#!/usr/bin/env python3
"""
train_unet.py - Simple U-Net training for event-to-depth estimation.

Inputs : voxels from events/voxels_cam0.h5 (5 temporal bins per frame)
Targets: depth from hdf5/depth_in_event_frame.h5 (float32 metres)
Data   : 3d_reconstruction/data/lego/  (lego_1 … lego_N)
Split  : 15% of objects → validation, rest → training (object-level split)
Metric : L1 loss on valid (non-zero depth) pixels, printed each epoch

Usage:
    python train_unet.py
    python train_unet.py --epochs 100 --batch_size 16 --out_dir checkpoints/unet_run1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from viz import VizLogger

# ---------------------------------------------------------------------------
# Constants (match 3d_reconstruction/config.py)
# ---------------------------------------------------------------------------
DEPTH_MIN = 0.05   # metres — minimum valid depth
D_MAX     = 0.60   # metres — maximum valid depth
NUM_BINS  = 5      # temporal bins per voxel grid

# Default data root relative to this file's parent directory
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT   = _SCRIPT_DIR.parent / "data" / "lego"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LegoDepthDataset(Dataset):
    """
    Per-frame dataset for a single lego sequence.

    Returns:
        voxels : (C, H, W) float32  event voxel grid (C = NUM_BINS)
        depth  : (1, H, W) float32  ground-truth depth in metres
        mask   : (1, H, W) float32  1 where depth is valid, 0 elsewhere
    """

    def __init__(self, seq_dir: Path):
        super().__init__()
        self.seq_dir     = seq_dir
        self.voxels_path = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path  = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.mask_path   = seq_dir / "hdf5"   / "spatial_mask.h5"

        import h5py  # imported here so the module is loaded lazily per worker
        with h5py.File(self.depth_path,  "r") as f:
            n_depth = int(f["depth"].shape[0])
        with h5py.File(self.voxels_path, "r") as f:
            n_vox = int(f["voxels"].shape[0])

        self.n_frames = min(n_depth, n_vox)
        self.has_mask = self.mask_path.exists()

        # HDF5 handles are opened lazily once per DataLoader worker to avoid
        # pickling issues and repeated open/close overhead per sample.
        self._ds_voxels = None
        self._ds_depth  = None
        self._ds_mask   = None

    def _open_handles(self):
        import h5py
        if self._ds_voxels is None:
            self._ds_voxels = h5py.File(self.voxels_path, "r")["voxels"]
        if self._ds_depth is None:
            self._ds_depth  = h5py.File(self.depth_path,  "r")["depth"]
        if self.has_mask and self._ds_mask is None:
            self._ds_mask   = h5py.File(self.mask_path,   "r")["mask"]

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int):
        self._open_handles()

        # Voxels: (C, H, W) — stored as float16 or float32
        voxels = self._ds_voxels[idx]
        if voxels.dtype == np.float16:
            voxels = voxels.astype(np.float32)

        # Depth: (H, W) in metres
        depth = self._ds_depth[idx].astype(np.float32)

        # Valid mask: pixels with positive depth and within spatial mask
        valid = (depth > 0).astype(np.float32)
        if self.has_mask:
            spatial = self._ds_mask[idx].astype(np.float32)
            valid   = valid * spatial

        # Convert to tensors
        voxels_t = torch.from_numpy(voxels)                    # (C, Hv, Wv)
        depth_t  = torch.from_numpy(depth).unsqueeze(0)        # (1, Hd, Wd)
        valid_t  = torch.from_numpy(valid).unsqueeze(0)        # (1, Hd, Wd)

        # Resize depth and mask to match voxel spatial resolution if they differ
        _, Hv, Wv = voxels_t.shape
        if depth_t.shape[-2] != Hv or depth_t.shape[-1] != Wv:
            depth_t = F.interpolate(
                depth_t.unsqueeze(0), size=(Hv, Wv), mode="nearest"
            ).squeeze(0)
            # nearest-neighbour for mask to avoid interpolation artefacts
            valid_t = F.interpolate(
                valid_t.unsqueeze(0), size=(Hv, Wv), mode="nearest"
            ).squeeze(0)

        return voxels_t, depth_t, valid_t


# ---------------------------------------------------------------------------
# U-Net architecture
# Strided-conv encoder, bilinear decoder, residual bottleneck (3 stages)
# ---------------------------------------------------------------------------
class _EncoderBlock(nn.Module):
    """Strided conv (÷2) + plain conv — downsamples by 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Bilinear upsample + concat skip + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _ResidualBlock(nn.Module):
    """Bottleneck residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class UNet(nn.Module):
    """
    Event-to-depth U-Net.
    Output: sigmoid → [0, 1] linear-normalised depth.
    Convert to metres: pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN.

    Encoder: stem + num_encoders strided-conv blocks (default 3).
    Bottleneck: num_residuals residual blocks (default 2).
    Decoder: num_encoders bilinear-upsample blocks with skip connections.
    """

    def __init__(
        self,
        in_ch:         int = NUM_BINS,
        base:          int = 32,
        num_encoders:  int = 3,
        num_residuals: int = 2,
    ):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # Encoder
        self.encoders = nn.ModuleList()
        ch = base
        for _ in range(num_encoders):
            self.encoders.append(_EncoderBlock(ch, ch * 2))
            ch *= 2
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[_ResidualBlock(ch) for _ in range(num_residuals)]
        )
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            out_ch  = ch // 2
            self.decoders.append(_DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch
        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.stem(x)
        skips = [feat]
        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i < len(self.encoders) - 1:
                skips.append(feat)
        feat = self.bottleneck(feat)
        for i, dec in enumerate(self.decoders):
            feat = dec(feat, skips[-(i + 1)])
        return self.head(feat)   # (B, 1, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# Loss  (e2depth_loss: charbonnier + gradient + smoothness + mean-depth)
# ---------------------------------------------------------------------------
def _charbonnier(pred: torch.Tensor, gt: torch.Tensor,
                 mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (torch.sqrt((pred - gt) ** 2 + eps ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_loss(pred: torch.Tensor, gt: torch.Tensor,
                   mask: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
    def gx(t): return t[:, :, :, 1:] - t[:, :, :, :-1]
    def gy(t): return t[:, :, 1:, :] - t[:, :, :-1, :]

    total = 0.0
    for s in range(num_scales):
        if s > 0:
            m    = F.avg_pool2d(mask, 2)
            pred = F.avg_pool2d(pred * mask, 2) / m.clamp_min(1e-6)
            gt   = F.avg_pool2d(gt   * mask, 2) / m.clamp_min(1e-6)
            mask = (m > 0.5).float()
        r  = pred - gt
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        total = total + (torch.abs(gx(r)) * mx).sum() / mx.sum().clamp_min(1.0)
        total = total + (torch.abs(gy(r)) * my).sum() / my.sum().clamp_min(1.0)
    return total / num_scales


def _smoothness_loss(pred: torch.Tensor, events: torch.Tensor,
                     mask: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    activity = events.abs().sum(dim=1, keepdim=True)
    a_max    = activity.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp_min(1e-6)
    activity = activity / a_max

    dx_p = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_p = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_e = (activity[:, :, :, 1:] + activity[:, :, :, :-1]) * 0.5
    dy_e = (activity[:, :, 1:, :] + activity[:, :, :-1, :]) * 0.5
    mx   = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my   = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    lx = (dx_p * torch.exp(-gamma * dx_e) * mx).sum() / mx.sum().clamp_min(1.0)
    ly = (dy_p * torch.exp(-gamma * dy_e) * my).sum() / my.sum().clamp_min(1.0)
    return lx + ly


def _mean_loss(pred: torch.Tensor, gt: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum().clamp_min(1.0)
    return torch.abs((pred * mask).sum() / n - (gt * mask).sum() / n)


def compute_loss(
    pred:           torch.Tensor,   # (B,1,H,W) model output in [0, 1]
    gt_norm:        torch.Tensor,   # (B,1,H,W) GT normalised to [0, 1]
    mask:           torch.Tensor,   # (B,1,H,W) binary valid mask
    events:         torch.Tensor,   # (B,C,H,W) voxels (for smoothness)
    K:              torch.Tensor | None = None,  # (3,3) scaled camera intrinsics
    lambda_grad:    float = 0.5,
    lambda_smooth:  float = 0.01,
    lambda_mean:    float = 0.1,
    lambda_normal:  float = 0.1,
) -> tuple:
    l_charb  = _charbonnier(pred, gt_norm, mask)
    l_grad   = _gradient_loss(pred, gt_norm, mask)
    l_smooth = _smoothness_loss(pred, events, mask)
    l_mean   = _mean_loss(pred, gt_norm, mask)
    total    = (l_charb
                + lambda_grad   * l_grad
                + lambda_smooth * l_smooth
                + lambda_mean   * l_mean)
    l_normal_val = 0.0
    if K is not None:
        l_normal      = _normal_loss(pred, gt_norm, mask, K)
        total         = total + lambda_normal * l_normal
        l_normal_val  = l_normal.item()
    return total, {
        "charb":  l_charb.item(),
        "grad":   l_grad.item(),
        "smooth": l_smooth.item(),
        "mean":   l_mean.item(),
        "normal": l_normal_val,
    }


def _l1_metres(pred_norm: torch.Tensor, depth_gt: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """L1 in metres for reporting. pred_norm in [0, 1], depth_gt in metres."""
    pred_m = pred_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    return (torch.abs(pred_m - depth_gt) * mask).sum() / mask.sum().clamp_min(1.0)


_pixel_grid_cache: dict = {}


def _get_pixel_grid(H: int, W: int, device: torch.device):
    key = (H, W, str(device))
    if key not in _pixel_grid_cache:
        u  = torch.arange(W, device=device, dtype=torch.float32)
        v  = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pix = torch.stack(
            [uu.reshape(-1), vv.reshape(-1), torch.ones(H * W, device=device)], dim=0
        )
        _pixel_grid_cache[key] = (uu, vv, pix)
    return _pixel_grid_cache[key]


def _compute_normals(depth_m: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Geometrically correct surface normals via backprojection + cross product."""
    B, _, H, W = depth_m.shape
    device = depth_m.device
    K  = K.to(device=device, dtype=torch.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uu, vv, _ = _get_pixel_grid(H, W, device)
    D = depth_m[:, 0]
    points = torch.stack([
        (uu - cx) * D / fx,
        (vv - cy) * D / fy,
        D,
    ], dim=1)

    du = points[:, :, :, 2:] - points[:, :, :, :-2]
    dv = points[:, :, 2:, :] - points[:, :, :-2, :]
    du = F.pad(du, (1, 1, 0, 0), mode="replicate")
    dv = F.pad(dv, (0, 0, 1, 1), mode="replicate")

    nx = du[:, 1] * dv[:, 2] - du[:, 2] * dv[:, 1]
    ny = du[:, 2] * dv[:, 0] - du[:, 0] * dv[:, 2]
    nz = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
    return F.normalize(torch.stack([nx, ny, nz], dim=1), dim=1)


def _normal_loss(pred: torch.Tensor, gt: torch.Tensor,
                 mask: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Surface normal cosine loss. pred and gt are normalised [0, 1]."""
    pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    gt_m   = gt   * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    n_pred = _compute_normals(pred_m, K)
    n_gt   = _compute_normals(gt_m,   K)
    cosine = (n_pred * n_gt).sum(dim=1, keepdim=True)
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    K:         torch.Tensor | None = None,
    viz:       VizLogger | None = None,
) -> tuple:
    """
    Run one training or validation epoch.
    Returns (mean_total_loss, mean_l1_metres).
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_l1   = 0.0
    with ctx:
        for voxels, depth, mask in loader:
            voxels = voxels.to(device)
            depth  = depth.to(device)
            mask   = mask.to(device)

            # Normalise GT from metres → [0, 1]
            depth_norm = ((depth - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)

            pred = model(voxels)
            loss, _ = compute_loss(pred, depth_norm, mask, voxels, K=K)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_l1 += _l1_metres(pred, depth, mask).item()

            if viz is not None:
                # Convert prediction to metres for visualisation
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                viz.add_batch(voxels, depth, mask, pred_m)

    n = len(loader)
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_l1 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet for event-to-depth")
    parser.add_argument("--data_dir",       type=Path, default=DATA_ROOT,
                        help="Either a single sequence directory (e.g. data/lego/lego_1) "
                             "or a parent folder containing multiple sequence sub-directories "
                             "(e.g. data/lego). Detected automatically.")
    parser.add_argument("--epochs",         type=int,  default=50)
    parser.add_argument("--batch_size",     type=int,  default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--workers",        type=int,  default=4)
    parser.add_argument("--base_channels",  type=int,  default=32,
                        help="Base channel count; doubled at each encoder level")
    parser.add_argument("--out_dir",        type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "unet",
                        help="Directory for checkpoints and TensorBoard logs")
    parser.add_argument("--seed",           type=int,  default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Discover valid sequence directories
    # Auto-detect: if --data_dir itself is a sequence, use single-object mode.
    # Otherwise treat it as a parent folder and collect sub-directories.
    # ------------------------------------------------------------------
    def _is_sequence(p: Path) -> bool:
        return (
            (p / "events" / "voxels_cam0.h5").exists()
            and (p / "hdf5" / "depth_in_event_frame.h5").exists()
        )

    if _is_sequence(args.data_dir):
        seq_dirs     = [args.data_dir]
        single_object = True
    else:
        seq_dirs = sorted([d for d in args.data_dir.iterdir()
                           if d.is_dir() and _is_sequence(d)])
        single_object = len(seq_dirs) == 1

    if not seq_dirs:
        sys.exit(f"[ERROR] No valid sequences found at or under {args.data_dir}\n"
                 "        Make sure precompute_voxels.py and "
                 "project_realsense_to_event.py have been run.")

    # ------------------------------------------------------------------
    # Object-level split: 15% of sequences → validation
    # (single-object mode: use all data for both train and val)
    # ------------------------------------------------------------------
    if single_object:
        train_seqs = seq_dirs
        val_seqs   = seq_dirs
        print(f"Found {len(seq_dirs)} sequence(s) in {args.data_dir} "
              f"[single-object mode: all data used for train and val]")
        print(f"  Sequences: {[d.name for d in seq_dirs]}")
    else:
        rng = np.random.default_rng(args.seed)
        order = np.arange(len(seq_dirs))
        rng.shuffle(order)

        n_val = max(1, round(len(seq_dirs) * 0.15))
        val_seqs   = [seq_dirs[i] for i in order[:n_val]]
        train_seqs = [seq_dirs[i] for i in order[n_val:]]

        print(f"Found {len(seq_dirs)} sequence(s) in {args.data_dir}")
        print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
        print(f"  Val   ({len(val_seqs)}):   {[d.name for d in val_seqs]}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_ds = ConcatDataset([LegoDepthDataset(d) for d in train_seqs])
    val_ds   = ConcatDataset([LegoDepthDataset(d) for d in val_seqs])

    print(f"  Train frames: {len(train_ds)},  Val frames: {len(val_ds)}")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    # drop_last=True prevents a trailing batch of size 1, which would crash
    # BatchNorm2d in training mode (requires > 1 sample per batch).
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_ch=NUM_BINS, base=args.base_channels).to(device)

    # Load event camera intrinsics and scale K to the voxel crop resolution
    _cam_data = np.load(_SCRIPT_DIR.parent / "camera_data" / "event_intrinsics.npz")
    _K_full   = _cam_data["camera_matrix"].astype(np.float32)
    _H_full, _W_full = 720, 1280   # original event-camera resolution
    _H_crop, _W_crop = 240, 320    # TRAIN_CROP_HW (voxel resolution)
    _K_scaled = _K_full.copy()
    _K_scaled[0, :] *= _W_crop / _W_full   # scale fx and cx
    _K_scaled[1, :] *= _H_crop / _H_full   # scale fy and cy
    K = torch.from_numpy(_K_scaled).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nUNet  base={args.base_channels}  parameters: {n_params:,}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))

    viz_train = VizLogger(writer, n_samples=4, tag="viz/train")
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_l1 = run_epoch(model, train_loader, optimizer, device, K=K, viz=viz_train)
        val_loss,   val_l1   = run_epoch(model, val_loader,   None,      device, K=K, viz=viz_val)
        scheduler.step()

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)

        vram_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_res   = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train loss: {train_loss:.4f}  train L1: {train_l1:.4f} m  "
              f"val loss: {val_loss:.4f}  val L1: {val_l1:.4f} m  "
              f"VRAM: {vram_alloc:.0f}/{vram_res:.0f} MB")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("l1/train",   train_l1,   epoch)
        writer.add_scalar("l1/val",     val_l1,     epoch)
        writer.add_scalar("lr",         scheduler.get_last_lr()[0], epoch)

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            ckpt_path   = args.out_dir / "best.pth"
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "val_l1":    val_l1,
                "base":      args.base_channels,
            }, ckpt_path)
            print(f"  → new best checkpoint saved (val L1 = {val_l1:.4f} m)")

    # Save final checkpoint
    torch.save({
        "epoch":  args.epochs,
        "model":  model.state_dict(),
        "val_l1": val_l1,
        "base":   args.base_channels,
    }, args.out_dir / "last.pth")

    writer.close()
    print(f"\nDone. Best val L1: {best_val_l1:.4f} m")


if __name__ == "__main__":
    main()
