#!/usr/bin/env python3
"""
train_transformer_table.py - Single-frame transformer depth model with table-plane prior channel.

Feeds a transformer-based dense depth model the target event voxels together with one additional channel
that encodes the table plane directly in the image plane:

    [x_t | table_plane_channel]

    x_t                 : target event voxel grid    (NUM_BINS channels)
    table_plane_channel : per-pixel z-depth [m] to the table plane,
                          normalised to [0, 1] using the training depth range.
                          Computed by intersecting per-pixel camera rays with
                          the horizontal plane  z = table_z  in the robot
                          base frame, using the frame's end-effector pose.

Default input channels: NUM_BINS + 1

Usage:
    python3 training/train_transformer_table.py
    python3 training/train_transformer_table.py --data_dir data/lego/lego_1
    python3 training/train_transformer_table.py --patch_size 4 --embed_dim 192 --depth 8 --num_heads 6
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from train_unet import (
    DEPTH_MIN, D_MAX, NUM_BINS, _SCRIPT_DIR, DATA_ROOT,
    compute_loss, _l1_metres, _worst_percent_l1_metres,
)
from tensorboard_runs import DEFAULT_TB_ROOT, tensorboard_run_dir
from viz import (
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
    UncertaintyErrorLogger,
    VizLogger,
)

# Camera calibration paths
_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_event_K_native() -> tuple[np.ndarray, int, int]:
    """Return event-camera intrinsics at native resolution, plus native (H, W)."""
    d = np.load(_CAM_DATA / "event_intrinsics.npz")
    K = d["camera_matrix"].astype(np.float32).copy()
    W = int(d["image_size"][0])
    H = int(d["image_size"][1])
    return K, H, W


# ---------------------------------------------------------------------------
# Table-plane channel  (kept here so reconstruction.py can import it)
# ---------------------------------------------------------------------------

def _compute_table_plane_channel(
    T_base_from_event: np.ndarray,  # (4, 4)
    K_native: np.ndarray,            # (3, 3) at native camera resolution
    native_H: int,
    native_W: int,
    vox_H: int,
    vox_W: int,
    table_z_base: float,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> np.ndarray:
    """
    Return (vox_H, vox_W) float32 channel where each pixel holds the
    z-depth [m] (along the optical axis) at which the ray through that
    pixel hits the table plane  z = table_z_base  in the robot base frame.

    Values are normalised to [0, 1] using [depth_min, depth_max], matching
    how GT depth is normalised during training.
    Pixels whose rays are parallel to the plane, or face away from it, get 0.
    """
    # Scale K from native resolution to voxel grid resolution
    K = K_native.copy().astype(np.float64)
    K[0, :] *= vox_W / native_W
    K[1, :] *= vox_H / native_H
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u = np.arange(vox_W, dtype=np.float64)
    v = np.arange(vox_H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)   # (H, W)

    # Ray directions in camera frame (z = 1 plane → z-depth parameterisation)
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    dz = np.ones_like(dx)

    # Rotate ray directions into robot base frame
    R = T_base_from_event[:3, :3].astype(np.float64)
    rays_flat = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (HW, 3)
    d_base = (R @ rays_flat.T).T   # (HW, 3)

    # Camera origin in base frame
    t_cam = T_base_from_event[:3, 3].astype(np.float64)  # (3,)

    # Solve: (t_cam + depth * d_base)[2] = table_z_base
    #   => depth = (table_z_base - t_cam[2]) / d_base[:, 2]
    denom = d_base[:, 2]
    numer = float(table_z_base) - float(t_cam[2])
    depth = np.where(np.abs(denom) > 1e-6, numer / denom, depth_max)
    depth = np.where(depth > 0.0, depth, depth_max)   # rays that miss the table → max depth

    # Normalise to [0, 1] (same scheme as GT depth normalisation)
    channel = ((depth - depth_min) / max(float(depth_max - depth_min), 1e-6)).clip(0.0, 1.0)
    return channel.reshape(vox_H, vox_W).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TablePriorDataset(Dataset):
    """
    Single-frame dataset: event voxels + precomputed table-plane channel → GT depth.

    Requires hdf5/table_plane.h5 produced by
    data_precomputation/precompute_table_plane.py.

    Returns (all torch.Tensor on CPU):
        inp    : (NUM_BINS + 1, H, W)  voxels concatenated with table-plane channel
        dep_t  : (1, H, W)             GT depth [m]
        mask_t : (1, H, W)             valid depth mask
    """

    def __init__(self, seq_dir: Path, use_mask: bool = True,
                 fill_invalid: bool = False):
        super().__init__()
        self.seq_dir         = seq_dir
        self.fill_invalid    = fill_invalid
        self.voxels_path     = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path      = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.mask_path       = seq_dir / "hdf5"   / "spatial_mask.h5"
        self.table_plane_path = seq_dir / "hdf5"  / "table_plane.h5"

        if not self.table_plane_path.exists():
            raise FileNotFoundError(
                f"Missing: {self.table_plane_path}\n"
                "Run data_precomputation/precompute_table_plane.py first."
            )

        import h5py
        with h5py.File(self.depth_path,       "r") as f: n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path,      "r") as f: n_v = f["voxels"].shape[0]
        with h5py.File(self.table_plane_path, "r") as f: n_t = f["table_plane"].shape[0]

        n_frames      = min(n_d, n_v, n_t)
        self.has_mask = use_mask and self.mask_path.exists()

        self.valid_indices = np.arange(n_frames)
        self._vox = None   # lazy HDF5 handles, opened per DataLoader worker
        self._dep = None
        self._msk = None
        self._tbl = None

    def _open(self):
        import h5py
        if self._vox is None:
            self._vox = h5py.File(self.voxels_path,      "r")["voxels"]
        if self._dep is None:
            self._dep = h5py.File(self.depth_path,       "r")["depth"]
        if self.has_mask and self._msk is None:
            self._msk = h5py.File(self.mask_path,        "r")["mask"]
        if self._tbl is None:
            self._tbl = h5py.File(self.table_plane_path, "r")["table_plane"]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, item: int):
        self._open()
        idx = int(self.valid_indices[item])

        # Voxels
        vox = self._vox[idx]
        if vox.dtype == np.float16:
            vox = vox.astype(np.float32)
        vox_t = torch.from_numpy(vox)   # (C, H, W)
        _, Hv, Wv = vox_t.shape

        # Depth + mask
        dep   = self._dep[idx].astype(np.float32)
        dep   = np.minimum(dep, D_MAX)   # cap valid depth at D_MAX
        valid = (dep > 0).astype(np.float32)
        if self.has_mask:
            valid *= self._msk[idx].astype(np.float32)
        dep_t = torch.from_numpy(dep).unsqueeze(0)
        msk_t = torch.from_numpy(valid).unsqueeze(0)
        if dep_t.shape[-2] != Hv or dep_t.shape[-1] != Wv:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)

        # Table-plane channel (precomputed)
        tbl_np = self._tbl[idx].astype(np.float32)             # (H_stored, W_stored)
        tbl_t  = torch.from_numpy(tbl_np).unsqueeze(0)         # (1, H_stored, W_stored)
        if tbl_t.shape[-2] != Hv or tbl_t.shape[-1] != Wv:
            tbl_t = F.interpolate(
                tbl_t.unsqueeze(0), (Hv, Wv),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # Optionally fill invalid depth pixels with the table-plane prior [m]
        if self.fill_invalid:
            tbl_m = tbl_t * (D_MAX - DEPTH_MIN) + DEPTH_MIN  # (1, H, W) metres
            dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
            msk_t = torch.ones_like(msk_t)  # all pixels now have a valid target

        inp = torch.cat([vox_t, tbl_t], dim=0)   # (C+1, H, W)
        return inp, dep_t, msk_t


# ---------------------------------------------------------------------------
# Transformer depth model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Small convolutional refinement block used before/after the transformer."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_2d_sincos_position_embedding(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return fixed 2D sin/cos positional embeddings with shape (1, H*W, dim)."""
    if dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin/cos positional encoding")

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))

    out_x = x.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = y.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)
    return pos.unsqueeze(0)


class EventDepthTransformer(nn.Module):
    """
    Single-frame transformer for dense event-camera depth prediction.

    Pipeline:
        input image/grid -> convolutional stem -> patch tokens -> transformer encoder
        -> reshape tokens to feature map -> convolutional upsampling decoder -> depth map.

    This is intentionally close to a SegFormer/DPT-style dense predictor, but small
    enough to train from scratch on project-scale data.
    """

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        dropout: float = 0.0,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")

        self.patch_size = patch_size
        self.predict_uncertainty = predict_uncertainty

        self.stem = nn.Sequential(
            ConvBlock(in_ch, base),
            ConvBlock(base, base),
        )
        self.patch_embed = nn.Conv2d(
            base, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            ConvBlock(embed_dim + base, base * 4),
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base * 2, base, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(base, 2 if predict_uncertainty else 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        stem = self.stem(x)

        # Pad so patch embedding works for arbitrary H/W.
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        stem_pad = F.pad(stem, (0, pad_w, 0, pad_h), mode="replicate")

        feat = self.patch_embed(stem_pad)          # (B, D, Hp, Wp)
        _, d, hp, wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, Hp*Wp, D)
        tokens = tokens + _build_2d_sincos_position_embedding(
            hp, wp, d, tokens.device, tokens.dtype
        )
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        feat = tokens.transpose(1, 2).reshape(b, d, hp, wp)
        feat = F.interpolate(feat, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([feat, stem], dim=1)
        feat = self.decoder(feat)

        out = self.head(feat)
        if out.shape[-2:] != (h, w):
            out = out[..., :h, :w]

        if self.predict_uncertainty:
            pred = torch.sigmoid(out[:, :1])
            log_var = out[:, 1:2].clamp(min=-6.0, max=3.0)
            return pred, log_var
        return torch.sigmoid(out)


def uncertainty_nll_loss(
    pred_norm: torch.Tensor,
    log_var: torch.Tensor,
    gt_norm: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    residual_sq = (pred_norm - gt_norm) ** 2
    nll = 0.5 * (torch.exp(-log_var) * residual_sq + log_var)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def uncertainty_metres(log_var: torch.Tensor) -> torch.Tensor:
    return torch.exp(0.5 * log_var) * (D_MAX - DEPTH_MIN)


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    K:         torch.Tensor,
    viz=None,
    activity_diag=None,
    error_diag=None,
    uncertainty_diag=None,
    uncertainty_weight: float = 1.0,
    depth_aux_weight: float = 0.0,
) -> tuple:
    """One epoch. Returns (mean_total_loss, mean_l1_metres, mean_worst10_l1_metres)."""
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_l1 = total_worst10_l1 = 0.0
    _phase     = "train" if is_train else "val"
    _n_total   = len(loader)
    _n_batches = 0
    _t_last    = time.time()

    with ctx:
        for inp, dep_t, mask_t in loader:
            inp, dep_t, mask_t = inp.to(device), dep_t.to(device), mask_t.to(device)

            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            out = model(inp)
            if isinstance(out, tuple):
                pred, log_var = out
            else:
                pred, log_var = out, None

            if log_var is not None:
                loss = uncertainty_weight * uncertainty_nll_loss(
                    pred, log_var, dep_norm, mask_t
                )
                if depth_aux_weight > 0.0:
                    depth_aux_loss, _ = compute_loss(
                        pred, dep_norm, mask_t, inp[:, :NUM_BINS], K=K
                    )
                    loss = loss + depth_aux_weight * depth_aux_loss
            else:
                loss, _ = compute_loss(pred, dep_norm, mask_t, inp[:, :NUM_BINS], K=K)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_l1 += _l1_metres(pred, dep_t, mask_t).item()
                total_worst10_l1 += _worst_percent_l1_metres(pred, dep_t, mask_t).item()

            _n_batches += 1
            _now = time.time()
            if _now - _t_last >= 20.0:
                print(f"  [{_phase}  {_n_batches:4d}/{_n_total} batches]  "
                      f"loss {total_loss / _n_batches:.4f}  "
                      f"L1 {total_l1 / _n_batches:.4f} m  "
                      f"worst10 {total_worst10_l1 / _n_batches:.4f} m")
                _t_last = _now

            if viz is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                tbl_ch = inp[:, NUM_BINS:NUM_BINS + 1].detach()   # (B, 1, H, W) normalised [0,1]
                viz.add_batch(inp[:, :NUM_BINS], dep_t, mask_t, pred_m, table_depth=tbl_ch)
            if activity_diag is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                activity_diag.add_batch(inp[:, :NUM_BINS], dep_t, mask_t, pred_m)
            if error_diag is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                error_diag.add_batch(pred_m, dep_t, mask_t)
            if uncertainty_diag is not None and log_var is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                uncertainty_diag.add_batch(
                    uncertainty_metres(log_var).detach(),
                    pred_m,
                    dep_t,
                    mask_t,
                )

    n = max(len(loader), 1)
    return total_loss / n, total_l1 / n, total_worst10_l1 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-frame transformer depth model with table-plane prior channel"
    )
    parser.add_argument("--data_dir",      type=Path, default=DATA_ROOT,
                        help="Single sequence dir or parent of multiple sequences")
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--batch_size",    type=int,  default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--workers",       type=int,  default=4)
    parser.add_argument("--base_channels", type=int,  default=32)
    parser.add_argument("--embed_dim",     type=int,   default=192,
                        help="Transformer token dimension; must be divisible by --num_heads and 4")
    parser.add_argument("--depth",         type=int,   default=8,
                        help="Number of transformer encoder layers")
    parser.add_argument("--num_heads",     type=int,   default=6,
                        help="Number of transformer attention heads")
    parser.add_argument("--mlp_ratio",     type=float, default=4.0,
                        help="Transformer MLP expansion ratio")
    parser.add_argument("--patch_size",    type=int,   default=4,
                        help="Patch size / token stride. Use 4 for 240x320; 8 for lower VRAM.")
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--out_dir",       type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "transformer_table")
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--table_z",       type=float, default=None,
                        help="Optional: override for --debug / informational use. "
                             "The actual table_z is read from hdf5/table_plane.h5.")
    parser.add_argument("--no_mask",       action="store_true",
                        help="Ignore spatial mask; dep>0 validity is always applied")
    parser.add_argument("--fill_invalid",  action="store_true",
                        help="Fill pixels with no depth measurement using the table-plane prior")
    parser.add_argument("--predict_uncertainty", action="store_true",
                        help="Predict a per-pixel uncertainty map and train it with a heteroscedastic loss.")
    parser.add_argument("--uncertainty_weight", type=float, default=1.0,
                        help="Weight for the primary uncertainty negative-log-likelihood term.")
    parser.add_argument("--depth_aux_weight", type=float, default=0.0,
                        help="Optional auxiliary weight for the original depth loss when uncertainty is enabled.")
    parser.add_argument("--name",          type=str, default=None,
                        help="Run name used in checkpoint filenames. Prompted if not provided.")
    parser.add_argument("--tb_root",       type=Path, default=DEFAULT_TB_ROOT,
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/transformer_table/<name>.")
    args = parser.parse_args()

    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import h5py
    K_native, native_H, native_W = _load_event_K_native()

    # ── Discover sequences ────────────────────────────────────────────────
    def _is_sequence(p: Path) -> bool:
        return (
            (p / "events" / "voxels_cam0.h5").exists()
            and (p / "hdf5" / "depth_in_event_frame.h5").exists()
            and (p / "hdf5" / "poses.h5").exists()
            and (p / "hdf5" / "table_plane.h5").exists()
        )

    if _is_sequence(args.data_dir):
        seq_dirs      = [args.data_dir]
        single_object = True
    else:
        seq_dirs = sorted([d for d in args.data_dir.iterdir()
                           if d.is_dir() and _is_sequence(d)])
        single_object = len(seq_dirs) == 1

    if not seq_dirs:
        sys.exit(
            f"[ERROR] No valid sequences found at or under {args.data_dir}\n"
            "        Make sure voxels, depth, poses.h5, and table_plane.h5 all exist.\n"
            "        Run: python3 data_precomputation/precompute_table_plane.py"
        )

    # ── Object-level split ────────────────────────────────────────────────
    if single_object:
        train_seqs = val_seqs = seq_dirs
        print(f"Found {len(seq_dirs)} sequence(s) [single-object mode]")
        print(f"  Sequences: {[d.name for d in seq_dirs]}")
    else:
        rng   = np.random.default_rng(args.seed)
        order = np.arange(len(seq_dirs))
        rng.shuffle(order)
        n_val      = max(1, round(len(seq_dirs) * 0.15))
        val_seqs   = [seq_dirs[i] for i in order[:n_val]]
        train_seqs = [seq_dirs[i] for i in order[n_val:]]
        print(f"Found {len(seq_dirs)} sequence(s) in {args.data_dir}")
        print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
        print(f"  Val   ({len(val_seqs)}):   {[d.name for d in val_seqs]}")

    # ── Read table_z stored in precomputed file (used only for checkpoint) ───
    with h5py.File(seq_dirs[0] / "hdf5" / "table_plane.h5", "r") as _tf:
        _stored_table_z = float(_tf.attrs.get("table_z_m", args.table_z or 0.0))
    if args.table_z is not None and abs(args.table_z - _stored_table_z) > 1e-5:
        print(f"  WARNING: --table_z {args.table_z} differs from precomputed "
              f"table_z_m={_stored_table_z:.4f}. Using precomputed value.")
    table_z_ckpt = _stored_table_z

    # ── DataLoaders ───────────────────────────────────────────────────────
    ds_kw = dict(
        use_mask=not args.no_mask,
        fill_invalid=args.fill_invalid,
    )
    train_ds = ConcatDataset([TablePriorDataset(d, **ds_kw) for d in train_seqs])
    val_ds   = ConcatDataset([TablePriorDataset(d, **ds_kw) for d in val_seqs])
    print(f"  Train frames: {len(train_ds)},  Val frames: {len(val_ds)}")
    print(f"  Table plane Z: {table_z_ckpt:.4f} m (from precomputed table_plane.h5)\n")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ── Model ──────────────────────────────────────────────────────────────
    in_ch  = NUM_BINS + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventDepthTransformer(
        in_ch=in_ch,
        base=args.base_channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        dropout=args.dropout,
        predict_uncertainty=args.predict_uncertainty,
    ).to(device)

    # K for loss: scale native K to a representative crop resolution
    K_loss = K_native.copy()
    K_loss[0, :] *= 320 / native_W
    K_loss[1, :] *= 240 / native_H
    K_tensor = torch.from_numpy(K_loss).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "EventDepthTransformer+uncertainty" if args.predict_uncertainty else "EventDepthTransformer"
    print(f"{model_name}  in_ch={in_ch}  base={args.base_channels}  "
          f"embed={args.embed_dim} depth={args.depth} heads={args.num_heads} "
          f"patch={args.patch_size}  parameters: {n_params:,}")
    print(f"  {NUM_BINS} (voxels) + 1 (table-plane channel) = {in_ch} channels")
    if args.predict_uncertainty:
        print(f"  Uncertainty head enabled; primary NLL weight = {args.uncertainty_weight:g}")
        print(f"  Auxiliary depth-loss weight = {args.depth_aux_weight:g}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ── Logging ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("transformer_table", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")
    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val",   show_mask=not args.no_mask)
    activity_train = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/train")
    activity_val   = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/val")
    error_train = ErrorDistributionSpatialLogger(writer, tag="error/train")
    error_val   = ErrorDistributionSpatialLogger(writer, tag="error/val")
    uncertainty_train = (
        UncertaintyErrorLogger(writer, tag="uncertainty_error/train")
        if args.predict_uncertainty else None
    )
    uncertainty_val = (
        UncertaintyErrorLogger(writer, tag="uncertainty_error/val")
        if args.predict_uncertainty else None
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_l1 = float("inf")
    ckpt: dict = {}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, tr_worst10 = run_epoch(model, train_loader, optimizer, device,
                                    K_tensor, viz=viz_train,
                                    activity_diag=activity_train,
                                    error_diag=error_train,
                                    uncertainty_diag=uncertainty_train,
                                    uncertainty_weight=args.uncertainty_weight,
                                    depth_aux_weight=args.depth_aux_weight)
        va_loss, va_l1, va_worst10 = run_epoch(model, val_loader,   None,      device,
                                    K_tensor, viz=viz_val,
                                    activity_diag=activity_val,
                                    error_diag=error_val,
                                    uncertainty_diag=uncertainty_val,
                                    uncertainty_weight=args.uncertainty_weight,
                                    depth_aux_weight=args.depth_aux_weight)
        scheduler.step()

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)
        activity_train.flush(step=epoch)
        activity_val.flush(step=epoch)
        error_train.flush(step=epoch)
        error_val.flush(step=epoch)
        if uncertainty_train is not None:
            uncertainty_train.flush(step=epoch)
        if uncertainty_val is not None:
            uncertainty_val.flush(step=epoch)

        vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
            f"L1: {tr_l1:.4f}/{va_l1:.4f} m  "
            f"worst10: {tr_worst10:.4f}/{va_worst10:.4f} m  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val",   va_loss, epoch)
        writer.add_scalar("l1/train",   tr_l1,   epoch)
        writer.add_scalar("l1/val",     va_l1,   epoch)
        writer.add_scalar("l1_worst10/train", tr_worst10, epoch)
        writer.add_scalar("l1_worst10/val",   va_worst10, epoch)
        writer.add_scalar("lr",         scheduler.get_last_lr()[0], epoch)

        ckpt = {
            "epoch":   epoch,
            "model":   model.state_dict(),
            "val_l1":  va_l1,
            "base":    args.base_channels,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "patch_size": args.patch_size,
            "dropout": args.dropout,
            "architecture": "EventDepthTransformer",
            "in_ch":   in_ch,
            "table_z": table_z_ckpt,
            "predict_uncertainty": args.predict_uncertainty,
            "uncertainty_weight": args.uncertainty_weight,
            "depth_aux_weight": args.depth_aux_weight,
        }
        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            torch.save(ckpt, args.out_dir / f"best_{args.name}.pth")
            print(f"  → new best checkpoint  (val L1 = {va_l1:.4f} m)")

    torch.save(ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(f"\nDone. Best val L1: {best_val_l1:.4f} m")


if __name__ == "__main__":
    main()
