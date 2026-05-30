#!/usr/bin/env python3
"""
train_pose_unet.py - Two-stage pose-guided U-Net for event-to-depth.

Stage 1: For every frame in a temporal window centred on the target, predict
         an initial depth from its own event voxels alone (shared UNet_s1).

Stage 2: Forward-project each stage-1 depth into the target frame's view using
         known camera poses, then feed the projected depth channels together
         with the target's voxels into a second UNet_s2 for the final prediction.

Data layout (per sequence directory):
  events/voxels_cam0.h5          – (N, C, H, W)  event voxels
  hdf5/depth_in_event_frame.h5   – (N, H_full, W_full)  GT depth in metres
  hdf5/spatial_mask.h5           – (N, H_full, W_full)  binary mask
  hdf5/poses.h5 / ee_T           – (N, 4, 4)  T_world_from_ee

Usage:
    python3 training/train_pose_unet.py
    python3 training/train_pose_unet.py --window_half 2 --window_stride 3 --data_dir data/lego/lego_1
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

# Shared building-blocks from train_unet (same directory)
from train_unet import (
    DEPTH_MIN, D_MAX, NUM_BINS, _SCRIPT_DIR, DATA_ROOT,
    _EncoderBlock, _DecoderBlock, _ResidualBlock, UNet,
    compute_loss, _l1_metres,
    _get_pixel_grid,
)
from viz import VizLogger

# Camera calibration paths
_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _load_event_K_scaled(h_crop: int = 240, w_crop: int = 320,
                          h_full: int = 720, w_full: int = 1280) -> np.ndarray:
    """Load and scale event-camera intrinsics to the voxel crop resolution."""
    K = np.load(_CAM_DATA / "event_intrinsics.npz")["camera_matrix"].astype(np.float32).copy()
    K[0, :] *= w_crop / w_full   # scale fx and cx
    K[1, :] *= h_crop / h_full   # scale fy and cy
    return K  # (3, 3)


def _load_T_event_from_ee() -> np.ndarray:
    """T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee  (4×4 float64)."""
    T_er = np.load(_CAM_DATA / "T_event_from_rgb.npz")["T"]
    T_re = np.load(_CAM_DATA / "T_rgb_from_ee.npz")["T"]
    return (T_er @ T_re).astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PoseWindowDataset(Dataset):
    """
    Per-frame dataset returning a temporal window of 2*window_half+1 frames.

    Returns (all are torch.Tensor):
        voxels_t    : (C, H, W)          – target voxels
        depth_t     : (1, H, W)          – target GT depth [m]
        mask_t      : (1, H, W)          – target valid mask
        voxels_nbr  : (nwin, C, H, W)    – nwin = 2*window_half neighbour voxels
        depth_nbr   : (nwin, 1, H, W)    – neighbour GT depths [m]
        mask_nbr    : (nwin, 1, H, W)    – neighbour valid masks
        T_tgt_nbr   : (nwin, 4, 4)       – T_tgt_from_nbr for each neighbour
    """

    def __init__(self, seq_dir: Path, window_half: int = 2, stride: int = 3):
        super().__init__()
        self.seq_dir     = seq_dir
        self.window_half = window_half
        self.stride      = stride
        self.voxels_path = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path  = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.mask_path   = seq_dir / "hdf5"   / "spatial_mask.h5"
        self.poses_path  = seq_dir / "hdf5"   / "poses.h5"

        import h5py
        with h5py.File(self.depth_path,  "r") as f: n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path, "r") as f: n_v = f["voxels"].shape[0]
        with h5py.File(self.poses_path,  "r") as f: n_p = f["ee_T"].shape[0]

        n_frames = min(n_d, n_v, n_p)
        self.has_mask = self.mask_path.exists()

        # Precompute T_world_from_event for every frame (small: N*4*4*8 bytes)
        T_event_from_ee = _load_T_event_from_ee()           # (4,4)
        T_ee_from_event = np.linalg.inv(T_event_from_ee)    # (4,4)
        with h5py.File(self.poses_path, "r") as f:
            ee_T = f["ee_T"][:n_frames]                      # (N, 4,4) float64
        # T_world_from_event[i] = ee_T[i] @ T_ee_from_event
        self.T_world_from_event = (ee_T @ T_ee_from_event[None]).astype(np.float32)  # (N,4,4)

        # Restrict valid target indices so the full window is always available
        # (margin = window_half * stride frames on each side)
        W = window_half
        margin = W * stride
        self.valid_indices = np.arange(margin, n_frames - margin)

        self._vox = None   # lazy HDF5 handles (opened per DataLoader worker)
        self._dep = None
        self._msk = None

    def _open(self):
        import h5py
        if self._vox is None:
            self._vox = h5py.File(self.voxels_path, "r")["voxels"]
        if self._dep is None:
            self._dep = h5py.File(self.depth_path,  "r")["depth"]
        if self.has_mask and self._msk is None:
            self._msk = h5py.File(self.mask_path,   "r")["mask"]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_frame(self, idx: int):
        """Return (voxels, depth, mask) tensors for absolute frame index idx."""
        vox = self._vox[idx]
        if vox.dtype == np.float16:
            vox = vox.astype(np.float32)
        dep = self._dep[idx].astype(np.float32)
        valid = (dep > 0).astype(np.float32)
        if self.has_mask:
            valid *= self._msk[idx].astype(np.float32)

        vox_t = torch.from_numpy(vox)                  # (C, Hv, Wv)
        dep_t = torch.from_numpy(dep).unsqueeze(0)     # (1, Hd, Wd)
        msk_t = torch.from_numpy(valid).unsqueeze(0)   # (1, Hd, Wd)

        # Resize depth/mask to voxel spatial resolution if needed
        _, Hv, Wv = vox_t.shape
        if dep_t.shape[-2] != Hv or dep_t.shape[-1] != Wv:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)

        return vox_t, dep_t, msk_t

    def __getitem__(self, item: int):
        self._open()
        t = int(self.valid_indices[item])
        W = self.window_half
        S = self.stride

        # Target frame
        vox_t, dep_t, msk_t = self._load_frame(t)

        # Neighbour frame indices: offsets ±1…±W, each multiplied by stride
        nbr_idxs = [t + off * S for off in range(-W, W + 1) if off != 0]

        vox_nbr_list, dep_nbr_list, msk_nbr_list, T_list = [], [], [], []
        T_tgt_from_world = np.linalg.inv(self.T_world_from_event[t])  # (4,4)

        for ni in nbr_idxs:
            vn, dn, mn = self._load_frame(ni)
            vox_nbr_list.append(vn)
            dep_nbr_list.append(dn)
            msk_nbr_list.append(mn)
            # T_tgt_from_nbr = inv(T_world_from_event[t]) @ T_world_from_event[ni]
            T_ti = (T_tgt_from_world @ self.T_world_from_event[ni]).astype(np.float32)
            T_list.append(torch.from_numpy(T_ti))

        vox_nbr = torch.stack(vox_nbr_list)   # (nwin, C, H, W)
        dep_nbr = torch.stack(dep_nbr_list)   # (nwin, 1, H, W)
        msk_nbr = torch.stack(msk_nbr_list)   # (nwin, 1, H, W)
        T_tgt_nbr = torch.stack(T_list)       # (nwin, 4, 4)

        return vox_t, dep_t, msk_t, vox_nbr, dep_nbr, msk_nbr, T_tgt_nbr


# ---------------------------------------------------------------------------
# Forward depth projection
# ---------------------------------------------------------------------------
@torch.no_grad()
def project_depth(
    depth_src_norm: torch.Tensor,   # (B, 1, H, W) stage-1 depth in [0, 1]
    T_tgt_from_src: torch.Tensor,   # (B, 4, 4) relative camera pose
    K:              torch.Tensor,   # (3, 3) scaled intrinsics
) -> torch.Tensor:                  # (B, 1, H, W) projected depth in [0, 1]
    """
    Forward-project normalised depth from source to target frame.

    Converts to metric depth, backprojects each source pixel to 3-D, transforms
    to the target frame, and scatters into the target image using a z-buffer
    (farther pixels processed first, closer pixels overwrite).

    Returns a normalised [0, 1] depth map; pixels without a projection are 0.
    No gradients — projected maps are used as fixed inputs to stage 2.
    """
    depth_m = (depth_src_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN)
    B, _, H, W = depth_m.shape
    device = depth_m.device

    K_f   = K.to(device=device, dtype=torch.float32)
    K_inv = torch.linalg.inv(K_f)
    T     = T_tgt_from_src.to(device=device, dtype=torch.float32)

    uu, vv, pix = _get_pixel_grid(H, W, device)   # pixel-coord grids
    N = H * W

    # Back-project source pixels to 3-D
    pix_b  = pix.unsqueeze(0).expand(B, -1, -1)                        # (B, 3, N)
    rays   = torch.bmm(K_inv.unsqueeze(0).expand(B, -1, -1), pix_b)    # (B, 3, N)
    D_flat = depth_m.reshape(B, 1, N)
    pts_s  = rays * D_flat                                               # (B, 3, N)
    pts_sh = torch.cat([pts_s, torch.ones(B, 1, N, device=device)], 1)  # (B, 4, N)

    # Transform to target frame
    pts_t = torch.bmm(T, pts_sh)[:, :3]                                  # (B, 3, N)

    # Project to target image plane
    z_t  = pts_t[:, 2]                                                    # (B, N)
    K_b  = K_f.unsqueeze(0).expand(B, -1, -1)
    uvz  = torch.bmm(K_b, pts_t)                                          # (B, 3, N)
    u_t  = (uvz[:, 0] / uvz[:, 2].clamp_min(1e-6)).round().long()        # (B, N)
    v_t  = (uvz[:, 1] / uvz[:, 2].clamp_min(1e-6)).round().long()        # (B, N)

    valid = (z_t > 0) & (u_t >= 0) & (u_t < W) & (v_t >= 0) & (v_t < H)

    # Vectorised z-buffer scatter — no Python loop over B.
    # Each source pixel maps to a flat position in a B*N buffer.
    b_off    = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * N  # (B, 1)
    flat_idx = (b_off + v_t * W + u_t).reshape(-1)   # (B*N,)
    zi_flat  = z_t.reshape(-1)                         # (B*N,)
    vf       = valid.reshape(-1)                       # (B*N,) bool

    zi_v  = zi_flat[vf]
    idx_v = flat_idx[vf]
    # Sort far→close so closer depths overwrite (z-buffer)
    order = torch.argsort(zi_v, descending=True)

    out = torch.zeros(B * N, device=device)
    out.scatter_(0, idx_v[order], zi_v[order])
    out = out.view(B, N)

    mask_out = (out > 0).float()
    out_norm = ((out - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
    return (out_norm * mask_out).view(B, 1, H, W)


# ---------------------------------------------------------------------------
# Two-stage model
# ---------------------------------------------------------------------------
class PoseUNet(nn.Module):
    """
    Two-stage pose-guided depth estimation network.

    Stage 1 (self.s1): shared UNet – any frame's voxels → initial depth [0, 1].
    Stage 2 (self.s2): UNet taking:
        - target voxels          (NUM_BINS channels)
        - target stage-1 depth   (1 channel)
        - 2*window_half projected neighbour depths (1 channel each)
    Output: refined depth [0, 1].
    """

    def __init__(
        self,
        num_bins:     int = NUM_BINS,
        window_half:  int = 2,
        base:         int = 32,
        num_encoders: int = 3,
        num_residuals:int = 2,
    ):
        super().__init__()
        self.window_half = window_half
        n_nbr  = 2 * window_half
        in_s2  = num_bins + 1 + n_nbr   # voxels + s1_tgt + projected neighbours

        self.s1 = UNet(in_ch=num_bins, base=base,
                       num_encoders=num_encoders, num_residuals=num_residuals)
        self.s2 = UNet(in_ch=in_s2,    base=base,
                       num_encoders=num_encoders, num_residuals=num_residuals)


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------
def run_epoch(
    model:     PoseUNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    K:         torch.Tensor,
    viz_s1:    VizLogger | None = None,
    viz_s2:    VizLogger | None = None,
) -> tuple:
    """
    Run one epoch.  Returns (mean_total_loss, mean_s1_l1_m, mean_s2_l1_m).
    Stage-1 and stage-2 each have their own e2depth loss; total = s1 + s2.
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_s1_l1 = total_s2_l1 = 0.0

    with ctx:
        for batch in loader:
            (vox_t, dep_t, msk_t,
             vox_nbr, dep_nbr, msk_nbr, T_tgt_nbr) = [b.to(device) for b in batch]

            B, nwin, C, H, W = vox_nbr.shape

            # ── Stage 1 ────────────────────────────────────────────────────
            # Run s1 on all frames in one batched forward pass
            # vox_all: (B*(nwin+1), C, H, W)
            vox_all = torch.cat(
                [vox_t.unsqueeze(1), vox_nbr], dim=1
            ).reshape(B * (nwin + 1), C, H, W)

            dep_s1_all = model.s1(vox_all).reshape(B, nwin + 1, 1, H, W)

            dep_t_s1   = dep_s1_all[:, 0]        # (B, 1, H, W) target frame
            dep_nbr_s1 = dep_s1_all[:, 1:]        # (B, nwin, 1, H, W) neighbours

            # Stage-1 loss: average over target + all neighbours
            dep_t_norm   = ((dep_t   - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            dep_nbr_norm = ((dep_nbr - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)

            l_s1_t, _ = compute_loss(dep_t_s1, dep_t_norm, msk_t, vox_t, K=K)

            # Reshape all neighbours into the batch dim → single compute_loss call
            l_s1_nbr, _ = compute_loss(
                dep_nbr_s1.reshape(B * nwin, 1, H, W),
                dep_nbr_norm.reshape(B * nwin, 1, H, W),
                msk_nbr.reshape(B * nwin, 1, H, W),
                vox_nbr.reshape(B * nwin, C, H, W),
                K=K,
            )
            l_s1 = (l_s1_t + l_s1_nbr) * 0.5   # equal weight s1_tgt vs s1_nbr avg

            # ── Projection ────────────────────────────────────────────────
            # Project each neighbour's stage-1 depth into the target frame.
            # Detached: gradients don't flow through the scatter operation.
            proj_depths = []
            for i in range(nwin):
                proj_i = project_depth(
                    dep_nbr_s1[:, i].detach(),
                    T_tgt_nbr[:, i],
                    K,
                )
                proj_depths.append(proj_i)              # (B, 1, H, W)

            # ── Stage 2 ───────────────────────────────────────────────────
            # Input: voxels_t | target_s1_depth | projected_neighbour_depths
            s2_inp = torch.cat(
                [vox_t, dep_t_s1] + proj_depths, dim=1
            )   # (B, C+1+nwin, H, W)

            dep_final = model.s2(s2_inp)   # (B, 1, H, W) in [0, 1]

            l_s2, _ = compute_loss(dep_final, dep_t_norm, msk_t, vox_t, K=K)

            loss = l_s1 + l_s2

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_s1_l1 += _l1_metres(dep_t_s1,  dep_t, msk_t).item()
                total_s2_l1 += _l1_metres(dep_final,  dep_t, msk_t).item()

            # Visualisation (metric depth for VizLogger)
            if viz_s1 is not None:
                pred_s1_m = (dep_t_s1  * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                viz_s1.add_batch(vox_t, dep_t, msk_t, pred_s1_m)
            if viz_s2 is not None:
                pred_s2_m = (dep_final * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                viz_s2.add_batch(vox_t, dep_t, msk_t, pred_s2_m)

    n = max(len(loader), 1)
    return total_loss / n, total_s1_l1 / n, total_s2_l1 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage pose-guided UNet for event-to-depth"
    )
    parser.add_argument("--data_dir",      type=Path, default=DATA_ROOT)
    parser.add_argument("--window_half",   type=int,  default=2,
                        help="Temporal half-window; total window = 2*W+1 frames")
    parser.add_argument("--window_stride", type=int,  default=3,
                        help="Frame stride between neighbours (default 3 = 15 voxel bins apart)")
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--batch_size",    type=int,  default=16,
                        help="Per-batch samples (lower than train_unet due to multi-frame)")
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--workers",       type=int,  default=4)
    parser.add_argument("--base_channels", type=int,  default=32)
    parser.add_argument("--out_dir",       type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "pose_unet")
    parser.add_argument("--seed",          type=int,  default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Discover sequences ────────────────────────────────────────────────
    def _is_sequence(p: Path) -> bool:
        return (
            (p / "events" / "voxels_cam0.h5").exists()
            and (p / "hdf5" / "depth_in_event_frame.h5").exists()
            and (p / "hdf5" / "poses.h5").exists()
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
            "        Make sure voxels, depth, and poses.h5 all exist."
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

    # ── DataLoaders ───────────────────────────────────────────────────────
    W = args.window_half
    S = args.window_stride
    train_ds = ConcatDataset([PoseWindowDataset(d, W, S) for d in train_seqs])
    val_ds   = ConcatDataset([PoseWindowDataset(d, W, S) for d in val_seqs])
    print(f"  Train frames: {len(train_ds)},  Val frames: {len(val_ds)}")
    print(f"  Window: {2*W} neighbours + 1 target = {2*W+1} frames total  "
          f"(stride={S}, max offset={W*S} frames)\n")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PoseUNet(
        num_bins=NUM_BINS,
        window_half=W,
        base=args.base_channels,
    ).to(device)

    # Load and scale event-camera intrinsics
    K = torch.from_numpy(_load_event_K_scaled()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PoseUNet  base={args.base_channels}  window_half={W}  "
          f"parameters: {n_params:,}")
    print(f"  S1 in_ch={NUM_BINS}   S2 in_ch={NUM_BINS + 1 + 2*W}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ── Logging ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))

    viz_s1_train = VizLogger(writer, n_samples=4, tag="viz/s1_train")
    viz_s1_val   = VizLogger(writer, n_samples=4, tag="viz/s1_val")
    viz_s2_train = VizLogger(writer, n_samples=4, tag="viz/s2_train")
    viz_s2_val   = VizLogger(writer, n_samples=4, tag="viz/s2_val")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_s1, tr_s2 = run_epoch(
            model, train_loader, optimizer, device, K,
            viz_s1=viz_s1_train, viz_s2=viz_s2_train,
        )
        va_loss, va_s1, va_s2 = run_epoch(
            model, val_loader, None, device, K,
            viz_s1=viz_s1_val, viz_s2=viz_s2_val,
        )
        scheduler.step()

        viz_s1_train.flush(step=epoch)
        viz_s1_val.flush(step=epoch)
        viz_s2_train.flush(step=epoch)
        viz_s2_val.flush(step=epoch)

        vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
            f"S1 L1: {tr_s1:.4f}/{va_s1:.4f} m  "
            f"S2 L1: {tr_s2:.4f}/{va_s2:.4f} m  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train",    tr_loss, epoch)
        writer.add_scalar("loss/val",      va_loss, epoch)
        writer.add_scalar("l1_s1/train",   tr_s1,   epoch)
        writer.add_scalar("l1_s1/val",     va_s1,   epoch)
        writer.add_scalar("l1_s2/train",   tr_s2,   epoch)
        writer.add_scalar("l1_s2/val",     va_s2,   epoch)
        writer.add_scalar("lr",            scheduler.get_last_lr()[0], epoch)

        if va_s2 < best_val_l1:
            best_val_l1 = va_s2
            torch.save({
                "epoch":         epoch,
                "model_s1":      model.s1.state_dict(),
                "model_s2":      model.s2.state_dict(),
                "val_s2_l1":     va_s2,
                "base":          args.base_channels,
                "window_half":   W,
                "window_stride": S,
            }, args.out_dir / "best.pth")
            print(f"  → new best checkpoint  (val S2 L1 = {va_s2:.4f} m)")

    torch.save({
        "epoch":         args.epochs,
        "model_s1":      model.s1.state_dict(),
        "model_s2":      model.s2.state_dict(),
        "val_s2_l1":     va_s2,
        "base":          args.base_channels,
        "window_half":   W,
        "window_stride": S,
    }, args.out_dir / "last.pth")

    writer.close()
    print(f"\nDone. Best val S2 L1: {best_val_l1:.4f} m")


if __name__ == "__main__":
    main()
