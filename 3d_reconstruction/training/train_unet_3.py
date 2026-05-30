#!/usr/bin/env python3
"""
train_unet_3.py - Multi-frame UNet for event-to-depth with pose ablations.

Feeds the U-Net the target event voxels together with one previous and
one next neighbour's voxels and their relative poses:

    [x_t | x_{t-s} | x_{t+s} | p_{t-s→t} | p_{t+s→t}]

    x_t          : target event voxel grid          (NUM_BINS channels)
    x_{t±s}      : neighbouring event voxels        (NUM_BINS channels each)
    p_{i→t}      : relative pose encoded as         (6 channels each)
                   constant-image channels
                   [tx/T_SCALE, ty/T_SCALE, tz/T_SCALE, rx, ry, rz]
                   rotation as axis-angle

Default input channels: 5 + 5 + 5 + 6 + 6 = 27
Default stride: 3 frames  (15 event-bin offset)

Usage:
    python3 training/train_unet_3.py
    python3 training/train_unet_3.py --stride 3 --data_dir data/lego/lego_1
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
    UNet, compute_loss, _l1_metres, _get_pixel_grid,
)
from viz import VizLogger

# Camera calibration paths
_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"

# Translation normalisation scales [metres]
T_SCALE     = 0.1   # relative translations between nearby frames (~cm range)
T_SCALE_ABS = 1.0   # absolute world-frame positions (~m range)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_event_K_scaled(h_crop: int = 240, w_crop: int = 320,
                          h_full: int = 720, w_full: int = 1280) -> np.ndarray:
    """Load and scale event-camera intrinsics to the voxel crop resolution."""
    K = np.load(_CAM_DATA / "event_intrinsics.npz")["camera_matrix"].astype(np.float32).copy()
    K[0, :] *= w_crop / w_full
    K[1, :] *= h_crop / h_full
    return K  # (3, 3)


def _load_T_event_from_ee() -> np.ndarray:
    """T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee  (4×4 float64)."""
    T_er = np.load(_CAM_DATA / "T_event_from_rgb.npz")["T"]
    T_re = np.load(_CAM_DATA / "T_rgb_from_ee.npz")["T"]
    return (T_er @ T_re).astype(np.float64)


# ---------------------------------------------------------------------------
# Pose encoding
# ---------------------------------------------------------------------------

def _rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix → (3,) axis-angle vector (Rodrigues formula)."""
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(angle) < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float32) / (2.0 * np.sin(angle))
    return (axis * angle).astype(np.float32)


def _pose_to_map(T: np.ndarray, H: int, W: int,
                  t_scale: float = T_SCALE) -> np.ndarray:
    """
    Encode a 4×4 pose matrix as a (6, H, W) constant-channel image.

    Channels: [tx/t_scale, ty/t_scale, tz/t_scale, rx, ry, rz]
    where (rx, ry, rz) is the axis-angle rotation vector.
    Use t_scale=T_SCALE_ABS for world-frame (absolute) poses.
    """
    t = (T[:3, 3] / t_scale).astype(np.float32)   # (3,)
    r = _rot_to_axis_angle(T[:3, :3])               # (3,)
    pose_vec = np.concatenate([t, r])                # (6,)
    return np.broadcast_to(pose_vec[:, None, None], (6, H, W)).copy()  # (6, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiFrameDepthDataset(Dataset):
    """
    Per-frame dataset returning target + prev + next event frames and
    their relative-pose maps.

    The temporal stride between target t and its neighbours is configurable:
        prev = t - stride,  next = t + stride

    Returns (all torch.Tensor):
        vox_t        : (C, H, W)    target event voxels
        dep_t        : (1, H, W)    GT depth [m]
        mask_t       : (1, H, W)    valid depth mask
        vox_prev     : (C, H, W)    prev-frame voxels  (t - stride)
        vox_next     : (C, H, W)    next-frame voxels  (t + stride)
        pose_prev_rel: (6, H, W)    T_tgt_from_prev  (relative, T_SCALE)
        pose_next_rel: (6, H, W)    T_tgt_from_next  (relative, T_SCALE)
        pose_t_abs   : (6, H, W)    target absolute pose  (T_SCALE_ABS)
        pose_prev_abs: (6, H, W)    prev absolute pose    (T_SCALE_ABS)
        pose_next_abs: (6, H, W)    next absolute pose    (T_SCALE_ABS)
    """

    def __init__(self, seq_dir: Path, stride: int = 3,
                 split: str = "all", val_ratio: float = 0.15):
        super().__init__()
        self.seq_dir     = seq_dir
        self.stride      = stride
        self.voxels_path = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path  = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.mask_path   = seq_dir / "hdf5"   / "spatial_mask.h5"
        self.poses_path  = seq_dir / "hdf5"   / "poses.h5"

        import h5py
        with h5py.File(self.depth_path,  "r") as f: n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path, "r") as f: n_v = f["voxels"].shape[0]
        with h5py.File(self.poses_path,  "r") as f: n_p = f["ee_T"].shape[0]

        n_frames      = min(n_d, n_v, n_p)
        self.has_mask = self.mask_path.exists()

        # Precompute T_world_from_event for every frame
        T_event_from_ee = _load_T_event_from_ee()          # (4, 4)
        T_ee_from_event = np.linalg.inv(T_event_from_ee)   # (4, 4)
        with h5py.File(self.poses_path, "r") as f:
            ee_T = f["ee_T"][:n_frames]                     # (N, 4, 4)
        self.T_world_from_event = (ee_T @ T_ee_from_event[None]).astype(np.float32)

        # Valid target indices: neighbours at ±stride must exist
        self.valid_indices = np.arange(stride, n_frames - stride)

        if split not in ("all", "train", "val"):
            raise ValueError(f"Unknown split: {split}")
        if split != "all" and len(self.valid_indices) > 1:
            n = len(self.valid_indices)
            n_val = max(1, int(round(n * float(np.clip(val_ratio, 0.01, 0.99)))))
            n_train = max(1, n - n_val)
            # Temporal holdout: use the tail of each sequence as validation.
            self.valid_indices = self.valid_indices[:n_train] if split == "train" else self.valid_indices[n_train:]

        self._vox = None   # lazy HDF5 handles, opened per DataLoader worker
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

    def _load_voxels(self, idx: int) -> torch.Tensor:
        """Load only voxels for frame idx — no depth/mask I/O."""
        vox = self._vox[idx]
        if vox.dtype == np.float16:
            vox = vox.astype(np.float32)
        return torch.from_numpy(vox)   # (C, Hv, Wv)

    def _load_frame(self, idx: int):
        """Load (voxels, depth, mask) for frame idx."""
        vox = self._vox[idx]
        if vox.dtype == np.float16:
            vox = vox.astype(np.float32)
        dep   = self._dep[idx].astype(np.float32)
        valid = (dep > 0).astype(np.float32)
        if self.has_mask:
            valid *= self._msk[idx].astype(np.float32)

        vox_t = torch.from_numpy(vox)
        dep_t = torch.from_numpy(dep).unsqueeze(0)
        msk_t = torch.from_numpy(valid).unsqueeze(0)

        _, Hv, Wv = vox_t.shape
        if dep_t.shape[-2] != Hv or dep_t.shape[-1] != Wv:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)

        return vox_t, dep_t, msk_t

    def __getitem__(self, item: int):
        self._open()
        t = int(self.valid_indices[item])
        S = self.stride

        # Target frame (voxels + depth + mask)
        vox_t, dep_t, mask_t = self._load_frame(t)
        _, H, W = vox_t.shape

        # Neighbour voxels only (no depth needed)
        vox_prev = self._load_voxels(t - S)   # (C, H, W)
        vox_next = self._load_voxels(t + S)   # (C, H, W)

        # Relative poses: T_tgt_from_nbr
        T_tgt_from_world = np.linalg.inv(self.T_world_from_event[t]).astype(np.float32)
        T_tgt_from_prev  = (T_tgt_from_world @ self.T_world_from_event[t - S]).astype(np.float32)
        T_tgt_from_next  = (T_tgt_from_world @ self.T_world_from_event[t + S]).astype(np.float32)

        pose_prev_rel = torch.from_numpy(_pose_to_map(T_tgt_from_prev, H, W))  # (6, H, W)
        pose_next_rel = torch.from_numpy(_pose_to_map(T_tgt_from_next, H, W))  # (6, H, W)

        # Absolute pose maps (world frame, normalised by T_SCALE_ABS)
        pose_t_abs    = torch.from_numpy(
            _pose_to_map(self.T_world_from_event[t],     H, W, t_scale=T_SCALE_ABS))
        pose_prev_abs = torch.from_numpy(
            _pose_to_map(self.T_world_from_event[t - S], H, W, t_scale=T_SCALE_ABS))
        pose_next_abs = torch.from_numpy(
            _pose_to_map(self.T_world_from_event[t + S], H, W, t_scale=T_SCALE_ABS))

        return (vox_t, dep_t, mask_t, vox_prev, vox_next,
                pose_prev_rel, pose_next_rel,
                pose_t_abs, pose_prev_abs, pose_next_abs)


def inverse_warp_voxels(
    vox_src: torch.Tensor,
    depth_tgt_norm: torch.Tensor,
    T_tgt_from_src: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp source voxel grid into the target view using target-depth hypothesis.

    This is the inverse-warp formulation: each target pixel is backprojected
    with the current target depth estimate, transformed into the source frame,
    and sampled from the source voxel grid with bilinear interpolation.

    Returns:
        warped_vox: (B, C, H, W)
        valid_mask: (B, 1, H, W)
    """
    B, C, H, W = vox_src.shape
    device = vox_src.device

    depth_tgt_m = depth_tgt_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    K_f = K.to(device=device, dtype=torch.float32)
    K_inv = torch.linalg.inv(K_f)
    T_src_from_tgt = torch.linalg.inv(T_tgt_from_src.to(device=device, dtype=torch.float32))

    _, _, pix = _get_pixel_grid(H, W, device)
    N = H * W
    pix_b = pix.unsqueeze(0).expand(B, -1, -1)
    rays_t = torch.bmm(K_inv.unsqueeze(0).expand(B, -1, -1), pix_b)
    depth_flat = depth_tgt_m.reshape(B, 1, N)
    pts_t = rays_t * depth_flat
    pts_th = torch.cat([pts_t, torch.ones(B, 1, N, device=device)], dim=1)

    pts_s = torch.bmm(T_src_from_tgt, pts_th)[:, :3]
    z_s = pts_s[:, 2].clamp_min(1e-6)
    uvz_s = torch.bmm(K_f.unsqueeze(0).expand(B, -1, -1), pts_s)
    u_s = uvz_s[:, 0] / z_s
    v_s = uvz_s[:, 1] / z_s

    valid = (pts_s[:, 2] > 0) & (u_s >= 0) & (u_s <= (W - 1)) & (v_s >= 0) & (v_s <= (H - 1))
    grid_x = (u_s / max(W - 1, 1)) * 2.0 - 1.0
    grid_y = (v_s / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(B, H, W, 2)

    warped = F.grid_sample(
        vox_src,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    valid_mask = valid.view(B, 1, H, W).float()
    return warped * valid_mask, valid_mask


class WarpedVoxelUNet(nn.Module):
    """Two-stage model: target-only depth proposal, then warped-neighbour refinement."""

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        base: int = 32,
    ):
        super().__init__()
        self.s1 = UNet(in_ch=num_bins, base=base)
        self.s2 = UNet(in_ch=3 * num_bins + 3, base=base)


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

# Input-channel groups for attribution
# With pose (27 ch):  vox_t | vox_prev | vox_next | pose_prev | pose_next
_ATTR_GROUPS_POSE: list[tuple[str, slice]] = [
    ("vox_t",    slice(0,            NUM_BINS)),
    ("vox_prev", slice(NUM_BINS,         2 * NUM_BINS)),
    ("vox_next", slice(2 * NUM_BINS,     3 * NUM_BINS)),
    ("pose_prev",slice(3 * NUM_BINS,     3 * NUM_BINS + 6)),
    ("pose_next",slice(3 * NUM_BINS + 6, 3 * NUM_BINS + 12)),
]
# Without pose (15 ch): vox_t | vox_prev | vox_next
_ATTR_GROUPS_VOX: list[tuple[str, slice]] = [
    ("vox_t",    slice(0,            NUM_BINS)),
    ("vox_prev", slice(NUM_BINS,     2 * NUM_BINS)),
    ("vox_next", slice(2 * NUM_BINS, 3 * NUM_BINS)),
]
# Absolute pose (33 ch): vox_t | vox_prev | vox_next | pose_t_abs | pose_prev_abs | pose_next_abs
_ATTR_GROUPS_ABS: list[tuple[str, slice]] = [
    ("vox_t",         slice(0,                  NUM_BINS)),
    ("vox_prev",      slice(NUM_BINS,           2 * NUM_BINS)),
    ("vox_next",      slice(2 * NUM_BINS,       3 * NUM_BINS)),
    ("pose_t_abs",    slice(3 * NUM_BINS,       3 * NUM_BINS + 6)),
    ("pose_prev_abs", slice(3 * NUM_BINS + 6,   3 * NUM_BINS + 12)),
    ("pose_next_abs", slice(3 * NUM_BINS + 12,  3 * NUM_BINS + 18)),
]

_ATTR_GROUPS_TGT: list[tuple[str, slice]] = [
    ("vox_t", slice(0, NUM_BINS)),
]


def _adapt_voxel_bins(vox: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Adapt voxel tensor (B, C, H, W) to exactly n_bins channels.

    If C > n_bins, truncates to the first n_bins channels.
    If C < n_bins, zero-pads missing channels.
    """
    c = vox.shape[1]
    if c == n_bins:
        return vox
    if c > n_bins:
        return vox[:, :n_bins]
    pad = torch.zeros(
        (vox.shape[0], n_bins - c, vox.shape[2], vox.shape[3]),
        dtype=vox.dtype,
        device=vox.device,
    )
    return torch.cat([vox, pad], dim=1)


def _silog_loss(pred_m: torch.Tensor, dep_m: torch.Tensor,
                mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant log-depth loss on valid pixels only."""
    valid = (mask > 0.5) & (dep_m > 0.0) & (pred_m > 0.0)
    if not valid.any():
        return pred_m.new_tensor(0.0)
    d = torch.log(pred_m[valid] + eps) - torch.log(dep_m[valid] + eps)
    return torch.sqrt(torch.clamp(d.pow(2).mean() - 0.5 * d.mean().pow(2), min=0.0))


def _grad_loss(pred_m: torch.Tensor, dep_m: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Edge-aware L1 on first-order depth gradients over valid neighbours."""
    valid = mask > 0.5
    px = pred_m[:, :, :, 1:] - pred_m[:, :, :, :-1]
    gx = dep_m[:, :, :, 1:] - dep_m[:, :, :, :-1]
    mx = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    py = pred_m[:, :, 1:, :] - pred_m[:, :, :-1, :]
    gy = dep_m[:, :, 1:, :] - dep_m[:, :, :-1, :]
    my = valid[:, :, 1:, :] & valid[:, :, :-1, :]

    loss = pred_m.new_tensor(0.0)
    if mx.any():
        loss = loss + (px[mx] - gx[mx]).abs().mean()
    if my.any():
        loss = loss + (py[my] - gy[my]).abs().mean()
    return loss


def run_epoch(
    model,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    K:         torch.Tensor,
    viz=None,
    compute_attribution: bool = False,
    pose_mode: str = "relative",
    input_mode: str = "target_neighbors_pose",
    target_nbins: int = 15,
    silog_weight: float = 0.1,
    grad_weight: float = 0.1,
    log_interval_s: float = 20.0,
) -> tuple:
    """
    One epoch.  Returns (mean_total_loss, mean_l1_metres, attribution | None).

    pose_mode:
        "relative"  -> cat([vox_t, vox_prev, vox_next, pose_prev_rel, pose_next_rel])       27 ch
        "absolute"  -> cat([vox_t, vox_prev, vox_next, pose_t_abs, pose_prev_abs, pose_next_abs]) 33 ch
        "none"      -> cat([vox_t, vox_prev, vox_next])                                     15 ch

    attribution is a dict {group_name: relative_contribution} when
    compute_attribution=True and is_train=True, mapping each input group to
    its share of the mean absolute input gradient (sums to 1.0).
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    use_warped_model = input_mode == "target_neighbors_pose"

    total_loss = total_l1 = 0.0
    total_base = total_silog = total_grad = 0.0
    attr_accum = None   # running sum of per-channel mean |∂loss/∂inp|
    n_batches = 0
    t_last_log = time.time()
    phase = "train" if is_train else "val"

    with ctx:
        for batch in loader:
            (vox_t, dep_t, mask_t, vox_prev, vox_next,
             pose_prev_rel, pose_next_rel,
             pose_t_abs, pose_prev_abs, pose_next_abs) = [b.to(device) for b in batch]

            # Normalise GT depth to [0, 1]
            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)

            if use_warped_model:
                # Proper geometric usage of neighbours: predict target depth,
                # inverse-warp prev/next voxel grids into the target view, then refine.
                dep_s1 = model.s1(vox_t)
                if pose_mode == "absolute":
                    raise ValueError("Absolute pose mode is not supported with geometric warping.")
                T_tgt_from_prev = pose_prev_rel.new_zeros((pose_prev_rel.shape[0], 4, 4))
                T_tgt_from_next = pose_next_rel.new_zeros((pose_next_rel.shape[0], 4, 4))
                # Reconstruct rigid transforms from dataset tensors passed in the batch.
                # The dataset already carries the exact relative pose maps as constant images,
                # but geometric warping needs the underlying 4x4 transforms, so recover them
                # from the per-batch absolute poses when available.
                if pose_mode == "relative":
                    # Use absolute pose maps to form target-from-neighbour transforms.
                    def _map_to_T(pose_map: torch.Tensor, t_scale: float) -> torch.Tensor:
                        pose_vec = pose_map[:, :, 0, 0]
                        t = pose_vec[:, :3] * t_scale
                        r = pose_vec[:, 3:]
                        angle = torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1e-8)
                        axis = r / angle
                        K_skew = torch.zeros((r.shape[0], 3, 3), device=r.device, dtype=r.dtype)
                        K_skew[:, 0, 1] = -axis[:, 2]
                        K_skew[:, 0, 2] = axis[:, 1]
                        K_skew[:, 1, 0] = axis[:, 2]
                        K_skew[:, 1, 2] = -axis[:, 0]
                        K_skew[:, 2, 0] = -axis[:, 1]
                        K_skew[:, 2, 1] = axis[:, 0]
                        eye = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0).expand(r.shape[0], -1, -1)
                        sin = torch.sin(angle).unsqueeze(-1)
                        cos = torch.cos(angle).unsqueeze(-1)
                        R = eye + sin * K_skew + (1.0 - cos) * torch.bmm(K_skew, K_skew)
                        small = (torch.linalg.norm(r, dim=1) < 1e-6)
                        if small.any():
                            R[small] = eye[small]
                        T = torch.eye(4, device=r.device, dtype=r.dtype).unsqueeze(0).repeat(r.shape[0], 1, 1)
                        T[:, :3, :3] = R
                        T[:, :3, 3] = t
                        return T

                    T_world_t = _map_to_T(pose_t_abs, T_SCALE_ABS)
                    T_world_prev = _map_to_T(pose_prev_abs, T_SCALE_ABS)
                    T_world_next = _map_to_T(pose_next_abs, T_SCALE_ABS)
                    T_tgt_from_prev = torch.bmm(torch.linalg.inv(T_world_t), T_world_prev)
                    T_tgt_from_next = torch.bmm(torch.linalg.inv(T_world_t), T_world_next)
                else:
                    T_tgt_from_prev = torch.eye(4, device=device).unsqueeze(0).repeat(vox_t.shape[0], 1, 1)
                    T_tgt_from_next = torch.eye(4, device=device).unsqueeze(0).repeat(vox_t.shape[0], 1, 1)

                vox_prev_warp, prev_valid = inverse_warp_voxels(vox_prev, dep_s1, T_tgt_from_prev, K)
                vox_next_warp, next_valid = inverse_warp_voxels(vox_next, dep_s1, T_tgt_from_next, K)
                s2_inp = torch.cat([vox_t, vox_prev_warp, vox_next_warp, dep_s1, prev_valid, next_valid], dim=1)
                pred = model.s2(s2_inp)

                base_loss_s1, _ = compute_loss(dep_s1, dep_norm, mask_t, vox_t, K=K)
                base_loss_s2, _ = compute_loss(pred, dep_norm, mask_t, vox_t, K=K)
                pred_s1_m = dep_s1 * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                silog_s1 = _silog_loss(pred_s1_m, dep_t, mask_t)
                silog_s2 = _silog_loss(pred_m, dep_t, mask_t)
                grad_s1 = _grad_loss(pred_s1_m, dep_t, mask_t)
                grad_s2 = _grad_loss(pred_m, dep_t, mask_t)
                base_loss = base_loss_s1 + base_loss_s2
                silog = silog_s1 + silog_s2
                grad = grad_s1 + grad_s2
                loss = base_loss + silog_weight * silog + grad_weight * grad
            else:
                # Build input tensor for the selected ablation mode.
                if input_mode == "target":
                    inp = vox_t
                elif input_mode == "target_nbins":
                    inp = _adapt_voxel_bins(vox_t, target_nbins)
                else:
                    inp = torch.cat([vox_t, vox_prev, vox_next], dim=1)

                if compute_attribution and is_train:
                    inp.requires_grad_(True)

                pred = model(inp)
                base_loss, _ = compute_loss(pred, dep_norm, mask_t, vox_t, K=K)
                pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                silog = _silog_loss(pred_m, dep_t, mask_t)
                grad = _grad_loss(pred_m, dep_t, mask_t)
                loss = base_loss + silog_weight * silog + grad_weight * grad

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Accumulate mean absolute gradient per input channel
                if (not use_warped_model) and compute_attribution and inp.grad is not None:
                    g = inp.grad.detach().abs().mean(dim=(0, 2, 3))  # (C_in,)
                    attr_accum = g if attr_accum is None else attr_accum + g
                optimizer.step()

            n_batches += 1
            total_loss += loss.item()
            total_base += base_loss.item()
            total_silog += silog.item()
            total_grad += grad.item()
            with torch.no_grad():
                total_l1 += _l1_metres(pred, dep_t, mask_t).item()

            if viz is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                viz.add_batch(vox_t, dep_t, mask_t, pred_m)

            now = time.time()
            if log_interval_s > 0 and (now - t_last_log) >= log_interval_s:
                print(
                    f"  [{phase:5s} {n_batches:4d}/{len(loader):4d}] "
                    f"loss {total_loss / n_batches:.4f} "
                    f"(base {total_base / n_batches:.4f}, "
                    f"silog {total_silog / n_batches:.4f}, "
                    f"grad {total_grad / n_batches:.4f}) "
                    f"L1 {total_l1 / n_batches:.4f} m"
                )
                t_last_log = now

    n = max(len(loader), 1)

    # Compute relative per-group attribution (fraction of total gradient signal)
    attribution = None
    if attr_accum is not None:
        g      = attr_accum / n
        if input_mode == "target":
            groups = _ATTR_GROUPS_TGT
        elif input_mode == "target_nbins":
            groups = [("vox_t_bins", slice(0, target_nbins))]
        elif input_mode == "target_neighbors":
            groups = _ATTR_GROUPS_VOX
        else:
            groups = {"relative": _ATTR_GROUPS_POSE,
                      "absolute": _ATTR_GROUPS_ABS,
                      "none":     _ATTR_GROUPS_VOX}[pose_mode]
        raw    = {name: g[sl].mean().item() for name, sl in groups}
        total  = sum(raw.values()) + 1e-8
        attribution = {k: v / total for k, v in raw.items()}

    return total_loss / n, total_l1 / n, attribution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pose-aware multi-frame UNet for event-to-depth"
    )
    parser.add_argument("--data_dir",      type=Path, default=DATA_ROOT,
                        help="Single sequence dir or parent of multiple sequences")
    parser.add_argument("--stride",        type=int,  default=3,
                        help="Frame stride to neighbours (default 3 = 15 event-bin offset)")
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--batch_size",    type=int,  default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--workers",       type=int,  default=4)
    parser.add_argument("--base_channels", type=int,  default=32)
    parser.add_argument("--out_dir",       type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "unet_3")
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--input_mode",    type=str, default="target_neighbors_pose",
                        choices=("target", "target_nbins", "target_neighbors", "target_neighbors_pose"),
                        help="Input ablation mode")
    parser.add_argument("--target_nbins",  type=int, default=15,
                        help="Number of voxel bins used by --input_mode target_nbins")
    parser.add_argument("--no_pose",   action="store_true",
                        help="Feed only voxel channels (15 ch); omit pose maps")
    parser.add_argument("--abs_pose",  action="store_true",
                        help="Use absolute world-frame poses (33 ch) instead of relative (27 ch). "
                            "Usually overfits; relative poses are recommended.")
    parser.add_argument("--single_object_val_ratio", type=float, default=0.15,
                        help="When only one sequence/object is found, hold out this tail fraction for validation")
    parser.add_argument("--silog_weight", type=float, default=0.1,
                        help="Weight for scale-invariant log-depth loss")
    parser.add_argument("--grad_weight", type=float, default=0.1,
                        help="Weight for depth-gradient loss")
    parser.add_argument("--log_interval_s", type=float, default=20.0,
                        help="Print running batch metrics every N seconds (0 disables)")
    parser.add_argument("--name",      type=str, default=None,
                        help="Run name used in checkpoint filenames (best_<name>.pth / last_<name>.pth). "
                             "Prompted interactively if not provided.")
    parser.add_argument("--tb_run_name", type=str, default="",
                        help="Optional TensorBoard run subdir name under <out_dir>/tb. "
                            "Defaults to --name")
    args = parser.parse_args()

    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

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
        print(f"  Validation split: last {args.single_object_val_ratio * 100:.1f}% of each sequence")
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
    S = args.stride
    if single_object:
        train_ds = ConcatDataset([
            MultiFrameDepthDataset(d, S, split="train", val_ratio=args.single_object_val_ratio)
            for d in train_seqs
        ])
        val_ds = ConcatDataset([
            MultiFrameDepthDataset(d, S, split="val", val_ratio=args.single_object_val_ratio)
            for d in val_seqs
        ])
    else:
        train_ds = ConcatDataset([MultiFrameDepthDataset(d, S) for d in train_seqs])
        val_ds   = ConcatDataset([MultiFrameDepthDataset(d, S) for d in val_seqs])
    print(f"  Train frames: {len(train_ds)},  Val frames: {len(val_ds)}")
    print(f"  Neighbours: t±{S} frames  ({S * NUM_BINS} event-bin offset)  "
          f"T_SCALE={T_SCALE} m\n")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ── Pose mode & Model ───────────────────────────────────────────────────
    if args.no_pose and args.abs_pose:
        sys.exit("[ERROR] --no_pose and --abs_pose are mutually exclusive")
    pose_mode = "none" if args.no_pose else "absolute" if args.abs_pose else "relative"
    if args.input_mode != "target_neighbors_pose":
        pose_mode = "none"
    elif pose_mode == "absolute":
        sys.exit("[ERROR] Geometric warping currently supports relative poses only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.input_mode == "target":
        in_ch = NUM_BINS
    elif args.input_mode == "target_nbins":
        in_ch = args.target_nbins
    elif args.input_mode == "target_neighbors":
        in_ch = 3 * NUM_BINS
    else:
        in_ch = 3 * NUM_BINS + 3
    if args.input_mode == "target_neighbors_pose":
        model = WarpedVoxelUNet(num_bins=NUM_BINS, base=args.base_channels).to(device)
    else:
        model = UNet(in_ch=in_ch, base=args.base_channels).to(device)

    K = torch.from_numpy(_load_event_K_scaled()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet  in_ch={in_ch}  base={args.base_channels}  parameters: {n_params:,}  "
          f"[input_mode={args.input_mode}, pose_mode={pose_mode}]")
    if args.input_mode == "target":
        ch_desc = f"{NUM_BINS} (vox_t)"
    elif args.input_mode == "target_nbins":
        ch_desc = f"{args.target_nbins} (adapted vox_t bins)"
    elif args.input_mode == "target_neighbors":
        ch_desc = f"{NUM_BINS} (vox_t) + {NUM_BINS} (vox_prev) + {NUM_BINS} (vox_next)"
    else:
        ch_desc = (f"{NUM_BINS} (vox_t) + {NUM_BINS} (warped vox_prev) + {NUM_BINS} (warped vox_next)"
                   f" + 1 (stage-1 depth) + 1 (prev valid mask) + 1 (next valid mask)")
    print(f"  {ch_desc} = {in_ch} channels")
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ── Logging ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_run_name = args.tb_run_name.strip() if args.tb_run_name else args.name
    tb_logdir = args.out_dir / "tb" / tb_run_name
    writer = SummaryWriter(log_dir=str(tb_logdir))
    print(f"TensorBoard run dir: {tb_logdir}")

    viz_train = VizLogger(writer, n_samples=4, tag="viz/train")
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, attr = run_epoch(
            model, train_loader, optimizer, device, K,
            viz=viz_train, compute_attribution=True, pose_mode=pose_mode,
            input_mode=args.input_mode,
            target_nbins=args.target_nbins,
            silog_weight=args.silog_weight,
            grad_weight=args.grad_weight,
            log_interval_s=args.log_interval_s,
        )
        va_loss, va_l1, _ = run_epoch(
            model, val_loader, None, device, K,
            viz=viz_val, pose_mode=pose_mode,
            input_mode=args.input_mode,
            target_nbins=args.target_nbins,
            silog_weight=args.silog_weight,
            grad_weight=args.grad_weight,
            log_interval_s=args.log_interval_s,
        )
        scheduler.step()

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)

        vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
            f"L1: {tr_l1:.4f}/{va_l1:.4f} m  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val",   va_loss, epoch)
        writer.add_scalar("l1/train",   tr_l1,   epoch)
        writer.add_scalar("l1/val",     va_l1,   epoch)
        writer.add_scalar("lr",         scheduler.get_last_lr()[0], epoch)

        # Log input attribution every epoch; print every 5
        if attr is not None:
            for name, val in attr.items():
                writer.add_scalar(f"attribution/{name}", val, epoch)
            if epoch % 5 == 0 or epoch == 1:
                parts = "  ".join(f"{k}={v:.1%}" for k, v in attr.items())
                print(f"  Attribution: {parts}")

        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "val_l1":   va_l1,
                "base":     args.base_channels,
                "in_ch":    in_ch,
                "stride":   S,
                "pose_mode": pose_mode,
                "input_mode": args.input_mode,
                "target_nbins": args.target_nbins,
                "silog_weight": args.silog_weight,
                "grad_weight": args.grad_weight,
            }, args.out_dir / f"best_{args.name}.pth")
            print(f"  → new best checkpoint  (val L1 = {va_l1:.4f} m)")

    torch.save({
        "epoch":    args.epochs,
        "model":    model.state_dict(),
        "val_l1":   va_l1,
        "base":     args.base_channels,
        "in_ch":    in_ch,
        "stride":   S,
        "pose_mode": pose_mode,
        "input_mode": args.input_mode,
        "target_nbins": args.target_nbins,
        "silog_weight": args.silog_weight,
        "grad_weight": args.grad_weight,
    }, args.out_dir / f"last_{args.name}.pth")

    writer.close()
    print(f"\nDone. Best val L1: {best_val_l1:.4f} m")


if __name__ == "__main__":
    main()
