#!/usr/bin/env python3
"""
train_unet_table.py - Early-fusion UNet with table-plane prior channels.

Feeds the U-Net one or more temporal indices. Each index contributes its event
voxels and one channel encoding the table plane directly in the image plane:

    [x_t | table_t | x_t-k | table_t-k | x_t+k | table_t+k | ...]

    x_t                 : target event voxel grid    (NUM_BINS channels)
    table_plane_channel : per-pixel z-depth [m] to the table plane,
                          normalised to [0, 1] using the training depth range.
                          Computed by intersecting per-pixel camera rays with
                          the horizontal plane  z = table_z  in the robot
                          base frame, using the frame's end-effector pose.

Input channels: num_views * (NUM_BINS + 1). With --pose_channels, each view
gets six additional constant pose channels:

    [event camera x, y, z in robot base frame,
     event camera optical-axis direction x, y, z in robot base frame]

The default remains one view without pose channels.

Usage:
    python3 training/train_unet_table.py
    python3 training/train_unet_table.py --data_dir data/lego/lego_1
    python3 training/train_unet_table.py --table_z 0.02
    python3 training/train_unet_table.py --model_scale 2.0
    python3 training/train_unet_table.py --num_views 5 --view_interval 5
"""

import argparse
import copy
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
    UNet, compute_loss, _l1_metres, _worst_percent_l1_metres,
)
from config import TRAIN_CROP_HW, TRAIN_RESIZE_HW
from tensorboard_helper import (
    DEFAULT_TB_ROOT,
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
    tensorboard_run_dir,
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


def _load_T_event_from_ee() -> np.ndarray:
    """Return T_event_from_ee from saved RGB/event hand-eye calibration."""
    T_rgb_from_ee = np.load(_CAM_DATA / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(_CAM_DATA / "T_event_from_rgb.npz")["T"].astype(np.float32)
    return T_event_from_rgb @ T_rgb_from_ee


def _pose_channels_from_base_event(T_base_from_event: np.ndarray) -> np.ndarray:
    """Return six pose values: event-camera position and optical axis in base frame."""
    position = T_base_from_event[:3, 3].astype(np.float32)
    optical_axis = T_base_from_event[:3, 2].astype(np.float32)
    norm = float(np.linalg.norm(optical_axis))
    if norm > 1e-6:
        optical_axis = optical_axis / norm
    return np.concatenate([position, optical_axis]).astype(np.float32)


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
    # Transform K through the same resize + centred crop as the voxel grid.
    resize_H, resize_W = TRAIN_RESIZE_HW
    K = K_native.copy().astype(np.float64)
    K[0, :] *= resize_W / native_W
    K[1, :] *= resize_H / native_H
    K[0, 2] -= (resize_W - vox_W) / 2.0
    K[1, 2] -= (resize_H - vox_H) / 2.0
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
    Early-fusion dataset: target/source event voxels and table priors → target GT depth.

    Requires hdf5/table_plane.h5 produced by
    data_precomputation/precompute_table_plane.py.

    Returns (all torch.Tensor on CPU):
        inp    : (num_views * (NUM_BINS + 1), H, W), target view first
        dep_t  : (1, H, W)             GT depth [m]
        mask_t : (1, H, W)             valid depth mask
    """

    def __init__(
        self,
        seq_dir: Path,
        use_mask: bool = True,
        fill_invalid: bool = False,
        num_views: int = 1,
        view_interval: int = 5,
        pose_channels: bool = False,
        pose_view_selection: bool = False,
        pose_move_threshold: float = 0.01,
        recurrent: bool = False,
        recurrent_enrollment_range: int = 0,
    ):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be at least 1")
        if view_interval < 1:
            raise ValueError("view_interval must be at least 1")
        if pose_view_selection and num_views % 2 != 1:
            raise ValueError(
                "--pose_view_selection requires odd --num_views so source views "
                "can be balanced before and after the target"
            )
        if pose_move_threshold <= 0:
            raise ValueError(f"pose_move_threshold must be > 0, got {pose_move_threshold}")
        if recurrent and num_views != 1:
            raise ValueError("--recurrent only supports single-view input; set --num_views 1")
        if recurrent and pose_view_selection:
            raise ValueError("--recurrent cannot be combined with --pose_view_selection")
        if recurrent_enrollment_range < 0:
            raise ValueError("recurrent_enrollment_range must be >= 0")
        self.seq_dir         = seq_dir
        self.fill_invalid    = fill_invalid
        self.num_views       = int(num_views)
        self.view_interval   = int(view_interval)
        self.pose_channels   = bool(pose_channels)
        self.pose_view_selection = bool(pose_view_selection)
        self.pose_move_threshold = float(pose_move_threshold)
        self.recurrent = bool(recurrent)
        self.recurrent_enrollment_range = int(recurrent_enrollment_range)
        self.voxels_path     = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path      = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.mask_path       = seq_dir / "hdf5"   / "spatial_mask.h5"
        self.poses_path      = seq_dir / "hdf5"   / "poses.h5"
        self.table_plane_path = seq_dir / "hdf5"  / "table_plane.h5"

        if not self.table_plane_path.exists():
            raise FileNotFoundError(
                f"Missing: {self.table_plane_path}\n"
                "Run data_precomputation/precompute_table_plane.py first."
            )

        import h5py
        with h5py.File(self.depth_path,       "r") as f: n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path,      "r") as f: n_v = f["voxels"].shape[0]
        with h5py.File(self.table_plane_path, "r") as f:
            n_t = f["table_plane"].shape[0]
            transform = f.attrs.get("intrinsics_transform", "")
            if isinstance(transform, bytes):
                transform = transform.decode("utf-8", errors="replace")
            corrected_table_transform = transform == "resize_center_crop"
        if not corrected_table_transform:
            raise RuntimeError(
                f"{self.table_plane_path} uses obsolete direct-scaling geometry. "
                "Regenerate it with: python3 "
                "data_precomputation/precompute_table_plane.py --overwrite "
                f"--data_dir {self.seq_dir}"
            )
        needs_poses = self.pose_channels or self.pose_view_selection
        if needs_poses:
            if not self.poses_path.exists():
                raise FileNotFoundError(
                    f"Missing: {self.poses_path}\n"
                    "--pose_channels/--pose_view_selection need recorded end-effector poses."
                )
            with h5py.File(self.poses_path, "r") as f:
                ee_T_all = f["ee_T"][:].astype(np.float32)
                n_p = ee_T_all.shape[0]
            self.T_ee_from_event = np.linalg.inv(_load_T_event_from_ee()).astype(np.float32)
        else:
            ee_T_all = None
            n_p = n_d

        n_frames      = min(n_d, n_v, n_t, n_p)
        self.has_mask = use_mask and self.mask_path.exists()

        self.pose_view_ids: dict[int, list[int]] = {}
        if self.pose_view_selection:
            ee_T_all = ee_T_all[:n_frames]
            T_base_from_event_all = (
                ee_T_all @ self.T_ee_from_event[None]
            ).astype(np.float32)
            self.cam_centers_base = T_base_from_event_all[:, :3, 3].astype(np.float32)
            self.src_offsets = []
            self.valid_indices = self._make_pose_view_ids(num_views, self.pose_move_threshold)
        else:
            self.cam_centers_base = None
            self.src_offsets = self._make_source_offsets(num_views, view_interval)
            if self.recurrent:
                self.valid_indices = np.arange(
                    self.recurrent_enrollment_range,
                    n_frames,
                    dtype=np.int64,
                )
            else:
                margin = max((abs(offset) for offset in self.src_offsets), default=0)
                self.valid_indices = np.arange(margin, n_frames - margin, dtype=np.int64)
        if len(self.valid_indices) == 0:
            mode = (
                f"pose threshold={pose_move_threshold:g} m"
                if self.pose_view_selection
                else f"view_interval={view_interval}"
            )
            raise RuntimeError(
                f"{self.seq_dir.name}: not enough frames ({n_frames}) for "
                f"num_views={num_views}, {mode}"
            )
        self._vox = None   # lazy HDF5 handles, opened per DataLoader worker
        self._dep = None
        self._msk = None
        self._tbl = None
        self._poses = None

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
        if self.pose_channels and self._poses is None:
            self._poses = h5py.File(self.poses_path, "r")["ee_T"]

    def __len__(self) -> int:
        return len(self.valid_indices)

    @staticmethod
    def _make_source_offsets(num_views: int, view_interval: int) -> list[int]:
        """Match MultiViewTableDataset's target, past, future ordering."""
        offsets: list[int] = []
        distance = 1
        while len(offsets) < num_views - 1:
            offsets.append(-distance * view_interval)
            if len(offsets) < num_views - 1:
                offsets.append(distance * view_interval)
            distance += 1
        return offsets

    def _find_pose_neighbours(
        self,
        idx: int,
        direction: int,
        per_direction: int,
        move_threshold: float,
    ) -> list[int] | None:
        neighbours: list[int] = []
        anchor = idx
        cursor = idx + direction
        while 0 <= cursor < len(self.cam_centers_base) and len(neighbours) < per_direction:
            moved = np.linalg.norm(self.cam_centers_base[cursor] - self.cam_centers_base[anchor])
            if moved >= move_threshold:
                neighbours.append(cursor)
                anchor = cursor
            cursor += direction
        if len(neighbours) != per_direction:
            return None
        return neighbours

    def _make_pose_view_ids(self, num_views: int, move_threshold: float) -> np.ndarray:
        per_direction = (num_views - 1) // 2
        valid: list[int] = []
        for idx in range(len(self.cam_centers_base)):
            before = self._find_pose_neighbours(idx, -1, per_direction, move_threshold)
            after = self._find_pose_neighbours(idx, 1, per_direction, move_threshold)
            if before is None or after is None:
                continue
            self.pose_view_ids[idx] = [idx] + before + after
            valid.append(idx)
        return np.array(valid, dtype=np.int64)

    def _load_input(self, idx: int) -> torch.Tensor:
        vox = self._vox[idx].astype(np.float32)
        vox_t = torch.from_numpy(vox)
        _, height, width = vox_t.shape
        tbl_t = torch.from_numpy(self._tbl[idx].astype(np.float32)).unsqueeze(0)
        if tbl_t.shape[-2:] != (height, width):
            tbl_t = F.interpolate(
                tbl_t.unsqueeze(0),
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        channels = [vox_t, tbl_t]
        if self.pose_channels:
            T_base_from_ee = self._poses[idx].astype(np.float32)
            T_base_from_event = T_base_from_ee @ self.T_ee_from_event
            pose_values = _pose_channels_from_base_event(T_base_from_event)
            pose_t = torch.from_numpy(pose_values).view(6, 1, 1).expand(-1, height, width)
            channels.append(pose_t)
        return torch.cat(channels, dim=0)

    def __getitem__(self, item: int):
        self._open()
        idx = int(self.valid_indices[item])
        if self.recurrent:
            view_ids = list(range(idx - self.recurrent_enrollment_range, idx + 1))
            view_inputs = [self._load_input(view_idx) for view_idx in view_ids]
            inp = torch.stack(view_inputs, dim=0)
        elif self.pose_view_selection:
            view_ids = self.pose_view_ids[idx]
            view_inputs = [self._load_input(view_idx) for view_idx in view_ids]
            inp = torch.cat(view_inputs, dim=0)
        else:
            view_ids = [idx] + [idx + offset for offset in self.src_offsets]
            # Concatenate complete per-view inputs along the channel dimension.
            # The target is first, followed by -interval, +interval, ... sources.
            view_inputs = [self._load_input(view_idx) for view_idx in view_ids]
            inp = torch.cat(view_inputs, dim=0)
        _, Hv, Wv = view_inputs[-1].shape

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

        # Optionally fill invalid depth pixels with the table-plane prior [m]
        if self.fill_invalid:
            target_tbl = view_inputs[-1][NUM_BINS:NUM_BINS + 1]
            tbl_m = target_tbl * (D_MAX - DEPTH_MIN) + DEPTH_MIN
            dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
            msk_t = torch.ones_like(msk_t)  # all pixels now have a valid target

        return inp, dep_t, msk_t


# ---------------------------------------------------------------------------
# Optional uncertainty head
# ---------------------------------------------------------------------------

class UncertaintyUNet(UNet):
    """UNet variant that predicts depth plus log variance in normalised depth units."""

    def __init__(self, in_ch: int, base: int):
        super().__init__(in_ch=in_ch, base=base)
        self.head = nn.Conv2d(base, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        skips = [feat]
        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i < len(self.encoders) - 1:
                skips.append(feat)
        feat = self.bottleneck(feat)
        for i, dec in enumerate(self.decoders):
            feat = dec(feat, skips[-(i + 1)])

        out = self.head(feat)
        pred = torch.sigmoid(out[:, :1])
        log_var = out[:, 1:2].clamp(min=-6.0, max=3.0)
        return pred, log_var


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell used at the U-Net bottleneck."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros_like(x)
        reset, update = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class RecurrentUNet(UNet):
    """Single-view recurrent U-Net with a ConvGRU bottleneck state."""

    def __init__(self, in_ch: int, base: int):
        super().__init__(in_ch=in_ch, base=base)
        bottleneck_channels = base * (2 ** len(self.encoders))
        self.gru = ConvGRUCell(bottleneck_channels)

    def _encode_one(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feat = self.stem(x)
        skips = [feat]
        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i < len(self.encoders) - 1:
                skips.append(feat)
        feat = self.bottleneck(feat)
        return feat, skips

    def _decode_one(self, feat: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for i, dec in enumerate(self.decoders):
            feat = dec(feat, skips[-(i + 1)])
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"RecurrentUNet expects (B,T,C,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        hidden = None
        final_skips = None
        for t in range(x.shape[1]):
            feat, skips = self._encode_one(x[:, t])
            hidden = self.gru(feat, hidden)
            final_skips = skips
        return self._decode_one(hidden, final_skips)


class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, averaged in self.model.state_dict().items():
            current = source[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(
                    current.to(dtype=averaged.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                averaged.copy_(current)


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
    model:     UNet,
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
    lambda_grad: float = 0.5,
    lambda_smooth: float = 0.01,
    lambda_mean: float = 0.1,
    lambda_normal: float = 0.1,
    ema_model: ModelEMA | None = None,
) -> tuple:
    """Return mean loss, L1, p95 absolute error, and worst-10% L1."""
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_l1 = total_p95 = total_worst10_l1 = 0.0
    phase = "train" if is_train else "val"
    n_batches = 0
    t_phase_start = time.perf_counter()
    t_last = t_phase_start
    batches_at_last_log = 0
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with ctx:
        for inp, dep_t, mask_t in loader:
            inp, dep_t, mask_t = inp.to(device), dep_t.to(device), mask_t.to(device)
            event_for_loss = inp[:, -1, :NUM_BINS] if inp.dim() == 5 else inp[:, :NUM_BINS]

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
                        pred, dep_norm, mask_t, event_for_loss, K=K,
                        lambda_grad=lambda_grad,
                        lambda_smooth=lambda_smooth,
                        lambda_mean=lambda_mean,
                        lambda_normal=lambda_normal,
                    )
                    loss = loss + depth_aux_weight * depth_aux_loss
            else:
                loss, _ = compute_loss(
                    pred, dep_norm, mask_t, event_for_loss, K=K,
                    lambda_grad=lambda_grad,
                    lambda_smooth=lambda_smooth,
                    lambda_mean=lambda_mean,
                    lambda_normal=lambda_normal,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema_model is not None:
                    ema_model.update(model)

            total_loss += loss.item()
            with torch.no_grad():
                total_l1 += _l1_metres(pred, dep_t, mask_t).item()
                pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                valid_errors = torch.abs(pred_m - dep_t)[mask_t > 0.5]
                if valid_errors.numel() > 0:
                    total_p95 += float(torch.quantile(valid_errors.float(), 0.95))
                total_worst10_l1 += _worst_percent_l1_metres(pred, dep_t, mask_t).item()

            n_batches += 1
            now = time.perf_counter()
            if now - t_last >= 20.0:
                interval_s = now - t_last
                interval_batches = n_batches - batches_at_last_log
                batches_per_s = interval_batches / max(interval_s, 1e-9)
                seconds_per_batch = interval_s / max(interval_batches, 1)
                if use_cuda:
                    vram_allocated = torch.cuda.memory_allocated(device) / 1024**2
                    vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
                    vram_peak = torch.cuda.max_memory_allocated(device) / 1024**2
                    vram_text = (
                        f"  VRAM {vram_allocated:.0f} MB allocated / "
                        f"{vram_reserved:.0f} MB reserved / {vram_peak:.0f} MB peak"
                    )
                else:
                    vram_text = ""
                print(
                    f"  [{phase}  {n_batches:4d}/{len(loader)} batches]  "
                    f"{batches_per_s:.3f} batch/s ({seconds_per_batch:.2f} s/batch)  "
                    f"loss {total_loss / n_batches:.4f}  "
                    f"L1 {total_l1 / n_batches:.4f} m  "
                    f"p95 {total_p95 / n_batches:.4f} m  "
                    f"worst10 {total_worst10_l1 / n_batches:.4f} m"
                    f"{vram_text}",
                    flush=True,
                )
                t_last = now
                batches_at_last_log = n_batches

            event_for_diag = event_for_loss.detach()
            if viz is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                tbl_ch = (
                    inp[:, -1, NUM_BINS:NUM_BINS + 1]
                    if inp.dim() == 5
                    else inp[:, NUM_BINS:NUM_BINS + 1]
                ).detach()
                viz.add_batch(event_for_diag, dep_t, mask_t, pred_m, table_depth=tbl_ch)
            if activity_diag is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                activity_diag.add_batch(event_for_diag, dep_t, mask_t, pred_m)
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

    elapsed_s = time.perf_counter() - t_phase_start
    batches_per_s = n_batches / max(elapsed_s, 1e-9)
    seconds_per_batch = elapsed_s / max(n_batches, 1)
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved_vram_mb = torch.cuda.memory_reserved(device) / 1024**2
        vram_summary = (
            f"  peak VRAM {peak_vram_mb:.0f} MB, "
            f"reserved {reserved_vram_mb:.0f} MB"
        )
    else:
        vram_summary = ""
    print(
        f"  [{phase} complete] {n_batches} batches in {elapsed_s:.1f} s  "
        f"({batches_per_s:.3f} batch/s, {seconds_per_batch:.2f} s/batch)"
        f"{vram_summary}",
        flush=True,
    )

    n = max(n_batches, 1)
    return total_loss / n, total_l1 / n, total_p95 / n, total_worst10_l1 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Early-fusion UNet with per-view table-plane prior channels"
    )
    parser.add_argument("--data_dir",      type=Path, default=DATA_ROOT,
                        help="Single sequence dir or parent of multiple sequences")
    parser.add_argument("--epochs",        type=int,  default=50)
    parser.add_argument("--batch_size",    type=int,  default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--workers",       type=int,  default=4)
    parser.add_argument("--base_channels", type=int,  default=32)
    parser.add_argument("--model_scale",   type=float, default=1.0,
                        help="Width multiplier for base_channels")
    parser.add_argument("--num_views",     type=int, default=1,
                        help="Number of target/source indices concatenated as U-Net input")
    parser.add_argument("--view_interval", type=int, default=5,
                        help="Frame interval between target and successive source indices")
    parser.add_argument("--pose_view_selection", action="store_true",
                        help="Select source views by travelled camera distance instead of fixed frame interval. "
                             "Requires odd --num_views.")
    parser.add_argument("--pose_move_threshold", type=float, default=0.01,
                        help="Minimum event-camera translation in metres between pose-selected views")
    parser.add_argument("--pose_channels", "--pose_bins", action="store_true",
                        help="Append six constant pose channels per view: event-camera "
                             "position xyz and optical-axis direction xyz in robot base frame")
    parser.add_argument("--recurrent", action="store_true",
                        help="Use a single-view recurrent U-Net with a ConvGRU bottleneck. "
                             "The dataset feeds previous frames as enrollment context.")
    parser.add_argument("--recurrent_enrollment_range", type=int, default=0,
                        help="Number of previous frames used to enroll/warm up the recurrent state "
                             "before predicting the target frame. 0 means only the target frame.")
    parser.add_argument("--out_dir",       type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "unet_table")
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
    parser.add_argument("--lambda_grad", type=float, default=0.5,
                        help="Weight for the multi-scale gradient loss.")
    parser.add_argument("--lambda_smooth", type=float, default=0.01,
                        help="Weight for the event-aware smoothness loss.")
    parser.add_argument("--lambda_mean", type=float, default=0.1,
                        help="Weight for the mean-depth consistency loss.")
    parser.add_argument("--lambda_normal", type=float, default=0.1,
                        help="Weight for the surface-normal loss.")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay used for validation/checkpoints; 0 disables EMA")
    parser.add_argument("--name",          type=str, default=None,
                        help="Run name used in checkpoint filenames. Prompted if not provided.")
    parser.add_argument("--tb_root",       type=Path, default=DEFAULT_TB_ROOT,
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/unet_table/<name>.")
    args = parser.parse_args()

    if args.model_scale <= 0:
        parser.error("--model_scale must be > 0")
    if args.num_views < 1:
        parser.error("--num_views must be at least 1")
    if args.view_interval < 1:
        parser.error("--view_interval must be at least 1")
    if args.pose_view_selection and args.num_views % 2 != 1:
        parser.error("--pose_view_selection requires odd --num_views")
    if args.pose_move_threshold <= 0:
        parser.error("--pose_move_threshold must be > 0")
    if args.recurrent and args.num_views != 1:
        parser.error("--recurrent only supports single-view input; use --num_views 1")
    if args.recurrent and args.pose_view_selection:
        parser.error("--recurrent cannot be combined with --pose_view_selection")
    if args.recurrent and args.predict_uncertainty:
        parser.error("--recurrent currently supports the depth head only, not --predict_uncertainty")
    for loss_flag in ("lambda_grad", "lambda_smooth", "lambda_mean", "lambda_normal"):
        if getattr(args, loss_flag) < 0:
            parser.error(f"--{loss_flag} must be >= 0")
    if args.recurrent_enrollment_range < 0:
        parser.error("--recurrent_enrollment_range must be >= 0")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        parser.error("--ema_decay must be in [0, 1)")

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

    train_root = args.data_dir / "train"
    eval_root = args.data_dir / "eval"
    explicit_train_seqs = (
        sorted(d for d in train_root.iterdir() if d.is_dir() and _is_sequence(d))
        if train_root.is_dir() else []
    )
    explicit_eval_seqs = (
        sorted(d for d in eval_root.iterdir() if d.is_dir() and _is_sequence(d))
        if eval_root.is_dir() else []
    )

    if explicit_train_seqs and explicit_eval_seqs:
        train_seqs = explicit_train_seqs
        val_seqs = explicit_eval_seqs
        seq_dirs = train_seqs + val_seqs
        single_object = False
    elif _is_sequence(args.data_dir):
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
    if explicit_train_seqs and explicit_eval_seqs:
        print(f"Using explicit train/eval split from {args.data_dir}")
        print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
        print(f"  Eval  ({len(val_seqs)}): {[d.name for d in val_seqs]}")
    elif single_object:
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
        num_views=args.num_views,
        view_interval=args.view_interval,
        pose_channels=args.pose_channels,
        pose_view_selection=args.pose_view_selection,
        pose_move_threshold=args.pose_move_threshold,
        recurrent=args.recurrent,
        recurrent_enrollment_range=args.recurrent_enrollment_range,
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
    pose_channel_count = 6 if args.pose_channels else 0
    per_view_channels = NUM_BINS + 1 + pose_channel_count
    in_ch = args.num_views * per_view_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = max(1, int(round(args.base_channels * args.model_scale)))
    if args.recurrent:
        model = RecurrentUNet(in_ch=in_ch, base=base_channels).to(device)
    elif args.predict_uncertainty:
        model = UncertaintyUNet(in_ch=in_ch, base=base_channels).to(device)
    else:
        model = UNet(in_ch=in_ch, base=base_channels).to(device)

    # K for loss: use the canonical resize + centred-crop transform.
    resize_H, resize_W = TRAIN_RESIZE_HW
    crop_H, crop_W = TRAIN_CROP_HW
    K_loss = K_native.copy()
    K_loss[0, :] *= resize_W / native_W
    K_loss[1, :] *= resize_H / native_H
    K_loss[0, 2] -= (resize_W - crop_W) / 2.0
    K_loss[1, 2] -= (resize_H - crop_H) / 2.0
    K_tensor = torch.from_numpy(K_loss).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.recurrent:
        model_name = "RecurrentUNet"
    else:
        model_name = "UNet+uncertainty" if args.predict_uncertainty else "UNet"
    print(
        f"{model_name}  in_ch={in_ch}  model_scale={args.model_scale:g}  "
        f"base={base_channels}  parameters: {n_params:,}"
    )
    print(
        f"  {args.num_views} views x ({NUM_BINS} voxel + 1 table-plane"
        f"{' + 6 pose' if args.pose_channels else ''}) "
        f"= {in_ch} channels; "
        + (
            f"pose_move_threshold={args.pose_move_threshold:g} m"
            if args.pose_view_selection
            else f"view_interval={args.view_interval}"
        )
    )
    if args.predict_uncertainty:
        print(f"  Uncertainty head enabled; primary NLL weight = {args.uncertainty_weight:g}")
        print(f"  Auxiliary depth-loss weight = {args.depth_aux_weight:g}")
    print(
        f"  Depth-loss weights: gradient={args.lambda_grad:g}, "
        f"smoothness={args.lambda_smooth:g}, mean={args.lambda_mean:g}, "
        f"normal={args.lambda_normal:g}"
    )
    if args.recurrent:
        print(
            f"  Recurrent mode: ConvGRU bottleneck, enrollment_range="
            f"{args.recurrent_enrollment_range} previous frame(s)"
        )
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(
            f"EMA: enabled (decay={args.ema_decay:g}); "
            "validation/checkpoints use EMA weights"
        )

    # ── Logging ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("unet_table", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")
    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val",   show_mask=not args.no_mask)
    activity_train = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/train")
    activity_val   = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/val")
    error_train = ErrorDistributionSpatialLogger(
        writer, tag="error/train", images_only=False
    )
    error_val = ErrorDistributionSpatialLogger(
        writer, tag="error/val", images_only=False
    )
    uncertainty_train = (
        UncertaintyErrorLogger(
            writer, tag="uncertainty/train", images_only=True
        )
        if args.predict_uncertainty else None
    )
    uncertainty_val = (
        UncertaintyErrorLogger(
            writer, tag="uncertainty/val", images_only=True
        )
        if args.predict_uncertainty else None
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_l1 = float("inf")
    best_val_p95 = float("inf")
    best_val_worst10 = float("inf")
    ckpt: dict = {}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, tr_p95, tr_worst10 = run_epoch(model, train_loader, optimizer, device,
                                    K_tensor, viz=viz_train,
                                    activity_diag=activity_train,
                                    error_diag=error_train,
                                    uncertainty_diag=uncertainty_train,
                                    uncertainty_weight=args.uncertainty_weight,
                                    depth_aux_weight=args.depth_aux_weight,
                                    lambda_grad=args.lambda_grad,
                                    lambda_smooth=args.lambda_smooth,
                                    lambda_mean=args.lambda_mean,
                                    lambda_normal=args.lambda_normal,
                                    ema_model=ema)
        validation_model = ema.model if ema is not None else model
        va_loss, va_l1, va_p95, va_worst10 = run_epoch(validation_model, val_loader, None, device,
                                    K_tensor, viz=viz_val,
                                    activity_diag=activity_val,
                                    error_diag=error_val,
                                    uncertainty_diag=uncertainty_val,
                                    uncertainty_weight=args.uncertainty_weight,
                                    depth_aux_weight=args.depth_aux_weight,
                                    lambda_grad=args.lambda_grad,
                                    lambda_smooth=args.lambda_smooth,
                                    lambda_mean=args.lambda_mean,
                                    lambda_normal=args.lambda_normal)
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
            f"p95: {tr_p95:.4f}/{va_p95:.4f} m  "
            f"worst10: {tr_worst10:.4f}/{va_worst10:.4f} m  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val",   va_loss, epoch)
        writer.add_scalar("l1/train",   tr_l1,   epoch)
        writer.add_scalar("l1/val",     va_l1,   epoch)
        writer.add_scalar("p95/train", tr_p95, epoch)
        writer.add_scalar("p95/val", va_p95, epoch)
        writer.add_scalar("l1_worst10/train", tr_worst10, epoch)
        writer.add_scalar("l1_worst10/val",   va_worst10, epoch)
        writer.add_scalar("lr",         scheduler.get_last_lr()[0], epoch)

        ckpt = {
            "epoch":   epoch,
            "model":   validation_model.state_dict(),
            "model_arch": model_name,
            "val_l1":  va_l1,
            "val_p95": va_p95,
            "val_l1_worst10": va_worst10,
            "base":    base_channels,
            "base_channels_arg": args.base_channels,
            "model_scale": args.model_scale,
            "in_ch":   in_ch,
            "num_views": args.num_views,
            "view_interval": args.view_interval,
            "pose_view_selection": args.pose_view_selection,
            "pose_move_threshold": args.pose_move_threshold,
            "pose_channels": args.pose_channels,
            "pose_channel_count": pose_channel_count,
            "recurrent": args.recurrent,
            "recurrent_enrollment_range": args.recurrent_enrollment_range,
            "early_fusion_views": True,
            "intrinsics_transform": "resize_center_crop",
            "table_z": table_z_ckpt,
            "predict_uncertainty": args.predict_uncertainty,
            "uncertainty_weight": args.uncertainty_weight,
            "depth_aux_weight": args.depth_aux_weight,
            "lambda_grad": args.lambda_grad,
            "lambda_smooth": args.lambda_smooth,
            "lambda_mean": args.lambda_mean,
            "lambda_normal": args.lambda_normal,
            "ema_decay": args.ema_decay,
        }
        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            torch.save(ckpt, args.out_dir / f"best_l1_{args.name}.pth")
            print(f"  → new best L1 checkpoint  (val L1 = {va_l1:.4f} m)")
        if va_p95 < best_val_p95:
            best_val_p95 = va_p95
            torch.save(ckpt, args.out_dir / f"best_p95_{args.name}.pth")
            print(f"  → new best p95 checkpoint  (val p95 = {va_p95:.4f} m)")
        if va_worst10 < best_val_worst10:
            best_val_worst10 = va_worst10
            torch.save(ckpt, args.out_dir / f"best_l1_worst10_{args.name}.pth")
            print(
                "  → new best worst10 checkpoint  "
                f"(val worst10 L1 = {va_worst10:.4f} m)"
            )

    torch.save(ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(
        f"\nDone. Best val L1: {best_val_l1:.4f} m, "
        f"p95: {best_val_p95:.4f} m, "
        f"worst10 L1: {best_val_worst10:.4f} m"
    )


if __name__ == "__main__":
    main()
