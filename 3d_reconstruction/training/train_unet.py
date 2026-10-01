#!/usr/bin/env python3
"""
train_unet.py - Early-fusion UNet with table-plane prior channels.

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

The entry point trains a standard early-fusion U-Net. The default is a single
view without pose channels.

Usage:
    python3 training/train_unet.py
    python3 training/train_unet.py --data_dir data/new/train
    python3 training/train_unet.py --table_z 0.02
    python3 training/train_unet.py --model_scale 2.0
    python3 training/train_unet.py --num_views 5 --view_interval 5
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEPTH_MIN, D_MAX, NUM_BINS, PREPROCESS_CROP_HW, PREPROCESS_RESIZE_HW,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    ModelEMA,
    build_pose_view_ids,
    fixed_source_offsets,
    is_precomputed_sequence,
    load_event_from_ee,
    load_event_intrinsics,
    pose_layout_counts,
    pose_channels_from_base_event,
    transform_intrinsics,
)
from depth_losses import (
    combined_depth_loss,
    l1_metres,
    worst_fraction_l1_metres,
)
_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = _SCRIPT_DIR.parent / "data" / "lego"
from tensorboard_helper import (
    DEFAULT_TB_ROOT,
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
    tensorboard_run_dir,
    VizLogger,
)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


# Camera calibration paths
_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat((inputs, skip), dim=1))


class _ResidualBlock(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(inputs + self.net(inputs))


class UNet(nn.Module):
    """Event-to-depth U-Net producing linear-normalized depth in ``[0, 1]``."""

    def __init__(
        self,
        in_ch: int = NUM_BINS,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.encoders = nn.ModuleList()
        channels = base
        for _ in range(num_encoders):
            self.encoders.append(_EncoderBlock(channels, channels * 2))
            channels *= 2
        self.bottleneck = nn.Sequential(
            *[_ResidualBlock(channels) for _ in range(num_residuals)]
        )
        self.decoders = nn.ModuleList()
        for index in range(num_encoders):
            skip_channels = channels // 2 if index < num_encoders - 1 else base
            self.decoders.append(
                _DecoderBlock(channels, skip_channels, channels // 2)
            )
            channels //= 2
        self.head = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        skips = [features]
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.encoders) - 1:
                skips.append(features)
        features = self.bottleneck(features)
        for index, decoder in enumerate(self.decoders):
            features = decoder(features, skips[-(index + 1)])
        return self.head(features)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

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
    resize_hw = PREPROCESS_RESIZE_HW
    crop_hw = PREPROCESS_CROP_HW
    K = transform_intrinsics(
        K_native.astype(np.float64), (native_H, native_W),
        resize_hw, crop_hw,
    )
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
        fill_invalid: bool = False,
        num_views: int = 1,
        view_interval: int = 5,
        pose_channels: bool = False,
        pose_view_selection: bool = False,
        pose_move_threshold: float = 0.01,
        allow_unbalanced_pose_views: bool = True,
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
        self.seq_dir         = seq_dir
        self.fill_invalid    = fill_invalid
        self.num_views       = int(num_views)
        self.view_interval   = int(view_interval)
        self.pose_channels   = bool(pose_channels)
        self.pose_view_selection = bool(pose_view_selection)
        self.pose_move_threshold = float(pose_move_threshold)
        self.allow_unbalanced_pose_views = bool(allow_unbalanced_pose_views)
        self.voxels_path     = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path      = seq_dir / "hdf5"   / "depth_in_event_frame.h5"
        self.poses_path      = seq_dir / "hdf5"   / "poses.h5"
        self.table_plane_path = seq_dir / "hdf5"  / "table_plane.h5"

        if not self.table_plane_path.exists():
            raise FileNotFoundError(
                f"Missing: {self.table_plane_path}\n"
                "Run data_precomputation/precompute_table_plane.py first."
            )

        import h5py
        with h5py.File(self.depth_path,       "r") as f: n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path, "r") as f:
            n_v = f["voxels"].shape[0]
            voxel_transform = f["voxels"].attrs.get("intrinsics_transform", "")
            if isinstance(voxel_transform, bytes):
                voxel_transform = voxel_transform.decode("utf-8", errors="replace")
        with h5py.File(self.table_plane_path, "r") as f:
            n_t = f["table_plane"].shape[0]
            transform = f.attrs.get("intrinsics_transform", "")
            if isinstance(transform, bytes):
                transform = transform.decode("utf-8", errors="replace")
            expected_transform = INTRINSICS_TRANSFORM
            corrected_table_transform = transform == expected_transform
        if not corrected_table_transform:
            raise RuntimeError(
                f"{self.table_plane_path} uses obsolete direct-scaling geometry. "
                "Regenerate it with: python3 "
                "data_precomputation/precompute_table_plane.py --overwrite "
                f"--data_dir {self.seq_dir}"
            )
        if voxel_transform != expected_transform:
            raise RuntimeError(
                f"{self.voxels_path} uses {voxel_transform!r}, requested "
                f"{expected_transform!r}. Regenerate the sequence."
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
                self.T_ee_from_event = np.linalg.inv(load_event_from_ee(_CAM_DATA)).astype(np.float32)
        else:
            ee_T_all = None
            n_p = n_d

        n_frames      = min(n_d, n_v, n_t, n_p)
        self.n_frames = n_frames

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
            self.src_offsets = fixed_source_offsets(num_views, view_interval)
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
        self._tbl = None
        self._poses = None

    def _open(self):
        import h5py
        if self._vox is None:
            self._vox = h5py.File(self.voxels_path,      "r")["voxels"]
        if self._dep is None:
            self._dep = h5py.File(self.depth_path,       "r")["depth"]
        if self._tbl is None:
            self._tbl = h5py.File(self.table_plane_path, "r")["table_plane"]
        if self.pose_channels and self._poses is None:
            self._poses = h5py.File(self.poses_path, "r")["ee_T"]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _make_pose_view_ids(self, num_views: int, move_threshold: float) -> np.ndarray:
        self.pose_view_ids, valid = build_pose_view_ids(
            self.cam_centers_base,
            num_views,
            move_threshold,
            self.allow_unbalanced_pose_views,
        )
        return valid

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
            pose_values = pose_channels_from_base_event(T_base_from_event)
            pose_t = torch.from_numpy(pose_values).view(6, 1, 1).expand(-1, height, width)
            channels.append(pose_t)
        return torch.cat(channels, dim=0)

    def __getitem__(self, item: int):
        self._open()
        idx = int(self.valid_indices[item])
        if self.pose_view_selection:
            view_ids = self.pose_view_ids[idx]
            target_input = self._load_input(idx)
            view_inputs = [
                target_input if view_idx == idx else
                self._load_input(view_idx) if view_idx >= 0 else
                torch.zeros_like(target_input)
                for view_idx in view_ids
            ]
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
        dep_t = torch.from_numpy(dep).unsqueeze(0)
        msk_t = torch.from_numpy(valid).unsqueeze(0)
        if dep_t.shape[-2] != Hv or dep_t.shape[-1] != Wv:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (Hv, Wv), mode="nearest").squeeze(0)

        # Optionally fill invalid depth pixels with the table-plane prior [m]
        if self.fill_invalid:
            target_tbl = view_inputs[0][NUM_BINS:NUM_BINS + 1]
            tbl_m = target_tbl * (D_MAX - DEPTH_MIN) + DEPTH_MIN
            dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
            msk_t = torch.ones_like(msk_t)  # all pixels now have a valid target

        return inp, dep_t, msk_t


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
    lambda_grad: float = 0.5,
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
            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            pred = model(inp)
            loss, _ = combined_depth_loss(
                pred, dep_norm, mask_t, K=K,
                lambda_grad=lambda_grad,
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
                total_l1 += l1_metres(pred, dep_t, mask_t).item()
                pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                valid_errors = torch.abs(pred_m - dep_t)[mask_t > 0.5]
                if valid_errors.numel() > 0:
                    total_p95 += float(torch.quantile(valid_errors.float(), 0.95))
                total_worst10_l1 += worst_fraction_l1_metres(pred, dep_t, mask_t).item()

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

            event_for_diag = inp[:, :NUM_BINS].detach()
            if viz is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                tbl_ch = inp[:, NUM_BINS:NUM_BINS + 1].detach()
                viz.add_batch(event_for_diag, dep_t, mask_t, pred_m, table_depth=tbl_ch)
            if activity_diag is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                activity_diag.add_batch(event_for_diag, dep_t, mask_t, pred_m)
            if error_diag is not None:
                pred_m = (pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                error_diag.add_batch(pred_m, dep_t, mask_t)

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
    parser.add_argument("--weight_decay",  type=float, default=5e-4,
                        help="AdamW weight decay")
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
    pose_layout_group = parser.add_mutually_exclusive_group()
    pose_layout_group.add_argument(
        "--allow_fewer_pose_views",
        "--allow-fewer-pose-views",
        "--allow_unbalanced_pose_views",
        "--allow-unbalanced-pose-views",
        dest="allow_unbalanced_pose_views",
        action="store_true",
        help=(
            "Allow pose-selected targets near sequence boundaries to use fewer "
            "sources; missing past/future input blocks are zero-padded (default)"
        ),
    )
    pose_layout_group.add_argument(
        "--strict_balanced_pose_views",
        "--strict-balanced-pose-views",
        dest="allow_unbalanced_pose_views",
        action="store_false",
        help="Require equal numbers of pose-selected sources before and after every target",
    )
    parser.set_defaults(allow_unbalanced_pose_views=True)
    parser.add_argument("--pose_channels", "--pose_bins", action="store_true",
                        help="Append six constant pose channels per view: event-camera "
                             "position xyz and optical-axis direction xyz in robot base frame")
    parser.add_argument("--out_dir",       type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "unet_table")
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--table_z",       type=float, default=None,
                        help="Optional: override for --debug / informational use. "
                             "The actual table_z is read from hdf5/table_plane.h5.")
    parser.add_argument("--fill_invalid",  action="store_true",
                        help="Fill pixels with no depth measurement using the table-plane prior")
    parser.add_argument("--lambda_grad", type=float, default=0.5,
                        help="Weight for the multi-scale gradient loss.")
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
    if args.weight_decay < 0:
        parser.error("--weight_decay must be >= 0")
    if args.num_views < 1:
        parser.error("--num_views must be at least 1")
    if args.view_interval < 1:
        parser.error("--view_interval must be at least 1")
    if args.pose_view_selection and args.num_views % 2 != 1:
        parser.error("--pose_view_selection requires odd --num_views")
    if args.pose_move_threshold <= 0:
        parser.error("--pose_move_threshold must be > 0")
    for loss_flag in ("lambda_grad", "lambda_normal"):
        if getattr(args, loss_flag) < 0:
            parser.error(f"--{loss_flag} must be >= 0")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        parser.error("--ema_decay must be in [0, 1)")

    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import h5py
    K_native, (native_H, native_W) = load_event_intrinsics(_CAM_DATA)

    # ── Discover sequences ────────────────────────────────────────────────
    train_root = args.data_dir / "train"
    eval_root = args.data_dir / "eval"
    explicit_train_seqs = (
        sorted(d for d in train_root.iterdir() if d.is_dir() and is_precomputed_sequence(d))
        if train_root.is_dir() else []
    )
    explicit_eval_seqs = (
        sorted(d for d in eval_root.iterdir() if d.is_dir() and is_precomputed_sequence(d))
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
        fill_invalid=args.fill_invalid,
        num_views=args.num_views,
        view_interval=args.view_interval,
        pose_channels=args.pose_channels,
        pose_view_selection=args.pose_view_selection,
        pose_move_threshold=args.pose_move_threshold,
        allow_unbalanced_pose_views=args.allow_unbalanced_pose_views,
    )
    train_sets = [TablePriorDataset(d, **ds_kw) for d in train_seqs]
    val_sets = [TablePriorDataset(d, **ds_kw) for d in val_seqs]
    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    train_total_frames = sum(ds.n_frames for ds in train_sets)
    val_total_frames = sum(ds.n_frames for ds in val_sets)
    print("  Train target-frame usage by sequence:")
    for sequence, dataset in zip(train_seqs, train_sets):
        layout = pose_layout_counts(dataset.pose_view_ids)
        layout_text = (
            f"; layouts balanced={layout['balanced']}, "
            f"asymmetric={layout['asymmetric']}, one-sided={layout['one_sided']}, "
            f"reference-only={layout['reference_only']}"
            if dataset.pose_view_selection else ""
        )
        print(
            f"    {sequence.name}: {len(dataset)}/{dataset.n_frames} "
            f"({100.0 * len(dataset) / dataset.n_frames:.1f}%){layout_text}"
        )
    print("  Val target-frame usage by sequence:")
    for sequence, dataset in zip(val_seqs, val_sets):
        layout = pose_layout_counts(dataset.pose_view_ids)
        layout_text = (
            f"; layouts balanced={layout['balanced']}, "
            f"asymmetric={layout['asymmetric']}, one-sided={layout['one_sided']}, "
            f"reference-only={layout['reference_only']}"
            if dataset.pose_view_selection else ""
        )
        print(
            f"    {sequence.name}: {len(dataset)}/{dataset.n_frames} "
            f"({100.0 * len(dataset) / dataset.n_frames:.1f}%){layout_text}"
        )
    print(
        f"  Train target frames used: {len(train_ds)}/{train_total_frames} "
        f"({100.0 * len(train_ds) / train_total_frames:.1f}%)"
    )
    print(
        f"  Val target frames used: {len(val_ds)}/{val_total_frames} "
        f"({100.0 * len(val_ds) / val_total_frames:.1f}%)"
    )
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
    model = UNet(in_ch=in_ch, base=base_channels).to(device)

    # K for loss: use the canonical native center crop followed by resize.
    resize_hw = PREPROCESS_RESIZE_HW
    crop_hw = PREPROCESS_CROP_HW
    K_loss = transform_intrinsics(
        K_native, (native_H, native_W), resize_hw, crop_hw,
    )
    K_tensor = torch.from_numpy(K_loss).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "UNet"
    print(
        f"{model_name}  in_ch={in_ch}  model_scale={args.model_scale:g}  "
        f"base={base_channels}  parameters: {n_params:,}"
    )
    print(
        f"  {args.num_views} views x ({NUM_BINS} voxel + 1 table-plane"
        f"{' + 6 pose' if args.pose_channels else ''}) "
        f"= {in_ch} channels; "
        + (
            f"pose_move_threshold={args.pose_move_threshold:g} m; "
            f"layout={'fewer boundary views allowed' if args.allow_unbalanced_pose_views else 'strictly balanced'}"
            if args.pose_view_selection
            else f"view_interval={args.view_interval}"
        )
    )
    print(
        f"  Depth-loss weights: gradient={args.lambda_grad:g}, "
        f"normal={args.lambda_normal:g}"
    )
    print(
        f"  Optimizer: AdamW, lr={args.lr:g}, "
        f"weight_decay={args.weight_decay:g}"
    )
    print(f"Device: {device}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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
    viz_train = VizLogger(writer, n_samples=4, tag="viz/train")
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val")
    activity_train = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/train")
    activity_val   = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/val")
    error_train = ErrorDistributionSpatialLogger(
        writer, tag="error/train", images_only=False
    )
    error_val = ErrorDistributionSpatialLogger(
        writer, tag="error/val", images_only=False
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
                                    lambda_grad=args.lambda_grad,
                                    lambda_normal=args.lambda_normal,
                                    ema_model=ema)
        validation_model = ema.model if ema is not None else model
        va_loss, va_l1, va_p95, va_worst10 = run_epoch(validation_model, val_loader, None, device,
                                    K_tensor, viz=viz_val,
                                    activity_diag=activity_val,
                                    error_diag=error_val,
                                    lambda_grad=args.lambda_grad,
                                    lambda_normal=args.lambda_normal)
        scheduler.step()

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)
        activity_train.flush(step=epoch)
        activity_val.flush(step=epoch)
        error_train.flush(step=epoch)
        error_val.flush(step=epoch)

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
            "allow_unbalanced_pose_views": args.allow_unbalanced_pose_views,
            "allow_fewer_pose_views": args.allow_unbalanced_pose_views,
            "pose_channels": args.pose_channels,
            "pose_channel_count": pose_channel_count,
            "early_fusion_views": True,
            "intrinsics_transform": INTRINSICS_TRANSFORM,
            "table_z": table_z_ckpt,
            "lambda_grad": args.lambda_grad,
            "lambda_normal": args.lambda_normal,
            "optimizer": "adamw",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
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
