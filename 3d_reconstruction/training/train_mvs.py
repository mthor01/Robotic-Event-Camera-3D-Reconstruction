#!/usr/bin/env python3
"""
train_mvs.py - Three-stage multi-view event-depth training with a table-plane prior.

This is the multi-view counterpart to train_unet.py. Each sample uses a
target event voxel frame plus neighbouring source frames:

    [x_i | table_plane_channel_i] for i in target + source frames

A deep FPN extracts shared H/8, H/4, and H/2 features. Source features are
warped into the target frustum using poses from hdf5/poses.h5 and event-camera
calibration from camera_data/. Variance cost volumes regress coarse depth,
middle local candidates, and final fine candidates:

    d_i(u, v) = d_hat(u, v) + sigma(u, v) * epsilon_i

The coarse stage samples inverse depth globally. Middle and fine stages sample
local linear-depth windows around the preceding prediction; the fine window
can optionally be predicted from target features. Only the final prediction is
supervised and returned.

Usage:
    python3 training/train_mvs.py --data_dir data/new/train
    python3 training/train_mvs.py --data_dir data/new/train --num_views 5
    python3 training/train_mvs.py --coarse_depths 32 --fine_depths 5 --view_interval 5
    python3 training/train_mvs.py --pose_view_selection --pose_move_threshold 0.01 --num_views 5
    python3 training/train_mvs.py --base_channels 64 --feature_channels 256 --cost_channels 32
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import DEPTH_MIN, D_MAX, NUM_BINS
from depth_losses import (
    charbonnier_loss,
    gradient_loss,
    l1_metres,
    normal_loss,
    worst_fraction_l1_metres,
)
from tensorboard_helper import (
    DEFAULT_TB_ROOT,
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
    tensorboard_run_dir,
    UncertaintyErrorLogger,
    VizLogger,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    ModelEMA,
    build_pose_view_ids,
    camera_centers_world,
    find_precomputed_sequences,
    fixed_source_offsets,
    load_event_calibration,
    pose_layout_counts,
    pose_channels_from_base_event,
    set_3d_axes_equal,
    transform_intrinsics,
)

_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"
DATA_ROOT = _SCRIPT_DIR.parent / "data" / "lego"


@dataclass(frozen=True)
class MultiViewAugConfig:
    """Training-time multi-view augmentation toggles and conservative defaults."""
    enabled: bool = False
    source_view_dropout: bool = True
    pose_noise: bool = True
    event_noise_per_view: bool = True
    cross_view_event_dropout: bool = True
    occlusion_view_masking: bool = True
    source_view_dropout_prob: float = 0.10
    pose_translation_std: float = 0.001
    pose_rotation_std_deg: float = 0.15
    event_noise_std: float = 0.01
    cross_view_event_dropout_prob: float = 0.02
    occlusion_prob: float = 0.20
    occlusion_max_rects: int = 1
    occlusion_frac_range: tuple[float, float] = (0.05, 0.15)


# ---------------------------------------------------------------------------
# Calibration / geometry
# ---------------------------------------------------------------------------

def _inverse_depth_candidates(coarse_depths: int, depth_min: float, depth_max: float) -> np.ndarray:
    if coarse_depths < 2:
        raise ValueError(f"--coarse_depths must be >= 2, got {coarse_depths}")
    inv = np.linspace(1.0 / depth_min, 1.0 / depth_max, coarse_depths, dtype=np.float32)
    return (1.0 / inv).astype(np.float32)


def _linear_depth_candidates(coarse_depths: int, depth_min: float, depth_max: float) -> np.ndarray:
    """Return uniformly spaced metric-depth hypotheses."""
    if coarse_depths < 2:
        raise ValueError(f"--coarse_depths must be >= 2, got {coarse_depths}")
    return np.linspace(depth_min, depth_max, coarse_depths, dtype=np.float32)


def homo_warp_features(
    src_feat: torch.Tensor,
    src_T_cam_from_world: torch.Tensor,
    ref_T_cam_from_world: torch.Tensor,
    K: torch.Tensor,
    depth_values: torch.Tensor,
    return_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Warp source features into the target camera frustum for every depth.

    Args:
        src_feat:             (B, C, H, W)
        src_T_cam_from_world: (B, 4, 4)
        ref_T_cam_from_world: (B, 4, 4)
        K:                    (B, 3, 3), scaled to feature resolution
        depth_values:         (B, D), (D,), or (B, D, H, W), metric z-depths

    Returns:
        warped: (B, C, D, H, W) source features sampled in the target frustum.
        valid:  (B, 1, D, H, W) if return_valid_mask=True, with 1 for
                projections in front of the camera and inside the source image.
    """
    B, C, H, W = src_feat.shape
    device = src_feat.device
    dtype = src_feat.dtype

    K = K.to(device=device, dtype=dtype)
    depth_values = depth_values.to(device=device, dtype=dtype)
    if depth_values.dim() == 1:
        depth_values = depth_values[None].expand(B, -1)
    if depth_values.dim() == 2:
        D = depth_values.shape[1]
        depth_hw = depth_values[:, :, None].expand(-1, -1, H * W)
    elif depth_values.dim() == 4:
        D = depth_values.shape[1]
        if depth_values.shape[-2:] != (H, W):
            depth_values = F.interpolate(
                depth_values, size=(H, W), mode="bilinear", align_corners=False
            )
        depth_hw = depth_values.flatten(2)
    else:
        raise ValueError(f"Unsupported depth_values shape: {tuple(depth_values.shape)}")

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack(
        [x.reshape(-1), y.reshape(-1), torch.ones(H * W, device=device, dtype=dtype)],
        dim=0,
    )
    pix = pix.unsqueeze(0).expand(B, -1, -1)  # (B, 3, HW)

    K_inv = torch.linalg.inv(K.float()).to(dtype)
    rays_ref = K_inv @ pix
    pts_ref = rays_ref.unsqueeze(2) * depth_hw[:, None]  # (B, 3, D, HW)
    ones = torch.ones((B, 1, D, H * W), device=device, dtype=dtype)
    pts_ref_h = torch.cat([pts_ref, ones], dim=1).flatten(2)  # (B, 4, D*HW)

    T_src_from_ref = (
        src_T_cam_from_world.float() @ torch.linalg.inv(ref_T_cam_from_world.float())
    ).to(dtype)
    pts_src = (T_src_from_ref @ pts_ref_h).view(B, 4, D, H * W)[:, :3]

    z = pts_src[:, 2:3]
    in_front = z > 1e-6
    pts_src_norm = pts_src / z.clamp_min(1e-6)
    pix_src = (K @ pts_src_norm.flatten(2)).view(B, 3, D, H * W)

    x_grid = 2.0 * (pix_src[:, 0] / max(W - 1, 1)) - 1.0
    y_grid = 2.0 * (pix_src[:, 1] / max(H - 1, 1)) - 1.0
    finite_grid = torch.isfinite(x_grid) & torch.isfinite(y_grid)
    sample_valid = in_front[:, 0] & finite_grid
    mask_valid = (
        sample_valid
        & (x_grid >= -1.0)
        & (x_grid <= 1.0)
        & (y_grid >= -1.0)
        & (y_grid <= 1.0)
    )
    x_grid = torch.where(sample_valid, x_grid, torch.full_like(x_grid, 2.0))
    y_grid = torch.where(sample_valid, y_grid, torch.full_like(y_grid, 2.0))
    x_grid = torch.nan_to_num(x_grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
    y_grid = torch.nan_to_num(y_grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)

    grid = torch.stack([x_grid, y_grid], dim=-1).view(B, D * H, W, 2)
    warped = F.grid_sample(
        src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    warped = warped.view(B, C, D, H, W)
    if return_valid_mask:
        valid_mask = mask_valid.to(dtype=dtype).view(B, 1, D, H, W)
        return warped, valid_mask
    return warped


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiViewTableDataset(Dataset):
    """
    Target + source event voxels with table-plane prior, GT target depth, poses.

    Returns:
        imgs         : (V, NUM_BINS + 1, H, W)
        cam_mats     : (V, 4, 4), T_event_from_base/world
        K            : (3, 3), intrinsics scaled to H,W
        depth_values : (D,)
        dep_t        : (1, H, W), metres
        mask_t       : (1, H, W)
    """

    def __init__(
        self,
        seq_dir: Path,
        calib: dict,
        num_views: int = 5,
        view_interval: int = 5,
        pose_view_selection: bool = False,
        pose_move_threshold: float = 0.01,
        allow_unbalanced_pose_views: bool = True,
        coarse_depths: int = 32,
        linear_depth_candidates: bool = False,
        fill_invalid: bool = False,
        pose_channels: bool = False,
        split_indices: np.ndarray | None = None,
        aug: MultiViewAugConfig | None = None,
    ):
        super().__init__()
        if num_views < 1:
            raise ValueError("--num_views must be at least 1")
        if pose_view_selection and num_views % 2 != 1:
            raise ValueError(
                "--pose_view_selection requires odd --num_views so sources are balanced "
                "before and after the target"
            )
        if pose_move_threshold <= 0:
            raise ValueError(f"--pose_move_threshold must be > 0, got {pose_move_threshold}")
        self.seq_dir = Path(seq_dir)
        self.num_views = num_views
        self.view_interval = view_interval
        self.pose_view_selection = pose_view_selection
        self.pose_move_threshold = float(pose_move_threshold)
        self.allow_unbalanced_pose_views = bool(allow_unbalanced_pose_views)
        self.fill_invalid = fill_invalid
        self.pose_channels = bool(pose_channels)
        self.aug = aug or MultiViewAugConfig(enabled=False)
        expected_transform = INTRINSICS_TRANSFORM

        self.voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path = self.seq_dir / "hdf5" / "depth_in_event_frame.h5"
        self.poses_path = self.seq_dir / "hdf5" / "poses.h5"
        self.table_plane_path = self.seq_dir / "hdf5" / "table_plane.h5"

        for p in (self.voxels_path, self.depth_path, self.poses_path, self.table_plane_path):
            if not p.exists():
                raise FileNotFoundError(p)

        import h5py
        with h5py.File(self.depth_path, "r") as f:
            n_d = f["depth"].shape[0]
        with h5py.File(self.voxels_path, "r") as f:
            n_v = f["voxels"].shape[0]
            _, vox_h, vox_w = f["voxels"].shape[1:]
            vox_attrs = dict(f["voxels"].attrs)
        with h5py.File(self.table_plane_path, "r") as f:
            n_t = f["table_plane"].shape[0]
            transform = f.attrs.get("intrinsics_transform", "")
            if isinstance(transform, bytes):
                transform = transform.decode("utf-8", errors="replace")
            corrected_table_transform = transform == expected_transform
        with h5py.File(self.poses_path, "r") as f:
            ee_T = f["ee_T"][:].astype(np.float32)

        self.n_frames = min(n_d, n_v, n_t, len(ee_T))
        if not corrected_table_transform:
            raise RuntimeError(
                f"{self.table_plane_path} uses obsolete direct-scaling geometry. "
                "Regenerate it with: python3 "
                "data_precomputation/precompute_table_plane.py --overwrite "
                f"--data_dir {self.seq_dir}"
            )
        native_h, native_w = calib["native_hw"]
        resize_h = int(vox_attrs.get("resize_h", vox_h))
        resize_w = int(vox_attrs.get("resize_w", vox_w))
        crop_h = int(vox_attrs.get("crop_h", vox_h))
        crop_w = int(vox_attrs.get("crop_w", vox_w))
        voxel_transform = vox_attrs.get("intrinsics_transform", "")
        if isinstance(voxel_transform, bytes):
            voxel_transform = voxel_transform.decode("utf-8", errors="replace")
        if voxel_transform != expected_transform:
            raise RuntimeError(
                f"{self.voxels_path} uses {voxel_transform!r}, requested "
                f"{expected_transform!r}. Regenerate the sequence."
            )
        self.K = transform_intrinsics(
            calib["K_native"],
            (native_h, native_w),
            (resize_h, resize_w),
            (crop_h, crop_w),
        ).astype(np.float32)
        self.linear_depth_candidates = bool(linear_depth_candidates)
        if self.linear_depth_candidates:
            self.depth_values = _linear_depth_candidates(
                coarse_depths, DEPTH_MIN, D_MAX
            )
        else:
            self.depth_values = _inverse_depth_candidates(
                coarse_depths, DEPTH_MIN, D_MAX
            )

        ee_T = ee_T[:self.n_frames]
        T_ee_inv = np.linalg.inv(ee_T)
        self.T_cam_from_world = np.einsum(
            "ij,njk->nik", calib["T_event_from_ee"], T_ee_inv
        ).astype(np.float32)
        self.cam_centers_world = camera_centers_world(self.T_cam_from_world)
        if self.pose_channels:
            T_base_from_event = np.linalg.inv(self.T_cam_from_world).astype(np.float32)
            self.pose_values = np.stack(
                [pose_channels_from_base_event(T) for T in T_base_from_event],
                axis=0,
            ).astype(np.float32)
        else:
            self.pose_values = None

        self.pose_view_ids: dict[int, list[int]] = {}
        if num_views == 1:
            self.src_offsets = []
            valid = np.arange(self.n_frames, dtype=np.int64)
        elif self.pose_view_selection:
            valid = self._make_pose_view_ids(num_views, self.pose_move_threshold)
        else:
            self.src_offsets = fixed_source_offsets(num_views, view_interval)
            margin = max(abs(o) for o in self.src_offsets)
            valid = np.arange(margin, self.n_frames - margin, dtype=np.int64)
        if len(valid) == 0:
            mode = (
                f"pose threshold={self.pose_move_threshold:g} m"
                if self.pose_view_selection
                else f"view_interval={view_interval}"
            )
            raise RuntimeError(
                f"{self.seq_dir.name}: not enough frames ({self.n_frames}) for "
                f"num_views={num_views}, {mode}"
            )
        if split_indices is not None:
            valid_set = set(valid.tolist())
            valid = np.array([i for i in split_indices if int(i) in valid_set], dtype=np.int64)
            if len(valid) == 0:
                mode = (
                    f"pose threshold={self.pose_move_threshold:g} m"
                    if self.pose_view_selection
                    else f"view_interval={view_interval}"
                )
                raise RuntimeError(
                    f"{self.seq_dir.name}: split has no valid samples for "
                    f"num_views={num_views}, {mode}"
                )
        self.valid_indices = valid

        self._vox = None
        self._dep = None
        self._tbl = None

    def _make_pose_view_ids(self, num_views: int, move_threshold: float) -> np.ndarray:
        self.pose_view_ids, valid = build_pose_view_ids(
            self.cam_centers_world,
            num_views,
            move_threshold,
            self.allow_unbalanced_pose_views,
        )
        return valid

    def _open(self) -> None:
        import h5py
        if self._vox is None:
            self._vox = h5py.File(self.voxels_path, "r")["voxels"]
        if self._dep is None:
            self._dep = h5py.File(self.depth_path, "r")["depth"]
        if self._tbl is None:
            self._tbl = h5py.File(self.table_plane_path, "r")["table_plane"]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_input(self, idx: int) -> torch.Tensor:
        vox = self._vox[idx]
        if vox.dtype == np.float16:
            vox = vox.astype(np.float32)
        else:
            vox = vox.astype(np.float32)
        vox_t = torch.from_numpy(vox)
        _, h, w = vox_t.shape

        tbl = self._tbl[idx].astype(np.float32)
        tbl_t = torch.from_numpy(tbl).unsqueeze(0)
        if tbl_t.shape[-2] != h or tbl_t.shape[-1] != w:
            tbl_t = F.interpolate(
                tbl_t.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
        channels = [vox_t, tbl_t]
        if self.pose_channels:
            pose_values = self.pose_values[idx]
            pose_t = torch.from_numpy(pose_values).view(6, 1, 1).expand(-1, h, w)
            channels.append(pose_t)
        return torch.cat(channels, dim=0)

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(axis_angle).clamp_min(1e-12)
        axis = axis_angle / theta
        x, y, z = axis
        zero = torch.zeros((), dtype=axis_angle.dtype, device=axis_angle.device)
        K = torch.stack([
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ])
        eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

    def _apply_source_view_dropout(
        self,
        imgs: torch.Tensor,
        view_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if imgs.shape[0] <= 1:
            return imgs
        p = self.aug.source_view_dropout_prob
        source_valid = view_valid_mask[1:].bool()
        drop = (torch.rand(imgs.shape[0] - 1) < p) & source_valid
        valid_indices = torch.nonzero(source_valid, as_tuple=False).flatten()
        if len(valid_indices) > 0 and drop[source_valid].all():
            keep_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            drop[keep_idx] = False
        # Drop only source event measurements.  The table-plane channel is a
        # deterministic geometric input and should not disappear with events.
        source_events = imgs[1:, :NUM_BINS]
        source_events[drop] = 0.0
        return imgs

    def _apply_pose_noise(self, cam_mats: torch.Tensor) -> torch.Tensor:
        trans_std = self.aug.pose_translation_std
        rot_std = np.deg2rad(self.aug.pose_rotation_std_deg)
        for v in range(1, cam_mats.shape[0]):
            delta = torch.eye(4, dtype=cam_mats.dtype)
            axis = torch.randn(3, dtype=cam_mats.dtype)
            axis = axis / torch.linalg.norm(axis).clamp_min(1e-12)
            angle = torch.randn((), dtype=cam_mats.dtype) * rot_std
            delta[:3, :3] = self._axis_angle_to_matrix(
                axis * angle
            )
            delta[:3, 3] = torch.randn(3, dtype=cam_mats.dtype) * trans_std
            cam_mats[v] = delta @ cam_mats[v]
        return cam_mats

    def _apply_event_noise_per_view(self, imgs: torch.Tensor) -> torch.Tensor:
        events = imgs[:, :NUM_BINS]
        # Perturb recorded activity without inventing events in empty voxels.
        active = events != 0
        events.add_(torch.randn_like(events) * self.aug.event_noise_std * active)
        return imgs

    def _apply_cross_view_event_dropout(self, imgs: torch.Tensor) -> torch.Tensor:
        keep = (
            torch.rand((1, NUM_BINS, imgs.shape[-2], imgs.shape[-1]), dtype=imgs.dtype)
            >= self.aug.cross_view_event_dropout_prob
        ).to(dtype=imgs.dtype)
        imgs[:, :NUM_BINS] = imgs[:, :NUM_BINS] * keep
        return imgs

    def _apply_occlusion_view_masking(self, imgs: torch.Tensor) -> torch.Tensor:
        h, w = imgs.shape[-2:]
        lo, hi = self.aug.occlusion_frac_range
        for v in range(imgs.shape[0]):
            if torch.rand(()) >= self.aug.occlusion_prob:
                continue
            n_rects = int(torch.randint(1, self.aug.occlusion_max_rects + 1, (1,)).item())
            for _ in range(n_rects):
                rect_h = max(1, int(round(float(torch.empty(()).uniform_(lo, hi)) * h)))
                rect_w = max(1, int(round(float(torch.empty(()).uniform_(lo, hi)) * w)))
                y0 = int(torch.randint(0, max(h - rect_h + 1, 1), (1,)).item())
                x0 = int(torch.randint(0, max(w - rect_w + 1, 1), (1,)).item())
                # Occlusion represents missing event measurements, not missing
                # deterministic table geometry.
                imgs[v, :NUM_BINS, y0:y0 + rect_h, x0:x0 + rect_w] = 0.0
        return imgs

    def _apply_augmentations(
        self,
        imgs: torch.Tensor,
        cam_mats: torch.Tensor,
        view_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.aug.enabled:
            return imgs, cam_mats
        imgs = imgs.clone()
        cam_mats = cam_mats.clone()
        if self.aug.source_view_dropout:
            imgs = self._apply_source_view_dropout(imgs, view_valid_mask)
        if self.aug.pose_noise:
            cam_mats = self._apply_pose_noise(cam_mats)
        if self.aug.event_noise_per_view:
            imgs = self._apply_event_noise_per_view(imgs)
        if self.aug.cross_view_event_dropout:
            imgs = self._apply_cross_view_event_dropout(imgs)
        if self.aug.occlusion_view_masking:
            imgs = self._apply_occlusion_view_masking(imgs)
        return imgs, cam_mats

    def __getitem__(self, item: int):
        self._open()
        idx = int(self.valid_indices[item])
        if self.num_views == 1:
            view_ids = [idx]
        elif self.pose_view_selection:
            view_ids = self.pose_view_ids[idx]
        else:
            view_ids = [idx] + [idx + o for o in self.src_offsets]

        target_input = self._load_input(idx)
        imgs = torch.stack(
            [
                target_input if view_idx == idx else
                self._load_input(view_idx) if view_idx >= 0 else
                torch.zeros_like(target_input)
                for view_idx in view_ids
            ],
            dim=0,
        )
        view_valid_mask = torch.tensor(
            [view_idx >= 0 for view_idx in view_ids], dtype=torch.bool
        )
        _, _, h, w = imgs.shape

        dep = self._dep[idx].astype(np.float32)
        dep = np.minimum(dep, D_MAX)
        valid = (dep > 0).astype(np.float32)

        dep_t = torch.from_numpy(dep).unsqueeze(0)
        msk_t = torch.from_numpy(valid).unsqueeze(0)
        if dep_t.shape[-2] != h or dep_t.shape[-1] != w:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (h, w), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (h, w), mode="nearest").squeeze(0)

        if self.fill_invalid:
            tbl_m = imgs[0, NUM_BINS:NUM_BINS + 1] * (D_MAX - DEPTH_MIN) + DEPTH_MIN
            dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
            msk_t = torch.ones_like(msk_t)

        cam_mats = torch.from_numpy(
            np.stack([
                self.T_cam_from_world[view_idx]
                if view_idx >= 0 else self.T_cam_from_world[idx]
                for view_idx in view_ids
            ])
        )
        imgs, cam_mats = self._apply_augmentations(
            imgs, cam_mats, view_valid_mask
        )
        return {
            "imgs": imgs,
            "cam_mats": cam_mats,
            "K": torch.from_numpy(self.K),
            "depth_values": torch.from_numpy(self.depth_values),
            "dep_t": dep_t,
            "mask_t": msk_t,
            "ref_idx": torch.tensor(idx, dtype=torch.long),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
            "view_valid_mask": view_valid_mask,
        }


class PerSequenceFractionSampler(Sampler[int]):
    """Select a new deterministic random fraction from every sequence each epoch."""

    def __init__(self, sequence_lengths: list[int], fraction: float, seed: int):
        self.sequence_lengths = [int(n) for n in sequence_lengths]
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.epoch = 0
        self.offsets = np.cumsum([0, *self.sequence_lengths[:-1]]).tolist()
        self.sample_counts = [
            min(n, max(1, int(round(n * self.fraction))))
            for n in self.sequence_lengths
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(self.sample_counts)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        selected: list[torch.Tensor] = []
        for offset, length, count in zip(
            self.offsets, self.sequence_lengths, self.sample_counts
        ):
            local = torch.randperm(length, generator=generator)[:count]
            selected.append(local + offset)
        indices = torch.cat(selected)
        order = torch.randperm(len(indices), generator=generator)
        return iter(indices[order].tolist())


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Residual2dBlock(nn.Module):
    """Basic residual block used by the deeper FPN encoder."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.projection = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


def _residual2d_stage(
    in_ch: int,
    out_ch: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    if blocks < 1:
        raise ValueError("A residual stage must contain at least one block")
    layers: list[nn.Module] = [Residual2dBlock(in_ch, out_ch, stride=stride)]
    layers.extend(Residual2dBlock(out_ch, out_ch) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual feature paths."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class FeaturePyramid(nn.Module):
    """Build the fixed deep, three-stage feature pyramid."""

    def __init__(
        self,
        in_ch: int,
        feature_ch: int,
        base: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        middle_feature_ch: int = 0,
        fine_feature_ch: int = 0,
        lateral_convolutions: bool = True,
    ):
        super().__init__()
        self.feature_ch = feature_ch
        if middle_feature_ch < 0 or fine_feature_ch < 0:
            raise ValueError("FPN feature-channel overrides must be >= 0")
        self.middle_ch = middle_feature_ch or max(16, feature_ch // 2)
        self.fine_ch = fine_feature_ch or max(8, feature_ch // 4)
        self.lateral_convolutions = bool(lateral_convolutions)

        # Bottom-up residual hierarchy (H/2, H/4, H/8), followed by
        # lateral/top-down fusion. Each map drives one cascade stage.
        c0 = max(8, feature_ch // 8)
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        self.casmvs_stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.casmvs_half = _residual2d_stage(c0, c1, 3, stride=2)
        self.casmvs_quarter = _residual2d_stage(c1, c2, 4, stride=2)
        self.casmvs_eighth = _residual2d_stage(c2, c3, 6, stride=2)

        if self.lateral_convolutions:
            # Original FPN: project the lateral maps up to the coarser width.
            self.casmvs_lateral_quarter = nn.Conv2d(c2, c3, 1, bias=False)
            self.casmvs_lateral_half = nn.Conv2d(c1, c2, 1, bias=False)
            self.casmvs_middle_reduce = nn.Identity()
            middle_fused_ch = c3
            self.casmvs_fine_reduce = nn.Conv2d(c3, c2, 1, bias=False)
            fine_fused_ch = c2
        else:
            # Reduced-width FPN: preserve the lateral maps and project only
            # the top-down path before element-wise addition.
            if self.middle_ch != c2 or self.fine_ch != c1:
                raise ValueError(
                    "--no_fpn_lateral_convolutions requires the middle and "
                    f"fine feature widths to match the bottom-up hierarchy "
                    f"({c2} and {c1} channels), but received "
                    f"{self.middle_ch} and {self.fine_ch}"
                )
            self.casmvs_lateral_quarter = nn.Identity()
            self.casmvs_lateral_half = nn.Identity()
            self.casmvs_middle_reduce = nn.Conv2d(c3, c2, 1, bias=False)
            middle_fused_ch = c2
            self.casmvs_fine_reduce = nn.Conv2d(c2, c1, 1, bias=False)
            fine_fused_ch = c1

        self.casmvs_coarse_out = nn.Sequential(
            nn.Conv2d(c3, feature_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU(inplace=True),
        )
        self.casmvs_middle_out = nn.Sequential(
            nn.Conv2d(middle_fused_ch, self.middle_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.middle_ch),
            nn.ReLU(inplace=True),
        )
        self.casmvs_fine_out = nn.Sequential(
            nn.Conv2d(fine_fused_ch, self.fine_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.fine_ch),
            nn.ReLU(inplace=True),
        )
        self.coarse_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.middle_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fine_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fusion_drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stem = self.casmvs_stem(x)
        half = self.casmvs_half(stem)
        quarter = self.casmvs_quarter(half)
        eighth = self.casmvs_eighth(quarter)
        quarter_fused = self.casmvs_lateral_quarter(quarter) + self.fusion_drop_path(
            F.interpolate(
                self.casmvs_middle_reduce(eighth),
                size=quarter.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        half_fused = self.casmvs_lateral_half(half) + self.fusion_drop_path(
            F.interpolate(
                self.casmvs_fine_reduce(quarter_fused),
                size=half.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        coarse = self.coarse_dropout(self.casmvs_coarse_out(eighth))
        middle = self.middle_dropout(self.casmvs_middle_out(quarter_fused))
        fine = self.fine_dropout(self.casmvs_fine_out(half_fused))
        return coarse, middle, fine


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = math.gcd(out_ch, min(8, out_ch))
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostHourglass3D(nn.Module):
    """Configurable 3-D encoder/decoder; wide layers run on smaller volumes."""

    def __init__(
        self,
        in_ch: int,
        base: int,
        levels: int = 2,
        bottleneck_dropout: float = 0.0,
    ):
        super().__init__()
        base = max(4, base)
        if levels < 1:
            raise ValueError("hourglass levels must be >= 1")
        self.stem = nn.Sequential(
            Conv3dBlock(in_ch, base),
            Conv3dBlock(base, base),
        )
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        channels = [base]
        for level in range(levels):
            in_width = channels[-1]
            out_width = base * (2 ** (level + 1))
            self.down.append(nn.Sequential(
                Conv3dBlock(in_width, out_width, stride=2),
                Conv3dBlock(out_width, out_width),
            ))
            channels.append(out_width)
        for level in range(levels, 0, -1):
            self.up.append(Conv3dBlock(channels[level], channels[level - 1]))
        self.bottleneck_dropout = (
            nn.Dropout3d(bottleneck_dropout)
            if bottleneck_dropout > 0 else nn.Identity()
        )
        self.out = nn.Conv3d(base, 1, 3, padding=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(volume)]
        for down in self.down:
            skips.append(down(skips[-1]))
        x = self.bottleneck_dropout(skips[-1])
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(
                up(x), size=skip.shape[-3:], mode="trilinear", align_corners=False
            ) + skip
        return self.out(x).squeeze(1)


class FullResolutionRefiner(nn.Module):
    def __init__(
        self,
        in_ch: int,
        fine_ch: int,
        width: int,
        max_residual_m: float,
        use_reference_input: bool = True,
    ):
        super().__init__()
        self.max_residual_m = float(max_residual_m)
        self.use_reference_input = bool(use_reference_input)
        width = max(16, width)
        body_in_ch = fine_ch + 1 + (in_ch if self.use_reference_input else 0)
        self.body = nn.Sequential(
            nn.Conv2d(body_in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(width // 2, 1, 3, padding=1)
        self.feature_channels = width // 2

    def forward(
        self,
        target: torch.Tensor,
        fine_feat: torch.Tensor,
        depth_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine_full = F.interpolate(
            fine_feat, size=depth_m.shape[-2:], mode="bilinear", align_corners=False
        )
        refiner_inputs = [fine_full, depth_m]
        if self.use_reference_input:
            refiner_inputs.insert(0, target)
        features = self.body(torch.cat(refiner_inputs, dim=1))
        residual = torch.tanh(self.residual(features)) * self.max_residual_m
        return (depth_m + residual).clamp(DEPTH_MIN, D_MAX), features


class ModernMVSNet(nn.Module):
    """Modern three-stage coarse-to-fine MVS network with a deep FPN."""

    architecture_name = "ModernMVSNet"

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        feature_ch: int | None = None,
        cost_base: int | None = None,
        fine_depths: int = 5,
        fine_window: float = 0.08,
        fine_offset_radius: float = 2.0,
        fine_window_min: float = 0.04,
        fine_window_max: float = 0.16,
        learned_fine_window: bool = False,
        reference_channels: int = 0,
        coarse_cost_channels: int = 0,
        fine_cost_channels: int = 0,
        refiner_channels: int = 0,
        refiner_max_residual_m: float | None = None,
        coarse_hourglass_levels: int = 2,
        fine_hourglass_levels: int = 2,
        fpn_dropout: float = 0.0,
        reference_dropout: float = 0.0,
        hourglass_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        middle_depths: int = 8,
        middle_window: float = 0.12,
        middle_cost_channels: int = 0,
        middle_hourglass_levels: int = 2,
        middle_feature_channels: int = 0,
        fine_feature_channels: int = 0,
        fpn_lateral_convolutions: bool = True,
        refiner_reference_input: bool = True,
        no_2d_refinement: bool = False,
    ):
        super().__init__()
        if fine_depths < 3:
            raise ValueError("fine_depths must be >= 3")
        if middle_depths < 3:
            raise ValueError("middle_depths must be >= 3")
        if middle_window <= 0:
            raise ValueError("middle_window must be > 0")
        if fine_window_min <= 0:
            raise ValueError("fine_window_min must be > 0")
        if fine_window_max < fine_window_min:
            raise ValueError("fine_window_max must be >= fine_window_min")
        feature_ch = feature_ch or max(32, base)
        cost_base = cost_base or max(8, base // 4)
        coarse_cost_base = coarse_cost_channels or cost_base
        middle_cost_base = middle_cost_channels or cost_base
        fine_cost_base = fine_cost_channels or cost_base
        self.fine_depths = int(fine_depths)
        self.middle_depths = int(middle_depths)
        self.middle_window = float(middle_window)
        self.fine_window = float(fine_window)
        self.fine_window_min = float(fine_window_min)
        self.fine_window_max = float(fine_window_max)
        self.refiner_max_residual_override = refiner_max_residual_m is not None
        if refiner_max_residual_m is not None and refiner_max_residual_m <= 0:
            raise ValueError("--refiner_max_residual_m must be > 0")
        self.refiner_max_residual_m = (
            float(refiner_max_residual_m)
            if refiner_max_residual_m is not None
            else min(0.05, self.fine_window)
        )
        self.fine_offset_radius = float(fine_offset_radius)
        self.learned_fine_window = bool(learned_fine_window)
        self.reference_channels = int(reference_channels)
        self.coarse_hourglass_levels = int(coarse_hourglass_levels)
        self.middle_hourglass_levels = int(middle_hourglass_levels)
        self.fine_hourglass_levels = int(fine_hourglass_levels)
        self.refiner_reference_input = bool(refiner_reference_input)
        self.no_2d_refinement = bool(no_2d_refinement)
        self.reference_dropout = float(reference_dropout)
        self.feature = FeaturePyramid(
            in_ch,
            feature_ch,
            base,
            dropout=fpn_dropout,
            drop_path=drop_path_rate,
            middle_feature_ch=middle_feature_channels,
            fine_feature_ch=fine_feature_channels,
            lateral_convolutions=fpn_lateral_convolutions,
        )
        if self.no_2d_refinement and not self.refiner_reference_input:
            raise ValueError(
                "--no_refiner_reference_input has no effect when "
                "--no_2d_refinement is active"
            )
        if self.refiner_max_residual_override and self.no_2d_refinement:
            raise ValueError(
                "--refiner_max_residual_m requires the 2-D residual refiner"
            )
        if self.reference_channels > 0:
            self.coarse_reference = nn.Conv2d(
                feature_ch, self.reference_channels, 1, bias=False
            )
            self.fine_reference = nn.Conv2d(
                self.feature.fine_ch, self.reference_channels, 1, bias=False
            )
            self.middle_reference = nn.Conv2d(
                self.feature.middle_ch, self.reference_channels, 1, bias=False
            )
        else:
            self.coarse_reference = None
            self.middle_reference = None
            self.fine_reference = None
        primary_coarse_channels = feature_ch
        primary_fine_channels = self.feature.fine_ch
        primary_middle_channels = self.feature.middle_ch
        volume_channels_coarse = (
            primary_coarse_channels + self.reference_channels + 1
        )
        volume_channels_fine = (
            primary_fine_channels + self.reference_channels + 1
        )
        volume_channels_middle = (
            primary_middle_channels
            + self.reference_channels
            + 1
        )
        self.coarse_cost = CostHourglass3D(
            volume_channels_coarse,
            coarse_cost_base,
            self.coarse_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.fine_cost = CostHourglass3D(
            volume_channels_fine,
            fine_cost_base,
            self.fine_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.middle_cost = CostHourglass3D(
            volume_channels_middle,
            middle_cost_base,
            self.middle_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        geometry_scales = "H/8/H/4/H/2"
        refiner_summary = (
            "none (bilinear only)"
            if self.no_2d_refinement
            else (
                f"{refiner_channels or max(32, min(base // 2, 128))}"
                f"/raw_reference={self.refiner_reference_input}"
                f"/max_residual={self.refiner_max_residual_m:g}m"
            )
        )
        self.capacity_summary = (
            f"decoder=deep_three_stage_fpn  "
            f"cascade_stages=3  "
            f"feature_coarse/middle/fine={feature_ch}"
            f"/{self.feature.middle_ch}"
            f"/{self.feature.fine_ch}  "
            f"volume_type=variance  "
            f"reference={self.reference_channels}  "
            f"cost_coarse/middle/fine={coarse_cost_base}"
            f"/{middle_cost_base}"
            f"/{fine_cost_base}  "
            f"geometry_scales={geometry_scales}  "
            f"fpn_lateral_convolutions={self.feature.lateral_convolutions}  "
            f"refiner={refiner_summary}  "
            f"hourglass_levels_coarse/middle/fine={self.coarse_hourglass_levels}"
            f"/{self.middle_hourglass_levels}"
            f"/{self.fine_hourglass_levels}  "
            f"  dropout(fpn/ref/hg)={fpn_dropout:g}/{reference_dropout:g}/{hourglass_dropout:g}"
            f"  drop_path={drop_path_rate:g}"
        )

        window_hidden = max(8, cost_base)
        if self.learned_fine_window:
            self.window_head = nn.Sequential(
                nn.Conv2d(self.feature.fine_ch + 1, window_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(window_hidden, 1, 3, padding=1),
            )

        refine_width = refiner_channels or max(32, min(base // 2, 128))
        self.refiner = (
            None
            if self.no_2d_refinement
            else FullResolutionRefiner(
                in_ch,
                self.feature.fine_ch,
                refine_width,
                max_residual_m=self.refiner_max_residual_m,
                use_reference_input=self.refiner_reference_input,
            )
        )
        refinement_feature_channels = (
            self.feature.fine_ch
            if self.no_2d_refinement
            else self.refiner.feature_channels
        )
        confidence_in = refinement_feature_channels + 5
        self.confidence_head = nn.Sequential(
            nn.Conv2d(confidence_in, max(16, cost_base * 2), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, cost_base * 2), 1, 1),
        )

    def _variance_volume(
        self,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        reference_projection: nn.Module | None,
        view_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MVSNet feature variance, including the reference as one observation."""
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref = feats[:, 0].unsqueeze(2).expand(-1, -1, D, -1, -1)
        feature_sum = ref.clone()
        feature_sq_sum = ref.square()
        observation_count = torch.ones(
            (B, 1, D, H, W), device=feats.device, dtype=feats.dtype
        )
        geometric_valid_sum = torch.zeros_like(observation_count)
        if view_valid_mask is None:
            view_valid_mask = torch.ones(
                (B, V), device=feats.device, dtype=torch.bool
            )
        elif view_valid_mask.shape != (B, V):
            raise ValueError(
                f"Expected view_valid_mask with shape {(B, V)}, got "
                f"{tuple(view_valid_mask.shape)}"
            )

        for view in range(1, V):
            warped, valid = homo_warp_features(
                feats[:, view],
                cam_mats[:, view],
                cam_mats[:, 0],
                K,
                depth_values,
                return_valid_mask=True,
            )
            sample_valid = view_valid_mask[:, view].to(valid.dtype).view(
                B, 1, 1, 1, 1
            )
            valid = valid * sample_valid
            # Invalid samples must not be treated as zero-valued observations.
            feature_sum = feature_sum + warped * valid
            feature_sq_sum = feature_sq_sum + warped.square() * valid
            observation_count = observation_count + valid
            geometric_valid_sum = geometric_valid_sum + valid

        mean = feature_sum / observation_count
        feature_variance = (
            feature_sq_sum / observation_count - mean.square()
        ).clamp_min(0.0)
        source_count = view_valid_mask[:, 1:].sum(dim=1).clamp_min(1).to(feats.dtype)
        valid_ratio = geometric_valid_sum / source_count[:, None, None, None, None]
        volume_parts = [feature_variance]
        if reference_projection is not None:
            reference = reference_projection(feats[:, 0])
            if self.reference_dropout > 0:
                reference = F.dropout2d(
                    reference, p=self.reference_dropout, training=self.training
                )
            volume_parts.append(reference.unsqueeze(2).expand(-1, -1, D, -1, -1))
        volume_parts.append(valid_ratio)
        return torch.cat(volume_parts, dim=1), valid_ratio

    @staticmethod
    def _regress(prob: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
        return torch.sum(prob * depth_values.to(prob.dtype), dim=1, keepdim=True)

    @staticmethod
    def _local_values(
        center_m: torch.Tensor,
        target_ref: torch.Tensor,
        count: int,
        half_window: float,
    ) -> torch.Tensor:
        center_up = F.interpolate(
            center_m, size=target_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        offsets = torch.linspace(
            -half_window,
            half_window,
            count,
            device=center_up.device,
            dtype=center_up.dtype,
        )
        return (center_up + offsets.view(1, count, 1, 1)).clamp(
            DEPTH_MIN, D_MAX
        )

    def _fine_values(
        self,
        coarse_m: torch.Tensor,
        fine_ref: torch.Tensor,
    ) -> torch.Tensor:
        coarse_up = F.interpolate(
            coarse_m, size=fine_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.learned_fine_window:
            half_window = self.fine_window_min + (
                self.fine_window_max - self.fine_window_min
            ) * torch.sigmoid(
                self.window_head(torch.cat([fine_ref, coarse_up], dim=1))
            )
            offset_radius = 1.0
        else:
            half_window = torch.full_like(
                coarse_up, self.fine_window * self.fine_offset_radius
            )
            offset_radius = 1.0
        offsets = torch.linspace(
            -offset_radius,
            offset_radius,
            self.fine_depths,
            device=coarse_up.device,
            dtype=coarse_up.dtype,
        )
        return (
            coarse_up + half_window * offsets.view(1, self.fine_depths, 1, 1)
        ).clamp(DEPTH_MIN, D_MAX)

    def forward(
        self,
        imgs: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        return_uncertainty: bool = False,
        view_valid_mask: torch.Tensor | None = None,
        detach_confidence_inputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        B, V, C, H, W = imgs.shape
        if view_valid_mask is None:
            view_valid_mask = torch.ones(
                (B, V), device=imgs.device, dtype=torch.bool
            )
        else:
            view_valid_mask = view_valid_mask.to(device=imgs.device, dtype=torch.bool)
        if not torch.all(view_valid_mask[:, 0]):
            raise ValueError("The reference view must be valid for every sample")
        if depth_values.dim() == 1:
            depth_values = depth_values.unsqueeze(0).expand(B, -1)
        elif depth_values.dim() != 2:
            raise ValueError(
                f"Expected global depth_values with shape (D,) or (B,D), got "
                f"{tuple(depth_values.shape)}"
            )
        flat_images = imgs.reshape(B * V, C, H, W)
        flat_valid = view_valid_mask.reshape(B * V)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        valid_feature_levels = self.feature(flat_images[flat_valid])
        feature_levels = tuple(
            valid_features.new_zeros(
                (B * V, *valid_features.shape[1:])
            ).index_copy(0, valid_indices, valid_features)
            for valid_features in valid_feature_levels
        )
        coarse_flat, middle_flat, fine_flat = feature_levels
        middle = middle_flat.view(B, V, *middle_flat.shape[1:])
        coarse = coarse_flat.view(B, V, *coarse_flat.shape[1:])
        fine = fine_flat.view(B, V, *fine_flat.shape[1:])

        K_coarse = K.clone()
        K_coarse[:, 0, :] *= coarse.shape[-1] / W
        K_coarse[:, 1, :] *= coarse.shape[-2] / H
        coarse_values = depth_values[:, :, None, None].expand(
            -1, -1, coarse.shape[-2], coarse.shape[-1]
        )
        coarse_volume, _ = self._variance_volume(
            coarse,
            cam_mats,
            K_coarse,
            coarse_values,
            self.coarse_reference,
            view_valid_mask,
        )
        coarse_logits = self.coarse_cost(coarse_volume)
        coarse_prob = F.softmax(-coarse_logits.float(), dim=1).to(coarse_logits.dtype)
        coarse_m = self._regress(coarse_prob, coarse_values)

        middle_values = self._local_values(
            coarse_m,
            middle[:, 0],
            self.middle_depths,
            self.middle_window,
        )
        K_middle = K.clone()
        K_middle[:, 0, :] *= middle.shape[-1] / W
        K_middle[:, 1, :] *= middle.shape[-2] / H
        middle_volume, _ = self._variance_volume(
            middle,
            cam_mats,
            K_middle,
            middle_values,
            self.middle_reference,
            view_valid_mask,
        )
        middle_logits = self.middle_cost(middle_volume)
        middle_prob = F.softmax(-middle_logits.float(), dim=1).to(
            middle_logits.dtype
        )
        middle_m = self._regress(middle_prob, middle_values)

        fine_values = self._fine_values(middle_m, fine[:, 0])
        K_fine = K.clone()
        K_fine[:, 0, :] *= fine.shape[-1] / W
        K_fine[:, 1, :] *= fine.shape[-2] / H
        fine_volume, fine_valid = self._variance_volume(
            fine,
            cam_mats,
            K_fine,
            fine_values,
            self.fine_reference,
            view_valid_mask,
        )
        fine_logits = self.fine_cost(fine_volume)
        fine_prob = F.softmax(-fine_logits.float(), dim=1).to(fine_logits.dtype)
        fine_m = self._regress(fine_prob, fine_values)
        depth_full = F.interpolate(
            fine_m, size=(H, W), mode="bilinear", align_corners=False
        )
        if self.no_2d_refinement:
            refined_m = depth_full.clamp(DEPTH_MIN, D_MAX)
            refinement_features = F.interpolate(
                fine[:, 0], size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            assert self.refiner is not None
            refined_m, refinement_features = self.refiner(
                imgs[:, 0], fine[:, 0], depth_full
            )

        final_norm = (
            (refined_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
        ).clamp(0.0, 1.0)

        confidence = None
        if return_uncertainty:
            eps = 1e-8
            posterior_std = torch.sum(
                fine_prob * (fine_values - fine_m).square(), dim=1, keepdim=True
            ).clamp_min(0.0).sqrt()
            entropy = -(fine_prob * torch.log(fine_prob + eps)).sum(dim=1, keepdim=True)
            max_prob = fine_prob.max(dim=1, keepdim=True).values
            valid_ratio = fine_valid.mean(dim=2)
            previous_up = F.interpolate(
                middle_m, size=fine_m.shape[-2:], mode="bilinear", align_corners=False
            )
            coarse_fine = torch.abs(fine_m - previous_up)
            diagnostics = torch.cat(
                [posterior_std, entropy, max_prob, valid_ratio, coarse_fine], dim=1
            )
            diagnostics = F.interpolate(
                diagnostics, size=(H, W), mode="bilinear", align_corners=False
            )
            if detach_confidence_inputs:
                refinement_features = refinement_features.detach()
                diagnostics = diagnostics.detach()
            confidence = torch.sigmoid(
                self.confidence_head(
                    torch.cat([refinement_features, diagnostics], dim=1)
                )
            )

        if return_uncertainty:
            return final_norm, confidence
        return final_norm



# Training / validation
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    viz=None,
    activity_diag=None,
    error_diag=None,
    uncertainty_diag=None,
    lambda_grad: float = 0.5,
    lambda_normal: float = 0.1,
    l1_loss_only: bool = False,
    uncertainty: bool = False,
    lambda_confidence: float = 0.1,
    confidence_abs_tolerance: float = 0.01,
    confidence_rel_tolerance: float = 0.01,
    detach_confidence_inputs: bool = False,
    ema_model=None,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_l1 = total_p95 = total_worst10_l1 = 0.0
    n_batches = 0
    phase = "train" if is_train else "val"
    t_phase_start = time.perf_counter()
    t_last = t_phase_start
    batches_at_last_log = 0
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with ctx:
        for batch in loader:
            imgs = batch["imgs"].to(device, non_blocking=True)
            cam_mats = batch["cam_mats"].to(device, non_blocking=True)
            K = batch["K"].to(device, non_blocking=True)
            depth_values = batch["depth_values"].to(device, non_blocking=True)
            dep_t = batch["dep_t"].to(device, non_blocking=True)
            mask_t = batch["mask_t"].to(device, non_blocking=True)
            view_valid_mask = batch["view_valid_mask"].to(device, non_blocking=True)

            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            model_out = model(
                imgs,
                cam_mats,
                K,
                depth_values,
                return_uncertainty=uncertainty,
                view_valid_mask=view_valid_mask,
                detach_confidence_inputs=detach_confidence_inputs,
            )
            pred_unc = None
            if uncertainty:
                pred, pred_unc = model_out
            else:
                pred = model_out

            pred_loss = pred
            if l1_loss_only:
                loss_final = l1_metres(pred_loss, dep_t, mask_t)
            else:
                loss_final = charbonnier_loss(pred_loss, dep_norm, mask_t)
                loss_final = loss_final + lambda_grad * gradient_loss(
                    pred_loss, dep_norm, mask_t
                )
                loss_final = loss_final + lambda_normal * normal_loss(
                    pred_loss, dep_norm, mask_t, K[0]
                )
            loss = loss_final
            if uncertainty and pred_unc is not None and lambda_confidence > 0:
                pred_m = pred_loss * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                with torch.no_grad():
                    error = torch.abs(pred_m.detach() - dep_t)
                    tolerance = confidence_abs_tolerance + confidence_rel_tolerance * dep_t
                    confidence_target = ((error < tolerance) & (mask_t > 0.5)).float()
                    valid = mask_t > 0.5
                    positive = confidence_target[valid].sum()
                    negative = (1.0 - confidence_target[valid]).sum()
                    pos_weight = negative / positive.clamp_min(1.0)
                    bce_weight = torch.where(
                        confidence_target > 0.5,
                        pos_weight.to(dtype=pred_unc.dtype),
                        torch.ones((), device=pred_unc.device, dtype=pred_unc.dtype),
                    )
                confidence_loss = F.binary_cross_entropy(
                    pred_unc.clamp(1e-6, 1.0 - 1e-6),
                    confidence_target,
                    reduction="none",
                )
                confidence_loss = (
                    confidence_loss * bce_weight * mask_t
                ).sum() / mask_t.sum().clamp_min(1.0)
                loss = loss + lambda_confidence * confidence_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if ema_model is not None:
                    ema_model.update(model)

            total_loss += float(loss.detach())
            with torch.no_grad():
                total_l1 += float(l1_metres(pred_loss, dep_t, mask_t))
                pred_m = pred_loss * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                valid_errors = torch.abs(pred_m - dep_t)[mask_t > 0.5]
                if valid_errors.numel() > 0:
                    total_p95 += float(torch.quantile(valid_errors.float(), 0.95))
                total_worst10_l1 += float(worst_fraction_l1_metres(pred_loss, dep_t, mask_t))
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

            with torch.no_grad():
                pred_m = pred_loss * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                tbl_ch = imgs[:, 0, NUM_BINS:NUM_BINS + 1]
                if viz is not None:
                    viz.add_batch(imgs[:, 0, :NUM_BINS], dep_t, mask_t, pred_m, table_depth=tbl_ch)
                if activity_diag is not None:
                    activity_diag.add_batch(imgs[:, 0, :NUM_BINS], dep_t, mask_t, pred_m)
                if error_diag is not None:
                    error_diag.add_batch(pred_m, dep_t, mask_t)
                if uncertainty and pred_unc is not None and uncertainty_diag is not None:
                    uncertainty_diag.add_batch(
                        1.0 - pred_unc,
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


def _event_activity_image(voxels: np.ndarray) -> np.ndarray:
    activity = np.abs(voxels).sum(axis=0).astype(np.float32)
    lo = float(activity.min())
    hi = float(activity.max())
    if hi - lo < 1e-8:
        return np.zeros_like(activity, dtype=np.float32)
    return np.clip((activity - lo) / (hi - lo), 0.0, 1.0)


def _write_pose_debug_text(
    out_txt: Path,
    view_ids: np.ndarray,
    cam_mats: np.ndarray,
) -> None:
    ref_T = cam_mats[0]
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Camera matrices used by multiview pose warp\n")
        f.write("cam_mats are T_event_from_world = T_event_from_ee @ inv(T_base_from_ee)\n")
        f.write("relative warp matrix is T_src_from_ref = T_src_cam_from_world @ inv(T_ref_cam_from_world)\n\n")
        for view_i, frame_idx in enumerate(view_ids):
            f.write(f"view {view_i}  frame {int(frame_idx)}\n")
            f.write("T_cam_from_world:\n")
            np.savetxt(f, cam_mats[view_i], fmt="% .8f")
            f.write("T_world_from_cam:\n")
            np.savetxt(f, np.linalg.inv(cam_mats[view_i]), fmt="% .8f")
            if view_i > 0:
                f.write("T_src_from_ref:\n")
                np.savetxt(f, cam_mats[view_i] @ np.linalg.inv(ref_T), fmt="% .8f")
            f.write("\n")


def debug_multiview_samples(
    dataset: Dataset,
    out_dir: Path,
    n_samples: int,
) -> None:
    """
    Save target/source event images and camera poses using the same sample path
    and pose matrices that training passes to homo_warp_features().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(max(int(n_samples), 1), len(dataset))
    sample_ids = np.linspace(0, len(dataset) - 1, n, dtype=np.int64)

    for debug_i, ds_i in enumerate(sample_ids):
        sample = dataset[int(ds_i)]
        imgs = sample["imgs"].numpy()
        cam_mats = sample["cam_mats"].numpy()
        view_ids = sample["view_ids"].numpy()
        ref_idx = int(sample["ref_idx"])

        V = imgs.shape[0]
        fig = plt.figure(figsize=(max(12, 2.6 * V), 7.5))
        gs = fig.add_gridspec(2, V, height_ratios=[1.0, 1.25])

        for v in range(V):
            ax = fig.add_subplot(gs[0, v])
            ax.imshow(_event_activity_image(imgs[v, :NUM_BINS]), cmap="gray", vmin=0.0, vmax=1.0)
            role = "target" if v == 0 else f"src {v}"
            ax.set_title(f"{role}\nframe {int(view_ids[v])}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        ax3d = fig.add_subplot(gs[1, :], projection="3d")
        centers = []
        axis_len = 0.04
        colors = ["tab:red", "tab:green", "tab:blue"]
        labels = ["x", "y", "z"]
        for v in range(V):
            T_world_from_cam = np.linalg.inv(cam_mats[v])
            center = T_world_from_cam[:3, 3]
            R = T_world_from_cam[:3, :3]
            centers.append(center)
            marker_color = "black" if v == 0 else "tab:orange"
            ax3d.scatter(center[0], center[1], center[2], color=marker_color, s=35)
            ax3d.text(center[0], center[1], center[2], f" {v}:{int(view_ids[v])}", fontsize=8)
            for axis_i, (color, label) in enumerate(zip(colors, labels)):
                direction = R[:, axis_i] * axis_len
                ax3d.quiver(
                    center[0], center[1], center[2],
                    direction[0], direction[1], direction[2],
                    color=color, linewidth=1.2,
                    arrow_length_ratio=0.25,
                )
                if v == 0:
                    tip = center + direction
                    ax3d.text(tip[0], tip[1], tip[2], label, color=color, fontsize=8)

        centers_np = np.stack(centers)
        ax3d.plot(centers_np[:, 0], centers_np[:, 1], centers_np[:, 2], color="0.35", linewidth=1.0)
        ax3d.set_title(
            "Camera poses used for warp: T_world_from_cam = inv(T_cam_from_world)",
            fontsize=10,
        )
        ax3d.set_xlabel("base/world x [m]")
        ax3d.set_ylabel("base/world y [m]")
        ax3d.set_zlabel("base/world z [m]")
        set_3d_axes_equal(ax3d)
        ax3d.view_init(elev=25, azim=-60)

        fig.suptitle(
            f"multiview debug sample {debug_i}  dataset item {int(ds_i)}  target frame {ref_idx}",
            fontsize=12,
        )
        fig.tight_layout()
        out_png = out_dir / f"sample_{debug_i:02d}_target_{ref_idx:06d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        out_txt = out_dir / f"sample_{debug_i:02d}_target_{ref_idx:06d}_poses.txt"
        _write_pose_debug_text(out_txt, view_ids, cam_mats)
        print(f"  debug sample {debug_i}: {out_png}")
        print(f"                  poses: {out_txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _make_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    min_lr = args.min_lr if args.min_lr is not None else args.lr * 1e-2
    schedule_epochs = max(1, args.epochs - args.warmup_epochs)
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=schedule_epochs,
            eta_min=min_lr,
        )
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_gamma,
            patience=args.lr_plateau_patience,
            min_lr=min_lr,
        )
    raise ValueError(f"Unsupported LR scheduler: {args.lr_scheduler}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-view event depth model with warped feature cost volume"
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT,
                        help="Dataset root containing train/ and eval/ sequence folders")
    parser.add_argument("--train_sequence_count", type=int, default=0,
                        help="Use only N deterministically ordered training sequences; 0 uses all")
    parser.add_argument("--train_sequence_seed", type=int, default=42,
                        help="Seed defining the nested deterministic training-sequence ordering")
    parser.add_argument("--train_frame_fraction", type=float, default=1.0,
                        help="Random fraction sampled independently from each training sequence per epoch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate.")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Optimizer weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum used only with --optimizer sgd.")
    parser.add_argument("--lr_scheduler", choices=("cosine", "step", "plateau", "none"),
                        default="cosine",
                        help="Learning-rate schedule. Defaults to the original cosine schedule.")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum LR for cosine/plateau; default is lr * 1e-2.")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linearly warm LR from lr/warmup_epochs to lr over this many epochs.")
    parser.add_argument("--lr_step_size", type=int, default=15,
                        help="Epoch interval for --lr_scheduler step.")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="LR decay factor for step/plateau schedules.")
    parser.add_argument("--lr_plateau_patience", type=int, default=5,
                        help="Validation-L1 patience for --lr_scheduler plateau.")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay used for validation/checkpoints; 0 disables EMA")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop after this many epochs without validation improvement; 0 disables")
    parser.add_argument("--lambda_grad", type=float, default=0.5,
                        help="Weight for multi-scale gradient loss term.")
    parser.add_argument("--lambda_normal", type=float, default=0.1,
                        help="Weight for surface-normal loss term.")
    parser.add_argument("--l1_loss_only", action="store_true",
                        help="Train directly with masked metric L1 in metres, ignoring auxiliary loss terms.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--feature_channels", type=int, default=0,
                        help="Shared CNN output channels; 0 means base_channels*4")
    parser.add_argument("--cost_channels", type=int, default=0,
                        help="3D cost CNN base width; 0 means max(base_channels//2, 8)")
    parser.add_argument("--reference_channels", type=int, default=0,
                        help="Compressed reference channels added to each cost volume")
    parser.add_argument("--coarse_cost_channels", type=int, default=0,
                        help="Coarse 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--fine_cost_channels", type=int, default=0,
                        help="Fine 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--middle_cost_channels", type=int, default=0,
                        help="Middle-stage 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--refiner_channels", type=int, default=0,
                        help="Full-resolution 2D refiner width; 0 selects automatically")
    parser.add_argument(
        "--refiner_max_residual_m",
        type=float,
        default=None,
        help=(
            "Override the symmetric maximum metric-depth "
            "correction of the 2-D refiner in metres; omitted uses "
            "min(0.05, fine_window)."
        ),
    )
    parser.add_argument(
        "--no_refiner_reference_input",
        action="store_true",
        help=(
            "Omit the raw full-resolution reference event/prior "
            "tensor from the 2-D refiner while retaining fine FPN features and depth."
        ),
    )
    parser.add_argument(
        "--no_2d_refinement",
        action="store_true",
        help=(
            "Bypass full-resolution 2-D residual refinement and "
            "use the bilinearly upsampled fine-stage depth directly."
        ),
    )
    parser.add_argument("--coarse_hourglass_levels", type=int, default=2,
                        help="Coarse-stage 3D hourglass levels")
    parser.add_argument("--fine_hourglass_levels", type=int, default=2,
                        help="Fine-stage 3D hourglass levels")
    parser.add_argument("--middle_hourglass_levels", type=int, default=2,
                        help="Middle-stage 3D hourglass levels")
    parser.add_argument("--middle_feature_channels", type=int, default=0,
                        help="Three-stage FPN only: middle-stage output channels; "
                             "0 uses feature_channels/2")
    parser.add_argument("--fine_feature_channels", type=int, default=0,
                        help="Three-stage FPN only: final-stage output channels; "
                             "0 uses feature_channels/4")
    parser.add_argument(
        "--no_fpn_lateral_convolutions",
        action="store_true",
        help=(
            "Keep the H/4 and H/2 bottom-up FPN branches unchanged and "
            "project only the top-down path before addition. The requested "
            "middle/fine feature widths must match the bottom-up widths."
        ),
    )
    parser.add_argument("--fpn_dropout", type=float, default=0.0,
                        help="Dropout2d probability on FPN outputs")
    parser.add_argument("--reference_dropout", type=float, default=0.0,
                        help="Dropout2d probability on reference features")
    parser.add_argument("--hourglass_dropout", type=float, default=0.0,
                        help="Dropout3d probability at hourglass bottlenecks")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                        help="Stochastic-depth rate on FPN fusion")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--view_interval", type=int, default=5)
    parser.add_argument("--pose_view_selection", action="store_true",
                        help="Select balanced before/after source views by camera motion instead of fixed frame offsets")
    parser.add_argument("--pose_move_threshold", type=float, default=0.01,
                        help="Minimum camera-center translation in metres between consecutive selected pose views")
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
            "sources; missing past/future slots are masked (default)"
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
    parser.add_argument("--coarse_depths", type=int, default=32,
                        help="Number of global/coarse depth planes between DEPTH_MIN and D_MAX")
    parser.add_argument(
        "--linear_depth_candidates",
        action="store_true",
        help=(
            "Use uniformly spaced metric-depth hypotheses for the global/coarse "
            "cost volume instead of the default inverse-depth spacing."
        ),
    )
    parser.add_argument("--fine_depths", type=int, default=5,
                        help="Number of fine per-pixel depth hypotheses around the coarse estimate")
    parser.add_argument("--middle_depths", type=int, default=8,
                        help="Middle-stage H/4 local depth hypotheses")
    parser.add_argument("--middle_window", type=float, default=0.12,
                        help="Middle-stage H/4 local search half-window in metres")
    parser.add_argument("--fine_window", type=float, default=0.08,
                        help="Fine-stage sigma/window in metres before multiplying by offsets")
    parser.add_argument("--fine_offset_radius", type=float, default=2.0,
                        help="Fine offsets span [-radius, radius]; 2 with 5 planes gives [-2,-1,0,1,2]")
    parser.add_argument("--fine_window_min", type=float, default=0.04,
                        help="Minimum learned fine-stage search half-width in metres")
    parser.add_argument("--fine_window_max", type=float, default=0.16,
                        help="Maximum learned fine-stage search half-width in metres")
    parser.add_argument("--learned_fine_window", action="store_true",
                        help="Predict a per-pixel fine-stage window from target features and coarse depth")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Train, return, and log learned per-pixel fusion confidence")
    parser.add_argument("--lambda_confidence", type=float, default=0.1,
                        help="Auxiliary BCE loss weight for learned confidence when --uncertainty is enabled")
    parser.add_argument("--confidence_abs_tolerance", type=float, default=0.01,
                        help="Absolute safe-to-fuse confidence tolerance in metres")
    parser.add_argument("--confidence_rel_tolerance", type=float, default=0.01,
                        help="Relative safe-to-fuse confidence tolerance as a fraction of GT depth")
    parser.add_argument(
        "--detach_confidence_inputs",
        action="store_true",
        help=(
            "Stop confidence-loss gradients at the confidence-head inputs so "
            "confidence training cannot alter the shared depth network"
        ),
    )
    parser.add_argument("--out_dir", type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "mvs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fill_invalid", action="store_true",
                        help="Fill pixels with no depth measurement using the table-plane prior")
    parser.add_argument("--no_augmentations", action="store_true",
                        help="Disable all training-time multi-view data augmentations")
    parser.add_argument("--no_source_view_dropout", action="store_true",
                        help="Disable randomly zeroing whole source views during training")
    parser.add_argument("--no_pose_noise", action="store_true",
                        help="Disable small random source-pose perturbations during training")
    parser.add_argument("--no_event_noise_per_view", action="store_true",
                        help="Disable independent additive event-voxel noise per view during training")
    parser.add_argument("--no_cross_view_event_dropout", action="store_true",
                        help="Disable shared cross-view event-bin dropout during training")
    parser.add_argument("--no_occlusion_view_masking", action="store_true",
                        help="Disable random rectangular occlusion/view masks during training")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name used in checkpoint filenames. Prompted if not provided.")
    parser.add_argument("--tb_root", type=Path, default=DEFAULT_TB_ROOT,
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/mvs/<name>.")
    parser.add_argument("--debug_views", action="store_true",
                        help="Save target/source event images and camera-pose plots, then exit.")
    parser.add_argument("--debug_samples", type=int, default=4,
                        help="Number of target samples to visualise with --debug_views.")
    parser.add_argument("--debug_out", type=Path,
                        default=_SCRIPT_DIR / "debug" / "mvs",
                        help="Output directory for --debug_views PNGs and pose matrices.")
    args = parser.parse_args()

    if args.pose_view_selection and args.num_views % 2 != 1:
        parser.error("--pose_view_selection requires odd --num_views for balanced before/after sources")
    if args.base_channels <= 0:
        parser.error("--base_channels must be > 0")
    if args.train_sequence_count < 0:
        parser.error("--train_sequence_count must be >= 0")
    if not (0 < args.train_frame_fraction <= 1):
        parser.error("--train_frame_fraction must be in (0, 1]")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.min_lr is not None and args.min_lr < 0:
        parser.error("--min_lr must be >= 0")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be >= 0")
    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs must be >= 0")
    if args.warmup_epochs >= args.epochs:
        parser.error("--warmup_epochs must be smaller than --epochs")
    if args.lr_step_size <= 0:
        parser.error("--lr_step_size must be > 0")
    if not (0 < args.lr_gamma < 1):
        parser.error("--lr_gamma must be between 0 and 1")
    if args.lr_plateau_patience < 0:
        parser.error("--lr_plateau_patience must be >= 0")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        parser.error("--ema_decay must be in [0, 1)")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience must be >= 0")
    if args.lambda_confidence < 0:
        parser.error("--lambda_confidence must be >= 0")
    if args.reference_channels < 0:
        parser.error("--reference_channels must be >= 0")
    if any(x < 0 for x in (
        args.coarse_cost_channels, args.middle_cost_channels, args.fine_cost_channels
    )):
        parser.error("stage-specific cost channel counts must be >= 0")
    if args.refiner_channels < 0:
        parser.error("--refiner_channels must be >= 0")
    if (
        args.refiner_max_residual_m is not None
        and args.refiner_max_residual_m <= 0
    ):
        parser.error("--refiner_max_residual_m must be > 0")
    if (
        args.refiner_max_residual_m is not None
        and args.no_2d_refinement
    ):
        parser.error(
            "--refiner_max_residual_m requires the 2-D residual refiner"
        )
    if args.no_2d_refinement and args.no_refiner_reference_input:
        parser.error(
            "--no_refiner_reference_input has no effect with --no_2d_refinement"
        )
    if any(x < 1 for x in (
        args.coarse_hourglass_levels,
        args.middle_hourglass_levels,
        args.fine_hourglass_levels,
    )):
        parser.error("stage-specific hourglass levels must be >= 1")
    if args.middle_depths < 3:
        parser.error("--middle_depths must be >= 3")
    if args.middle_window <= 0:
        parser.error("--middle_window must be > 0")
    for name in ("fpn_dropout", "reference_dropout", "hourglass_dropout", "drop_path_rate"):
        if not (0 <= getattr(args, name) < 1):
            parser.error(f"--{name} must be in [0, 1)")
    if args.confidence_abs_tolerance < 0:
        parser.error("--confidence_abs_tolerance must be >= 0")
    if args.confidence_rel_tolerance < 0:
        parser.error("--confidence_rel_tolerance must be >= 0")
    if args.name is None and not args.debug_views:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calib = load_event_calibration(_CAM_DATA)

    train_root = args.data_dir / "train"
    eval_root = args.data_dir / "eval"
    train_seqs = find_precomputed_sequences(train_root)
    val_seqs = find_precomputed_sequences(eval_root)

    if not train_seqs or not val_seqs:
        missing = []
        if not train_seqs:
            missing.append(str(train_root))
        if not val_seqs:
            missing.append(str(eval_root))
        sys.exit(
            f"[ERROR] No valid sequences found in: {', '.join(missing)}\n"
            "        --data_dir must contain train/ and eval/, each holding one or more\n"
            "        valid sequences with voxels, depth, poses.h5, and table_plane.h5.\n"
            "        Run: python3 data_precomputation/precompute_table_plane.py"
        )

    all_train_seqs = train_seqs
    sequence_rng = np.random.default_rng(args.train_sequence_seed)
    sequence_order = sequence_rng.permutation(len(all_train_seqs))
    ordered_train_seqs = [all_train_seqs[int(i)] for i in sequence_order]
    if args.train_sequence_count > len(all_train_seqs):
        parser.error(
            f"--train_sequence_count={args.train_sequence_count} exceeds the "
            f"{len(all_train_seqs)} available training sequences"
        )
    train_seqs = (
        ordered_train_seqs[:args.train_sequence_count]
        if args.train_sequence_count > 0
        else ordered_train_seqs
    )

    print(f"Dataset root: {args.data_dir}")
    print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
    print(f"  Eval  ({len(val_seqs)}): {[d.name for d in val_seqs]}")
    print(
        f"  Train sequence subset: {len(train_seqs)}/{len(all_train_seqs)} "
        f"(seed={args.train_sequence_seed})"
    )

    ds_kw = dict(
        calib=calib,
        num_views=args.num_views,
        view_interval=args.view_interval,
        pose_view_selection=args.pose_view_selection,
        pose_move_threshold=args.pose_move_threshold,
        allow_unbalanced_pose_views=args.allow_unbalanced_pose_views,
        coarse_depths=args.coarse_depths,
        linear_depth_candidates=args.linear_depth_candidates,
        fill_invalid=args.fill_invalid,
        pose_channels=False,
    )
    train_aug = MultiViewAugConfig(
        enabled=not args.no_augmentations,
        source_view_dropout=not args.no_source_view_dropout,
        pose_noise=not args.no_pose_noise,
        event_noise_per_view=not args.no_event_noise_per_view,
        cross_view_event_dropout=not args.no_cross_view_event_dropout,
        occlusion_view_masking=not args.no_occlusion_view_masking,
    )
    val_aug = MultiViewAugConfig(enabled=False)

    train_sets = [MultiViewTableDataset(d, **ds_kw, aug=train_aug) for d in train_seqs]
    val_sets = [MultiViewTableDataset(d, **ds_kw, aug=val_aug) for d in val_seqs]

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]
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
    if args.train_frame_fraction < 1.0:
        sampled_per_epoch = sum(
            min(len(ds), max(1, int(round(len(ds) * args.train_frame_fraction))))
            for ds in train_sets
        )
        print(
            f"  Train frame sampling: {100.0 * args.train_frame_fraction:.1f}% "
            f"per sequence per epoch ({sampled_per_epoch}/{len(train_ds)} samples)"
        )
    else:
        print("  Train frame sampling: all frames")
    coarse_candidate_distribution = (
        "linear-depth" if args.linear_depth_candidates else "inverse-depth"
    )
    print(
        f"  Multi-view: views={args.num_views}, interval={args.view_interval}, "
        f"coarse {coarse_candidate_distribution} "
        f"planes={args.coarse_depths} [{DEPTH_MIN:.3f}, {D_MAX:.3f}] m"
    )
    if args.pose_view_selection:
        print(
            f"  View selection: pose-based, "
            f"{'up to four sources per side; missing boundary views masked' if args.allow_unbalanced_pose_views else 'strictly balanced before/after'}, "
            f"translation threshold={args.pose_move_threshold:g} m"
        )
    else:
        print("  View selection: fixed frame offsets")
    print(
        f"  Fine stage: planes={args.fine_depths}, window={args.fine_window:.4f} m, "
        f"offset radius={args.fine_offset_radius:g}, "
        f"learned half-width=[{args.fine_window_min:g}, {args.fine_window_max:g}] m, "
        f"learned window={args.learned_fine_window}\n"
    )
    print(
        f"  Loss weights: grad={args.lambda_grad:g}, normal={args.lambda_normal:g}, "
        f"l1_loss_only={args.l1_loss_only}, "
        f"confidence={args.lambda_confidence:g} "
        f"(abs_tol={args.confidence_abs_tolerance:g} m, "
        f"rel_tol={args.confidence_rel_tolerance:g}, "
        f"detached={args.detach_confidence_inputs}), "
        "final-stage supervision only\n"
    )
    min_lr = args.min_lr if args.min_lr is not None else args.lr * 1e-2
    print(
        f"  Optimizer: {args.optimizer}, lr={args.lr:g}, weight_decay={args.weight_decay:g}, "
        f"scheduler={args.lr_scheduler}, min_lr={min_lr:g}, "
        f"warmup_epochs={args.warmup_epochs}, lr_gamma={args.lr_gamma:g}\n"
    )
    print(
        "  Train augmentations: "
        f"enabled={train_aug.enabled}, "
        f"source_view_dropout={train_aug.source_view_dropout}, "
        f"p={train_aug.source_view_dropout_prob:g}; "
        f"pose_noise={train_aug.pose_noise}, "
        f"std={train_aug.pose_translation_std:g}m/"
        f"{train_aug.pose_rotation_std_deg:g}deg; "
        f"event_noise_per_view={train_aug.event_noise_per_view}, "
        f"std={train_aug.event_noise_std:g}; "
        f"cross_view_event_dropout={train_aug.cross_view_event_dropout}, "
        f"p={train_aug.cross_view_event_dropout_prob:g}; "
        f"occlusion_view_masking={train_aug.occlusion_view_masking}, "
        f"p={train_aug.occlusion_prob:g}, rects<= {train_aug.occlusion_max_rects}, "
        f"fraction={train_aug.occlusion_frac_range}\n"
    )

    if args.debug_views:
        print(f"Debug mode: saving {args.debug_samples} sample(s) to {args.debug_out}")
        debug_multiview_samples(train_ds, args.debug_out, args.debug_samples)
        print("Debug mode complete; exiting before training.")
        return

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_sampler = None
    if args.train_frame_fraction < 1.0:
        train_sampler = PerSequenceFractionSampler(
            [len(ds) for ds in train_sets],
            args.train_frame_fraction,
            args.seed,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kw,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    in_ch = NUM_BINS + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = args.base_channels
    feature_channels = (
        args.feature_channels if args.feature_channels > 0 else base_channels * 4
    )
    cost_channels = (
        args.cost_channels if args.cost_channels > 0 else max(base_channels // 2, 8)
    )
    model = ModernMVSNet(
        in_ch=in_ch,
        base=base_channels,
        feature_ch=feature_channels,
        cost_base=cost_channels,
        fine_depths=args.fine_depths,
        fine_window=args.fine_window,
        fine_offset_radius=args.fine_offset_radius,
        fine_window_min=args.fine_window_min,
        fine_window_max=args.fine_window_max,
        learned_fine_window=args.learned_fine_window,
        reference_channels=args.reference_channels,
        coarse_cost_channels=args.coarse_cost_channels,
        fine_cost_channels=args.fine_cost_channels,
        refiner_channels=args.refiner_channels,
        refiner_max_residual_m=args.refiner_max_residual_m,
        refiner_reference_input=not args.no_refiner_reference_input,
        no_2d_refinement=args.no_2d_refinement,
        coarse_hourglass_levels=args.coarse_hourglass_levels,
        fine_hourglass_levels=args.fine_hourglass_levels,
        fpn_dropout=args.fpn_dropout,
        reference_dropout=args.reference_dropout,
        hourglass_dropout=args.hourglass_dropout,
        drop_path_rate=args.drop_path_rate,
        middle_depths=args.middle_depths,
        middle_window=args.middle_window,
        middle_cost_channels=args.middle_cost_channels,
        middle_hourglass_levels=args.middle_hourglass_levels,
        middle_feature_channels=args.middle_feature_channels,
        fine_feature_channels=args.fine_feature_channels,
        fpn_lateral_convolutions=not args.no_fpn_lateral_convolutions,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_arch = getattr(model, "architecture_name", model.__class__.__name__)
    print(
        f"{model_arch}  in_ch={in_ch}  "
        f"base={base_channels}  feature={feature_channels}  "
        f"feature_encoder=deep_fpn  cost={cost_channels}  "
        f"coarse_depths={args.coarse_depths}  middle_depths={args.middle_depths}  "
        f"fine_depths={args.fine_depths}  "
        f"cost_volume=variance  "
        f"uncertainty={args.uncertainty}  "
        f"parameters: {n_params:,}"
    )
    if hasattr(model, "capacity_summary"):
        print(f"  Modern capacity: {model.capacity_summary}")
    print(f"Device: {device}\n")

    optimizer = _make_optimizer(args, model)
    scheduler = _make_lr_scheduler(args, optimizer)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA: enabled (decay={args.ema_decay:g}); validation/checkpoints use EMA weights")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("mvs", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")

    viz_train = VizLogger(writer, n_samples=4, tag="viz/train")
    viz_val = VizLogger(writer, n_samples=4, tag="viz/val")
    activity_train = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/train")
    activity_val = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/val")
    error_train = ErrorDistributionSpatialLogger(
        writer, tag="error/train", images_only=False
    )
    error_val = ErrorDistributionSpatialLogger(
        writer, tag="error/val", images_only=False
    )
    uncertainty_train = (
        UncertaintyErrorLogger(
            writer,
            tag="uncertainty/train",
            images_only=True,
        )
        if args.uncertainty else None
    )
    uncertainty_val = (
        UncertaintyErrorLogger(
            writer,
            tag="uncertainty/val",
            images_only=True,
        )
        if args.uncertainty else None
    )

    best_val_l1 = float("inf")
    best_val_p95 = float("inf")
    best_val_worst10 = float("inf")
    epochs_without_improvement = 0
    ckpt: dict = {}
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            _set_optimizer_lr(optimizer, warmup_lr)

        tr_loss, tr_l1, tr_p95, tr_worst10 = run_epoch(
            model, train_loader, optimizer, device,
            viz=viz_train,
            activity_diag=activity_train,
            error_diag=error_train,
            uncertainty_diag=uncertainty_train,
            lambda_grad=args.lambda_grad,
            lambda_normal=args.lambda_normal,
            l1_loss_only=args.l1_loss_only,
            uncertainty=args.uncertainty,
            lambda_confidence=args.lambda_confidence,
            confidence_abs_tolerance=args.confidence_abs_tolerance,
            confidence_rel_tolerance=args.confidence_rel_tolerance,
            detach_confidence_inputs=args.detach_confidence_inputs,
            ema_model=ema,
        )
        validation_model = ema.model if ema is not None else model
        va_loss, va_l1, va_p95, va_worst10 = run_epoch(
            validation_model, val_loader, None, device,
            viz=viz_val,
            activity_diag=activity_val,
            error_diag=error_val,
            uncertainty_diag=uncertainty_val,
            lambda_grad=args.lambda_grad,
            lambda_normal=args.lambda_normal,
            l1_loss_only=args.l1_loss_only,
            uncertainty=args.uncertainty,
            lambda_confidence=args.lambda_confidence,
            confidence_abs_tolerance=args.confidence_abs_tolerance,
            confidence_rel_tolerance=args.confidence_rel_tolerance,
            detach_confidence_inputs=args.detach_confidence_inputs,
        )
        if scheduler is not None and epoch > args.warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_l1)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

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
        vram_r = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
            f"L1: {tr_l1:.4f}/{va_l1:.4f} m  "
            f"p95: {tr_p95:.4f}/{va_p95:.4f} m  "
            f"worst10: {tr_worst10:.4f}/{va_worst10:.4f} m  "
            f"lr: {current_lr:.3e}  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("l1/train", tr_l1, epoch)
        writer.add_scalar("l1/val", va_l1, epoch)
        writer.add_scalar("p95/train", tr_p95, epoch)
        writer.add_scalar("p95/val", va_p95, epoch)
        writer.add_scalar("l1_worst10/train", tr_worst10, epoch)
        writer.add_scalar("l1_worst10/val", va_worst10, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        ckpt = {
            "epoch": epoch,
            "model": validation_model.state_dict(),
            "model_arch": model_arch,
            "intrinsics_transform": INTRINSICS_TRANSFORM,
            "val_l1": va_l1,
            "val_p95": va_p95,
            "val_l1_worst10": va_worst10,
            "train_sequence_count": len(train_seqs),
            "train_sequence_seed": args.train_sequence_seed,
            "train_sequences": [d.name for d in train_seqs],
            "train_frame_fraction": args.train_frame_fraction,
            "base": base_channels,
            "base_channels_arg": args.base_channels,
            "feature_channels": feature_channels,
            "cost_channels": cost_channels,
            "reference_channels": args.reference_channels,
            "coarse_cost_channels": args.coarse_cost_channels,
            "middle_cost_channels": args.middle_cost_channels,
            "fine_cost_channels": args.fine_cost_channels,
            "refiner_channels": args.refiner_channels,
            "refiner_max_residual_m": args.refiner_max_residual_m,
            "no_refiner_reference_input": args.no_refiner_reference_input,
            "no_2d_refinement": args.no_2d_refinement,
            "coarse_hourglass_levels": args.coarse_hourglass_levels,
            "middle_hourglass_levels": args.middle_hourglass_levels,
            "fine_hourglass_levels": args.fine_hourglass_levels,
            "middle_feature_channels": args.middle_feature_channels,
            "fine_feature_channels": args.fine_feature_channels,
            "no_fpn_lateral_convolutions": args.no_fpn_lateral_convolutions,
            "fpn_dropout": args.fpn_dropout,
            "reference_dropout": args.reference_dropout,
            "hourglass_dropout": args.hourglass_dropout,
            "drop_path_rate": args.drop_path_rate,
            "ema_decay": args.ema_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "in_ch": in_ch,
            "num_views": args.num_views,
            "view_interval": args.view_interval,
            "pose_view_selection": args.pose_view_selection,
            "pose_move_threshold": args.pose_move_threshold,
            "allow_unbalanced_pose_views": args.allow_unbalanced_pose_views,
            "allow_fewer_pose_views": args.allow_unbalanced_pose_views,
            "coarse_depths": args.coarse_depths,
            "linear_depth_candidates": args.linear_depth_candidates,
            "middle_depths": args.middle_depths,
            "middle_window": args.middle_window,
            "fine_depths": args.fine_depths,
            "fine_window": args.fine_window,
            "fine_offset_radius": args.fine_offset_radius,
            "fine_window_min": args.fine_window_min,
            "fine_window_max": args.fine_window_max,
            "learned_fine_window": args.learned_fine_window,
            "uncertainty": args.uncertainty,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "lr_scheduler": args.lr_scheduler,
            "min_lr": min_lr,
            "warmup_epochs": args.warmup_epochs,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "lr_plateau_patience": args.lr_plateau_patience,
            "lambda_grad": args.lambda_grad,
            "lambda_normal": args.lambda_normal,
            "lambda_confidence": args.lambda_confidence,
            "confidence_abs_tolerance": args.confidence_abs_tolerance,
            "confidence_rel_tolerance": args.confidence_rel_tolerance,
            "detach_confidence_inputs": args.detach_confidence_inputs,
            "l1_loss_only": args.l1_loss_only,
            "augmentation": {
                "enabled": train_aug.enabled,
                "source_view_dropout": train_aug.source_view_dropout,
                "source_view_dropout_prob": train_aug.source_view_dropout_prob,
                "pose_noise": train_aug.pose_noise,
                "pose_translation_std_m": train_aug.pose_translation_std,
                "pose_rotation_std_deg": train_aug.pose_rotation_std_deg,
                "event_noise_per_view": train_aug.event_noise_per_view,
                "event_noise_std": train_aug.event_noise_std,
                "cross_view_event_dropout": train_aug.cross_view_event_dropout,
                "cross_view_event_dropout_prob": train_aug.cross_view_event_dropout_prob,
                "occlusion_view_masking": train_aug.occlusion_view_masking,
                "occlusion_prob": train_aug.occlusion_prob,
                "occlusion_max_rects": train_aug.occlusion_max_rects,
                "occlusion_frac_range": train_aug.occlusion_frac_range,
            },
            "depth_min": DEPTH_MIN,
            "depth_max": D_MAX,
        }
        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            epochs_without_improvement = 0
            torch.save(ckpt, args.out_dir / f"best_l1_{args.name}.pth")
            print(f"  -> new best L1 checkpoint  (val L1 = {va_l1:.4f} m)")
        else:
            epochs_without_improvement += 1
        if va_p95 < best_val_p95:
            best_val_p95 = va_p95
            torch.save(ckpt, args.out_dir / f"best_p95_{args.name}.pth")
            print(f"  -> new best p95 checkpoint  (val p95 = {va_p95:.4f} m)")
        if va_worst10 < best_val_worst10:
            best_val_worst10 = va_worst10
            torch.save(ckpt, args.out_dir / f"best_l1_worst10_{args.name}.pth")
            print(
                "  -> new best worst10 checkpoint  "
                f"(val worst10 L1 = {va_worst10:.4f} m)"
            )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                f"without validation improvement."
            )
            break

    torch.save(ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(
        f"\nDone. Best val L1: {best_val_l1:.4f} m, "
        f"p95: {best_val_p95:.4f} m, "
        f"worst10 L1: {best_val_worst10:.4f} m"
    )


if __name__ == "__main__":
    main()
