#!/usr/bin/env python3
"""
multiview.py - Multi-view event depth training with a table-plane prior.

This is the multi-view counterpart to train_unet_table.py.  Each sample uses a
target event voxel frame plus neighbouring source frames:

    [x_i | table_plane_channel_i] for i in target + source frames

A shared 2D CNN extracts per-view features.  Source features are first warped
into the target frustum for coarse inverse-depth candidates using poses from
hdf5/poses.h5 and event-camera calibration from camera_data/.  The coarse
distribution is regressed to a rough depth map, then a second cost volume uses
per-pixel fine hypotheses around that rough depth:

    d_i(u, v) = d_hat(u, v) + sigma(u, v) * epsilon_i

sigma is either a fixed window or predicted from target features.

Usage:
    python3 training/multiview.py --data_dir data/real
    python3 training/multiview.py --data_dir data/real --num_views 5
    python3 training/multiview.py --num_depths 32 --fine_depths 5 --view_interval 5
    python3 training/multiview.py --pose_view_selection --pose_move_threshold 0.01 --num_views 5
    python3 training/multiview.py --masked_warp_aggregation
    python3 training/multiview.py --cost_volume_ref_features
    python3 training/multiview.py --model_scale 1.5
    python3 training/multiview.py --feature_encoder resnet18_h4
"""

from __future__ import annotations

import argparse
import copy
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

from train_unet import (
    DEPTH_MIN, D_MAX, NUM_BINS, _SCRIPT_DIR, DATA_ROOT,
    compute_loss, _l1_metres, _worst_percent_l1_metres,
)
from tensorboard_helper import (
    DEFAULT_TB_ROOT,
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
    tensorboard_run_dir,
    UncertaintyErrorLogger,
    VizLogger,
)

_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"


@dataclass(frozen=True)
class MultiViewAugConfig:
    """Training-time multi-view augmentation toggles and conservative defaults."""
    enabled: bool = False
    source_view_dropout: bool = True
    pose_noise: bool = True
    event_noise_per_view: bool = True
    cross_view_event_dropout: bool = True
    occlusion_view_masking: bool = True
    source_view_dropout_prob: float = 0.25
    pose_translation_std: float = 0.002
    pose_rotation_std_deg: float = 0.35
    event_noise_std: float = 0.03
    cross_view_event_dropout_prob: float = 0.04
    occlusion_prob: float = 0.45
    occlusion_max_rects: int = 2
    occlusion_frac_range: tuple[float, float] = (0.08, 0.28)


# ---------------------------------------------------------------------------
# Calibration / geometry
# ---------------------------------------------------------------------------

def _load_event_calibration() -> dict:
    """Load event intrinsics plus T_event_from_ee from camera_data."""
    ev = np.load(_CAM_DATA / "event_intrinsics.npz")
    K_native = ev["camera_matrix"].astype(np.float32)
    native_w = int(ev["image_size"][0])
    native_h = int(ev["image_size"][1])

    T_rgb_from_ee = np.load(_CAM_DATA / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(_CAM_DATA / "T_event_from_rgb.npz")["T"].astype(np.float32)
    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee

    return {
        "K_native": K_native,
        "native_hw": (native_h, native_w),
        "T_event_from_ee": T_event_from_ee,
    }


def _scale_K(K: np.ndarray, native_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    native_h, native_w = native_hw
    target_h, target_w = target_hw
    K_scaled = K.copy()
    K_scaled[0, :] *= target_w / native_w
    K_scaled[1, :] *= target_h / native_h
    return K_scaled.astype(np.float32)


def _scale_K_resize_crop(
    K: np.ndarray,
    native_hw: tuple[int, int],
    resize_hw: tuple[int, int],
    crop_hw: tuple[int, int],
) -> np.ndarray:
    """Scale event intrinsics through resize + centred crop.

    This matches the preprocessing used by project_realsense_to_event.py and
    precompute_voxels.py. The historical direct native->tensor scaling remains
    unchanged unless --fix_transform is enabled.
    """
    native_h, native_w = native_hw
    resize_h, resize_w = resize_hw
    crop_h, crop_w = crop_hw
    K_scaled = K.copy()
    K_scaled[0, :] *= resize_w / native_w
    K_scaled[1, :] *= resize_h / native_h
    K_scaled[0, 2] -= (resize_w - crop_w) / 2.0
    K_scaled[1, 2] -= (resize_h - crop_h) / 2.0
    return K_scaled.astype(np.float32)


def _inverse_depth_candidates(num_depths: int, depth_min: float, depth_max: float) -> np.ndarray:
    if num_depths < 2:
        raise ValueError(f"--num_depths must be >= 2, got {num_depths}")
    inv = np.linspace(1.0 / depth_min, 1.0 / depth_max, num_depths, dtype=np.float32)
    return (1.0 / inv).astype(np.float32)


def _table_plane_filename(fix_transform: bool) -> str:
    return "fixed_table_plane.h5" if fix_transform else "table_plane.h5"


def _pose_channels_from_base_event(T_base_from_event: np.ndarray) -> np.ndarray:
    """Return six pose values: event-camera position and optical axis in base frame."""
    position = T_base_from_event[:3, 3].astype(np.float32)
    optical_axis = T_base_from_event[:3, 2].astype(np.float32)
    norm = float(np.linalg.norm(optical_axis))
    if norm > 1e-6:
        optical_axis = optical_axis / norm
    return np.concatenate([position, optical_axis]).astype(np.float32)


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
        num_depths: int = 32,
        use_mask: bool = True,
        fill_invalid: bool = False,
        fix_transform: bool = False,
        pose_channels: bool = False,
        recurrent: bool = False,
        recurrent_enrollment_range: int = 0,
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
        if recurrent and num_views != 1:
            raise ValueError("--recurrent only supports single-view input; set --num_views 1")
        if recurrent and pose_view_selection:
            raise ValueError("--recurrent cannot be combined with --pose_view_selection")
        if recurrent_enrollment_range < 0:
            raise ValueError("--recurrent_enrollment_range must be >= 0")

        self.seq_dir = Path(seq_dir)
        self.num_views = num_views
        self.view_interval = view_interval
        self.pose_view_selection = pose_view_selection
        self.pose_move_threshold = float(pose_move_threshold)
        self.fill_invalid = fill_invalid
        self.pose_channels = bool(pose_channels)
        self.recurrent = bool(recurrent)
        self.recurrent_enrollment_range = int(recurrent_enrollment_range)
        self.aug = aug or MultiViewAugConfig(enabled=False)

        self.voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path = self.seq_dir / "hdf5" / "depth_in_event_frame.h5"
        self.mask_path = self.seq_dir / "hdf5" / "spatial_mask.h5"
        self.poses_path = self.seq_dir / "hdf5" / "poses.h5"
        self.table_plane_path = self.seq_dir / "hdf5" / _table_plane_filename(fix_transform)

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
            table_fix_transform = bool(f.attrs.get("fix_transform", False))
        with h5py.File(self.poses_path, "r") as f:
            ee_T = f["ee_T"][:].astype(np.float32)

        self.n_frames = min(n_d, n_v, n_t, len(ee_T))
        self.has_mask = use_mask and self.mask_path.exists()
        if fix_transform and not table_fix_transform:
            print(
                f"  [{self.seq_dir.name}] WARNING: -fix_transform is active for MVS K, "
                f"but hdf5/{self.table_plane_path.name} was not marked as fixed. Re-run "
                "data_precomputation/precompute_table_plane.py -fix_transform for a "
                "fully consistent table prior."
            )
        if fix_transform:
            native_h, native_w = calib["native_hw"]
            resize_h = int(vox_attrs.get("resize_h", vox_h))
            resize_w = int(vox_attrs.get("resize_w", vox_w))
            crop_h = int(vox_attrs.get("crop_h", vox_h))
            crop_w = int(vox_attrs.get("crop_w", vox_w))
            self.K = _scale_K_resize_crop(
                calib["K_native"],
                (native_h, native_w),
                (resize_h, resize_w),
                (crop_h, crop_w),
            )
        else:
            self.K = _scale_K(calib["K_native"], calib["native_hw"], (vox_h, vox_w))
        self.depth_values = _inverse_depth_candidates(num_depths, DEPTH_MIN, D_MAX)

        ee_T = ee_T[:self.n_frames]
        T_ee_inv = np.linalg.inv(ee_T)
        self.T_cam_from_world = np.einsum(
            "ij,njk->nik", calib["T_event_from_ee"], T_ee_inv
        ).astype(np.float32)
        self.cam_centers_world = self._camera_centers_world(self.T_cam_from_world)
        if self.pose_channels:
            T_base_from_event = np.linalg.inv(self.T_cam_from_world).astype(np.float32)
            self.pose_values = np.stack(
                [_pose_channels_from_base_event(T) for T in T_base_from_event],
                axis=0,
            ).astype(np.float32)
        else:
            self.pose_values = None

        self.pose_view_ids: dict[int, list[int]] = {}
        if num_views == 1:
            self.src_offsets = []
            if self.recurrent:
                valid = np.arange(self.recurrent_enrollment_range, self.n_frames, dtype=np.int64)
            else:
                valid = np.arange(self.n_frames, dtype=np.int64)
        elif self.pose_view_selection:
            valid = self._make_pose_view_ids(num_views, self.pose_move_threshold)
        else:
            self.src_offsets = self._make_source_offsets(num_views, view_interval)
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
        self._msk = None
        self._tbl = None

    @staticmethod
    def _make_source_offsets(num_views: int, view_interval: int) -> list[int]:
        offsets: list[int] = []
        k = 1
        while len(offsets) < num_views - 1:
            offsets.append(-k * view_interval)
            if len(offsets) < num_views - 1:
                offsets.append(k * view_interval)
            k += 1
        return offsets

    @staticmethod
    def _camera_centers_world(T_cam_from_world: np.ndarray) -> np.ndarray:
        R = T_cam_from_world[:, :3, :3]
        t = T_cam_from_world[:, :3, 3]
        return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t).astype(np.float32)

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
        while 0 <= cursor < self.n_frames and len(neighbours) < per_direction:
            moved = np.linalg.norm(self.cam_centers_world[cursor] - self.cam_centers_world[anchor])
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
        for idx in range(self.n_frames):
            before = self._find_pose_neighbours(idx, -1, per_direction, move_threshold)
            after = self._find_pose_neighbours(idx, 1, per_direction, move_threshold)
            if before is None or after is None:
                continue
            self.pose_view_ids[idx] = [idx] + before + after
            valid.append(idx)
        return np.array(valid, dtype=np.int64)

    def _open(self) -> None:
        import h5py
        if self._vox is None:
            self._vox = h5py.File(self.voxels_path, "r")["voxels"]
        if self._dep is None:
            self._dep = h5py.File(self.depth_path, "r")["depth"]
        if self.has_mask and self._msk is None:
            self._msk = h5py.File(self.mask_path, "r")["mask"]
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

    def _apply_source_view_dropout(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.shape[0] <= 1:
            return imgs
        p = self.aug.source_view_dropout_prob
        drop = torch.rand(imgs.shape[0] - 1) < p
        if len(drop) > 1 and drop.all():
            drop[torch.randint(0, len(drop), (1,)).item()] = False
        imgs[1:][drop] = 0.0
        return imgs

    def _apply_pose_noise(self, cam_mats: torch.Tensor) -> torch.Tensor:
        trans_std = self.aug.pose_translation_std
        rot_std = np.deg2rad(self.aug.pose_rotation_std_deg)
        for v in range(1, cam_mats.shape[0]):
            delta = torch.eye(4, dtype=cam_mats.dtype)
            delta[:3, :3] = self._axis_angle_to_matrix(
                torch.randn(3, dtype=cam_mats.dtype) * rot_std
            )
            delta[:3, 3] = torch.randn(3, dtype=cam_mats.dtype) * trans_std
            cam_mats[v] = delta @ cam_mats[v]
        return cam_mats

    def _apply_event_noise_per_view(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs[:, :NUM_BINS] = imgs[:, :NUM_BINS] + (
            torch.randn_like(imgs[:, :NUM_BINS]) * self.aug.event_noise_std
        )
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
                imgs[v, :, y0:y0 + rect_h, x0:x0 + rect_w] = 0.0
        return imgs

    def _apply_augmentations(
        self,
        imgs: torch.Tensor,
        cam_mats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.aug.enabled:
            return imgs, cam_mats
        imgs = imgs.clone()
        cam_mats = cam_mats.clone()
        if self.aug.source_view_dropout:
            imgs = self._apply_source_view_dropout(imgs)
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
        if self.pose_view_selection:
            view_ids = self.pose_view_ids[idx]
        elif self.recurrent:
            view_ids = list(range(idx - self.recurrent_enrollment_range, idx + 1))
        else:
            view_ids = [idx] + [idx + o for o in self.src_offsets]

        imgs = torch.stack([self._load_input(i) for i in view_ids], dim=0)
        _, _, h, w = imgs.shape

        dep = self._dep[idx].astype(np.float32)
        dep = np.minimum(dep, D_MAX)
        valid = (dep > 0).astype(np.float32)
        if self.has_mask:
            valid *= self._msk[idx].astype(np.float32)

        dep_t = torch.from_numpy(dep).unsqueeze(0)
        msk_t = torch.from_numpy(valid).unsqueeze(0)
        if dep_t.shape[-2] != h or dep_t.shape[-1] != w:
            dep_t = F.interpolate(dep_t.unsqueeze(0), (h, w), mode="nearest").squeeze(0)
            msk_t = F.interpolate(msk_t.unsqueeze(0), (h, w), mode="nearest").squeeze(0)

        if self.fill_invalid:
            target_view = -1 if self.recurrent else 0
            tbl_m = imgs[target_view, NUM_BINS:NUM_BINS + 1] * (D_MAX - DEPTH_MIN) + DEPTH_MIN
            dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
            msk_t = torch.ones_like(msk_t)

        cam_mats = torch.from_numpy(np.stack([self.T_cam_from_world[i] for i in view_ids]))
        imgs, cam_mats = self._apply_augmentations(imgs, cam_mats)
        return {
            "imgs": imgs,
            "cam_mats": cam_mats,
            "K": torch.from_numpy(self.K),
            "depth_values": torch.from_numpy(self.depth_values),
            "dep_t": dep_t,
            "mask_t": msk_t,
            "ref_idx": torch.tensor(idx, dtype=torch.long),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
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

class SharedFeatureCNN(nn.Module):
    """Shared encoder applied to target and source frames."""

    def __init__(self, in_ch: int, base: int, feature_ch: int, output_stride: int = 4):
        super().__init__()
        if output_stride not in (4, 8):
            raise ValueError(f"Unsupported SharedFeatureCNN output stride: {output_stride}")

        layers = [
            nn.Conv2d(in_ch, base, 5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, feature_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU(inplace=True),
        ]
        if output_stride == 8:
            layers.extend([
                nn.Conv2d(feature_ch, feature_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature_ch),
                nn.ReLU(inplace=True),
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _replace_first_conv(module: nn.Module, in_ch: int) -> bool:
    """
    Replace the first Conv2d found in a torchvision feature stem so non-RGB
    event/table inputs can use the same backbone topology.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(
                module,
                name,
                nn.Conv2d(
                    in_ch,
                    child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=1,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                ),
            )
            return True
        if _replace_first_conv(child, in_ch):
            return True
    return False


class TorchvisionFeatureEncoder(nn.Module):
    """ResNet/EfficientNet feature encoder projected to the cost-volume width."""

    def __init__(self, name: str, in_ch: int, feature_ch: int):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, resnet18, resnet34, resnet50
        except ImportError as exc:
            raise ImportError(
                f"--feature_encoder {name} requires torchvision to be installed"
            ) from exc

        if name == "resnet18":
            backbone = resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            out_ch = 128
        elif name == "resnet18_h4":
            backbone = resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            backbone.layer2[0].conv1.stride = (1, 1)
            backbone.layer2[0].downsample[0].stride = (1, 1)
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            out_ch = 128
        elif name == "resnet34":
            backbone = resnet34(weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            out_ch = 128
        elif name == "resnet34_h4":
            backbone = resnet34(weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            backbone.layer2[0].conv1.stride = (1, 1)
            backbone.layer2[0].downsample[0].stride = (1, 1)
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            out_ch = 128
        elif name == "resnet50":
            backbone = resnet50(weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.encoder = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            out_ch = 512
        elif name == "efficientnet_b0":
            backbone = efficientnet_b0(weights=None)
            if not _replace_first_conv(backbone.features[0], in_ch):
                raise RuntimeError("Could not replace EfficientNet-B0 input convolution")
            # Stages 0..3 yield stride-8 features; deeper stages make the
            # cost volume very coarse for this depth-regression task.
            self.encoder = nn.Sequential(*list(backbone.features.children())[:4])
            out_ch = 40
        else:
            raise ValueError(f"Unsupported feature encoder: {name}")

        self.project = nn.Sequential(
            nn.Conv2d(out_ch, feature_ch, 1, bias=False),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.encoder(x))


def make_feature_encoder(name: str, in_ch: int, base: int, feature_ch: int) -> nn.Module:
    if name == "cnn":
        return SharedFeatureCNN(in_ch, base, feature_ch)
    if name == "cnn_8":
        return SharedFeatureCNN(in_ch, base, feature_ch, output_stride=8)
    return TorchvisionFeatureEncoder(name, in_ch, feature_ch)


class CostVolumeCNN(nn.Module):
    """Final CNN that regularises the depth cost volume."""

    def __init__(self, in_ch: int, base: int):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_ch, base),
            block(base, base * 2),
            block(base * 2, base * 2),
            block(base * 2, base),
            nn.Conv3d(base, 1, 3, padding=1),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.net(volume).squeeze(1)  # (B, D, H, W)


class SingleViewFallbackHead(nn.Module):
    """Small target-only decoder and fusion gate for the multi-view estimate."""

    def __init__(self, feature_ch: int, base: int):
        super().__init__()
        hidden = max(base, 16)
        fusion_ch = max(base // 2, 8)
        self.low = nn.Sequential(
            nn.Conv2d(feature_ch, hidden * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.high = nn.Sequential(
            nn.Conv2d(hidden, fusion_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(fusion_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Conv2d(fusion_ch + 2, fusion_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ref_feat: torch.Tensor,
        depth_mv: torch.Tensor,
        out_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.low(ref_feat)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.mid(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        fusion_feat = self.high(x)
        depth_single = self.depth_head(fusion_feat)
        alpha = self.fusion_head(torch.cat([fusion_feat, depth_mv, depth_single], dim=1))
        final = alpha * depth_mv + (1.0 - alpha) * depth_single
        return final.clamp(0.0, 1.0), depth_single, alpha


class MultiViewDepthNet(nn.Module):
    """Shared feature CNN with coarse-to-fine warped cost volumes."""

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        feature_ch: int | None = None,
        cost_base: int | None = None,
        fine_depths: int = 5,
        fine_window: float = 0.08,
        fine_offset_radius: float = 2.0,
        learned_fine_window: bool = False,
        masked_warp_aggregation: bool = False,
        cost_volume_ref_features: bool = False,
        single_view_fallback: bool = False,
        feature_encoder: str = "cnn",
        correlation_groups: int = 0,
        reference_channels: int = 0,
        coarse_cost_channels: int = 0,
        fine_cost_channels: int = 0,
        refiner_channels: int = 0,
        hourglass_levels: int = 2,
        coarse_hourglass_levels: int = 0,
        fine_hourglass_levels: int = 0,
        learned_view_weighting: bool = False,
        two_mode_fine_candidates: bool = False,
        fine_supervision: bool = False,
        fine_loss_weight: float = 0.3,
        variance_channels: int = 0,
        convex_upsampling: bool = False,
        fullres_geometry: bool = False,
        fullres_depths: int = 3,
        fullres_window: float = 0.01,
        fpn_dropout: float = 0.0,
        reference_dropout: float = 0.0,
        hourglass_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        cost_volume_type: str = "correlation",
        middle_depths: int = 8,
        middle_window: float = 0.12,
        middle_cost_channels: int = 0,
        middle_hourglass_levels: int = 0,
        middle_feature_channels: int = 0,
        fine_feature_channels: int = 0,
        middle_supervision: bool = False,
        middle_loss_weight: float = 0.3,
        fullres_fine_volume: bool = False,
        h4_coarse_volume: bool = False,
        decoder_type: str = "auto",
    ):
        super().__init__()
        del (
            correlation_groups,
            reference_channels,
            coarse_cost_channels,
            fine_cost_channels,
            refiner_channels,
            hourglass_levels,
            coarse_hourglass_levels,
            fine_hourglass_levels,
            learned_view_weighting,
            two_mode_fine_candidates,
            fine_supervision,
            fine_loss_weight,
            variance_channels,
            convex_upsampling,
            fullres_geometry,
            fullres_depths,
            fullres_window,
            fpn_dropout,
            reference_dropout,
            hourglass_dropout,
            drop_path_rate,
            cost_volume_type,
            middle_depths,
            middle_window,
            middle_cost_channels,
            middle_hourglass_levels,
            middle_feature_channels,
            fine_feature_channels,
            middle_supervision,
            middle_loss_weight,
            fullres_fine_volume,
            h4_coarse_volume,
            decoder_type,
        )
        if fine_depths < 3:
            raise ValueError(f"fine_depths must be >= 3, got {fine_depths}")
        feature_ch = feature_ch or base * 4
        cost_base = cost_base or max(base // 2, 8)
        self.fine_depths = fine_depths
        self.fine_window = float(fine_window)
        self.fine_offset_radius = float(fine_offset_radius)
        self.learned_fine_window = learned_fine_window
        self.masked_warp_aggregation = masked_warp_aggregation
        self.cost_volume_ref_features = cost_volume_ref_features
        self.single_view_fallback = single_view_fallback
        self.feature_encoder = feature_encoder
        self.feature = make_feature_encoder(feature_encoder, in_ch, base, feature_ch)
        cost_volume_ch = feature_ch * 2 + 1 if cost_volume_ref_features else feature_ch
        self.coarse_cost_cnn = CostVolumeCNN(cost_volume_ch, cost_base)
        self.fine_cost_cnn = CostVolumeCNN(cost_volume_ch, cost_base)
        confidence_ch = feature_ch + 7
        self.confidence_head = nn.Sequential(
            nn.Conv2d(confidence_ch, max(cost_base * 2, 32), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(cost_base * 2, 32), max(cost_base, 16), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(cost_base, 16), 1, 1),
        )
        if single_view_fallback:
            self.single_view_head = SingleViewFallbackHead(feature_ch, base)
        if learned_fine_window:
            self.window_head = nn.Sequential(
                nn.Conv2d(feature_ch + 1, max(cost_base, 8), 3, padding=1, bias=False),
                nn.BatchNorm2d(max(cost_base, 8)),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(cost_base, 8), 1, 3, padding=1),
            )

    def _build_cost_volume(
        self,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K_feat: torch.Tensor,
        depth_values: torch.Tensor,
        return_valid_ratio: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, V, Fch, Hf, Wf = feats.shape
        D = depth_values.shape[0] if depth_values.dim() == 1 else depth_values.shape[1]
        ref_feat = feats[:, 0]
        ref_T = cam_mats[:, 0]
        volume_sum = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
        volume_sq_sum = volume_sum ** 2
        if self.masked_warp_aggregation:
            valid_sum = torch.ones(
                (B, 1, D, Hf, Wf),
                device=feats.device,
                dtype=feats.dtype,
            )

        for v in range(1, V):
            if self.masked_warp_aggregation:
                warped, valid_mask = homo_warp_features(
                    feats[:, v],
                    cam_mats[:, v],
                    ref_T,
                    K_feat,
                    depth_values,
                    return_valid_mask=True,
                )
                volume_sum = volume_sum + warped * valid_mask
                volume_sq_sum = volume_sq_sum + (warped ** 2) * valid_mask
                valid_sum = valid_sum + valid_mask
            else:
                warped = homo_warp_features(
                    feats[:, v],
                    cam_mats[:, v],
                    ref_T,
                    K_feat,
                    depth_values,
                )
                volume_sum = volume_sum + warped
                volume_sq_sum = volume_sq_sum + warped ** 2

        if self.masked_warp_aggregation:
            denom = valid_sum.clamp_min(1.0)
            mean = volume_sum / denom
            variance = volume_sq_sum / denom - mean ** 2
            valid_ratio = valid_sum / float(V)
        else:
            variance = volume_sq_sum / V - (volume_sum / V) ** 2
            valid_ratio = torch.ones(
                (B, 1, D, Hf, Wf),
                device=feats.device,
                dtype=feats.dtype,
            )

        if self.cost_volume_ref_features:
            ref_volume = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
            volume = torch.cat([variance, ref_volume, valid_ratio], dim=1)
        else:
            volume = variance
        if return_valid_ratio:
            return volume, valid_ratio
        return volume

    @staticmethod
    def _regress_depth(prob: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
        if depth_values.dim() == 1:
            dv = depth_values[None, :, None, None]
        elif depth_values.dim() == 2:
            dv = depth_values[:, :, None, None]
        elif depth_values.dim() == 4:
            dv = depth_values
        else:
            raise ValueError(f"Unsupported depth_values shape: {tuple(depth_values.shape)}")
        return torch.sum(prob * dv.to(device=prob.device, dtype=prob.dtype), dim=1, keepdim=True)

    @staticmethod
    def _depth_uncertainty(
        prob: torch.Tensor,
        depth_values: torch.Tensor,
        mean_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-pixel depth standard deviation in metres."""
        if depth_values.dim() == 1:
            dv = depth_values[None, :, None, None]
        elif depth_values.dim() == 2:
            dv = depth_values[:, :, None, None]
        elif depth_values.dim() == 4:
            dv = depth_values
        else:
            raise ValueError(f"Unsupported depth_values shape: {tuple(depth_values.shape)}")
        dv = dv.to(device=prob.device, dtype=prob.dtype)
        variance = torch.sum(prob * (dv - mean_depth).square(), dim=1, keepdim=True)
        return variance.clamp_min(0.0).sqrt()

    def _fine_depth_values(
        self,
        coarse_depth: torch.Tensor,
        ref_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, _, Hf, Wf = coarse_depth.shape
        dtype = coarse_depth.dtype
        device = coarse_depth.device
        offsets = torch.linspace(
            -self.fine_offset_radius,
            self.fine_offset_radius,
            self.fine_depths,
            device=device,
            dtype=dtype,
        )
        if self.learned_fine_window:
            window01 = torch.sigmoid(self.window_head(torch.cat([ref_feat, coarse_depth], dim=1)))
            sigma = self.fine_window * (0.25 + 0.75 * window01)
        else:
            sigma = torch.full_like(coarse_depth, self.fine_window)
        fine = coarse_depth + sigma * offsets.view(1, self.fine_depths, 1, 1)
        return fine.clamp(DEPTH_MIN, D_MAX).view(B, self.fine_depths, Hf, Wf)

    def forward(
        self,
        imgs: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        return_coarse: bool = False,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        B, V, C, H, W = imgs.shape

        feats = self.feature(imgs.reshape(B * V, C, H, W))
        _, Fch, Hf, Wf = feats.shape
        feats = feats.view(B, V, Fch, Hf, Wf)

        K_feat = K.clone()
        K_feat[:, 0, :] *= Wf / W
        K_feat[:, 1, :] *= Hf / H

        coarse_volume = self._build_cost_volume(feats, cam_mats, K_feat, depth_values)
        coarse_cost = self.coarse_cost_cnn(coarse_volume)
        coarse_prob = F.softmax(-coarse_cost.float(), dim=1).to(coarse_cost.dtype)
        coarse_depth = self._regress_depth(coarse_prob, depth_values)

        fine_values = self._fine_depth_values(coarse_depth, feats[:, 0])
        fine_volume, fine_valid_ratio = self._build_cost_volume(
            feats,
            cam_mats,
            K_feat,
            fine_values,
            return_valid_ratio=True,
        )
        fine_cost = self.fine_cost_cnn(fine_volume)
        fine_prob = F.softmax(-fine_cost.float(), dim=1).to(fine_cost.dtype)
        depth_low = self._regress_depth(fine_prob, fine_values)

        depth_mv_m = F.interpolate(depth_low, size=(H, W), mode="bilinear", align_corners=False)
        depth_mv_norm = ((depth_mv_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
        if self.single_view_fallback:
            final_depth_norm, depth_single_norm, fusion_alpha = self.single_view_head(
                feats[:, 0],
                depth_mv_norm,
                (H, W),
            )
            depth_single_low = F.interpolate(
                depth_single_norm,
                size=(Hf, Wf),
                mode="bilinear",
                align_corners=False,
            ) * (D_MAX - DEPTH_MIN) + DEPTH_MIN
            alpha_low = F.interpolate(
                fusion_alpha,
                size=(Hf, Wf),
                mode="bilinear",
                align_corners=False,
            )
        else:
            final_depth_norm = depth_mv_norm
            depth_single_low = depth_low
            alpha_low = torch.ones_like(depth_low)

        confidence_m = None
        if return_uncertainty:
            eps = 1e-8
            posterior_std = self._depth_uncertainty(fine_prob, fine_values, depth_low)
            entropy = -(fine_prob * torch.log(fine_prob + eps)).sum(dim=1, keepdim=True)
            max_prob = fine_prob.max(dim=1, keepdim=True).values
            coarse_fine_diff = torch.abs(depth_low - coarse_depth)
            valid_ratio = fine_valid_ratio.mean(dim=2)
            single_mv_diff = torch.abs(depth_single_low - depth_low)
            confidence_features = torch.cat(
                [
                    feats[:, 0],
                    posterior_std,
                    entropy,
                    max_prob,
                    coarse_fine_diff,
                    valid_ratio,
                    single_mv_diff,
                    alpha_low,
                ],
                dim=1,
            )
            confidence_low = torch.sigmoid(self.confidence_head(confidence_features))
            confidence_m = F.interpolate(
                confidence_low,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            confidence_m = confidence_m.clamp(0.0, 1.0)

        if not return_coarse:
            if return_uncertainty:
                return final_depth_norm, confidence_m
            return final_depth_norm

        coarse_depth_m = F.interpolate(coarse_depth, size=(H, W), mode="bilinear", align_corners=False)
        coarse_depth_norm = ((coarse_depth_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
        if return_uncertainty:
            return coarse_depth_norm, final_depth_norm, confidence_m
        return coarse_depth_norm, final_depth_norm


# ---------------------------------------------------------------------------
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
    lambda_smooth: float = 0.01,
    lambda_mean: float = 0.1,
    lambda_normal: float = 0.1,
    l1_loss_only: bool = False,
    coarse_supervision: bool = False,
    coarse_loss_weight: float = 0.3,
    uncertainty: bool = False,
    lambda_confidence: float = 0.1,
    confidence_abs_tolerance: float = 0.01,
    confidence_rel_tolerance: float = 0.01,
    fine_supervision: bool = False,
    fine_loss_weight: float = 0.3,
    middle_supervision: bool = False,
    middle_loss_weight: float = 0.3,
    lambda_worst_percent: float = 0.0,
    worst_percent: float = 0.10,
    ema_model=None,
    freeze_batch_norm: bool = False,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
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

            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            model_out = model(
                imgs,
                cam_mats,
                K,
                depth_values,
                return_coarse=coarse_supervision,
                return_uncertainty=uncertainty,
            )
            pred_unc = None
            if coarse_supervision:
                if uncertainty:
                    coarse_pred, pred, pred_unc = model_out
                else:
                    coarse_pred, pred = model_out
            else:
                if uncertainty:
                    pred, pred_unc = model_out
                else:
                    pred = model_out

            pred_loss = pred
            coarse_pred_loss = None
            if coarse_supervision:
                coarse_pred_loss = coarse_pred
            if l1_loss_only:
                loss_final = _l1_metres(pred_loss, dep_t, mask_t)
            else:
                loss_final, _ = compute_loss(
                    pred_loss,
                    dep_norm,
                    mask_t,
                    imgs[:, 0, :NUM_BINS],
                    K=K[0],
                    lambda_grad=lambda_grad,
                    lambda_smooth=lambda_smooth,
                    lambda_mean=lambda_mean,
                    lambda_normal=lambda_normal,
                )
            loss = loss_final
            if lambda_worst_percent > 0:
                loss_worst = _worst_percent_l1_metres(
                    pred_loss,
                    dep_t,
                    mask_t,
                    percent=worst_percent,
                )
                loss = loss + lambda_worst_percent * loss_worst
            if fine_supervision:
                fine_pred = getattr(model, "aux_fine_pred", None)
                if fine_pred is None:
                    raise RuntimeError(
                        "--fine_supervision requires a model exposing aux_fine_pred"
                    )
                fine_hw = fine_pred.shape[-2:]
                fine_depth = F.interpolate(dep_t, size=fine_hw, mode="nearest")
                fine_mask = F.interpolate(mask_t, size=fine_hw, mode="nearest")
                loss_fine = _l1_metres(fine_pred, fine_depth, fine_mask)
                loss = loss + fine_loss_weight * loss_fine
            if middle_supervision:
                middle_pred = getattr(model, "aux_middle_pred", None)
                if middle_pred is None:
                    raise RuntimeError(
                        "--middle_supervision requires a three-stage model "
                        "exposing aux_middle_pred"
                    )
                middle_hw = middle_pred.shape[-2:]
                middle_depth = F.interpolate(dep_t, size=middle_hw, mode="nearest")
                middle_mask = F.interpolate(mask_t, size=middle_hw, mode="nearest")
                loss_middle = _l1_metres(
                    middle_pred, middle_depth, middle_mask
                )
                loss = loss + middle_loss_weight * loss_middle
            if coarse_supervision:
                if l1_loss_only:
                    loss_coarse = _l1_metres(coarse_pred_loss, dep_t, mask_t)
                else:
                    loss_coarse, _ = compute_loss(
                        coarse_pred_loss,
                        dep_norm,
                        mask_t,
                        imgs[:, 0, :NUM_BINS],
                        K=K[0],
                        lambda_grad=lambda_grad,
                        lambda_smooth=lambda_smooth,
                        lambda_mean=lambda_mean,
                        lambda_normal=lambda_normal,
                    )
                loss = loss + coarse_loss_weight * loss_coarse
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
                total_l1 += float(_l1_metres(pred_loss, dep_t, mask_t))
                pred_m = pred_loss * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                valid_errors = torch.abs(pred_m - dep_t)[mask_t > 0.5]
                if valid_errors.numel() > 0:
                    total_p95 += float(torch.quantile(valid_errors.float(), 0.95))
                total_worst10_l1 += float(_worst_percent_l1_metres(pred_loss, dep_t, mask_t))
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


def _set_axes_equal(ax) -> None:
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    ranges = [abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])]
    centers = [
        (xlim[0] + xlim[1]) * 0.5,
        (ylim[0] + ylim[1]) * 0.5,
        (zlim[0] + zlim[1]) * 0.5,
    ]
    radius = max(ranges) * 0.5
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


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
        _set_axes_equal(ax3d)
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

def _is_sequence(p: Path, fix_transform: bool = False) -> bool:
    return (
        (p / "events" / "voxels_cam0.h5").exists()
        and (p / "hdf5" / "depth_in_event_frame.h5").exists()
        and (p / "hdf5" / "poses.h5").exists()
        and (p / "hdf5" / _table_plane_filename(fix_transform)).exists()
    )


def _find_sequences(root: Path, fix_transform: bool = False) -> list[Path]:
    """Return a sequence at root, or all valid immediate child sequences."""
    if _is_sequence(root, fix_transform=fix_transform):
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and _is_sequence(d, fix_transform=fix_transform)
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
    parser.add_argument("--freeze_batch_norm", action="store_true",
                        help="Keep BatchNorm layers in eval mode during training so running stats do not drift")
    parser.add_argument("--lambda_grad", type=float, default=0.5,
                        help="Weight for multi-scale gradient loss term.")
    parser.add_argument("--lambda_smooth", type=float, default=0.01,
                        help="Weight for event-aware smoothness loss term.")
    parser.add_argument("--lambda_mean", type=float, default=0.1,
                        help="Weight for mean-depth consistency loss term.")
    parser.add_argument("--lambda_normal", type=float, default=0.1,
                        help="Weight for surface-normal loss term.")
    parser.add_argument("--l1_loss_only", action="store_true",
                        help="Train directly with masked metric L1 in metres, ignoring auxiliary loss terms.")
    parser.add_argument("--lambda_worst_percent", type=float, default=0.0,
                        help="Weight for metric L1 over the worst valid prediction pixels")
    parser.add_argument("--worst_percent", type=float, default=0.10,
                        help="Fraction of valid pixels used by --lambda_worst_percent")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--feature_channels", type=int, default=0,
                        help="Shared CNN output channels; 0 means base_channels*4")
    parser.add_argument("--feature_encoder",
                        choices=("cnn", "cnn_8", "casmvsnet_fpn", "deep_fpn", "resnet18", "resnet18_h4",
                                 "resnet34", "resnet34_h4", "resnet50", "efficientnet_b0"),
                        default="cnn",
                        help="Shared per-view feature encoder; casmvsnet_fpn is available in modern_multiview.py")
    parser.add_argument(
        "--decoder_type",
        choices=("auto", "two_stage_fpn", "three_stage_fpn"),
        default="auto",
        help=(
            "Modern MVS decoder/cascade layout: auto preserves existing "
            "behavior; two_stage_fpn uses H/4 global then H/2 local geometry "
            "with casmvsnet_fpn/deep_fpn; three_stage_fpn explicitly selects "
            "the existing three-stage FPN decoder"
        ),
    )
    parser.add_argument("--cost_channels", type=int, default=0,
                        help="3D cost CNN base width; 0 means max(base_channels//2, 8)")
    parser.add_argument("--correlation_groups", type=int, default=0,
                        help="Modern MVS only: group-correlation channels; for casmvsnet_fpn "
                             "this is the per-stage maximum; 0 selects automatically")
    parser.add_argument("--cost_volume_type", choices=("correlation", "variance"),
                        default="correlation",
                        help="Modern MVS only: primary multi-view cost-volume aggregation")
    parser.add_argument("--reference_channels", type=int, default=0,
                        help="Modern MVS only: compressed reference channels added to each cost volume")
    parser.add_argument("--coarse_cost_channels", type=int, default=0,
                        help="Modern MVS only: coarse 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--fine_cost_channels", type=int, default=0,
                        help="Modern MVS only: fine 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--middle_cost_channels", type=int, default=0,
                        help="CasMVSNet FPN only: H/4 3D hourglass base width; 0 uses --cost_channels")
    parser.add_argument("--refiner_channels", type=int, default=0,
                        help="Modern MVS only: full-resolution 2D refiner width; 0 selects automatically")
    parser.add_argument("--hourglass_levels", type=int, default=2,
                        help="Modern MVS only: fallback number of 3D hourglass levels for both stages")
    parser.add_argument("--coarse_hourglass_levels", type=int, default=0,
                        help="Modern MVS only: coarse hourglass levels; 0 uses --hourglass_levels")
    parser.add_argument("--fine_hourglass_levels", type=int, default=0,
                        help="Modern MVS only: fine hourglass levels; 0 uses --hourglass_levels")
    parser.add_argument("--middle_hourglass_levels", type=int, default=0,
                        help="CasMVSNet FPN only: H/4 hourglass levels; 0 uses --hourglass_levels")
    parser.add_argument("--learned_view_weighting", action="store_true",
                        help="Modern MVS only: learn per-view, per-depth reliability before aggregation")
    parser.add_argument("--two_mode_fine_candidates", action="store_true",
                        help="Modern MVS only: sample fine hypotheses around two separated coarse modes")
    parser.add_argument("--fine_supervision", action="store_true",
                        help="Modern MVS only: directly supervise the H/2 fine-stage depth")
    parser.add_argument("--fine_loss_weight", type=float, default=0.3,
                        help="Weight of direct fine-stage metric L1 supervision")
    parser.add_argument("--middle_supervision", action="store_true",
                        help="Three-stage FPN only: directly supervise middle-stage depth")
    parser.add_argument("--middle_loss_weight", type=float, default=0.3,
                        help="Weight of direct H/4 middle-stage metric L1 supervision")
    parser.add_argument("--middle_feature_channels", type=int, default=0,
                        help="Three-stage FPN only: middle-stage output channels; "
                             "0 uses feature_channels/2")
    parser.add_argument("--fine_feature_channels", type=int, default=0,
                        help="Three-stage FPN only: final-stage output channels; "
                             "0 uses feature_channels/4")
    parser.add_argument("--fullres_fine_volume", action="store_true",
                        help="Three-stage FPN only: construct the final cost volume at input "
                             "resolution and bypass depth upsampling and the 2-D refiner")
    parser.add_argument("--h4_coarse_volume", action="store_true",
                        help="Full-resolution FPN mode only: construct the global coarse "
                             "cost volume at H/4 instead of H/8")
    parser.add_argument("--variance_channels", type=int, default=0,
                        help="Modern MVS only: compressed channels for cross-view correlation variance")
    parser.add_argument("--convex_upsampling", action="store_true",
                        help="Modern MVS only: use learned RAFT-style convex H/2-to-full upsampling")
    parser.add_argument("--fullres_geometry", action="store_true",
                        help="Modern MVS only: run a tiny local full-resolution geometry stage")
    parser.add_argument("--fullres_depths", type=int, default=3,
                        help="Modern MVS only: odd number of full-resolution local depth hypotheses")
    parser.add_argument("--fullres_window", type=float, default=0.01,
                        help="Modern MVS only: full-resolution local search half-window in metres")
    parser.add_argument("--fpn_dropout", type=float, default=0.0,
                        help="Modern MVS only: Dropout2d probability on FPN outputs")
    parser.add_argument("--reference_dropout", type=float, default=0.0,
                        help="Modern MVS only: Dropout2d probability on reference features")
    parser.add_argument("--hourglass_dropout", type=float, default=0.0,
                        help="Modern MVS only: Dropout3d probability at hourglass bottlenecks")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                        help="Modern MVS only: stochastic-depth rate on FPN fusion")
    parser.add_argument("--model_scale", type=float, default=1.0,
                        help="Width multiplier for base_channels and derived feature/cost channels; explicit feature/cost overrides still win")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--view_interval", type=int, default=5)
    parser.add_argument("--pose_view_selection", action="store_true",
                        help="Select balanced before/after source views by camera motion instead of fixed frame offsets")
    parser.add_argument("--pose_move_threshold", type=float, default=0.01,
                        help="Minimum camera-center translation in metres between consecutive selected pose views")
    parser.add_argument("--num_depths", type=int, default=32,
                        help="Number of coarse inverse-depth planes between DEPTH_MIN and D_MAX")
    parser.add_argument("--fine_depths", type=int, default=5,
                        help="Number of fine per-pixel depth hypotheses around the coarse estimate")
    parser.add_argument("--middle_depths", type=int, default=8,
                        help="CasMVSNet FPN only: H/4 local depth hypotheses")
    parser.add_argument("--middle_window", type=float, default=0.12,
                        help="CasMVSNet FPN only: H/4 local search half-window in metres")
    parser.add_argument("--fine_window", type=float, default=0.08,
                        help="Fine-stage sigma/window in metres before multiplying by offsets")
    parser.add_argument("--fine_offset_radius", type=float, default=2.0,
                        help="Fine offsets span [-radius, radius]; 2 with 5 planes gives [-2,-1,0,1,2]")
    parser.add_argument("--learned_fine_window", action="store_true",
                        help="Predict a per-pixel fine-stage window from target features and coarse depth")
    parser.add_argument("--masked_warp_aggregation", action="store_true",
                        help="Ignore out-of-image and behind-camera warped source features when aggregating cost-volume variance")
    parser.add_argument("--cost_volume_ref_features", action="store_true",
                        help="Concatenate target/reference features and valid-view ratio over depth with the variance cost volume before the 3D CNN")
    parser.add_argument("--coarse_supervision", action="store_true",
                        help="Train with an auxiliary coarse-depth loss: loss_final + 0.3 * loss_coarse")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Train, return, and log learned per-pixel fusion confidence")
    parser.add_argument("--lambda_confidence", type=float, default=0.1,
                        help="Auxiliary BCE loss weight for learned confidence when --uncertainty is enabled")
    parser.add_argument("--confidence_abs_tolerance", type=float, default=0.01,
                        help="Absolute safe-to-fuse confidence tolerance in metres")
    parser.add_argument("--confidence_rel_tolerance", type=float, default=0.01,
                        help="Relative safe-to-fuse confidence tolerance as a fraction of GT depth")
    parser.add_argument("--single_view_fallback", action="store_true",
                        help="Add a target-only decoder and learn a per-pixel fusion of multi-view and single-view depth")
    parser.add_argument("--out_dir", type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "multiview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mask", action="store_true",
                        help="Ignore spatial mask; dep>0 validity is always applied")
    parser.add_argument("--fill_invalid", action="store_true",
                        help="Fill pixels with no depth measurement using the table-plane prior")
    parser.add_argument("-fix_transform", "--fix_transform", action="store_true",
                        help="Opt-in geometry fix: scale event intrinsics through the stored "
                             "resize + centred crop transform instead of the historical direct "
                             "native-resolution -> tensor-resolution scaling")
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
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/multiview/<name>.")
    parser.add_argument("--debug_views", action="store_true",
                        help="Save target/source event images and camera-pose plots, then exit.")
    parser.add_argument("--debug_samples", type=int, default=4,
                        help="Number of target samples to visualise with --debug_views.")
    parser.add_argument("--debug_out", type=Path,
                        default=_SCRIPT_DIR / "debug" / "multiview",
                        help="Output directory for --debug_views PNGs and pose matrices.")
    args = parser.parse_args()

    if args.pose_view_selection and args.num_views % 2 != 1:
        parser.error("--pose_view_selection requires odd --num_views for balanced before/after sources")
    if args.model_scale <= 0:
        parser.error("--model_scale must be > 0")
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
    if args.lambda_worst_percent < 0:
        parser.error("--lambda_worst_percent must be >= 0")
    if not (0 < args.worst_percent <= 1):
        parser.error("--worst_percent must be in (0, 1]")
    if args.correlation_groups < 0:
        parser.error("--correlation_groups must be >= 0")
    if args.reference_channels < 0:
        parser.error("--reference_channels must be >= 0")
    if any(x < 0 for x in (
        args.coarse_cost_channels, args.middle_cost_channels, args.fine_cost_channels
    )):
        parser.error("stage-specific cost channel counts must be >= 0")
    if args.refiner_channels < 0:
        parser.error("--refiner_channels must be >= 0")
    if args.hourglass_levels < 1:
        parser.error("--hourglass_levels must be >= 1")
    if any(x < 0 for x in (
        args.coarse_hourglass_levels,
        args.middle_hourglass_levels,
        args.fine_hourglass_levels,
    )):
        parser.error("stage-specific hourglass levels must be >= 0")
    if args.middle_depths < 3:
        parser.error("--middle_depths must be >= 3")
    if args.middle_window <= 0:
        parser.error("--middle_window must be > 0")
    if args.fine_loss_weight < 0:
        parser.error("--fine_loss_weight must be >= 0")
    if args.variance_channels < 0:
        parser.error("--variance_channels must be >= 0")
    if args.fullres_depths < 3 or args.fullres_depths % 2 != 1:
        parser.error("--fullres_depths must be an odd integer >= 3")
    if args.fullres_window <= 0:
        parser.error("--fullres_window must be > 0")
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

    calib = _load_event_calibration()

    train_root = args.data_dir / "train"
    eval_root = args.data_dir / "eval"
    train_seqs = _find_sequences(train_root, fix_transform=args.fix_transform)
    val_seqs = _find_sequences(eval_root, fix_transform=args.fix_transform)

    if not train_seqs or not val_seqs:
        missing = []
        if not train_seqs:
            missing.append(str(train_root))
        if not val_seqs:
            missing.append(str(eval_root))
        sys.exit(
            f"[ERROR] No valid sequences found in: {', '.join(missing)}\n"
            "        --data_dir must contain train/ and eval/, each holding one or more\n"
            f"        valid sequences with voxels, depth, poses.h5, and {_table_plane_filename(args.fix_transform)}.\n"
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
        num_depths=args.num_depths,
        use_mask=not args.no_mask,
        fill_invalid=args.fill_invalid,
        fix_transform=args.fix_transform,
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
    print(f"  Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")
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
    print(
        f"  Multi-view: views={args.num_views}, interval={args.view_interval}, "
        f"coarse inverse-depth planes={args.num_depths} [{DEPTH_MIN:.3f}, {D_MAX:.3f}] m"
    )
    if args.pose_view_selection:
        print(
            f"  View selection: pose-based, balanced before/after, "
            f"translation threshold={args.pose_move_threshold:g} m"
        )
    else:
        print("  View selection: fixed frame offsets")
    print(
        f"  Fine stage: planes={args.fine_depths}, window={args.fine_window:.4f} m, "
        f"offset radius={args.fine_offset_radius:g}, "
        f"learned window={args.learned_fine_window}\n"
    )
    print(
        f"  Loss weights: grad={args.lambda_grad:g}, smooth={args.lambda_smooth:g}, "
        f"mean={args.lambda_mean:g}, normal={args.lambda_normal:g}, "
        f"l1_loss_only={args.l1_loss_only}, "
        f"worst={args.lambda_worst_percent:g} "
        f"(top {100.0 * args.worst_percent:g}%), "
        f"confidence={args.lambda_confidence:g} "
        f"(abs_tol={args.confidence_abs_tolerance:g} m, "
        f"rel_tol={args.confidence_rel_tolerance:g}), "
        f"middle={args.middle_loss_weight:g} enabled={args.middle_supervision}, "
        f"fine={args.fine_loss_weight:g} enabled={args.fine_supervision}\n"
    )
    min_lr = args.min_lr if args.min_lr is not None else args.lr * 1e-2
    print(
        f"  Optimizer: {args.optimizer}, lr={args.lr:g}, weight_decay={args.weight_decay:g}, "
        f"scheduler={args.lr_scheduler}, min_lr={min_lr:g}, "
        f"warmup_epochs={args.warmup_epochs}, lr_gamma={args.lr_gamma:g}, "
        f"freeze_batch_norm={args.freeze_batch_norm}\n"
    )
    print(
        "  Train augmentations: "
        f"enabled={train_aug.enabled}, "
        f"source_view_dropout={train_aug.source_view_dropout}, "
        f"pose_noise={train_aug.pose_noise}, "
        f"event_noise_per_view={train_aug.event_noise_per_view}, "
        f"cross_view_event_dropout={train_aug.cross_view_event_dropout}, "
        f"occlusion_view_masking={train_aug.occlusion_view_masking}\n"
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
    base_channels = max(1, int(round(args.base_channels * args.model_scale)))
    feature_channels = (
        args.feature_channels if args.feature_channels > 0 else base_channels * 4
    )
    cost_channels = (
        args.cost_channels if args.cost_channels > 0 else max(base_channels // 2, 8)
    )
    model = MultiViewDepthNet(
        in_ch=in_ch,
        base=base_channels,
        feature_ch=feature_channels,
        cost_base=cost_channels,
        fine_depths=args.fine_depths,
        fine_window=args.fine_window,
        fine_offset_radius=args.fine_offset_radius,
        learned_fine_window=args.learned_fine_window,
        masked_warp_aggregation=args.masked_warp_aggregation,
        cost_volume_ref_features=args.cost_volume_ref_features,
        single_view_fallback=args.single_view_fallback,
        feature_encoder=args.feature_encoder,
        correlation_groups=args.correlation_groups,
        reference_channels=args.reference_channels,
        coarse_cost_channels=args.coarse_cost_channels,
        fine_cost_channels=args.fine_cost_channels,
        refiner_channels=args.refiner_channels,
        hourglass_levels=args.hourglass_levels,
        coarse_hourglass_levels=args.coarse_hourglass_levels,
        fine_hourglass_levels=args.fine_hourglass_levels,
        learned_view_weighting=args.learned_view_weighting,
        two_mode_fine_candidates=args.two_mode_fine_candidates,
        fine_supervision=args.fine_supervision,
        fine_loss_weight=args.fine_loss_weight,
        variance_channels=args.variance_channels,
        convex_upsampling=args.convex_upsampling,
        fullres_geometry=args.fullres_geometry,
        fullres_depths=args.fullres_depths,
        fullres_window=args.fullres_window,
        fpn_dropout=args.fpn_dropout,
        reference_dropout=args.reference_dropout,
        hourglass_dropout=args.hourglass_dropout,
        drop_path_rate=args.drop_path_rate,
        cost_volume_type=args.cost_volume_type,
        middle_depths=args.middle_depths,
        middle_window=args.middle_window,
        middle_cost_channels=args.middle_cost_channels,
        middle_hourglass_levels=args.middle_hourglass_levels,
        middle_feature_channels=args.middle_feature_channels,
        fine_feature_channels=args.fine_feature_channels,
        middle_supervision=args.middle_supervision,
        middle_loss_weight=args.middle_loss_weight,
        fullres_fine_volume=args.fullres_fine_volume,
        h4_coarse_volume=args.h4_coarse_volume,
        decoder_type=args.decoder_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_arch = getattr(model, "architecture_name", model.__class__.__name__)
    print(
        f"{model_arch}  in_ch={in_ch}  model_scale={args.model_scale:g}  "
        f"base={base_channels}  feature={feature_channels}  "
        f"feature_encoder={args.feature_encoder}  cost={cost_channels}  "
        f"coarse_depths={args.num_depths}  middle_depths={args.middle_depths}  "
        f"fine_depths={args.fine_depths}  "
        f"masked_warp_aggregation={args.masked_warp_aggregation}  "
        f"cost_volume_type={args.cost_volume_type}  "
        f"cost_volume_ref_features={args.cost_volume_ref_features}  "
        f"coarse_supervision={args.coarse_supervision}  "
        f"uncertainty={args.uncertainty}  "
        f"single_view_fallback={args.single_view_fallback}  "
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
    tb_log_dir = tensorboard_run_dir("multiview", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")

    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val = VizLogger(writer, n_samples=4, tag="viz/val", show_mask=not args.no_mask)
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
            lambda_smooth=args.lambda_smooth,
            lambda_mean=args.lambda_mean,
            lambda_normal=args.lambda_normal,
            l1_loss_only=args.l1_loss_only,
            coarse_supervision=args.coarse_supervision,
            uncertainty=args.uncertainty,
            lambda_confidence=args.lambda_confidence,
            confidence_abs_tolerance=args.confidence_abs_tolerance,
            confidence_rel_tolerance=args.confidence_rel_tolerance,
            fine_supervision=args.fine_supervision,
            fine_loss_weight=args.fine_loss_weight,
            middle_supervision=args.middle_supervision,
            middle_loss_weight=args.middle_loss_weight,
            lambda_worst_percent=args.lambda_worst_percent,
            worst_percent=args.worst_percent,
            ema_model=ema,
            freeze_batch_norm=args.freeze_batch_norm,
        )
        validation_model = ema.model if ema is not None else model
        va_loss, va_l1, va_p95, va_worst10 = run_epoch(
            validation_model, val_loader, None, device,
            viz=viz_val,
            activity_diag=activity_val,
            error_diag=error_val,
            uncertainty_diag=uncertainty_val,
            lambda_grad=args.lambda_grad,
            lambda_smooth=args.lambda_smooth,
            lambda_mean=args.lambda_mean,
            lambda_normal=args.lambda_normal,
            l1_loss_only=args.l1_loss_only,
            coarse_supervision=args.coarse_supervision,
            uncertainty=args.uncertainty,
            lambda_confidence=args.lambda_confidence,
            confidence_abs_tolerance=args.confidence_abs_tolerance,
            confidence_rel_tolerance=args.confidence_rel_tolerance,
            fine_supervision=args.fine_supervision,
            fine_loss_weight=args.fine_loss_weight,
            middle_supervision=args.middle_supervision,
            middle_loss_weight=args.middle_loss_weight,
            lambda_worst_percent=args.lambda_worst_percent,
            worst_percent=args.worst_percent,
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
            "feature_encoder": args.feature_encoder,
            "decoder_type": args.decoder_type,
            "cost_channels": cost_channels,
            "correlation_groups": args.correlation_groups,
            "cost_volume_type": args.cost_volume_type,
            "reference_channels": args.reference_channels,
            "coarse_cost_channels": args.coarse_cost_channels,
            "middle_cost_channels": args.middle_cost_channels,
            "fine_cost_channels": args.fine_cost_channels,
            "refiner_channels": args.refiner_channels,
            "hourglass_levels": args.hourglass_levels,
            "coarse_hourglass_levels": args.coarse_hourglass_levels,
            "middle_hourglass_levels": args.middle_hourglass_levels,
            "fine_hourglass_levels": args.fine_hourglass_levels,
            "learned_view_weighting": args.learned_view_weighting,
            "two_mode_fine_candidates": args.two_mode_fine_candidates,
            "fine_supervision": args.fine_supervision,
            "fine_loss_weight": args.fine_loss_weight,
            "middle_supervision": args.middle_supervision,
            "middle_loss_weight": args.middle_loss_weight,
            "middle_feature_channels": args.middle_feature_channels,
            "fine_feature_channels": args.fine_feature_channels,
            "fullres_fine_volume": args.fullres_fine_volume,
            "h4_coarse_volume": args.h4_coarse_volume,
            "variance_channels": args.variance_channels,
            "convex_upsampling": args.convex_upsampling,
            "fullres_geometry": args.fullres_geometry,
            "fullres_depths": args.fullres_depths,
            "fullres_window": args.fullres_window,
            "fpn_dropout": args.fpn_dropout,
            "reference_dropout": args.reference_dropout,
            "hourglass_dropout": args.hourglass_dropout,
            "drop_path_rate": args.drop_path_rate,
            "ema_decay": args.ema_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "freeze_batch_norm": args.freeze_batch_norm,
            "model_scale": args.model_scale,
            "in_ch": in_ch,
            "num_views": args.num_views,
            "view_interval": args.view_interval,
            "pose_view_selection": args.pose_view_selection,
            "pose_move_threshold": args.pose_move_threshold,
            "num_depths": args.num_depths,
            "middle_depths": args.middle_depths,
            "middle_window": args.middle_window,
            "fine_depths": args.fine_depths,
            "fine_window": args.fine_window,
            "fine_offset_radius": args.fine_offset_radius,
            "learned_fine_window": args.learned_fine_window,
            "masked_warp_aggregation": args.masked_warp_aggregation,
            "cost_volume_ref_features": args.cost_volume_ref_features,
            "coarse_supervision": args.coarse_supervision,
            "uncertainty": args.uncertainty,
            "single_view_fallback": args.single_view_fallback,
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
            "lambda_smooth": args.lambda_smooth,
            "lambda_mean": args.lambda_mean,
            "lambda_normal": args.lambda_normal,
            "lambda_confidence": args.lambda_confidence,
            "confidence_abs_tolerance": args.confidence_abs_tolerance,
            "confidence_rel_tolerance": args.confidence_rel_tolerance,
            "l1_loss_only": args.l1_loss_only,
            "lambda_worst_percent": args.lambda_worst_percent,
            "worst_percent": args.worst_percent,
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
