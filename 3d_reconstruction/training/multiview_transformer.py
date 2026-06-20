#!/usr/bin/env python3
"""
multiview_transformer.py - Multi-view event depth training with streaming view transformers.

This is the multi-view counterpart to train_unet_table.py.  Each sample uses a
target event voxel frame plus neighbouring source frames:

    [x_i | table_plane_channel_i] for i in target + source frames

A shared 2D CNN extracts per-view features. Source features are warped into
the target frustum one view at a time, projected to compact view tokens, and
discarded. A transformer aggregates only over the view dimension for each
pixel-depth hypothesis. A small 2D CNN refines the resulting depth-cost logits.
The coarse distribution is regressed to a rough depth map, then the same
streaming view-transformer path is applied to fine per-pixel hypotheses around
that rough depth:

    d_i(u, v) = d_hat(u, v) + sigma(u, v) * epsilon_i

sigma is either a fixed window or predicted from target features.

Usage:
    python3 training/multiview.py --data_dir data/lego
    python3 training/multiview.py --data_dir data/lego/lego_1 --num_views 5
    python3 training/multiview.py --num_depths 32 --fine_depths 5 --view_interval 5
    python3 training/multiview.py --pose_view_selection --pose_move_threshold 0.01 --num_views 5
    python3 training/multiview.py --masked_warp_aggregation
    python3 training/multiview_transformer.py --model_scale 0.75 --transformer_dim 32
    python3 training/multiview_transformer.py --transformer_dim 48 --transformer_heads 4
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from train_unet import (
    DEPTH_MIN, D_MAX, NUM_BINS, _SCRIPT_DIR, DATA_ROOT,
    compute_loss, _l1_metres, _worst_percent_l1_metres,
)
from tensorboard_runs import DEFAULT_TB_ROOT, tensorboard_run_dir
from viz import (
    ErrorDistributionSpatialLogger,
    EventActivityAccuracyLogger,
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


def _inverse_depth_candidates(num_depths: int, depth_min: float, depth_max: float) -> np.ndarray:
    if num_depths < 2:
        raise ValueError(f"--num_depths must be >= 2, got {num_depths}")
    inv = np.linspace(1.0 / depth_min, 1.0 / depth_max, num_depths, dtype=np.float32)
    return (1.0 / inv).astype(np.float32)


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
        split_indices: np.ndarray | None = None,
        aug: MultiViewAugConfig | None = None,
    ):
        super().__init__()
        if num_views < 2:
            raise ValueError("--num_views must be at least 2 for multi-view training")
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
        self.fill_invalid = fill_invalid
        self.aug = aug or MultiViewAugConfig(enabled=False)

        self.voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path = self.seq_dir / "hdf5" / "depth_in_event_frame.h5"
        self.mask_path = self.seq_dir / "hdf5" / "spatial_mask.h5"
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
        with h5py.File(self.table_plane_path, "r") as f:
            n_t = f["table_plane"].shape[0]
        with h5py.File(self.poses_path, "r") as f:
            ee_T = f["ee_T"][:].astype(np.float32)

        self.n_frames = min(n_d, n_v, n_t, len(ee_T))
        self.has_mask = use_mask and self.mask_path.exists()
        self.K = _scale_K(calib["K_native"], calib["native_hw"], (vox_h, vox_w))
        self.depth_values = _inverse_depth_candidates(num_depths, DEPTH_MIN, D_MAX)

        ee_T = ee_T[:self.n_frames]
        T_ee_inv = np.linalg.inv(ee_T)
        self.T_cam_from_world = np.einsum(
            "ij,njk->nik", calib["T_event_from_ee"], T_ee_inv
        ).astype(np.float32)
        self.cam_centers_world = self._camera_centers_world(self.T_cam_from_world)

        self.pose_view_ids: dict[int, list[int]] = {}
        if self.pose_view_selection:
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
        return torch.cat([vox_t, tbl_t], dim=0)

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
            tbl_m = imgs[0, NUM_BINS:NUM_BINS + 1] * (D_MAX - DEPTH_MIN) + DEPTH_MIN
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


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class SharedFeatureCNN(nn.Module):
    """Shared encoder applied to target and source frames."""

    def __init__(self, in_ch: int, base: int, feature_ch: int):
        super().__init__()
        self.net = nn.Sequential(
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostRefinement2D(nn.Module):
    """Small 2D refinement head over compact depth logits."""

    def __init__(self, num_depths: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_depths, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_depths, 3, padding=1),
        )

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        return cost + self.net(cost)


class CostVolumeViewTransformer(nn.Module):
    """
    Streaming view transformer regulariser for multi-view cost volumes.

    Source views are warped one at a time, projected to compact tokens, and then
    discarded. Attention is only across the small view dimension for each
    pixel/depth hypothesis.

    Inputs:
        ref_feat:      (B, C, H, W)
        feats:         (B, V, C, H, W)
        cam_mats:      (B, V, 4, 4)
        K_feat:        (B, 3, 3)
        depth_values:  (D,), (B, D), or (B, D, H, W)

    Returns:
        cost logits:   (B, D, H, W)
    """

    def __init__(
        self,
        feature_ch: int,
        model_dim: int,
        num_depths: int,
        num_heads: int = 4,
        view_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        attention_chunk: int = 32768,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"transformer model_dim must be divisible by num_heads, got "
                f"model_dim={model_dim}, num_heads={num_heads}"
            )
        token_ch = feature_ch * 4 + 1 + 1 + model_dim
        self.model_dim = model_dim
        self.feature_ch = feature_ch
        self.gradient_checkpointing = gradient_checkpointing
        self.attention_chunk = max(1, int(attention_chunk))
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.pose_mlp = nn.Sequential(
            nn.Linear(12, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.view_token_proj = nn.Sequential(
            nn.Linear(token_ch, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.view_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=int(model_dim * mlp_ratio),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(view_layers)
            ]
        )
        self.cost_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.refine = CostRefinement2D(num_depths, hidden=max(model_dim, 16))

    @staticmethod
    def _normalise_depth_values(
        depth_values: torch.Tensor,
        B: int,
        D: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        depth_values = depth_values.to(device=device, dtype=dtype)
        if depth_values.dim() == 1:
            dv = depth_values.view(1, D, 1, 1).expand(B, -1, H, W)
        elif depth_values.dim() == 2:
            dv = depth_values.view(B, D, 1, 1).expand(-1, -1, H, W)
        elif depth_values.dim() == 4:
            dv = depth_values
            if dv.shape[-2:] != (H, W):
                dv = F.interpolate(dv, size=(H, W), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported depth_values shape: {tuple(depth_values.shape)}")
        return ((dv - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)

    @staticmethod
    def _relative_pose_features(
        ref_T: torch.Tensor,
        src_T: torch.Tensor,
    ) -> torch.Tensor:
        B, S = src_T.shape[:2]
        T_src_from_ref = src_T.float() @ torch.linalg.inv(ref_T.float()).unsqueeze(1)
        return T_src_from_ref[:, :, :3, :4].reshape(B, S, 12).to(src_T.dtype)

    def _run_view_layers(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.view_layers:
            if tokens.shape[0] <= self.attention_chunk:
                if self.gradient_checkpointing and self.training:
                    tokens = checkpoint(layer, tokens, use_reentrant=False)
                else:
                    tokens = layer(tokens)
                continue

            chunks = []
            for start in range(0, tokens.shape[0], self.attention_chunk):
                chunk = tokens[start:start + self.attention_chunk]
                if self.gradient_checkpointing and self.training:
                    chunk = checkpoint(layer, chunk, use_reentrant=False)
                else:
                    chunk = layer(chunk)
                chunks.append(chunk)
            tokens = torch.cat(chunks, dim=0)
        return tokens

    def _make_view_token(
        self,
        view_feat: torch.Tensor,
        ref_d: torch.Tensor,
        valid: torch.Tensor,
        depth_scalar: torch.Tensor,
        depth_embed: torch.Tensor,
        pose_embed: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(view_feat - ref_d) * valid
        prod = view_feat * ref_d * valid
        view_feat = view_feat * valid
        token = torch.cat(
            [view_feat, ref_d, diff, prod, valid, depth_scalar, pose_embed + depth_embed],
            dim=-1,
        )
        return self.view_token_proj(token.reshape(-1, token.shape[-1]))

    def forward(
        self,
        ref_feat: torch.Tensor,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K_feat: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = ref_feat.shape
        B2, V, C2, H2, W2 = feats.shape
        if (B2, C2, H2, W2) != (B, C, H, W):
            raise ValueError(
                "feature shape does not match ref feature shape: "
                f"ref={tuple(ref_feat.shape)}, feats={tuple(feats.shape)}"
            )
        device = ref_feat.device
        dtype = ref_feat.dtype
        ref_T = cam_mats[:, 0]
        src_T = cam_mats[:, 1:]

        if depth_values.dim() == 1:
            D = int(depth_values.shape[0])
        elif depth_values.dim() in (2, 4):
            D = int(depth_values.shape[1])
        else:
            raise ValueError(f"Unsupported depth_values shape: {tuple(depth_values.shape)}")

        ref_d = ref_feat.unsqueeze(2).expand(-1, -1, D, -1, -1)
        ref_d = ref_d.permute(0, 2, 3, 4, 1)                                  # (B,D,H,W,C)
        depth01 = self._normalise_depth_values(depth_values, B, D, H, W, device, dtype)
        depth_scalar = depth01.unsqueeze(-1)                                  # (B,D,H,W,1)
        depth_embed = self.depth_mlp(depth_scalar.reshape(-1, 1)).view(
            B, D, H, W, self.model_dim
        )

        src_pose = self._relative_pose_features(ref_T, src_T)
        zero_pose = torch.zeros((B, 1, 12), device=device, dtype=dtype)
        pose = torch.cat([zero_pose, src_pose], dim=1)
        pose_embed = self.pose_mlp(pose.reshape(B * V, 12)).view(B, V, self.model_dim)
        pose_embed = pose_embed[:, :, None, None, None, :].expand(-1, -1, D, H, W, -1)

        target_valid = torch.ones((B, D, H, W, 1), device=device, dtype=dtype)
        view_tokens = [
            self._make_view_token(
                ref_d,
                ref_d,
                target_valid,
                depth_scalar,
                depth_embed,
                pose_embed[:, 0],
            )
        ]

        for v in range(1, V):
            warped, valid_mask = homo_warp_features(
                feats[:, v],
                cam_mats[:, v],
                ref_T,
                K_feat,
                depth_values,
                return_valid_mask=True,
            )
            warped = warped.permute(0, 2, 3, 4, 1)                            # (B,D,H,W,C)
            valid = valid_mask.permute(0, 2, 3, 4, 1)                         # (B,D,H,W,1)
            view_tokens.append(
                self._make_view_token(
                    warped,
                    ref_d,
                    valid,
                    depth_scalar,
                    depth_embed,
                    pose_embed[:, v],
                )
            )
            del warped, valid_mask, valid

        # Attention across views for each pixel/depth hypothesis.
        tokens = torch.stack(view_tokens, dim=1).view(B * D * H * W, V, self.model_dim)
        fused = self._run_view_layers(tokens)[:, 0]                           # target token output
        cost = self.cost_head(fused).view(B, D, H, W)
        return self.refine(cost)


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.low(ref_feat)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.mid(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        fusion_feat = self.high(x)
        depth_single = self.depth_head(fusion_feat)
        alpha = self.fusion_head(torch.cat([fusion_feat, depth_mv, depth_single], dim=1))
        final = alpha * depth_mv + (1.0 - alpha) * depth_single
        return final.clamp(0.0, 1.0), depth_single


class MultiViewDepthNet(nn.Module):
    """Shared feature CNN with coarse-to-fine view/depth transformer aggregation."""

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
        num_depths: int = 32,
        transformer_dim: int | None = None,
        transformer_heads: int = 4,
        view_transformer_layers: int = 2,
        depth_transformer_layers: int = 0,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        view_attention_chunk: int = 32768,
    ):
        super().__init__()
        if fine_depths < 3:
            raise ValueError(f"fine_depths must be >= 3, got {fine_depths}")
        feature_ch = feature_ch or base * 4
        cost_base = cost_base or max(base // 2, 8)
        transformer_dim = transformer_dim or feature_ch
        self.fine_depths = fine_depths
        self.fine_window = float(fine_window)
        self.fine_offset_radius = float(fine_offset_radius)
        self.learned_fine_window = learned_fine_window
        self.masked_warp_aggregation = masked_warp_aggregation
        self.cost_volume_ref_features = cost_volume_ref_features
        self.single_view_fallback = single_view_fallback
        self.num_depths = int(num_depths)
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.view_transformer_layers = view_transformer_layers
        self.depth_transformer_layers = depth_transformer_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.view_attention_chunk = max(1, int(view_attention_chunk))
        self.feature = SharedFeatureCNN(in_ch, base, feature_ch)
        self.coarse_cost_cnn = CostVolumeViewTransformer(
            feature_ch=feature_ch,
            model_dim=transformer_dim,
            num_depths=self.num_depths,
            num_heads=transformer_heads,
            view_layers=view_transformer_layers,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
            gradient_checkpointing=gradient_checkpointing,
            attention_chunk=self.view_attention_chunk,
        )
        self.fine_cost_cnn = CostVolumeViewTransformer(
            feature_ch=feature_ch,
            model_dim=transformer_dim,
            num_depths=fine_depths,
            num_heads=transformer_heads,
            view_layers=view_transformer_layers,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
            gradient_checkpointing=gradient_checkpointing,
            attention_chunk=self.view_attention_chunk,
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, V, C, H, W = imgs.shape

        feats = self.feature(imgs.reshape(B * V, C, H, W))
        _, Fch, Hf, Wf = feats.shape
        feats = feats.view(B, V, Fch, Hf, Wf)

        K_feat = K.clone()
        K_feat[:, 0, :] *= Wf / W
        K_feat[:, 1, :] *= Hf / H

        coarse_cost = self.coarse_cost_cnn(
            feats[:, 0],
            feats,
            cam_mats,
            K_feat,
            depth_values,
        )
        coarse_prob = F.softmax(-coarse_cost.float(), dim=1).to(coarse_cost.dtype)
        coarse_depth = self._regress_depth(coarse_prob, depth_values)

        fine_values = self._fine_depth_values(coarse_depth, feats[:, 0])
        fine_cost = self.fine_cost_cnn(
            feats[:, 0],
            feats,
            cam_mats,
            K_feat,
            fine_values,
        )
        fine_prob = F.softmax(-fine_cost.float(), dim=1).to(fine_cost.dtype)
        depth_low = self._regress_depth(fine_prob, fine_values)

        depth_mv_m = F.interpolate(depth_low, size=(H, W), mode="bilinear", align_corners=False)
        depth_mv_norm = ((depth_mv_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
        if self.single_view_fallback:
            final_depth_norm, _ = self.single_view_head(feats[:, 0], depth_mv_norm, (H, W))
        else:
            final_depth_norm = depth_mv_norm
        if not return_coarse:
            return final_depth_norm

        coarse_depth_m = F.interpolate(coarse_depth, size=(H, W), mode="bilinear", align_corners=False)
        coarse_depth_norm = ((coarse_depth_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
        return coarse_depth_norm, final_depth_norm


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    viz=None,
    activity_diag=None,
    error_diag=None,
    lambda_grad: float = 0.5,
    lambda_smooth: float = 0.01,
    lambda_mean: float = 0.1,
    lambda_normal: float = 0.1,
    coarse_supervision: bool = False,
    coarse_loss_weight: float = 0.3,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = total_l1 = total_worst10_l1 = 0.0
    n_batches = 0
    phase = "train" if is_train else "val"
    t_last = time.time()
    amp_enabled = bool(use_amp and device.type == "cuda")

    with ctx:
        for batch in loader:
            imgs = batch["imgs"].to(device, non_blocking=True)
            cam_mats = batch["cam_mats"].to(device, non_blocking=True)
            K = batch["K"].to(device, non_blocking=True)
            depth_values = batch["depth_values"].to(device, non_blocking=True)
            dep_t = batch["dep_t"].to(device, non_blocking=True)
            mask_t = batch["mask_t"].to(device, non_blocking=True)

            dep_norm = ((dep_t - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                model_out = model(imgs, cam_mats, K, depth_values, return_coarse=coarse_supervision)
                if coarse_supervision:
                    coarse_pred, pred = model_out
                else:
                    pred = model_out

                loss_final, _ = compute_loss(
                    pred,
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
                if coarse_supervision:
                    loss_coarse, _ = compute_loss(
                        coarse_pred,
                        dep_norm,
                        mask_t,
                        imgs[:, 0, :NUM_BINS],
                        K=K[0],
                        lambda_grad=lambda_grad,
                        lambda_smooth=lambda_smooth,
                        lambda_mean=lambda_mean,
                        lambda_normal=lambda_normal,
                    )
                    loss = loss_final + coarse_loss_weight * loss_coarse

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach())
            with torch.no_grad():
                total_l1 += float(_l1_metres(pred, dep_t, mask_t))
                total_worst10_l1 += float(_worst_percent_l1_metres(pred, dep_t, mask_t))
            n_batches += 1

            now = time.time()
            if now - t_last >= 20.0:
                print(
                    f"  [{phase}  {n_batches:4d}/{len(loader)} batches]  "
                    f"loss {total_loss / n_batches:.4f}  "
                    f"L1 {total_l1 / n_batches:.4f} m  "
                    f"worst10 {total_worst10_l1 / n_batches:.4f} m",
                    flush=True,
                )
                t_last = now

            with torch.no_grad():
                pred_m = pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                tbl_ch = imgs[:, 0, NUM_BINS:NUM_BINS + 1]
                if viz is not None:
                    viz.add_batch(imgs[:, 0, :NUM_BINS], dep_t, mask_t, pred_m, table_depth=tbl_ch)
                if activity_diag is not None:
                    activity_diag.add_batch(imgs[:, 0, :NUM_BINS], dep_t, mask_t, pred_m)
                if error_diag is not None:
                    error_diag.add_batch(pred_m, dep_t, mask_t)

    n = max(n_batches, 1)
    return total_loss / n, total_l1 / n, total_worst10_l1 / n


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

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

def _is_sequence(p: Path) -> bool:
    return (
        (p / "events" / "voxels_cam0.h5").exists()
        and (p / "hdf5" / "depth_in_event_frame.h5").exists()
        and (p / "hdf5" / "poses.h5").exists()
        and (p / "hdf5" / "table_plane.h5").exists()
    )


def _frame_block_split(n_frames: int, margin: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.arange(margin, n_frames - margin, dtype=np.int64)
    rng = np.random.default_rng(seed)
    block_size = max(20, 2 * margin + 1)
    n_blocks = max(1, int(np.ceil(len(valid) / block_size)))
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)

    val_mask = np.zeros(len(valid), dtype=bool)
    n_val = max(1, int(round(len(valid) * val_ratio)))
    count = 0
    for b in block_ids:
        if count >= n_val:
            break
        s = b * block_size
        e = min((b + 1) * block_size, len(valid))
        val_mask[s:e] = True
        count += e - s
    if val_mask.all() and len(valid) > 1:
        val_mask[:] = False
        val_mask[-min(n_val, len(valid) - 1):] = True
    if not val_mask.any() and len(valid) > 1:
        val_mask[-1] = True
    return valid[~val_mask], valid[val_mask]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-view event depth model with warped feature cost volume"
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT,
                        help="Single sequence dir or parent of multiple sequences")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_grad", type=float, default=0.5,
                        help="Weight for multi-scale gradient loss term.")
    parser.add_argument("--lambda_smooth", type=float, default=0.01,
                        help="Weight for event-aware smoothness loss term.")
    parser.add_argument("--lambda_mean", type=float, default=0.1,
                        help="Weight for mean-depth consistency loss term.")
    parser.add_argument("--lambda_normal", type=float, default=0.1,
                        help="Weight for surface-normal loss term.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--feature_channels", type=int, default=0,
                        help="Shared CNN output channels; 0 means base_channels*4")
    parser.add_argument("--cost_channels", type=int, default=0,
                        help="Auxiliary width used by the learned fine-window head; 0 means max(base_channels//2, 8)")
    parser.add_argument("--model_scale", type=float, default=1.0,
                        help="Width multiplier for base_channels and derived feature/transformer channels; explicit feature/cost overrides still win")
    parser.add_argument("--transformer_dim", type=int, default=32,
                        help="View transformer width; set 0 to use feature_channels")
    parser.add_argument("--transformer_heads", type=int, default=4,
                        help="Number of attention heads in both view and depth transformers")
    parser.add_argument("--view_transformer_layers", type=int, default=2,
                        help="Number of transformer layers for per-pixel/depth view aggregation")
    parser.add_argument("--depth_transformer_layers", type=int, default=0,
                        help="Deprecated compatibility option; depth-token transformer is disabled to save VRAM")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=4.0,
                        help="Feed-forward expansion ratio inside transformer blocks")
    parser.add_argument("--transformer_dropout", type=float, default=0.0,
                        help="Dropout inside transformer blocks")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable checkpointing of view-transformer layers")
    parser.add_argument("--view_attention_chunk", type=int, default=32768,
                        help="Max flattened pixel/depth tokens per view-attention call")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable CUDA automatic mixed precision training")
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
    parser.add_argument("--fine_window", type=float, default=0.08,
                        help="Fine-stage sigma/window in metres before multiplying by offsets")
    parser.add_argument("--fine_offset_radius", type=float, default=2.0,
                        help="Fine offsets span [-radius, radius]; 2 with 5 planes gives [-2,-1,0,1,2]")
    parser.add_argument("--learned_fine_window", action="store_true",
                        help="Predict a per-pixel fine-stage window from target features and coarse depth")
    parser.add_argument("--masked_warp_aggregation", action="store_true",
                        help="Kept for CLI compatibility; transformer mode always uses valid warp masks as token inputs")
    parser.add_argument("--cost_volume_ref_features", action="store_true",
                        help="Kept for CLI compatibility; transformer mode always uses reference features as token inputs")
    parser.add_argument("--coarse_supervision", action="store_true",
                        help="Train with an auxiliary coarse-depth loss: loss_final + 0.3 * loss_coarse")
    parser.add_argument("--single_view_fallback", action="store_true",
                        help="Add a target-only decoder and learn a per-pixel fusion of multi-view and single-view depth")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Used for single-sequence frame-block validation split")
    parser.add_argument("--out_dir", type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "multiview_trans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mask", action="store_true",
                        help="Ignore spatial mask; dep>0 validity is always applied")
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
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/multiview_trans/<name>.")
    parser.add_argument("--debug_views", action="store_true",
                        help="Save target/source event images and camera-pose plots, then exit.")
    parser.add_argument("--debug_samples", type=int, default=4,
                        help="Number of target samples to visualise with --debug_views.")
    parser.add_argument("--debug_out", type=Path,
                        default=_SCRIPT_DIR / "debug" / "multiview_trans",
                        help="Output directory for --debug_views PNGs and pose matrices.")
    args = parser.parse_args()

    if args.pose_view_selection and args.num_views % 2 != 1:
        parser.error("--pose_view_selection requires odd --num_views for balanced before/after sources")
    if args.model_scale <= 0:
        parser.error("--model_scale must be > 0")
    if args.transformer_dim < 0:
        parser.error("--transformer_dim must be >= 0")
    if args.view_attention_chunk <= 0:
        parser.error("--view_attention_chunk must be > 0")

    if args.name is None and not args.debug_views:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calib = _load_event_calibration()

    if _is_sequence(args.data_dir):
        seq_dirs = [args.data_dir]
        single_object = True
    else:
        seq_dirs = sorted([d for d in args.data_dir.iterdir() if d.is_dir() and _is_sequence(d)])
        single_object = len(seq_dirs) == 1

    if not seq_dirs:
        sys.exit(
            f"[ERROR] No valid sequences found at or under {args.data_dir}\n"
            "        Make sure voxels, depth, poses.h5, and table_plane.h5 all exist.\n"
            "        Run: python3 data_precomputation/precompute_table_plane.py"
        )

    if single_object:
        train_seqs = val_seqs = seq_dirs
        print(f"Found {len(seq_dirs)} sequence(s) [single-object frame-block mode]")
        print(f"  Sequences: {[d.name for d in seq_dirs]}")
    else:
        rng = np.random.default_rng(args.seed)
        order = np.arange(len(seq_dirs))
        rng.shuffle(order)
        n_val = max(1, round(len(seq_dirs) * 0.15))
        val_seqs = [seq_dirs[i] for i in order[:n_val]]
        train_seqs = [seq_dirs[i] for i in order[n_val:]]
        print(f"Found {len(seq_dirs)} sequence(s) in {args.data_dir}")
        print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
        print(f"  Val   ({len(val_seqs)}):   {[d.name for d in val_seqs]}")

    ds_kw = dict(
        calib=calib,
        num_views=args.num_views,
        view_interval=args.view_interval,
        pose_view_selection=args.pose_view_selection,
        pose_move_threshold=args.pose_move_threshold,
        num_depths=args.num_depths,
        use_mask=not args.no_mask,
        fill_invalid=args.fill_invalid,
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

    train_sets = []
    val_sets = []
    if single_object:
        if args.pose_view_selection:
            margin = 0
        else:
            margin = max(abs(o) for o in MultiViewTableDataset._make_source_offsets(
                args.num_views, args.view_interval
            ))
        for seq in seq_dirs:
            import h5py
            with h5py.File(seq / "events" / "voxels_cam0.h5", "r") as f:
                n_frames = f["voxels"].shape[0]
            tr_idx, va_idx = _frame_block_split(n_frames, margin, args.val_ratio, args.seed)
            train_sets.append(MultiViewTableDataset(seq, **ds_kw, split_indices=tr_idx, aug=train_aug))
            val_sets.append(MultiViewTableDataset(seq, **ds_kw, split_indices=va_idx, aug=val_aug))
    else:
        train_sets = [MultiViewTableDataset(d, **ds_kw, aug=train_aug) for d in train_seqs]
        val_sets = [MultiViewTableDataset(d, **ds_kw, aug=val_aug) for d in val_seqs]

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]
    print(f"  Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")
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
        f"mean={args.lambda_mean:g}, normal={args.lambda_normal:g}\n"
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
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
    transformer_dim = args.transformer_dim if args.transformer_dim > 0 else feature_channels
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
        num_depths=args.num_depths,
        transformer_dim=transformer_dim,
        transformer_heads=args.transformer_heads,
        view_transformer_layers=args.view_transformer_layers,
        depth_transformer_layers=args.depth_transformer_layers,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_dropout=args.transformer_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        view_attention_chunk=args.view_attention_chunk,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"MultiViewDepthNet  in_ch={in_ch}  model_scale={args.model_scale:g}  "
        f"base={base_channels}  feature={feature_channels}  transformer_dim={transformer_dim}  "
        f"heads={args.transformer_heads}  view_layers={args.view_transformer_layers}  "
        f"depth_layers=disabled  "
        f"coarse_depths={args.num_depths}  fine_depths={args.fine_depths}  "
        f"coarse_supervision={args.coarse_supervision}  "
        f"single_view_fallback={args.single_view_fallback}  "
        f"checkpointing={not args.no_gradient_checkpointing}  "
        f"view_attention_chunk={args.view_attention_chunk}  "
        f"amp={not args.no_amp and device.type == 'cuda'}  "
        f"parameters: {n_params:,}"
    )
    print(f"Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and device.type == "cuda"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("multiview", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")

    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val = VizLogger(writer, n_samples=4, tag="viz/val", show_mask=not args.no_mask)
    activity_train = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/train")
    activity_val = EventActivityAccuracyLogger(writer, tag="event_activity_accuracy/val")
    error_train = ErrorDistributionSpatialLogger(writer, tag="error/train")
    error_val = ErrorDistributionSpatialLogger(writer, tag="error/val")

    best_val_l1 = float("inf")
    ckpt: dict = {}
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, tr_worst10 = run_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler,
            use_amp=not args.no_amp,
            viz=viz_train, activity_diag=activity_train, error_diag=error_train,
            lambda_grad=args.lambda_grad,
            lambda_smooth=args.lambda_smooth,
            lambda_mean=args.lambda_mean,
            lambda_normal=args.lambda_normal,
            coarse_supervision=args.coarse_supervision,
        )
        va_loss, va_l1, va_worst10 = run_epoch(
            model, val_loader, None, device,
            scaler=None,
            use_amp=not args.no_amp,
            viz=viz_val, activity_diag=activity_val, error_diag=error_val,
            lambda_grad=args.lambda_grad,
            lambda_smooth=args.lambda_smooth,
            lambda_mean=args.lambda_mean,
            lambda_normal=args.lambda_normal,
            coarse_supervision=args.coarse_supervision,
        )
        scheduler.step()

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)
        activity_train.flush(step=epoch)
        activity_val.flush(step=epoch)
        error_train.flush(step=epoch)
        error_val.flush(step=epoch)

        vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
            f"L1: {tr_l1:.4f}/{va_l1:.4f} m  "
            f"worst10: {tr_worst10:.4f}/{va_worst10:.4f} m  "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("l1/train", tr_l1, epoch)
        writer.add_scalar("l1/val", va_l1, epoch)
        writer.add_scalar("l1_worst10/train", tr_worst10, epoch)
        writer.add_scalar("l1_worst10/val", va_worst10, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_l1": va_l1,
            "base": base_channels,
            "base_channels_arg": args.base_channels,
            "feature_channels": feature_channels,
            "cost_channels": cost_channels,
            "model_scale": args.model_scale,
            "transformer_dim": transformer_dim,
            "transformer_heads": args.transformer_heads,
            "view_transformer_layers": args.view_transformer_layers,
            "depth_transformer_layers": args.depth_transformer_layers,
            "transformer_mlp_ratio": args.transformer_mlp_ratio,
            "transformer_dropout": args.transformer_dropout,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "view_attention_chunk": args.view_attention_chunk,
            "amp": not args.no_amp,
            "in_ch": in_ch,
            "num_views": args.num_views,
            "view_interval": args.view_interval,
            "pose_view_selection": args.pose_view_selection,
            "pose_move_threshold": args.pose_move_threshold,
            "num_depths": args.num_depths,
            "fine_depths": args.fine_depths,
            "fine_window": args.fine_window,
            "fine_offset_radius": args.fine_offset_radius,
            "learned_fine_window": args.learned_fine_window,
            "masked_warp_aggregation": args.masked_warp_aggregation,
            "cost_volume_ref_features": args.cost_volume_ref_features,
            "coarse_supervision": args.coarse_supervision,
            "single_view_fallback": args.single_view_fallback,
            "lambda_grad": args.lambda_grad,
            "lambda_smooth": args.lambda_smooth,
            "lambda_mean": args.lambda_mean,
            "lambda_normal": args.lambda_normal,
            "depth_min": DEPTH_MIN,
            "depth_max": D_MAX,
        }
        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            torch.save(ckpt, args.out_dir / f"best_{args.name}.pth")
            print(f"  -> new best checkpoint  (val L1 = {va_l1:.4f} m)")

    torch.save(ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(f"\nDone. Best val L1: {best_val_l1:.4f} m")


if __name__ == "__main__":
    main()
