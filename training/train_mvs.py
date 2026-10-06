#!/usr/bin/env python3
"""
train_mvs.py - Three-stage multi-view event-depth training with a table-plane prior.

Each sample uses a target event voxel frame plus neighbouring source frames:

    [x_i | table_plane_channel_i] for i in target + source frames

A deep FPN extracts shared H/8, H/4, and H/2 features. Source features are
warped into the target frustum using poses from hdf5/poses.h5 and event-camera
calibration from camera_data/. Variance cost volumes regress coarse depth,
middle local candidates, and final fine candidates:

    d_i(u, v) = d_hat(u, v) + sigma(u, v) * epsilon_i

The coarse stage samples inverse depth globally. Middle and fine stages sample
local linear-depth windows around the preceding prediction; the fine window is
predicted from target features. Only the final prediction is supervised and
returned. Source views are selected by camera motion between frames.

All defaults reproduce the configuration of the model used in the thesis.

Usage:
    python3 training/train_mvs.py --data_dir data/Event_and_Depth --name my_run
    python3 training/train_mvs.py --name small --num_views 5 --feature_channels 64
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import DATA_ROOT as _DATA_ROOT, DEPTH_MIN, D_MAX, NUM_BINS
from depth_losses import (
    charbonnier_loss,
    gradient_loss,
    l1_metres,
    normal_loss,
    worst_fraction_l1_metres,
)
from tensorboard_helper import (
    DEFAULT_TB_ROOT,
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
    load_event_calibration,
    transform_intrinsics,
)

_CAM_DATA = _SCRIPT_DIR.parent / "camera_data"
DATA_ROOT = _SCRIPT_DIR.parent / _DATA_ROOT


@dataclass(frozen=True)
class MultiViewAugConfig:
    """Training-time multi-view augmentation toggles and conservative defaults."""
    enabled: bool = False
    source_view_dropout: bool = True
    event_noise_per_view: bool = True
    cross_view_event_dropout: bool = True
    occlusion_view_masking: bool = True
    source_view_dropout_prob: float = 0.10
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
        num_views: int = 9,
        pose_move_threshold: float = 0.05,
        allow_unbalanced_pose_views: bool = True,
        coarse_depths: int = 32,
        fill_invalid: bool = False,
        aug: MultiViewAugConfig | None = None,
    ):
        super().__init__()
        if num_views < 1 or num_views % 2 != 1:
            raise ValueError(
                "--num_views must be odd so sources are balanced before and after the target"
            )
        if pose_move_threshold <= 0:
            raise ValueError(f"--pose_move_threshold must be > 0, got {pose_move_threshold}")
        self.seq_dir = Path(seq_dir)
        self.num_views = num_views
        self.pose_move_threshold = float(pose_move_threshold)
        self.allow_unbalanced_pose_views = bool(allow_unbalanced_pose_views)
        self.fill_invalid = fill_invalid
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
        self.depth_values = _inverse_depth_candidates(coarse_depths, DEPTH_MIN, D_MAX)

        ee_T = ee_T[:self.n_frames]
        T_ee_inv = np.linalg.inv(ee_T)
        self.T_cam_from_world = np.einsum(
            "ij,njk->nik", calib["T_event_from_ee"], T_ee_inv
        ).astype(np.float32)
        self.cam_centers_world = camera_centers_world(self.T_cam_from_world)

        self.pose_view_ids, valid = build_pose_view_ids(
            self.cam_centers_world,
            num_views,
            self.pose_move_threshold,
            self.allow_unbalanced_pose_views,
        )
        if len(valid) == 0:
            raise RuntimeError(
                f"{self.seq_dir.name}: not enough frames ({self.n_frames}) for "
                f"num_views={num_views}, pose threshold={self.pose_move_threshold:g} m"
            )
        self.valid_indices = valid

        self._vox = None
        self._dep = None
        self._tbl = None

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
        return torch.cat([vox_t, tbl_t], dim=0)

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
        view_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.aug.enabled:
            return imgs
        imgs = imgs.clone()
        if self.aug.source_view_dropout:
            imgs = self._apply_source_view_dropout(imgs, view_valid_mask)
        if self.aug.event_noise_per_view:
            imgs = self._apply_event_noise_per_view(imgs)
        if self.aug.cross_view_event_dropout:
            imgs = self._apply_cross_view_event_dropout(imgs)
        if self.aug.occlusion_view_masking:
            imgs = self._apply_occlusion_view_masking(imgs)
        return imgs

    def __getitem__(self, item: int):
        self._open()
        idx = int(self.valid_indices[item])
        view_ids = self.pose_view_ids[idx]

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
        imgs = self._apply_augmentations(imgs, view_valid_mask)
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
    """Build the fixed deep, three-stage feature pyramid.

    A bottom-up residual hierarchy (H/2, H/4, H/8) is fused top-down. Only the
    top-down path is projected before each element-wise addition, so the
    middle and fine outputs keep the bottom-up widths.
    """

    def __init__(
        self,
        in_ch: int,
        feature_ch: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.feature_ch = feature_ch
        c0 = max(8, feature_ch // 8)
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        self.middle_ch = c2
        self.fine_ch = c1
        self.casmvs_stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.casmvs_half = _residual2d_stage(c0, c1, 3, stride=2)
        self.casmvs_quarter = _residual2d_stage(c1, c2, 4, stride=2)
        self.casmvs_eighth = _residual2d_stage(c2, c3, 6, stride=2)
        self.casmvs_middle_reduce = nn.Conv2d(c3, c2, 1, bias=False)
        self.casmvs_fine_reduce = nn.Conv2d(c2, c1, 1, bias=False)

        self.casmvs_coarse_out = nn.Sequential(
            nn.Conv2d(c3, feature_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU(inplace=True),
        )
        self.casmvs_middle_out = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.casmvs_fine_out = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
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
        quarter_fused = quarter + self.fusion_drop_path(
            F.interpolate(
                self.casmvs_middle_reduce(eighth),
                size=quarter.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        half_fused = half + self.fusion_drop_path(
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
    ):
        super().__init__()
        self.max_residual_m = float(max_residual_m)
        width = max(16, width)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch + fine_ch + 1, width, 3, padding=1, bias=False),
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
        features = self.body(torch.cat([target, fine_full, depth_m], dim=1))
        residual = torch.tanh(self.residual(features)) * self.max_residual_m
        return (depth_m + residual).clamp(DEPTH_MIN, D_MAX), features


class ModernMVSNet(nn.Module):
    """Modern three-stage coarse-to-fine MVS network with a deep FPN.

    Constructor arguments are named like the checkpoint keys, so a network can
    be rebuilt with ``ModernMVSNet(in_ch, **{k: ckpt[k] for k in MODEL_ARGS})``.
    """

    architecture_name = "ModernMVSNet"

    def __init__(
        self,
        in_ch: int,
        feature_channels: int = 128,
        cost_channels: int = 64,
        reference_channels: int = 16,
        coarse_hourglass_levels: int = 3,
        middle_hourglass_levels: int = 2,
        fine_hourglass_levels: int = 1,
        refiner_channels: int = 64,
        refiner_max_residual_m: float = 0.01,
        fpn_dropout: float = 0.1,
        reference_dropout: float = 0.1,
        hourglass_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        middle_depths: int = 16,
        middle_window: float = 0.12,
        fine_depths: int = 8,
        fine_window_min: float = 0.02,
        fine_window_max: float = 0.08,
    ):
        super().__init__()
        if min(middle_depths, fine_depths) < 3:
            raise ValueError("middle_depths and fine_depths must be >= 3")
        if min(middle_window, fine_window_min, refiner_max_residual_m) <= 0:
            raise ValueError(
                "middle_window, fine_window_min and refiner_max_residual_m must be > 0"
            )
        if fine_window_max < fine_window_min:
            raise ValueError("fine_window_max must be >= fine_window_min")
        self.fine_depths = int(fine_depths)
        self.middle_depths = int(middle_depths)
        self.middle_window = float(middle_window)
        self.fine_window_min = float(fine_window_min)
        self.fine_window_max = float(fine_window_max)
        self.reference_dropout = float(reference_dropout)
        self.feature = FeaturePyramid(
            in_ch, feature_channels, dropout=fpn_dropout, drop_path=drop_path_rate
        )
        middle_ch, fine_ch = self.feature.middle_ch, self.feature.fine_ch
        if reference_channels > 0:
            self.coarse_reference = nn.Conv2d(
                feature_channels, reference_channels, 1, bias=False
            )
            self.fine_reference = nn.Conv2d(fine_ch, reference_channels, 1, bias=False)
            self.middle_reference = nn.Conv2d(middle_ch, reference_channels, 1, bias=False)
        else:
            self.coarse_reference = None
            self.middle_reference = None
            self.fine_reference = None
        self.coarse_cost = CostHourglass3D(
            feature_channels + reference_channels + 1,
            cost_channels,
            coarse_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.fine_cost = CostHourglass3D(
            fine_ch + reference_channels + 1,
            cost_channels,
            fine_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.middle_cost = CostHourglass3D(
            middle_ch + reference_channels + 1,
            cost_channels,
            middle_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )

        window_hidden = max(8, cost_channels)
        self.window_head = nn.Sequential(
            nn.Conv2d(fine_ch + 1, window_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(window_hidden, 1, 3, padding=1),
        )
        self.refiner = FullResolutionRefiner(
            in_ch, fine_ch, refiner_channels, refiner_max_residual_m
        )
        confidence_hidden = max(16, cost_channels * 2)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(self.refiner.feature_channels + 5, confidence_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(confidence_hidden, 1, 1),
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
        """Sample fine hypotheses in a per-pixel window predicted by window_head."""
        coarse_up = F.interpolate(
            coarse_m, size=fine_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        half_window = self.fine_window_min + (
            self.fine_window_max - self.fine_window_min
        ) * torch.sigmoid(
            self.window_head(torch.cat([fine_ref, coarse_up], dim=1))
        )
        offsets = torch.linspace(
            -1.0,
            1.0,
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
        refined_m, refinement_features = self.refiner(imgs[:, 0], fine[:, 0], depth_full)

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
            confidence = torch.sigmoid(
                self.confidence_head(
                    torch.cat([refinement_features, diagnostics], dim=1)
                )
            )

        if return_uncertainty:
            return final_norm, confidence
        return final_norm



# Network hyperparameters: each one is a ModernMVSNet argument, a command-line
# option of this script, and a checkpoint key with the same name.
MODEL_ARGS = (
    "feature_channels",
    "cost_channels",
    "reference_channels",
    "coarse_hourglass_levels",
    "middle_hourglass_levels",
    "fine_hourglass_levels",
    "refiner_channels",
    "refiner_max_residual_m",
    "fpn_dropout",
    "reference_dropout",
    "hourglass_dropout",
    "drop_path_rate",
    "middle_depths",
    "middle_window",
    "fine_depths",
    "fine_window_min",
    "fine_window_max",
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ModernMVSNet, dict]:
    """Rebuild the network stored in a train_mvs.py checkpoint of either branch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("model_arch", "ModernMVSNet") != "ModernMVSNet":
        raise ValueError(f"{path} is not a train_mvs.py checkpoint")
    if ckpt.get("intrinsics_transform") != INTRINSICS_TRANSFORM:
        raise RuntimeError(
            f"{path} uses {ckpt.get('intrinsics_transform')!r}, expected "
            f"{INTRINSICS_TRANSFORM!r}. Use a crop-then-resize checkpoint."
        )
    if not ckpt.get("pose_view_selection") or ckpt.get("linear_depth_candidates", False):
        raise ValueError(
            f"{path} uses fixed-interval views or linear depth planes, which "
            "this version does not implement"
        )
    hparams = {key: ckpt[key] for key in MODEL_ARGS}
    if hparams["refiner_max_residual_m"] is None:  # main-branch default
        hparams["refiner_max_residual_m"] = min(0.05, ckpt["fine_window"])
    model = ModernMVSNet(ckpt["in_ch"], **hparams).to(device)
    # Strict loading rejects network variants this version does not build.
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


# Training / validation
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    viz=None,
    uncertainty_diag=None,
    lambda_grad: float = 0.1,
    lambda_normal: float = 0.1,
    uncertainty: bool = True,
    lambda_confidence: float = 0.1,
    confidence_abs_tolerance: float = 0.01,
    confidence_rel_tolerance: float = 0.0,
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
            )
            pred_unc = None
            if uncertainty:
                pred, pred_unc = model_out
            else:
                pred = model_out

            pred_loss = pred
            loss = charbonnier_loss(pred_loss, dep_norm, mask_t)
            loss = loss + lambda_grad * gradient_loss(pred_loss, dep_norm, mask_t)
            loss = loss + lambda_normal * normal_loss(pred_loss, dep_norm, mask_t, K[0])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the multi-view event depth model. The defaults reproduce the "
            "configuration of the thesis model."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT,
                        help="Dataset root containing train/ and eval/ sequence folders")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name used in checkpoint filenames. Prompted if not provided.")
    parser.add_argument("--out_dir", type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "mvs")
    parser.add_argument("--tb_root", type=Path, default=DEFAULT_TB_ROOT,
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/mvs/<name>.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization: AdamW with a cosine learning-rate schedule.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Learning rate at the end of the cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9995,
                        help="EMA decay used for validation/checkpoints; 0 disables EMA")

    # Loss
    parser.add_argument("--lambda_grad", type=float, default=0.1,
                        help="Weight for multi-scale gradient loss term")
    parser.add_argument("--lambda_normal", type=float, default=0.1,
                        help="Weight for surface-normal loss term")
    parser.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=True,
                        help="Train, return, and log learned per-pixel fusion confidence")
    parser.add_argument("--lambda_confidence", type=float, default=0.1,
                        help="Auxiliary BCE loss weight for learned confidence")
    parser.add_argument("--confidence_abs_tolerance", type=float, default=0.01,
                        help="Absolute safe-to-fuse confidence tolerance in metres")
    parser.add_argument("--confidence_rel_tolerance", type=float, default=0.0,
                        help="Relative safe-to-fuse confidence tolerance as a fraction of GT depth")

    # Network
    parser.add_argument("--feature_channels", type=int, default=128,
                        help="Coarse FPN output channels; the middle and fine widths follow from it")
    parser.add_argument("--cost_channels", type=int, default=64,
                        help="Base width of the 3-D cost-volume hourglasses")
    parser.add_argument("--reference_channels", type=int, default=16,
                        help="Compressed reference channels added to each cost volume; 0 disables")
    parser.add_argument("--coarse_hourglass_levels", type=int, default=3)
    parser.add_argument("--middle_hourglass_levels", type=int, default=2)
    parser.add_argument("--fine_hourglass_levels", type=int, default=1)
    parser.add_argument("--refiner_channels", type=int, default=64,
                        help="Width of the full-resolution 2-D refiner")
    parser.add_argument("--refiner_max_residual_m", type=float, default=0.01,
                        help="Maximum metric-depth correction of the 2-D refiner in metres")
    parser.add_argument("--fpn_dropout", type=float, default=0.1,
                        help="Dropout2d probability on FPN outputs")
    parser.add_argument("--reference_dropout", type=float, default=0.1,
                        help="Dropout2d probability on reference features")
    parser.add_argument("--hourglass_dropout", type=float, default=0.1,
                        help="Dropout3d probability at hourglass bottlenecks")
    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                        help="Stochastic-depth rate on FPN fusion")

    # Depth hypotheses
    parser.add_argument("--coarse_depths", type=int, default=32,
                        help="Global inverse-depth planes between DEPTH_MIN and D_MAX")
    parser.add_argument("--middle_depths", type=int, default=16,
                        help="Middle-stage H/4 local depth hypotheses")
    parser.add_argument("--middle_window", type=float, default=0.12,
                        help="Middle-stage H/4 local search half-window in metres")
    parser.add_argument("--fine_depths", type=int, default=8,
                        help="Fine-stage H/2 depth hypotheses")
    parser.add_argument("--fine_window_min", type=float, default=0.02,
                        help="Minimum learned fine-stage search half-width in metres")
    parser.add_argument("--fine_window_max", type=float, default=0.08,
                        help="Maximum learned fine-stage search half-width in metres")

    # Source views
    parser.add_argument("--num_views", type=int, default=9,
                        help="Target plus source views; must be odd")
    parser.add_argument("--pose_move_threshold", type=float, default=0.05,
                        help="Minimum camera-center translation in metres between consecutive selected views")
    pose_layout_group = parser.add_mutually_exclusive_group()
    pose_layout_group.add_argument(
        "--allow_fewer_pose_views",
        "--allow-fewer-pose-views",
        "--allow_unbalanced_pose_views",
        "--allow-unbalanced-pose-views",
        dest="allow_unbalanced_pose_views",
        action="store_true",
        help=(
            "Allow targets near sequence boundaries to use fewer sources; missing "
            "past/future slots are masked (default)"
        ),
    )
    pose_layout_group.add_argument(
        "--strict_balanced_pose_views",
        "--strict-balanced-pose-views",
        dest="allow_unbalanced_pose_views",
        action="store_false",
        help="Require equal numbers of sources before and after every target",
    )
    parser.set_defaults(allow_unbalanced_pose_views=True)

    # Data and augmentation
    parser.add_argument("--fill_invalid", action="store_true",
                        help="Fill pixels with no depth measurement using the table-plane prior")
    parser.add_argument("--no_augmentations", action="store_true",
                        help="Disable all training-time multi-view data augmentations")
    parser.add_argument("--no_source_view_dropout", action="store_true",
                        help="Disable randomly zeroing whole source views during training")
    parser.add_argument("--no_event_noise_per_view", action="store_true",
                        help="Disable independent additive event-voxel noise per view during training")
    parser.add_argument("--no_cross_view_event_dropout", action="store_true",
                        help="Disable shared cross-view event-bin dropout during training")
    parser.add_argument("--no_occlusion_view_masking", action="store_true",
                        help="Disable random rectangular occlusion/view masks during training")
    args = parser.parse_args()

    if args.num_views < 1 or args.num_views % 2 != 1:
        parser.error("--num_views must be odd for balanced before/after sources")
    for name in (
        "epochs", "batch_size", "lr", "pose_move_threshold", "feature_channels",
        "cost_channels", "refiner_channels", "coarse_depths",
        "coarse_hourglass_levels", "middle_hourglass_levels", "fine_hourglass_levels",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be > 0")
    for name in (
        "min_lr", "weight_decay", "reference_channels", "lambda_grad", "lambda_normal",
        "lambda_confidence", "confidence_abs_tolerance", "confidence_rel_tolerance",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")
    for name in ("ema_decay", "fpn_dropout", "reference_dropout", "hourglass_dropout", "drop_path_rate"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"--{name} must be in [0, 1)")
    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")
    return args


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calib = load_event_calibration(_CAM_DATA)
    train_seqs = find_precomputed_sequences(args.data_dir / "train")
    val_seqs = find_precomputed_sequences(args.data_dir / "eval")
    if not train_seqs or not val_seqs:
        sys.exit(
            f"[ERROR] No valid sequences found in {args.data_dir}/train or "
            f"{args.data_dir}/eval\n"
            "        --data_dir must contain train/ and eval/, each holding one or more\n"
            "        valid sequences with voxels, depth, poses.h5, and table_plane.h5.\n"
            "        Run: data_precomputation/precompute_all.sh"
        )
    print(f"Dataset root: {args.data_dir}")
    print(f"  Train ({len(train_seqs)}): {[d.name for d in train_seqs]}")
    print(f"  Eval  ({len(val_seqs)}): {[d.name for d in val_seqs]}")

    ds_kw = dict(
        calib=calib,
        num_views=args.num_views,
        pose_move_threshold=args.pose_move_threshold,
        allow_unbalanced_pose_views=args.allow_unbalanced_pose_views,
        coarse_depths=args.coarse_depths,
        fill_invalid=args.fill_invalid,
    )
    train_aug = MultiViewAugConfig(
        enabled=not args.no_augmentations,
        source_view_dropout=not args.no_source_view_dropout,
        event_noise_per_view=not args.no_event_noise_per_view,
        cross_view_event_dropout=not args.no_cross_view_event_dropout,
        occlusion_view_masking=not args.no_occlusion_view_masking,
    )
    train_sets = [MultiViewTableDataset(d, **ds_kw, aug=train_aug) for d in train_seqs]
    val_sets = [
        MultiViewTableDataset(d, **ds_kw, aug=MultiViewAugConfig(enabled=False))
        for d in val_seqs
    ]
    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]
    print(f"  Target frames: train {len(train_ds)}, eval {len(val_ds)}")
    print(f"  Train augmentations: {asdict(train_aug)}\n")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    in_ch = NUM_BINS + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernMVSNet(in_ch, **{key: getattr(args, key) for key in MODEL_ARGS}).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model.architecture_name}  in_ch={in_ch}  parameters: {n_params:,}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA: enabled (decay={args.ema_decay:g}); validation/checkpoints use EMA weights")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("mvs", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")
    loggers = {
        phase: dict(
            viz=VizLogger(writer, n_samples=4, tag=f"viz/{phase}"),
            uncertainty_diag=(
                UncertaintyErrorLogger(writer, tag=f"uncertainty/{phase}", images_only=True)
                if args.uncertainty else None
            ),
        )
        for phase in ("train", "val")
    }
    loss_kw = dict(
        lambda_grad=args.lambda_grad,
        lambda_normal=args.lambda_normal,
        uncertainty=args.uncertainty,
        lambda_confidence=args.lambda_confidence,
        confidence_abs_tolerance=args.confidence_abs_tolerance,
        confidence_rel_tolerance=args.confidence_rel_tolerance,
    )
    # Fixed architecture choices of this version, stored under the option
    # names of the main branch so that its scripts rebuild the same network.
    # "base" is unused by the network once the widths are set explicitly.
    main_branch_keys = {
        "pose_view_selection": True,
        "learned_fine_window": True,
        "no_fpn_lateral_convolutions": True,
        "middle_feature_channels": model.feature.middle_ch,
        "fine_feature_channels": model.feature.fine_ch,
        "base": 32,
    }
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    best = {"l1": float("inf"), "p95": float("inf"), "l1_worst10": float("inf")}
    ckpt: dict = {}
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, tr_p95, tr_worst10 = run_epoch(
            model, train_loader, optimizer, device,
            **loggers["train"], **loss_kw, ema_model=ema,
        )
        validation_model = ema.model if ema is not None else model
        va_loss, va_l1, va_p95, va_worst10 = run_epoch(
            validation_model, val_loader, None, device,
            **loggers["val"], **loss_kw,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        for phase_loggers in loggers.values():
            for logger in phase_loggers.values():
                if logger is not None:
                    logger.flush(step=epoch)

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
        for tag, train_value, val_value in (
            ("loss", tr_loss, va_loss),
            ("l1", tr_l1, va_l1),
            ("p95", tr_p95, va_p95),
            ("l1_worst10", tr_worst10, va_worst10),
        ):
            writer.add_scalar(f"{tag}/train", train_value, epoch)
            writer.add_scalar(f"{tag}/val", val_value, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        ckpt = {
            "epoch": epoch,
            "model": validation_model.state_dict(),
            "model_arch": model.architecture_name,
            "intrinsics_transform": INTRINSICS_TRANSFORM,
            "in_ch": in_ch,
            "depth_min": DEPTH_MIN,
            "depth_max": D_MAX,
            "val_l1": va_l1,
            "val_p95": va_p95,
            "val_l1_worst10": va_worst10,
            "train_sequences": [d.name for d in train_seqs],
            "augmentation": asdict(train_aug),
            **run_config,
            **main_branch_keys,
        }
        for metric, value in (("l1", va_l1), ("p95", va_p95), ("l1_worst10", va_worst10)):
            if value < best[metric]:
                best[metric] = value
                torch.save(ckpt, args.out_dir / f"best_{metric}_{args.name}.pth")
                print(f"  -> new best {metric} checkpoint  (val {metric} = {value:.4f} m)")

    torch.save(ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(
        f"\nDone. Best val L1: {best['l1']:.4f} m, "
        f"p95: {best['p95']:.4f} m, "
        f"worst10 L1: {best['l1_worst10']:.4f} m"
    )


if __name__ == "__main__":
    main()
