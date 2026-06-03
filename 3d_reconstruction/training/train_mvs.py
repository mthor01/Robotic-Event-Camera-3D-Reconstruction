#!/usr/bin/env python3
"""
train_mvsnet_event.py

MVSNet-style depth training for event voxel inputs.

Input per view:
    [event voxel bins, table-plane/pose-depth channel]

Expected data layout:
    data/real/<object_name>/
        hdf5/
            depth_in_event_frame.h5   dataset "depth"  (N,H,W), metres
            poses.h5                  dataset "ee_T"   (N,4,4), T_base_from_ee
            spatial_mask.h5           dataset "mask"   (optional)
            table_plane_depth.h5      dataset "depth" or "table_depth" (optional)
        events/
            voxels_cam0.h5            dataset "voxels" (N,C,H,W)
            voxels_pose_cam0/         optional fallback, voxel_000000.npy, last channel = pose/table depth

        camera_data/ or configured calib dir:
            event_intrinsics.npz      keys "camera_matrix", "image_size"=[W,H]
            T_rgb_from_ee.npz         key "T"
            T_event_from_rgb.npz      key "T"

This is deliberately independent from your model registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from viz import VizLogger


# ----------------------------
# Geometry
# ----------------------------

def load_calibration(calib_dir: Path) -> dict:
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K = ev["camera_matrix"].astype(np.float32)
    image_size = ev["image_size"]  # [W,H]

    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float32)
    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee

    return {
        "K": K,
        "native_hw": (int(image_size[1]), int(image_size[0])),
        "T_event_from_ee": T_event_from_ee,
    }


def scale_K(K: np.ndarray, native_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    native_h, native_w = native_hw
    target_h, target_w = target_hw
    K2 = K.copy()
    K2[0, :] *= target_w / native_w
    K2[1, :] *= target_h / native_h
    return K2.astype(np.float32)


def projection_matrix(K: np.ndarray, T_cam_from_world: np.ndarray) -> np.ndarray:
    return (K @ T_cam_from_world[:3, :]).astype(np.float32)


def make_depth_values(depth_min: float, depth_max: float, num_depth: int) -> np.ndarray:
    return np.linspace(depth_min, depth_max, num_depth, dtype=np.float32)


def homo_warping(
    src_feat: torch.Tensor,
    src_proj: torch.Tensor,
    ref_proj: torch.Tensor,
    depth_values: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable homography warping.

    src_feat:     (B,C,H,W)
    src_proj:     (B,3,4), K_s [R_s|t_s]
    ref_proj:     (B,3,4), K_r [R_r|t_r]
    depth_values: (B,D) or (D,)

    Returns:
        warped source features in reference frustum: (B,C,D,H,W)
    """
    B, C, H, W = src_feat.shape
    device = src_feat.device
    dtype = src_feat.dtype

    if depth_values.dim() == 1:
        depth_values = depth_values[None].repeat(B, 1)
    D = depth_values.shape[1]

    ref_4x4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    src_4x4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    ref_4x4[:, :3, :] = ref_proj
    src_4x4[:, :3, :] = src_proj

    proj = (src_4x4.float() @ torch.linalg.inv(ref_4x4.float())).to(dtype)
    R = proj[:, :3, :3]
    t = proj[:, :3, 3:4]

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    xyz = torch.stack((x.reshape(-1), y.reshape(-1), torch.ones(H * W, device=device, dtype=dtype)), dim=0)
    xyz = xyz.unsqueeze(0).repeat(B, 1, 1)  # (B,3,HW)

    rot_xyz = R @ xyz  # (B,3,HW)
    rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values[:, None, :, None]  # (B,3,D,HW)
    proj_xyz = rot_depth_xyz + t[:, :, None, :]  # (B,3,D,HW)

    z = proj_xyz[:, 2:3].clamp(min=1e-6)
    x_norm = proj_xyz[:, 0:1] / z
    y_norm = proj_xyz[:, 1:2] / z

    x_grid = 2.0 * (x_norm / max(W - 1, 1)) - 1.0
    y_grid = 2.0 * (y_norm / max(H - 1, 1)) - 1.0

    grid = torch.stack((x_grid.squeeze(1), y_grid.squeeze(1)), dim=-1)  # (B,D,HW,2)
    grid = grid.view(B, D * H, W, 2)

    warped = F.grid_sample(src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.view(B, C, D, H, W)
    return warped


# ----------------------------
# Dataset
# ----------------------------

class EventMVSObjectDataset(Dataset):
    """
    Samples one reference frame plus neighbouring source frames.

    Returned dict:
        imgs         (V,C+1,H,W): event voxel bins + table-plane/pose-depth channel
        proj_mats    (V,3,4)
        depth_values (D,)
        depth        (H,W), metric metres
        mask         (H,W)
    """

    def __init__(
        self,
        sequence_dir: Path,
        calib: dict,
        num_views: int,
        view_interval: int,
        num_depth: int,
        depth_min: float,
        depth_max: float,
        resize_hw: Optional[Tuple[int, int]],
        val_ratio: float,
        split: str,
        seed: int = 42,
        use_spatial_mask: bool = True,
    ):
        self.sequence_dir = Path(sequence_dir)
        self.num_views = num_views
        self.view_interval = view_interval
        self.depth_values = make_depth_values(depth_min, depth_max, num_depth)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.resize_hw = resize_hw
        self.use_spatial_mask = use_spatial_mask

        self.vox_path = self.sequence_dir / "events" / "voxels_cam0.h5"
        self.depth_path = self.sequence_dir / "hdf5" / "depth_in_event_frame.h5"
        self.pose_path = self.sequence_dir / "hdf5" / "poses.h5"
        self.spatial_mask_path = self.sequence_dir / "hdf5" / "spatial_mask.h5"

        for p in [self.vox_path, self.depth_path, self.pose_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        self.table_h5_path = self.sequence_dir / "hdf5" / "table_plane.h5"
        self.pose_voxel_dir = self.sequence_dir / "events" / "voxels_pose_cam0"

        with h5py.File(self.depth_path, "r") as f:
            self.n_frames = int(f["depth"].shape[0])
            self.orig_hw = (int(f["depth"].shape[1]), int(f["depth"].shape[2]))

        target_hw = resize_hw or self.orig_hw
        self.K = scale_K(calib["K"], calib["native_hw"], target_hw)

        with h5py.File(self.pose_path, "r") as f:
            ee_T = f["ee_T"][:].astype(np.float32)
        T_ee_inv = np.linalg.inv(ee_T)
        self.T_cam_from_world = np.einsum(
            "ij,njk->nik", calib["T_event_from_ee"], T_ee_inv
        ).astype(np.float32)

        n_src = num_views - 1
        offsets = []
        k = 1
        while len(offsets) < n_src:
            offsets.append(-k * view_interval)
            if len(offsets) < n_src:
                offsets.append(k * view_interval)
            k += 1
        self.src_offsets = offsets

        margin = max(abs(o) for o in self.src_offsets) if self.src_offsets else 0
        valid = np.arange(margin, self.n_frames - margin, dtype=np.int64)

        rng = np.random.default_rng(seed)
        block_size = max(20, 2 * margin + 1)
        n_blocks = max(1, int(np.ceil(len(valid) / block_size)))
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)

        val_mask = np.zeros(len(valid), dtype=bool)
        n_val = int(round(len(valid) * val_ratio))
        count = 0
        for b in block_ids:
            if count >= n_val:
                break
            s = b * block_size
            e = min((b + 1) * block_size, len(valid))
            val_mask[s:e] = True
            count += e - s

        self.indices = valid[val_mask] if split == "val" else valid[~val_mask]
        print(
            f"[{self.sequence_dir.name}] {split}: {len(self.indices)} MVS samples, "
            f"views={num_views}, interval={view_interval}, input_hw={target_hw}"
        )

        # Lazy HDF5 handles — opened once per DataLoader worker, not per item.
        self._h5_vox:   h5py.Dataset | None = None
        self._h5_dep:   h5py.Dataset | None = None
        self._h5_msk:   h5py.Dataset | None = None
        self._h5_tbl:   h5py.Dataset | None = None
        self._tbl_key:  str | None = None

    def _open_handles(self) -> None:
        if self._h5_vox is None:
            self._h5_vox = h5py.File(self.vox_path,   "r")["voxels"]
        if self._h5_dep is None:
            self._h5_dep = h5py.File(self.depth_path, "r")["depth"]
        if self._h5_msk is None and self.use_spatial_mask and self.spatial_mask_path.exists():
            self._h5_msk = h5py.File(self.spatial_mask_path, "r")["mask"]
        if self._h5_tbl is None and self.table_h5_path.exists():
            f = h5py.File(self.table_h5_path, "r")
            key = "table_plane" if "table_plane" in f else ("depth" if "depth" in f else "table_depth")
            self._tbl_key = key
            self._h5_tbl = f[key]

    def __len__(self) -> int:
        return len(self.indices)

    def _resize_chw(self, x: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        if self.resize_hw is None:
            return x
        t = torch.from_numpy(x).float().unsqueeze(0)
        y = F.interpolate(
            t,
            size=self.resize_hw,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        return y.squeeze(0).numpy()

    def _resize_hw(self, x: np.ndarray, mode: str = "nearest") -> np.ndarray:
        if self.resize_hw is None:
            return x
        t = torch.from_numpy(x).float()[None, None]
        y = F.interpolate(
            t,
            size=self.resize_hw,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        return y[0, 0].numpy()

    def _load_voxel(self, idx: int) -> np.ndarray:
        self._open_handles()
        v = self._h5_vox[idx]
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        else:
            v = v.astype(np.float32)
        return self._resize_chw(v, mode="bilinear")

    def _load_table_depth(self, idx: int, hw: Tuple[int, int]) -> np.ndarray:
        """
        Loads the table-plane/pose-depth channel.

        Preferred:
            hdf5/table_plane.h5 with key "table_plane" (or "depth"/"table_depth").
        Fallback:
            events/voxels_pose_cam0/voxel_XXXXXX.npy, last channel.
        Last fallback:
            zeros.
        """
        self._open_handles()
        if self._h5_tbl is not None:
            x = self._h5_tbl[idx].astype(np.float32)
            x = self._resize_hw(x, mode="bilinear")
        else:
            p = self.pose_voxel_dir / f"voxel_{idx:06d}.npy"
            if p.exists():
                arr = np.load(p).astype(np.float32)
                x = arr[-1]
                x = self._resize_hw(x, mode="bilinear")
            else:
                x = np.zeros(hw, dtype=np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.max() > 2.0:
            x = np.clip((x - self.depth_min) / (self.depth_max - self.depth_min), 0.0, 1.0)
        return x.astype(np.float32)

    def _load_input(self, idx: int) -> np.ndarray:
        vox = self._load_voxel(idx)
        _, h, w = vox.shape
        table = self._load_table_depth(idx, (h, w))[None]
        return np.concatenate([vox, table], axis=0).astype(np.float32)

    def _load_depth_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._open_handles()
        d = self._h5_dep[idx].astype(np.float32)
        d = self._resize_hw(d, mode="nearest")
        mask = ((d > self.depth_min) & (d < self.depth_max)).astype(np.float32)

        if self._h5_msk is not None:
            sp = self._h5_msk[idx].astype(np.float32)
            sp = self._resize_hw(sp, mode="nearest")
            mask[sp == 0] = 0.0

        return d.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        ref_idx = int(self.indices[i])
        view_ids = [ref_idx] + [ref_idx + o for o in self.src_offsets]

        imgs = []
        proj_mats = []
        for idx in view_ids:
            imgs.append(self._load_input(idx))
            proj_mats.append(projection_matrix(self.K, self.T_cam_from_world[idx]))

        depth, mask = self._load_depth_and_mask(ref_idx)

        return {
            "imgs": torch.from_numpy(np.stack(imgs)),             # (V,C,H,W)
            "proj_mats": torch.from_numpy(np.stack(proj_mats)),   # (V,3,4)
            "depth_values": torch.from_numpy(self.depth_values),  # (D,)
            "depth": torch.from_numpy(depth),                     # (H,W)
            "mask": torch.from_numpy(mask),                       # (H,W)
        }


# ----------------------------
# MVSNet
# ----------------------------

class FeatureNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostRegNet(nn.Module):
    def __init__(self, in_channels: int = 32):
        super().__init__()

        def conv3d(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            conv3d(in_channels, 8),
            conv3d(8, 16),
            conv3d(16, 16),
            conv3d(16, 8),
            nn.Conv3d(8, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B,D,H,W)


class EventMVSNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 8):
        super().__init__()
        self.feature = FeatureNet(in_channels, base_channels)
        self.cost_reg = CostRegNet(32)

    def forward(
        self,
        imgs: torch.Tensor,
        proj_mats: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        imgs:         (B,V,C,H,W)
        proj_mats:    (B,V,3,4)
        depth_values: (B,D) or (D,)

        Returns:
            depth: (B,H,W), metric
            prob:  (B,D,H/4,W/4), depth probability at feature resolution
        """
        B, V, C, H, W = imgs.shape
        D = depth_values.shape[-1]

        imgs_flat = imgs.reshape(B * V, C, H, W)
        feats = self.feature(imgs_flat)
        _, Fch, Hf, Wf = feats.shape
        feats = feats.view(B, V, Fch, Hf, Wf)

        # Projection matrices must be scaled to feature resolution.
        proj_feat = proj_mats.clone()
        proj_feat[:, :, 0, :] *= Wf / W
        proj_feat[:, :, 1, :] *= Hf / H

        ref_feat = feats[:, 0]
        ref_proj = proj_feat[:, 0]

        volume_sum = ref_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)
        volume_sq_sum = volume_sum ** 2

        for v in range(1, V):
            warped = homo_warping(feats[:, v], proj_feat[:, v], ref_proj, depth_values)
            volume_sum = volume_sum + warped
            volume_sq_sum = volume_sq_sum + warped ** 2

        volume_variance = volume_sq_sum / V - (volume_sum / V) ** 2

        cost = self.cost_reg(volume_variance)
        prob = F.softmax(-cost, dim=1)

        if depth_values.dim() == 1:
            dv = depth_values[None, :, None, None]
        else:
            dv = depth_values[:, :, None, None]
        depth_low = torch.sum(prob * dv, dim=1)
        depth = F.interpolate(depth_low[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        return depth, prob


# ----------------------------
# Training
# ----------------------------

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return (pred[valid] - target[valid]).abs().mean()


@torch.no_grad()
def validate(model, loader, device, viz: VizLogger | None = None,
             num_bins: int = 5) -> dict:
    model.eval()
    total_l1 = 0.0
    total_abs_rel = 0.0
    n = 0
    n_total = len(loader)
    _t_last = time.time()
    print(f"  [val    starting ({n_total} batches)]", flush=True)

    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        proj_mats = batch["proj_mats"].to(device, non_blocking=True)
        depth_values = batch["depth_values"].to(device, non_blocking=True)
        gt = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        pred, _ = model(imgs, proj_mats, depth_values)
        valid = mask > 0.5
        if valid.sum() == 0:
            continue

        l1 = (pred[valid] - gt[valid]).abs().mean()
        abs_rel = ((pred[valid] - gt[valid]).abs() / gt[valid].clamp(min=1e-3)).mean()

        total_l1 += float(l1)
        total_abs_rel += float(abs_rel)
        n += 1

        _now = time.time()
        if _now - _t_last >= 20.0:
            print(
                f"  [val    {n:4d}/{n_total} batches]  "
                f"L1 {total_l1 / n:.4f} m",
                flush=True,
            )
            _t_last = _now

        if viz is not None:
            ref_vox = imgs[:, 0, :num_bins].detach()           # (B, C, H, W)
            tbl_ch  = imgs[:, 0, num_bins:num_bins + 1].detach()  # (B, 1, H, W)
            viz.add_batch(
                ref_vox,
                gt.unsqueeze(1),
                mask.unsqueeze(1),
                pred.unsqueeze(1).detach(),
                table_depth=tbl_ch,
            )

    return {
        "l1": total_l1 / max(n, 1),
        "abs_rel": total_abs_rel / max(n, 1),
    }


def _is_sequence(p: Path) -> bool:
    return (
        (p / "events" / "voxels_cam0.h5").exists()
        and (p / "hdf5" / "depth_in_event_frame.h5").exists()
        and (p / "hdf5" / "poses.h5").exists()
    )


def find_sequences(data_root: Path) -> List[Path]:
    return sorted(d for d in data_root.iterdir() if d.is_dir() and _is_sequence(d))


def build_datasets(args):
    import traceback as _tb
    calib = load_calibration(Path(args.calib_dir))

    if args.data_dir:
        seqs = []
        for p in [Path(x) for x in args.data_dir]:
            if _is_sequence(p):
                seqs.append(p)
            else:
                found = find_sequences(p)
                if found:
                    seqs.extend(found)
                else:
                    print(f"WARNING: {p} is neither a sequence nor a parent with sequences — skipping")
    else:
        seqs = find_sequences(Path(args.data_root))

    if not seqs:
        raise RuntimeError("No valid sequences found.")

    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (args.resize_h, args.resize_w)

    use_spatial_mask = not args.no_mask
    train_sets, val_sets = [], []
    for seq in seqs:
        try:
            train_sets.append(EventMVSObjectDataset(
                seq, calib, args.num_views, args.view_interval, args.num_depth,
                args.depth_min, args.depth_max, resize_hw, args.val_ratio, "train",
                use_spatial_mask=use_spatial_mask,
            ))
            val_sets.append(EventMVSObjectDataset(
                seq, calib, args.num_views, args.view_interval, args.num_depth,
                args.depth_min, args.depth_max, resize_hw, args.val_ratio, "val",
                use_spatial_mask=use_spatial_mask,
            ))
        except Exception as e:
            print(f"ERROR loading {seq}: {e}")
            _tb.print_exc()

    if not train_sets:
        raise RuntimeError("No usable sequences after filtering.")

    train = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    default=str(_SCRIPT_DIR.parent / "data" / "real"))
    ap.add_argument("--data_dir", type=str, nargs="*", default=None)
    ap.add_argument("--calib_dir", type=str,
                    default=str(_SCRIPT_DIR.parent / "camera_data"))
    ap.add_argument("--out", type=str,
                    default=str(_SCRIPT_DIR / "checkpoints" / "event_mvsnet.pt"))

    ap.add_argument("--num_bins", type=int, default=5)
    ap.add_argument("--num_views", type=int, default=5)
    ap.add_argument("--view_interval", type=int, default=5)
    ap.add_argument("--num_depth", type=int, default=96)
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=2.0)

    ap.add_argument("--resize_h", type=int, default=256)
    ap.add_argument("--resize_w", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--base_channels", type=int, default=8)
    ap.add_argument("--no_mask", action="store_true",
                    help="Ignore spatial_mask.h5; only pixels with valid depth (>depth_min) are used")
    ap.add_argument("--name", type=str, default=None,
                    help="Run name for checkpoint filename and TensorBoard logs")
    args = ap.parse_args()

    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds = build_datasets(args)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    writer    = SummaryWriter(log_dir=str(out_dir / "tb" / args.name))
    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val   = VizLogger(writer, n_samples=4, tag="viz/val",   show_mask=not args.no_mask)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = args.num_bins + 1  # voxel bins + table-plane/pose-depth channel
    model = EventMVSNet(in_channels=in_channels, base_channels=args.base_channels).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = float("inf")
    out = out_dir / f"best_{args.name}.pth"

    n_train_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        _t_last = time.time()

        for batch in train_loader:
            imgs = batch["imgs"].to(device, non_blocking=True)
            proj_mats = batch["proj_mats"].to(device, non_blocking=True)
            depth_values = batch["depth_values"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred, _ = model(imgs, proj_mats, depth_values)
                loss = masked_l1(pred, gt, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            count += 1

            _now = time.time()
            if _now - _t_last >= 20.0:
                vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                print(
                    f"  [train  epoch {epoch}  {count:4d}/{n_train_batches} batches]  "
                    f"loss {running / count:.4f} m  "
                    f"VRAM {vram_a:.0f} MB",
                    flush=True,
                )
                _t_last = _now

            with torch.no_grad():
                ref_vox = imgs[:, 0, :args.num_bins].detach()              # (B, C, H, W)
                tbl_ch  = imgs[:, 0, args.num_bins:args.num_bins + 1].detach()  # (B, 1, H, W)
                viz_train.add_batch(
                    ref_vox,
                    gt.unsqueeze(1),
                    mask.unsqueeze(1),
                    pred.unsqueeze(1).detach(),
                    table_depth=tbl_ch,
                )

        metrics = validate(model, val_loader, device,
                           viz=viz_val, num_bins=args.num_bins)
        train_l1 = running / max(count, 1)

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)

        writer.add_scalar("loss/train",   train_l1,          epoch)
        writer.add_scalar("l1/val",        metrics["l1"],     epoch)
        writer.add_scalar("abs_rel/val",   metrics["abs_rel"], epoch)

        vram_a = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        print(
            f"epoch {epoch:03d} | train_l1={train_l1:.4f} m | "
            f"val_l1={metrics['l1']:.4f} m | val_abs_rel={metrics['abs_rel']:.4f} | "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB",
            flush=True,
        )

        if metrics["l1"] < best:
            best = metrics["l1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_l1": best,
                },
                out,
            )
            print(f"  -> saved {out} with val_l1={best:.4f} m", flush=True)

    writer.close()
    print(f"\nDone. Best val L1: {best:.4f} m", flush=True)


if __name__ == "__main__":
    main()
