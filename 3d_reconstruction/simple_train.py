"""
Train a simple (non-recurrent) UNet for Event-to-Depth prediction.

Same architecture size as E2Depth (real_train.py) but with plain Conv layers
instead of ConvLSTM — no hidden state, no sequence unrolling.  Cheaper to
train and easier to debug.  Drop-in replacement: same data pipeline, same
loss, same CLI flags (minus --seq_len which is always 1).

Usage:
    python3 simple_train.py --data_root data/real
    python3 simple_train.py --data_dir data/real/box --use_pose
    python3 simple_train.py --data_dir data/real/box --spatial_mask --use_pose
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import h5py

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

from reconstruction_config import (
    D_MAX, ALPHA, DEPTH_MIN, WHITE_THRESH, NUM_BINS,
    DATA_ROOT as _DATA_ROOT, DEFAULT_OUT_DIR,
)

DATA_ROOT = _DATA_ROOT


# ═══════════════════════════════════════════════════════════════════
#  Model — plain UNet (no recurrence)
# ═══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class EncoderBlock(nn.Module):
    """Downsample conv (stride-2, k=5) + refine conv (k=3) — mirrors E2Depth encoder size."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Bilinear upsample + concat skip + conv (k=5) + conv (k=3)."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SimpleUNet(nn.Module):
    """
    Stateless UNet with the same channel widths as E2DepthNet.

    base=32, num_encoders=3 → encoder channels: 64, 128, 256
    bottleneck: 256
    decoder: 128, 64, 32
    """
    def __init__(
        self,
        in_channels: int = 5,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.num_encoders = num_encoders

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Encoders
        self.encoders = nn.ModuleList()
        ch = base
        for _ in range(num_encoders):
            self.encoders.append(EncoderBlock(ch, ch * 2))
            ch *= 2

        # Bottleneck
        self.residuals = nn.ModuleList([ResidualBlock(ch) for _ in range(num_residuals)])

        # Decoders (skip channels mirror encoder out-channels in reverse)
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            self.decoders.append(DecoderBlock(ch, skip_ch, ch // 2))
            ch //= 2

        # Prediction head
        self.pred = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            depth: (B, 1, H, W) in [0, 1]
        """
        x = self.head(x)

        skips = [x]
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < self.num_encoders - 1:
                skips.append(x)

        for res in self.residuals:
            x = res(x)

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])

        return self.pred(x)


# ═══════════════════════════════════════════════════════════════════
#  Loss functions  (identical to real_train.py)
# ═══════════════════════════════════════════════════════════════════

def scale_invariant_loss(pred, gt, mask):
    diff = (pred - gt) * mask
    n = mask.sum().clamp_min(1.0)
    return (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)


def multi_scale_gradient_loss(pred, gt, mask, num_scales=4):
    def gx(t): return t[:, :, :, 1:] - t[:, :, :, :-1]
    def gy(t): return t[:, :, 1:, :] - t[:, :, :-1, :]

    total = 0.0
    for scale in range(num_scales):
        if scale > 0:
            pred = F.avg_pool2d(pred, 2)
            gt   = F.avg_pool2d(gt,   2)
            mask = F.avg_pool2d(mask, 2)
            mask = (mask > 0.5).float()
        r = pred - gt
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        total = total + (torch.abs(gx(r)) * mx).sum() / mx.sum().clamp_min(1.0)
        total = total + (torch.abs(gy(r)) * my).sum() / my.sum().clamp_min(1.0)
    return total / num_scales


def e2depth_loss(pred, gt, mask, lambda_grad=0.5, lambda_mean=0.2):
    l_si   = scale_invariant_loss(pred, gt, mask)
    l_grad = multi_scale_gradient_loss(pred, gt, mask)
    n = mask.sum().clamp_min(1.0)
    l_mean = (((pred - gt) * mask).sum() / n) ** 2
    total  = l_si + lambda_grad * l_grad + lambda_mean * l_mean
    return total, {"si": l_si.item(), "grad": l_grad.item(),
                   "mean": l_mean.item(), "total": total.item()}


# ═══════════════════════════════════════════════════════════════════
#  Depth conversion utilities  (identical to real_train.py)
# ═══════════════════════════════════════════════════════════════════

def log_normalized_to_depth(pred, d_max=D_MAX, alpha=ALPHA):
    return d_max * torch.exp(-alpha * (1.0 - pred))

def linear_normalized_to_depth(pred, d_min=DEPTH_MIN, d_max=D_MAX):
    return pred * (d_max - d_min) + d_min


# ═══════════════════════════════════════════════════════════════════
#  Dataset  (identical to real_train.py — seq_len always forced to 1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    seq_len: int = 1
    crop_size: Optional[Tuple[int, int]] = None
    resize_hw: Optional[Tuple[int, int]] = None
    depth_max: float = D_MAX
    depth_min: float = DEPTH_MIN
    augment: bool = True
    num_bins: int = NUM_BINS
    use_pose: bool = False
    rgb_mask: bool = False
    spatial_mask: bool = False
    log_depth: bool = False


class RealDataset(Dataset):
    def __init__(self, sequence_dir, cfg: DataConfig, split="train",
                 val_ratio=0.1, seed=42):
        super().__init__()
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.split = split

        projected = self.sequence_dir / "hdf5" / "depth_in_event_frame.h5"
        if projected.exists():
            self.depth_h5_path = projected
            self._depth_key = "depth"
            self._depth_is_metric = True
        else:
            self.depth_h5_path = self.sequence_dir / "hdf5" / "realsense.h5"
            self._depth_key = "depth"
            self._depth_is_metric = False
        self._ts_h5_path = self.sequence_dir / "hdf5" / "realsense.h5"
        self._ts_key = "t_sys_ns"

        self.use_pose = cfg.use_pose
        if self.use_pose:
            pose_vox = self.sequence_dir / "events" / "voxels_pose_cam0"
            if pose_vox.exists():
                self.voxels_dir = pose_vox
            else:
                raise FileNotFoundError(
                    f"--use_pose requires precomputed pose voxels. "
                    f"Run: python precompute_pose_depth.py --data_dir {self.sequence_dir}"
                )
        elif (self.sequence_dir / "events" / "voxels_cam0").exists():
            self.voxels_dir = self.sequence_dir / "events" / "voxels_cam0"
        else:
            self.voxels_dir = self.sequence_dir / "events" / "voxels"

        if not self.depth_h5_path.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5_path}")

        self.voxel_files = sorted(self.voxels_dir.glob("voxel_*.npy")) if self.voxels_dir.exists() else []
        if not self.voxel_files:
            raise FileNotFoundError(f"No precomputed voxels in {self.voxels_dir}")

        with h5py.File(self.depth_h5_path, "r") as f:
            self.n_frames = f[self._depth_key].shape[0]
            self.H = f[self._depth_key].shape[1]
            self.W = f[self._depth_key].shape[2]
        with h5py.File(self._ts_h5_path, "r") as f:
            self.depth_timestamps = f[self._ts_key][:] // 1000

        n_voxels = len(self.voxel_files)
        if n_voxels != self.n_frames:
            print(f"Warning: {n_voxels} voxels != {self.n_frames} frames")
            self.n_frames = min(n_voxels, self.n_frames)

        self.rgb_mask = cfg.rgb_mask
        self._rgb_h5_path = None
        if self.rgb_mask:
            rgb_proj = self.sequence_dir / "hdf5" / "rgb_in_event_frame.h5"
            if not rgb_proj.exists():
                raise FileNotFoundError(
                    f"--rgb_mask requires projected RGB. "
                    f"Run: python project_realsense_to_event.py --data_dir {self.sequence_dir}"
                )
            self._rgb_h5_path = rgb_proj

        self.spatial_mask = cfg.spatial_mask
        self._spatial_h5_path = None
        if self.spatial_mask:
            sp = self.sequence_dir / "hdf5" / "spatial_mask.h5"
            if not sp.exists():
                raise FileNotFoundError(
                    f"--spatial_mask requires precomputed masks. "
                    f"Run: python precompute_spatial_mask.py --data_dir {self.sequence_dir}"
                )
            self._spatial_h5_path = sp

        self._compute_valid_indices(val_ratio, seed)

        depth_src = "projected" if self._depth_is_metric else "raw realsense"
        depth_enc = "log" if cfg.log_depth else "linear"
        pose_str = "with pose" if self.use_pose else "no pose"
        rgb_str = ", rgb_mask" if self.rgb_mask else ""
        spatial_str = ", spatial_mask" if self.spatial_mask else ""
        print(f"[{self.sequence_dir.name}] {split}: {len(self.indices)} samples "
              f"(frames: {self.n_frames}, depth: {depth_src} [{depth_enc}], "
              f"{pose_str}{rgb_str}{spatial_str}, res: {self.W}x{self.H})")

    def _compute_valid_indices(self, val_ratio, seed):
        idx_max = self.n_frames - 1
        all_indices = np.arange(0, idx_max + 1, dtype=np.int64)
        n_total = len(all_indices)
        rng = np.random.default_rng(seed)
        block_size = max(10, 2)
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val = int(round(n_total * val_ratio))
        val_mask = np.zeros(n_total, dtype=bool)
        val_count = 0
        for bid in block_ids:
            start = bid * block_size
            end = min(start + block_size, n_total)
            block_len = end - start
            if val_count < n_val:
                val_mask[start:end] = True
                val_count += block_len
        if self.split == "val":
            self.indices = all_indices[val_mask]
        else:
            self.indices = all_indices[~val_mask]

    def __len__(self):
        return len(self.indices)

    def _get_voxel(self, idx):
        path = self.voxels_dir / f"voxel_{idx:06d}.npy"
        voxel = np.load(path).astype(np.float32)
        if voxel.ndim == 2:
            voxel = voxel[None]
        return voxel

    def _get_depth(self, idx):
        with h5py.File(self.depth_h5_path, "r") as f:
            depth = f[self._depth_key][idx].astype(np.float32)
        if not self._depth_is_metric:
            depth = depth / 1000.0
        return depth

    def __getitem__(self, i):
        idx = int(self.indices[i])

        voxel = self._get_voxel(idx)
        depth = self._get_depth(idx)

        mask = ((depth > self.cfg.depth_min) & (depth < self.cfg.depth_max)).astype(np.float32)

        if self.rgb_mask:
            with h5py.File(self._rgb_h5_path, "r") as rf:
                rgb = rf["rgb"][idx]
            white = np.all(rgb > WHITE_THRESH, axis=-1)
            no_rgb = np.all(rgb == 0, axis=-1)
            mask[white | no_rgb] = 0.0

        if self.spatial_mask:
            with h5py.File(self._spatial_h5_path, "r") as sf:
                sp = sf["mask"][idx]
            mask[sp == 0] = 0.0

        depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
        if self.cfg.log_depth:
            depth = 1.0 + (1.0 / ALPHA) * np.log(depth / self.cfg.depth_max)
        else:
            depth = (depth - self.cfg.depth_min) / (self.cfg.depth_max - self.cfg.depth_min)
        depth = np.clip(depth, 0, 1)

        # Shape: (C, H, W) and (1, H, W)
        events = voxel
        depth  = depth[None]
        mask   = mask[None]

        # Resize
        if self.cfg.resize_hw is not None:
            rh, rw = self.cfg.resize_hw
            events = F.interpolate(torch.from_numpy(events[None]), size=(rh, rw), mode="bilinear", align_corners=False).numpy()[0]
            depth  = F.interpolate(torch.from_numpy(depth[None]),  size=(rh, rw), mode="bilinear", align_corners=False).numpy()[0]
            mask   = F.interpolate(torch.from_numpy(mask[None]),   size=(rh, rw), mode="nearest").numpy()[0]

        # Center crop
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            cur_H, cur_W = events.shape[1], events.shape[2]
            y0 = (cur_H - ch) // 2
            x0 = (cur_W - cw) // 2
            events = events[:, y0:y0+ch, x0:x0+cw]
            depth  = depth[:,  y0:y0+ch, x0:x0+cw]
            mask   = mask[:,   y0:y0+ch, x0:x0+cw]

        # Horizontal flip (train only)
        if self.cfg.augment and self.split == "train" and np.random.rand() > 0.5:
            events = np.flip(events, axis=2).copy()
            depth  = np.flip(depth,  axis=2).copy()
            mask   = np.flip(mask,   axis=2).copy()

        return (
            torch.from_numpy(events).float(),
            torch.from_numpy(depth).float(),
            torch.from_numpy(mask).float(),
        )


def create_multi_sequence_dataset(sequence_dirs, cfg, split="train", val_ratio=0.1):
    datasets = []
    for seq_dir in sequence_dirs:
        try:
            ds = RealDataset(seq_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir}: {e}")
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root):
    dirs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        has_depth = (d / "hdf5" / "depth_in_event_frame.h5").exists() or (d / "hdf5" / "realsense.h5").exists()
        has_voxels = (d / "events" / "voxels_cam0").exists() or (d / "events" / "voxels").exists()
        if has_depth and has_voxels:
            dirs.append(d)
    return dirs


# ═══════════════════════════════════════════════════════════════════
#  Training utilities
# ═══════════════════════════════════════════════════════════════════

def get_gpu_stats(device):
    stats = {}
    if device.type != "cuda":
        return stats
    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
    stats["vram_used_mb"]     = torch.cuda.memory_allocated(gpu_idx) / 1024 ** 2
    stats["vram_reserved_mb"] = torch.cuda.memory_reserved(gpu_idx) / 1024 ** 2
    if _NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        stats["gpu_util_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    return stats


def train_one_epoch(model, loader, optimizer, device, lambda_grad=0.5, epoch=0, log_interval=10.0):
    model.train()
    total_loss = total_si = total_grad = total_mean = 0.0
    n_batches = 0
    n_total = len(loader)
    last_log = epoch_start = time.time()

    for events, depths, masks in loader:
        # events: (B, C, H, W)  depths/masks: (B, 1, H, W)
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        pred = model(events)
        loss, metrics = e2depth_loss(pred, depths, masks, lambda_grad)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_si   += metrics["si"]
        total_grad += metrics["grad"]
        total_mean += metrics["mean"]
        n_batches  += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - epoch_start
            bps = n_batches / elapsed if elapsed > 0 else 0
            eta = (n_total - n_batches) / bps if bps > 0 else 0
            avg = total_loss / n_batches
            print(f"  [Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                  f"| loss: {avg:.5f} | {bps:.1f} batch/s "
                  f"| ETA: {int(eta//60):02d}:{int(eta%60):02d}", flush=True)
            last_log = now

    n = max(1, n_batches)
    return {"total": total_loss/n, "si": total_si/n, "grad": total_grad/n, "mean": total_mean/n}


@torch.no_grad()
def validate(model, loader, device, log_depth=False, depth_min=DEPTH_MIN, depth_max=D_MAX):
    model.eval()
    total_si = total_l1 = total_abs_rel = 0.0
    n_batches = 0
    last_pred_mean = last_gt_mean = last_pred_m = last_gt_m = 0.0

    for events, depths, masks in loader:
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        pred = model(events)

        total_si += scale_invariant_loss(pred, depths, masks).item()

        if log_depth:
            pred_m = log_normalized_to_depth(pred)
            gt_m   = log_normalized_to_depth(depths)
        else:
            pred_m = linear_normalized_to_depth(pred, depth_min, depth_max)
            gt_m   = linear_normalized_to_depth(depths, depth_min, depth_max)

        n_valid = masks.sum().clamp_min(1.0)
        diff = torch.abs(pred_m - gt_m) * masks
        total_l1      += (diff.sum() / n_valid).item()
        total_abs_rel += ((diff / gt_m.clamp_min(1e-6)).sum() / n_valid).item()

        last_pred_mean = ((pred * masks).sum() / n_valid).item()
        last_gt_mean   = ((depths * masks).sum() / n_valid).item()
        last_pred_m    = ((pred_m * masks).sum() / n_valid).item()
        last_gt_m      = ((gt_m * masks).sum() / n_valid).item()
        n_batches += 1

    n = max(1, n_batches)
    return {
        "si": total_si / n,
        "l1_metric": total_l1 / n,
        "abs_rel": total_abs_rel / n,
        "pred_log_mean": last_pred_mean,
        "gt_log_mean": last_gt_mean,
        "pred_metric_mean": last_pred_m,
        "gt_metric_mean": last_gt_m,
    }


def _depth_to_rgb(t):
    import matplotlib.cm as cm
    arr = t[0].cpu().float().numpy() if t.dim() == 3 else t.cpu().float().numpy()
    rgba = cm.turbo(arr)
    return torch.from_numpy(rgba[:, :, :3].transpose(2, 0, 1).astype("float32"))


def log_images(writer, model, loader, device, epoch, split="val", n_images=3):
    from torch.utils.data import Subset
    model.eval()
    dataset = loader.dataset
    indices = torch.randperm(len(dataset))[:n_images].tolist()
    sub_loader = DataLoader(Subset(dataset, indices), batch_size=n_images, shuffle=False, num_workers=0)
    with torch.no_grad():
        for events, depths, masks in sub_loader:
            events = events.to(device)
            depths = depths.to(device)
            masks  = masks.to(device)
            pred   = model(events)
            for j in range(events.shape[0]):
                ev_img = events[j]
                ev_img = ev_img[:3] if ev_img.shape[0] >= 3 else ev_img[0:1]
                ev_img = (ev_img - ev_img.min()) / (ev_img.max() - ev_img.min() + 1e-6)
                m = masks[j]
                tag = f"{split}/sample_{j}"
                writer.add_image(f"{tag}/input_events", ev_img, epoch)
                writer.add_image(f"{tag}/gt_depth",    _depth_to_rgb(depths[j] * m), epoch)
                writer.add_image(f"{tag}/pred_depth",  _depth_to_rgb(pred[j]   * m), epoch)
                writer.add_image(f"{tag}/total_mask",  m, epoch)
                err = torch.abs(pred[j] - depths[j]) * m
                writer.add_image(f"{tag}/error", _depth_to_rgb(err / (err.max() + 1e-6)), epoch)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train a simple (non-recurrent) UNet for event-to-depth prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir",  nargs="+", type=str, default=None)
    data_group.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    data_group.add_argument("--val_ratio", type=float, default=0.2)
    data_group.add_argument("--num_bins",  type=int,   default=5)
    data_group.add_argument("--depth_max", type=float, default=D_MAX)
    data_group.add_argument("--depth_min", type=float, default=0.05)
    data_group.add_argument("--use_pose",    action="store_true")
    data_group.add_argument("--rgb_mask",    action="store_true")
    data_group.add_argument("--spatial_mask",action="store_true")
    data_group.add_argument("--log_depth",   action="store_true")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--base",          type=int, default=32)
    model_group.add_argument("--num_encoders",  type=int, default=3)
    model_group.add_argument("--num_residuals", type=int, default=2)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs",      type=int,   default=50)
    train_group.add_argument("--batch",       type=int,   default=12)
    train_group.add_argument("--lr",          type=float, default=1e-4)
    train_group.add_argument("--lambda_grad", type=float, default=0.5)
    train_group.add_argument("--num_workers", type=int,   default=8)
    train_group.add_argument("--crop_h",      type=int,   default=240)
    train_group.add_argument("--crop_w",      type=int,   default=320)
    train_group.add_argument("--resize_h",    type=int,   default=288)
    train_group.add_argument("--resize_w",    type=int,   default=384)

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out_dir",    type=str, default="checkpoints_simple")
    out_group.add_argument("--save_every", type=int, default=10)
    out_group.add_argument("--resume",     type=str, default=None)

    args = parser.parse_args()

    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequences found in {args.data_root}")
            return

    print(f"Found {len(sequence_dirs)} sequences:")
    for d in sequence_dirs:
        print(f"  - {d.name}")

    crop_size = (args.crop_h, args.crop_w) if args.crop_h > 0 and args.crop_w > 0 else None
    resize_hw  = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None

    cfg     = DataConfig(seq_len=1, crop_size=crop_size, resize_hw=resize_hw,
                         depth_max=args.depth_max, depth_min=args.depth_min, augment=True,
                         num_bins=args.num_bins, use_pose=args.use_pose, rgb_mask=args.rgb_mask,
                         spatial_mask=args.spatial_mask, log_depth=args.log_depth)
    cfg_val = DataConfig(seq_len=1, crop_size=crop_size, resize_hw=resize_hw,
                         depth_max=args.depth_max, depth_min=args.depth_min, augment=False,
                         num_bins=args.num_bins, use_pose=args.use_pose, rgb_mask=args.rgb_mask,
                         spatial_mask=args.spatial_mask, log_depth=args.log_depth)

    if len(sequence_dirs) > 1:
        rng = np.random.default_rng(42)
        dirs_shuffled = list(sequence_dirs)
        rng.shuffle(dirs_shuffled)
        n_val_dirs = max(1, int(round(len(dirs_shuffled) * args.val_ratio)))
        val_dirs   = dirs_shuffled[:n_val_dirs]
        train_dirs = dirs_shuffled[n_val_dirs:] or val_dirs
        print(f"  Object-level split: {len(train_dirs)} train, {len(val_dirs)} val")
        train_ds = create_multi_sequence_dataset([str(d) for d in train_dirs], cfg,     split="train", val_ratio=0.0)
        val_ds   = create_multi_sequence_dataset([str(d) for d in val_dirs],   cfg_val, split="val",   val_ratio=1.0)
    else:
        train_ds = create_multi_sequence_dataset([str(d) for d in sequence_dirs], cfg,     split="train", val_ratio=args.val_ratio)
        val_ds   = create_multi_sequence_dataset([str(d) for d in sequence_dirs], cfg_val, split="val",   val_ratio=args.val_ratio)

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Sequences: {len(sequence_dirs)}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Valid samples: {len(val_ds)}")
    print(f"  Depth range: {args.depth_min:.2f}m - {args.depth_max:.2f}m")
    print(f"  Depth encoding: {'log' if args.log_depth else 'linear'}")
    print(f"  Voxel bins: {args.num_bins}")
    print(f"  Use pose: {args.use_pose}")
    print(f"  RGB mask: {args.rgb_mask}")
    print(f"{'='*60}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    in_channels = args.num_bins + (1 if args.use_pose else 0)
    print(f"Input channels: {in_channels} (bins={args.num_bins}, pose={'yes' if args.use_pose else 'no'})")

    model = SimpleUNet(in_channels=in_channels, base=args.base,
                       num_encoders=args.num_encoders,
                       num_residuals=args.num_residuals).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20)

    start_epoch, best_val_loss = 1, float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(args.out_dir, "runs", run_name)
    writer   = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_grad=args.lambda_grad, epoch=epoch)
        val_metrics = validate(model, val_loader, device,
                               log_depth=args.log_depth,
                               depth_min=args.depth_min,
                               depth_max=args.depth_max)
        scheduler.step(val_metrics["si"])

        writer.add_scalar("loss/train_total",    train_metrics["total"], epoch)
        writer.add_scalar("loss/train_si",       train_metrics["si"],    epoch)
        writer.add_scalar("loss/train_grad",     train_metrics["grad"],  epoch)
        writer.add_scalar("loss/train_mean",     train_metrics["mean"],  epoch)
        writer.add_scalar("loss/val_si",         val_metrics["si"],      epoch)
        writer.add_scalar("loss/val_l1_metric",  val_metrics["l1_metric"], epoch)
        writer.add_scalar("loss/val_abs_rel",    val_metrics["abs_rel"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"],         epoch)

        gpu_stats = get_gpu_stats(device)
        if gpu_stats:
            writer.add_scalar("gpu/vram_used_mb",     gpu_stats["vram_used_mb"],     epoch)
            writer.add_scalar("gpu/vram_reserved_mb", gpu_stats["vram_reserved_mb"], epoch)
            if "gpu_util_pct" in gpu_stats:
                writer.add_scalar("gpu/utilization_pct", gpu_stats["gpu_util_pct"], epoch)

        print(f"Epoch {epoch:03d} | train: {train_metrics['total']:.5f} "
              f"| val SI: {val_metrics['si']:.5f} | val L1: {val_metrics['l1_metric']:.3f}m "
              f"| val AbsRel: {val_metrics['abs_rel']:.4f}")
        print(f"          | pred_log: {val_metrics['pred_log_mean']:.3f} vs "
              f"gt_log: {val_metrics['gt_log_mean']:.3f} "
              f"| pred_m: {val_metrics['pred_metric_mean']:.3f}m vs "
              f"gt_m: {val_metrics['gt_metric_mean']:.3f}m")
        if gpu_stats:
            util_str = f" | GPU util: {gpu_stats['gpu_util_pct']:.0f}%" if "gpu_util_pct" in gpu_stats else ""
            print(f"          | VRAM: {gpu_stats['vram_used_mb']:.0f}/{gpu_stats['vram_reserved_mb']:.0f} MB"
                  f" (used/reserved){util_str}")

        if val_metrics["si"] < best_val_loss:
            best_val_loss = val_metrics["si"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "in_channels": in_channels, "base": args.base,
                    "num_encoders": args.num_encoders,
                    "num_residuals": args.num_residuals,
                    "depth_max": args.depth_max, "depth_min": args.depth_min,
                    "use_pose": args.use_pose,
                },
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> Saved best model (val SI: {best_val_loss:.5f})")

        if epoch % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))

        if epoch % 5 == 0 or epoch == 1:
            log_images(writer, model, val_loader,   device, epoch, split="val",   n_images=3)
            log_images(writer, model, train_loader, device, epoch, split="train", n_images=3)

    writer.close()
    print(f"\nTraining complete! Best val SI: {best_val_loss:.5f}")
    print(f"Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
