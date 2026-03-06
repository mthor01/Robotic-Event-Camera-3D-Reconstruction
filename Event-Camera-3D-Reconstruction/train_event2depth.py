"""
Train UNet for Event-to-Depth prediction.

Uses event frames as input and RealSense depth as ground truth.
Supports the new multi-object folder structure from synchronized_recording.py.

Usage:
    # Single object (5 consecutive frames as input):
    python train_event2depth.py --data_dir data/red_cube --num_frames 5

    # Multiple objects:
    python train_event2depth.py --data_dir data/red_cube data/blue_sphere --num_frames 5

    # All objects in data folder:
    python train_event2depth.py --data_root data --all_objects --num_frames 5

    # With frame skipping (use every 3rd frame, spanning 12 frames total):
    python train_event2depth.py --all_objects --num_frames 5 --frame_skip 2

    # With stereo event cameras (10 input channels = 5 frames x 2 cameras):
    python train_event2depth.py --all_objects --num_frames 5 --stereo

TensorBoard:
    python -m tensorboard.main --logdir checkpoints_event2depth/runs
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ================= DEFAULT PATHS =================
DATA_ROOT = Path("data")
DEFAULT_OUT_DIR = Path("checkpoints_event2depth")
# =================================================


# -----------------------------
# UNet Architecture
# -----------------------------
class DoubleConv(nn.Module):
    """Two consecutive conv-batchnorm-relu blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Standard UNet encoder-decoder with skip connections.
    
    Args:
        in_channels: Number of input channels (1 for single event camera, 2 for stereo)
        base: Base number of filters (doubled at each encoder level)
        out_channels: Number of output channels (1 for depth)
    """
    def __init__(self, in_channels: int = 1, base: int = 32, out_channels: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._match_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.out(d1)

    def _match_and_cat(self, upsampled, skip):
        """Handle potential size mismatch due to odd dimensions."""
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([upsampled, skip], dim=1)


# -----------------------------
# Loss Functions
# -----------------------------
def masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss computed only on valid (masked) pixels."""
    diff = torch.abs(pred - gt) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_mse_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss computed only on valid (masked) pixels."""
    diff = ((pred - gt) ** 2) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_charbonnier(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (smooth L1 variant) on valid pixels."""
    diff = (pred - gt) * mask
    loss = torch.sqrt(diff * diff + eps * eps)
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def edge_aware_smoothness(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness loss using reference image gradients."""
    def grad_x(t): return t[:, :, :, 1:] - t[:, :, :, :-1]
    def grad_y(t): return t[:, :, 1:, :] - t[:, :, :-1, :]

    pred_masked = pred * mask
    ref_masked = ref * mask

    px, py = grad_x(pred_masked), grad_y(pred_masked)
    rx, ry = torch.abs(grad_x(ref_masked)), torch.abs(grad_y(ref_masked))

    wx = torch.exp(-rx)
    wy = torch.exp(-ry)

    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = (torch.abs(px) * wx * mx).sum() / mx.sum().clamp_min(1.0)
    loss_y = (torch.abs(py) * wy * my).sum() / my.sum().clamp_min(1.0)
    return loss_x + loss_y


def scale_invariant_log_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al.)."""
    # Clamp to avoid log(0)
    pred_log = torch.log(pred.clamp_min(1e-6)) * mask
    gt_log = torch.log(gt.clamp_min(1e-6)) * mask
    
    diff = pred_log - gt_log
    n_valid = mask.sum().clamp_min(1.0)
    
    term1 = (diff ** 2).sum() / n_valid
    term2 = alpha * (diff.sum() ** 2) / (n_valid ** 2)
    
    return term1 - term2


# -----------------------------
# Dataset Configuration
# -----------------------------
@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    offset: int = 0  # event_index = depth_index + offset
    num_frames: int = 1  # Number of consecutive event frames to stack as input
    frame_skip: int = 0  # Frames to skip between each input frame (0 = consecutive)
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) for center crop
    resize_to_depth: bool = True  # Resize event frames to match depth resolution
    depth_scale: float = 0.001  # Convert depth units (e.g., mm -> meters)
    clamp_depth: Optional[Tuple[float, float]] = (0.1, 10.0)  # Valid depth range (meters)
    normalize_depth: bool = True  # Normalize depth to [0, 1] using clamp range
    event_norm: str = "01"  # "01" for /255, "none" for raw
    event_log1p: bool = False  # Apply log1p to event frames
    use_stereo: bool = False  # Use both event cameras (2 channels)
    augment: bool = True  # Apply data augmentation during training


class SingleObjectDataset(Dataset):
    """
    Dataset for a single recorded object.
    
    Expected folder structure:
        object_dir/
            hdf5/
                synchronized_recording.h5  (contains realsense/depth, poses/*)
            raw_event_data/
                events_cam0.raw
                events_cam1.raw
    
    Event frames must be pre-generated using raw_to_frames.py into:
        object_dir/
            hdf5/
                events_cam0.h5
                events_cam1.h5
    """
    def __init__(
        self,
        object_dir: str,
        cfg: DataConfig,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.object_dir = Path(object_dir)
        self.cfg = cfg
        self.split = split
        
        # Paths
        self.hdf5_dir = self.object_dir / "hdf5"
        self.rs_h5_path = self.hdf5_dir / "synchronized_recording.h5"
        self.ev0_h5_path = self.hdf5_dir / "events_cam0.h5"
        self.ev1_h5_path = self.hdf5_dir / "events_cam1.h5"
        
        # Verify files exist
        if not self.rs_h5_path.exists():
            raise FileNotFoundError(f"RealSense HDF5 not found: {self.rs_h5_path}")
        if not self.ev0_h5_path.exists():
            raise FileNotFoundError(
                f"Event camera 0 HDF5 not found: {self.ev0_h5_path}\n"
                f"Run raw_to_frames.py first to convert .raw files to .h5"
            )
        
        # Load metadata
        with h5py.File(self.rs_h5_path, "r") as h5r:
            self.n_depth = h5r["realsense/depth"].shape[0]
            self.depth_h = h5r["realsense/depth"].shape[1]
            self.depth_w = h5r["realsense/depth"].shape[2]
        
        with h5py.File(self.ev0_h5_path, "r") as h5e:
            self.n_events = h5e["events/frames"].shape[0]
            self.event_h = h5e["events/frames"].shape[1]
            self.event_w = h5e["events/frames"].shape[2]
        
        if cfg.use_stereo and self.ev1_h5_path.exists():
            self.use_stereo = True
        else:
            self.use_stereo = False
        
        # Compute valid indices (frames that have both event and depth data)
        self._compute_valid_indices(val_ratio, seed)
        
        print(f"[{self.object_dir.name}] {split}: {len(self.indices)} samples "
              f"(depth: {self.n_depth}, events: {self.n_events})")

    def _compute_valid_indices(self, val_ratio: float, seed: int):
        """Compute valid depth indices that have paired event frames."""
        offset = self.cfg.offset
        num_frames = self.cfg.num_frames
        frame_skip = self.cfg.frame_skip
        
        # Calculate how many event frames we need before the target frame
        # For num_frames=5, frame_skip=2: we need frames at t-8, t-6, t-4, t-2, t
        # That's (num_frames - 1) * (frame_skip + 1) frames before target
        lookback = (num_frames - 1) * (frame_skip + 1) if num_frames > 1 else 0
        
        # Valid depth range considering multi-frame lookback
        if offset >= 0:
            # event_idx = depth_idx + offset
            # We need event frames from (event_idx - lookback) to event_idx
            # So event_idx - lookback >= 0, meaning event_idx >= lookback
            # And event_idx < n_events, meaning depth_idx + offset < n_events
            depth_min = max(0, lookback - offset) if offset < lookback else 0
            depth_max = min(self.n_depth - 1, self.n_events - 1 - offset)
        else:
            depth_min = max(0, -offset, lookback - offset)
            depth_max = self.n_depth - 1
        
        if depth_max < depth_min:
            raise RuntimeError(
                f"No overlap between depth ({self.n_depth}) and events ({self.n_events}) "
                f"with offset={offset}, num_frames={num_frames}, frame_skip={frame_skip}"
            )
        
        all_indices = np.arange(depth_min, depth_max + 1, dtype=np.int64)
        n_total = len(all_indices)
        
        # Block-based split (keeps neighboring frames together)
        rng = np.random.default_rng(seed)
        block_size = 10
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)
        
        n_val = int(round(n_total * val_ratio))
        val_mask = np.zeros(n_total, dtype=bool)
        val_count = 0
        
        for b in block_ids:
            if val_count >= n_val:
                break
            start = b * block_size
            end = min((b + 1) * block_size, n_total)
            val_mask[start:end] = True
            val_count += end - start
        
        if self.split == "train":
            self.indices = all_indices[~val_mask]
        else:
            self.indices = all_indices[val_mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        depth_idx = int(self.indices[i])
        event_idx = depth_idx + self.cfg.offset
        num_frames = self.cfg.num_frames
        frame_skip = self.cfg.frame_skip
        step = frame_skip + 1  # actual step between frames
        
        # Compute event frame indices to load
        # For num_frames=5, step=3: load frames at [event_idx-12, event_idx-9, event_idx-6, event_idx-3, event_idx]
        event_indices = [event_idx - (num_frames - 1 - k) * step for k in range(num_frames)]
        
        # Load depth
        with h5py.File(self.rs_h5_path, "r") as h5r:
            depth = h5r["realsense/depth"][depth_idx].astype(np.float32)
        
        # Load event frames (multiple consecutive frames)
        ev0_frames = []
        with h5py.File(self.ev0_h5_path, "r") as h5e:
            for idx in event_indices:
                ev0_frames.append(h5e["events/frames"][idx].astype(np.float32))
        
        ev1_frames = []
        if self.use_stereo:
            with h5py.File(self.ev1_h5_path, "r") as h5e:
                for idx in event_indices:
                    ev1_frames.append(h5e["events/frames"][idx].astype(np.float32))
        
        # Process depth
        depth = depth * self.cfg.depth_scale
        mask = (depth > 0).astype(np.float32)
        
        if self.cfg.clamp_depth is not None:
            dmin, dmax = self.cfg.clamp_depth
            depth = np.clip(depth, dmin, dmax)
            if self.cfg.normalize_depth:
                depth = (depth - dmin) / (dmax - dmin)
        
        # Process events
        if self.cfg.event_norm == "01":
            ev0_frames = [f / 255.0 for f in ev0_frames]
            if self.use_stereo:
                ev1_frames = [f / 255.0 for f in ev1_frames]
        
        if self.cfg.event_log1p:
            ev0_frames = [np.log1p(f) for f in ev0_frames]
            if self.use_stereo:
                ev1_frames = [np.log1p(f) for f in ev1_frames]
        
        # Resize events to depth resolution
        if self.cfg.resize_to_depth and (ev0_frames[0].shape[0] != depth.shape[0] or ev0_frames[0].shape[1] != depth.shape[1]):
            resized_ev0 = []
            for f in ev0_frames:
                f_t = torch.from_numpy(f)[None, None]
                f_t = F.interpolate(f_t, size=(depth.shape[0], depth.shape[1]), mode='bilinear', align_corners=False)
                resized_ev0.append(f_t[0, 0].numpy())
            ev0_frames = resized_ev0
            
            if self.use_stereo:
                resized_ev1 = []
                for f in ev1_frames:
                    f_t = torch.from_numpy(f)[None, None]
                    f_t = F.interpolate(f_t, size=(depth.shape[0], depth.shape[1]), mode='bilinear', align_corners=False)
                    resized_ev1.append(f_t[0, 0].numpy())
                ev1_frames = resized_ev1
        
        # Stack frames into channels
        # For mono: (num_frames, H, W)
        # For stereo: (num_frames * 2, H, W) - interleaved [cam0_t0, cam1_t0, cam0_t1, cam1_t1, ...]
        if self.use_stereo:
            # Interleave: [ev0_0, ev1_0, ev0_1, ev1_1, ...]
            events_list = []
            for ev0, ev1 in zip(ev0_frames, ev1_frames):
                events_list.append(ev0)
                events_list.append(ev1)
            events = np.stack(events_list, axis=0)  # (num_frames * 2, H, W)
        else:
            events = np.stack(ev0_frames, axis=0)  # (num_frames, H, W)
        
        depth = depth[None]  # (1, H, W)
        mask = mask[None]  # (1, H, W)
        
        # Optional center crop
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            events = self._center_crop(events, ch, cw)
            depth = self._center_crop(depth, ch, cw)
            mask = self._center_crop(mask, ch, cw)
        
        # Data augmentation (training only)
        if self.cfg.augment and self.split == "train":
            events, depth, mask = self._augment(events, depth, mask)
        
        return (
            torch.from_numpy(events).float(),
            torch.from_numpy(depth).float(),
            torch.from_numpy(mask).float(),
        )

    def _center_crop(self, x: np.ndarray, h: int, w: int) -> np.ndarray:
        _, oh, ow = x.shape
        y0 = (oh - h) // 2
        x0 = (ow - w) // 2
        return x[:, y0:y0+h, x0:x0+w]

    def _augment(self, events: np.ndarray, depth: np.ndarray, mask: np.ndarray):
        """Simple augmentation: random horizontal flip."""
        if np.random.rand() > 0.5:
            events = np.flip(events, axis=2).copy()
            depth = np.flip(depth, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return events, depth, mask


def create_multi_object_dataset(
    object_dirs: List[str],
    cfg: DataConfig,
    split: str = "train",
    val_ratio: float = 0.1,
) -> Dataset:
    """Create a concatenated dataset from multiple object directories."""
    datasets = []
    for obj_dir in object_dirs:
        try:
            ds = SingleObjectDataset(obj_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {obj_dir}: {e}")
    
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    
    return ConcatDataset(datasets)


def find_object_dirs(data_root: Path) -> List[Path]:
    """Find all valid object directories in data root."""
    object_dirs = []
    for d in data_root.iterdir():
        if d.is_dir():
            # Check if it has the expected structure
            rs_h5 = d / "hdf5" / "synchronized_recording.h5"
            ev_h5 = d / "hdf5" / "events_cam0.h5"
            if rs_h5.exists() and ev_h5.exists():
                object_dirs.append(d)
    return sorted(object_dirs)


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: str = "charbonnier",
    lambda_smooth: float = 0.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for events, depth, mask in loader:
        events = events.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        pred = model(events)
        
        # Compute loss
        if loss_fn == "l1":
            loss = masked_l1_loss(pred, depth, mask)
        elif loss_fn == "mse":
            loss = masked_mse_loss(pred, depth, mask)
        elif loss_fn == "charbonnier":
            loss = masked_charbonnier(pred, depth, mask)
        elif loss_fn == "si_log":
            loss = scale_invariant_log_loss(pred, depth, mask)
        else:
            loss = masked_charbonnier(pred, depth, mask)
        
        # Optional smoothness regularization
        if lambda_smooth > 0:
            loss = loss + lambda_smooth * edge_aware_smoothness(pred, events[:, :1], mask)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate and compute metrics."""
    model.eval()
    
    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0
    
    for events, depth, mask in loader:
        events = events.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        pred = model(events)
        
        total_l1 += masked_l1_loss(pred, depth, mask).item()
        total_mse += masked_mse_loss(pred, depth, mask).item()
        total_samples += 1
    
    n = max(1, total_samples)
    return {
        "l1": total_l1 / n,
        "mse": total_mse / n,
        "rmse": np.sqrt(total_mse / n),
    }


def log_images(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_images: int = 4,
):
    """Log sample predictions to TensorBoard."""
    model.eval()
    
    def normalize_for_display(x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x
    
    with torch.no_grad():
        for i, (events, depth, mask) in enumerate(loader):
            if i >= 1:  # Just one batch
                break
            
            events = events.to(device)
            depth = depth.to(device)
            mask = mask.to(device)
            
            pred = model(events)
            
            for j in range(min(max_images, events.shape[0])):
                ev_img = normalize_for_display(events[j, 0:1])
                gt_img = normalize_for_display(depth[j])
                pred_img = normalize_for_display(pred[j])
                err_img = normalize_for_display(torch.abs(pred[j] - depth[j]) * mask[j])
                
                writer.add_image(f"sample_{j}/input_events", ev_img, epoch)
                writer.add_image(f"sample_{j}/gt_depth", gt_img, epoch)
                writer.add_image(f"sample_{j}/pred_depth", pred_img, epoch)
                writer.add_image(f"sample_{j}/error", err_img, epoch)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train UNet for event-to-depth prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", nargs="+", type=str, default=None,
                           help="Object directory(ies) to train on")
    data_group.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                           help="Root data directory")
    data_group.add_argument("--all_objects", action="store_true",
                           help="Use all objects found in data_root")
    data_group.add_argument("--val_ratio", type=float, default=0.1,
                           help="Fraction of data for validation")
    
    # Preprocessing
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument("--offset", type=int, default=0,
                              help="Frame offset: event_idx = depth_idx + offset")
    preproc_group.add_argument("--num_frames", type=int, default=5,
                              help="Number of consecutive event frames to use as input")
    preproc_group.add_argument("--frame_skip", type=int, default=0,
                              help="Frames to skip between input frames (0=consecutive)")
    preproc_group.add_argument("--depth_scale", type=float, default=0.001,
                              help="Depth scale factor (e.g., 0.001 for mm->m)")
    preproc_group.add_argument("--clamp_min", type=float, default=0.1,
                              help="Minimum valid depth (meters)")
    preproc_group.add_argument("--clamp_max", type=float, default=10.0,
                              help="Maximum valid depth (meters)")
    preproc_group.add_argument("--no_normalize_depth", action="store_true",
                              help="Don't normalize depth to [0,1]")
    preproc_group.add_argument("--stereo", action="store_true",
                              help="Use both event cameras (2 input channels)")
    preproc_group.add_argument("--no_augment", action="store_true",
                              help="Disable data augmentation")
    preproc_group.add_argument("--crop_h", type=int, default=0)
    preproc_group.add_argument("--crop_w", type=int, default=0)
    
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--base", type=int, default=32,
                            help="Base number of filters in UNet")
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--batch", type=int, default=8)
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-4)
    train_group.add_argument("--loss", type=str, default="charbonnier",
                            choices=["l1", "mse", "charbonnier", "si_log"])
    train_group.add_argument("--lambda_smooth", type=float, default=0.0,
                            help="Weight for edge-aware smoothness loss")
    train_group.add_argument("--num_workers", type=int, default=4)
    
    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    out_group.add_argument("--save_every", type=int, default=10,
                          help="Save checkpoint every N epochs")
    out_group.add_argument("--resume", type=str, default=None,
                          help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Determine object directories
    if args.all_objects:
        object_dirs = find_object_dirs(Path(args.data_root))
        if not object_dirs:
            print(f"No valid object directories found in {args.data_root}")
            print("Expected structure: data/<object>/hdf5/synchronized_recording.h5")
            print("                    data/<object>/hdf5/events_cam0.h5")
            print("\nRun raw_to_frames.py on each object's raw event data first.")
            return
        print(f"Found {len(object_dirs)} objects: {[d.name for d in object_dirs]}")
    elif args.data_dir:
        object_dirs = [Path(d) for d in args.data_dir]
    else:
        print("Error: Specify --data_dir or --all_objects")
        return
    
    # Create config
    crop_size = None
    if args.crop_h > 0 and args.crop_w > 0:
        crop_size = (args.crop_h, args.crop_w)
    
    cfg = DataConfig(
        offset=args.offset,
        num_frames=args.num_frames,
        frame_skip=args.frame_skip,
        crop_size=crop_size,
        resize_to_depth=True,
        depth_scale=args.depth_scale,
        clamp_depth=(args.clamp_min, args.clamp_max),
        normalize_depth=not args.no_normalize_depth,
        event_norm="01",
        event_log1p=False,
        use_stereo=args.stereo,
        augment=not args.no_augment,
    )
    
    # Create datasets
    train_ds = create_multi_object_dataset(
        [str(d) for d in object_dirs], cfg, split="train", val_ratio=args.val_ratio
    )
    val_ds = create_multi_object_dataset(
        [str(d) for d in object_dirs], cfg, split="val", val_ratio=args.val_ratio
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
    
    # Calculate input channels: num_frames * cameras_per_frame
    cameras_per_frame = 2 if args.stereo else 1
    in_channels = args.num_frames * cameras_per_frame
    print(f"Input: {args.num_frames} frames x {cameras_per_frame} camera(s) = {in_channels} channels")
    if args.frame_skip > 0:
        print(f"Frame skip: {args.frame_skip} (temporal span: {(args.num_frames - 1) * (args.frame_skip + 1)} frames)")
    
    model = UNet(in_channels=in_channels, base=args.base, out_channels=1).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 1
    best_val_loss = float("inf")
    
    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")
    
    # TensorBoard
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.out_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_fn=args.loss, lambda_smooth=args.lambda_smooth,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        
        # Logging
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val_l1", val_metrics["l1"], epoch)
        writer.add_scalar("loss/val_rmse", val_metrics["rmse"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"Epoch {epoch:03d} | train: {train_loss:.5f} | val L1: {val_metrics['l1']:.5f} | val RMSE: {val_metrics['rmse']:.5f}")
        
        # Save best model
        if val_metrics["l1"] < best_val_loss:
            best_val_loss = val_metrics["l1"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> Saved best model (val L1: {best_val_loss:.5f})")
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
        
        # Log images
        if epoch % 5 == 0 or epoch == 1:
            log_images(writer, model, val_loader, device, epoch)
    
    writer.close()
    print(f"\nTraining complete! Best val L1: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
