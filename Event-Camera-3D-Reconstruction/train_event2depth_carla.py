"""
Train UNet for Event-to-Depth prediction using CARLA simulation data.

Uses pre-rendered event frames (PNG) as input and depth maps (NPY) as ground truth.
This script is designed for the test_data folder structure from CARLA simulations.

Expected folder structure:
    test_data/
        train_sequence_00_town01/
            depth/
                data/
                    depth_0000000000.npy
                    depth_0000000001.npy
                    ...
            events/
                frames/
                    frame_0000000000.png
                    frame_0000000001.png
                    ...

Usage:
    # Single sequence:
    python train_event2depth_carla.py --data_dir test_data/train_sequence_00_town01

    # Multiple sequences:
    python train_event2depth_carla.py --data_dir test_data/train_sequence_00_town01 test_data/train_sequence_01_town02

    # All sequences in test_data:
    python train_event2depth_carla.py --data_root test_data --all_sequences

    # With multi-frame input (5 consecutive frames):
    python train_event2depth_carla.py --data_dir test_data/train_sequence_00_town01 --num_frames 5

TensorBoard:
    python -m tensorboard.main --logdir checkpoints_carla/runs
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ================= DEFAULT PATHS =================
DATA_ROOT = Path("test_data")
DEFAULT_OUT_DIR = Path("checkpoints_carla")
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
        in_channels: Number of input channels (3 for RGB event frames, or 3*num_frames for stacked)
        base: Base number of filters (doubled at each encoder level)
        out_channels: Number of output channels (1 for depth)
    """
    def __init__(self, in_channels: int = 3, base: int = 32, out_channels: int = 1):
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
    num_frames: int = 1  # Number of consecutive event frames to stack as input
    frame_skip: int = 0  # Frames to skip between each input frame (0 = consecutive)
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) for center crop
    clamp_depth: Optional[Tuple[float, float]] = (0.1, 100.0)  # Valid depth range (meters)
    normalize_depth: bool = True  # Normalize depth to [0, 1] using clamp range
    grayscale_events: bool = False  # Convert RGB event frames to grayscale
    augment: bool = True  # Apply data augmentation during training


class CarlaSequenceDataset(Dataset):
    """
    Dataset for a single CARLA simulation sequence.
    
    Expected folder structure:
        sequence_dir/
            depth/
                data/
                    depth_0000000000.npy
                    ...
            events/
                frames/
                    frame_0000000000.png
                    ...
    """
    def __init__(
        self,
        sequence_dir: str,
        cfg: DataConfig,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.split = split
        
        # Paths
        self.depth_dir = self.sequence_dir / "depth" / "data"
        self.events_dir = self.sequence_dir / "events" / "frames"
        
        # Verify directories exist
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        if not self.events_dir.exists():
            raise FileNotFoundError(f"Events directory not found: {self.events_dir}")
        
        # Get sorted list of depth files
        self.depth_files = sorted([
            f for f in self.depth_dir.iterdir() 
            if f.suffix == '.npy' and f.name.startswith('depth_')
        ])
        
        # Get sorted list of event frame files
        self.event_files = sorted([
            f for f in self.events_dir.iterdir() 
            if f.suffix == '.png' and f.name.startswith('frame_')
        ])
        
        self.n_depth = len(self.depth_files)
        self.n_events = len(self.event_files)
        
        if self.n_depth == 0:
            raise RuntimeError(f"No depth files found in {self.depth_dir}")
        if self.n_events == 0:
            raise RuntimeError(f"No event frame files found in {self.events_dir}")
        
        # Load one sample to get dimensions
        sample_depth = np.load(self.depth_files[0])
        sample_event = np.array(Image.open(self.event_files[0]))
        self.depth_h, self.depth_w = sample_depth.shape
        self.event_h, self.event_w = sample_event.shape[:2]
        self.event_channels = sample_event.shape[2] if len(sample_event.shape) == 3 else 1
        
        # Compute valid indices
        self._compute_valid_indices(val_ratio, seed)
        
        print(f"[{self.sequence_dir.name}] {split}: {len(self.indices)} samples "
              f"(depth: {self.n_depth}, events: {self.n_events})")

    def _compute_valid_indices(self, val_ratio: float, seed: int):
        """Compute valid indices that have paired event frames."""
        num_frames = self.cfg.num_frames
        frame_skip = self.cfg.frame_skip
        
        # Calculate how many event frames we need before the target frame
        lookback = (num_frames - 1) * (frame_skip + 1) if num_frames > 1 else 0
        
        # Valid range: need enough frames for lookback
        n_paired = min(self.n_depth, self.n_events)
        idx_min = lookback
        idx_max = n_paired - 1
        
        if idx_max < idx_min:
            raise RuntimeError(
                f"Not enough frames for num_frames={num_frames}, frame_skip={frame_skip}. "
                f"Available: {n_paired}, needed lookback: {lookback}"
            )
        
        all_indices = np.arange(idx_min, idx_max + 1, dtype=np.int64)
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
        idx = int(self.indices[i])
        num_frames = self.cfg.num_frames
        frame_skip = self.cfg.frame_skip
        step = frame_skip + 1
        
        # Compute event frame indices to load
        event_indices = [idx - (num_frames - 1 - k) * step for k in range(num_frames)]
        
        # Load depth (NPY file)
        depth = np.load(self.depth_files[idx]).astype(np.float32)
        
        # Load event frames (PNG files)
        event_frames = []
        for ev_idx in event_indices:
            img = Image.open(self.event_files[ev_idx])
            if self.cfg.grayscale_events:
                img = img.convert('L')
            event_frames.append(np.array(img, dtype=np.float32))
        
        # Process depth
        mask = (depth > 0).astype(np.float32)
        
        if self.cfg.clamp_depth is not None:
            dmin, dmax = self.cfg.clamp_depth
            # Mask out invalid depth (e.g., sky at 1000m in CARLA)
            mask = mask * (depth < dmax).astype(np.float32)
            depth = np.clip(depth, dmin, dmax)
            if self.cfg.normalize_depth:
                depth = (depth - dmin) / (dmax - dmin)
        
        # Process events (normalize to [0, 1])
        event_frames = [f / 255.0 for f in event_frames]
        
        # Stack frames into channels
        # For RGB: (num_frames * 3, H, W)
        # For grayscale: (num_frames, H, W)
        if self.cfg.grayscale_events or len(event_frames[0].shape) == 2:
            events = np.stack(event_frames, axis=0)  # (num_frames, H, W)
        else:
            # Stack RGB channels: [R1,G1,B1, R2,G2,B2, ...]
            events = np.concatenate([f.transpose(2, 0, 1) for f in event_frames], axis=0)
        
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


def create_multi_sequence_dataset(
    sequence_dirs: List[str],
    cfg: DataConfig,
    split: str = "train",
    val_ratio: float = 0.1,
) -> Dataset:
    """Create a concatenated dataset from multiple sequence directories."""
    datasets = []
    for seq_dir in sequence_dirs:
        try:
            ds = CarlaSequenceDataset(seq_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir}: {e}")
    
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find all valid sequence directories in data root."""
    sequence_dirs = []
    for d in data_root.iterdir():
        if d.is_dir():
            # Check if it has the expected structure
            depth_dir = d / "depth" / "data"
            events_dir = d / "events" / "frames"
            if depth_dir.exists() and events_dir.exists():
                sequence_dirs.append(d)
    return sorted(sequence_dirs)


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
                # For RGB input, take first 3 channels
                if events.shape[1] >= 3:
                    ev_img = events[j, :3]
                else:
                    ev_img = events[j, 0:1]
                ev_img = normalize_for_display(ev_img)
                
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
        description="Train UNet for event-to-depth prediction on CARLA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", nargs="+", type=str, default=None,
                           help="Sequence directory(ies) to train on")
    data_group.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                           help="Root data directory")
    data_group.add_argument("--all_sequences", action="store_true",
                           help="Use all sequences found in data_root")
    data_group.add_argument("--val_ratio", type=float, default=0.1,
                           help="Fraction of data for validation")
    
    # Preprocessing
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument("--num_frames", type=int, default=1,
                              help="Number of consecutive event frames to use as input")
    preproc_group.add_argument("--frame_skip", type=int, default=0,
                              help="Frames to skip between input frames (0=consecutive)")
    preproc_group.add_argument("--clamp_min", type=float, default=0.1,
                              help="Minimum valid depth (meters)")
    preproc_group.add_argument("--clamp_max", type=float, default=100.0,
                              help="Maximum valid depth (meters)")
    preproc_group.add_argument("--no_normalize_depth", action="store_true",
                              help="Don't normalize depth to [0,1]")
    preproc_group.add_argument("--grayscale", action="store_true",
                              help="Convert RGB event frames to grayscale (1 channel per frame)")
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
    
    # Determine sequence directories
    if args.all_sequences:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequence directories found in {args.data_root}")
            print("Expected structure: data_root/<sequence>/depth/data/*.npy")
            print("                    data_root/<sequence>/events/frames/*.png")
            return
        print(f"Found {len(sequence_dirs)} sequences: {[d.name for d in sequence_dirs]}")
    elif args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        print("Error: Specify --data_dir or --all_sequences")
        return
    
    # Create config
    crop_size = None
    if args.crop_h > 0 and args.crop_w > 0:
        crop_size = (args.crop_h, args.crop_w)
    
    cfg = DataConfig(
        num_frames=args.num_frames,
        frame_skip=args.frame_skip,
        crop_size=crop_size,
        clamp_depth=(args.clamp_min, args.clamp_max),
        normalize_depth=not args.no_normalize_depth,
        grayscale_events=args.grayscale,
        augment=not args.no_augment,
    )
    
    # Create datasets
    train_ds = create_multi_sequence_dataset(
        [str(d) for d in sequence_dirs], cfg, split="train", val_ratio=args.val_ratio
    )
    val_ds = create_multi_sequence_dataset(
        [str(d) for d in sequence_dirs], cfg, split="val", val_ratio=args.val_ratio
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
    
    # Calculate input channels
    if args.grayscale:
        in_channels = args.num_frames
    else:
        in_channels = args.num_frames * 3  # RGB
    
    print(f"Input: {args.num_frames} frames x {'1 (grayscale)' if args.grayscale else '3 (RGB)'} = {in_channels} channels")
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
