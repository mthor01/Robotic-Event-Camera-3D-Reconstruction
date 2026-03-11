"""
Train E2Depth: Recurrent UNet for Event-to-Depth prediction.

Implementation based on:
"Learning Monocular Dense Depth from Events" (Hidalgo-Carrió et al., 3DV 2020)

Key differences from standard UNet:
- ConvLSTM in encoder layers for temporal recurrence
- Residual blocks in bottleneck
- Log depth output with sigmoid activation
- Scale-invariant + multi-scale gradient loss
- Bilinear upsampling in decoder

Expected folder structure (synthetic data from franka_pipeline):
    data/synthetic_data/
        bottle/
            hdf5/
                depth.h5    # realsense/depth (N, H, W) uint16 mm, realsense/t_sys_ns
                rgb.h5      # rgb (N, H, W, 3), t_sys_ns
            events/
                events.npy  # structured array (x, y, p, t)
        cube_medium/
            ...

Usage:
    python train_e2depth.py --data_root data/synthetic_data

    # With specific objects:
    python train_e2depth.py --data_dir data/synthetic_data/bottle data/synthetic_data/cube_medium

TensorBoard:
    tensorboard --logdir checkpoints_e2depth/runs
"""

import argparse
import os
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


# ================= DEFAULT PATHS =================
DATA_ROOT = Path("data/synthetic_data")
DEFAULT_OUT_DIR = Path("checkpoints_e2depth")
# =================================================

# ================= DEPTH PARAMETERS (tabletop scene) =================
D_MAX = 2.0   # Maximum depth in meters (tabletop range)
ALPHA = 3.7   # Log depth parameter
# D_metric = D_MAX * exp(-ALPHA * (1 - D_pred))
# At D_pred=0: D_metric = D_MAX * exp(-ALPHA) ≈ 0.05m (5cm minimum)
# At D_pred=1: D_metric = D_MAX = 2.0m
# =====================================================================


# -----------------------------
# ConvLSTM Module
# -----------------------------
class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    
    From paper: "Each encoder layer is composed of a downsampling convolution 
    with kernel size 5 and stride 2 and a ConvLSTM module with kernel size 3."
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Combined gates: input, forget, output, cell candidate
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            state: Tuple of (h, c) or None for initial state
        
        Returns:
            h: Hidden state (B, hidden_channels, H, W)
            (h, c): New state tuple
        """
        B, _, H, W = x.shape
        
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        
        # Split into individual gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell candidate
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


# -----------------------------
# E2Depth Architecture (Paper)
# -----------------------------
class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class EncoderLayer(nn.Module):
    """
    Encoder layer with downsampling conv + ConvLSTM.
    
    From paper: "downsampling convolution with kernel size 5 and stride 2 
    and a ConvLSTM module with kernel size 3"
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTMCell(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x = self.downsample(x)
        h, new_state = self.convlstm(x, state)
        return h, new_state


class DecoderLayer(nn.Module):
    """
    Decoder layer with bilinear upsampling + convolution.
    
    From paper: "each decoder layer is composed of a bilinear upsampling 
    operation followed by convolution with kernel size 5"
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # Bilinear upsampling to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class E2DepthNet(nn.Module):
    """
    E2Depth: Recurrent UNet for event-to-depth prediction.
    
    Architecture from paper:
    - Head layer (H)
    - NE=3 recurrent encoder layers with ConvLSTM
    - NR=2 residual blocks
    - NE=3 decoder layers
    - Prediction layer (P) with sigmoid
    
    Args:
        in_channels: Number of input channels (e.g., 5 for voxel grid bins)
        base: Base number of filters (Nb=32 in paper)
        num_encoders: Number of encoder layers (NE=3 in paper)
        num_residuals: Number of residual blocks (NR=2 in paper)
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
        
        # Head layer
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        
        # Encoder layers with ConvLSTM
        self.encoders = nn.ModuleList()
        ch = base
        for i in range(num_encoders):
            out_ch = ch * 2
            self.encoders.append(EncoderLayer(ch, out_ch))
            ch = out_ch
        
        # Residual blocks at bottleneck
        self.residuals = nn.ModuleList([
            ResidualBlock(ch) for _ in range(num_residuals)
        ])
        
        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            out_ch = ch // 2
            self.decoders.append(DecoderLayer(ch, skip_ch, out_ch))
            ch = out_ch
        
        # Prediction layer (depth-wise conv with kernel 1, sigmoid output)
        self.pred = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),  # Output normalized log depth in [0, 1]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with recurrent state.
        
        Args:
            x: Input tensor (B, C, H, W)
            states: List of (h, c) tuples for each encoder layer, or None
        
        Returns:
            pred: Predicted normalized log depth (B, 1, H, W)
            new_states: Updated states for next frame
        """
        if states is None:
            states = [None] * self.num_encoders
        
        # Head
        x = self.head(x)
        
        # Encoder with skip connections
        skips = [x]
        new_states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, states[i])
            new_states.append(state)
            if i < self.num_encoders - 1:
                skips.append(x)
        
        # Residual blocks
        for residual in self.residuals:
            x = residual(x)
        
        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)
        
        # Prediction
        pred = self.pred(x)
        
        return pred, new_states
    
    def forward_sequence(
        self,
        sequence: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process a sequence of frames with persistent state.
        
        Args:
            sequence: (B, T, C, H, W) tensor of T frames
            states: Initial states or None
        
        Returns:
            predictions: List of T predictions
            final_states: States after processing sequence
        """
        B, T, C, H, W = sequence.shape
        predictions = []
        
        for t in range(T):
            pred, states = self.forward(sequence[:, t], states)
            predictions.append(pred)
        
        return predictions, states


# -----------------------------
# Loss Functions (from paper)
# -----------------------------
def scale_invariant_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant loss from paper (Equation 3).
    
    L_si = (1/n) * sum(R^2) - (1/n^2) * sum(R)^2
    where R = pred_log - gt_log
    """
    # Both pred and gt should be in log space already (normalized [0, 1])
    diff = (pred - gt) * mask
    n = mask.sum().clamp_min(1.0)
    
    term1 = (diff ** 2).sum() / n
    term2 = (diff.sum() ** 2) / (n ** 2)
    
    return term1 - term2


def multi_scale_gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, num_scales: int = 4) -> torch.Tensor:
    """
    Multi-scale gradient matching loss from paper (Equation 4).
    
    Encourages smooth depth and sharp discontinuities.
    """
    def gradient_x(t):
        return t[:, :, :, 1:] - t[:, :, :, :-1]
    
    def gradient_y(t):
        return t[:, :, 1:, :] - t[:, :, :-1, :]
    
    total_loss = 0.0
    
    for scale in range(num_scales):
        if scale > 0:
            pred = F.avg_pool2d(pred, 2)
            gt = F.avg_pool2d(gt, 2)
            mask = F.avg_pool2d(mask, 2)
            mask = (mask > 0.5).float()
        
        # Compute residual
        residual = pred - gt
        
        # Gradients of residual
        grad_x = gradient_x(residual)
        grad_y = gradient_y(residual)
        
        # Mask for valid gradient pixels
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        
        # L1 norm of gradients
        loss_x = (torch.abs(grad_x) * mask_x).sum() / mask_x.sum().clamp_min(1.0)
        loss_y = (torch.abs(grad_y) * mask_y).sum() / mask_y.sum().clamp_min(1.0)
        
        total_loss = total_loss + loss_x + loss_y
    
    return total_loss / num_scales


def e2depth_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    lambda_grad: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss from paper (Equation 5).
    
    L_tot = L_si + λ * L_grad
    where λ = 0.5
    """
    l_si = scale_invariant_loss(pred, gt, mask)
    l_grad = multi_scale_gradient_loss(pred, gt, mask)
    
    total = l_si + lambda_grad * l_grad
    
    return total, {"si": l_si.item(), "grad": l_grad.item(), "total": total.item()}


# -----------------------------
# Depth Conversion Utilities
# -----------------------------
def depth_to_log_normalized(depth: torch.Tensor, d_max: float = D_MAX, alpha: float = ALPHA) -> torch.Tensor:
    """
    Convert metric depth to normalized log depth [0, 1].
    
    From paper (inverse of Equation 2):
    D_normalized = 1 + (1/α) * log(D_metric / D_max)
    """
    # Clamp to avoid log(0)
    depth = depth.clamp_min(1e-6)
    log_depth = 1.0 + (1.0 / alpha) * torch.log(depth / d_max)
    return log_depth.clamp(0, 1)


def log_normalized_to_depth(pred: torch.Tensor, d_max: float = D_MAX, alpha: float = ALPHA) -> torch.Tensor:
    """
    Convert normalized log depth [0, 1] to metric depth.
    
    From paper (Equation 2):
    D_metric = D_max * exp(-α * (1 - D_pred))
    """
    return d_max * torch.exp(-alpha * (1.0 - pred))


# -----------------------------
# Dataset Configuration
# -----------------------------
@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    seq_len: int = 1  # Sequence length for recurrent training
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) for random crop
    depth_max: float = D_MAX  # Maximum depth in meters
    depth_min: float = 0.05   # Minimum depth in meters (5cm for tabletop)
    augment: bool = True  # Apply data augmentation during training
    num_bins: int = 5  # Number of temporal bins for voxel grid


def events_to_voxel_grid(
    events: np.ndarray,
    height: int,
    width: int,
    num_bins: int = 5,
) -> np.ndarray:
    """
    Convert raw events to a voxel grid representation.
    
    Args:
        events: Structured array with fields (x, y, p, t)
        height: Image height
        width: Image width
        num_bins: Number of temporal bins
        
    Returns:
        Voxel grid of shape (num_bins, height, width)
    """
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    
    if len(events) == 0:
        return voxel
    
    # Extract event fields
    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    p = events['p'].astype(np.float32)  # 0 or 1
    t = events['t'].astype(np.float64)
    
    # Convert polarity: 0 -> -1, 1 -> +1
    p = p * 2 - 1
    
    # Normalize timestamps to [0, num_bins-1]
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        t_norm = np.zeros_like(t)
    
    # Bilinear interpolation across time bins
    t_floor = np.floor(t_norm).astype(np.int32)
    t_ceil = np.minimum(t_floor + 1, num_bins - 1)
    t_frac = t_norm - t_floor
    
    # Accumulate events into voxel grid
    for i in range(len(events)):
        if 0 <= x[i] < width and 0 <= y[i] < height:
            voxel[t_floor[i], y[i], x[i]] += p[i] * (1 - t_frac[i])
            voxel[t_ceil[i], y[i], x[i]] += p[i] * t_frac[i]
    
    # Normalize voxel grid
    nonzero_mask = voxel != 0
    if nonzero_mask.any():
        mean = voxel[nonzero_mask].mean()
        std = voxel[nonzero_mask].std()
        if std > 0:
            voxel = (voxel - mean) / std
    
    return voxel


class SyntheticDataset(Dataset):
    """
    Dataset for synthetic data from franka_pipeline.
    
    Supports two modes:
    1. Precomputed voxels (fast, low memory) - uses events/voxels/*.npy
    2. Raw events (slow, high memory) - uses events/events.npy
    
    Run precompute_voxels.py first for best performance.
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
        self.depth_h5_path = self.sequence_dir / "hdf5" / "depth.h5"
        self.voxels_dir = self.sequence_dir / "events" / "voxels"
        self.events_path = self.sequence_dir / "events" / "events.npy"
        
        # Check for precomputed voxels
        self.use_precomputed = self.voxels_dir.exists() and len(list(self.voxels_dir.glob("voxel_*.npy"))) > 0
        
        # Verify files exist
        if not self.depth_h5_path.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5_path}")
        if not self.use_precomputed and not self.events_path.exists():
            raise FileNotFoundError(f"Neither voxels nor events found in {self.sequence_dir}")
        
        # Load depth metadata only (not full data)
        with h5py.File(self.depth_h5_path, 'r') as f:
            self.n_frames = f['realsense/depth'].shape[0]
            self.H = f['realsense/depth'].shape[1]
            self.W = f['realsense/depth'].shape[2]
            # HDF5 timestamps are nanoseconds; event timestamps are microseconds.
            self.depth_timestamps = f['realsense/t_sys_ns'][:] // 1000
        
        if self.use_precomputed:
            # Count voxel files
            self.voxel_files = sorted(self.voxels_dir.glob("voxel_*.npy"))
            n_voxels = len(self.voxel_files)
            if n_voxels != self.n_frames:
                print(f"Warning: {n_voxels} voxels != {self.n_frames} frames")
                self.n_frames = min(n_voxels, self.n_frames)
            self.events = None
            self.n_events = 0
        else:
            # Fall back to loading raw events (high memory!)
            print(f"[{self.sequence_dir.name}] Warning: No precomputed voxels, loading raw events...")
            self.events = np.load(self.events_path)
            self.n_events = len(self.events)
            self.voxel_files = None
        
        # Compute valid indices
        self._compute_valid_indices(val_ratio, seed)
        
        mode = "precomputed" if self.use_precomputed else f"raw ({self.n_events:,} events)"
        print(f"[{self.sequence_dir.name}] {split}: {len(self.indices)} samples "
              f"(frames: {self.n_frames}, mode: {mode}, res: {self.W}x{self.H})")
    
    def _compute_valid_indices(self, val_ratio: float, seed: int):
        """Compute valid starting indices for sequences."""
        seq_len = self.cfg.seq_len
        
        # Need seq_len consecutive frames
        idx_max = self.n_frames - seq_len
        if idx_max < 0:
            raise RuntimeError(f"Not enough frames ({self.n_frames}) for seq_len={seq_len}")
        
        all_indices = np.arange(0, idx_max + 1, dtype=np.int64)
        n_total = len(all_indices)
        
        # Block-based split to avoid temporal leakage
        rng = np.random.default_rng(seed)
        block_size = max(10, seq_len * 2)
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
    
    def _get_voxel(self, frame_idx: int) -> np.ndarray:
        """Get voxel grid for a frame."""
        if self.use_precomputed:
            voxel_path = self.voxels_dir / f"voxel_{frame_idx:06d}.npy"
            return np.load(voxel_path)
        else:
            # Compute on the fly (slow)
            frame_events = self._get_events_for_frame(frame_idx)
            return events_to_voxel_grid(frame_events, self.H, self.W, self.cfg.num_bins)
    
    def _get_events_for_frame(self, frame_idx: int) -> np.ndarray:
        """Get events between frame_idx and frame_idx+1 timestamps."""
        if frame_idx >= len(self.depth_timestamps) - 1:
            t_start = self.depth_timestamps[frame_idx]
            mask = self.events['t'] >= t_start
        else:
            t_start = self.depth_timestamps[frame_idx]
            t_end = self.depth_timestamps[frame_idx + 1]
            mask = (self.events['t'] >= t_start) & (self.events['t'] < t_end)
        return self.events[mask]
    
    def _get_depth(self, frame_idx: int) -> np.ndarray:
        """Get depth frame (lazy load from HDF5)."""
        with h5py.File(self.depth_h5_path, 'r') as f:
            depth = f['realsense/depth'][frame_idx].astype(np.float32) / 1000.0
        return depth
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i: int):
        start_idx = int(self.indices[i])
        seq_len = self.cfg.seq_len
        
        events_seq = []
        depth_seq = []
        mask_seq = []
        
        for t in range(seq_len):
            idx = start_idx + t
            
            # Get voxel grid
            voxel = self._get_voxel(idx)
            
            # Get depth frame (lazy load)
            depth = self._get_depth(idx)
            
            # Create validity mask
            mask = ((depth > self.cfg.depth_min) & (depth < self.cfg.depth_max)).astype(np.float32)
            
            # Convert depth to log normalized [0, 1]
            depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
            depth = 1.0 + (1.0 / ALPHA) * np.log(depth / D_MAX)
            depth = np.clip(depth, 0, 1)
            
            events_seq.append(voxel)
            depth_seq.append(depth[None])  # (1, H, W)
            mask_seq.append(mask[None])
        
        # Stack sequences: (T, C, H, W)
        events = np.stack(events_seq, axis=0)
        depths = np.stack(depth_seq, axis=0)
        masks = np.stack(mask_seq, axis=0)
        
        # Random crop (training)
        if self.cfg.crop_size is not None and self.cfg.augment and self.split == "train":
            ch, cw = self.cfg.crop_size
            if self.H > ch and self.W > cw:
                y0 = np.random.randint(0, self.H - ch + 1)
                x0 = np.random.randint(0, self.W - cw + 1)
                events = events[:, :, y0:y0+ch, x0:x0+cw]
                depths = depths[:, :, y0:y0+ch, x0:x0+cw]
                masks = masks[:, :, y0:y0+ch, x0:x0+cw]
        
        # Horizontal flip (training)
        if self.cfg.augment and self.split == "train" and np.random.rand() > 0.5:
            events = np.flip(events, axis=3).copy()
            depths = np.flip(depths, axis=3).copy()
            masks = np.flip(masks, axis=3).copy()
        
        return (
            torch.from_numpy(events).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(masks).float(),
        )


def create_multi_sequence_dataset(
    sequence_dirs: List[str],
    cfg: DataConfig,
    split: str = "train",
    val_ratio: float = 0.1,
) -> Dataset:
    """Create concatenated dataset from multiple sequences."""
    datasets = []
    for seq_dir in sequence_dirs:
        try:
            ds = SyntheticDataset(seq_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir}: {e}")
    
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """
    Find valid synthetic data directories.
    
    Each valid directory should have:
    - hdf5/depth.h5
    - events/events.npy
    """
    sequence_dirs = []
    for d in data_root.iterdir():
        if d.is_dir():
            depth_h5 = d / "hdf5" / "depth.h5"
            events_npy = d / "events" / "events.npy"
            if depth_h5.exists() and events_npy.exists():
                sequence_dirs.append(d)
    return sorted(sequence_dirs)


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(
    model: E2DepthNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_grad: float = 0.5,
) -> Dict[str, float]:
    """Train for one epoch with sequence processing."""
    model.train()
    
    total_loss = 0.0
    total_si = 0.0
    total_grad = 0.0
    n_batches = 0
    
    for events, depths, masks in loader:
        # events: (B, T, C, H, W)
        # depths: (B, T, 1, H, W)
        # masks: (B, T, 1, H, W)
        B, T = events.shape[:2]
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Process sequence
        states = None
        batch_loss = 0.0
        batch_si = 0.0
        batch_grad = 0.0
        
        for t in range(T):
            pred, states = model(events[:, t], states)
            loss, metrics = e2depth_loss(pred, depths[:, t], masks[:, t], lambda_grad)
            batch_loss += loss
            batch_si += metrics["si"]
            batch_grad += metrics["grad"]
        
        # Average over sequence
        batch_loss = batch_loss / T
        
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_si += batch_si / T
        total_grad += batch_grad / T
        n_batches += 1
    
    return {
        "total": total_loss / max(1, n_batches),
        "si": total_si / max(1, n_batches),
        "grad": total_grad / max(1, n_batches),
    }


@torch.no_grad()
def validate(
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate and compute metrics."""
    model.eval()
    
    total_si = 0.0
    total_l1 = 0.0
    total_abs_rel = 0.0
    n_batches = 0
    
    for events, depths, masks in loader:
        B, T = events.shape[:2]
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        states = None
        for t in range(T):
            pred, states = model(events[:, t], states)
        
        # Metrics on last frame
        gt = depths[:, -1]
        mask = masks[:, -1]
        
        # Scale-invariant loss
        total_si += scale_invariant_loss(pred, gt, mask).item()
        
        # Convert to metric depth for L1 and abs_rel
        pred_metric = log_normalized_to_depth(pred)
        gt_metric = log_normalized_to_depth(gt)
        
        # Debug: track prediction statistics
        pred_mean = (pred * mask).sum() / mask.sum()
        gt_mean = (gt * mask).sum() / mask.sum()
        
        diff = torch.abs(pred_metric - gt_metric) * mask
        n_valid = mask.sum().clamp_min(1.0)
        
        total_l1 += (diff.sum() / n_valid).item()
        total_abs_rel += ((diff / gt_metric.clamp_min(1e-6)).sum() / n_valid).item()
        n_batches += 1
        
        # Store last batch stats for debugging
        last_pred_mean = pred_mean.item()
        last_gt_mean = gt_mean.item()
        last_pred_metric_mean = (pred_metric * mask).sum().item() / n_valid.item()
        last_gt_metric_mean = (gt_metric * mask).sum().item() / n_valid.item()
    
    n = max(1, n_batches)
    return {
        "si": total_si / n,
        "l1_metric": total_l1 / n,
        "abs_rel": total_abs_rel / n,
        "pred_log_mean": last_pred_mean,
        "gt_log_mean": last_gt_mean,
        "pred_metric_mean": last_pred_metric_mean,
        "gt_metric_mean": last_gt_metric_mean,
    }


def log_images(
    writer: SummaryWriter,
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_images: int = 4,
):
    """Log sample predictions to TensorBoard."""
    model.eval()
    
    with torch.no_grad():
        for i, (events, depths, masks) in enumerate(loader):
            if i >= 1:
                break
            
            B, T = events.shape[:2]
            events = events.to(device)
            depths = depths.to(device)
            masks = masks.to(device)
            
            # Process full sequence
            states = None
            for t in range(T):
                pred, states = model(events[:, t], states)
            
            # Log last frame predictions
            for j in range(min(max_images, B)):
                ev_img = events[j, -1]
                if ev_img.shape[0] >= 3:
                    ev_img = ev_img[:3]
                else:
                    ev_img = ev_img[0:1]
                
                # Normalize for display
                ev_img = (ev_img - ev_img.min()) / (ev_img.max() - ev_img.min() + 1e-6)
                gt_img = depths[j, -1]
                pred_img = pred[j]
                err_img = torch.abs(pred[j] - depths[j, -1]) * masks[j, -1]
                
                writer.add_image(f"sample_{j}/input_events", ev_img, epoch)
                writer.add_image(f"sample_{j}/gt_log_depth", gt_img, epoch)
                writer.add_image(f"sample_{j}/pred_log_depth", pred_img, epoch)
                writer.add_image(f"sample_{j}/error", err_img / (err_img.max() + 1e-6), epoch)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train E2Depth (recurrent UNet) for event-to-depth prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", nargs="+", type=str, default=None,
                           help="Sequence directory(ies) to train on")
    data_group.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                           help="Root data directory (will use all subdirs)")
    data_group.add_argument("--val_ratio", type=float, default=0.1,
                           help="Fraction of data for validation")
    data_group.add_argument("--num_bins", type=int, default=5,
                           help="Number of temporal bins for voxel grid")
    data_group.add_argument("--depth_max", type=float, default=D_MAX,
                           help="Maximum depth in meters")
    data_group.add_argument("--depth_min", type=float, default=0.05,
                           help="Minimum depth in meters")
    
    # Model (paper defaults)
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--base", type=int, default=32,
                            help="Base filters (Nb in paper)")
    model_group.add_argument("--num_encoders", type=int, default=3,
                            help="Number of encoder layers (NE in paper)")
    model_group.add_argument("--num_residuals", type=int, default=2,
                            help="Number of residual blocks (NR in paper)")
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=300,
                            help="Number of epochs")
    train_group.add_argument("--batch", type=int, default=8,
                            help="Batch size")
    train_group.add_argument("--seq_len", type=int, default=8,
                            help="Sequence length for recurrent training")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
    train_group.add_argument("--lambda_grad", type=float, default=0.5,
                            help="Weight for gradient loss (λ in paper)")
    train_group.add_argument("--num_workers", type=int, default=4)
    train_group.add_argument("--crop_h", type=int, default=0,
                            help="Random crop height (0=no crop)")
    train_group.add_argument("--crop_w", type=int, default=0,
                            help="Random crop width (0=no crop)")
    
    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    out_group.add_argument("--save_every", type=int, default=10,
                          help="Save checkpoint every N epochs")
    out_group.add_argument("--resume", type=str, default=None,
                          help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Find sequences
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequences found in {args.data_root}")
            print("Expected structure: <dir>/hdf5/depth.h5 and <dir>/events/events.npy")
            return
    
    print(f"Found {len(sequence_dirs)} sequences:")
    for d in sequence_dirs:
        print(f"  - {d.name}")
    
    # Config
    crop_size = None
    if args.crop_h > 0 and args.crop_w > 0:
        crop_size = (args.crop_h, args.crop_w)
    
    cfg = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        augment=True,
        num_bins=args.num_bins,
    )
    
    # Create config without augmentation for validation
    cfg_val = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        augment=False,
        num_bins=args.num_bins,
    )
    
    # Datasets (split each sequence into train/val)
    train_ds = create_multi_sequence_dataset(
        [str(d) for d in sequence_dirs], cfg, split="train", val_ratio=args.val_ratio
    )
    val_ds = create_multi_sequence_dataset(
        [str(d) for d in sequence_dirs], cfg_val, split="val", val_ratio=args.val_ratio
    )
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Sequences: {len(sequence_dirs)}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Valid samples: {len(val_ds)}")
    print(f"  Depth range: {args.depth_min:.2f}m - {args.depth_max:.2f}m")
    print(f"  Voxel bins: {args.num_bins}")
    print(f"{'='*60}\n")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Input channels = num_bins
    in_channels = args.num_bins
    print(f"Input channels: {in_channels}")
    
    # Model
    model = E2DepthNet(
        in_channels=in_channels,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    start_epoch = 1
    best_val_loss = float("inf")
    
    # Resume
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
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, lambda_grad=args.lambda_grad
        )
        val_metrics = validate(model, val_loader, device)
        
        # Step scheduler based on validation loss
        scheduler.step(val_metrics["si"])
        
        # Logging
        writer.add_scalar("loss/train_total", train_metrics["total"], epoch)
        writer.add_scalar("loss/train_si", train_metrics["si"], epoch)
        writer.add_scalar("loss/train_grad", train_metrics["grad"], epoch)
        writer.add_scalar("loss/val_si", val_metrics["si"], epoch)
        writer.add_scalar("loss/val_l1_metric", val_metrics["l1_metric"], epoch)
        writer.add_scalar("loss/val_abs_rel", val_metrics["abs_rel"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"Epoch {epoch:03d} | train: {train_metrics['total']:.5f} "
              f"| val SI: {val_metrics['si']:.5f} | val L1: {val_metrics['l1_metric']:.3f}m "
              f"| val AbsRel: {val_metrics['abs_rel']:.4f}")
        print(f"          | pred_log: {val_metrics['pred_log_mean']:.3f} vs gt_log: {val_metrics['gt_log_mean']:.3f} "
              f"| pred_m: {val_metrics['pred_metric_mean']:.3f}m vs gt_m: {val_metrics['gt_metric_mean']:.3f}m")
        
        # Save best
        if val_metrics["si"] < best_val_loss:
            best_val_loss = val_metrics["si"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "in_channels": in_channels,
                    "base": args.base,
                    "num_encoders": args.num_encoders,
                    "num_residuals": args.num_residuals,
                    "depth_max": args.depth_max,
                    "depth_min": args.depth_min,
                },
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> Saved best model (val SI: {best_val_loss:.5f})")
        
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
    print(f"\nTraining complete! Best val SI: {best_val_loss:.5f}")
    print(f"Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
