"""
Train E2Depth: Recurrent UNet for Event-to-Depth prediction.

Implementation based on:
"Learning Monocular Dense Depth from Events" (Hidalgo-Carrió et al., 3DV 2020)

Key differences from standard UNet:
- ConvLSTM in encoder layers for temporal recurrence
- Residual blocks in bottleneck
- Sigmoid output [0,1]: log depth (--log_depth) or linear normalization (default)
- Scale-invariant + multi-scale gradient loss
- Bilinear upsampling in decoder

Expected folder structure (real data):
    data/real/
        <object_name>/
            hdf5/
                realsense.h5              # depth (N, H, W) uint16 mm, t_sys_ns
                depth_in_event_frame.h5   # depth (N, H, W) float32 metres (preferred)
                rgb_in_event_frame.h5     # rgb (N, H, W, 3) uint8 (for --rgb_mask)
                poses.h5                  # ee_T (N, 4, 4), joint_positions, gripper_q
            events/
                voxels_cam0/              # precomputed voxel_NNNNNN.npy files

Usage:
    python real_train.py --data_root data/real

    # With specific objects:
    python3 real_train.py --data_dir data/real/bottle data/real/cube_medium

    # With pose input (requires precomputed pose voxels):
    python3 precompute_pose_depth.py --data_root data/real
    python3 real_train.py --data_root data/real --use_pose

    # With RGB white-pixel masking (requires projected RGB):
    python3 project_realsense_to_event.py --data_root data/real
    python3 real_train.py --data_root data/real --rgb_mask

TensorBoard:
    tensorboard --logdir checkpoints_e2depth/runs
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

# ================= DEFAULT PATHS =================
DATA_ROOT = _DATA_ROOT
# =================================================




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
            nn.Sigmoid(),  # Output normalized depth in [0, 1]
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
    lambda_grad: float = 0.5,
    lambda_mean: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss.

    L_tot = L_si + λ_grad * L_grad + λ_mean * L_mean

    L_mean penalises systematic global bias (mean residual) that L_si ignores
    because its second term cancels constant offsets.
    """
    l_si = scale_invariant_loss(pred, gt, mask)
    l_grad = multi_scale_gradient_loss(pred, gt, mask)

    n = mask.sum().clamp_min(1.0)
    mean_diff = ((pred - gt) * mask).sum() / n
    l_mean = mean_diff ** 2

    total = l_si + lambda_grad * l_grad + lambda_mean * l_mean

    return total, {
        "si": l_si.item(),
        "grad": l_grad.item(),
        "mean": l_mean.item(),
        "total": total.item(),
    }


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


def depth_to_linear_normalized(depth: torch.Tensor, d_min: float = DEPTH_MIN, d_max: float = D_MAX) -> torch.Tensor:
    """Convert metric depth to linearly normalized [0, 1]: (d - d_min) / (d_max - d_min)."""
    return ((depth - d_min) / (d_max - d_min)).clamp(0, 1)


def linear_normalized_to_depth(pred: torch.Tensor, d_min: float = DEPTH_MIN, d_max: float = D_MAX) -> torch.Tensor:
    """Convert linearly normalized [0, 1] back to metric depth."""
    return pred * (d_max - d_min) + d_min


# -----------------------------
# Dataset Configuration
# -----------------------------
@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    seq_len: int = 1  # Sequence length for recurrent training
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) center crop applied after resize
    resize_hw: Optional[Tuple[int, int]] = None  # (H, W) resize before crop/augmentation
    depth_max: float = D_MAX  # Maximum depth in meters
    depth_min: float = DEPTH_MIN   # Minimum depth in meters (5cm for tabletop)
    augment: bool = True  # Apply data augmentation during training
    num_bins: int = NUM_BINS  # Number of temporal bins for voxel grid
    use_pose: bool = False  # Use precomputed pose depth channel (from voxels_pose_cam0/)
    rgb_mask: bool = False  # Mask out white pixels using projected RGB
    spatial_mask: bool = False  # Mask pixels outside cube around EE (from spatial_mask.h5)
    log_depth: bool = False  # Use log depth encoding (paper default); False = linear normalization


class RealDataset(Dataset):
    """
    Dataset for real data recorded with franka_pipeline + synchronised_recording.

    Requires precomputed voxels (run precompute_voxels.py first).
    When ``cfg.use_pose`` is ``True``, loads from voxels_pose_cam0/
    (run precompute_pose_depth.py first) which has the pose depth
    channel already baked in as the last channel.

    When ``cfg.rgb_mask`` is ``True``, additionally masks out white pixels
    in the projected RGB (from rgb_in_event_frame.h5, run
    project_realsense_to_event.py first).
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
        
        # --- Depth ---
        projected = self.sequence_dir / "hdf5" / "depth_in_event_frame.h5"
        if projected.exists():
            self.depth_h5_path = projected
            self._depth_key = "depth"
            self._depth_is_metric = True   # already float32 metres
        else:
            self.depth_h5_path = self.sequence_dir / "hdf5" / "realsense.h5"
            self._depth_key = "depth"
            self._depth_is_metric = False  # uint16 mm
        self._ts_h5_path = self.sequence_dir / "hdf5" / "realsense.h5"
        self._ts_key = "t_sys_ns"

        # --- Voxels (precomputed) ---
        self.use_pose = cfg.use_pose
        if self.use_pose:
            # Prefer voxels_pose_cam0 (has pose depth channel appended)
            if (self.sequence_dir / "events" / "voxels_pose_cam0").exists():
                self.voxels_dir = self.sequence_dir / "events" / "voxels_pose_cam0"
            else:
                raise FileNotFoundError(
                    f"--use_pose requires precomputed pose voxels. "
                    f"Run: python precompute_pose_depth.py --data_dir {self.sequence_dir}"
                )
        elif (self.sequence_dir / "events" / "voxels_cam0").exists():
            self.voxels_dir = self.sequence_dir / "events" / "voxels_cam0"
        else:
            self.voxels_dir = self.sequence_dir / "events" / "voxels"
        
        # Verify files exist
        if not self.depth_h5_path.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5_path}")
        
        self.voxel_files = sorted(self.voxels_dir.glob("voxel_*.npy")) if self.voxels_dir.exists() else []
        if not self.voxel_files:
            raise FileNotFoundError(f"No precomputed voxels in {self.voxels_dir}")
        
        # --- Metadata ---
        with h5py.File(self.depth_h5_path, 'r') as f:
            self.n_frames = f[self._depth_key].shape[0]
            self.H = f[self._depth_key].shape[1]
            self.W = f[self._depth_key].shape[2]
        with h5py.File(self._ts_h5_path, 'r') as f:
            self.depth_timestamps = f[self._ts_key][:] // 1000
        
        n_voxels = len(self.voxel_files)
        if n_voxels != self.n_frames:
            print(f"Warning: {n_voxels} voxels != {self.n_frames} frames")
            self.n_frames = min(n_voxels, self.n_frames)
        
        # --- RGB mask (optional) ---
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

        # --- Spatial mask (optional, precomputed) ---
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
        
        # --- Train / val split ---
        self._compute_valid_indices(val_ratio, seed)
        
        depth_src = "projected" if self._depth_is_metric else "raw realsense"
        pose_str = "with pose" if self.use_pose else "no pose"
        rgb_str = ", rgb_mask" if self.rgb_mask else ""
        spatial_str = ", spatial_mask" if self.spatial_mask else ""
        depth_enc = "log" if cfg.log_depth else "linear"
        print(f"[{self.sequence_dir.name}] {split}: {len(self.indices)} samples "
              f"(frames: {self.n_frames}, depth: {depth_src} [{depth_enc}], {pose_str}{rgb_str}{spatial_str}, res: {self.W}x{self.H})")
    
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
        """Get precomputed voxel grid for a frame."""
        voxel_path = self.voxels_dir / f"voxel_{frame_idx:06d}.npy"
        return np.load(voxel_path)
    
    def _get_depth(self, frame_idx: int) -> np.ndarray:
        """Get depth frame (lazy load from HDF5)."""
        with h5py.File(self.depth_h5_path, 'r') as f:
            if self._depth_is_metric:
                depth = f[self._depth_key][frame_idx].astype(np.float32)
            else:
                depth = f[self._depth_key][frame_idx].astype(np.float32) / 1000.0
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
            
            # Create validity mask (depth range)
            mask = ((depth > self.cfg.depth_min) & (depth < self.cfg.depth_max)).astype(np.float32)
            
            # Mask out white pixels from projected RGB
            if self.rgb_mask:
                with h5py.File(self._rgb_h5_path, 'r') as rf:
                    rgb = rf["rgb"][idx]  # (H, W, 3) uint8
                # White = all channels above threshold
                white = np.all(rgb > WHITE_THRESH, axis=-1)  # (H, W)
                mask[white] = 0.0

            # Apply precomputed spatial mask (cube around EE)
            if self.spatial_mask:
                with h5py.File(self._spatial_h5_path, 'r') as sf:
                    sp = sf["mask"][idx]  # (H, W) uint8
                mask[sp == 0] = 0.0
            
            # Normalize depth to [0, 1]
            depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
            if self.cfg.log_depth:
                depth = 1.0 + (1.0 / ALPHA) * np.log(depth / self.cfg.depth_max)
            else:
                depth = (depth - self.cfg.depth_min) / (self.cfg.depth_max - self.cfg.depth_min)
            depth = np.clip(depth, 0, 1)
            
            events_seq.append(voxel)
            depth_seq.append(depth[None])  # (1, H, W)
            mask_seq.append(mask[None])
        
        # Stack sequences: (T, C, H, W)
        events = np.stack(events_seq, axis=0)
        depths = np.stack(depth_seq, axis=0)
        masks = np.stack(mask_seq, axis=0)

        # Resize to target resolution (applied before crop/augmentation)
        if self.cfg.resize_hw is not None:
            rh, rw = self.cfg.resize_hw
            events = F.interpolate(torch.from_numpy(events), size=(rh, rw), mode="bilinear", align_corners=False).numpy()
            depths = F.interpolate(torch.from_numpy(depths), size=(rh, rw), mode="bilinear", align_corners=False).numpy()
            masks = F.interpolate(torch.from_numpy(masks), size=(rh, rw), mode="nearest").numpy()

        # Spatial dims after optional resize (used for crop bounds)
        cur_H = events.shape[2]
        cur_W = events.shape[3]

        # Center crop (applied to both train and val whenever crop_size is set)
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            if cur_H < ch or cur_W < cw:
                raise ValueError(
                    f"Center crop {cw}x{ch} larger than image {cur_W}x{cur_H}"
                )
            y0 = (cur_H - ch) // 2
            x0 = (cur_W - cw) // 2
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
            ds = RealDataset(seq_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir}: {e}")
    
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """
    Find valid real data sequence directories.

    Required layout:
        hdf5/realsense.h5  (or depth_in_event_frame.h5)
        events/voxels_cam0/*.npy  (or events/voxels/*.npy)
    """
    sequence_dirs = []
    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        has_depth = (
            (d / "hdf5" / "depth_in_event_frame.h5").exists()
            or (d / "hdf5" / "realsense.h5").exists()
        )
        has_voxels = (
            (d / "events" / "voxels_cam0").exists()
            or (d / "events" / "voxels").exists()
        )
        if has_depth and has_voxels:
            sequence_dirs.append(d)
    return sorted(sequence_dirs)


# -----------------------------
# Debug Visualization
# -----------------------------
def debug_visualize(
    sequence_dir: str,
    n_samples: int = 3,
    out_path: str = "debug_viz.png",
    num_bins: int = 5,
    seed: int = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    use_pose: bool = False,
    rgb_mask: bool = False,
    spatial_mask: bool = False,
) -> None:
    """
    Save a debug PNG with n_samples × 2 rows (raw + preprocessed per timestamp).

    RAW row columns:
      1. Depth – Realsense raw            (plasma, metres)
      2. Depth Mask                        (gray)
      3. Projected Depth – event plane     (plasma, metres)  [N/A if absent]
      4. Event Plane Mask                  (gray)            [N/A if absent]
      5. Events Accumulated – sum of bins  (gray)
      6. RGB (Realsense raw)               [N/A if absent]
      7. Projected RGB – event plane       [N/A if absent]

    PREPROCESSED row columns (after resize then center-crop):
      1. Projected Depth – preprocessed    (plasma)
      2. Event Plane Mask – preprocessed   (gray)
      3. Events Accumulated – preprocessed (gray)
      4. RGB – preprocessed                [N/A if absent]
      5. Pose Depth – raw event resolution (viridis)  [N/A if --use_pose off]
      6. Pose Depth – preprocessed         (viridis)  [N/A if --use_pose off]
      7. Total Mask – preprocessed         (gray)     [N/A if no depth projected]

    Resolution (W×H) is annotated in the top-left corner of every panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_dir = Path(sequence_dir)
    rng = np.random.default_rng(seed)

    # ---- Locate raw depth & timestamps ----
    realsense_path = seq_dir / "hdf5" / "realsense.h5"
    projected_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"

    if not realsense_path.exists():
        raise FileNotFoundError(f"No realsense.h5 found in {seq_dir / 'hdf5'}")
    raw_depth_path = realsense_path
    raw_depth_key = "depth"
    ts_key = "t_sys_ns"

    # ---- Locate RGB ----
    rgb_path: Optional[Path] = None
    rgb_key: Optional[str] = None
    for _rp, _rk in [
        (seq_dir / "hdf5" / "realsense.h5", "rgb"),
        (seq_dir / "hdf5" / "rgb.h5", "rgb"),
    ]:
        if _rp.exists():
            with h5py.File(_rp, "r") as _f:
                if _rk in _f:
                    rgb_path, rgb_key = _rp, _rk
                    break

    # ---- Locate precomputed voxels ----
    voxels_dir: Optional[Path] = None
    for _vd in [
        seq_dir / "events" / "voxels_cam0",
        seq_dir / "events" / "voxels",
    ]:
        if _vd.exists() and list(_vd.glob("voxel_*.npy")):
            voxels_dir = _vd
            break
    # ---- Load metadata ----
    with h5py.File(raw_depth_path, "r") as _f:
        n_frames   = _f[raw_depth_key].shape[0]
        depth_H    = int(_f[raw_depth_key].shape[1])
        depth_W    = int(_f[raw_depth_key].shape[2])

    proj_H: Optional[int] = None
    proj_W: Optional[int] = None
    if projected_path.exists():
        with h5py.File(projected_path, "r") as _f:
            proj_H = int(_f["depth"].shape[1])
            proj_W = int(_f["depth"].shape[2])

    # ---- Pose voxels (precomputed, optional) ----
    pose_voxels_dir: Optional[Path] = None
    if use_pose:
        _pvd = seq_dir / "events" / "voxels_pose_cam0"
        if _pvd.exists() and list(_pvd.glob("voxel_*.npy")):
            pose_voxels_dir = _pvd
        else:
            print("[debug_viz] Warning: voxels_pose_cam0 not found, skipping pose depth")

    # ---- Projected RGB (for rgb_mask visualisation) ----
    proj_rgb_path: Optional[Path] = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    if not proj_rgb_path.exists():
        proj_rgb_path = None
        if rgb_mask:
            print("[debug_viz] Warning: rgb_in_event_frame.h5 not found, skipping RGB mask")

    # ---- Spatial mask (precomputed) ----
    spatial_mask_path: Optional[Path] = seq_dir / "hdf5" / "spatial_mask.h5"
    if not spatial_mask_path.exists():
        spatial_mask_path = None
        if spatial_mask:
            print("[debug_viz] Warning: spatial_mask.h5 not found, skipping spatial mask")

    # ---- Sample random frame indices ----
    n_samples = min(n_samples, n_frames)
    frame_indices: List[int] = sorted(
        rng.choice(n_frames, size=n_samples, replace=False).tolist()
    )

    # ---- Helpers ----
    def _apply_resize_crop(img: np.ndarray) -> np.ndarray:
        """img: (H, W) or (H, W, C) — returns resized+cropped numpy array."""
        is_rgb = img.ndim == 3
        # add batch+channel dims expected by F.interpolate: (1, C, H, W)
        if is_rgb:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        else:
            t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        if resize_hw is not None:
            mode = "bilinear" if is_rgb or img.dtype != bool else "nearest"
            t = F.interpolate(t, size=resize_hw, mode=mode, align_corners=False if mode == "bilinear" else None)
        if crop_size is not None:
            ch, cw = crop_size
            cur_H2, cur_W2 = t.shape[2], t.shape[3]
            y0 = (cur_H2 - ch) // 2
            x0 = (cur_W2 - cw) // 2
            t = t[:, :, y0:y0 + ch, x0:x0 + cw]
        out = t.squeeze(0).numpy()
        if is_rgb:
            out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        else:
            out = out.squeeze(0)
        return out

    # ---- Layout ----
    # Each sample occupies 2 rows: raw (top) + preprocessed (bottom)
    N_COLS = 7
    N_ROWS = n_samples * 2
    row_h = 3.5
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 4.2, N_ROWS * row_h))
    if N_ROWS == 1:
        axes = axes[np.newaxis, :]

    preproc_label = ""
    if resize_hw is not None:
        preproc_label += f"resize→{resize_hw[1]}×{resize_hw[0]}"
    if crop_size is not None:
        if preproc_label:
            preproc_label += " + "
        preproc_label += f"center-crop→{crop_size[1]}×{crop_size[0]}"
    if not preproc_label:
        preproc_label = "no resize/crop"

    def _annotate(ax, W: int, H: int, title: str, frame_idx: Optional[int] = None) -> None:
        full_title = f"[HDF idx {frame_idx}]  {title}" if frame_idx is not None else title
        ax.set_title(full_title, fontsize=8, pad=3)
        ax.text(
            0.01, 0.99, f"{W}×{H}",
            color="white", fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
        )
        ax.axis("off")

    def _na(ax, title: str) -> None:
        ax.set_title(title, fontsize=8, pad=3)
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")

    def _blank(ax) -> None:
        ax.axis("off")

    for sample_i, frame_idx in enumerate(frame_indices):
        raw_row = sample_i * 2          # top row  — raw data
        pre_row = sample_i * 2 + 1     # bottom row — after preprocessing

        # ================================================================
        # RAW ROW
        # ================================================================

        # --- 1. Raw depth ---
        with h5py.File(raw_depth_path, "r") as _f:
            raw_d = _f[raw_depth_key][frame_idx].astype(np.float32)
        raw_d_m = raw_d / 1000.0 if raw_d.max() > 100.0 else raw_d
        ax = axes[raw_row, 0]
        ax.imshow(raw_d_m, cmap="plasma", vmin=0.0, vmax=2.5)
        _annotate(ax, depth_W, depth_H, "Depth (Realsense)", frame_idx=frame_idx)

        # --- 2. Raw depth mask ---
        raw_mask_arr = ((raw_d_m > 0.01) & (raw_d_m < 5.0)).astype(np.float32)
        ax = axes[raw_row, 1]
        ax.imshow(raw_mask_arr, cmap="gray", vmin=0, vmax=1)
        _annotate(ax, depth_W, depth_H, "Depth Mask")

        # --- 3. Projected depth (event plane) ---
        proj_d: Optional[np.ndarray] = None
        ax = axes[raw_row, 2]
        if projected_path.exists():
            with h5py.File(projected_path, "r") as _f:
                proj_d = _f["depth"][frame_idx].astype(np.float32)
            ax.imshow(proj_d, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, proj_W, proj_H, "Projected Depth")
        else:
            _na(ax, "Projected Depth")

        # --- 4. Event plane mask ---
        ax = axes[raw_row, 3]
        proj_mask_arr: Optional[np.ndarray] = None
        if proj_d is not None:
            proj_mask_arr = ((proj_d > 0.01) & (proj_d < 5.0)).astype(np.float32)
            ax.imshow(proj_mask_arr, cmap="gray", vmin=0, vmax=1)
            _annotate(ax, proj_W, proj_H, "Event Plane Mask")
        else:
            _na(ax, "Event Plane Mask")

        # --- 5. Accumulated events ---
        voxel: Optional[np.ndarray] = None
        ax = axes[raw_row, 4]
        if voxels_dir is not None:
            vp = voxels_dir / f"voxel_{frame_idx:06d}.npy"
            if vp.exists():
                voxel = np.load(vp)
        if voxel is not None:
            accum = voxel.sum(axis=0)
            ev_H_v, ev_W_v = accum.shape
            vmax_ev = float(max(np.abs(accum).max(), 1e-6))
            ax.imshow(accum, cmap="gray", vmin=-vmax_ev, vmax=vmax_ev)
            _annotate(ax, ev_W_v, ev_H_v, "Events (Accumulated)")
        else:
            _na(ax, "Events (Accumulated)")

        # --- 6. RGB (Realsense raw) ---
        rgb_frame_raw: Optional[np.ndarray] = None
        ax = axes[raw_row, 5]
        if rgb_path is not None:
            with h5py.File(rgb_path, "r") as _f:
                rgb_frame_raw = _f[rgb_key][frame_idx]
            if rgb_frame_raw.dtype != np.uint8:
                rgb_frame_raw = np.clip(rgb_frame_raw, 0, 255).astype(np.uint8)
            rgb_H2, rgb_W2 = int(rgb_frame_raw.shape[0]), int(rgb_frame_raw.shape[1])
            ax.imshow(rgb_frame_raw)
            _annotate(ax, rgb_W2, rgb_H2, "RGB")
        else:
            _na(ax, "RGB")

        # --- 7. Projected RGB (event plane) ---
        proj_rgb_frame: Optional[np.ndarray] = None
        ax = axes[raw_row, 6]
        if proj_rgb_path is not None:
            with h5py.File(proj_rgb_path, "r") as _f:
                proj_rgb_frame = _f["rgb"][frame_idx]  # (EH, EW, 3) uint8
            prh, prw = int(proj_rgb_frame.shape[0]), int(proj_rgb_frame.shape[1])
            ax.imshow(proj_rgb_frame)
            _annotate(ax, prw, prh, "Projected RGB")
        else:
            _na(ax, "Projected RGB")

        # Row label on the left
        axes[raw_row, 0].set_ylabel(
            f"sample {sample_i}  |  RAW", fontsize=9, rotation=90, labelpad=6
        )
        axes[raw_row, 0].axis("off")  # restore after set_ylabel touched it
        _annotate(axes[raw_row, 0], depth_W, depth_H, "Depth (Realsense)", frame_idx=frame_idx)

        # ================================================================
        # PREPROCESSED ROW  (resize → center-crop applied to event-space data)
        # ================================================================

        # --- col 0: projected depth after preprocessing ---
        ax = axes[pre_row, 0]
        if proj_d is not None:
            pd_pre = _apply_resize_crop(proj_d)
            ph, pw = pd_pre.shape
            ax.imshow(pd_pre, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, pw, ph, f"Proj Depth  [{preproc_label}]")
        else:
            _na(ax, f"Proj Depth  [{preproc_label}]")

        # --- col 1: event plane mask after preprocessing ---
        ax = axes[pre_row, 1]
        if proj_mask_arr is not None:
            pm_pre = _apply_resize_crop(proj_mask_arr)
            ph, pw = pm_pre.shape
            ax.imshow(pm_pre, cmap="gray", vmin=0, vmax=1)
            _annotate(ax, pw, ph, f"Event Mask  [{preproc_label}]")
        else:
            _na(ax, f"Event Mask  [{preproc_label}]")

        # --- col 2: accumulated events after preprocessing ---
        ax = axes[pre_row, 2]
        if voxel is not None:
            accum_pre = _apply_resize_crop(voxel.sum(axis=0))
            eh, ew = accum_pre.shape
            vmax_ev2 = float(max(np.abs(accum_pre).max(), 1e-6))
            ax.imshow(accum_pre, cmap="gray", vmin=-vmax_ev2, vmax=vmax_ev2)
            _annotate(ax, ew, eh, f"Events  [{preproc_label}]")
        else:
            _na(ax, f"Events  [{preproc_label}]")

        # --- col 3: RGB after preprocessing ---
        ax = axes[pre_row, 3]
        if rgb_frame_raw is not None:
            rgb_pre = _apply_resize_crop(rgb_frame_raw)
            rh2, rw2 = int(rgb_pre.shape[0]), int(rgb_pre.shape[1])
            ax.imshow(rgb_pre)
            _annotate(ax, rw2, rh2, f"RGB  [{preproc_label}]")
        else:
            _na(ax, f"RGB  [{preproc_label}]")

        # --- col 4: pose depth (raw, event resolution) ---
        ax = axes[pre_row, 4]
        pose_depth_raw: Optional[np.ndarray] = None
        if pose_voxels_dir is not None:
            pvp = pose_voxels_dir / f"voxel_{frame_idx:06d}.npy"
            if pvp.exists():
                pose_voxel = np.load(pvp)
                pose_depth_raw = pose_voxel[-1]  # last channel is pose depth
                pdh0, pdw0 = pose_depth_raw.shape
                ax.imshow(pose_depth_raw, cmap="viridis", vmin=0, vmax=1)
                _annotate(ax, pdw0, pdh0, "Pose Depth (raw)")
            else:
                _na(ax, "Pose Depth (raw)")
        else:
            _na(ax, "Pose Depth (raw)")

        # --- col 5: pose depth (preprocessed) ---
        ax = axes[pre_row, 5]
        if pose_depth_raw is not None:
            pd_pre2 = _apply_resize_crop(pose_depth_raw)
            pdh, pdw = pd_pre2.shape
            ax.imshow(pd_pre2, cmap="viridis", vmin=0, vmax=1)
            _annotate(ax, pdw, pdh, f"Pose Depth  [{preproc_label}]")
        else:
            _na(ax, f"Pose Depth  [{preproc_label}]")

        axes[pre_row, 0].set_ylabel(
            f"sample {sample_i}  |  PREPROCESSED", fontsize=9, rotation=90, labelpad=6
        )
        axes[pre_row, 0].axis("off")
        if proj_d is not None:
            _annotate(axes[pre_row, 0], pw, ph, f"Proj Depth  [{preproc_label}]")
        else:
            _na(axes[pre_row, 0], f"Proj Depth  [{preproc_label}]")

        # --- col 6: total mask (depth mask & ~white RGB mask) after preprocessing ---
        ax = axes[pre_row, 6]
        if proj_mask_arr is not None:
            total_mask = proj_mask_arr.copy()  # depth range mask
            if rgb_mask and proj_rgb_frame is not None:
                white_mask = np.all(proj_rgb_frame > 200, axis=-1)  # (H, W)
                total_mask[white_mask] = 0.0
            if spatial_mask and spatial_mask_path is not None:
                with h5py.File(spatial_mask_path, 'r') as _sf:
                    sp_raw = _sf["mask"][frame_idx].astype(np.float32)
                total_mask[sp_raw == 0] = 0.0
            tm_pre = _apply_resize_crop(total_mask)
            tmh, tmw = tm_pre.shape
            ax.imshow(tm_pre, cmap="gray", vmin=0, vmax=1)
            parts = ["depth"]
            if rgb_mask and proj_rgb_frame is not None:
                parts.append("RGB")
            if spatial_mask and spatial_mask_path is not None:
                parts.append("spatial")
            label = f"Total Mask ({' + '.join(parts)})"
            _annotate(ax, tmw, tmh, f"{label}  [{preproc_label}]")
        else:
            _na(ax, f"Total Mask  [{preproc_label}]")

    fig.suptitle(
        f"Debug — {seq_dir.name}   "
        f"(HDF frame indices: {', '.join(str(i) for i in frame_indices)})  "
        f"— preprocessing: {preproc_label}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_viz] Saved → {out_path}")


# -----------------------------
# GPU Monitoring
# -----------------------------
def get_gpu_stats(device: torch.device) -> Dict[str, float]:
    """Return GPU utilization (%) and VRAM usage (MB) for the given device."""
    stats: Dict[str, float] = {}
    if device.type != "cuda":
        return stats

    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()

    # VRAM via PyTorch (always available)
    stats["vram_used_mb"] = torch.cuda.memory_allocated(gpu_idx) / 1024 ** 2
    stats["vram_reserved_mb"] = torch.cuda.memory_reserved(gpu_idx) / 1024 ** 2

    # GPU utilization via pynvml (optional)
    if _NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["gpu_util_pct"] = float(util.gpu)

    return stats


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(
    model: E2DepthNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_grad: float = 0.5,
    epoch: int = 0,
    log_interval: float = 10.0,
) -> Dict[str, float]:
    """Train for one epoch with sequence processing."""
    model.train()
    
    total_loss = 0.0
    total_si = 0.0
    total_grad = 0.0
    total_mean = 0.0
    n_batches = 0
    n_total = len(loader)

    last_log_time = time.time()
    epoch_start = time.time()

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
        batch_mean = 0.0
        
        for t in range(T):
            pred, states = model(events[:, t], states)
            loss, metrics = e2depth_loss(pred, depths[:, t], masks[:, t], lambda_grad)
            batch_loss += loss
            batch_si += metrics["si"]
            batch_grad += metrics["grad"]
            batch_mean += metrics["mean"]
        
        # Average over sequence
        batch_loss = batch_loss / T
        
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_si += batch_si / T
        total_grad += batch_grad / T
        total_mean += batch_mean / T
        n_batches += 1

        now = time.time()
        if now - last_log_time >= log_interval:
            elapsed = now - epoch_start
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0
            eta_sec = (n_total - n_batches) / batches_per_sec if batches_per_sec > 0 else 0
            avg_loss = total_loss / n_batches
            print(
                f"  [Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {avg_loss:.5f} "
                f"| {batches_per_sec:.1f} batch/s "
                f"| ETA: {int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}",
                flush=True,
            )
            last_log_time = now

    return {
        "total": total_loss / max(1, n_batches),
        "si": total_si / max(1, n_batches),
        "grad": total_grad / max(1, n_batches),
        "mean": total_mean / max(1, n_batches),
    }


@torch.no_grad()
def validate(
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
    log_depth: bool = False,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
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
        if log_depth:
            pred_metric = log_normalized_to_depth(pred)
            gt_metric = log_normalized_to_depth(gt)
        else:
            pred_metric = linear_normalized_to_depth(pred, d_min=depth_min, d_max=depth_max)
            gt_metric = linear_normalized_to_depth(gt, d_min=depth_min, d_max=depth_max)
        
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


def _depth_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """Apply turbo colormap to a (1, H, W) or (H, W) float tensor in [0, 1].

    Returns a (3, H, W) float32 tensor suitable for writer.add_image.
    """
    import matplotlib.cm as cm

    arr = t[0].cpu().float().numpy() if t.dim() == 3 else t.cpu().float().numpy()
    rgba = cm.turbo(arr)  # (H, W, 4) float64
    rgb = rgba[:, :, :3].transpose(2, 0, 1).astype("float32")  # (3, H, W)
    return torch.from_numpy(rgb)


def log_images(
    writer: SummaryWriter,
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    split: str = "val",
    n_images: int = 3,
):
    """Log n_images sample predictions to TensorBoard under '<split>/sample_N/…'."""
    from torch.utils.data import Subset

    model.eval()

    dataset = loader.dataset
    n = len(dataset)
    indices = torch.randperm(n)[:n_images].tolist()
    subset_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=n_images,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn if hasattr(loader, "collate_fn") and loader.collate_fn is not None else None,
    )

    with torch.no_grad():
        for events, depths, masks in subset_loader:
            B, T = events.shape[:2]
            events = events.to(device)
            depths = depths.to(device)
            masks = masks.to(device)

            # Process full sequence
            states = None
            for t in range(T):
                pred, states = model(events[:, t], states)

            # Log last frame predictions
            for j in range(B):
                ev_img = events[j, -1]
                if ev_img.shape[0] >= 3:
                    ev_img = ev_img[:3]
                else:
                    ev_img = ev_img[0:1]

                ev_img = (ev_img - ev_img.min()) / (ev_img.max() - ev_img.min() + 1e-6)
                mask_img = masks[j, -1]          # total mask (depth + rgb if enabled)
                gt_img = depths[j, -1] * mask_img
                pred_img = pred[j] * mask_img
                err_img = torch.abs(pred[j] - depths[j, -1]) * mask_img

                tag = f"{split}/sample_{j}"
                writer.add_image(f"{tag}/input_events", ev_img, epoch)
                writer.add_image(f"{tag}/gt_depth", _depth_to_rgb(gt_img), epoch)
                writer.add_image(f"{tag}/pred_depth", _depth_to_rgb(pred_img), epoch)
                writer.add_image(f"{tag}/total_mask", mask_img, epoch)
                writer.add_image(f"{tag}/error", _depth_to_rgb(err_img / (err_img.max() + 1e-6)), epoch)


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
    data_group.add_argument("--val_ratio", type=float, default=0.2,
                           help="Fraction of data for validation (object-level split when >1 sequence, temporal otherwise)")
    data_group.add_argument("--num_bins", type=int, default=5,
                           help="Number of temporal bins for voxel grid")
    data_group.add_argument("--depth_max", type=float, default=D_MAX,
                           help="Maximum depth in meters")
    data_group.add_argument("--depth_min", type=float, default=0.05,
                           help="Minimum depth in meters")
    data_group.add_argument("--use_pose", action="store_true",
                           help="Use precomputed pose depth channel (from voxels_pose_cam0/)")
    data_group.add_argument("--rgb_mask", action="store_true",
                           help="Mask out white pixels using projected RGB (from rgb_in_event_frame.h5)")
    data_group.add_argument("--spatial_mask", action="store_true",
                           help="Mask pixels outside cube around EE (from spatial_mask.h5, "
                                "run precompute_spatial_mask.py first)")
    data_group.add_argument("--log_depth", action="store_true",
                           help="Use log depth encoding (paper method); default is linear normalization")
    
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
    train_group.add_argument("--epochs", type=int, default=50,
                            help="Number of epochs")
    train_group.add_argument("--batch", type=int, default=10,
                            help="Batch size")
    train_group.add_argument("--seq_len", type=int, default=10,
                            help="Sequence length for recurrent training")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
    train_group.add_argument("--lambda_grad", type=float, default=0.5,
                            help="Weight for gradient loss (λ in paper)")
    train_group.add_argument("--num_workers", type=int, default=8)
    train_group.add_argument("--crop_h", type=int, default=240,
                            help="Center crop height in pixels (0=no crop)")
    train_group.add_argument("--crop_w", type=int, default=320,
                            help="Center crop width in pixels (0=no crop)")
    train_group.add_argument("--resize_h", type=int, default=288,
                            help="Resize height before crop/augmentation (0=no resize)")
    train_group.add_argument("--resize_w", type=int, default=384,
                            help="Resize width before crop/augmentation (0=no resize)")
    
    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    out_group.add_argument("--save_every", type=int, default=10,
                          help="Save checkpoint every N epochs")
    out_group.add_argument("--resume", type=str, default=None,
                          help="Path to checkpoint to resume from")

    # Debug
    dbg_group = parser.add_argument_group("Debug")
    dbg_group.add_argument("--debug_viz", action="store_true",
                           help="Generate a debug PNG of sample inputs and exit (no training)")
    dbg_group.add_argument("--debug_n", type=int, default=3,
                           help="Number of random timestamps shown in debug PNG")
    dbg_group.add_argument("--debug_out", type=str, default="debug_viz.png",
                           help="Output path for the debug PNG")
    dbg_group.add_argument("--debug_seed", type=int, default=None,
                           help="Random seed for timestamp sampling in debug mode")
    dbg_group.add_argument("--debug_seq", type=str, default=None,
                           help="Specific sequence directory to visualize "
                                "(default: first found via --data_dir / --data_root)")

    args = parser.parse_args()
    
    # Find sequences
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequences found in {args.data_root}")
            print("Expected: <dir>/hdf5/realsense.h5 + <dir>/events/voxels_cam0/")
            return
    
    print(f"Found {len(sequence_dirs)} sequences:")
    for d in sequence_dirs:
        print(f"  - {d.name}")

    # Debug visualization mode — generate PNG and exit
    if args.debug_viz:
        if args.debug_seq:
            dbg_seq = Path(args.debug_seq)
        else:
            dbg_seq = sequence_dirs[0]
        dbg_resize_hw = None
        if args.resize_h > 0 and args.resize_w > 0:
            dbg_resize_hw = (args.resize_h, args.resize_w)
        dbg_crop_size = None
        if args.crop_h > 0 and args.crop_w > 0:
            dbg_crop_size = (args.crop_h, args.crop_w)
        debug_visualize(
            str(dbg_seq),
            n_samples=args.debug_n,
            out_path=args.debug_out,
            num_bins=args.num_bins,
            seed=args.debug_seed,
            resize_hw=dbg_resize_hw,
            crop_size=dbg_crop_size,
            use_pose=args.use_pose,
            rgb_mask=args.rgb_mask,
            spatial_mask=args.spatial_mask,
        )
        return

    # Config
    crop_size = None
    if args.crop_h > 0 and args.crop_w > 0:
        crop_size = (args.crop_h, args.crop_w)

    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (args.resize_h, args.resize_w)
    
    cfg = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        resize_hw=resize_hw,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        augment=True,
        num_bins=args.num_bins,
        use_pose=args.use_pose,
        rgb_mask=args.rgb_mask,
        spatial_mask=args.spatial_mask,
        log_depth=args.log_depth,
    )
    
    # Create config without augmentation for validation
    cfg_val = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        resize_hw=resize_hw,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        augment=False,
        num_bins=args.num_bins,
        use_pose=args.use_pose,
        rgb_mask=args.rgb_mask,
        spatial_mask=args.spatial_mask,
        log_depth=args.log_depth,
    )
    
    # Datasets — object-level split when multiple sequences are available,
    # fallback to temporal block-split within the single sequence.
    if len(sequence_dirs) > 1:
        rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)
        dirs_shuffled = list(sequence_dirs)
        rng.shuffle(dirs_shuffled)
        n_val_dirs = max(1, int(round(len(dirs_shuffled) * args.val_ratio)))
        val_dirs = dirs_shuffled[:n_val_dirs]
        train_dirs = dirs_shuffled[n_val_dirs:]
        if not train_dirs:
            # Edge case: only 1 dir total, fall through to temporal split
            train_dirs = val_dirs
        print(f"  Object-level split: {len(train_dirs)} train dirs, {len(val_dirs)} val dirs")
        train_ds = create_multi_sequence_dataset(
            [str(d) for d in train_dirs], cfg, split="train", val_ratio=0.0
        )
        val_ds = create_multi_sequence_dataset(
            [str(d) for d in val_dirs], cfg_val, split="val", val_ratio=1.0
        )
    else:
        # Single sequence: temporal block-split within that sequence
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
    print(f"  Depth encoding: {'log' if args.log_depth else 'linear'}")
    print(f"  Voxel bins: {args.num_bins}")
    print(f"  Use pose: {args.use_pose}")
    print(f"  RGB mask: {args.rgb_mask}")
    print(f"{'='*60}\n")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Input channels = num_bins (+ 1 if using pose)
    in_channels = args.num_bins + (1 if args.use_pose else 0)
    print(f"Input channels: {in_channels} (bins={args.num_bins}, pose={'yes' if args.use_pose else 'no'})")
    
    # Model
    model = E2DepthNet(
        in_channels=in_channels,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
    ).to(device)

    #model = torch.compile(model)
    
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
            model, train_loader, optimizer, device, lambda_grad=args.lambda_grad, epoch=epoch
        )
        val_metrics = validate(model, val_loader, device,
                               log_depth=args.log_depth,
                               depth_min=args.depth_min,
                               depth_max=args.depth_max)
        
        # Step scheduler based on validation loss
        scheduler.step(val_metrics["si"])
        
        # Logging
        writer.add_scalar("loss/train_total", train_metrics["total"], epoch)
        writer.add_scalar("loss/train_si", train_metrics["si"], epoch)
        writer.add_scalar("loss/train_grad", train_metrics["grad"], epoch)
        writer.add_scalar("loss/train_mean", train_metrics["mean"], epoch)
        writer.add_scalar("loss/val_si", val_metrics["si"], epoch)
        writer.add_scalar("loss/val_l1_metric", val_metrics["l1_metric"], epoch)
        writer.add_scalar("loss/val_abs_rel", val_metrics["abs_rel"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # GPU stats
        gpu_stats = get_gpu_stats(device)
        if gpu_stats:
            writer.add_scalar("gpu/vram_used_mb", gpu_stats["vram_used_mb"], epoch)
            writer.add_scalar("gpu/vram_reserved_mb", gpu_stats["vram_reserved_mb"], epoch)
            if "gpu_util_pct" in gpu_stats:
                writer.add_scalar("gpu/utilization_pct", gpu_stats["gpu_util_pct"], epoch)

        print(f"Epoch {epoch:03d} | train: {train_metrics['total']:.5f} "
              f"| val SI: {val_metrics['si']:.5f} | val L1: {val_metrics['l1_metric']:.3f}m "
              f"| val AbsRel: {val_metrics['abs_rel']:.4f}")
        print(f"          | pred_log: {val_metrics['pred_log_mean']:.3f} vs gt_log: {val_metrics['gt_log_mean']:.3f} "
              f"| pred_m: {val_metrics['pred_metric_mean']:.3f}m vs gt_m: {val_metrics['gt_metric_mean']:.3f}m")
        if gpu_stats:
            util_str = f" | GPU util: {gpu_stats['gpu_util_pct']:.0f}%" if "gpu_util_pct" in gpu_stats else ""
            print(f"          | VRAM: {gpu_stats['vram_used_mb']:.0f}/{gpu_stats['vram_reserved_mb']:.0f} MB (used/reserved){util_str}")
        
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
                    "use_pose": args.use_pose,
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
            log_images(writer, model, val_loader, device, epoch, split="val", n_images=3)
            log_images(writer, model, train_loader, device, epoch, split="train", n_images=3)
    
    writer.close()
    print(f"\nTraining complete! Best val SI: {best_val_loss:.5f}")
    print(f"Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
