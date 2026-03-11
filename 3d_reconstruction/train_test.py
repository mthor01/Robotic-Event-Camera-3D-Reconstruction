"""
Train E2Depth on CARLA DENSE data: Recurrent UNet for Event-to-Depth prediction.

Faithful implementation of:
"Learning Monocular Dense Depth from Events" (Hidalgo-Carrió, Gehrig, Scaramuzza)
IEEE International Conference on 3D Vision (3DV), 2020.

Paper hyperparameters:
    - Resolution: 346 x 260 (DAVIS346B)
    - Voxel grid: B=5 temporal bins, ΔT=50ms
    - D_max = 80m, α = 3.7 (min depth ≈ 2m)
    - Architecture: NE=3 encoders, NR=2 residual blocks, Nb=32 base filters
    - Unroll length: L=40 steps
    - Batch size: 20, learning rate: 1e-4 (Adam)
    - Loss: L_si + 0.5 * L_grad (scale-invariant + multi-scale gradient)
    - Data augmentation: random crop + horizontal flip
    - Training: 300 epochs

Supported data formats (auto-detected per sequence directory):

  1. CARLA DENSE format (default: data/test_data/):
        <seq>/depth/data/depth_NNNNNNNNNN.npy          # (260, 346) float64, meters
        <seq>/events/voxels/event_tensor_NNNNNNNNNN.npy  # (5, 260, 346) float32
     Recommended: --depth_max 80 --depth_min 2.0

  2. Synthetic HDF5 format (tabletop: data/synthetic_data/):
        <seq>/hdf5/depth.h5                             # 'realsense/depth' (N, 260, 346) uint16 mm
        <seq>/events/voxels/voxel_NNNNNN.npy            # (5, 260, 346) float32
     Recommended: --depth_max 2.0 --depth_min 0.05 --sky_threshold 10.0

IMPORTANT: Do not mix both formats in one training run (different depth ranges
give incompatible log-normalized targets).

Usage:
    # CARLA DENSE (paper settings):
    python train_test.py --data_root data/test_data

    # Tabletop synthetic data:
    python train_test.py --data_root data/synthetic_data \
                         --depth_max 2.0 --depth_min 0.05 --sky_threshold 10.0 \
                         --out_dir checkpoints_e2depth_synthetic

TensorBoard:
    tensorboard --logdir checkpoints_e2depth_carla/runs
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
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
DATA_ROOT = Path("data/test_data")
DEFAULT_OUT_DIR = Path("checkpoints_e2depth_carla")
# =================================================

# ================= DEPTH PARAMETERS (paper Section 3.3) =================
D_MAX = 80.0   # Maximum expected depth in meters
ALPHA = 3.7    # Log depth parameter → min depth ≈ D_MAX * exp(-α) ≈ 2m
# D_metric = D_MAX * exp(-α * (1 - D_pred))
# At D_pred=0: D_metric = D_MAX * exp(-3.7) ≈ 1.97m
# At D_pred=1: D_metric = D_MAX = 80.0m
# =====================================================================

# ================= SENSOR PARAMETERS (DAVIS346B) =================
SENSOR_HEIGHT = 260
SENSOR_WIDTH = 346
# =================================================================


# ─────────────────────────────────────────────────────────────────
# ConvLSTM Cell
# ─────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.

    Paper §3.2: "Each encoder layer is composed of a downsampling convolution
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
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


# ─────────────────────────────────────────────────────────────────
# Network Architecture (Paper §3.2 / Figure 2)
# ─────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Residual block with kernel size 3 and summation skip (paper §3.2)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class EncoderLayer(nn.Module):
    """
    Encoder: downsampling conv (k=5, s=2) → BN → ReLU → ConvLSTM (k=3).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTMCell(out_channels, out_channels, kernel_size=3)

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.downsample(x)
        h, new_state = self.convlstm(x, state)
        return h, new_state


class DecoderLayer(nn.Module):
    """
    Decoder: bilinear upsample → concat skip → conv k=5 → BN → ReLU.
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class E2DepthNet(nn.Module):
    """
    E2Depth: Recurrent UNet for event-to-depth prediction.

    Paper §3.2:
    - Head layer H → Nb channels
    - NE=3 encoder layers with ConvLSTM (channels doubled each layer)
    - NR=2 residual blocks at bottleneck
    - NE=3 decoder layers with bilinear upsampling
    - Prediction layer P: 1×1 conv → sigmoid  (normalized log depth ∈ [0,1])
    """

    def __init__(
        self,
        in_channels: int = 5,     # B = 5 voxel grid bins
        base: int = 32,           # Nb = 32
        num_encoders: int = 3,    # NE = 3
        num_residuals: int = 2,   # NR = 2
    ):
        super().__init__()
        self.num_encoders = num_encoders

        # Head layer
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Encoder layers (channels: base → 2*base → 4*base → 8*base)
        self.encoders = nn.ModuleList()
        ch = base
        for _ in range(num_encoders):
            out_ch = ch * 2
            self.encoders.append(EncoderLayer(ch, out_ch))
            ch = out_ch

        # Residual blocks at bottleneck
        self.residuals = nn.ModuleList([ResidualBlock(ch) for _ in range(num_residuals)])

        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            out_ch = ch // 2
            self.decoders.append(DecoderLayer(ch, skip_ch, out_ch))
            ch = out_ch

        # Prediction layer: depth-wise 1×1 conv → sigmoid
        self.pred = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Single-step forward with recurrent state.

        Args:
            x: (B, C, H, W) voxel grid
            states: list of (h, c) per encoder, or None

        Returns:
            pred: (B, 1, H, W) normalized log depth ∈ [0, 1]
            new_states: updated recurrent states
        """
        if states is None:
            states = [None] * self.num_encoders

        x = self.head(x)
        skips = [x]  # skip from head for last decoder
        new_states = []

        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, states[i])
            new_states.append(state)
            if i < self.num_encoders - 1:
                skips.append(x)

        for residual in self.residuals:
            x = residual(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        pred = self.pred(x)
        return pred, new_states

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process a sequence of T frames with persistent recurrent state.

        Args:
            sequence: (B, T, C, H, W)
            states: initial states or None

        Returns:
            predictions: list of T tensors each (B, 1, H, W)
            final_states: states after the last step
        """
        B, T, C, H, W = sequence.shape
        predictions = []
        for t in range(T):
            pred, states = self.forward(sequence[:, t], states)
            predictions.append(pred)
        return predictions, states


# ─────────────────────────────────────────────────────────────────
# Loss Functions (Paper §3.4, Equations 3-5)
# ─────────────────────────────────────────────────────────────────
def scale_invariant_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant loss (Eq. 3):
        L_si = (1/n) Σ R² − (1/n²) (Σ R)²
    where R = D̂ − D (in normalized log depth space).
    """
    diff = (pred - gt) * mask
    n = mask.sum().clamp_min(1.0)
    return (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)


def multi_scale_gradient_loss(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, num_scales: int = 4
) -> torch.Tensor:
    """
    Multi-scale scale-invariant gradient matching loss (Eq. 4):
        L_grad = (1/n) Σ_s Σ_u |∇_x R^s| + |∇_y R^s|
    Using L1 to enforce sharp depth discontinuities.
    """

    def grad_x(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, :, 1:] - t[:, :, :, :-1]

    def grad_y(t: torch.Tensor) -> torch.Tensor:
        return t[:, :, 1:, :] - t[:, :, :-1, :]

    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    for s in range(num_scales):
        if s > 0:
            pred = F.avg_pool2d(pred, 2)
            gt = F.avg_pool2d(gt, 2)
            mask = F.avg_pool2d(mask, 2)
            mask = (mask > 0.5).float()

        residual = pred - gt
        gx = grad_x(residual)
        gy = grad_y(residual)
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = (torch.abs(gx) * mx).sum() / mx.sum().clamp_min(1.0)
        loss_y = (torch.abs(gy) * my).sum() / my.sum().clamp_min(1.0)
        total_loss = total_loss + loss_x + loss_y

    return total_loss / num_scales


def e2depth_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    lambda_grad: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss (Eq. 5):
        L_tot = Σ_k  L_{k,si} + λ L_{k,grad}
    with λ = 0.5 chosen by cross-validation.
    """
    l_si = scale_invariant_loss(pred, gt, mask)
    l_grad = multi_scale_gradient_loss(pred, gt, mask)
    total = l_si + lambda_grad * l_grad
    return total, {"si": l_si.item(), "grad": l_grad.item(), "total": total.item()}


# ─────────────────────────────────────────────────────────────────
# Depth Conversion (Paper §3.3, Equation 2)
# ─────────────────────────────────────────────────────────────────
def depth_to_log_normalized(
    depth: torch.Tensor, d_max: float = D_MAX, alpha: float = ALPHA
) -> torch.Tensor:
    """
    Convert metric depth → normalized log depth ∈ [0, 1].
    Inverse of Eq. 2:  D̂ = 1 + (1/α) ln(D / D_max)
    """
    depth = depth.clamp_min(1e-6)
    return (1.0 + (1.0 / alpha) * torch.log(depth / d_max)).clamp(0.0, 1.0)


def log_normalized_to_depth(
    pred: torch.Tensor, d_max: float = D_MAX, alpha: float = ALPHA
) -> torch.Tensor:
    """
    Convert normalized log depth [0, 1] → metric depth (Eq. 2):
        D_m = D_max · exp(−α (1 − D̂))
    """
    return d_max * torch.exp(-alpha * (1.0 - pred))


# ─────────────────────────────────────────────────────────────────
# Dataset for CARLA DENSE format
# ─────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Configuration for CARLA DENSE data loading."""
    seq_len: int = 40         # L=40 unroll steps (paper §3.2)
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) random crop
    depth_max: float = D_MAX  # 80m
    depth_min: float = 2.0    # ~D_MAX * exp(-α) ≈ 2m
    sky_threshold: float = 500.0  # Depth values above this are sky (CARLA uses 1000)
    augment: bool = True
    num_bins: int = 5         # B=5 temporal bins


class CARLADenseDataset(Dataset):
    """
    Dataset for CARLA DENSE per-frame data.

    Each sequence directory contains:
        depth/data/depth_NNNNNNNNNN.npy       → (260, 346) float64 depth in meters
        events/voxels/event_tensor_NNNNNNNNNN.npy  → (5, 260, 346) float32 voxel grid
    """

    def __init__(
        self,
        sequence_dir: str,
        cfg: DataConfig,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.split = split

        # Locate files
        self.voxel_dir = self.sequence_dir / "events" / "voxels"
        self.depth_dir = self.sequence_dir / "depth" / "data"

        if not self.voxel_dir.exists():
            raise FileNotFoundError(f"Voxel directory not found: {self.voxel_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")

        # Count matching frames
        self.voxel_files = sorted(self.voxel_dir.glob("event_tensor_*.npy"))
        self.depth_files = sorted(self.depth_dir.glob("depth_*.npy"))
        self.n_frames = min(len(self.voxel_files), len(self.depth_files))

        if self.n_frames == 0:
            raise RuntimeError(f"No matching frames found in {self.sequence_dir}")

        # Detect resolution from first voxel
        sample_voxel = np.load(self.voxel_files[0])
        self.num_bins, self.H, self.W = sample_voxel.shape

        # Compute valid sequence start indices and split
        self._compute_indices(train_ratio, seed)

        print(
            f"[{self.sequence_dir.name}] {split}: {len(self.indices)} sequences "
            f"(frames: {self.n_frames}, seq_len: {cfg.seq_len}, "
            f"res: {self.W}×{self.H})"
        )

    def _compute_indices(self, train_ratio: float, seed: int):
        """Split into train/val ensuring no temporal overlap."""
        seq_len = self.cfg.seq_len
        max_start = self.n_frames - seq_len
        if max_start < 0:
            raise RuntimeError(
                f"Not enough frames ({self.n_frames}) for seq_len={seq_len}"
            )

        all_starts = np.arange(0, max_start + 1)
        n_total = len(all_starts)

        # Block-based split to avoid temporal leakage between train/val
        rng = np.random.default_rng(seed)
        block_size = max(20, seq_len)
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)

        n_train_blocks = int(round(n_blocks * train_ratio))
        train_blocks = set(block_ids[:n_train_blocks].tolist())

        train_indices = []
        val_indices = []
        for idx in all_starts:
            block = idx // block_size
            if block in train_blocks:
                train_indices.append(idx)
            else:
                val_indices.append(idx)

        if self.split == "train":
            self.indices = np.array(train_indices)
        else:
            self.indices = np.array(val_indices)

    def _load_voxel(self, frame_idx: int) -> np.ndarray:
        """Load precomputed voxel grid (5, 260, 346)."""
        path = self.voxel_dir / f"event_tensor_{frame_idx:010d}.npy"
        return np.load(path).astype(np.float32)

    def _load_depth(self, frame_idx: int) -> np.ndarray:
        """Load depth map (260, 346) in meters."""
        path = self.depth_dir / f"depth_{frame_idx:010d}.npy"
        return np.load(path).astype(np.float64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        start = int(self.indices[i])
        seq_len = self.cfg.seq_len

        voxels_seq = []
        depths_seq = []
        masks_seq = []

        for t in range(seq_len):
            idx = start + t

            # Load precomputed voxel grid
            voxel = self._load_voxel(idx)  # (5, H, W)

            # Load depth
            depth = self._load_depth(idx)  # (H, W)

            # Validity mask: exclude sky (1000m in CARLA) and very close depths
            valid = (depth > self.cfg.depth_min) & (depth < self.cfg.sky_threshold)
            mask = valid.astype(np.float32)

            # Clip depth to [depth_min, depth_max] and convert to log normalized
            depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
            log_depth = 1.0 + (1.0 / ALPHA) * np.log(depth / self.cfg.depth_max)
            log_depth = np.clip(log_depth, 0.0, 1.0)

            voxels_seq.append(voxel)
            depths_seq.append(log_depth[None])  # (1, H, W)
            masks_seq.append(mask[None])         # (1, H, W)

        # Stack: (T, C, H, W) and (T, 1, H, W)
        voxels = np.stack(voxels_seq, axis=0)   # (T, 5, H, W)
        depths = np.stack(depths_seq, axis=0)    # (T, 1, H, W)
        masks = np.stack(masks_seq, axis=0)      # (T, 1, H, W)

        # ── Data augmentation (paper §4: random crop + horizontal flip) ──
        if self.cfg.augment and self.split == "train":
            # Random crop
            if self.cfg.crop_size is not None:
                ch, cw = self.cfg.crop_size
                if self.H > ch and self.W > cw:
                    y0 = np.random.randint(0, self.H - ch + 1)
                    x0 = np.random.randint(0, self.W - cw + 1)
                    voxels = voxels[:, :, y0 : y0 + ch, x0 : x0 + cw]
                    depths = depths[:, :, y0 : y0 + ch, x0 : x0 + cw]
                    masks = masks[:, :, y0 : y0 + ch, x0 : x0 + cw]

            # Random horizontal flip
            if np.random.rand() > 0.5:
                voxels = np.flip(voxels, axis=3).copy()
                depths = np.flip(depths, axis=3).copy()
                masks = np.flip(masks, axis=3).copy()

        return (
            torch.from_numpy(voxels).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(masks).float(),
        )


# ─────────────────────────────────────────────────────────────────
# Dataset for Synthetic HDF5 format (tabletop robotic data)
# ─────────────────────────────────────────────────────────────────
class SyntheticHDF5Dataset(Dataset):
    """
    Dataset for tabletop synthetic data (data/synthetic_data/).

    Each sequence directory contains:
        hdf5/depth.h5                       → 'realsense/depth' (N, H, W) uint16 mm
        events/voxels/voxel_NNNNNN.npy      → (5, H, W) float32 voxel grid

    Typical depth range: 0.04–2.0m (tabletop robot workspace).
    Set DataConfig with depth_max=2.0, depth_min=0.05, sky_threshold=10.0.

    Note: voxel count may exceed depth frame count by a few frames; the
    surplus is dropped (we use min(n_voxels, n_depth_frames)).
    """

    def __init__(
        self,
        sequence_dir: str,
        cfg: DataConfig,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        super().__init__()
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.split = split

        # Locate files
        self.voxel_dir = self.sequence_dir / "events" / "voxels"
        self.depth_h5 = self.sequence_dir / "hdf5" / "depth.h5"

        if not self.voxel_dir.exists():
            raise FileNotFoundError(f"Voxel directory not found: {self.voxel_dir}")
        if not self.depth_h5.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5}")

        # Count voxel files (6-digit naming: voxel_NNNNNN.npy)
        self.voxel_files = sorted(self.voxel_dir.glob("voxel_*.npy"))
        n_voxels = len(self.voxel_files)
        if n_voxels == 0:
            raise RuntimeError(
                f"No voxel files in {self.voxel_dir}. "
                "Run precompute_voxels.py first."
            )

        # Read only shape metadata — depth is loaded lazily in _load_depth.
        # Caching all frames in __init__ would be copied into every DataLoader
        # worker process (fork), causing gigabytes of unnecessary RAM use and
        # long startup times.
        with h5py.File(self.depth_h5, "r") as f:
            shape = f["realsense/depth"].shape  # (N, H, W)
            n_depth = shape[0]
            self.H, self.W = shape[1], shape[2]
            self.n_frames = min(n_voxels, n_depth)

        # Detect voxel bins
        self.num_bins = np.load(self.voxel_files[0]).shape[0]

        # Compute valid sequence start indices and split
        self._compute_indices(train_ratio, seed)

        print(
            f"[{self.sequence_dir.name}] {split}: {len(self.indices)} sequences "
            f"(frames: {self.n_frames} = min({n_voxels} voxels, {n_depth} depths), "
            f"seq_len: {cfg.seq_len}, res: {self.W}×{self.H})"
        )

    def _compute_indices(self, train_ratio: float, seed: int):
        """Split into train/val ensuring no temporal overlap (block-based)."""
        seq_len = self.cfg.seq_len
        max_start = self.n_frames - seq_len
        if max_start < 0:
            raise RuntimeError(
                f"Not enough frames ({self.n_frames}) for seq_len={seq_len}"
            )

        all_starts = np.arange(0, max_start + 1)
        n_total = len(all_starts)

        rng = np.random.default_rng(seed)
        block_size = max(20, seq_len)
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)

        n_train_blocks = int(round(n_blocks * train_ratio))
        train_blocks = set(block_ids[:n_train_blocks].tolist())

        train_indices, val_indices = [], []
        for idx in all_starts:
            block = idx // block_size
            if block in train_blocks:
                train_indices.append(idx)
            else:
                val_indices.append(idx)

        self.indices = np.array(train_indices if self.split == "train" else val_indices)

    def _load_voxel(self, frame_idx: int) -> np.ndarray:
        """Load precomputed voxel grid (5, H, W). Named voxel_NNNNNN.npy."""
        return np.load(self.voxel_dir / f"voxel_{frame_idx:06d}.npy").astype(np.float32)

    def _load_depth(self, frame_idx: int) -> np.ndarray:
        """Load one depth frame from HDF5 as float32 meters (uint16 mm → ÷1000).
        Opens the file on each call so it is safe with DataLoader num_workers > 0
        (each forked worker gets its own file handle)."""
        with h5py.File(self.depth_h5, "r") as f:
            return f["realsense/depth"][frame_idx].astype(np.float32) / 1000.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        start = int(self.indices[i])
        seq_len = self.cfg.seq_len

        voxels_seq, depths_seq, masks_seq = [], [], []

        for t in range(seq_len):
            idx = start + t

            voxel = self._load_voxel(idx)   # (5, H, W) float32
            depth = self._load_depth(idx)   # (H, W) float32 meters

            # Validity mask: finite + within [depth_min, depth_max]
            valid = (
                np.isfinite(depth)
                & (depth > self.cfg.depth_min)
                & (depth < self.cfg.depth_max)
            )
            mask = valid.astype(np.float32)

            # Clip and convert to log-normalized depth ∈ [0, 1]
            depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
            log_depth = 1.0 + (1.0 / ALPHA) * np.log(depth / self.cfg.depth_max)
            log_depth = np.clip(log_depth, 0.0, 1.0)

            voxels_seq.append(voxel)
            depths_seq.append(log_depth[None])  # (1, H, W)
            masks_seq.append(mask[None])         # (1, H, W)

        voxels = np.stack(voxels_seq, axis=0)   # (T, 5, H, W)
        depths = np.stack(depths_seq, axis=0)    # (T, 1, H, W)
        masks = np.stack(masks_seq, axis=0)      # (T, 1, H, W)

        # ── Data augmentation: random crop + horizontal flip ──
        if self.cfg.augment and self.split == "train":
            if self.cfg.crop_size is not None:
                ch, cw = self.cfg.crop_size
                if self.H > ch and self.W > cw:
                    y0 = np.random.randint(0, self.H - ch + 1)
                    x0 = np.random.randint(0, self.W - cw + 1)
                    voxels = voxels[:, :, y0 : y0 + ch, x0 : x0 + cw]
                    depths = depths[:, :, y0 : y0 + ch, x0 : x0 + cw]
                    masks = masks[:, :, y0 : y0 + ch, x0 : x0 + cw]

            if np.random.rand() > 0.5:
                voxels = np.flip(voxels, axis=3).copy()
                depths = np.flip(depths, axis=3).copy()
                masks = np.flip(masks, axis=3).copy()

        return (
            torch.from_numpy(voxels).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(masks).float(),
        )


# ─────────────────────────────────────────────────────────────────
# Format detection & dataset factory
# ─────────────────────────────────────────────────────────────────
def detect_format(seq_dir: Path) -> str:
    """
    Auto-detect the dataset format of a sequence directory.

    Returns:
        'synthetic_hdf5' – has hdf5/depth.h5 + events/voxels/voxel_*.npy
        'carla'          – has depth/data/depth_*.npy + events/voxels/event_tensor_*.npy
        'unknown'        – does not match either format
    """
    if (seq_dir / "hdf5" / "depth.h5").exists():
        voxel_dir = seq_dir / "events" / "voxels"
        if voxel_dir.exists() and len(list(voxel_dir.glob("voxel_*.npy"))) > 0:
            return "synthetic_hdf5"
    voxel_dir = seq_dir / "events" / "voxels"
    depth_dir = seq_dir / "depth" / "data"
    if voxel_dir.exists() and depth_dir.exists():
        if (
            len(list(voxel_dir.glob("event_tensor_*.npy"))) > 0
            and len(list(depth_dir.glob("depth_*.npy"))) > 0
        ):
            return "carla"
    return "unknown"


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find all valid sequence directories (CARLA DENSE or Synthetic HDF5) under data_root."""
    dirs = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and detect_format(d) != "unknown":
            dirs.append(d)
    return dirs


def create_datasets(
    sequence_dirs: List[Path],
    cfg: DataConfig,
    cfg_val: DataConfig,
    train_ratio: float = 0.8,
) -> Tuple[Dataset, Dataset]:
    """Create train and validation datasets, auto-detecting format per sequence."""
    train_datasets = []
    val_datasets = []
    for seq_dir in sequence_dirs:
        fmt = detect_format(seq_dir)
        cls = SyntheticHDF5Dataset if fmt == "synthetic_hdf5" else CARLADenseDataset
        try:
            ds_train = cls(str(seq_dir), cfg, split="train", train_ratio=train_ratio)
            ds_val = cls(str(seq_dir), cfg_val, split="val", train_ratio=train_ratio)
            train_datasets.append(ds_train)
            val_datasets.append(ds_val)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir.name} [{fmt}]: {e}")

    if not train_datasets:
        raise RuntimeError("No valid datasets found!")

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────
# Evaluation Metrics (Paper Tables 2-4)
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_depth_metrics(
    pred_metric: torch.Tensor,
    gt_metric: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all depth metrics from Tables 2 & 3 of the paper.

    Returns dict with:
        abs_rel, sq_rel, rmse, rmse_log, si_log,
        delta_1, delta_2, delta_3  (δ < 1.25, 1.25², 1.25³)
        avg_err_10m, avg_err_20m, avg_err_30m
    """
    valid = mask.squeeze(1) > 0.5  # (B, H, W)

    pred = pred_metric.squeeze(1)  # (B, H, W)
    gt = gt_metric.squeeze(1)

    # Gather valid pixels
    p = pred[valid]
    g = gt[valid]

    if p.numel() == 0:
        return {k: 0.0 for k in [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "si_log",
            "delta_1", "delta_2", "delta_3",
            "avg_err_10m", "avg_err_20m", "avg_err_30m",
        ]}

    # Clamp for safe log
    p = p.clamp_min(1e-3)
    g = g.clamp_min(1e-3)
    diff = torch.abs(p - g)

    # Standard metrics
    abs_rel = (diff / g).mean().item()
    sq_rel = ((diff ** 2) / g).mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    log_p = torch.log(p)
    log_g = torch.log(g)
    log_diff = log_p - log_g
    rmse_log = torch.sqrt((log_diff ** 2).mean()).item()
    si_log = (log_diff ** 2).mean().item() - (log_diff.mean() ** 2).item()

    # Threshold accuracy (δ)
    ratio = torch.max(p / g, g / p)
    delta_1 = (ratio < 1.25).float().mean().item()
    delta_2 = (ratio < 1.25 ** 2).float().mean().item()
    delta_3 = (ratio < 1.25 ** 3).float().mean().item()

    # Average absolute error at depth cutoffs (Table 3)
    def avg_err_at_cutoff(cutoff: float) -> float:
        m = valid & (gt_metric.squeeze(1) < cutoff)
        if m.sum() == 0:
            return 0.0
        return torch.abs(pred.squeeze(1)[m] - gt.squeeze(1)[m]).mean().item()

    # Note: we reuse gt (which was clamped to 80m) here but use the original
    # gt_metric for distance thresholding
    avg_10 = avg_err_at_cutoff(10.0)
    avg_20 = avg_err_at_cutoff(20.0)
    avg_30 = avg_err_at_cutoff(30.0)

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "si_log": si_log,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "avg_err_10m": avg_10,
        "avg_err_20m": avg_20,
        "avg_err_30m": avg_30,
    }


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: E2DepthNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_grad: float = 0.5,
) -> Dict[str, float]:
    """Train one epoch, unrolling the recurrent net across the sequence."""
    model.train()
    total_loss = 0.0
    total_si = 0.0
    total_grad = 0.0
    n_batches = 0

    for voxels, depths, masks in loader:
        # voxels: (B, T, C, H, W)  depths/masks: (B, T, 1, H, W)
        B, T = voxels.shape[:2]
        voxels = voxels.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Unroll sequence through recurrent network
        states = None
        seq_loss = torch.tensor(0.0, device=device)
        seq_si = 0.0
        seq_grad = 0.0

        for t in range(T):
            pred, states = model(voxels[:, t], states)
            loss, metrics = e2depth_loss(pred, depths[:, t], masks[:, t], lambda_grad)
            seq_loss = seq_loss + loss
            seq_si += metrics["si"]
            seq_grad += metrics["grad"]

        # Average loss over unroll length (Eq. 5: sum over k)
        seq_loss = seq_loss / T

        optimizer.zero_grad(set_to_none=True)
        seq_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += seq_loss.item()
        total_si += seq_si / T
        total_grad += seq_grad / T
        n_batches += 1

    n = max(1, n_batches)
    return {"total": total_loss / n, "si": total_si / n, "grad": total_grad / n}


@torch.no_grad()
def validate(
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
    d_max: float = D_MAX,
) -> Dict[str, float]:
    """Validate with full depth metrics."""
    model.eval()

    all_metrics = {
        "si_loss": 0.0,
        "abs_rel": 0.0, "sq_rel": 0.0,
        "rmse": 0.0, "rmse_log": 0.0, "si_log": 0.0,
        "delta_1": 0.0, "delta_2": 0.0, "delta_3": 0.0,
        "avg_err_10m": 0.0, "avg_err_20m": 0.0, "avg_err_30m": 0.0,
    }
    n_batches = 0

    for voxels, depths, masks in loader:
        B, T = voxels.shape[:2]
        voxels = voxels.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Unroll full sequence
        states = None
        for t in range(T):
            pred, states = model(voxels[:, t], states)

        # Evaluate on last frame of each sequence
        gt_log = depths[:, -1]
        mask = masks[:, -1]

        all_metrics["si_loss"] += scale_invariant_loss(pred, gt_log, mask).item()

        # Convert to metric depth for paper metrics
        pred_metric = log_normalized_to_depth(pred, d_max=d_max)
        gt_metric = log_normalized_to_depth(gt_log, d_max=d_max)
        batch_metrics = compute_depth_metrics(pred_metric, gt_metric, mask)

        for k, v in batch_metrics.items():
            all_metrics[k] += v
        n_batches += 1

    n = max(1, n_batches)
    return {k: v / n for k, v in all_metrics.items()}


def log_images(
    writer: SummaryWriter,
    model: E2DepthNet,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    d_max: float = D_MAX,
    max_images: int = 4,
):
    """Log sample predictions to TensorBoard."""
    model.eval()
    with torch.no_grad():
        for i, (voxels, depths, masks) in enumerate(loader):
            if i >= 1:
                break
            B, T = voxels.shape[:2]
            voxels = voxels.to(device)
            depths = depths.to(device)
            masks = masks.to(device)

            states = None
            for t in range(T):
                pred, states = model(voxels[:, t], states)

            for j in range(min(max_images, B)):
                # Event voxel visualization (first 3 bins as RGB)
                ev_img = voxels[j, -1, :3]
                ev_img = (ev_img - ev_img.min()) / (ev_img.max() - ev_img.min() + 1e-6)

                gt_img = depths[j, -1]
                pred_img = pred[j]
                err_img = torch.abs(pred[j] - depths[j, -1]) * masks[j, -1]

                writer.add_image(f"sample_{j}/events", ev_img, epoch)
                writer.add_image(f"sample_{j}/gt_log_depth", gt_img, epoch)
                writer.add_image(f"sample_{j}/pred_log_depth", pred_img, epoch)
                writer.add_image(
                    f"sample_{j}/error", err_img / (err_img.max() + 1e-6), epoch
                )

                # Also log metric depth
                pred_m = log_normalized_to_depth(pred[j], d_max=d_max)
                gt_m = log_normalized_to_depth(depths[j, -1], d_max=d_max)
                writer.add_image(
                    f"sample_{j}/pred_metric_depth",
                    pred_m / d_max,
                    epoch,
                )
                writer.add_image(
                    f"sample_{j}/gt_metric_depth",
                    gt_m / d_max,
                    epoch,
                )


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train E2Depth on CARLA DENSE data "
            "(Hidalgo-Carrió et al., 3DV 2020)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_root", type=str, default=str(DATA_ROOT),
        help=(
            "Root dir with sequence subdirectories. Format auto-detected: "
            "CARLA DENSE (depth/data/*.npy) or Synthetic HDF5 (hdf5/depth.h5). "
            "Do NOT mix formats in the same run."
        ),
    )
    data.add_argument(
        "--data_dir", nargs="+", type=str, default=None,
        help="Explicit sequence directory(ies) to train on (overrides --data_root)",
    )
    data.add_argument("--train_ratio", type=float, default=0.8)
    data.add_argument("--num_bins", type=int, default=5, help="B = temporal bins")
    data.add_argument(
        "--depth_max", type=float, default=D_MAX,
        help="D_max in meters. CARLA: 80.0 (default). Tabletop synthetic: 2.0.",
    )
    data.add_argument(
        "--depth_min", type=float, default=2.0,
        help="Min valid depth in meters. CARLA: 2.0 (default). Tabletop synthetic: 0.05.",
    )
    data.add_argument(
        "--sky_threshold", type=float, default=500.0,
        help=(
            "Depths above this are treated as sky/invalid (CARLA only). "
            "For synthetic/indoor data use a large value (e.g. 10.0)."
        ),
    )

    # ── Model (paper defaults) ──
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument("--base", type=int, default=32, help="Nb base filters")
    model_grp.add_argument("--num_encoders", type=int, default=3, help="NE encoder layers")
    model_grp.add_argument("--num_residuals", type=int, default=2, help="NR residual blocks")

    # ── Training (paper defaults) ──
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--epochs", type=int, default=300)
    train_grp.add_argument("--batch", type=int, default=20, help="Paper uses batch=20")
    train_grp.add_argument(
        "--seq_len", type=int, default=40,
        help="Recurrent unroll length L (paper uses L=40)",
    )
    train_grp.add_argument("--lr", type=float, default=1e-4)
    train_grp.add_argument("--lambda_grad", type=float, default=0.5, help="λ for gradient loss")
    train_grp.add_argument("--num_workers", type=int, default=4)
    train_grp.add_argument("--crop_h", type=int, default=0, help="Random crop height (0=disable)")
    train_grp.add_argument("--crop_w", type=int, default=0, help="Random crop width (0=disable)")

    # ── Output ──
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    out_grp.add_argument("--save_every", type=int, default=10)
    out_grp.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()

    # ── Discover sequences ──
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequences found in {args.data_root}")
            print("  CARLA DENSE:  <seq>/depth/data/depth_*.npy + events/voxels/event_tensor_*.npy")
            print("  Synthetic:    <seq>/hdf5/depth.h5 + events/voxels/voxel_*.npy")
            return

    print(f"Found {len(sequence_dirs)} sequence(s):")
    for d in sequence_dirs:
        print(f"  • {d.name}  [{detect_format(d)}]")
    print()

    # ── Build configs ──
    crop_size = (args.crop_h, args.crop_w) if args.crop_h > 0 and args.crop_w > 0 else None

    cfg_train = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        sky_threshold=args.sky_threshold,
        augment=True,
        num_bins=args.num_bins,
    )
    cfg_val = DataConfig(
        seq_len=args.seq_len,
        crop_size=crop_size,
        depth_max=args.depth_max,
        depth_min=args.depth_min,
        sky_threshold=args.sky_threshold,
        augment=False,
        num_bins=args.num_bins,
    )

    train_ds, val_ds = create_datasets(sequence_dirs, cfg_train, cfg_val, args.train_ratio)

    print(f"{'=' * 65}")
    print(f"  Sequences      : {len(sequence_dirs)}")
    print(f"  Train samples  : {len(train_ds)}")
    print(f"  Val samples    : {len(val_ds)}")
    print(f"  Seq length (L) : {args.seq_len}")
    print(f"  Depth range    : {args.depth_min:.1f}m – {args.depth_max:.1f}m")
    print(f"  Voxel bins (B) : {args.num_bins}")
    print(f"  Crop           : {crop_size or 'none'}")
    print(f"{'=' * 65}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ──
    model = E2DepthNet(
        in_channels=args.num_bins,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    # ── Optimizer (paper §3.4: Adam, lr=1e-4) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    # ── Resume ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")

    # ── TensorBoard ──
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.out_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log  : {log_dir}\n")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, lambda_grad=args.lambda_grad
        )
        val_metrics = validate(model, val_loader, device, d_max=args.depth_max)

        scheduler.step(val_metrics["si_loss"])
        lr = optimizer.param_groups[0]["lr"]

        # ── TensorBoard logging ──
        writer.add_scalar("loss/train_total", train_metrics["total"], epoch)
        writer.add_scalar("loss/train_si", train_metrics["si"], epoch)
        writer.add_scalar("loss/train_grad", train_metrics["grad"], epoch)
        writer.add_scalar("loss/val_si", val_metrics["si_loss"], epoch)
        writer.add_scalar("metric/abs_rel", val_metrics["abs_rel"], epoch)
        writer.add_scalar("metric/sq_rel", val_metrics["sq_rel"], epoch)
        writer.add_scalar("metric/rmse", val_metrics["rmse"], epoch)
        writer.add_scalar("metric/rmse_log", val_metrics["rmse_log"], epoch)
        writer.add_scalar("metric/si_log", val_metrics["si_log"], epoch)
        writer.add_scalar("metric/delta_1.25", val_metrics["delta_1"], epoch)
        writer.add_scalar("metric/delta_1.25^2", val_metrics["delta_2"], epoch)
        writer.add_scalar("metric/delta_1.25^3", val_metrics["delta_3"], epoch)
        writer.add_scalar("metric/avg_err_10m", val_metrics["avg_err_10m"], epoch)
        writer.add_scalar("metric/avg_err_20m", val_metrics["avg_err_20m"], epoch)
        writer.add_scalar("metric/avg_err_30m", val_metrics["avg_err_30m"], epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.flush()  # Ensure data is written to disk immediately

        # ── Console output ──
        print(
            f"Epoch {epoch:03d}/{args.epochs} │ "
            f"train: {train_metrics['total']:.5f} │ "
            f"val SI: {val_metrics['si_loss']:.5f} │ "
            f"AbsRel: {val_metrics['abs_rel']:.4f} │ "
            f"RMSE: {val_metrics['rmse']:.2f}m │ "
            f"δ<1.25: {val_metrics['delta_1']:.3f} │ "
            f"lr: {lr:.1e}"
        )
        print(
            f"          │ "
            f"SqRel: {val_metrics['sq_rel']:.3f} │ "
            f"RMSE_log: {val_metrics['rmse_log']:.4f} │ "
            f"SI_log: {val_metrics['si_log']:.4f} │ "
            f"δ<1.25²: {val_metrics['delta_2']:.3f} │ "
            f"δ<1.25³: {val_metrics['delta_3']:.3f}"
        )
        print(
            f"          │ "
            f"Avg err 10m: {val_metrics['avg_err_10m']:.2f}m │ "
            f"20m: {val_metrics['avg_err_20m']:.2f}m │ "
            f"30m: {val_metrics['avg_err_30m']:.2f}m"
        )

        # ── Save best ──
        if val_metrics["si_loss"] < best_val_loss:
            best_val_loss = val_metrics["si_loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "config": {
                        "in_channels": args.num_bins,
                        "base": args.base,
                        "num_encoders": args.num_encoders,
                        "num_residuals": args.num_residuals,
                        "depth_max": args.depth_max,
                        "depth_min": args.depth_min,
                        "alpha": ALPHA,
                    },
                },
                os.path.join(args.out_dir, "best.pt"),
            )
            print(f"  ✓ Saved best model (val SI: {best_val_loss:.5f})")

        # ── Periodic checkpoint ──
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
            )

        # ── Log images every 5 epochs ──
        if epoch % 5 == 0 or epoch == 1:
            log_images(writer, model, val_loader, device, epoch, d_max=args.depth_max)

        print()

    writer.close()
    print(f"Training complete! Best val SI: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
