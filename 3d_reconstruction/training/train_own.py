#!/usr/bin/env python3
"""
Recurrent GRU-UNet for Event Frame → Depth prediction.

Input:   event frames  (hdf5/events_cam0.h5  → events/frames, uint8)
Target:  depth frames  (hdf5/depth_in_event_frame.h5 → depth, float32 metres)
Mask:    spatial mask  (hdf5/spatial_mask.h5 → mask, uint8)
Loss:    masked per-pixel L1

Architecture:
  - UNet with ConvGRU cells in every encoder block
  - Head + 3 encoder levels + 2 residual blocks + 3 decoder levels + sigmoid output

Usage:
    python3 train_own.py --data_root data/lego
    python3 train_own.py --data_dir data/lego/lego_1 data/lego/lego_2
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  ConvGRU cell
# ═══════════════════════════════════════════════════════════════════

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell (reset + update gates)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # Reset (r) and update (z) gates — jointly computed for efficiency
        self.gates = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels,
            kernel_size, padding=pad, bias=True,
        )
        # Candidate hidden state — uses reset-gated previous hidden
        self.candidate = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels,
            kernel_size, padding=pad, bias=True,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W,
                            device=x.device, dtype=x.dtype)
        rz = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        r, z = rz.chunk(2, dim=1)
        h_cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_cand


# ═══════════════════════════════════════════════════════════════════
#  UNet building blocks
# ═══════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class EncoderBlock(nn.Module):
    """Stride-2 conv (k=5) + ConvGRU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.gru = ConvGRUCell(out_ch, out_ch, kernel_size=3)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.down(x)
        h_new = self.gru(x, h)
        return h_new, h_new  # (feature_out, new_hidden_state)


class DecoderBlock(nn.Module):
    """Bilinear upsample → concat skip → conv (k=5) → conv (k=3)."""

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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ═══════════════════════════════════════════════════════════════════
#  Recurrent UNet
# ═══════════════════════════════════════════════════════════════════

class GRUUNet(nn.Module):
    """
    Recurrent UNet with ConvGRU encoders.

    Channel layout (base=32, num_encoders=3):
      head:      in  →  32
      enc0:       32 →  64  + GRU(64)
      enc1:       64 → 128  + GRU(128)
      enc2:      128 → 256  + GRU(256)
      bottleneck: ResBlock × num_residuals
      dec0:      256 + 128  → 128
      dec1:      128 +  64  →  64
      dec2:       64 +  32  →  32
      out:        32 →   1  (Sigmoid)

    skip connections:
      dec0 ← enc1 output
      dec1 ← enc0 output
      dec2 ← head output
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.num_encoders = num_encoders

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.encoders = nn.ModuleList()
        ch = base
        for _ in range(num_encoders):
            self.encoders.append(EncoderBlock(ch, ch * 2))
            ch *= 2

        self.bottleneck = nn.ModuleList([ResBlock(ch) for _ in range(num_residuals)])

        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            self.decoders.append(DecoderBlock(ch, skip_ch, ch // 2))
            ch //= 2

        self.out_head = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x:      (B, C, H, W)
            states: one hidden tensor per encoder level, or None for cold start
        Returns:
            pred:       (B, 1, H, W) in [0, 1]
            new_states: updated hidden states
        """
        if states is None:
            states = [None] * self.num_encoders

        feat = self.head(x)
        skips = [feat]  # skip from head (full resolution)

        new_states: List[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            feat, h_new = enc(feat, states[i])
            new_states.append(h_new)
            if i < self.num_encoders - 1:
                skips.append(feat)  # skips from enc0 and enc1

        for res in self.bottleneck:
            feat = res(feat)

        for i, dec in enumerate(self.decoders):
            feat = dec(feat, skips[-(i + 1)])

        return self.out_head(feat), new_states


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """
    Returns non-overlapping windows of seq_len consecutive frames.

    Per-item shapes (after optional resize):
        events: (seq_len, 1, H, W)  float32  [0, 1]
        depths: (seq_len, 1, H, W)  float32  [0, 1]  (normalized)
        masks:  (seq_len, 1, H, W)  float32  {0, 1}
        rgbs:   (seq_len, 3, H, W)  float32  [0, 1]  (visualization only)
    """

    def __init__(
        self,
        sequence_dir: Path,
        seq_len: int = 8,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        depth_min: float = 0.05,
        depth_max: float = 0.6,
        resize_hw: Optional[Tuple[int, int]] = None,
    ):
        self.sequence_dir = Path(sequence_dir)
        self.seq_len   = seq_len
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.resize_hw = resize_hw

        self.events_h5 = self.sequence_dir / "hdf5" / "events_cam0.h5"
        self.depth_h5  = self.sequence_dir / "hdf5" / "depth_in_event_frame.h5"
        self.rgb_h5    = self.sequence_dir / "hdf5" / "rgb_in_event_frame.h5"
        self.mask_h5   = self.sequence_dir / "hdf5" / "spatial_mask.h5"

        for p in (self.events_h5, self.depth_h5, self.mask_h5):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        with h5py.File(self.depth_h5, 'r') as f:
            self.n_frames = f['depth'].shape[0]

        if self.n_frames < seq_len:
            raise RuntimeError(
                f"{self.sequence_dir.name}: only {self.n_frames} frames, "
                f"need at least seq_len={seq_len}"
            )

        # Non-overlapping windows
        n_windows   = self.n_frames // seq_len
        all_starts  = np.arange(n_windows) * seq_len
        rng         = np.random.default_rng(seed)
        shuffled    = np.arange(n_windows)
        rng.shuffle(shuffled)
        n_val = int(round(n_windows * val_ratio))

        if split == "val":
            self.window_starts = all_starts[shuffled[:n_val]]
        else:
            self.window_starts = all_starts[shuffled[n_val:]]

        print(f"  [{self.sequence_dir.name}] {split}: "
              f"{len(self.window_starts)} windows "
              f"(seq_len={seq_len}, total_frames={self.n_frames})")

        # Lazy-open HDF5 handles per worker
        self._ev_ds  = None
        self._dep_ds = None
        self._msk_ds = None
        self._rgb_ds = None

    def __len__(self) -> int:
        return len(self.window_starts)

    def _open(self):
        if self._ev_ds is None:
            self._ev_ds  = h5py.File(self.events_h5, 'r')['events/frames']
            self._dep_ds = h5py.File(self.depth_h5,  'r')['depth']
            self._msk_ds = h5py.File(self.mask_h5,   'r')['mask']
            if self.rgb_h5.exists():
                self._rgb_ds = h5py.File(self.rgb_h5, 'r')['rgb']

    def __getitem__(self, i: int):
        self._open()
        start = int(self.window_starts[i])
        sl    = slice(start, start + self.seq_len)

        # Load raw data
        events = self._ev_ds[sl].astype(np.float32) / 255.0   # (T, H, W)
        depths = self._dep_ds[sl].astype(np.float32)           # (T, H, W) metres
        masks  = self._msk_ds[sl].astype(np.float32)           # (T, H, W) uint8→float

        has_rgb = self._rgb_ds is not None
        if has_rgb:
            rgbs = self._rgb_ds[sl].astype(np.float32) / 255.0  # (T, H, W, 3)

        # Combine depth validity with spatial mask
        depth_valid   = (depths > self.depth_min) & (depths < self.depth_max)
        combined_mask = depth_valid & (masks > 0)               # (T, H, W)

        # Normalize depth to [0, 1]
        depths = np.clip(depths, self.depth_min, self.depth_max)
        depths = (depths - self.depth_min) / (self.depth_max - self.depth_min)

        # Add channel dim → (T, 1, H, W)
        events = events[:, None]
        depths = depths[:, None]
        masks  = combined_mask[:, None].astype(np.float32)

        if has_rgb:
            rgbs = rgbs.transpose(0, 3, 1, 2)   # (T, 3, H, W)
        else:
            T, _, H, W = events.shape
            rgbs = np.zeros((T, 3, H, W), dtype=np.float32)

        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            events_t = F.interpolate(torch.from_numpy(events), (rh, rw),
                                     mode='bilinear', align_corners=False)
            depths_t = F.interpolate(torch.from_numpy(depths), (rh, rw),
                                     mode='bilinear', align_corners=False)
            masks_t  = F.interpolate(torch.from_numpy(masks), (rh, rw),
                                     mode='nearest')
            rgbs_t   = F.interpolate(torch.from_numpy(rgbs), (rh, rw),
                                     mode='bilinear', align_corners=False)
            return events_t, depths_t, masks_t, rgbs_t

        return (
            torch.from_numpy(events),
            torch.from_numpy(depths),
            torch.from_numpy(masks),
            torch.from_numpy(rgbs),
        )


def make_dataset(
    sequence_dirs: List[Path],
    seq_len: int,
    split: str,
    val_ratio: float,
    depth_min: float,
    depth_max: float,
    resize_hw: Optional[Tuple[int, int]],
) -> ConcatDataset:
    datasets = []
    for d in sequence_dirs:
        try:
            ds = SequenceDataset(
                d, seq_len=seq_len, split=split, val_ratio=val_ratio,
                depth_min=depth_min, depth_max=depth_max, resize_hw=resize_hw,
            )
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Warning: skipping {d.name}: {e}")
    if not datasets:
        raise RuntimeError("No valid sequences found!")
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root: Path) -> List[Path]:
    dirs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "hdf5" / "events_cam0.h5").exists() and \
           (d / "hdf5" / "depth_in_event_frame.h5").exists():
            dirs.append(d)
    return dirs


# ═══════════════════════════════════════════════════════════════════
#  Training & validation
# ═══════════════════════════════════════════════════════════════════

def masked_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    n = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - gt) * mask).sum() / n


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: float = 15.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    n_total    = len(loader)
    last_log   = t0 = time.time()

    for events, depths, masks, _ in loader:
        # (B, T, 1, H, W)
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        T = events.shape[1]

        states = None
        loss   = torch.tensor(0.0, device=device)
        for t in range(T):
            pred, states = model(events[:, t], states)
            loss = loss + masked_l1(pred, depths[:, t], masks[:, t])
        loss = loss / T

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Detach hidden states between batches (truncated BPTT)
        states = [h.detach() for h in states]

        total_loss += loss.item()
        n_batches  += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - t0
            bps = n_batches / elapsed if elapsed > 0 else 0.0
            eta = (n_total - n_batches) / bps if bps > 0 else 0.0
            avg = total_loss / n_batches
            print(f"  [Epoch {epoch:03d}] {n_batches}/{n_total} | "
                  f"loss: {avg:.5f} | {bps:.1f} batch/s | "
                  f"ETA: {int(eta//60):02d}:{int(eta%60):02d}", flush=True)
            last_log = now

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for events, depths, masks, _ in loader:
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        T = events.shape[1]

        states = None
        loss   = torch.tensor(0.0, device=device)
        for t in range(T):
            pred, states = model(events[:, t], states)
            loss = loss + masked_l1(pred, depths[:, t], masks[:, t])
        loss = loss / T

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


# ═══════════════════════════════════════════════════════════════════
#  Image logging
# ═══════════════════════════════════════════════════════════════════

def _depth_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """Colorize a (1, H, W) or (H, W) depth tensor with turbo colormap → (3, H, W)."""
    import matplotlib.cm as cm
    arr = t[0].cpu().float().numpy() if t.dim() == 3 else t.cpu().float().numpy()
    rgba = cm.turbo(arr)
    return torch.from_numpy(rgba[:, :, :3].transpose(2, 0, 1).astype("float32"))


def _run_and_log(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    tag_prefix: str,
    n_samples: int = 4,
) -> None:
    """Run model on the first n_samples items from loader and log each as a 4-panel strip."""
    # Collect enough individual samples (one window each, no batching)
    collected = []
    for events_b, depths_b, masks_b, rgbs_b in loader:
        for i in range(events_b.shape[0]):
            collected.append((
                events_b[i:i+1],   # (1, T, 1, H, W)
                depths_b[i:i+1],
                masks_b[i:i+1],
                rgbs_b[i:i+1],
            ))
            if len(collected) >= n_samples:
                break
        if len(collected) >= n_samples:
            break

    for idx, (events, depths, masks, rgbs) in enumerate(collected):
        events_dev = events.to(device)

        states = None
        preds  = []
        for t in range(events_dev.shape[1]):
            pred, states = model(events_dev[:, t], states)
            preds.append(pred.cpu())

        # Visualise the last frame in the window
        t_vis = -1
        ev  = events_dev[0, t_vis, 0].cpu().numpy()
        msk = masks[0, t_vis, 0]           # (H, W) tensor
        pr  = preds[t_vis][0]              # (1, H, W) tensor
        gt  = depths[0, t_vis]             # (1, H, W) tensor
        rgb = rgbs[0, t_vis].numpy().transpose(1, 2, 0)

        ev_rgb   = np.stack([ev, ev, ev], axis=-1)
        gt_rgb   = _depth_to_rgb(gt  * msk).numpy().transpose(1, 2, 0)
        pred_rgb = _depth_to_rgb(pr  * msk).numpy().transpose(1, 2, 0)
        rgb_clip = np.clip(rgb, 0.0, 1.0)

        # Horizontal strip: event | GT depth | pred depth | RGB
        grid   = np.concatenate([ev_rgb, gt_rgb, pred_rgb, rgb_clip], axis=1)
        grid_t = torch.from_numpy(grid.transpose(2, 0, 1))
        writer.add_image(f"viz/{tag_prefix}/sample_{idx:02d}", grid_t, epoch)


@torch.no_grad()
def log_images(
    writer: SummaryWriter,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    n_samples: int = 4,
) -> None:
    """Log n_samples strips for both train and val sets."""
    model.eval()
    _run_and_log(writer, model, train_loader, device, epoch, "train", n_samples)
    _run_and_log(writer, model, val_loader,   device, epoch, "val",   n_samples)


# ═══════════════════════════════════════════════════════════════════
#  GPU stats
# ═══════════════════════════════════════════════════════════════════

def gpu_stats(device: torch.device) -> dict:
    stats = {}
    if device.type != "cuda":
        return stats
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stats["vram_used_mb"]     = torch.cuda.memory_allocated(idx) / 1024 ** 2
    stats["vram_reserved_mb"] = torch.cuda.memory_reserved(idx)  / 1024 ** 2
    if _NVML_AVAILABLE:
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        stats["gpu_util_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    return stats


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Recurrent GRU-UNet: event frames → depth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Data")
    g.add_argument("--data_dir",  nargs="+", type=str, default=None,
                   help="Explicit sequence directory(ies)")
    g.add_argument("--data_root", type=str, default="data/lego",
                   help="Root directory — all subdirs with events_cam0.h5 are used")
    g.add_argument("--val_ratio", type=float, default=0.15)
    g.add_argument("--depth_min", type=float, default=0.05,
                   help="Minimum valid depth in metres")
    g.add_argument("--depth_max", type=float, default=0.6,
                   help="Maximum valid depth in metres")
    g.add_argument("--resize_h",  type=int, default=240,
                   help="Resize height (0 = no resize)")
    g.add_argument("--resize_w",  type=int, default=320,
                   help="Resize width (0 = no resize)")

    g = parser.add_argument_group("Sequence")
    g.add_argument("--seq_len", type=int, default=8,
                   help="Number of consecutive frames per training window")

    g = parser.add_argument_group("Model")
    g.add_argument("--base",          type=int, default=32)
    g.add_argument("--num_encoders",  type=int, default=3)
    g.add_argument("--num_residuals", type=int, default=2)

    g = parser.add_argument_group("Training")
    g.add_argument("--epochs",      type=int,   default=100)
    g.add_argument("--batch",       type=int,   default=16)
    g.add_argument("--lr",          type=float, default=1e-4)
    g.add_argument("--num_workers", type=int,   default=4)

    g = parser.add_argument_group("Output")
    g.add_argument("--out_dir",    type=str, default="checkpoints_gru")
    g.add_argument("--save_every", type=int, default=10)
    g.add_argument("--resume",     type=str, default=None)

    args = parser.parse_args()

    # ── Resolve sequence dirs ────────────────────────────────────────
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

    resize_hw = (args.resize_h, args.resize_w) \
        if (args.resize_h > 0 and args.resize_w > 0) else None

    # ── Object-level train/val split ────────────────────────────────
    rng   = np.random.default_rng(42)
    dirs  = list(sequence_dirs)
    rng.shuffle(dirs)
    n_val = max(1, int(round(len(dirs) * args.val_ratio)))
    val_dirs   = dirs[:n_val]
    train_dirs = dirs[n_val:] or val_dirs

    print(f"  Object split: {len(train_dirs)} train, {len(val_dirs)} val")

    print("\nBuilding train dataset...")
    train_ds = make_dataset(train_dirs, args.seq_len, "train", 0.0,
                            args.depth_min, args.depth_max, resize_hw)
    print("Building val dataset...")
    val_ds   = make_dataset(val_dirs,   args.seq_len, "val",   1.0,
                            args.depth_min, args.depth_max, resize_hw)

    print(f"\n{'='*60}")
    print(f"  Train windows : {len(train_ds)}")
    print(f"  Val   windows : {len(val_ds)}")
    print(f"  Seq len       : {args.seq_len}")
    print(f"  Resize        : {resize_hw or 'none'}")
    print(f"  Depth range   : {args.depth_min}m – {args.depth_max}m")
    print(f"{'='*60}\n")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ───────────────────────────────────────────────────────
    model = GRUUNet(
        in_channels=1,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(args.out_dir, "runs", run_name)
    writer   = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}\n")

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch)
        val_loss   = validate(model, val_loader, device)
        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        gs = gpu_stats(device)
        if gs:
            writer.add_scalar("gpu/vram_used_mb",     gs["vram_used_mb"],     epoch)
            writer.add_scalar("gpu/vram_reserved_mb", gs["vram_reserved_mb"], epoch)
            if "gpu_util_pct" in gs:
                writer.add_scalar("gpu/utilization_pct", gs["gpu_util_pct"], epoch)

        util_str = (f" | GPU {gs['gpu_util_pct']:.0f}%"
                    if gs and "gpu_util_pct" in gs else "")
        vram_str = (f" | VRAM {gs['vram_used_mb']:.0f}/{gs['vram_reserved_mb']:.0f} MB"
                    if gs else "")
        print(f"Epoch {epoch:03d} | train L1: {train_loss:.5f} | "
              f"val L1: {val_loss:.5f}{vram_str}{util_str}")

        # Visualization after every epoch
        log_images(writer, model, train_loader, val_loader, device, epoch)

        # Checkpoint — best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "epoch":         epoch,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  → new best (val L1: {best_val_loss:.5f})")

        # Checkpoint — periodic
        if epoch % args.save_every == 0:
            torch.save({
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "epoch":         epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))

    writer.close()
    print(f"\nDone. Best val L1: {best_val_loss:.5f}")
    print(f"Checkpoints in: {args.out_dir}")


if __name__ == "__main__":
    main()
