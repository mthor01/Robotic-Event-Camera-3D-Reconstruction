#!/usr/bin/env python3
"""
train_ere.py - EReFormer-style recurrent transformer for event-to-depth.

This script implements a recurrent transformer depth estimator inspired by
"Event-based Monocular Dense Depth Estimation with Recurrent Transformers"
(Liu et al., 2022), adapted to this repository's input format:

    [5-bin event voxel grid | 1 table-plane depth channel]

So each timestep is a 6-channel image-like tensor. The model processes
temporal sequences recurrently and predicts one dense depth map per timestep.

Key pieces mirrored from the paper:
  - Hierarchical transformer encoder-decoder backbone
  - Spatial Transformer Fusion (STF) skip connections
  - Gate Recurrent Vision Transformer (GRViT) units at each encoder scale

Training/logging behaviour mirrors train_unet_table.py:
  - same table-plane prior input channel
  - same loss function and L1 metric
  - same TensorBoard scalar/image logging via VizLogger
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from train_unet import DEPTH_MIN, D_MAX, NUM_BINS, _SCRIPT_DIR, DATA_ROOT, compute_loss, _l1_metres
from train_unet_table import _load_event_K_native
from tensorboard_runs import DEFAULT_TB_ROOT, tensorboard_run_dir
from viz import VizLogger


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def _unflatten_hw(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = tokens.shape
    return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()


class Mlp2d(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = windows.shape[0] // ((h // window_size) * (w // window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


def _build_shift_mask(h: int, w: int, window_size: int, shift_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if shift_size <= 0:
        return None

    img_mask = torch.zeros((1, h, w, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))

    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = count
            count += 1

    mask_windows = _window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-100.0"))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


# ---------------------------------------------------------------------------
# Swin-style attention blocks
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b_win, n, c = x.shape
        qkv = self.qkv(x).reshape(b_win, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b_win // n_windows, n_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        return self.proj(out)


class CrossWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_win, n, c = query.shape
        q = self.q(query).reshape(b_win, n, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        kv = self.kv(context).reshape(b_win, n, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b_win // n_windows, n_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp2d(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        h_orig, w_orig = x.shape[-2:]
        window_size = min(self.window_size, h_orig, w_orig)
        shift_size = self.shift_size if window_size > self.shift_size else 0

        x, pad_h, pad_w = _pad_to_multiple(x, window_size)
        _, _, h_pad, w_pad = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            mask = _build_shift_mask(h_pad, w_pad, window_size, shift_size, x.device)
        else:
            mask = None

        windows = _window_partition(x, window_size)
        windows = self.attn(windows, mask=mask)
        x = _window_reverse(windows, window_size, h_pad, w_pad)

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        x = x[:, :h_orig, :w_orig, :].permute(0, 3, 1, 2).contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class CrossSwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm_q = LayerNorm2d(dim)
        self.norm_kv = LayerNorm2d(dim)
        self.attn = CrossWindowAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp2d(dim, mlp_ratio=mlp_ratio)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shortcut = query
        query = self.norm_q(query)
        context = self.norm_kv(context)

        h_orig, w_orig = query.shape[-2:]
        window_size = min(self.window_size, h_orig, w_orig)
        shift_size = self.shift_size if window_size > self.shift_size else 0

        query, pad_h, pad_w = _pad_to_multiple(query, window_size)
        if pad_h or pad_w:
            context = F.pad(context, (0, pad_w, 0, pad_h))

        _, _, h_pad, w_pad = query.shape
        query = query.permute(0, 2, 3, 1).contiguous()
        context = context.permute(0, 2, 3, 1).contiguous()

        if shift_size > 0:
            query = torch.roll(query, shifts=(-shift_size, -shift_size), dims=(1, 2))
            context = torch.roll(context, shifts=(-shift_size, -shift_size), dims=(1, 2))
            mask = _build_shift_mask(h_pad, w_pad, window_size, shift_size, query.device)
        else:
            mask = None

        q_windows = _window_partition(query, window_size)
        kv_windows = _window_partition(context, window_size)
        windows = self.attn(q_windows, kv_windows, mask=mask)
        x = _window_reverse(windows, window_size, h_pad, w_pad)

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        x = x[:, :h_orig, :w_orig, :].permute(0, 3, 1, 2).contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm2d(dim * 4)
        self.reduction = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        return self.reduction(self.norm(x))


class DecoderStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, depth: int, window_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(SwinBlock(out_dim, num_heads, window_size=window_size, shift_size=shift))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.blocks(x)


class STFModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8):
        super().__init__()
        self.block1 = CrossSwinBlock(dim, num_heads, window_size=window_size, shift_size=0)
        self.block2 = CrossSwinBlock(dim, num_heads, window_size=window_size, shift_size=window_size // 2)

    def forward(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor) -> torch.Tensor:
        fused = self.block1(decoder_feat, encoder_feat)
        fused = self.block2(fused, encoder_feat)
        return fused + decoder_feat


class GRViTUnit(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_x = LayerNorm2d(dim)
        self.norm_h = LayerNorm2d(dim)
        self.q_proj_x = nn.Linear(dim, dim)
        self.q_proj_h = nn.Linear(dim, dim)
        self.k_proj_x = nn.Linear(dim, dim)
        self.k_proj_h = nn.Linear(dim, dim)
        self.v_proj_x = nn.Linear(dim, dim)
        self.v_proj_h = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.norm_out = LayerNorm2d(dim)
        self.mlp = Mlp2d(dim, mlp_ratio=mlp_ratio)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        return out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], self.dim)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = torch.zeros_like(x)

        x_norm = self.norm_x(x)
        h_norm = self.norm_h(h_prev)

        h_spatial, w_spatial = x.shape[-2:]
        x_tok = _flatten_hw(x_norm)
        h_tok = _flatten_hw(h_norm)

        q = self._reshape_heads(self.q_proj_x(x_tok) + self.q_proj_h(h_tok))
        k = self._reshape_heads(self.k_proj_x(x_tok) + self.k_proj_h(h_tok))
        v = self._reshape_heads(self.v_proj_x(x_tok) + self.v_proj_h(h_tok))

        attn_tok = self._linear_attention(q, k, v)
        attn_tok = self.out_proj(attn_tok)
        attn = _unflatten_hw(attn_tok, h_spatial, w_spatial)

        fused = x + attn
        fused = fused + self.mlp(self.norm_out(fused))

        gate = torch.sigmoid(self.gate_proj(torch.cat([x_norm, h_norm], dim=1)))
        h_new = (1.0 - gate) * h_prev + gate * attn
        return fused, h_new


class EReFormer(nn.Module):
    def __init__(
        self,
        in_ch: int = NUM_BINS + 1,
        embed_dim: int = 96,
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        decoder_depths: tuple[int, int, int] = (2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 8,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.window_size = window_size

        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.patch_embed = PatchEmbed(in_ch, embed_dim)

        self.encoder_stages = nn.ModuleList()
        for stage_idx, (dim, depth, heads) in enumerate(zip(dims, depths, num_heads)):
            blocks = []
            for block_idx in range(depth):
                shift = 0 if block_idx % 2 == 0 else window_size // 2
                blocks.append(SwinBlock(dim, heads, window_size=window_size, shift_size=shift))
            self.encoder_stages.append(nn.Sequential(*blocks))

        self.downsamples = nn.ModuleList([PatchMerging(embed_dim), PatchMerging(embed_dim * 2), PatchMerging(embed_dim * 4)])
        self.grvits = nn.ModuleList([
            GRViTUnit(dims[0], num_heads[0]),
            GRViTUnit(dims[1], num_heads[1]),
            GRViTUnit(dims[2], num_heads[2]),
            GRViTUnit(dims[3], num_heads[3]),
        ])

        self.decoder3 = DecoderStage(dims[3], dims[2], num_heads[2], decoder_depths[0], window_size)
        self.decoder2 = DecoderStage(dims[2], dims[1], num_heads[1], decoder_depths[1], window_size)
        self.decoder1 = DecoderStage(dims[1], dims[0], num_heads[0], decoder_depths[2], window_size)

        self.stf2 = STFModule(dims[2], num_heads[2], window_size=window_size)
        self.stf1 = STFModule(dims[1], num_heads[1], window_size=window_size)
        self.stf0 = STFModule(dims[0], num_heads[0], window_size=window_size)

        self.head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        input_hw = x.shape[-2:]
        if states is None:
            states = [None, None, None, None]

        x = self.patch_embed(x)
        skips: list[torch.Tensor] = []
        new_states: list[torch.Tensor] = []

        for stage_idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            x, state = self.grvits[stage_idx](x, states[stage_idx])
            new_states.append(state)
            skips.append(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)

        x = self.decoder3(skips[3], skips[2].shape[-2:])
        x = self.stf2(x, skips[2])
        x = self.decoder2(x, skips[1].shape[-2:])
        x = self.stf1(x, skips[1])
        x = self.decoder1(x, skips[0].shape[-2:])
        x = self.stf0(x, skips[0])

        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
        return self.head(x), new_states


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceTablePriorDataset(Dataset):
    """
    Returns windows of consecutive frames, each timestep using:
        [event_voxel_bins(5), table_plane_channel(1)]

    Per-item shapes:
        inp    : (T, NUM_BINS + 1, H, W)
        depth  : (T, 1, H, W)   metres
        mask   : (T, 1, H, W)
    """

    def __init__(
        self,
        seq_dir: Path,
        seq_len: int = 8,
        frame_stride: int = 1,
        window_stride: Optional[int] = None,
        use_mask: bool = True,
        fill_invalid: bool = False,
    ):
        super().__init__()
        self.seq_dir = seq_dir
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.window_stride = seq_len if window_stride is None else window_stride
        self.use_mask = use_mask
        self.fill_invalid = fill_invalid

        self.voxels_path = seq_dir / "events" / "voxels_cam0.h5"
        self.depth_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"
        self.mask_path = seq_dir / "hdf5" / "spatial_mask.h5"
        self.table_plane_path = seq_dir / "hdf5" / "table_plane.h5"

        if not self.table_plane_path.exists():
            raise FileNotFoundError(
                f"Missing: {self.table_plane_path}\n"
                "Run data_precomputation/precompute_table_plane.py first."
            )

        import h5py

        with h5py.File(self.depth_path, "r") as f:
            n_d = int(f["depth"].shape[0])
        with h5py.File(self.voxels_path, "r") as f:
            n_v = int(f["voxels"].shape[0])
        with h5py.File(self.table_plane_path, "r") as f:
            n_t = int(f["table_plane"].shape[0])

        self.n_frames = min(n_d, n_v, n_t)
        self.has_mask = use_mask and self.mask_path.exists()

        last_start = self.n_frames - 1 - (seq_len - 1) * frame_stride
        if last_start < 0:
            raise RuntimeError(
                f"{seq_dir.name}: not enough frames ({self.n_frames}) for "
                f"seq_len={seq_len}, frame_stride={frame_stride}"
            )

        self.window_starts = np.arange(0, last_start + 1, self.window_stride, dtype=np.int64)

        self._vox = None
        self._dep = None
        self._msk = None
        self._tbl = None

    def _open(self):
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
        return len(self.window_starts)

    def __getitem__(self, item: int):
        self._open()
        start = int(self.window_starts[item])
        indices = [start + t * self.frame_stride for t in range(self.seq_len)]

        inputs = []
        depths = []
        masks = []

        for idx in indices:
            vox = self._vox[idx]
            if vox.dtype == np.float16:
                vox = vox.astype(np.float32)
            vox_t = torch.from_numpy(vox)
            _, h_vox, w_vox = vox_t.shape

            dep = self._dep[idx].astype(np.float32)
            dep = np.minimum(dep, D_MAX)
            valid = (dep > 0).astype(np.float32)
            if self.has_mask:
                valid *= self._msk[idx].astype(np.float32)

            dep_t = torch.from_numpy(dep).unsqueeze(0)
            msk_t = torch.from_numpy(valid).unsqueeze(0)
            if dep_t.shape[-2:] != (h_vox, w_vox):
                dep_t = F.interpolate(dep_t.unsqueeze(0), (h_vox, w_vox), mode="nearest").squeeze(0)
                msk_t = F.interpolate(msk_t.unsqueeze(0), (h_vox, w_vox), mode="nearest").squeeze(0)

            tbl_np = self._tbl[idx].astype(np.float32)
            tbl_t = torch.from_numpy(tbl_np).unsqueeze(0)
            if tbl_t.shape[-2:] != (h_vox, w_vox):
                tbl_t = F.interpolate(
                    tbl_t.unsqueeze(0),
                    (h_vox, w_vox),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            if self.fill_invalid:
                tbl_m = tbl_t * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                dep_t = torch.where(msk_t > 0.5, dep_t, tbl_m)
                msk_t = torch.ones_like(msk_t)

            inputs.append(torch.cat([vox_t, tbl_t], dim=0))
            depths.append(dep_t)
            masks.append(msk_t)

        return torch.stack(inputs, dim=0), torch.stack(depths, dim=0), torch.stack(masks, dim=0)


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: EReFormer,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR],
    device: torch.device,
    K: torch.Tensor,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    viz: Optional[VizLogger] = None,
) -> tuple[float, float, Optional[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_l1 = 0.0
    n_batches = 0
    phase = "train" if is_train else "val"
    last_log = time.time()

    with ctx:
        for inp_seq, dep_seq, mask_seq in loader:
            inp_seq = inp_seq.to(device, non_blocking=True)
            dep_seq = dep_seq.to(device, non_blocking=True)
            mask_seq = mask_seq.to(device, non_blocking=True)

            seq_len = inp_seq.shape[1]
            states = None
            seq_loss = 0.0
            seq_l1 = 0.0
            last_pred = None

            for t in range(seq_len):
                pred, states = model(inp_seq[:, t], states)
                dep_norm = ((dep_seq[:, t] - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
                loss_t, _ = compute_loss(
                    pred,
                    dep_norm,
                    mask_seq[:, t],
                    inp_seq[:, t, :NUM_BINS],
                    K=K,
                )
                seq_loss = seq_loss + loss_t
                seq_l1 = seq_l1 + _l1_metres(pred, dep_seq[:, t], mask_seq[:, t])
                last_pred = pred

            loss = seq_loss / seq_len
            l1 = seq_l1 / seq_len

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item())
            total_l1 += float(l1.item())
            n_batches += 1

            now = time.time()
            if now - last_log >= 20.0:
                step_value = global_step if global_step is not None else n_batches
                print(
                    f"  [{phase} {n_batches:4d}/{len(loader)} batches] "
                    f"loss {total_loss / n_batches:.4f} "
                    f"L1 {total_l1 / n_batches:.4f} m",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar(f"loss/{phase}_running", total_loss / n_batches, step_value)
                    writer.add_scalar(f"l1/{phase}_running", total_l1 / n_batches, step_value)
                    if is_train:
                        writer.add_scalar("lr/running", optimizer.param_groups[0]["lr"], step_value)
                    writer.flush()
                last_log = now

            if viz is not None and last_pred is not None:
                pred_m = (last_pred * (D_MAX - DEPTH_MIN) + DEPTH_MIN).detach()
                table_ch = inp_seq[:, -1, NUM_BINS:NUM_BINS + 1].detach()
                viz.add_batch(
                    inp_seq[:, -1, :NUM_BINS].detach(),
                    dep_seq[:, -1].detach(),
                    mask_seq[:, -1].detach(),
                    pred_m,
                    table_depth=table_ch,
                )

            if global_step is not None:
                global_step += 1

    n = max(n_batches, 1)
    return total_loss / n, total_l1 / n, global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EReFormer-style recurrent transformer with table-plane prior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT,
                        help="Single sequence dir or parent of multiple sequences")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--window_stride", type=int, default=0,
                        help="0 means seq_len (non-overlapping windows)")
    parser.add_argument("--lr", type=float, default=3.2e-5,
                        help="Max learning rate for OneCycleLR")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--out_dir", type=Path,
                        default=_SCRIPT_DIR / "checkpoints" / "ereformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mask", action="store_true",
                        help="Ignore spatial mask; depth>0 validity is still applied")
    parser.add_argument("--fill_invalid", action="store_true",
                        help="Fill pixels with no measurement using the table-plane prior")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name used in checkpoint filenames. Prompted if not provided.")
    parser.add_argument("--tb_root", type=Path, default=DEFAULT_TB_ROOT,
                        help="Shared TensorBoard root. Runs are logged under <tb_root>/ereformer/<name>.")
    args = parser.parse_args()

    if args.name is None:
        args.name = input("Enter a run name for the checkpoints: ").strip()
        if not args.name:
            sys.exit("[ERROR] Run name must not be empty.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def _is_sequence(p: Path) -> bool:
        return (
            (p / "events" / "voxels_cam0.h5").exists()
            and (p / "hdf5" / "depth_in_event_frame.h5").exists()
            and (p / "hdf5" / "poses.h5").exists()
            and (p / "hdf5" / "table_plane.h5").exists()
        )

    if _is_sequence(args.data_dir):
        seq_dirs = [args.data_dir]
        single_object = True
    else:
        seq_dirs = sorted([d for d in args.data_dir.iterdir() if d.is_dir() and _is_sequence(d)])
        single_object = len(seq_dirs) == 1

    if not seq_dirs:
        sys.exit(
            f"[ERROR] No valid sequences found at or under {args.data_dir}\n"
            "        Make sure voxels, depth, poses.h5, and table_plane.h5 all exist."
        )

    if single_object:
        train_seqs = val_seqs = seq_dirs
        print(f"Found {len(seq_dirs)} sequence(s) [single-object mode]")
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
        seq_len=args.seq_len,
        frame_stride=args.frame_stride,
        window_stride=(None if args.window_stride == 0 else args.window_stride),
        use_mask=not args.no_mask,
        fill_invalid=args.fill_invalid,
    )
    train_ds = ConcatDataset([SequenceTablePriorDataset(d, **ds_kw) for d in train_seqs])
    val_ds = ConcatDataset([SequenceTablePriorDataset(d, **ds_kw) for d in val_seqs])
    print(f"  Train windows: {len(train_ds)},  Val windows: {len(val_ds)}")

    loader_kw = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    sample_inp, _, _ = train_ds[0]
    input_h, input_w = sample_inp.shape[-2:]
    K_native, native_h, native_w = _load_event_K_native()
    K_loss = K_native.copy()
    K_loss[0, :] *= input_w / native_w
    K_loss[1, :] *= input_h / native_h

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EReFormer(
        in_ch=NUM_BINS + 1,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
    ).to(device)
    K_tensor = torch.from_numpy(K_loss).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EReFormer in_ch={NUM_BINS + 1} embed_dim={args.embed_dim} parameters: {n_params:,}")
    print(f"  {NUM_BINS} (voxels) + 1 (table-plane channel) = {NUM_BINS + 1} channels")
    print(f"  Sequence length: {args.seq_len}  frame_stride: {args.frame_stride}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = tensorboard_run_dir("ereformer", args.name, args.tb_root)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard: {tb_log_dir}")
    viz_train = VizLogger(writer, n_samples=4, tag="viz/train", show_mask=not args.no_mask)
    viz_val = VizLogger(writer, n_samples=4, tag="viz/val", show_mask=not args.no_mask)

    best_val_l1 = float("inf")
    last_ckpt: dict = {}
    train_global_step = 0
    val_global_step = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_l1, train_global_step = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            K_tensor,
            writer=writer,
            global_step=train_global_step,
            viz=viz_train,
        )
        va_loss, va_l1, val_global_step = run_epoch(
            model,
            val_loader,
            None,
            None,
            device,
            K_tensor,
            writer=writer,
            global_step=val_global_step,
            viz=viz_val,
        )

        viz_train.flush(step=epoch)
        viz_val.flush(step=epoch)

        vram_a = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        vram_r = torch.cuda.memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        lr_cur = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss: {tr_loss:.4f}/{va_loss:.4f} "
            f"L1: {tr_l1:.4f}/{va_l1:.4f} m "
            f"LR: {lr_cur:.2e} "
            f"VRAM: {vram_a:.0f}/{vram_r:.0f} MB",
            flush=True,
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("l1/train", tr_l1, epoch)
        writer.add_scalar("l1/val", va_l1, epoch)
        writer.add_scalar("lr", lr_cur, epoch)

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_l1": va_l1,
            "embed_dim": args.embed_dim,
            "window_size": args.window_size,
            "in_ch": NUM_BINS + 1,
            "seq_len": args.seq_len,
            "frame_stride": args.frame_stride,
            "args": vars(args),
        }

        if va_l1 < best_val_l1:
            best_val_l1 = va_l1
            torch.save(last_ckpt, args.out_dir / f"best_{args.name}.pth")
            print(f"  -> new best checkpoint (val L1 = {va_l1:.4f} m)", flush=True)

    torch.save(last_ckpt, args.out_dir / f"last_{args.name}.pth")
    writer.close()
    print(f"\nDone. Best val L1: {best_val_l1:.4f} m", flush=True)


if __name__ == "__main__":
    main()
