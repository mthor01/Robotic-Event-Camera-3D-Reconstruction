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
                voxels_cam0.h5            # precomputed voxels (N, C, H, W) HDF5

Usage:
    python real_train.py --data_root data/real

    # With specific objects:
    python3 real_train.py --data_dir data/real/bottle data/real/cube_medium

    # With pose-warp (warps hidden states using relative camera pose):
    python3 real_train.py --data_root data/real --use_pose_warp

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
    CALIB_DIR as _CALIB_DIR,
    TRAIN_RESIZE_HW, TRAIN_CROP_HW,
)

# ================= DEFAULT PATHS =================
DATA_ROOT = _DATA_ROOT
CALIB_DIR = Path(__file__).resolve().parent / _CALIB_DIR
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
        use_pose_warp: bool = False,
        K: Optional[np.ndarray] = None,
        input_hw: Optional[Tuple[int, int]] = None,
        pose_d_ref: float = 0.3,
    ):
        super().__init__()
        self.num_encoders = num_encoders
        self.use_pose_warp = use_pose_warp
        self.pose_d_ref = pose_d_ref

        if use_pose_warp:
            if K is None or input_hw is None:
                raise ValueError("K and input_hw are required when use_pose_warp=True")
            K_t = torch.from_numpy(K).float() if isinstance(K, np.ndarray) else K.float()
            self.register_buffer("K", K_t)  # (3, 3)
            self.input_hw = input_hw        # (H, W)
        else:
            self.K = None
            self.input_hw = None
        
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
    
    def _warp_states(
        self,
        states: List,
        T_rel: torch.Tensor,
    ) -> List:
        """
        Warp all ConvLSTM hidden states (h, c) with the relative camera pose.

        Args:
            states:  List of (h, c) or None per encoder level.
            T_rel:   (B, 4, 4) T_curr_from_prev in camera frame.
        """
        in_H, in_W = self.input_hw
        warped = []
        for state in states:
            if state is None:
                warped.append(None)
                continue
            h, c = state
            feat_H, feat_W = h.shape[2], h.shape[3]
            grid = build_warp_grid(
                T_rel, self.K, feat_H, feat_W, in_H, in_W, self.pose_d_ref
            )
            h_w = F.grid_sample(
                h, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            c_w = F.grid_sample(
                c, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            warped.append((h_w, c_w))
        return warped

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        T_rel: Optional[torch.Tensor] = None,
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

        # Warp hidden states to align with the current camera frame
        if self.use_pose_warp and T_rel is not None:
            states = self._warp_states(states, T_rel)

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
# Pose Warp Utilities
# -----------------------------
def build_warp_grid(
    T_rel: torch.Tensor,
    K: torch.Tensor,
    feat_H: int,
    feat_W: int,
    in_H: int,
    in_W: int,
    d_ref: float = 0.3,
) -> torch.Tensor:
    """
    Build an inverse-warp sampling grid for hidden-state alignment.

    For each pixel in the current frame at reference depth d_ref, computes
    where it came from in the previous frame under the given relative camera
    pose.  The resulting grid is suitable for F.grid_sample.

    Args:
        T_rel:          (B, 4, 4) SE3 T_curr_from_prev in camera frame.
        K:              (3, 3) camera intrinsics at the reference resolution.
        feat_H, feat_W: Spatial size of the feature map to warp.
        in_H, in_W:     Reference resolution that K was calibrated for.
        d_ref:          Reference depth (metres) used for unprojection.

    Returns:
        grid: (B, feat_H, feat_W, 2) sampling coordinates in [-1, 1].
    """
    B = T_rel.shape[0]
    device = T_rel.device

    # Scale K to feature-map resolution
    K_f = K.to(device=device, dtype=torch.float32).clone()
    K_f[0] = K_f[0] * (feat_W / in_W)   # scale fx and cx
    K_f[1] = K_f[1] * (feat_H / in_H)   # scale fy and cy

    # Inverse relative pose: T_prev_from_curr
    T_f = T_rel.float()
    R_inv = T_f[:, :3, :3].transpose(1, 2)          # (B, 3, 3)
    t_inv = -torch.bmm(R_inv, T_f[:, :3, 3:])       # (B, 3, 1)

    # Pixel grid for the current (destination) frame
    u = torch.arange(feat_W, device=device, dtype=torch.float32)
    v = torch.arange(feat_H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing='ij')     # (feat_H, feat_W)
    ones = torch.ones(feat_H, feat_W, device=device, dtype=torch.float32)
    pix = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)  # (3, N)

    # Unproject to camera coordinates at reference depth
    K_f_inv = torch.inverse(K_f)
    X = (K_f_inv @ pix) * d_ref                     # (3, N)
    X = X.unsqueeze(0).expand(B, -1, -1)            # (B, 3, N)

    # Transform to previous camera frame
    X_prev = torch.bmm(R_inv, X) + t_inv            # (B, 3, N)

    # Project onto previous image plane
    K_f_b = K_f.unsqueeze(0).expand(B, -1, -1)
    p = torch.bmm(K_f_b, X_prev)                    # (B, 3, N)
    z = p[:, 2:3].clamp(min=1e-6)
    p_xy = p[:, :2] / z                             # (B, 2, N)

    # Normalise to [-1, 1] for grid_sample (align_corners=True)
    norm_x = 2.0 * p_xy[:, 0] / max(feat_W - 1, 1) - 1.0
    norm_y = 2.0 * p_xy[:, 1] / max(feat_H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=2).view(B, feat_H, feat_W, 2)
    return grid


# -----------------------------
# Loss Functions (from paper)
# -----------------------------
def charbonnier_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier (pseudo-Huber) loss: mean sqrt((pred-gt)^2 + eps^2) over valid pixels.

    Preferred over plain L1: continuously differentiable at 0, robust to sensor
    outliers near depth discontinuities.
    """
    n = mask.sum().clamp_min(1.0)
    diff = (pred - gt) * mask
    return torch.sqrt(diff ** 2 + eps ** 2).sum() / n


def mean_alignment_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalises global mean offset: (mean(pred) - mean(gt))^2 over valid pixels.

    Gradient w.r.t. output bias = 2*(pred_mean - gt_mean)/n, which is non-zero
    whenever there is a systematic offset.  This directly closes the persistent
    pred_m vs gt_m gap that scale-invariant terms (gradient, smoothness, normal)
    cannot fix because they are insensitive to global offsets.
    """
    n = mask.sum().clamp_min(1.0)
    pred_mean = (pred * mask).sum() / n
    gt_mean   = (gt   * mask).sum() / n
    return (pred_mean - gt_mean) ** 2


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
            m    = F.avg_pool2d(mask, 2)
            pred = F.avg_pool2d(pred * mask, 2) / m.clamp_min(1e-6)
            gt   = F.avg_pool2d(gt   * mask, 2) / m.clamp_min(1e-6)
            mask = (m > 0.5).float()
        
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


def edge_aware_smoothness_loss(
    pred: torch.Tensor,
    events: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Edge-aware depth smoothness regulariser.

    L_smooth = |dx D̂| · exp(-γ |dx Ē|) + |dy D̂| · exp(-γ |dy Ē|)

    where Ē = per-sample normalised event activity summed over bins.
    Keeps depth smooth on flat regions but allows sharp discontinuities where
    the event camera sees real edges.
    """
    activity = events.abs().sum(dim=1, keepdim=True)                    # (B, 1, H, W)
    a_max    = activity.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp_min(1e-6)
    activity = activity / a_max                                          # normalised [0,1]

    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    dx_ev = (activity[:, :, :, 1:] + activity[:, :, :, :-1]) * 0.5
    dy_ev = (activity[:, :, 1:, :] + activity[:, :, :-1, :]) * 0.5

    loss_x = (dx_pred * torch.exp(-gamma * dx_ev)).mean()
    loss_y = (dy_pred * torch.exp(-gamma * dy_ev)).mean()
    return loss_x + loss_y


def _compute_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Geometrically correct surface normals via backprojection and cross product.

    Backprojects each pixel to 3-D using camera intrinsics, then estimates the
    surface normal at each pixel as the cross product of the horizontal and
    vertical 3-D central-difference vectors.

    Args:
        depth: (B, 1, H, W) **metric** depth in metres.
        K:     (3, 3) camera intrinsics at input resolution.
    Returns:
        normals: (B, 3, H, W) unit normals.
    """
    B, _, H, W = depth.shape
    device = depth.device
    K = K.to(device=device, dtype=torch.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel coordinate grids
    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing='ij')   # (H, W)

    # Backproject each pixel to 3-D: X=(u-cx)*D/fx, Y=(v-cy)*D/fy, Z=D
    D = depth[:, 0]  # (B, H, W)
    points = torch.stack([
        (uu - cx) * D / fx,   # X
        (vv - cy) * D / fy,   # Y
        D,                     # Z
    ], dim=1)  # (B, 3, H, W)

    # 3-D central differences along u and v
    du = points[:, :, :, 2:] - points[:, :, :, :-2]   # (B, 3, H, W-2)
    dv = points[:, :, 2:, :] - points[:, :, :-2, :]   # (B, 3, H-2, W)
    du = F.pad(du, (1, 1, 0, 0), mode='replicate')     # (B, 3, H, W)
    dv = F.pad(dv, (0, 0, 1, 1), mode='replicate')     # (B, 3, H, W)

    # Normal = cross(du, dv)
    nx = du[:, 1] * dv[:, 2] - du[:, 2] * dv[:, 1]
    ny = du[:, 2] * dv[:, 0] - du[:, 0] * dv[:, 2]
    nz = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
    normals = torch.stack([nx, ny, nz], dim=1)  # (B, 3, H, W)
    return F.normalize(normals, dim=1)


def normal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
    log_depth: bool = False,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> torch.Tensor:
    """Surface normal cosine loss using geometrically correct backprojected normals.

    Converts normalised predictions to metric depth before computing normals so
    that the cross-product vectors are in consistent metric units.
    """
    if log_depth:
        pred_m = log_normalized_to_depth(pred)
        gt_m   = log_normalized_to_depth(gt)
    else:
        pred_m = linear_normalized_to_depth(pred, depth_min, depth_max)
        gt_m   = linear_normalized_to_depth(gt, depth_min, depth_max)
    n_pred = _compute_normals(pred_m, K)
    n_gt   = _compute_normals(gt_m, K)
    cosine = (n_pred * n_gt).sum(dim=1, keepdim=True)   # (B, 1, H, W)
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


def multiview_consistency_loss(
    pred_prev: torch.Tensor,
    pred_curr: torch.Tensor,
    T_curr_from_prev: torch.Tensor,
    K: torch.Tensor,
    mask_prev: torch.Tensor,
    mask_curr: torch.Tensor,
    log_depth: bool = False,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> torch.Tensor:
    """Multi-view depth consistency using known relative camera pose.

    Projects pred_prev into the current frame and enforces agreement with
    pred_curr at the projected location.

    Args:
        pred_prev:        (B, 1, H, W) normalised depth at t-1.
        pred_curr:        (B, 1, H, W) normalised depth at t.
        T_curr_from_prev: (B, 4, 4) SE3 T_{t from t-1} in camera frame.
        K:                (3, 3) camera intrinsics at input resolution.
        mask_prev/curr:   (B, 1, H, W) valid-pixel masks.
    """
    if log_depth:
        d_prev = log_normalized_to_depth(pred_prev)
        d_curr = log_normalized_to_depth(pred_curr)
    else:
        d_prev = linear_normalized_to_depth(pred_prev, depth_min, depth_max)
        d_curr = linear_normalized_to_depth(pred_curr, depth_min, depth_max)

    B, _, H, W = d_prev.shape
    device = d_prev.device
    K_dev  = K.to(device=device, dtype=torch.float32)
    K_inv  = torch.inverse(K_dev)

    # Pixel grid for the prev frame
    u    = torch.arange(W, device=device, dtype=torch.float32)
    v    = torch.arange(H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing='ij')                     # (H, W)
    pix  = torch.stack([uu, vv, torch.ones_like(uu)], dim=0).reshape(3, -1)  # (3, N)

    # Unproject prev depth to 3-D points in camera_prev
    d_flat   = d_prev.reshape(B, 1, H * W)
    rays     = (K_inv @ pix).unsqueeze(0).expand(B, -1, -1)          # (B, 3, N)
    X_prev   = rays * d_flat                                          # (B, 3, N)
    X_prev_h = torch.cat(
        [X_prev, torch.ones(B, 1, H * W, device=device)], dim=1      # (B, 4, N)
    )

    # Transform to curr frame
    T      = T_curr_from_prev.to(device=device, dtype=torch.float32)
    X_curr = torch.bmm(T, X_prev_h)[:, :3]                           # (B, 3, N)

    # Project onto curr image plane
    K_b   = K_dev.unsqueeze(0).expand(B, -1, -1)
    p     = torch.bmm(K_b, X_curr)                                    # (B, 3, N)
    z     = p[:, 2:3].clamp(min=1e-6)
    p_xy  = p[:, :2] / z                                              # (B, 2, N)

    # Normalise to [-1, 1] for F.grid_sample
    norm_x = (2.0 * p_xy[:, 0] / max(W - 1, 1) - 1.0).view(B, H, W)
    norm_y = (2.0 * p_xy[:, 1] / max(H - 1, 1) - 1.0).view(B, H, W)
    grid   = torch.stack([norm_x, norm_y], dim=3)                     # (B, H, W, 2)

    # Sample d_curr at the projected locations
    d_curr_sampled = F.grid_sample(
        d_curr, grid, align_corners=True, mode='bilinear', padding_mode='zeros'
    )

    # Expected depth in curr frame = Z of the transformed 3-D point (metric)
    z_exp_m = X_curr[:, 2:3].view(B, 1, H, W).clamp_min(1e-6)

    # Validity mask: in-bounds + positive z + valid in both frames
    in_bounds = (
        (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
    ).unsqueeze(1).float()
    pos_z          = (z.view(B, 1, H, W) > 0.0).float()
    mask_c_sampled = F.grid_sample(
        mask_curr.float(), grid, align_corners=True,
        mode='nearest', padding_mode='zeros'
    )
    valid = mask_prev * in_bounds * pos_z * mask_c_sampled

    n    = valid.sum().clamp_min(1.0)
    diff = (d_curr_sampled - z_exp_m) * valid
    return torch.sqrt(diff ** 2 + 1e-3 ** 2).sum() / n


def e2depth_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    events: torch.Tensor,
    lambda_grad: float = 0.5,
    lambda_smooth: float = 0.01,
    lambda_normal: float = 0.1,
    lambda_mean: float = 0.1,
    pred_prev: Optional[torch.Tensor] = None,
    T_curr_from_prev: Optional[torch.Tensor] = None,
    mask_prev: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
    lambda_mv: float = 0.2,
    log_depth: bool = False,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined depth loss.

    L = L_charb + λ_grad·L_grad + λ_smooth·L_smooth + λ_normal·L_normal
      + λ_mean·L_mean + λ_mv·L_mv  (only when pred_prev and K are available)

    Design notes:
    - Charbonnier replaces plain L1: robust to sensor outliers, fully
      differentiable at 0 (no kink unlike Huber).
    - No separate log-depth term: the network already predicts in
      log-normalised space, so Charbonnier in that space IS a robust
      log-domain loss.  Adding it explicitly would double-count.
    - Multi-view consistency uses full gradients through pred_prev so
      both frames learn to be geometrically consistent.
    """
    l_charb  = charbonnier_loss(pred, gt, mask)
    l_grad   = multi_scale_gradient_loss(pred, gt, mask)
    l_smooth = edge_aware_smoothness_loss(pred, events)
    l_mean   = mean_alignment_loss(pred, gt, mask)

    total = (
        l_charb
        + lambda_grad   * l_grad
        + lambda_smooth * l_smooth
        + lambda_mean   * l_mean
    )

    l_normal_val = 0.0
    if K is not None:
        l_normal = normal_loss(
            pred, gt, mask, K,
            log_depth=log_depth,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        total        = total + lambda_normal * l_normal
        l_normal_val = l_normal.item()

    l_mv_val = 0.0
    if (
        pred_prev is not None
        and T_curr_from_prev is not None
        and mask_prev is not None
        and K is not None
    ):
        l_mv = multiview_consistency_loss(
            pred_prev, pred, T_curr_from_prev, K,
            mask_prev, mask,
            log_depth=log_depth,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        total    = total + lambda_mv * l_mv
        l_mv_val = l_mv.item()

    return total, {
        "charb":  l_charb.item(),
        "grad":   l_grad.item(),
        "smooth": l_smooth.item(),
        "normal": l_normal_val,
        "mean":   l_mean.item(),
        "mv":     l_mv_val,
        "total":  total.item(),
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
    use_pose_warp: bool = False  # Warp ConvLSTM hidden states using relative camera pose (from poses.h5)
    rgb_mask: bool = False  # Mask out white pixels using projected RGB
    spatial_mask: bool = True  # Mask pixels outside cube around EE (from spatial_mask.h5)
    log_depth: bool = False  # Use log depth encoding (paper default); False = linear normalization


class RealDataset(Dataset):
    """
    Dataset for real data recorded with franka_pipeline + synchronised_recording.

    Requires precomputed voxels (run precompute_voxels.py first).
    When ``cfg.use_pose_warp`` is ``True``, loads poses from hdf5/poses.h5
    and computes relative camera transforms per timestep.  These are returned
    alongside the event voxels and used by the model to warp ConvLSTM hidden
    states for ego-motion compensation.

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
        self.use_pose_warp = cfg.use_pose_warp
        _voxels_h5 = self.sequence_dir / "events" / "voxels_cam0.h5"
        if _voxels_h5.exists():
            self.voxels_h5_path: Optional[Path] = _voxels_h5
            self.voxels_dir: Optional[Path] = None
        elif (self.sequence_dir / "events" / "voxels_cam0").exists():
            self.voxels_h5_path = None
            self.voxels_dir = self.sequence_dir / "events" / "voxels_cam0"
        else:
            self.voxels_h5_path = None
            self.voxels_dir = self.sequence_dir / "events" / "voxels"
        # Lazy-opened per worker (None until first _get_voxel call in that worker)
        self._voxels_ds = None

        # --- Pose warp: load camera poses (optional) ---
        self._T_cam_from_world: Optional[np.ndarray] = None  # (N, 4, 4)
        if self.use_pose_warp:
            poses_path = self.sequence_dir / "hdf5" / "poses.h5"
            if not poses_path.exists():
                raise FileNotFoundError(
                    f"--use_pose_warp requires poses.h5. "
                    f"Expected at: {poses_path}"
                )
            T_rgb_from_ee = np.load(CALIB_DIR / "T_rgb_from_ee.npz")["T"]
            T_event_from_rgb = np.load(CALIB_DIR / "T_event_from_rgb.npz")["T"]
            T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee  # (4, 4)
            with h5py.File(poses_path, 'r') as pf:
                ee_T = pf["ee_T"][:]  # (N, 4, 4) T_base_from_ee
            # T_cam[i] = T_event_from_base[i] = T_event_from_ee @ inv(ee_T[i])
            T_ee_inv = np.linalg.inv(ee_T)  # (N, 4, 4)
            self._T_cam_from_world = np.einsum(
                'ij,njk->nik', T_event_from_ee, T_ee_inv
            ).astype(np.float32)
        
        # Verify files exist
        if not self.depth_h5_path.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5_path}")
        
        if self.voxels_h5_path is not None:
            with h5py.File(self.voxels_h5_path, 'r') as _f:
                n_voxels = int(_f["voxels"].shape[0])
            if n_voxels == 0:
                raise FileNotFoundError(f"Empty voxels dataset in {self.voxels_h5_path}")
        else:
            _voxel_files = sorted(self.voxels_dir.glob("voxel_*.npy")) if self.voxels_dir.exists() else []
            n_voxels = len(_voxel_files)
            if n_voxels == 0:
                raise FileNotFoundError(f"No precomputed voxels in {self.voxels_dir}")
        
        # --- Metadata ---
        with h5py.File(self.depth_h5_path, 'r') as f:
            self.n_frames = f[self._depth_key].shape[0]
            self.H = f[self._depth_key].shape[1]
            self.W = f[self._depth_key].shape[2]
        with h5py.File(self._ts_h5_path, 'r') as f:
            self.depth_timestamps = f[self._ts_key][:] // 1000
        
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
        pose_str = "pose_warp" if self.use_pose_warp else "no pose"
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
        """Get precomputed voxel grid for a frame (always returns float32)."""
        if self.voxels_h5_path is not None:
            # Open lazily per worker — avoids per-call open/close overhead.
            # Each DataLoader worker process opens the file once and keeps it open.
            if self._voxels_ds is None:
                self._voxels_ds = h5py.File(self.voxels_h5_path, 'r')["voxels"]
            v = self._voxels_ds[frame_idx]
        else:
            v = np.load(self.voxels_dir / f"voxel_{frame_idx:06d}.npy")
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        return v
    
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
        poses_seq = []

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

            # Relative camera pose: T_curr_from_prev (identity for first frame)
            if self.use_pose_warp and self._T_cam_from_world is not None and t > 0:
                T_curr = self._T_cam_from_world[idx]
                T_prev = self._T_cam_from_world[idx - 1]
                T_rel = (T_curr @ np.linalg.inv(T_prev)).astype(np.float32)
            else:
                T_rel = np.eye(4, dtype=np.float32)

            events_seq.append(voxel)
            depth_seq.append(depth[None])  # (1, H, W)
            mask_seq.append(mask[None])
            poses_seq.append(T_rel)

        # Stack sequences: (T, C, H, W)
        events = np.stack(events_seq, axis=0)
        depths = np.stack(depth_seq, axis=0)
        masks = np.stack(mask_seq, axis=0)
        poses = np.stack(poses_seq, axis=0)  # (T, 4, 4)

        # Resize to target resolution (applied before crop/augmentation).
        # Depths and masks are always at native HDF5 resolution and always need
        # resizing. Events (voxels) may have been precomputed at the final crop
        # resolution, in which case the resize step is skipped for them only.
        if self.cfg.resize_hw is not None:
            rh, rw = self.cfg.resize_hw
            depths = F.interpolate(torch.from_numpy(depths), size=(rh, rw), mode="bilinear", align_corners=False).numpy()
            masks  = F.interpolate(torch.from_numpy(masks),  size=(rh, rw), mode="nearest").numpy()
            ev_H, ev_W = events.shape[2], events.shape[3]
            out_H = self.cfg.crop_size[0] if self.cfg.crop_size is not None else rh
            out_W = self.cfg.crop_size[1] if self.cfg.crop_size is not None else rw
            if (ev_H != rh or ev_W != rw) and (ev_H != out_H or ev_W != out_W):
                events = F.interpolate(torch.from_numpy(events), size=(rh, rw), mode="bilinear", align_corners=False).numpy()

        # Spatial dims after optional resize (used for crop bounds).
        # Use depths shape as the reference — depths are always resized to resize_hw,
        # whereas events may have been precomputed at the final crop resolution.
        cur_H = depths.shape[2]
        cur_W = depths.shape[3]

        # Center crop (applied to both train and val whenever crop_size is set)
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            if cur_H < ch or cur_W < cw:
                raise ValueError(
                    f"Center crop {cw}x{ch} larger than image {cur_W}x{cur_H}"
                )
            y0 = (cur_H - ch) // 2
            x0 = (cur_W - cw) // 2
            depths = depths[:, :, y0:y0+ch, x0:x0+cw]
            masks  = masks[:, :, y0:y0+ch, x0:x0+cw]
            # Events may already be at crop size (precomputed at training resolution).
            if events.shape[2] != ch or events.shape[3] != cw:
                events = events[:, :, y0:y0+ch, x0:x0+cw]

        # Horizontal flip (training) — disabled when use_pose_warp is active because
        # flipping requires updating cx in K; without that update, unprojection in the
        # MV-consistency and pose-warp paths would be geometrically wrong.
        if self.cfg.augment and self.split == "train" and not self.use_pose_warp and np.random.rand() > 0.5:
            events = np.flip(events, axis=3).copy()
            depths = np.flip(depths, axis=3).copy()
            masks = np.flip(masks, axis=3).copy()

        return (
            torch.from_numpy(events).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(masks).float(),
            torch.from_numpy(poses).float(),
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
            (d / "events" / "voxels_cam0.h5").exists()
            or (d / "events" / "voxels_cam0").exists()
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
    voxels_h5_path: Optional[Path] = None
    voxels_dir: Optional[Path] = None
    _h5 = seq_dir / "events" / "voxels_cam0.h5"
    if _h5.exists():
        voxels_h5_path = _h5
    else:
        for _vd in [seq_dir / "events" / "voxels_cam0", seq_dir / "events" / "voxels"]:
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
        if voxels_h5_path is not None:
            with h5py.File(voxels_h5_path, 'r') as _f:
                voxel = _f["voxels"][frame_idx].astype(np.float32)
        elif voxels_dir is not None:
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
    lambda_smooth: float = 0.01,
    lambda_normal: float = 0.1,
    lambda_mean: float = 0.1,
    lambda_mv: float = 0.2,
    K: Optional[torch.Tensor] = None,
    log_depth: bool = False,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
    epoch: int = 0,
    log_interval: float = 10.0,
    profile: bool = False,
) -> Dict[str, float]:
    """Train for one epoch with sequence processing."""
    model.train()
    
    total_loss   = 0.0
    total_charb  = 0.0
    total_grad   = 0.0
    total_smooth = 0.0
    total_normal = 0.0
    total_mean_a = 0.0
    total_mv     = 0.0
    n_batches    = 0
    n_total = len(loader)

    # Per-phase timing accumulators (seconds)
    t_data    = 0.0  # DataLoader wait (data loading + collation)
    t_transfer = 0.0  # host → device (.to())
    t_forward  = 0.0  # model forward + loss
    t_backward = 0.0  # backward + optimizer step

    last_log_time = time.time()
    epoch_start = time.time()
    _t = time.perf_counter()  # tracks the start of the current "data loading" window

    for events, depths, masks, poses in loader:
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_data += time.perf_counter() - _t

        # events: (B, T, C, H, W)
        # depths: (B, T, 1, H, W)
        # masks:  (B, T, 1, H, W)
        # poses:  (B, T, 4, 4) relative T_curr_from_prev per step
        B, T = events.shape[:2]

        if profile:
            _t2 = time.perf_counter()
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        poses = poses.to(device, non_blocking=True)
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_transfer += time.perf_counter() - _t2

        # Process sequence
        states    = None
        pred_prev = None
        mask_prev = None
        batch_loss   = 0.0
        batch_charb  = 0.0
        batch_grad   = 0.0
        batch_smooth = 0.0
        batch_normal = 0.0
        batch_mean_a = 0.0
        batch_mv     = 0.0

        if profile:
            _t2 = time.perf_counter()
        for t in range(T):
            T_rel = poses[:, t] if model.use_pose_warp else None
            pred, states = model(events[:, t], states, T_rel=T_rel)
            loss, metrics = e2depth_loss(
                pred, depths[:, t], masks[:, t], events[:, t],
                lambda_grad=lambda_grad,
                lambda_smooth=lambda_smooth,
                lambda_normal=lambda_normal,
                lambda_mean=lambda_mean,
                pred_prev=pred_prev,
                T_curr_from_prev=poses[:, t] if t > 0 else None,
                mask_prev=mask_prev,
                K=K,
                lambda_mv=lambda_mv,
                log_depth=log_depth,
                depth_min=depth_min,
                depth_max=depth_max,
            )
            batch_loss   += loss
            batch_charb  += metrics["charb"]
            batch_grad   += metrics["grad"]
            batch_smooth += metrics["smooth"]
            batch_normal += metrics["normal"]
            batch_mean_a += metrics["mean"]
            batch_mv     += metrics["mv"]
            pred_prev = pred
            mask_prev = masks[:, t]
        
        # Average over sequence
        batch_loss = batch_loss / T
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward += time.perf_counter() - _t2

        if profile:
            _t2 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_backward += time.perf_counter() - _t2
        
        total_loss   += batch_loss.item()
        total_charb  += batch_charb  / T
        total_grad   += batch_grad   / T
        total_smooth += batch_smooth / T
        total_normal += batch_normal / T
        total_mean_a += batch_mean_a / T
        total_mv     += batch_mv     / T
        n_batches    += 1

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

        if profile:
            _t = time.perf_counter()  # reset for next data-loading window

    if profile and n_batches > 0:
        total_t = t_data + t_transfer + t_forward + t_backward
        def _pct(x): return 100.0 * x / total_t if total_t > 0 else 0.0
        def _ms(x): return 1000.0 * x / n_batches
        print(
            f"\n  [Epoch {epoch:03d}] Timing breakdown ({n_batches} batches):\n"
            f"    Data loading : {_ms(t_data):7.1f} ms/batch  ({_pct(t_data):.1f}%)\n"
            f"    Host→device  : {_ms(t_transfer):7.1f} ms/batch  ({_pct(t_transfer):.1f}%)\n"
            f"    Forward+loss : {_ms(t_forward):7.1f} ms/batch  ({_pct(t_forward):.1f}%)\n"
            f"    Backward+opt : {_ms(t_backward):7.1f} ms/batch  ({_pct(t_backward):.1f}%)\n"
            f"    Total        : {_ms(total_t):7.1f} ms/batch\n",
            flush=True,
        )

    n = max(1, n_batches)
    return {
        "total":  total_loss   / n,
        "charb":  total_charb  / n,
        "grad":   total_grad   / n,
        "smooth": total_smooth / n,
        "normal": total_normal / n,
        "mean":   total_mean_a / n,
        "mv":     total_mv     / n,
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
    """Validate and compute metrics over all frames after recurrent warmup.

    Evaluates frames t >= eval_start = min(3, T-1) so the ConvLSTM state has
    had a few steps to warm up before measurements begin.  All four standard
    metrics are accumulated:
      - L1 (metric)  — mean absolute error in metres
      - AbsRel        — |pred-gt|/gt
      - RMSE          — sqrt(mean squared error) in metres
      - δ<1.25        — fraction with max(pred/gt, gt/pred) < 1.25
    """
    model.eval()

    total_l1      = 0.0
    total_abs_rel = 0.0
    total_sq      = 0.0
    total_delta    = 0.0
    total_pixels   = 0.0
    n_frames       = 0  # count each evaluated frame, not each batch

    last_pred_log_mean    = 0.0
    last_gt_log_mean      = 0.0
    last_pred_metric_mean = 0.0
    last_gt_metric_mean   = 0.0

    with torch.no_grad():
        for events, depths, masks, poses in loader:
            B, T = events.shape[:2]
            events = events.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            poses  = poses.to(device, non_blocking=True)

            eval_start = min(3, T - 1)

            states = None
            preds  = []
            for t in range(T):
                T_rel = poses[:, t] if model.use_pose_warp else None
                pred, states = model(events[:, t], states, T_rel=T_rel)
                if t >= eval_start:
                    preds.append((pred, depths[:, t], masks[:, t]))

            for pred, gt, mask in preds:
                if log_depth:
                    pred_metric = log_normalized_to_depth(pred)
                    gt_metric   = log_normalized_to_depth(gt)
                else:
                    pred_metric = linear_normalized_to_depth(pred, d_min=depth_min, d_max=depth_max)
                    gt_metric   = linear_normalized_to_depth(gt,   d_min=depth_min, d_max=depth_max)

                gt_safe  = gt_metric.clamp_min(1e-6)
                diff     = torch.abs(pred_metric - gt_metric) * mask
                n_valid  = mask.sum().clamp_min(1.0)

                total_l1      += (diff.sum() / n_valid).item()
                total_abs_rel += ((diff / gt_safe).sum() / n_valid).item()
                total_sq      += ((diff ** 2).sum() / n_valid).item()

                ratio = torch.max(pred_metric / gt_safe, gt_safe / pred_metric.clamp_min(1e-6))
                total_delta   += ((ratio < 1.25).float() * mask).sum().item()
                total_pixels  += n_valid.item()

                n_frames += 1  # one frame evaluated

            # Keep last-frame stats from the final batch for the mean-gap debug print
            pred_last, gt_last, mask_last = preds[-1]
            if log_depth:
                pm = log_normalized_to_depth(pred_last)
                gm = log_normalized_to_depth(gt_last)
            else:
                pm = linear_normalized_to_depth(pred_last, d_min=depth_min, d_max=depth_max)
                gm = linear_normalized_to_depth(gt_last,   d_min=depth_min, d_max=depth_max)
            nv = mask_last.sum().clamp_min(1.0)
            last_pred_log_mean    = (pred_last * mask_last).sum().item() / nv.item()
            last_gt_log_mean      = (gt_last   * mask_last).sum().item() / nv.item()
            last_pred_metric_mean = (pm * mask_last).sum().item() / nv.item()
            last_gt_metric_mean   = (gm * mask_last).sum().item() / nv.item()

    n = max(1, n_frames)
    return {
        "l1_metric":         total_l1      / n,
        "abs_rel":           total_abs_rel / n,
        "rmse":              (total_sq     / n) ** 0.5,
        "delta_125":         total_delta   / max(1.0, total_pixels),
        "pred_log_mean":     last_pred_log_mean,
        "gt_log_mean":       last_gt_log_mean,
        "pred_metric_mean":  last_pred_metric_mean,
        "gt_metric_mean":    last_gt_metric_mean,
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
    fixed_indices: Optional[List[int]] = None,
):
    """Log n_images sample predictions to TensorBoard under '<split>/sample_N/…'.

    Uses ``fixed_indices`` (determined once before training) so the same samples
    are shown every epoch, making progress directly comparable across epochs.
    """
    from torch.utils.data import Subset

    model.eval()

    dataset = loader.dataset
    n = len(dataset)
    if fixed_indices is not None:
        indices = [i % n for i in fixed_indices[:n_images]]
    else:
        indices = torch.randperm(n)[:n_images].tolist()
    subset_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=n_images,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn if hasattr(loader, "collate_fn") and loader.collate_fn is not None else None,
    )

    with torch.no_grad():
        for events, depths, masks, poses in subset_loader:
            B, T = events.shape[:2]
            events = events.to(device)
            depths = depths.to(device)
            masks = masks.to(device)
            poses = poses.to(device)

            # Process full sequence
            states = None
            for t in range(T):
                T_rel = poses[:, t] if model.use_pose_warp else None
                pred, states = model(events[:, t], states, T_rel=T_rel)

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
    data_group.add_argument("--use_pose_warp", action="store_true", default=True,
                           help="Warp ConvLSTM hidden states with relative camera pose (from poses.h5)")
    data_group.add_argument("--rgb_mask", action="store_true",
                           help="Mask out white pixels using projected RGB (from rgb_in_event_frame.h5)")
    data_group.add_argument("--spatial_mask", action="store_true", default=True,
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
    train_group.add_argument("--epochs", type=int, default=15,
                            help="Number of epochs")
    train_group.add_argument("--batch", type=int, default=10,
                            help="Batch size")
    train_group.add_argument("--seq_len", type=int, default=10,
                            help="Sequence length for recurrent training")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Peak learning rate")
    train_group.add_argument("--lr_min", type=float, default=1e-6,
                            help="Minimum LR at end of cosine decay")
    train_group.add_argument("--warmup_epochs", type=int, default=0,
                            help="Linear warmup epochs (0 = no warmup)")
    train_group.add_argument("--lambda_grad", type=float, default=0.5,
                            help="Weight for multi-scale gradient loss")
    train_group.add_argument("--lambda_smooth", type=float, default=0.01,
                            help="Weight for edge-aware smoothness loss")
    train_group.add_argument("--lambda_normal", type=float, default=0.1,
                            help="Weight for surface normal cosine loss")
    train_group.add_argument("--lambda_mean", type=float, default=0.1,
                            help="Weight for global mean alignment loss")
    train_group.add_argument("--lambda_mv", type=float, default=0.2,
                            help="Weight for multi-view consistency loss (needs poses)")
    train_group.add_argument("--num_workers", type=int, default=8)
    train_group.add_argument("--crop_h", type=int, default=TRAIN_CROP_HW[0],
                            help="Center crop height in pixels (0=no crop)")
    train_group.add_argument("--crop_w", type=int, default=TRAIN_CROP_HW[1],
                            help="Center crop width in pixels (0=no crop)")
    train_group.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0],
                            help="Resize height before crop/augmentation (0=no resize)")
    train_group.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1],
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
    dbg_group.add_argument("--profile", action="store_true",
                           help="Print per-phase timing breakdown (data/transfer/forward/backward) "
                                "at the end of each training epoch")

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
            use_pose=False,  # pose depth channel visualization removed (now using hidden-state warp)
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
        use_pose_warp=args.use_pose_warp,
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
        use_pose_warp=args.use_pose_warp,
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
    print(f"  Pose warp: {args.use_pose_warp}")
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

    # Input channels = num_bins (pose is no longer an extra input channel)
    in_channels = args.num_bins
    print(f"Input channels: {in_channels} (bins={args.num_bins})")

    # Compute effective camera intrinsics after resize+crop.
    # Always computed (needed for multi-view consistency and pose warp).
    K_input: Optional[np.ndarray] = None
    input_hw: Optional[Tuple[int, int]] = None
    try:
        K_native  = np.load(str(CALIB_DIR / "event_intrinsics.npz"))["camera_matrix"].copy()
        native_ev = np.load(str(CALIB_DIR / "event_intrinsics.npz"))["image_size"]
        native_H, native_W = int(native_ev[1]), int(native_ev[0])
        K_input = K_native.copy()
        cur_H, cur_W = native_H, native_W
        if resize_hw is not None:
            rh, rw = resize_hw
            K_input[0] *= rw / native_W   # fx, cx
            K_input[1] *= rh / native_H   # fy, cy
            cur_H, cur_W = rh, rw
        if crop_size is not None:
            ch, cw = crop_size
            y0 = (cur_H - ch) // 2
            x0 = (cur_W - cw) // 2
            K_input[0, 2] -= x0           # cx
            K_input[1, 2] -= y0           # cy
            cur_H, cur_W = ch, cw
        input_hw = (cur_H, cur_W)
        print(f"Effective K (input res {cur_W}x{cur_H}):\n{K_input}")
    except Exception as e:
        print(f"Warning: could not load camera intrinsics: {e}")
        if args.use_pose_warp:
            raise

    # Model
    model = E2DepthNet(
        in_channels=in_channels,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
        use_pose_warp=args.use_pose_warp,
        K=K_input,
        input_hw=input_hw,
    ).to(device)

    #model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Cosine annealing with optional linear warmup.
    # During warmup epochs the LR rises linearly from 0 → args.lr.
    # Afterwards it decays as a cosine from args.lr → args.lr_min.
    _cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_cosine_epochs, eta_min=args.lr_min
    )
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6 / args.lr,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    start_epoch = 1
    best_val_loss = float("inf")

    # Fixed visualisation indices — chosen once with a deterministic seed so the
    # same dataset samples appear in every epoch's TensorBoard images.
    import json
    _rng_viz = np.random.default_rng(42)
    n_val   = len(val_loader.dataset)
    n_train = len(train_loader.dataset)
    viz_val_indices   = _rng_viz.choice(n_val,   size=min(3, n_val),   replace=False).tolist()
    viz_train_indices = _rng_viz.choice(n_train, size=min(3, n_train), replace=False).tolist()
    viz_indices_path = os.path.join(args.out_dir, "viz_indices.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(viz_indices_path, "w") as _f:
        json.dump({"val": viz_val_indices, "train": viz_train_indices}, _f)

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
            model, train_loader, optimizer, device,
            lambda_grad=args.lambda_grad,
            lambda_smooth=args.lambda_smooth,
            lambda_normal=args.lambda_normal,
            lambda_mean=args.lambda_mean,
            lambda_mv=args.lambda_mv,
            K=torch.from_numpy(K_input).float() if K_input is not None else None,
            log_depth=args.log_depth,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            epoch=epoch,
            profile=args.profile,
        )
        val_metrics = validate(model, val_loader, device,
                               log_depth=args.log_depth,
                               depth_min=args.depth_min,
                               depth_max=args.depth_max)
        
        # Step cosine/warmup scheduler once per epoch
        scheduler.step()

        # Logging
        writer.add_scalar("loss/train_total",  train_metrics["total"],  epoch)
        writer.add_scalar("loss/train_charb",  train_metrics["charb"],  epoch)
        writer.add_scalar("loss/train_grad",   train_metrics["grad"],   epoch)
        writer.add_scalar("loss/train_smooth", train_metrics["smooth"], epoch)
        writer.add_scalar("loss/train_normal", train_metrics["normal"], epoch)
        writer.add_scalar("loss/train_mean",   train_metrics["mean"],   epoch)
        writer.add_scalar("loss/train_mv",     train_metrics["mv"],     epoch)
        writer.add_scalar("loss/val_l1_metric", val_metrics["l1_metric"], epoch)
        writer.add_scalar("loss/val_abs_rel",   val_metrics["abs_rel"],   epoch)
        writer.add_scalar("loss/val_rmse",      val_metrics["rmse"],      epoch)
        writer.add_scalar("loss/val_delta125",  val_metrics["delta_125"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # GPU stats
        gpu_stats = get_gpu_stats(device)
        if gpu_stats:
            writer.add_scalar("gpu/vram_used_mb", gpu_stats["vram_used_mb"], epoch)
            writer.add_scalar("gpu/vram_reserved_mb", gpu_stats["vram_reserved_mb"], epoch)
            if "gpu_util_pct" in gpu_stats:
                writer.add_scalar("gpu/utilization_pct", gpu_stats["gpu_util_pct"], epoch)

        print(f"Epoch {epoch:03d} | train: {train_metrics['total']:.5f} "
              f"(c:{train_metrics['charb']:.4f} g:{train_metrics['grad']:.4f} "
              f"s:{train_metrics['smooth']:.5f} n:{train_metrics['normal']:.4f} "
              f"m:{train_metrics['mean']:.5f} mv:{train_metrics['mv']:.4f}) "
              f"| val L1: {val_metrics['l1_metric']:.4f} | AbsRel: {val_metrics['abs_rel']:.4f} "
              f"| RMSE: {val_metrics['rmse']:.4f} | δ<1.25: {val_metrics['delta_125']:.3f}")
        print(f"          | pred_log: {val_metrics['pred_log_mean']:.3f} vs gt_log: {val_metrics['gt_log_mean']:.3f} "
              f"| pred_m: {val_metrics['pred_metric_mean']:.3f}m vs gt_m: {val_metrics['gt_metric_mean']:.3f}m")
        if gpu_stats:
            util_str = f" | GPU util: {gpu_stats['gpu_util_pct']:.0f}%" if "gpu_util_pct" in gpu_stats else ""
            print(f"          | VRAM: {gpu_stats['vram_used_mb']:.0f}/{gpu_stats['vram_reserved_mb']:.0f} MB (used/reserved){util_str}")
        
        # Save best
        if val_metrics["l1_metric"] < best_val_loss:
            best_val_loss = val_metrics["l1_metric"]
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
                    "log_depth": args.log_depth,
                    "alpha": ALPHA,
                    "use_pose_warp": args.use_pose_warp,
                },
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> Saved best model (val L1: {best_val_loss:.4f})")
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
        
        # Log images every epoch (fixed indices for comparability across epochs)
        log_images(writer, model, val_loader,   device, epoch, split="val",   n_images=3, fixed_indices=viz_val_indices)
        log_images(writer, model, train_loader, device, epoch, split="train", n_images=3, fixed_indices=viz_train_indices)
    
    writer.close()
    print(f"\nTraining complete! Best val L1: {best_val_loss:.4f}")
    print(f"Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
