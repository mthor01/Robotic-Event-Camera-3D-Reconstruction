"""
E2Depth: Recurrent UNet for Event-to-Depth prediction.

Implementation based on:
"Learning Monocular Dense Depth from Events" (Hidalgo-Carrió et al., 3DV 2020)

Also contains:
  - Pose-warp grid builder (build_warp_grid)
  - All loss functions (charbonnier, gradient, smoothness, normal, multiview, combined)
  - Depth conversion utilities (depth_to_linear_normalized, linear_normalized_to_depth)
  - build_e2depth factory function for use with the model registry
"""

from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
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


# -----------------------------
# Building blocks
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
        return self.relu(out + residual)


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

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
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
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -----------------------------
# E2Depth Network
# -----------------------------
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
        in_channels:   Number of input channels (e.g., 5 for voxel grid bins).
        base:          Base number of filters (Nb=32 in paper).
        num_encoders:  Number of encoder layers (NE=3 in paper).
        num_residuals: Number of residual blocks (NR=2 in paper).
        use_pose_warp: Warp ConvLSTM hidden states with relative camera pose.
        K:             (3, 3) camera intrinsics array (required when use_pose_warp=True).
        input_hw:      (H, W) input resolution (required when use_pose_warp=True).
        pose_d_ref:    Reference depth used for pose-warp unprojection.
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
        # Loss hyperparameters — stored so compute_loss needs no external lambda args
        lambda_grad: float = 0.5,
        lambda_smooth: float = 0.01,
        lambda_normal: float = 0.1,
        lambda_mean: float = 0.1,
        lambda_mv: float = 0.2,
        depth_min: float = 0.05,
        depth_max: float = 3.0,
    ):
        super().__init__()
        self.num_encoders = num_encoders
        self.use_pose_warp = use_pose_warp
        self.pose_d_ref = pose_d_ref
        self.lambda_grad   = lambda_grad
        self.lambda_smooth = lambda_smooth
        self.lambda_normal = lambda_normal
        self.lambda_mean   = lambda_mean
        self.lambda_mv     = lambda_mv
        self.depth_min     = depth_min
        self.depth_max     = depth_max

        if use_pose_warp:
            if K is None or input_hw is None:
                raise ValueError("K and input_hw are required when use_pose_warp=True")
            K_t = torch.from_numpy(K).float() if isinstance(K, np.ndarray) else K.float()
            self.register_buffer("K", K_t)  # (3, 3)
            self.input_hw = input_hw
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

        # Prediction layer
        self.pred = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _warp_states(self, states: List, T_rel: torch.Tensor) -> List:
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
            h_w = F.grid_sample(h, grid, mode="bilinear", padding_mode="border", align_corners=True)
            c_w = F.grid_sample(c, grid, mode="bilinear", padding_mode="border", align_corners=True)
            warped.append((h_w, c_w))
        return warped

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        T_rel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if states is None:
            states = [None] * self.num_encoders

        if self.use_pose_warp and T_rel is not None:
            states = self._warp_states(states, T_rel)

        x = self.head(x)

        skips = [x]
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

    def compute_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        events: torch.Tensor,
        pred_prev: Optional[torch.Tensor] = None,
        T_curr_from_prev: Optional[torch.Tensor] = None,
        mask_prev: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined depth loss using this model's stored hyperparameters.

        Wraps ``e2depth_loss`` so that callers (train_one_epoch, validate) need
        not know about lambda weights or depth range — all of those are owned by
        the model instance and set at construction time via ``build_e2depth``.

        Args:
            pred:             (B, 1, H, W) predicted depth in linear-normalised [0, 1]
            gt:               (B, 1, H, W) ground-truth depth in linear-normalised [0, 1]
            mask:             (B, 1, H, W) binary validity mask
            events:           (B, C, H, W) event voxel grid (used for edge-aware smoothness)
            pred_prev:        previous timestep's prediction, for multi-view consistency
            T_curr_from_prev: (B, 4, 4) relative pose, for multi-view consistency
            mask_prev:        previous timestep's mask, for multi-view consistency
            K:                (3, 3) camera intrinsics tensor; pass None to skip
                              geometry-based losses (normal + multi-view)
        Returns:
            (loss, metrics) — same as e2depth_loss
        """
        return e2depth_loss(
            pred, gt, mask, events,
            lambda_grad=self.lambda_grad,
            lambda_smooth=self.lambda_smooth,
            lambda_normal=self.lambda_normal,
            lambda_mean=self.lambda_mean,
            pred_prev=pred_prev,
            T_curr_from_prev=T_curr_from_prev,
            mask_prev=mask_prev,
            K=K,
            lambda_mv=self.lambda_mv,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
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

    K_f = K.to(device=device, dtype=torch.float32).clone()
    K_f[0] = K_f[0] * (feat_W / in_W)
    K_f[1] = K_f[1] * (feat_H / in_H)

    T_f = T_rel.float()
    R_inv = T_f[:, :3, :3].transpose(1, 2)
    t_inv = -torch.bmm(R_inv, T_f[:, :3, 3:])

    u = torch.arange(feat_W, device=device, dtype=torch.float32)
    v = torch.arange(feat_H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    ones = torch.ones(feat_H, feat_W, device=device, dtype=torch.float32)
    pix = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)

    K_f_inv = torch.inverse(K_f)
    X = (K_f_inv @ pix) * d_ref
    X = X.unsqueeze(0).expand(B, -1, -1)

    X_prev = torch.bmm(R_inv, X) + t_inv

    K_f_b = K_f.unsqueeze(0).expand(B, -1, -1)
    p = torch.bmm(K_f_b, X_prev)
    z = p[:, 2:3].clamp(min=1e-6)
    p_xy = p[:, :2] / z

    norm_x = 2.0 * p_xy[:, 0] / max(feat_W - 1, 1) - 1.0
    norm_y = 2.0 * p_xy[:, 1] / max(feat_H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=2).view(B, feat_H, feat_W, 2)
    return grid


# -----------------------------
# Depth Conversion Utilities
# -----------------------------
def depth_to_linear_normalized(
    depth: torch.Tensor,
    d_min: float,
    d_max: float,
) -> torch.Tensor:
    """Convert metric depth to linearly normalized [0, 1]."""
    return ((depth - d_min) / (d_max - d_min)).clamp(0, 1)


def linear_normalized_to_depth(
    pred: torch.Tensor,
    d_min: float,
    d_max: float,
) -> torch.Tensor:
    """Convert linearly normalized [0, 1] back to metric depth."""
    return pred * (d_max - d_min) + d_min


# -----------------------------
# Loss Functions
# -----------------------------
def charbonnier_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier (pseudo-Huber) loss over valid pixels."""
    n = mask.sum().clamp_min(1.0)
    diff = pred - gt
    return (torch.sqrt(diff ** 2 + eps ** 2) * mask).sum() / n


def mean_alignment_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalises global mean offset over valid pixels."""
    n = mask.sum().clamp_min(1.0)
    pred_mean = (pred * mask).sum() / n
    gt_mean   = (gt   * mask).sum() / n
    return torch.abs(pred_mean - gt_mean)


def multi_scale_gradient_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    num_scales: int = 4,
) -> torch.Tensor:
    """Multi-scale gradient matching loss (Equation 4 in paper)."""
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

        residual = pred - gt
        grad_x = gradient_x(residual)
        grad_y = gradient_y(residual)
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = (torch.abs(grad_x) * mask_x).sum() / mask_x.sum().clamp_min(1.0)
        loss_y = (torch.abs(grad_y) * mask_y).sum() / mask_y.sum().clamp_min(1.0)
        total_loss = total_loss + loss_x + loss_y

    return total_loss / num_scales


def edge_aware_smoothness_loss(
    pred: torch.Tensor,
    events: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Edge-aware depth smoothness regulariser, restricted to valid pixels."""
    activity = events.abs().sum(dim=1, keepdim=True)
    a_max    = activity.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp_min(1e-6)
    activity = activity / a_max

    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    dx_ev = (activity[:, :, :, 1:] + activity[:, :, :, :-1]) * 0.5
    dy_ev = (activity[:, :, 1:, :] + activity[:, :, :-1, :]) * 0.5

    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = (dx_pred * torch.exp(-gamma * dx_ev) * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    loss_y = (dy_pred * torch.exp(-gamma * dy_ev) * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return loss_x + loss_y


# Module-level cache for pixel-coordinate grids.
# Keyed by (H, W, device_str) so the expensive meshgrid + stack allocation
# happens at most once per resolution.
_pixel_grid_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_pixel_grid(
    H: int, W: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (H, W, str(device))
    if key not in _pixel_grid_cache:
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pix = torch.stack(
            [uu.reshape(-1), vv.reshape(-1), torch.ones(H * W, device=device)], dim=0
        )
        _pixel_grid_cache[key] = (uu, vv, pix)
    return _pixel_grid_cache[key]


def _compute_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Geometrically correct surface normals via backprojection and cross product."""
    B, _, H, W = depth.shape
    device = depth.device
    K = K.to(device=device, dtype=torch.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uu, vv, _ = _get_pixel_grid(H, W, device)
    D = depth[:, 0]
    points = torch.stack([
        (uu - cx) * D / fx,
        (vv - cy) * D / fy,
        D,
    ], dim=1)

    du = points[:, :, :, 2:] - points[:, :, :, :-2]
    dv = points[:, :, 2:, :] - points[:, :, :-2, :]
    du = F.pad(du, (1, 1, 0, 0), mode="replicate")
    dv = F.pad(dv, (0, 0, 1, 1), mode="replicate")

    nx = du[:, 1] * dv[:, 2] - du[:, 2] * dv[:, 1]
    ny = du[:, 2] * dv[:, 0] - du[:, 0] * dv[:, 2]
    nz = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
    normals = torch.stack([nx, ny, nz], dim=1)
    return F.normalize(normals, dim=1)


def normal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """Surface normal cosine loss using geometrically correct backprojected normals."""
    pred_m = linear_normalized_to_depth(pred, depth_min, depth_max)
    gt_m   = linear_normalized_to_depth(gt,   depth_min, depth_max)
    n_pred = _compute_normals(pred_m, K)
    n_gt   = _compute_normals(gt_m,   K)
    cosine = (n_pred * n_gt).sum(dim=1, keepdim=True)
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


def multiview_consistency_loss(
    pred_prev: torch.Tensor,
    pred_curr: torch.Tensor,
    T_curr_from_prev: torch.Tensor,
    K: torch.Tensor,
    mask_prev: torch.Tensor,
    mask_curr: torch.Tensor,
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """Multi-view depth consistency using known relative camera pose."""
    d_prev = linear_normalized_to_depth(pred_prev, depth_min, depth_max)
    d_curr = linear_normalized_to_depth(pred_curr, depth_min, depth_max)

    B, _, H, W = d_prev.shape
    device = d_prev.device
    K_dev  = K.to(device=device, dtype=torch.float32)
    K_inv  = torch.inverse(K_dev)

    _, _, pix = _get_pixel_grid(H, W, device)

    d_flat   = d_prev.reshape(B, 1, H * W)
    rays     = (K_inv @ pix).unsqueeze(0).expand(B, -1, -1)
    X_prev   = rays * d_flat
    X_prev_h = torch.cat([X_prev, torch.ones(B, 1, H * W, device=device)], dim=1)

    T      = T_curr_from_prev.to(device=device, dtype=torch.float32)
    X_curr = torch.bmm(T, X_prev_h)[:, :3]

    K_b  = K_dev.unsqueeze(0).expand(B, -1, -1)
    p    = torch.bmm(K_b, X_curr)
    z    = p[:, 2:3].clamp(min=1e-6)
    p_xy = p[:, :2] / z

    norm_x = (2.0 * p_xy[:, 0] / max(W - 1, 1) - 1.0).view(B, H, W)
    norm_y = (2.0 * p_xy[:, 1] / max(H - 1, 1) - 1.0).view(B, H, W)
    grid   = torch.stack([norm_x, norm_y], dim=3)

    d_curr_sampled = F.grid_sample(
        d_curr, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
    )
    z_exp_m = X_curr[:, 2:3].view(B, 1, H, W).clamp_min(1e-6)

    in_bounds = (
        (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
    ).unsqueeze(1).float()
    pos_z = (z.view(B, 1, H, W) > 0.0).float()
    mask_c_sampled = F.grid_sample(
        mask_curr.float(), grid, align_corners=True,
        mode="nearest", padding_mode="zeros"
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
    depth_min: float = 0.05,
    depth_max: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined depth loss.

    L = L_charb + λ_grad·L_grad + λ_smooth·L_smooth + λ_normal·L_normal
      + λ_mean·L_mean + λ_mv·L_mv  (only when pred_prev and K are available)
    """
    l_charb  = charbonnier_loss(pred, gt, mask)
    l_grad   = multi_scale_gradient_loss(pred, gt, mask)
    l_smooth = edge_aware_smoothness_loss(pred, events, mask)
    pred_m_mean = linear_normalized_to_depth(pred, depth_min, depth_max)
    gt_m_mean   = linear_normalized_to_depth(gt,   depth_min, depth_max)
    l_mean   = mean_alignment_loss(pred_m_mean, gt_m_mean, mask)

    total = (
        l_charb
        + lambda_grad   * l_grad
        + lambda_smooth * l_smooth
        + lambda_mean   * l_mean
    )

    l_normal_val = 0.0
    if K is not None:
        l_normal = normal_loss(pred, gt, mask, K, depth_min=depth_min, depth_max=depth_max)
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
# Factory function (registry entry point)
# -----------------------------
def add_e2depth_args(parser) -> None:
    """Register e2depth-specific CLI arguments onto *parser* (or an argument group)."""
    parser.add_argument("--base", type=int, default=32,
                        help="Base filters (Nb in paper)")
    parser.add_argument("--num_encoders", type=int, default=3,
                        help="Number of encoder layers (NE in paper)")
    parser.add_argument("--num_residuals", type=int, default=2,
                        help="Number of residual blocks (NR in paper)")


def build_e2depth(args, in_channels: int, K_input, input_hw) -> E2DepthNet:
    """Build E2DepthNet from parsed CLI args.

    Args:
        args:        argparse.Namespace with base, num_encoders, num_residuals,
                     use_pose_warp, lambda_*, depth_min, depth_max fields.
        in_channels: Number of input event-voxel channels.
        K_input:     (3, 3) np.ndarray or None — effective camera intrinsics.
        input_hw:    (H, W) tuple or None — effective input resolution.

    Returns:
        Instantiated E2DepthNet (not yet moved to device).
    """
    return E2DepthNet(
        in_channels=in_channels,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
        use_pose_warp=args.use_pose_warp,
        K=K_input,
        input_hw=input_hw,
        lambda_grad=args.lambda_grad,
        lambda_smooth=args.lambda_smooth,
        lambda_normal=args.lambda_normal,
        lambda_mean=args.lambda_mean,
        lambda_mv=args.lambda_mv,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
