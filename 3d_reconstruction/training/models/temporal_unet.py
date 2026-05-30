"""Temporal UNet: Pose-warped multi-frame event fusion for depth estimation.

For each reference frame, N voxel grids are loaded at symmetric temporal
offsets (e.g. [-2*k, -k, 0, +k, +2*k] for N=5).  Each non-reference frame
is warped into the reference camera view using the relative pose with a
fronto-parallel plane approximation at depth ``pose_d_ref``.  The N warped
grids are concatenated along the channel axis and fed through a standard UNet.

  Input:  (B, N, C, H, W) event voxels  +  (B, N, 4, 4) relative poses
  Output: (B, 1, H, W) normalised depth in [0, 1]

The reference frame is at index ``ref_idx = N // 2`` (the middle frame for
odd N).  Its pose is the identity, so it is never warped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet
from .e2depth import build_warp_grid


class TemporalUNet(nn.Module):
    """Multi-frame UNet that fuses N event voxel grids via pose-based warping.

    Parameters
    ----------
    in_channels  : number of temporal bins per voxel frame (C)
    n_frames     : total number of frames N (reference + neighbours)
    base, num_encoders, num_residuals : UNet architecture hyper-params
    K            : (3, 3) intrinsic matrix at the training resolution
    input_hw     : (H, W) network input spatial size
    pose_d_ref   : reference depth for fronto-parallel warp approximation (m)
    lambda_*     : loss weights forwarded to the inner UNet
    depth_min/max: depth range for normalisation (metres)
    """

    use_pose_warp: bool = False  # warping is done internally — not via the external flag

    def __init__(
        self,
        in_channels: int,
        n_frames: int,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
        K: Optional[np.ndarray] = None,
        input_hw: Optional[Tuple[int, int]] = None,
        pose_d_ref: float = 0.3,
        lambda_grad: float = 0.5,
        lambda_smooth: float = 0.01,
        lambda_normal: float = 0.1,
        lambda_mean: float = 0.1,
        lambda_mv: float = 0.0,
        depth_min: float = 0.05,
        depth_max: float = 3.0,
    ):
        super().__init__()
        self.n_frames   = n_frames
        self.ref_idx    = n_frames // 2
        self.pose_d_ref = pose_d_ref
        self.depth_min  = depth_min
        self.depth_max  = depth_max
        self.input_hw   = input_hw if input_hw is not None else (240, 320)

        if K is not None:
            K_t = torch.from_numpy(K).float() if isinstance(K, np.ndarray) else K.float()
        else:
            K_t = torch.eye(3)
        self.register_buffer("K", K_t)

        # Inner UNet accepts N*C input channels
        self._unet = UNet(
            in_channels=n_frames * in_channels,
            base=base,
            num_encoders=num_encoders,
            num_residuals=num_residuals,
            lambda_grad=lambda_grad,
            lambda_smooth=lambda_smooth,
            lambda_normal=lambda_normal,
            lambda_mean=lambda_mean,
            lambda_mv=lambda_mv,
            depth_min=depth_min,
            depth_max=depth_max,
        )

    def forward(
        self,
        events: torch.Tensor,
        poses_rel: torch.Tensor,
        states=None,
        T_rel=None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Parameters
        ----------
        events    : (B, N, C, H, W)  voxel grids for N frames
        poses_rel : (B, N, 4, 4)     T_ref_from_frame[i] for each frame i
                    (identity at i == ref_idx — reference frame is not warped)

        Returns
        -------
        pred : (B, 1, H, W) normalised depth in [0, 1]
        []   : empty state list (stateless model)
        """
        B, N, C, H, W = events.shape
        in_H, in_W = self.input_hw

        warped_frames: List[torch.Tensor] = []
        for i in range(N):
            ev_i = events[:, i]  # (B, C, H, W)
            if i == self.ref_idx:
                warped_frames.append(ev_i)
            else:
                grid = build_warp_grid(
                    poses_rel[:, i], self.K, H, W, in_H, in_W, self.pose_d_ref
                ).to(dtype=ev_i.dtype)
                warped = F.grid_sample(
                    ev_i, grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                warped_frames.append(warped)

        x = torch.cat(warped_frames, dim=1)  # (B, N*C, H, W)
        return self._unet(x, None, None)

    def compute_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        events: torch.Tensor,
        pred_prev=None,
        T_curr_from_prev=None,
        mask_prev=None,
        K: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Delegate to the inner UNet's loss.

        Multi-view consistency is disabled (no temporal sequence here).
        ``events`` should be the reference-frame voxels for edge-aware smoothness.
        """
        return self._unet.compute_loss(
            pred, gt, mask, events,
            pred_prev=None,
            T_curr_from_prev=None,
            mask_prev=None,
            K=K,
        )


# ─────────────────────────────────────────────────────────────────────
#  Registry entry points
# ─────────────────────────────────────────────────────────────────────

def add_temporal_unet_args(parser) -> None:
    """Register TemporalUNet-specific CLI arguments."""
    parser.add_argument("--base", type=int, default=32,
                        help="Base number of filters for UNet backbone")
    parser.add_argument("--num_encoders", type=int, default=3,
                        help="Number of encoder/decoder levels")
    parser.add_argument("--num_residuals", type=int, default=2,
                        help="Number of bottleneck residual blocks")
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Number of frames to fuse (odd → symmetric window)")
    parser.add_argument("--frame_offset", type=int, default=15,
                        help="Temporal offset (frame indices) between adjacent fused frames")
    parser.add_argument("--pose_d_ref", type=float, default=0.3,
                        help="Reference depth for fronto-parallel warp approximation (m)")


def build_temporal_unet(args, in_channels: int, K_input, input_hw) -> TemporalUNet:
    """Build TemporalUNet from parsed CLI args."""
    return TemporalUNet(
        in_channels=in_channels,
        n_frames=args.n_frames,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
        K=K_input,
        input_hw=input_hw,
        pose_d_ref=args.pose_d_ref,
        lambda_grad=args.lambda_grad,
        lambda_smooth=args.lambda_smooth,
        lambda_normal=args.lambda_normal,
        lambda_mean=args.lambda_mean,
        lambda_mv=0.0,  # no temporal sequence → no MV consistency loss
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
