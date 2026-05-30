"""
Plain UNet for Event-to-Depth prediction (no recurrent units).

Same encoder-decoder structure as E2Depth but with ordinary conv layers
instead of ConvLSTM cells.  Each encoder block is a strided conv + BN + ReLU;
each decoder block is bilinear upsample + two convolutions.  The model is
fully stateless — states are always an empty list.

Compatible with the train.py sequence pipeline:
  forward(events, states=None, T_rel=None) -> (pred, [])
"""

from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .e2depth import e2depth_loss


# ─────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Strided-conv downsampling + double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class DecoderBlock(nn.Module):
    """Bilinear upsample + concat skip + double conv."""

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
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualBlock(nn.Module):
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
        return self.relu(x + self.net(x))


# ─────────────────────────────────────────────────────────────────────
#  UNet model
# ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Stateless UNet for event-to-depth prediction.

    Args:
        in_channels:   Number of input channels (event voxel bins).
        base:          Base number of filters (doubled each encoder level).
        num_encoders:  Number of encoder / decoder levels.
        num_residuals: Number of residual blocks at the bottleneck.
        lambda_*:      Loss weights (same as E2DepthNet).
        depth_min/max: Depth range in metres.
    """

    # Interface flags consumed by train.py
    use_pose_warp: bool = False

    def __init__(
        self,
        in_channels: int = 5,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
        lambda_grad: float = 0.5,
        lambda_smooth: float = 0.01,
        lambda_normal: float = 0.1,
        lambda_mean: float = 0.1,
        lambda_mv: float = 0.0,   # no pose warp → MV loss disabled by default
        depth_min: float = 0.05,
        depth_max: float = 3.0,
    ):
        super().__init__()
        self.num_encoders  = num_encoders
        self.lambda_grad   = lambda_grad
        self.lambda_smooth = lambda_smooth
        self.lambda_normal = lambda_normal
        self.lambda_mean   = lambda_mean
        self.lambda_mv     = lambda_mv
        self.depth_min     = depth_min
        self.depth_max     = depth_max

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        ch = base
        for _ in range(num_encoders):
            out_ch = ch * 2
            self.encoders.append(EncoderBlock(ch, out_ch))
            ch = out_ch

        # Bottleneck residuals
        self.bottleneck = nn.Sequential(*[ResidualBlock(ch) for _ in range(num_residuals)])

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_encoders):
            skip_ch = ch // 2 if i < num_encoders - 1 else base
            out_ch  = ch // 2
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # Output head
        self.out = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List] = None,
        T_rel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Args:
            x:      (B, C, H, W) event voxel grid for one timestep
            states: ignored (kept for interface compatibility)
            T_rel:  ignored (no pose warp)
        Returns:
            pred:   (B, 1, H, W) depth in linear-normalised [0, 1]
            []:     empty states list
        """
        feat = self.head(x)
        skips = [feat]

        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i < self.num_encoders - 1:
                skips.append(feat)

        feat = self.bottleneck(feat)

        for i, dec in enumerate(self.decoders):
            feat = dec(feat, skips[-(i + 1)])

        return self.out(feat), []

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


# ─────────────────────────────────────────────────────────────────────
#  Registry entry points
# ─────────────────────────────────────────────────────────────────────

def add_unet_args(parser) -> None:
    """Register UNet-specific CLI arguments (shared with e2depth naming)."""
    parser.add_argument("--base", type=int, default=32,
                        help="Base number of filters")
    parser.add_argument("--num_encoders", type=int, default=3,
                        help="Number of encoder/decoder levels")
    parser.add_argument("--num_residuals", type=int, default=2,
                        help="Number of bottleneck residual blocks")


def build_unet(args, in_channels: int, K_input, input_hw) -> UNet:
    """Build UNet from parsed CLI args."""
    return UNet(
        in_channels=in_channels,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
        lambda_grad=args.lambda_grad,
        lambda_smooth=args.lambda_smooth,
        lambda_normal=args.lambda_normal,
        lambda_mean=args.lambda_mean,
        lambda_mv=args.lambda_mv,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
