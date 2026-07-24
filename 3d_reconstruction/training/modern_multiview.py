#!/usr/bin/env python3
"""
Modern coarse-to-fine multi-view event-depth training.

This entry point reuses the dataset, losses, diagnostics, optimization, and
generic command-line flags from multiview.py, but replaces its single-scale
model with:

  * a two-level ResNet/FPN path or configurable two/three-stage CasMVSNet
    feature pyramid,
  * compact group-wise correlation volumes,
  * validity-aware source-view aggregation,
  * 3-D hourglass cost regularization,
  * H/8 -> H/4 -> H/2 geometric cascade for casmvsnet_fpn, while other
    encoders retain their original H/4 -> H/2 (or H/8 -> H/4) path,
  * learned full-resolution 2-D depth refinement.

The usual multiview.py flags remain available.  The legacy-only
--cost_volume_ref_features flag is accepted for CLI compatibility but ignored:
the modern model conditions refinement through its FPN/reference decoder
instead of repeating reference features across every depth plane.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiview as legacy
from multiview import DEPTH_MIN, D_MAX, homo_warp_features


def _conv2d_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Residual2dBlock(nn.Module):
    """Basic residual block used by the deeper FPN encoder."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.projection = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


def _residual2d_stage(
    in_ch: int,
    out_ch: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    if blocks < 1:
        raise ValueError("A residual stage must contain at least one block")
    layers: list[nn.Module] = [Residual2dBlock(in_ch, out_ch, stride=stride)]
    layers.extend(Residual2dBlock(out_ch, out_ch) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual feature paths."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class FeaturePyramid(nn.Module):
    """Build shared multi-scale features for the selected geometric decoder."""

    def __init__(
        self,
        name: str,
        in_ch: int,
        feature_ch: int,
        base: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        middle_feature_ch: int = 0,
        fine_feature_ch: int = 0,
        fullres_fine_volume: bool = False,
        h4_coarse_volume: bool = False,
        decoder_type: str = "auto",
    ):
        super().__init__()
        self.name = name
        self.feature_ch = feature_ch
        self.casmvsnet_fpn = name in ("casmvsnet_fpn", "deep_fpn")
        if decoder_type not in ("auto", "two_stage_fpn", "three_stage_fpn"):
            raise ValueError(
                f"Unsupported decoder_type={decoder_type!r}; expected auto, "
                "two_stage_fpn, or three_stage_fpn"
            )
        if decoder_type == "auto":
            self.decoder_type = (
                "three_stage_fpn" if self.casmvsnet_fpn else "native"
            )
        else:
            if not self.casmvsnet_fpn:
                raise ValueError(
                    f"--decoder_type {decoder_type} requires --feature_encoder "
                    "casmvsnet_fpn or deep_fpn"
                )
            self.decoder_type = decoder_type
        self.three_stage_fpn = self.decoder_type == "three_stage_fpn"
        self.fullres_fine_volume = bool(fullres_fine_volume)
        self.h4_coarse_volume = bool(h4_coarse_volume)
        if self.fullres_fine_volume and not self.three_stage_fpn:
            raise ValueError(
                "--fullres_fine_volume requires the three_stage_fpn decoder"
            )
        if self.h4_coarse_volume and not self.fullres_fine_volume:
            raise ValueError(
                "--h4_coarse_volume requires --fullres_fine_volume so the three "
                "geometry stages remain distinct (H/4, H/2, H/1)"
            )
        if middle_feature_ch < 0 or fine_feature_ch < 0:
            raise ValueError("FPN feature-channel overrides must be >= 0")
        if not self.casmvsnet_fpn and (middle_feature_ch > 0 or fine_feature_ch > 0):
            raise ValueError(
                "--middle_feature_channels and --fine_feature_channels are only "
                "supported by casmvsnet_fpn and deep_fpn"
            )
        self.middle_ch = middle_feature_ch or max(16, feature_ch // 2)
        self.fine_ch = (
            fine_feature_ch or max(8, feature_ch // 4)
            if self.casmvsnet_fpn else max(16, feature_ch // 2)
        )

        if self.casmvsnet_fpn:
            # CasMVSNet-style bottom-up feature hierarchy (H/2, H/4, H/8)
            # followed by lateral/top-down fusion. All three maps are returned
            # because each one drives a geometric stage in the cascade.
            c0 = max(8, feature_ch // 8)
            c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
            if name == "deep_fpn":
                self.casmvs_stem = nn.Sequential(
                    nn.Conv2d(in_ch, c0, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c0),
                    nn.ReLU(inplace=True),
                )
                self.casmvs_half = _residual2d_stage(c0, c1, 3, stride=2)
                self.casmvs_quarter = _residual2d_stage(c1, c2, 4, stride=2)
                self.casmvs_eighth = _residual2d_stage(c2, c3, 6, stride=2)
            else:
                self.casmvs_stem = _conv2d_block(in_ch, c0, stride=1)
                self.casmvs_half = _conv2d_block(c0, c1, stride=2)
                self.casmvs_quarter = _conv2d_block(c1, c2, stride=2)
                self.casmvs_eighth = _conv2d_block(c2, c3, stride=2)
            self.casmvs_lateral_quarter = nn.Conv2d(c2, c3, 1, bias=False)
            self.casmvs_lateral_half = nn.Conv2d(c1, c2, 1, bias=False)
            self.casmvs_lateral_full = (
                nn.Conv2d(c0, c2, 1, bias=False)
                if self.fullres_fine_volume else None
            )
            self.casmvs_coarse_out = nn.Sequential(
                nn.Conv2d(c3, feature_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(feature_ch),
                nn.ReLU(inplace=True),
            )
            if self.three_stage_fpn:
                middle_in_ch = c2 if self.fullres_fine_volume else c3
                self.casmvs_middle_out = nn.Sequential(
                    nn.Conv2d(
                        middle_in_ch,
                        self.middle_ch,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.middle_ch),
                    nn.ReLU(inplace=True),
                )
            else:
                self.casmvs_middle_out = None
            self.casmvs_fine_reduce = nn.Conv2d(c3, c2, 1, bias=False)
            self.casmvs_fine_out = nn.Sequential(
                nn.Conv2d(c2, self.fine_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.fine_ch),
                nn.ReLU(inplace=True),
            )
            self.coarse_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            self.middle_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            self.fine_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            self.fusion_drop_path = DropPath(drop_path)
            return
        elif name.startswith("resnet"):
            try:
                from torchvision.models import resnet18, resnet34, resnet50
            except ImportError as exc:
                raise ImportError(f"--feature_encoder {name} requires torchvision") from exc

            high_resolution = name.endswith("_h4")
            base_name = name.removesuffix("_h4")
            constructors = {
                "resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50,
            }
            if base_name not in constructors:
                raise ValueError(f"Unsupported modern ResNet encoder: {name}")
            backbone = constructors[base_name](weights=None)
            backbone.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if high_resolution:
                backbone.layer2[0].conv1.stride = (1, 1)
                if hasattr(backbone.layer2[0], "conv2") and base_name == "resnet50":
                    backbone.layer2[0].conv2.stride = (1, 1)
                backbone.layer2[0].downsample[0].stride = (1, 1)

            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            layer1_ch = 256 if base_name == "resnet50" else 64
            layer2_ch = 512 if base_name == "resnet50" else 128
            self.high_resolution = high_resolution
            fine_in_ch = 64 if high_resolution else layer1_ch
            coarse_in_ch = layer2_ch
        elif name in ("cnn", "cnn_8"):
            width = max(32, min(base, 128))
            self.stem = _conv2d_block(in_ch, width, stride=2)
            self.pool = nn.Identity()
            self.layer1 = _conv2d_block(width, width * 2, stride=2)
            self.layer2 = (
                _conv2d_block(width * 2, width * 4, stride=2)
                if name == "cnn_8"
                else _conv2d_block(width * 2, width * 4, stride=1)
            )
            self.high_resolution = name != "cnn_8"
            fine_in_ch = width if self.high_resolution else width * 2
            coarse_in_ch = width * 4
        else:
            raise ValueError(
                "modern_multiview.py supports cnn, cnn_8, casmvsnet_fpn, deep_fpn, "
                "resnet18/34/50, and their *_h4 variants"
            )

        self.coarse_proj = nn.Sequential(
            nn.Conv2d(coarse_in_ch, feature_ch, 1, bias=False),
            nn.BatchNorm2d(feature_ch),
            nn.ReLU(inplace=True),
        )
        self.fine_proj = nn.Sequential(
            nn.Conv2d(fine_in_ch, self.fine_ch, 1, bias=False),
            nn.BatchNorm2d(self.fine_ch),
        )
        self.coarse_to_fine = nn.Conv2d(feature_ch, self.fine_ch, 1, bias=False)
        self.coarse_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fine_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fusion_drop_path = DropPath(drop_path)
        self.fine_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fine_ch, self.fine_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.fine_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.casmvsnet_fpn:
            stem = self.casmvs_stem(x)
            half = self.casmvs_half(stem)
            quarter = self.casmvs_quarter(half)
            eighth = self.casmvs_eighth(quarter)
            quarter_fused = self.casmvs_lateral_quarter(quarter) + self.fusion_drop_path(
                F.interpolate(eighth, size=quarter.shape[-2:], mode="bilinear", align_corners=False)
            )
            half_fused = self.casmvs_lateral_half(half) + self.fusion_drop_path(
                F.interpolate(
                    self.casmvs_fine_reduce(quarter_fused),
                    size=half.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            if self.fullres_fine_volume:
                assert self.casmvs_lateral_full is not None
                fine_fused = self.casmvs_lateral_full(stem) + self.fusion_drop_path(
                    F.interpolate(
                        half_fused,
                        size=stem.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                fine_fused = half_fused
            if not self.three_stage_fpn:
                # Retain the current FPN's H/8 contextual path, but perform
                # geometry only at H/4 (global) and H/2 (local).
                coarse = self.coarse_dropout(
                    self.casmvs_coarse_out(quarter_fused)
                )
                fine = self.fine_dropout(self.casmvs_fine_out(half_fused))
                return coarse, fine
            coarse_source = quarter_fused if self.h4_coarse_volume else eighth
            coarse = self.coarse_dropout(self.casmvs_coarse_out(coarse_source))
            middle_source = half_fused if self.fullres_fine_volume else quarter_fused
            assert self.casmvs_middle_out is not None
            middle = self.middle_dropout(self.casmvs_middle_out(middle_source))
            fine = self.fine_dropout(self.casmvs_fine_out(fine_fused))
            return coarse, middle, fine

        stem = self.stem(x)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        fine_raw = stem if self.high_resolution else layer1
        coarse = self.coarse_dropout(self.coarse_proj(layer2))
        fine = self.fine_proj(fine_raw)
        fine = fine + self.fusion_drop_path(
            F.interpolate(
                self.coarse_to_fine(coarse),
                size=fine.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        return coarse, self.fine_dropout(self.fine_out(fine))


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = math.gcd(out_ch, min(8, out_ch))
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostHourglass3D(nn.Module):
    """Configurable 3-D encoder/decoder; wide layers run on smaller volumes."""

    def __init__(
        self,
        in_ch: int,
        base: int,
        levels: int = 2,
        bottleneck_dropout: float = 0.0,
    ):
        super().__init__()
        base = max(4, base)
        if levels < 1:
            raise ValueError("hourglass levels must be >= 1")
        self.stem = nn.Sequential(
            Conv3dBlock(in_ch, base),
            Conv3dBlock(base, base),
        )
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        channels = [base]
        for level in range(levels):
            in_width = channels[-1]
            out_width = base * (2 ** (level + 1))
            self.down.append(nn.Sequential(
                Conv3dBlock(in_width, out_width, stride=2),
                Conv3dBlock(out_width, out_width),
            ))
            channels.append(out_width)
        for level in range(levels, 0, -1):
            self.up.append(Conv3dBlock(channels[level], channels[level - 1]))
        self.bottleneck_dropout = (
            nn.Dropout3d(bottleneck_dropout)
            if bottleneck_dropout > 0 else nn.Identity()
        )
        self.out = nn.Conv3d(base, 1, 3, padding=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(volume)]
        for down in self.down:
            skips.append(down(skips[-1]))
        x = self.bottleneck_dropout(skips[-1])
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(
                up(x), size=skip.shape[-3:], mode="trilinear", align_corners=False
            ) + skip
        return self.out(x).squeeze(1)


class FullResolutionRefiner(nn.Module):
    def __init__(self, in_ch: int, fine_ch: int, width: int, max_residual_m: float):
        super().__init__()
        self.max_residual_m = float(max_residual_m)
        width = max(16, width)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch + fine_ch + 1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(width // 2, 1, 3, padding=1)
        self.feature_channels = width // 2

    def forward(
        self,
        target: torch.Tensor,
        fine_feat: torch.Tensor,
        depth_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine_full = F.interpolate(
            fine_feat, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.body(torch.cat([target, fine_full, depth_m], dim=1))
        residual = torch.tanh(self.residual(features)) * self.max_residual_m
        return (depth_m + residual).clamp(DEPTH_MIN, D_MAX), features


class ConvexDepthUpsampler(nn.Module):
    """RAFT-style learned convex upsampling for an integer spatial scale."""

    def __init__(self, feature_ch: int, scale: int = 2):
        super().__init__()
        self.scale = int(scale)
        self.mask = nn.Conv2d(feature_ch, 9 * self.scale * self.scale, 3, padding=1)

    def forward(self, depth: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth.shape
        mask = self.mask(features).view(
            B, 1, 9, self.scale, self.scale, H, W
        )
        mask = torch.softmax(mask, dim=2)
        padded_depth = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        neighbours = F.unfold(padded_depth, kernel_size=3).view(
            B, 1, 9, 1, 1, H, W
        )
        up = torch.sum(mask * neighbours, dim=2)
        return up.permute(0, 1, 4, 2, 5, 3).reshape(
            B, 1, H * self.scale, W * self.scale
        )


class ModernMVSNet(nn.Module):
    """Modern two/three-stage coarse-to-fine MVS network."""

    architecture_name = "ModernMVSNet"

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        feature_ch: int | None = None,
        cost_base: int | None = None,
        fine_depths: int = 5,
        fine_window: float = 0.08,
        fine_offset_radius: float = 2.0,
        learned_fine_window: bool = False,
        masked_warp_aggregation: bool = True,
        cost_volume_ref_features: bool = False,
        single_view_fallback: bool = False,
        feature_encoder: str = "resnet34_h4",
        correlation_groups: int = 0,
        reference_channels: int = 0,
        coarse_cost_channels: int = 0,
        fine_cost_channels: int = 0,
        refiner_channels: int = 0,
        hourglass_levels: int = 2,
        coarse_hourglass_levels: int = 0,
        fine_hourglass_levels: int = 0,
        learned_view_weighting: bool = False,
        two_mode_fine_candidates: bool = False,
        fine_supervision: bool = False,
        fine_loss_weight: float = 0.3,
        variance_channels: int = 0,
        convex_upsampling: bool = False,
        fullres_geometry: bool = False,
        fullres_depths: int = 3,
        fullres_window: float = 0.01,
        fpn_dropout: float = 0.0,
        reference_dropout: float = 0.0,
        hourglass_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        cost_volume_type: str = "correlation",
        middle_depths: int = 8,
        middle_window: float = 0.12,
        middle_cost_channels: int = 0,
        middle_hourglass_levels: int = 0,
        middle_feature_channels: int = 0,
        fine_feature_channels: int = 0,
        middle_supervision: bool = False,
        middle_loss_weight: float = 0.3,
        fullres_fine_volume: bool = False,
        h4_coarse_volume: bool = False,
        decoder_type: str = "auto",
    ):
        super().__init__()
        del cost_volume_ref_features  # legacy-only: never repeat ref features over depth
        if fine_depths < 3:
            raise ValueError("fine_depths must be >= 3")
        if middle_depths < 3:
            raise ValueError("middle_depths must be >= 3")
        if middle_window <= 0:
            raise ValueError("middle_window must be > 0")
        if middle_loss_weight < 0:
            raise ValueError("middle_loss_weight must be >= 0")
        feature_ch = feature_ch or max(32, base)
        cost_base = cost_base or max(8, base // 4)
        coarse_cost_base = coarse_cost_channels or cost_base
        middle_cost_base = middle_cost_channels or cost_base
        fine_cost_base = fine_cost_channels or cost_base
        self.fine_depths = int(fine_depths)
        self.middle_depths = int(middle_depths)
        self.middle_window = float(middle_window)
        self.fine_window = float(fine_window)
        self.fine_offset_radius = float(fine_offset_radius)
        self.learned_fine_window = bool(learned_fine_window)
        self.masked_warp_aggregation = bool(masked_warp_aggregation)
        self.single_view_fallback = bool(single_view_fallback)
        self.reference_channels = int(reference_channels)
        self.coarse_hourglass_levels = int(coarse_hourglass_levels or hourglass_levels)
        self.middle_hourglass_levels = int(middle_hourglass_levels or hourglass_levels)
        self.fine_hourglass_levels = int(fine_hourglass_levels or hourglass_levels)
        self.learned_view_weighting = bool(learned_view_weighting)
        self.two_mode_fine_candidates = bool(two_mode_fine_candidates)
        self.fine_supervision = bool(fine_supervision)
        self.fine_loss_weight = float(fine_loss_weight)
        self.middle_supervision = bool(middle_supervision)
        self.middle_loss_weight = float(middle_loss_weight)
        self.fullres_fine_volume = bool(fullres_fine_volume)
        self.h4_coarse_volume = bool(h4_coarse_volume)
        self.variance_channels = int(variance_channels)
        self.convex_upsampling = bool(convex_upsampling)
        self.fullres_geometry = bool(fullres_geometry)
        self.fullres_depths = int(fullres_depths)
        self.fullres_window = float(fullres_window)
        self.reference_dropout = float(reference_dropout)
        self.cost_volume_type = str(cost_volume_type)
        if self.cost_volume_type not in ("correlation", "variance"):
            raise ValueError(
                f"Unsupported cost_volume_type={self.cost_volume_type!r}; "
                "expected 'correlation' or 'variance'"
            )
        if self.cost_volume_type == "variance" and self.learned_view_weighting:
            raise ValueError(
                "--learned_view_weighting is currently only supported with "
                "--cost_volume_type correlation"
            )
        self.aux_fine_pred: torch.Tensor | None = None
        self.aux_middle_pred: torch.Tensor | None = None

        self.feature = FeaturePyramid(
            feature_encoder,
            in_ch,
            feature_ch,
            base,
            dropout=fpn_dropout,
            drop_path=drop_path_rate,
            middle_feature_ch=middle_feature_channels,
            fine_feature_ch=fine_feature_channels,
            fullres_fine_volume=self.fullres_fine_volume,
            h4_coarse_volume=self.h4_coarse_volume,
            decoder_type=decoder_type,
        )
        self.decoder_type = self.feature.decoder_type
        self.three_stage_cascade = self.feature.three_stage_fpn
        if self.middle_supervision and not self.three_stage_cascade:
            raise ValueError(
                "--middle_supervision requires the three_stage_fpn decoder"
            )
        if self.fullres_fine_volume and not self.three_stage_cascade:
            raise ValueError(
                "--fullres_fine_volume requires the three_stage_fpn decoder"
            )
        if self.h4_coarse_volume and not self.fullres_fine_volume:
            raise ValueError(
                "--h4_coarse_volume requires --fullres_fine_volume"
            )
        if self.fullres_fine_volume and self.convex_upsampling:
            raise ValueError(
                "--convex_upsampling is unnecessary with --fullres_fine_volume"
            )
        if self.fullres_fine_volume and self.fullres_geometry:
            raise ValueError(
                "--fullres_geometry cannot be combined with --fullres_fine_volume"
            )
        if self.fullres_fine_volume and self.single_view_fallback:
            raise ValueError(
                "--single_view_fallback currently requires the 2-D refiner and "
                "cannot be combined with --fullres_fine_volume"
            )
        requested_groups = correlation_groups if self.cost_volume_type == "correlation" else 0
        self.coarse_groups = self._resolve_groups(feature_ch, requested_groups)
        self.middle_groups = (
            self._resolve_groups(
                self.feature.middle_ch,
                requested_groups,
                allow_reduce=True,
            )
            if self.three_stage_cascade else 0
        )
        self.fine_groups = self._resolve_groups(
            self.feature.fine_ch,
            requested_groups,
            allow_reduce=self.three_stage_cascade,
        )
        if self.reference_channels > 0:
            self.coarse_reference = nn.Conv2d(
                feature_ch, self.reference_channels, 1, bias=False
            )
            self.fine_reference = nn.Conv2d(
                self.feature.fine_ch, self.reference_channels, 1, bias=False
            )
            self.middle_reference = (
                nn.Conv2d(
                    self.feature.middle_ch, self.reference_channels, 1, bias=False
                )
                if self.three_stage_cascade else None
            )
        else:
            self.coarse_reference = None
            self.middle_reference = None
            self.fine_reference = None
        primary_coarse_channels = (
            feature_ch if self.cost_volume_type == "variance" else self.coarse_groups
        )
        primary_fine_channels = (
            self.feature.fine_ch
            if self.cost_volume_type == "variance" else self.fine_groups
        )
        primary_middle_channels = (
            self.feature.middle_ch
            if self.cost_volume_type == "variance" else self.middle_groups
        ) if self.three_stage_cascade else 0
        if self.variance_channels > 0:
            self.coarse_variance = nn.Conv3d(
                primary_coarse_channels, self.variance_channels, 1, bias=False
            )
            self.fine_variance = nn.Conv3d(
                primary_fine_channels, self.variance_channels, 1, bias=False
            )
            self.middle_variance = (
                nn.Conv3d(
                    primary_middle_channels, self.variance_channels, 1, bias=False
                )
                if self.three_stage_cascade else None
            )
        else:
            self.coarse_variance = None
            self.middle_variance = None
            self.fine_variance = None
        if self.learned_view_weighting:
            self.coarse_view_weight = self._make_view_weight_head(self.coarse_groups)
            self.fine_view_weight = self._make_view_weight_head(self.fine_groups)
            self.middle_view_weight = (
                self._make_view_weight_head(self.middle_groups)
                if self.three_stage_cascade else None
            )
        else:
            self.coarse_view_weight = None
            self.middle_view_weight = None
            self.fine_view_weight = None
        volume_channels_coarse = (
            primary_coarse_channels + self.reference_channels + self.variance_channels + 1
        )
        volume_channels_fine = (
            primary_fine_channels + self.reference_channels + self.variance_channels + 1
        )
        volume_channels_middle = (
            primary_middle_channels
            + self.reference_channels
            + self.variance_channels
            + 1
        ) if self.three_stage_cascade else 0
        self.coarse_cost = CostHourglass3D(
            volume_channels_coarse,
            coarse_cost_base,
            self.coarse_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.fine_cost = CostHourglass3D(
            volume_channels_fine,
            fine_cost_base,
            self.fine_hourglass_levels,
            bottleneck_dropout=hourglass_dropout,
        )
        self.middle_cost = (
            CostHourglass3D(
                volume_channels_middle,
                middle_cost_base,
                self.middle_hourglass_levels,
                bottleneck_dropout=hourglass_dropout,
            )
            if self.three_stage_cascade else None
        )
        if self.three_stage_cascade:
            geometry_scales = (
                "H/4/H/2/H/1"
                if self.h4_coarse_volume
                else ("H/8/H/2/H/1" if self.fullres_fine_volume else "H/8/H/4/H/2")
            )
        elif self.feature.casmvsnet_fpn:
            geometry_scales = "H/4/H/2"
        else:
            geometry_scales = "H/4/H/2" if self.feature.high_resolution else "H/8/H/4"
        correlation_groups_summary = (
            f"{self.coarse_groups}"
            f"/{self.middle_groups if self.three_stage_cascade else '-'}"
            f"/{self.fine_groups}"
            if self.cost_volume_type == "correlation"
            else "n/a"
        )
        self.capacity_summary = (
            f"decoder={self.decoder_type}  "
            f"cascade_stages={3 if self.three_stage_cascade else 2}  "
            f"feature_coarse/middle/fine={feature_ch}"
            f"/{self.feature.middle_ch if self.three_stage_cascade else '-'}"
            f"/{self.feature.fine_ch}  "
            f"corr_groups={correlation_groups_summary}  "
            f"volume_type={self.cost_volume_type}  "
            f"reference={self.reference_channels}  "
            f"cost_coarse/middle/fine={coarse_cost_base}"
            f"/{middle_cost_base if self.three_stage_cascade else '-'}"
            f"/{fine_cost_base}  "
            f"geometry_scales={geometry_scales}  "
            f"refiner={'none' if self.fullres_fine_volume else (refiner_channels or max(32, min(base // 2, 128)))}  "
            f"hourglass_levels={self.coarse_hourglass_levels}"
            f"/{self.middle_hourglass_levels if self.three_stage_cascade else '-'}"
            f"/{self.fine_hourglass_levels}  "
            f"view_weighting={self.learned_view_weighting}  "
            f"two_mode={self.two_mode_fine_candidates}  variance={self.variance_channels}  "
            f"convex_up={self.convex_upsampling}  fullres_geometry={self.fullres_geometry}"
            f"  dropout(fpn/ref/hg)={fpn_dropout:g}/{reference_dropout:g}/{hourglass_dropout:g}"
            f"  drop_path={drop_path_rate:g}"
            f"  supervision(middle/fine)={self.middle_supervision}/{self.fine_supervision}"
        )

        window_hidden = max(8, cost_base)
        if self.learned_fine_window:
            self.window_head = nn.Sequential(
                nn.Conv2d(self.feature.fine_ch + 1, window_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(window_hidden, 1, 3, padding=1),
            )

        refine_width = refiner_channels or max(32, min(base // 2, 128))
        self.refiner = (
            None
            if self.fullres_fine_volume
            else FullResolutionRefiner(
                in_ch,
                self.feature.fine_ch,
                refine_width,
                max_residual_m=min(0.05, self.fine_window),
            )
        )
        refinement_feature_channels = (
            self.feature.fine_ch
            if self.fullres_fine_volume
            else self.refiner.feature_channels
        )
        self.convex_upsampler = (
            ConvexDepthUpsampler(self.feature.fine_ch, scale=2)
            if self.convex_upsampling else None
        )
        if self.fullres_geometry:
            full_feature_ch = max(8, min(16, self.feature.fine_ch // 2))
            self.full_feature = nn.Sequential(
                nn.Conv2d(in_ch, full_feature_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(full_feature_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(full_feature_ch, full_feature_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(full_feature_ch),
                nn.ReLU(inplace=True),
            )
            self.full_groups = self._resolve_groups(full_feature_ch, 4)
            self.full_view_weight = (
                self._make_view_weight_head(self.full_groups)
                if self.learned_view_weighting else None
            )
            full_volume_channels = (
                full_feature_ch
                if self.cost_volume_type == "variance" else self.full_groups
            )
            self.full_cost = nn.Sequential(
                Conv3dBlock(full_volume_channels + 1, 8),
                Conv3dBlock(8, 8),
                nn.Conv3d(8, 1, 3, padding=1),
            )
        confidence_in = refinement_feature_channels + 5
        self.confidence_head = nn.Sequential(
            nn.Conv2d(confidence_in, max(16, cost_base * 2), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(16, cost_base * 2), 1, 1),
        )
        if self.single_view_fallback:
            self.single_depth_head = nn.Sequential(
                nn.Conv2d(refinement_feature_channels, 1, 3, padding=1),
                nn.Sigmoid(),
            )
            self.fusion_head = nn.Sequential(
                nn.Conv2d(refinement_feature_channels + 2, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _resolve_groups(
        channels: int,
        requested: int,
        allow_reduce: bool = False,
    ) -> int:
        if requested > 0:
            if requested > channels or channels % requested != 0:
                if allow_reduce:
                    for groups in range(min(requested, channels), 0, -1):
                        if channels % groups == 0:
                            return groups
                raise ValueError(
                    f"correlation_groups={requested} must divide feature channels "
                    f"({channels}) and cannot exceed them"
                )
            return requested
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def _make_view_weight_head(groups: int) -> nn.Module:
        hidden = max(4, min(16, groups))
        return nn.Sequential(
            nn.Conv3d(groups + 1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, 1),
        )

    def _correlation_volume(
        self,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        groups: int,
        reference_projection: nn.Module | None,
        variance_projection: nn.Module | None = None,
        view_weight_head: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref = feats[:, 0]
        corr_sum = torch.zeros(
            (B, groups, D, H, W), device=feats.device, dtype=feats.dtype
        )
        corr_sq_sum = torch.zeros_like(corr_sum) if variance_projection is not None else None
        valid_sum = torch.zeros(
            (B, 1, D, H, W), device=feats.device, dtype=feats.dtype
        )
        geometric_valid_sum = torch.zeros_like(valid_sum)
        channels_per_group = C // groups
        ref_grouped = ref.reshape(B, groups, channels_per_group, H, W).unsqueeze(3)

        for view in range(1, V):
            warped, valid = homo_warp_features(
                feats[:, view],
                cam_mats[:, view],
                cam_mats[:, 0],
                K,
                depth_values,
                return_valid_mask=True,
            )
            warped = warped.reshape(B, groups, channels_per_group, D, H, W)
            corr = (ref_grouped * warped).mean(dim=2)
            geometric_valid_sum = geometric_valid_sum + valid
            if view_weight_head is not None:
                reliability = torch.sigmoid(
                    view_weight_head(torch.cat([corr, valid], dim=1))
                ) * valid
                corr_sum = corr_sum + corr * reliability
                if corr_sq_sum is not None:
                    corr_sq_sum = corr_sq_sum + corr.square() * reliability
                valid_sum = valid_sum + reliability
            elif self.masked_warp_aggregation:
                corr_sum = corr_sum + corr * valid
                if corr_sq_sum is not None:
                    corr_sq_sum = corr_sq_sum + corr.square() * valid
                valid_sum = valid_sum + valid
            else:
                corr_sum = corr_sum + corr
                if corr_sq_sum is not None:
                    corr_sq_sum = corr_sq_sum + corr.square()
                valid_sum = valid_sum + valid

        if self.masked_warp_aggregation or view_weight_head is not None:
            corr_mean = corr_sum / valid_sum.clamp_min(1.0)
        else:
            corr_mean = corr_sum / max(V - 1, 1)
        valid_ratio = geometric_valid_sum / max(V - 1, 1)
        volume_parts = [corr_mean]
        if reference_projection is not None:
            reference = reference_projection(ref)
            if self.reference_dropout > 0:
                reference = F.dropout2d(
                    reference,
                    p=self.reference_dropout,
                    training=self.training,
                )
            reference = reference.unsqueeze(2).expand(-1, -1, D, -1, -1)
            volume_parts.append(reference)
        if variance_projection is not None and corr_sq_sum is not None:
            if self.masked_warp_aggregation or view_weight_head is not None:
                corr_second = corr_sq_sum / valid_sum.clamp_min(1.0)
            else:
                corr_second = corr_sq_sum / max(V - 1, 1)
            corr_variance = (corr_second - corr_mean.square()).clamp_min(0.0)
            volume_parts.append(variance_projection(corr_variance))
        volume_parts.append(valid_ratio)
        return torch.cat(volume_parts, dim=1), valid_ratio

    def _variance_volume(
        self,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        reference_projection: nn.Module | None,
        variance_projection: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MVSNet feature variance, including the reference as one observation."""
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref = feats[:, 0].unsqueeze(2).expand(-1, -1, D, -1, -1)
        feature_sum = ref.clone()
        feature_sq_sum = ref.square()
        observation_count = torch.ones(
            (B, 1, D, H, W), device=feats.device, dtype=feats.dtype
        )
        geometric_valid_sum = torch.zeros_like(observation_count)

        for view in range(1, V):
            warped, valid = homo_warp_features(
                feats[:, view],
                cam_mats[:, view],
                cam_mats[:, 0],
                K,
                depth_values,
                return_valid_mask=True,
            )
            # Invalid samples must not be treated as zero-valued observations.
            feature_sum = feature_sum + warped * valid
            feature_sq_sum = feature_sq_sum + warped.square() * valid
            observation_count = observation_count + valid
            geometric_valid_sum = geometric_valid_sum + valid

        mean = feature_sum / observation_count
        feature_variance = (
            feature_sq_sum / observation_count - mean.square()
        ).clamp_min(0.0)
        valid_ratio = geometric_valid_sum / max(V - 1, 1)
        volume_parts = [feature_variance]
        if reference_projection is not None:
            reference = reference_projection(feats[:, 0])
            if self.reference_dropout > 0:
                reference = F.dropout2d(
                    reference, p=self.reference_dropout, training=self.training
                )
            volume_parts.append(reference.unsqueeze(2).expand(-1, -1, D, -1, -1))
        if variance_projection is not None:
            volume_parts.append(variance_projection(feature_variance))
        volume_parts.append(valid_ratio)
        return torch.cat(volume_parts, dim=1), valid_ratio

    def _cost_volume(
        self,
        feats: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        groups: int,
        reference_projection: nn.Module | None,
        variance_projection: nn.Module | None = None,
        view_weight_head: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cost_volume_type == "variance":
            return self._variance_volume(
                feats,
                cam_mats,
                K,
                depth_values,
                reference_projection,
                variance_projection,
            )
        return self._correlation_volume(
            feats,
            cam_mats,
            K,
            depth_values,
            groups,
            reference_projection,
            variance_projection,
            view_weight_head,
        )

    @staticmethod
    def _regress(prob: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
        return torch.sum(prob * depth_values.to(prob.dtype), dim=1, keepdim=True)

    @staticmethod
    def _local_values(
        center_m: torch.Tensor,
        target_ref: torch.Tensor,
        count: int,
        half_window: float,
    ) -> torch.Tensor:
        center_up = F.interpolate(
            center_m, size=target_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        offsets = torch.linspace(
            -half_window,
            half_window,
            count,
            device=center_up.device,
            dtype=center_up.dtype,
        )
        return (center_up + offsets.view(1, count, 1, 1)).clamp(
            DEPTH_MIN, D_MAX
        )

    def _fine_values(
        self,
        coarse_m: torch.Tensor,
        coarse_prob: torch.Tensor,
        coarse_values: torch.Tensor,
        fine_ref: torch.Tensor,
    ) -> torch.Tensor:
        coarse_up = F.interpolate(
            coarse_m, size=fine_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.learned_fine_window:
            scale = 0.25 + 0.75 * torch.sigmoid(
                self.window_head(torch.cat([fine_ref, coarse_up], dim=1))
            )
            sigma = self.fine_window * scale
        else:
            sigma = torch.full_like(coarse_up, self.fine_window)
        offsets = torch.linspace(
            -self.fine_offset_radius,
            self.fine_offset_radius,
            self.fine_depths,
            device=coarse_up.device,
            dtype=coarse_up.dtype,
        )
        if not self.two_mode_fine_candidates:
            return (
                coarse_up + sigma * offsets.view(1, self.fine_depths, 1, 1)
            ).clamp(DEPTH_MIN, D_MAX)

        top1 = coarse_prob.argmax(dim=1, keepdim=True)
        depth_axis = torch.arange(
            coarse_prob.shape[1], device=coarse_prob.device
        ).view(1, -1, 1, 1)
        separated_scores = coarse_prob.masked_fill(
            (depth_axis - top1).abs() <= 1, -1.0
        )
        top2 = separated_scores.argmax(dim=1, keepdim=True)
        mode1 = torch.gather(coarse_values, 1, top1)
        mode2 = torch.gather(coarse_values, 1, top2)
        mode1 = F.interpolate(
            mode1, size=fine_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        mode2 = F.interpolate(
            mode2, size=fine_ref.shape[-2:], mode="bilinear", align_corners=False
        )
        local_offsets = sigma * offsets.view(1, self.fine_depths, 1, 1)
        return torch.cat([mode1 + local_offsets, mode2 + local_offsets], dim=1).clamp(
            DEPTH_MIN, D_MAX
        )

    def forward(
        self,
        imgs: torch.Tensor,
        cam_mats: torch.Tensor,
        K: torch.Tensor,
        depth_values: torch.Tensor,
        return_coarse: bool = False,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        B, V, C, H, W = imgs.shape
        self.aux_middle_pred = None
        if depth_values.dim() == 1:
            depth_values = depth_values.unsqueeze(0).expand(B, -1)
        elif depth_values.dim() != 2:
            raise ValueError(
                f"Expected global depth_values with shape (D,) or (B,D), got "
                f"{tuple(depth_values.shape)}"
            )
        feature_levels = self.feature(imgs.reshape(B * V, C, H, W))
        if self.three_stage_cascade:
            coarse_flat, middle_flat, fine_flat = feature_levels
            middle = middle_flat.view(B, V, *middle_flat.shape[1:])
        else:
            coarse_flat, fine_flat = feature_levels
            middle = None
        coarse = coarse_flat.view(B, V, *coarse_flat.shape[1:])
        fine = fine_flat.view(B, V, *fine_flat.shape[1:])

        K_coarse = K.clone()
        K_coarse[:, 0, :] *= coarse.shape[-1] / W
        K_coarse[:, 1, :] *= coarse.shape[-2] / H
        coarse_values = depth_values[:, :, None, None].expand(
            -1, -1, coarse.shape[-2], coarse.shape[-1]
        )
        coarse_volume, _ = self._cost_volume(
            coarse,
            cam_mats,
            K_coarse,
            coarse_values,
            self.coarse_groups,
            self.coarse_reference,
            self.coarse_variance,
            self.coarse_view_weight,
        )
        coarse_logits = self.coarse_cost(coarse_volume)
        coarse_prob = F.softmax(-coarse_logits.float(), dim=1).to(coarse_logits.dtype)
        coarse_m = self._regress(coarse_prob, coarse_values)

        previous_m = coarse_m
        previous_prob = coarse_prob
        previous_values = coarse_values
        if self.three_stage_cascade:
            assert middle is not None and self.middle_cost is not None
            middle_values = self._local_values(
                coarse_m,
                middle[:, 0],
                self.middle_depths,
                self.middle_window,
            )
            K_middle = K.clone()
            K_middle[:, 0, :] *= middle.shape[-1] / W
            K_middle[:, 1, :] *= middle.shape[-2] / H
            middle_volume, _ = self._cost_volume(
                middle,
                cam_mats,
                K_middle,
                middle_values,
                self.middle_groups,
                self.middle_reference,
                self.middle_variance,
                self.middle_view_weight,
            )
            middle_logits = self.middle_cost(middle_volume)
            middle_prob = F.softmax(-middle_logits.float(), dim=1).to(
                middle_logits.dtype
            )
            middle_m = self._regress(middle_prob, middle_values)
            self.aux_middle_pred = (
                (middle_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
            ).clamp(0.0, 1.0)
            previous_m = middle_m
            previous_prob = middle_prob
            previous_values = middle_values

        fine_values = self._fine_values(
            previous_m, previous_prob, previous_values, fine[:, 0]
        )
        K_fine = K.clone()
        K_fine[:, 0, :] *= fine.shape[-1] / W
        K_fine[:, 1, :] *= fine.shape[-2] / H
        fine_volume, fine_valid = self._cost_volume(
            fine,
            cam_mats,
            K_fine,
            fine_values,
            self.fine_groups,
            self.fine_reference,
            self.fine_variance,
            self.fine_view_weight,
        )
        fine_logits = self.fine_cost(fine_volume)
        fine_prob = F.softmax(-fine_logits.float(), dim=1).to(fine_logits.dtype)
        fine_m = self._regress(fine_prob, fine_values)
        self.aux_fine_pred = (
            (fine_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
        ).clamp(0.0, 1.0)
        if self.fullres_fine_volume:
            if fine_m.shape[-2:] != (H, W):
                raise RuntimeError(
                    "--fullres_fine_volume expected full-resolution fine features, "
                    f"got {tuple(fine_m.shape[-2:])} for input {(H, W)}"
                )
            depth_full = fine_m
        elif (
            self.convex_upsampler is not None
            and H == fine_m.shape[-2] * 2
            and W == fine_m.shape[-1] * 2
        ):
            depth_full = self.convex_upsampler(fine_m, fine[:, 0])
        else:
            depth_full = F.interpolate(
                fine_m, size=(H, W), mode="bilinear", align_corners=False
            )
        if self.fullres_fine_volume:
            refined_m = depth_full
            refinement_features = fine[:, 0]
        else:
            assert self.refiner is not None
            refined_m, refinement_features = self.refiner(
                imgs[:, 0], fine[:, 0], depth_full
            )

        if self.fullres_geometry:
            full_flat = self.full_feature(imgs.reshape(B * V, C, H, W))
            full = full_flat.view(B, V, *full_flat.shape[1:])
            offsets = torch.linspace(
                -self.fullres_window,
                self.fullres_window,
                self.fullres_depths,
                device=refined_m.device,
                dtype=refined_m.dtype,
            )
            full_values = (
                refined_m + offsets.view(1, self.fullres_depths, 1, 1)
            ).clamp(DEPTH_MIN, D_MAX)
            full_volume, _ = self._cost_volume(
                full,
                cam_mats,
                K,
                full_values,
                self.full_groups,
                None,
                None,
                self.full_view_weight,
            )
            full_logits = self.full_cost(full_volume).squeeze(1)
            full_prob = F.softmax(-full_logits.float(), dim=1).to(full_logits.dtype)
            refined_m = self._regress(full_prob, full_values)

        if self.single_view_fallback:
            single_norm = self.single_depth_head(refinement_features)
            mv_norm = ((refined_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)).clamp(0.0, 1.0)
            alpha = self.fusion_head(
                torch.cat([refinement_features, mv_norm, single_norm], dim=1)
            )
            final_norm = alpha * mv_norm + (1.0 - alpha) * single_norm
        else:
            final_norm = (
                (refined_m - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
            ).clamp(0.0, 1.0)

        confidence = None
        if return_uncertainty:
            eps = 1e-8
            posterior_std = torch.sum(
                fine_prob * (fine_values - fine_m).square(), dim=1, keepdim=True
            ).clamp_min(0.0).sqrt()
            entropy = -(fine_prob * torch.log(fine_prob + eps)).sum(dim=1, keepdim=True)
            max_prob = fine_prob.max(dim=1, keepdim=True).values
            valid_ratio = fine_valid.mean(dim=2)
            previous_up = F.interpolate(
                previous_m, size=fine_m.shape[-2:], mode="bilinear", align_corners=False
            )
            coarse_fine = torch.abs(fine_m - previous_up)
            diagnostics = torch.cat(
                [posterior_std, entropy, max_prob, valid_ratio, coarse_fine], dim=1
            )
            diagnostics = F.interpolate(
                diagnostics, size=(H, W), mode="bilinear", align_corners=False
            )
            confidence = torch.sigmoid(
                self.confidence_head(
                    torch.cat([refinement_features, diagnostics], dim=1)
                )
            )

        if return_coarse:
            coarse_full = F.interpolate(
                coarse_m, size=(H, W), mode="bilinear", align_corners=False
            )
            coarse_norm = (
                (coarse_full - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
            ).clamp(0.0, 1.0)
            if return_uncertainty:
                return coarse_norm, final_norm, confidence
            return coarse_norm, final_norm
        if return_uncertainty:
            return final_norm, confidence
        return final_norm


def upgrade_legacy_modern_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map checkpoints from the original fixed two-level hourglass layout."""
    replacements = {
        ".down1.": ".down.0.",
        ".down2.": ".down.1.",
        ".up2.": ".up.0.",
        ".up1.": ".up.1.",
    }
    upgraded: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        upgraded[new_key] = value
    return upgraded


def main() -> None:
    if "--cost_volume_ref_features" in sys.argv:
        print(
            "[modern_multiview] --cost_volume_ref_features is a legacy-only flag "
            "and is ignored; compact group correlation is always used.",
            flush=True,
        )
    legacy.MultiViewDepthNet = ModernMVSNet
    legacy.main()


if __name__ == "__main__":
    main()
