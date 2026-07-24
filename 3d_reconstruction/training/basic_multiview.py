#!/usr/bin/env python3
"""Minimal one-stage MVS baseline using a global feature-variance volume."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiview as legacy
from train_unet import DEPTH_MIN, D_MAX


class BasicMVSNet(nn.Module):
    """Shared 2-D encoder -> variance volume -> small 3-D CNN -> depth."""

    architecture_name = "BasicMVSNet"

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        feature_ch: int | None = None,
        cost_base: int | None = None,
        feature_encoder: str = "cnn",
        masked_warp_aggregation: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        feature_ch = feature_ch or base * 2
        cost_base = cost_base or max(base // 2, 8)
        self.masked_warp_aggregation = bool(masked_warp_aggregation)
        self.feature = legacy.make_feature_encoder(
            feature_encoder, in_ch, base, feature_ch
        )
        self.cost_regularizer = legacy.CostVolumeCNN(feature_ch, cost_base)
        self.capacity_summary = (
            f"single_stage=True  feature={feature_ch}  cost={cost_base}  "
            "volume_type=variance  reference=none  refiner=none"
        )

    def _variance_volume(
        self,
        features: torch.Tensor,
        camera_matrices: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        batch, views, _, _, _ = features.shape
        depths = depth_values.shape[-1] if depth_values.ndim == 2 else depth_values.shape[0]
        reference = features[:, 0]
        reference_volume = reference.unsqueeze(2).expand(-1, -1, depths, -1, -1)
        feature_sum = reference_volume.clone()
        feature_sq_sum = reference_volume.square()

        if self.masked_warp_aggregation:
            count = torch.ones(
                (batch, 1, depths, reference.shape[-2], reference.shape[-1]),
                device=features.device,
                dtype=features.dtype,
            )

        for view in range(1, views):
            if self.masked_warp_aggregation:
                warped, valid = legacy.homo_warp_features(
                    features[:, view],
                    camera_matrices[:, view],
                    camera_matrices[:, 0],
                    intrinsics,
                    depth_values,
                    return_valid_mask=True,
                )
                feature_sum = feature_sum + warped * valid
                feature_sq_sum = feature_sq_sum + warped.square() * valid
                count = count + valid
            else:
                warped = legacy.homo_warp_features(
                    features[:, view],
                    camera_matrices[:, view],
                    camera_matrices[:, 0],
                    intrinsics,
                    depth_values,
                )
                feature_sum = feature_sum + warped
                feature_sq_sum = feature_sq_sum + warped.square()

        denominator = count.clamp_min(1.0) if self.masked_warp_aggregation else views
        mean = feature_sum / denominator
        return (feature_sq_sum / denominator - mean.square()).clamp_min(0.0)

    @staticmethod
    def _regress_depth(
        probability: torch.Tensor, depth_values: torch.Tensor
    ) -> torch.Tensor:
        if depth_values.ndim == 1:
            candidates = depth_values[None, :, None, None]
        elif depth_values.ndim == 2:
            candidates = depth_values[:, :, None, None]
        else:
            raise ValueError(
                f"Expected depth candidates shaped (D,) or (B,D), got {depth_values.shape}"
            )
        return torch.sum(
            probability * candidates.to(probability.device, probability.dtype),
            dim=1,
            keepdim=True,
        )

    def forward(
        self,
        images: torch.Tensor,
        camera_matrices: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_values: torch.Tensor,
        return_coarse: bool = False,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if return_uncertainty:
            raise ValueError("BasicMVSNet has no uncertainty mechanism")

        batch, views, channels, height, width = images.shape
        flat_features = self.feature(
            images.reshape(batch * views, channels, height, width)
        )
        _, feature_channels, feature_height, feature_width = flat_features.shape
        features = flat_features.view(
            batch, views, feature_channels, feature_height, feature_width
        )

        feature_intrinsics = intrinsics.clone()
        feature_intrinsics[:, 0, :] *= feature_width / width
        feature_intrinsics[:, 1, :] *= feature_height / height
        volume = self._variance_volume(
            features, camera_matrices, feature_intrinsics, depth_values
        )
        cost = self.cost_regularizer(volume)
        probability = F.softmax(-cost.float(), dim=1).to(cost.dtype)
        depth_low = self._regress_depth(probability, depth_values)
        depth = F.interpolate(
            depth_low, size=(height, width), mode="bilinear", align_corners=False
        )
        prediction = (
            (depth - DEPTH_MIN) / (D_MAX - DEPTH_MIN)
        ).clamp(0.0, 1.0)
        if return_coarse:
            return prediction, prediction
        return prediction


def main() -> None:
    legacy.MultiViewDepthNet = BasicMVSNet
    legacy.main()


if __name__ == "__main__":
    main()
