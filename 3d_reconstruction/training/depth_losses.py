"""Shared loss functions and reporting metrics for depth training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from config import DEPTH_MIN, D_MAX


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    error = torch.sqrt((prediction - target) ** 2 + eps**2)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_scales: int = 4,
) -> torch.Tensor:
    def gradient_x(values: torch.Tensor) -> torch.Tensor:
        return values[:, :, :, 1:] - values[:, :, :, :-1]

    def gradient_y(values: torch.Tensor) -> torch.Tensor:
        return values[:, :, 1:, :] - values[:, :, :-1, :]

    total = prediction.new_tensor(0.0)
    for scale in range(num_scales):
        if scale > 0:
            pooled_mask = F.avg_pool2d(mask, 2)
            prediction = F.avg_pool2d(prediction * mask, 2) / pooled_mask.clamp_min(1e-6)
            target = F.avg_pool2d(target * mask, 2) / pooled_mask.clamp_min(1e-6)
            mask = (pooled_mask > 0.5).float()
        residual = prediction - target
        valid_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        valid_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        total = total + (gradient_x(residual).abs() * valid_x).sum() / valid_x.sum().clamp_min(1.0)
        total = total + (gradient_y(residual).abs() * valid_y).sum() / valid_y.sum().clamp_min(1.0)
    return total / num_scales


_PIXEL_GRID_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _pixel_grid(
    height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device))
    if key not in _PIXEL_GRID_CACHE:
        u = torch.arange(width, device=device, dtype=torch.float32)
        v = torch.arange(height, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        _PIXEL_GRID_CACHE[key] = (uu, vv)
    return _PIXEL_GRID_CACHE[key]


def surface_normals(depth_m: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Calculate surface normals by backprojecting metric depth."""
    _, _, height, width = depth_m.shape
    K = K.to(device=depth_m.device, dtype=torch.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uu, vv = _pixel_grid(height, width, depth_m.device)
    depth = depth_m[:, 0]
    points = torch.stack(
        ((uu - cx) * depth / fx, (vv - cy) * depth / fy, depth), dim=1
    )
    du = F.pad(points[:, :, :, 2:] - points[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
    dv = F.pad(points[:, :, 2:, :] - points[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
    normal = torch.stack(
        (
            du[:, 1] * dv[:, 2] - du[:, 2] * dv[:, 1],
            du[:, 2] * dv[:, 0] - du[:, 0] * dv[:, 2],
            du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0],
        ),
        dim=1,
    )
    return F.normalize(normal, dim=1)


def normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    prediction_m = prediction * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    target_m = target * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    cosine = (surface_normals(prediction_m, K) * surface_normals(target_m, K)).sum(
        dim=1, keepdim=True
    )
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


def l1_metres(
    prediction: torch.Tensor, target_m: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    prediction_m = prediction * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    return ((prediction_m - target_m).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def worst_fraction_l1_metres(
    prediction: torch.Tensor,
    target_m: torch.Tensor,
    mask: torch.Tensor,
    fraction: float = 0.10,
) -> torch.Tensor:
    prediction_m = prediction * (D_MAX - DEPTH_MIN) + DEPTH_MIN
    errors = (prediction_m - target_m).abs()[mask > 0.5]
    if errors.numel() == 0:
        return prediction_m.sum() * 0.0
    count = max(1, math.ceil(errors.numel() * fraction))
    return torch.topk(errors, k=min(count, errors.numel()), largest=True).values.mean()
