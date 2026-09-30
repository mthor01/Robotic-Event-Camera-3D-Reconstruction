"""U-Net model definitions used by training, evaluation, and reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_BINS


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat((inputs, skip), dim=1))


class _ResidualBlock(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(inputs + self.net(inputs))


class UNet(nn.Module):
    """Event-to-depth U-Net producing linear-normalized depth in ``[0, 1]``."""

    def __init__(
        self,
        in_ch: int = NUM_BINS,
        base: int = 32,
        num_encoders: int = 3,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.encoders = nn.ModuleList()
        channels = base
        for _ in range(num_encoders):
            self.encoders.append(_EncoderBlock(channels, channels * 2))
            channels *= 2
        self.bottleneck = nn.Sequential(
            *[_ResidualBlock(channels) for _ in range(num_residuals)]
        )
        self.decoders = nn.ModuleList()
        for index in range(num_encoders):
            skip_channels = channels // 2 if index < num_encoders - 1 else base
            self.decoders.append(
                _DecoderBlock(channels, skip_channels, channels // 2)
            )
            channels //= 2
        self.head = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        skips = [features]
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.encoders) - 1:
                skips.append(features)
        features = self.bottleneck(features)
        for index, decoder in enumerate(self.decoders):
            features = decoder(features, skips[-(index + 1)])
        return self.head(features)


class UncertaintyUNet(UNet):
    """U-Net that predicts depth and log variance in normalized depth units."""

    def __init__(self, in_ch: int, base: int):
        super().__init__(in_ch=in_ch, base=base)
        self.head = nn.Conv2d(base, 2, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(inputs)
        skips = [features]
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.encoders) - 1:
                skips.append(features)
        features = self.bottleneck(features)
        for index, decoder in enumerate(self.decoders):
            features = decoder(features, skips[-(index + 1)])
        output = self.head(features)
        return torch.sigmoid(output[:, :1]), output[:, 1:2].clamp(-6.0, 3.0)


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros_like(inputs)
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class RecurrentUNet(UNet):
    """Single-view recurrent U-Net with a ConvGRU bottleneck state."""

    def __init__(self, in_ch: int, base: int):
        super().__init__(in_ch=in_ch, base=base)
        self.gru = ConvGRUCell(base * (2 ** len(self.encoders)))

    def _encode_one(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.stem(inputs)
        skips = [features]
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.encoders) - 1:
                skips.append(features)
        return self.bottleneck(features), skips

    def _decode_one(self, features: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for index, decoder in enumerate(self.decoders):
            features = decoder(features, skips[-(index + 1)])
        return self.head(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        if inputs.dim() != 5:
            raise ValueError(
                "RecurrentUNet expects (B,T,C,H,W) or (B,C,H,W), "
                f"got {tuple(inputs.shape)}"
            )
        hidden = None
        final_skips = None
        for index in range(inputs.shape[1]):
            features, final_skips = self._encode_one(inputs[:, index])
            hidden = self.gru(features, hidden)
        return self._decode_one(hidden, final_skips)
