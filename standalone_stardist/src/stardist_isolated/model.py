from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


def _initialize_decoder(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def _initialize_head(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class FixedUnpool(nn.Module):
    """TensorPack-style fixed unpooling used by the existing StarDist model."""

    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        self.scale_factor = scale_factor
        self.register_buffer(
            "unpool_mat",
            torch.ones((scale_factor, scale_factor), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = list(x.shape)
        x = x.unsqueeze(-1)
        mat = self.unpool_mat.unsqueeze(0)
        out = torch.tensordot(x, mat, dims=1)
        out = out.permute(0, 1, 2, 4, 3, 5)
        return out.reshape(
            -1,
            in_shape[1],
            in_shape[2] * self.scale_factor,
            in_shape[3] * self.scale_factor,
        )


class ConvRelu(nn.Sequential):
    """Bias-free same-padded 3x3 convolution followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )


class DecoderStage(nn.Module):
    """One StarDist decoder stage: upsample, optional U-Net skip, two convs."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.upsample = FixedUnpool(scale_factor)
        merged_channels = in_channels + skip_channels
        self.block = nn.Sequential(
            ConvRelu(merged_channels, out_channels),
            ConvRelu(out_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    "Encoder/decoder spatial shapes do not match: "
                    f"{tuple(x.shape[-2:])} vs {tuple(skip.shape[-2:])}. "
                    "Use input sizes compatible with the encoder reductions."
                )
            x = torch.cat((x, skip), dim=1)
        return self.block(x)


class ExcitationHead(nn.Sequential):
    """StarDist head: 3x3 excitation conv then bias-free 1x1 output conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        excitation_channels: int = 128,
    ) -> None:
        super().__init__(
            ConvRelu(in_channels, excitation_channels),
            nn.Conv2d(
                excitation_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )


class StarDist(nn.Module):
    """Minimal single-class 2D StarDist dense prediction network.

    This keeps the default operations used by the existing repository's
    StarDist implementation while removing the generic multi-task framework.
    The model predicts only objectness and radial distances.
    """

    def __init__(
        self,
        n_rays: int = 32,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        depth: int = 4,
        out_channels: tuple[int, ...] = (256, 128, 64, 32),
        enc_out_indices: tuple[int, ...] | None = None,
        head_channels: int = 128,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if n_rays <= 0:
            raise ValueError("n_rays must be > 0")
        if depth <= 0:
            raise ValueError("depth must be > 0")
        if len(out_channels) != depth:
            raise ValueError("len(out_channels) must equal depth")

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))
        if len(enc_out_indices) != depth:
            raise ValueError("len(enc_out_indices) must equal depth")

        self.n_rays = n_rays
        self.enc_name = enc_name
        self.enc_out_indices = enc_out_indices

        self.encoder = timm.create_model(
            enc_name,
            pretrained=enc_pretrain,
            features_only=False,
            **(encoder_kwargs or {}),
        )
        if not hasattr(self.encoder, "forward_intermediates"):
            raise AttributeError(
                f"Encoder {enc_name!r} does not support forward_intermediates()."
            )
        if not hasattr(self.encoder, "feature_info"):
            raise AttributeError(f"Encoder {enc_name!r} does not expose feature_info.")

        all_feature_info = list(self.encoder.feature_info)
        try:
            feature_info = [all_feature_info[i] for i in enc_out_indices]
        except IndexError as exc:
            raise ValueError(
                f"enc_out_indices={enc_out_indices} are invalid for {enc_name!r}"
            ) from exc

        reductions = [int(info["reduction"]) for info in feature_info]
        channels = [int(info["num_chs"]) for info in feature_info]

        if reductions != sorted(reductions):
            raise ValueError(
                "Selected encoder features must be ordered from shallow to deep."
            )

        rev_reductions = list(reversed(reductions))
        rev_channels = list(reversed(channels))

        stage_reductions = rev_reductions + [1]
        scale_factors: list[int] = []
        for current, target in zip(stage_reductions[:-1], stage_reductions[1:]):
            if current % target != 0:
                raise ValueError(
                    "Encoder reductions must have integer consecutive ratios; "
                    f"got {current} -> {target}."
                )
            scale_factors.append(current // target)

        stages = []
        current_channels = rev_channels[0]
        skip_channels = rev_channels[1:]
        for i in range(depth):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            stage = DecoderStage(
                in_channels=current_channels,
                skip_channels=skip_ch,
                out_channels=out_channels[i],
                scale_factor=scale_factors[i],
            )
            stages.append(stage)
            current_channels = out_channels[i]

        self.decoder = nn.ModuleList(stages)
        self.objectness_head = ExcitationHead(
            current_channels,
            1,
            excitation_channels=head_channels,
        )
        self.ray_head = ExcitationHead(
            current_channels,
            n_rays,
            excitation_channels=head_channels,
        )

        _initialize_decoder(self.decoder)
        _initialize_head(self.objectness_head)
        _initialize_head(self.ray_head)

        if enc_freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, intermediates = self.encoder.forward_intermediates(x)
        all_feature_info = list(self.encoder.feature_info)
        offset = len(intermediates) - len(all_feature_info)
        return [intermediates[i + offset] for i in self.enc_out_indices]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._encoder_features(x)
        feats = list(reversed(feats))

        y = feats[0]
        skips = feats[1:]
        for i, stage in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            y = stage(y, skip)

        return {
            "objectness": self.objectness_head(y),
            "rays": self.ray_head(y),
        }
