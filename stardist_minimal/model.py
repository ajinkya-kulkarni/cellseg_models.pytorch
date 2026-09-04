import timm
import torch
import torch.nn as nn


class _ConvRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(),
        )


class _DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.block = nn.Sequential(
            _ConvRelu(in_channels + skip_channels, out_channels),
            _ConvRelu(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = x.repeat_interleave(self.scale_factor, -2).repeat_interleave(
            self.scale_factor, -1
        )
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.block(x)


class _Head(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            _ConvRelu(in_channels, 128),
            nn.Conv2d(128, out_channels, 1, bias=False),
        )


class StarDist(nn.Module):
    """Single-class 2D StarDist network: probability map + radial distances."""

    def __init__(
        self,
        n_rays: int = 32,
        encoder_name: str = "efficientnet_b5",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n_rays = n_rays
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=False,
        )
        if not hasattr(self.encoder, "forward_intermediates"):
            raise ValueError(f"{encoder_name!r} does not support forward_intermediates")

        feature_info = list(self.encoder.feature_info)[:4]
        channels = [int(info["num_chs"]) for info in feature_info]
        reductions = [int(info["reduction"]) for info in feature_info]

        rev_channels = channels[::-1]
        rev_reductions = reductions[::-1]
        scales = [a // b for a, b in zip(rev_reductions, rev_reductions[1:] + [1])]

        decoder_channels = (256, 128, 64, 32)
        stages = []
        in_channels = rev_channels[0]
        skips = rev_channels[1:]
        for i, out_channels in enumerate(decoder_channels):
            skip_channels = skips[i] if i < len(skips) else 0
            stages.append(
                _DecoderStage(
                    in_channels,
                    skip_channels,
                    out_channels,
                    scales[i],
                )
            )
            in_channels = out_channels

        self.decoder = nn.ModuleList(stages)
        self.prob_head = _Head(in_channels, 1)
        self.dist_head = _Head(in_channels, n_rays)

        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        for head in (self.prob_head, self.dist_head):
            for module in head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, features = self.encoder.forward_intermediates(x)
        offset = len(features) - len(self.encoder.feature_info)
        features = [features[i + offset] for i in range(4)][::-1]

        y = features[0]
        skips = features[1:]
        for i, stage in enumerate(self.decoder):
            y = stage(y, skips[i] if i < len(skips) else None)

        return self.prob_head(y), self.dist_head(y)
