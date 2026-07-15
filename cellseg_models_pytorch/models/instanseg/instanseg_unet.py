from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import (
    DecoderSoftOutput,
    MultiTaskDecoder,
)
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.instanseg._conf import _create_instanseg_args
from cellseg_models_pytorch.models.instanseg.probability_net import ProbabilityNet

__all__ = ["InstanSegUnet", "instanseg_nuclei"]


class InstanSegUnet(nn.ModuleDict):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_indices: Tuple[int, ...] = None,
        upsampling: str = "fixed-unpool",
        long_skip: str = "unet",
        merge_policy: str = "cat",
        short_skip: str = "basic",
        block_type: str = "basic",
        normalization: str = None,
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        attention: str = None,
        preattend: bool = False,
        out_size: int = None,
        encoder_kws: Dict[str, Any] = None,
        skip_kws: Dict[str, Any] = None,
        stem_skip_kws: Dict[str, Any] = None,
        inst_key: str = "type",
        **kwargs,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.inst_key = inst_key
        self.aux_key = "instanseg"
        self.enc_name = enc_name

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        n_layers = (1,) * depth
        n_blocks = ((2,),) * depth
        stage_kws = _create_instanseg_args(
            depth,
            normalization,
            activation,
            convolution,
            attention,
            preactivate,
            preattend,
            short_skip,
            use_style,
            block_type,
            merge_policy,
            skip_kws,
            upsampling,
        )

        self.add_module(
            self.enc_name,
            Encoder(
                timm_encoder_name=enc_name,
                timm_encoder_out_indices=enc_out_indices,
                timm_encoder_pretrained=enc_pretrain,
                timm_extra_kwargs=encoder_kws,
            ),
        )

        self.decoder = MultiTaskDecoder(
            decoders=decoders,
            heads=heads,
            out_channels=out_channels,
            enc_feature_info=self[self.enc_name].feature_info,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stage_kws=stage_kws,
            stem_skip_kws=stem_skip_kws,
            long_skip=long_skip,
            out_size=out_size,
            style_channels=style_channels,
            head_excitation_channels=128,
        )

        self.decoder.initialize()

        if enc_freeze:
            self[self.enc_name].freeze_encoder()

        self.name = f"InstanSegUnet-{enc_name}"

    def attach_probability_net(self) -> None:
        self.probability_net = ProbabilityNet(embedding_dim=3)

    def forward(self, x: torch.Tensor, return_pred_only: bool = True) -> Dict[str, Any]:
        enc_output, feats = self[self.enc_name](x)
        dec_out: DecoderSoftOutput = self.decoder(feats, x)

        res = {
            "nuc": dec_out.nuc_map,
            "tissue": dec_out.tissue_map,
            "cyto": dec_out.cyto_map,
        }

        if not return_pred_only:
            res["enc_feats"] = dec_out.enc_feats
            res["dec_feats"] = dec_out.dec_feats
            res["enc_out"] = enc_output

        return res


def instanseg_nuclei(n_nuc_classes: int, **kwargs) -> nn.Module:
    model = InstanSegUnet(
        decoders=("type",),
        heads={
            "type": {
                "nuc_instanseg": 4,
                "nuc_type": n_nuc_classes,
            }
        },
        **kwargs,
    )
    model.attach_probability_net()
    return model
