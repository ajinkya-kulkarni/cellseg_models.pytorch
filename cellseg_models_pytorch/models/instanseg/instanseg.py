from typing import Any, Dict

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.instanseg.instanseg_unet import instanseg_nuclei

__all__ = ["InstanSeg"]


class InstanSeg(BaseModelInst):
    model_name = "instanseg"

    def __init__(
        self,
        n_nuc_classes: int,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.model = instanseg_nuclei(
            n_nuc_classes=n_nuc_classes,
            enc_name=enc_name,
            enc_pretrain=enc_pretrain,
            enc_freeze=enc_freeze,
            **model_kwargs,
        )

        self.device = device
        self.model.to(device)

    def set_inference_mode(self, mixed_precision: bool = True) -> None:
        self.model.eval()
        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "cpu":
            mixed_precision = False
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(
            postproc_method="instanseg",
            postproc_kwargs={"pixel_classifier": self.model.probability_net},
        )
        self.inference_mode = True
