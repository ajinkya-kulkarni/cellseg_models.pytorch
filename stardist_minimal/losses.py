import torch
import torch.nn as nn
import torch.nn.functional as F


class StarDistLoss(nn.Module):
    """Single-class StarDist objective.

    The score branch regresses the normalized per-instance EDT target with BCE + MSE.
    The ray branch uses foreground-normalized MAE plus the StarDist background
    regularizer. The ray branch weight (0.2) and background alpha (1e-4) match the
    StarDist training recipe used by the original repository.
    """

    def __init__(self, ray_weight: float = 0.2, background_reg: float = 1e-4) -> None:
        super().__init__()
        self.ray_weight = ray_weight
        self.background_reg = background_reg

    def forward(
        self,
        score_pred: torch.Tensor,
        ray_pred: torch.Tensor,
        score_target: torch.Tensor,
        ray_target: torch.Tensor,
        foreground: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        score_prob = torch.sigmoid(score_pred)
        fg = foreground.float()
        if fg.ndim == 3:
            fg = fg.unsqueeze(1)

        bce = F.binary_cross_entropy(score_prob, score_target.float(), reduction="none")
        score_bce = (bce * fg).sum() / fg.sum().clamp_min(1.0)
        score_mse = F.mse_loss(score_prob, score_target.float())
        score_loss = score_bce + score_mse

        ray_error = torch.abs(ray_pred - ray_target.float()).mean(dim=1, keepdim=True)
        ray_fg = (ray_error * fg).sum() / fg.sum().clamp_min(1.0)
        background = 1.0 - fg
        ray_bg = (torch.abs(ray_pred).mean(dim=1, keepdim=True) * background).mean()
        ray_loss = ray_fg + self.background_reg * ray_bg

        total = score_loss + self.ray_weight * ray_loss
        return {
            "loss": total,
            "score": score_loss,
            "score_bce": score_bce,
            "score_mse": score_mse,
            "rays": ray_loss,
        }
