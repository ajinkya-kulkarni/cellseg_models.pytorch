import torch
import torch.nn as nn

__all__ = ["ProbabilityNet"]


class ProbabilityNet(nn.Module):
    def __init__(self, embedding_dim: int = 3, width: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._relu_non_empty(self.fc1(x))
        x = self._relu_non_empty(self.fc2(x))
        x = self.fc3(x)
        return x

    def _relu_non_empty(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return torch.relu_(x)
