import torch
import torch.nn as nn


class RhythmHead(nn.Module):
    """Onset probability head operating on shared backbone features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(384, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities.

        Args:
            x: Feature tensor of shape (batch, 384, T).

        Returns:
            Onset probability tensor of shape (batch, 1, T).
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.sigmoid(self.conv3(x))
