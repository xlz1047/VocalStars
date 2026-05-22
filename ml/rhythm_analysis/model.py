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
            x: Feature tensor of shape (batch, 256, T).

        Returns:
            Onset probability tensor of shape (batch, 1, T).
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.sigmoid(self.conv3(x))

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
        """Binary cross-entropy loss with positive class weighting.

        Args:
            pred: Predicted onset probabilities, shape (batch, 1, T).
            target: Ground-truth onset labels, shape (batch, 1, T).
            pos_weight: Weight for positive (onset) class to handle class imbalance.

        Returns:
            Scalar loss tensor.
        """
        weight_tensor = torch.where(target == 1, torch.tensor(pos_weight), torch.tensor(1.0))
        criterion = nn.BCELoss(weight=weight_tensor)
        return criterion(pred, target)
