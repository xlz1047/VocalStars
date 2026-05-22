"""PyTorch nn.Module head for breath event classification."""

import torch
import torch.nn as nn


class BreathHead(nn.Module):
    """Binary classification head that detects breath events from backbone features.

    Accepts frame-level embeddings of shape (batch, 384, T) from the shared backbone,
    applies temporal average pooling, and outputs a per-sample breath probability.

    Args:
        None — architecture is fixed to match the shared backbone output width of 384.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(384, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify backbone embeddings as breath or non-breath.

        Args:
            x: Float tensor of shape (batch, 256, T).

        Returns:
            Float tensor of shape (batch, 1) with probabilities in [0, 1].
        """
        pooled = x.mean(dim=-1)  # (batch, 256)
        out = self.relu(self.fc1(pooled))  # (batch, 64)
        return self.sigmoid(self.fc2(out))  # (batch, 1)

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss.

        Args:
            pred: Predicted probabilities, shape (batch, 1).
            target: Ground-truth labels, shape (batch, 1), values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        return nn.BCELoss()(pred, target.float())
