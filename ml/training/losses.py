"""Multi-task loss functions: pitch regression, rhythm classification, and breath BCE."""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Weighted sum of per-task losses for pitch, rhythm, and breath prediction.

    Args:
        pitch_w:  Weight applied to the pitch BCE loss.
        rhythm_w: Weight applied to the rhythm/onset BCE loss.
        breath_w: Weight applied to the breath BCE loss.
    """

    def __init__(
        self,
        pitch_w: float = 1.0,
        rhythm_w: float = 0.5,
        breath_w: float = 0.3,
    ) -> None:
        super().__init__()
        self.pitch_w = pitch_w
        self.rhythm_w = rhythm_w
        self.breath_w = breath_w

        self._pitch_loss_fn = nn.BCEWithLogitsLoss()
        # pos_weight=10.0 counteracts class imbalance: onsets are rare vs. non-onset frames
        self._onset_loss_fn = nn.BCELoss()
        self._breath_loss_fn = nn.BCELoss()
        self._onset_pos_weight = 10.0

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute per-task losses and their weighted sum.

        Args:
            predictions: Dict with keys:
                ``pitch_logits`` — (batch, 360) raw logits.
                ``onset_probs``  — (batch, 1, T) or (batch, T) sigmoid probabilities.
                ``breath_prob``  — (batch, 1) or (batch,) sigmoid probability.
            targets: Dict with keys:
                ``pitch_bins``     — (batch, 360) binary multi-hot targets.
                ``onset_targets``  — (batch, T) binary onset targets.
                ``breath_target``  — (batch,) binary breath targets.

        Returns:
            Dict with keys ``pitch_loss``, ``rhythm_loss``, ``breath_loss``,
            ``total_loss`` (all scalar tensors).
        """
        pitch_logits: torch.Tensor = predictions["pitch_logits"]
        onset_probs: torch.Tensor = predictions["onset_probs"]
        breath_prob: torch.Tensor = predictions["breath_prob"]

        pitch_bins: torch.Tensor = targets["pitch_bins"]
        onset_targets: torch.Tensor = targets["onset_targets"]
        breath_target: torch.Tensor = targets["breath_target"]

        pitch_loss = self._pitch_loss_fn(
            pitch_logits,
            pitch_bins.float(),
        )

        # Flatten temporal dimension to match onset_probs and onset_targets
        onset_p = onset_probs.view(onset_probs.size(0), -1)  # (batch, T)
        onset_t = onset_targets.view(onset_targets.size(0), -1).float()

        # Manual pos-weight application: scale positive sample losses
        onset_weights = torch.ones_like(onset_t)
        onset_weights[onset_t > 0.5] = self._onset_pos_weight
        rhythm_loss = (onset_weights * nn.functional.binary_cross_entropy(
            onset_p, onset_t, reduction="none"
        )).mean()

        breath_p = breath_prob.view(-1)
        breath_t = breath_target.view(-1).float()
        breath_loss = self._breath_loss_fn(breath_p, breath_t)

        total_loss = (
            self.pitch_w * pitch_loss
            + self.rhythm_w * rhythm_loss
            + self.breath_w * breath_loss
        )

        return {
            "pitch_loss": pitch_loss,
            "rhythm_loss": rhythm_loss,
            "breath_loss": breath_loss,
            "total_loss": total_loss,
        }
