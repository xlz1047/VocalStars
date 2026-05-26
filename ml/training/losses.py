"""Multi-task loss functions: pitch, rhythm, breath, and VAD."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.5,
    pos_weight: float = 2.3,
) -> torch.Tensor:
    """Focal binary cross-entropy loss.

    Down-weights easy (high-confidence) frames and up-weights voiced frames to
    counteract class imbalance.  gamma=0 reduces to weighted BCE.

    Args:
        pred: Sigmoid probabilities, any shape.
        target: Binary targets, same shape.
        gamma: Focal modulation exponent.
        pos_weight: Weight applied to positive (voiced) samples.

    Returns:
        Scalar loss tensor.
    """
    pred = pred.clamp(1e-7, 1.0 - 1e-7)
    bce_per_elem = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    weight = torch.where(
        target == 1,
        torch.full_like(pt, pos_weight),
        torch.ones_like(pt),
    )
    return (weight * (1 - pt) ** gamma * bce_per_elem).mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of per-task losses for pitch, rhythm, breath, and VAD prediction.

    Pitch and VAD targets are per-frame ``(B, T, 360)`` / ``(B, T, 1)``.  When
    target and prediction T dimensions differ, both are truncated to the shorter
    length so the loss degrades gracefully under variable-length batches.

    Args:
        pitch_w:  Weight for pitch BCE loss.
        rhythm_w: Weight for rhythm/onset BCE loss.
        breath_w: Weight for breath BCE loss.
        vad_w:    Weight for VAD focal BCE loss.
    """

    def __init__(
        self,
        pitch_w: float = 1.0,
        rhythm_w: float = 0.5,
        breath_w: float = 0.3,
        vad_w: float = 0.1,
    ) -> None:
        super().__init__()
        self.pitch_w = pitch_w
        self.rhythm_w = rhythm_w
        self.breath_w = breath_w
        self.vad_w = vad_w

        self._pitch_loss_fn = nn.BCEWithLogitsLoss()
        self._breath_loss_fn = nn.BCELoss()
        self._onset_pos_weight = 10.0

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute per-task losses and their weighted sum.

        Args:
            predictions: Dict with keys:
                ``pitch_logits`` — ``(B, T, 360)`` raw logits.
                ``onset_probs``  — ``(B, 1, T)`` or ``(B, T)`` sigmoid probs.
                ``breath_prob``  — ``(B, 1)`` or ``(B,)`` sigmoid probability.
                ``vad_logits``   — ``(B, T, 1)`` raw logits (optional).
            targets: Dict with keys:
                ``pitch_bins``     — ``(B, T, 360)`` soft Gaussian targets.
                ``onset_targets``  — ``(B, 1, T)`` or ``(B, T)`` binary targets.
                ``breath_target``  — ``(B,)`` or ``(B, 1)`` binary targets.
                ``vad_target``     — ``(B, T, 1)`` binary targets (optional).

        Returns:
            Dict with scalar tensors: ``pitch_loss``, ``rhythm_loss``,
            ``breath_loss``, ``vad_loss``, ``total_loss``.
        """
        device = predictions["pitch_logits"].device

        # ── pitch loss ────────────────────────────────────────────────────────
        pitch_logits: torch.Tensor = predictions["pitch_logits"]   # (B, T, 360)
        pitch_bins: torch.Tensor   = targets["pitch_bins"]          # (B, T', 360)
        if pitch_logits.dim() == 3 and pitch_bins.dim() == 3:
            T_p = min(pitch_logits.shape[1], pitch_bins.shape[1])
            pitch_loss = self._pitch_loss_fn(
                pitch_logits[:, :T_p], pitch_bins[:, :T_p].float()
            )
        else:
            pitch_loss = self._pitch_loss_fn(pitch_logits, pitch_bins.float())

        # ── rhythm/onset loss ─────────────────────────────────────────────────
        onset_probs: torch.Tensor   = predictions["onset_probs"]
        onset_targets: torch.Tensor = targets["onset_targets"]
        onset_p = onset_probs.reshape(onset_probs.size(0), -1)       # (B, T)
        onset_t = onset_targets.reshape(onset_targets.size(0), -1).float()
        T_o = min(onset_p.shape[-1], onset_t.shape[-1])
        onset_p = onset_p[:, :T_o]
        onset_t = onset_t[:, :T_o]

        onset_weights = torch.ones_like(onset_t)
        onset_weights[onset_t > 0.5] = self._onset_pos_weight
        rhythm_loss = (
            onset_weights
            * F.binary_cross_entropy(onset_p, onset_t, reduction="none")
        ).mean()

        # ── breath loss ───────────────────────────────────────────────────────
        breath_prob: torch.Tensor   = predictions["breath_prob"].view(-1)
        breath_target: torch.Tensor = targets["breath_target"].view(-1).float()
        breath_loss = self._breath_loss_fn(breath_prob, breath_target)

        # ── VAD loss (optional) ───────────────────────────────────────────────
        vad_loss = torch.tensor(0.0, device=device)
        if "vad_logits" in predictions and "vad_target" in targets:
            vad_logits: torch.Tensor = predictions["vad_logits"]   # (B, T, 1)
            vad_target: torch.Tensor = targets["vad_target"]        # (B, T', 1)
            T_v = min(vad_logits.shape[1], vad_target.shape[1])
            vad_prob = torch.sigmoid(vad_logits[:, :T_v])
            vad_loss = focal_bce(vad_prob, vad_target[:, :T_v].float())

        total_loss = (
            self.pitch_w  * pitch_loss
            + self.rhythm_w * rhythm_loss
            + self.breath_w * breath_loss
            + self.vad_w    * vad_loss
        )

        return {
            "pitch_loss":  pitch_loss,
            "rhythm_loss": rhythm_loss,
            "breath_loss": breath_loss,
            "vad_loss":    vad_loss,
            "total_loss":  total_loss,
        }
