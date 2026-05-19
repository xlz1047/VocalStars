"""PyTorch nn.Module head for pitch estimation outputting pitch_hz and intonation_cents."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_N_BINS = 360
# 360 bins, each 20 cents wide, starting at C1 (32.7 Hz) — CREPE-style log spacing
_BINS_HZ: torch.Tensor = 32.7 * (
    2 ** (torch.arange(_N_BINS, dtype=torch.float32) * 20.0 / 1200.0)
)


class PitchHead(nn.Module):
    """Pitch classification head mapping backbone features to pitch bin logits.

    Accepts frame-level embeddings of shape (batch, 256, T) from the shared backbone
    and outputs (batch, 360) raw logits, one per pitch bin.  Bins are spaced 20 cents
    apart starting at 32.7 Hz (C1), trained with per-bin binary cross-entropy so
    adjacent bins can co-activate around the true pitch.

    Args:
        top_k: Number of top-probability bins used by ``logits_to_hz`` for the
            weighted-mean frequency estimate.
    """

    N_BINS: int = _N_BINS
    BINS_HZ: torch.Tensor = _BINS_HZ

    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k

        self.conv = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        # Projects the mean-pooled query before dot-product attention
        self.attn_proj = nn.Linear(256, 256, bias=False)
        self.fc = nn.Linear(256, self.N_BINS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map backbone features to pitch bin logits.

        Args:
            x: Float tensor of shape (batch, 256, T).

        Returns:
            Raw logit tensor of shape (batch, 360).
        """
        x = F.relu(self.conv(x))  # (batch, 256, T)

        # Mean-pool over T, project, then attend back over T
        query = self.attn_proj(x.mean(dim=-1))  # (batch, 256)
        attn_scores = torch.bmm(query.unsqueeze(1), x).squeeze(1)  # (batch, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, T)

        context = (x * attn_weights.unsqueeze(1)).sum(dim=-1)  # (batch, 256)
        return self.fc(context)  # (batch, 360)

    def logits_to_hz(self, logits: torch.Tensor) -> float:
        """Convert a single frame's pitch logits to a frequency estimate in Hz.

        Applies sigmoid to obtain per-bin probabilities, selects the ``top_k``
        highest-activated bins, and returns their frequency-weighted mean.

        Args:
            logits: Tensor of shape (360,) or (1, 360).

        Returns:
            Estimated fundamental frequency in Hz.
        """
        probs = torch.sigmoid(logits.view(-1))  # (360,)
        k = min(self.top_k, self.N_BINS)
        top_vals, top_idx = torch.topk(probs, k)
        bins = self.BINS_HZ.to(logits.device)
        weighted_hz = (bins[top_idx] * top_vals).sum() / top_vals.sum().clamp(min=1e-8)
        return weighted_hz.item()

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"{self.__class__.__name__}("
            f"n_bins={self.N_BINS}, "
            f"top_k={self.top_k}, "
            f"params={n_params:,})"
        )
