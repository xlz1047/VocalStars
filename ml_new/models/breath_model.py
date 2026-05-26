"""Lightweight causal GRU breath detector for real-time streaming inference.

Architecture
------------
Input: ``vad_features (B, 3, T)`` — RMS energy, spectral flatness, ZCR.

These three handcrafted features are sufficient to distinguish:
- Voiced singing  (high RMS, low ZCR, low spectral flatness)
- Breath frames   (moderate RMS, high ZCR, moderate spectral flatness)
- Silence frames  (near-zero RMS)

The model is intentionally separate from the VAD model so that it can be
updated, exported, or replaced independently.

Output
------
``breath_prob: (B, T)`` — per-frame probability of breath in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BreathModel(nn.Module):
    """Causal two-layer GRU breath frame classifier.

    Args:
        n_vad_feats: Number of input features per frame (default 3).
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability applied between GRU layers (0 = off).
    """

    def __init__(
        self,
        n_vad_feats: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.norm = nn.LayerNorm(n_vad_feats)
        self.gru = nn.GRU(
            input_size=n_vad_feats,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        vad_features: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            vad_features: ``(B, 3, T)`` handcrafted acoustic features.
            h: Optional GRU hidden state ``(num_layers, B, hidden_size)``
                for streaming inference.  Pass ``None`` to start a new sequence.

        Returns:
            Tuple of:
            - ``breath_prob``: ``(B, T)`` per-frame breath probability.
            - ``h_new``: Updated GRU hidden state for the next chunk.
        """
        x = vad_features.permute(0, 2, 1)  # (B, T, 3)
        x = self.norm(x)
        out, h_new = self.gru(x, h)        # (B, T, hidden_size)
        logits = self.fc(out).squeeze(-1)  # (B, T)
        return torch.sigmoid(logits), h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zeroed initial hidden state for streaming."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
