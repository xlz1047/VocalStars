"""Lightweight note onset detector using harmonic CQT features.

Architecture
------------
Input: ``hcqt_h0 (B, 180, T)`` — first harmonic of the HCQT (pitch salience map).

Using only the first harmonic keeps the model lightweight while retaining the
frequency resolution needed to detect pitch-onset events.

Pipeline:
1. ``Linear(180 → 32)`` per-frame frequency projection (applied as Conv1d with kernel=1).
2. ``LayerNorm(32)``
3. Single-layer causal ``GRU(32, 64)``
4. ``Linear(64 → 1) + Sigmoid``

Output
------
``onset_prob: (B, T)`` — per-frame probability of a note onset in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OnsetModel(nn.Module):
    """Causal onset detector operating on the first HCQT harmonic.

    Args:
        n_bins: Number of CQT frequency bins in the first harmonic (default 180).
        proj_size: Projected frequency dimension after the initial linear layer.
        hidden_size: GRU hidden dimension.
        dropout: Dropout on GRU output before the classifier head.
    """

    def __init__(
        self,
        n_bins: int = 180,
        proj_size: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Project 180 frequency bins → proj_size per frame (kernel=1 = per-frame linear)
        self.freq_proj = nn.Conv1d(n_bins, proj_size, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(proj_size)
        self.gru = nn.GRU(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hcqt_h0: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hcqt_h0: ``(B, 180, T)`` first harmonic of the HCQT.
            h: Optional GRU hidden state ``(1, B, hidden_size)`` for streaming.

        Returns:
            Tuple of:
            - ``onset_prob``: ``(B, T)`` per-frame onset probability.
            - ``h_new``: Updated GRU hidden state.
        """
        x = self.freq_proj(hcqt_h0)        # (B, proj_size, T)
        x = x.permute(0, 2, 1)             # (B, T, proj_size)
        x = self.norm(x)
        out, h_new = self.gru(x, h)        # (B, T, hidden_size)
        out = self.drop(out)
        logits = self.fc(out).squeeze(-1)  # (B, T)
        return torch.sigmoid(logits), h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zeroed initial hidden state for streaming."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
