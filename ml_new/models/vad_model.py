"""Lightweight causal VAD model for real-time streaming inference.

Architecture
------------
The model has two input streams:

1. **HCQT stream** ``(B, 6, 60, T)``
   Mean-pooled across the frequency axis → ``(B, 6, T)``.
   The 6 remaining values represent the mean log-magnitude per harmonic
   per frame.  Voiced speech has strong, coherent harmonics; silence and
   unvoiced fricatives do not.

2. **Handcrafted VAD stream** ``(B, 3, T)``
   RMS energy, spectral flatness, and zero-crossing rate — classic
   voiced/unvoiced discriminants.

Both streams are concatenated → ``(B, 9, T)``, then fed to a two-layer
causal GRU (no lookahead) followed by a linear classifier.

Output
------
``vad_prob: (B, T)`` — per-frame probability of voicing in [0, 1].

Streaming
---------
Hidden states are returned so the caller can pass them back on the next
chunk, enabling zero-latency frame-by-frame inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VADModel(nn.Module):
    """Causal two-layer GRU VAD classifier.

    Args:
        n_harmonics: Number of HCQT harmonic channels (default 6).
        n_vad_feats: Number of handcrafted VAD features per frame (default 3).
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability applied between GRU layers (0 = off).
    """

    def __init__(
        self,
        n_harmonics: int = 6,
        n_vad_feats: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        input_size = n_harmonics + n_vad_feats  # 9

        self.norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hcqt: torch.Tensor,
        vad_features: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hcqt: ``(B, 6, 60, T)`` log-magnitude HCQT.
            vad_features: ``(B, 3, T)`` handcrafted features.
            h: Optional GRU hidden state ``(num_layers, B, hidden_size)``
                for streaming inference.  Pass ``None`` to start a new
                sequence.

        Returns:
            Tuple of:
            - ``vad_prob``: ``(B, T)`` per-frame voicing probability.
            - ``h_new``: Updated GRU hidden state for the next chunk.
        """
        # Compress HCQT: mean over freq bins → (B, 6, T)
        hcqt_mean = hcqt.mean(dim=2)  # (B, n_harmonics, T)

        # Concatenate feature streams → (B, 9, T) → (B, T, 9)
        x = torch.cat([hcqt_mean, vad_features], dim=1).permute(0, 2, 1)

        x = self.norm(x)
        out, h_new = self.gru(x, h)            # (B, T, hidden_size)
        logits = self.fc(out).squeeze(-1)       # (B, T)
        return torch.sigmoid(logits), h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zeroed initial hidden state for streaming."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
