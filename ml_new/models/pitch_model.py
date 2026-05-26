"""Causal pitch detection model using HCQT features.

Architecture
------------
The model exploits the key property of HCQT: for a voice singing at pitch bin b,
*all* harmonics contribute energy at exactly bin b in their respective channels.
The first stage fuses the 6 harmonic channels at each frequency bin independently,
producing a per-bin salience map.  A causal GRU then smooths this over time and
outputs per-frame pitch logits and a voiced probability.

Stages
~~~~~~
1. **Harmonic fusion** (per frame, per bin):
   ``(B*T, 6, 60)`` → Conv1d(6→32, k=1) → Conv1d(32→16, k=3) → Conv1d(16→1, k=1)
   → ``(B, T, 60)`` pitch salience map

2. **Temporal GRU** (causal):
   ``(B, T, 60)`` → GRU(60, 128, layers=2) → ``(B, T, 128)``

3. **Pitch head**: Linear(128→60) — logits for 60 pitch bins

4. **Voiced head**: Linear(128 + 3 VAD feats → 1) + Sigmoid

Bin encoding
~~~~~~~~~~~~
- fmin = 32.7 Hz (C1)
- Default: bins_per_octave=36, n_bins=180 (5 octaves, 33¢/bin)
- Legacy: bins_per_octave=12, n_bins=60 (5 octaves, 100¢/bin)
- bin b → frequency: fmin × 2^(b / bins_per_octave)
- frequency f → bin: bins_per_octave × log2(f / fmin)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchModel(nn.Module):
    """Causal HCQT-based pitch detection and voicing classifier.

    Args:
        n_harmonics: Number of HCQT harmonic channels (default 6).
        n_bins: Number of pitch / CQT bins (default 180 = 5 oct × 36 bins/oct).
        bins_per_octave: CQT frequency resolution (default 36 = 33¢/bin).
        n_vad_feats: Number of handcrafted VAD features (default 3).
        gru_hidden: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers.
    """

    FMIN: float = 32.7

    def __init__(
        self,
        n_harmonics: int = 6,
        n_bins: int = 180,
        bins_per_octave: int = 36,
        n_vad_feats: int = 3,
        gru_hidden: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers

        # Stage 1: harmonic fusion — operates on (B*T, n_harmonics, n_bins)
        # Lightweight 1×1 → 5-wide → 1×1 conv; k=5 gives ±2-bin (±66¢) context
        self.harmonic_conv = nn.Sequential(
            nn.Conv1d(n_harmonics, 16, kernel_size=1),          # per-bin harmonic fusion
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),         # ±2 bin local context
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),                     # salience per bin
        )

        # Stage 2: single-layer causal GRU (web-friendly: no dropout needed)
        self.norm = nn.LayerNorm(n_bins)
        self.gru = nn.GRU(
            input_size=n_bins,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        # Stage 3: output heads
        self.pitch_head = nn.Linear(gru_hidden, n_bins)
        self.voiced_head = nn.Linear(gru_hidden + n_vad_feats, 1)

        # Register bin-centre frequencies as a buffer (no gradient)
        bins = torch.arange(n_bins, dtype=torch.float32)
        self.register_buffer("bin_hz", self.FMIN * (2.0 ** (bins / bins_per_octave)))

    def forward(
        self,
        hcqt: torch.Tensor,
        vad_features: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hcqt: ``(B, 6, 60, T)`` log-magnitude HCQT.
            vad_features: ``(B, 3, T)`` handcrafted VAD features.
            h: Optional GRU hidden state ``(num_layers, B, gru_hidden)``
                for streaming.  Pass ``None`` to start a new sequence.

        Returns:
            Tuple of:
            - ``pitch_logits``: ``(B, T, 60)`` unnormalized pitch bin scores.
            - ``voiced_prob``: ``(B, T)`` voicing probability in [0, 1].
            - ``h_new``: Updated GRU hidden state for the next chunk.
        """
        B, n_harm, _, T = hcqt.shape

        # Stage 1: harmonic fusion per frame
        # (B, H, n_bins, T) → (B, T, H, n_bins) → (B*T, H, n_bins)
        x = hcqt.permute(0, 3, 1, 2).reshape(B * T, n_harm, self.n_bins)
        x = self.harmonic_conv(x)           # (B*T, 1, 60)
        x = x.squeeze(1).reshape(B, T, self.n_bins)   # (B, T, 60)

        # Stage 2: temporal GRU
        x = self.norm(x)
        gru_out, h_new = self.gru(x, h)    # (B, T, gru_hidden)

        # Stage 3: pitch head
        pitch_logits = self.pitch_head(gru_out)   # (B, T, 60)

        # Voiced head: GRU context + VAD features
        vad_t = vad_features.permute(0, 2, 1)         # (B, T, 3)
        voiced_in = torch.cat([gru_out, vad_t], dim=-1)  # (B, T, 131)
        voiced_prob = torch.sigmoid(self.voiced_head(voiced_in)).squeeze(-1)  # (B, T)

        return pitch_logits, voiced_prob, h_new

    def predict_hz(self, pitch_logits: torch.Tensor) -> torch.Tensor:
        """Convert pitch logits to Hz via argmax decoding.

        Args:
            pitch_logits: ``(B, T, 60)`` logits.

        Returns:
            ``(B, T)`` predicted frequency in Hz.
        """
        pred_bins = pitch_logits.argmax(dim=-1)   # (B, T)
        return self.bin_hz[pred_bins]             # (B, T)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zeroed initial hidden state for streaming."""
        return torch.zeros(self.num_layers, batch_size, self.gru_hidden, device=device)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
