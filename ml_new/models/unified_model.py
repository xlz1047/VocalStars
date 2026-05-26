"""Unified multi-task singing voice analysis model.

One shared HCQT backbone feeds five task heads in a single forward pass:

  1. Pitch       — per-frame pitch bin logits (180 bins, 33 ¢/bin)
  2. Voiced/VAD  — per-frame voicing probability
  3. Breath      — per-frame breath probability
  4. Onset       — per-frame note-onset probability
  5. Technique   — clip-level singing technique classification (20 classes)

Architecture
------------
Backbone
  HCQT (B, 6, 180, T)
      → harmonic fusion  Conv1d 6→16→8→1 (per frequency bin, per frame)
      → LayerNorm
      → 2-layer causal GRU  (180 → 128 hidden)
      → gru_out  (B, T, 128)

Task heads (all causal — only use gru_out[:, :t, :])
  Pitch head:      Linear(128 → 180)
  Voiced head:     Linear(128 + 3 VAD feats → 1) + sigmoid
  Breath head:     Linear(128 + 3 VAD feats → 1) + sigmoid  [VAD features are
                   direct breath indicators: RMS, flatness, ZCR]
  Onset head:      Linear(128 → 64) → ReLU → Linear(64 → 1) + sigmoid
  Technique head:  temporal mean-pool → Linear(128 → 64) → ReLU →
                   Dropout(0.3) → Linear(64 → 20)

Technique vocabulary (20 classes)
----------------------------------
Shared across GTSinger and VocalSet; breathy/vibrato appear in both datasets.
See TECHNIQUE_VOCAB for the canonical index → name mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Canonical technique vocabulary — index order is fixed for checkpoint compat.
# ---------------------------------------------------------------------------
TECHNIQUE_VOCAB: list[str] = [
    "belt",                     # 0
    "breathy",                  # 1  (both GTSinger + VocalSet)
    "fast_forte",               # 2
    "fast_piano",               # 3
    "forte",                    # 4
    "glissando",                # 5  (GTSinger)
    "inhaled",                  # 6
    "lip_trill",                # 7
    "messa",                    # 8
    "mixed_voice_and_falsetto", # 9  (GTSinger)
    "pharyngeal",               # 10 (GTSinger)
    "pp",                       # 11
    "slow_forte",               # 12
    "slow_piano",               # 13
    "spoken",                   # 14
    "straight",                 # 15
    "trill",                    # 16
    "trillo",                   # 17
    "vibrato",                  # 18 (both GTSinger + VocalSet)
    "vocal_fry",                # 19
]
TECHNIQUE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TECHNIQUE_VOCAB)}
N_TECHNIQUES = len(TECHNIQUE_VOCAB)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UnifiedVocalModel(nn.Module):
    """Multi-task singing voice model with shared HCQT backbone.

    Args:
        n_harmonics: HCQT harmonic channels (default 6).
        n_bins: Pitch / CQT bins (default 180 = 5 oct × 36 bins/oct).
        bins_per_octave: CQT resolution (default 36 = 33 ¢/bin).
        n_vad_feats: Handcrafted VAD feature dimension (default 3).
        gru_hidden: GRU hidden size per layer (default 128).
        num_gru_layers: Number of stacked causal GRU layers (default 2).
        n_techniques: Technique vocabulary size (default 20).
        dropout: Dropout applied inside the technique head (default 0.3).
    """

    FMIN: float = 32.7  # C1 in Hz

    def __init__(
        self,
        n_harmonics: int = 6,
        n_bins: int = 180,
        bins_per_octave: int = 36,
        n_vad_feats: int = 3,
        gru_hidden: int = 128,
        num_gru_layers: int = 2,
        n_techniques: int = N_TECHNIQUES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers
        self.n_vad_feats = n_vad_feats

        # ── Backbone: harmonic fusion ────────────────────────────────────────
        # Applied per-frame per-frequency bin: (B*T, n_harmonics, n_bins)
        self.harmonic_conv = nn.Sequential(
            nn.Conv1d(n_harmonics, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),  # ±2-bin local context
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )

        # ── Backbone: 2-layer causal GRU ────────────────────────────────────
        self.input_norm = nn.LayerNorm(n_bins)
        self.gru = nn.GRU(
            input_size=n_bins,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # ── Task heads ───────────────────────────────────────────────────────
        self.pitch_head = nn.Linear(gru_hidden, n_bins)

        # Voiced + breath heads get VAD features appended
        self.voiced_head = nn.Linear(gru_hidden + n_vad_feats, 1)
        self.breath_head = nn.Linear(gru_hidden + n_vad_feats, 1)

        # Onset head: two-layer MLP for more discriminative onset detection
        self.onset_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Technique head: mean-pool over time → MLP
        self.technique_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_techniques),
        )

        # Bin-centre frequencies buffer (no gradient)
        bins = torch.arange(n_bins, dtype=torch.float32)
        self.register_buffer("bin_hz", self.FMIN * (2.0 ** (bins / bins_per_octave)))

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        hcqt: torch.Tensor,
        vad_features: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,  # pitch_logits   (B, T, n_bins)
        torch.Tensor,  # voiced_prob    (B, T)
        torch.Tensor,  # breath_prob    (B, T)
        torch.Tensor,  # onset_prob     (B, T)
        torch.Tensor,  # tech_logits    (B, n_techniques)
        torch.Tensor,  # h_new          (num_layers, B, gru_hidden)
    ]:
        """Run all five task heads from a single backbone forward pass.

        Args:
            hcqt: ``(B, 6, n_bins, T)`` log-magnitude HCQT.
            vad_features: ``(B, 3, T)`` handcrafted features
                (RMS energy, spectral flatness, ZCR).
            h: Optional GRU hidden state for streaming inference.

        Returns:
            pitch_logits, voiced_prob, breath_prob, onset_prob,
            technique_logits, h_new
        """
        B, n_harm, _, T = hcqt.shape

        # ── Harmonic fusion ─────────────────────────────────────────────────
        # Reshape to (B*T, n_harmonics, n_bins), apply conv, restore shape
        x = hcqt.permute(0, 3, 1, 2).reshape(B * T, n_harm, self.n_bins)
        x = self.harmonic_conv(x)                          # (B*T, 1, n_bins)
        x = x.squeeze(1).reshape(B, T, self.n_bins)       # (B, T, n_bins)

        # ── 2-layer causal GRU ──────────────────────────────────────────────
        x = self.input_norm(x)
        gru_out, h_new = self.gru(x, h)                   # (B, T, gru_hidden)

        # VAD features reshaped for concatenation: (B, T, 3)
        vad_t = vad_features.permute(0, 2, 1)

        # ── Pitch head ──────────────────────────────────────────────────────
        pitch_logits = self.pitch_head(gru_out)            # (B, T, n_bins)

        # ── Voiced head ─────────────────────────────────────────────────────
        voiced_in = torch.cat([gru_out, vad_t], dim=-1)   # (B, T, 131)
        voiced_prob = torch.sigmoid(
            self.voiced_head(voiced_in)
        ).squeeze(-1)                                      # (B, T)

        # ── Breath head ─────────────────────────────────────────────────────
        breath_prob = torch.sigmoid(
            self.breath_head(voiced_in)                    # reuse same cat
        ).squeeze(-1)                                      # (B, T)

        # ── Onset head ──────────────────────────────────────────────────────
        onset_prob = torch.sigmoid(
            self.onset_head(gru_out)
        ).squeeze(-1)                                      # (B, T)

        # ── Technique head (clip-level via mean pooling) ─────────────────────
        clip_repr = gru_out.mean(dim=1)                    # (B, gru_hidden)
        tech_logits = self.technique_head(clip_repr)       # (B, n_techniques)

        return pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits, h_new

    # ── Convenience helpers ─────────────────────────────────────────────────

    def encode_clip(
        self,
        hcqt: torch.Tensor,
        vad_features: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean-pooled backbone representation without running task heads.

        Args:
            hcqt: ``(B, 6, n_bins, T)``
            vad_features: ``(B, 3, T)``
            h: optional GRU hidden state.

        Returns:
            ``(clip_repr, h_new)`` where ``clip_repr`` is ``(B, gru_hidden)``.
        """
        B, n_harm, _, T = hcqt.shape
        x = hcqt.permute(0, 3, 1, 2).reshape(B * T, n_harm, self.n_bins)
        x = self.harmonic_conv(x).squeeze(1).reshape(B, T, self.n_bins)
        x = self.input_norm(x)
        gru_out, h_new = self.gru(x, h)
        return gru_out.mean(dim=1), h_new

    def predict_hz(self, pitch_logits: torch.Tensor) -> torch.Tensor:
        """Argmax decode pitch logits to Hz."""
        return self.bin_hz[pitch_logits.argmax(dim=-1)]

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zeroed initial hidden state for streaming."""
        return torch.zeros(
            self.num_gru_layers, batch_size, self.gru_hidden, device=device
        )

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
