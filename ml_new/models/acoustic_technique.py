"""Acoustic-feature-augmented technique classifier for VocalStars.

Problem with the original technique head
-----------------------------------------
The shared backbone was trained primarily for pitch / VAD / breath / onset.
Its mean-pooled clip representation carries limited technique-specific signal,
so the frozen-backbone fine-tune plateaued at ~46 % accuracy.

Solution
--------
Append 10 clip-level acoustic features to the backbone's clip representation
before the technique MLP.  These features are derived directly from the NPZ
arrays that are already stored for every clip — no audio re-read needed at
training time, and trivially computed from model outputs at inference time.

Acoustic feature vector (10-D, all from NPZ)
----------------------------------------------
  0  rms_mean          — overall energy level
  1  rms_std           — energy variation (belting is more stable than breathy)
  2  flatness_mean     — spectral flatness (high → noisy/breathy)
  3  flatness_std      — flatness variation
  4  zcr_mean          — zero-crossing rate (high for fricative/breathy noise)
  5  zcr_std           — ZCR variation
  6  voiced_ratio      — fraction of voiced frames
  7  f0_mean_norm      — mean F0 in cents above C1, divided by 2400
  8  f0_std_norm       — F0 std in cents, divided by 100
  9  f0_range_norm     — F0 peak-to-peak range in cents, divided by 100

These capture breathiness (flatness, ZCR), energy profile (RMS), voicing
amount (voiced_ratio), and pitch behaviour (vibrato ↔ high F0 std, narrow
range ↔ monotone technique like vocal_fry / spoken).

Normalisation
-------------
Running mean / std are computed from the training split before training and
stored as module buffers so they are saved inside the checkpoint and
automatically restored at inference without any external state file.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ml_new.models.unified_model import N_TECHNIQUES

N_ACOUSTIC = 10   # dimension of the acoustic feature vector
FMIN_C1 = 32.7    # Hz — lowest CQT bin (C1)


# ---------------------------------------------------------------------------
# Feature extraction helper (used at both training and inference time)
# ---------------------------------------------------------------------------

def extract_acoustic_features(
    vad_features: np.ndarray,  # (3, T) — [rms, flatness, zcr]
    f0_hz:        np.ndarray,  # (T,)
    vad:          np.ndarray,  # (T,) binary voiced label (or voiced_bool array)
) -> np.ndarray:
    """Compute the 10-D clip-level acoustic feature vector from NPZ arrays.

    All inputs are already available in every NPZ file produced by
    ``extract_all.py``, so no audio re-read is required.

    Args:
        vad_features: Per-frame [RMS, spectral flatness, ZCR].
        f0_hz: Per-frame F0 in Hz (0 = unvoiced).
        vad: Binary voiced labels (1 = voiced, 0 = unvoiced / silence).

    Returns:
        ``(10,)`` float32 array.
    """
    rms      = vad_features[0]
    flatness = vad_features[1]
    zcr      = vad_features[2]

    voiced = (vad > 0) & (f0_hz > 0)
    voiced_ratio = float(vad.mean())

    f0_voiced = f0_hz[voiced]
    if len(f0_voiced) >= 5:
        cents    = 1200.0 * np.log2(f0_voiced / FMIN_C1)
        f0_mean  = float(cents.mean()) / 2400.0        # roughly 0–1 over vocal range
        f0_std   = float(cents.std())  / 100.0
        f0_range = float(cents.max() - cents.min()) / 100.0
    else:
        f0_mean = f0_std = f0_range = 0.0

    return np.array([
        float(rms.mean()),
        float(rms.std()),
        float(flatness.mean()),
        float(flatness.std()),
        float(zcr.mean()),
        float(zcr.std()),
        voiced_ratio,
        f0_mean,
        f0_std,
        f0_range,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Classifier module
# ---------------------------------------------------------------------------

class AcousticTechniqueClassifier(nn.Module):
    """Technique MLP that fuses backbone clip representation + acoustic features.

    Args:
        gru_dim:     Dimension of the backbone's mean-pooled GRU output (128).
        n_acoustic:  Acoustic feature dimension (10).
        hidden:      Hidden units in the two-layer MLP.
        n_techniques: Output classes (20).
        dropout:     Dropout rate inside the MLP.
    """

    def __init__(
        self,
        gru_dim:      int = 128,
        n_acoustic:   int = N_ACOUSTIC,
        hidden:       int = 128,
        n_techniques: int = N_TECHNIQUES,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = gru_dim + n_acoustic
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_techniques),
        )
        # Normalisation statistics — stored as buffers so they live in the
        # state_dict and are restored from the checkpoint automatically.
        self.register_buffer("feat_mean", torch.zeros(n_acoustic))
        self.register_buffer("feat_std",  torch.ones(n_acoustic))

    def forward(
        self,
        clip_repr:     torch.Tensor,   # (B, gru_dim)
        acoustic_feats: torch.Tensor,  # (B, n_acoustic)  — raw, unnormalised
    ) -> torch.Tensor:
        """Return technique logits ``(B, n_techniques)``."""
        norm = (acoustic_feats - self.feat_mean) / (self.feat_std + 1e-8)
        x = torch.cat([clip_repr, norm], dim=-1)
        return self.net(x)

    def set_normalisation(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set normalisation stats computed from the training split."""
        self.feat_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.feat_std.copy_(torch.from_numpy(std.astype(np.float32)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
