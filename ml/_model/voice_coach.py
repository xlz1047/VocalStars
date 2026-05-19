"""VoiceCoach multi-task model: backbone + pitch, rhythm, and breath heads."""

import torch
import torch.nn as nn

from ml._model.backbone import AudioEncoder
from ml.pitch_detection.model import PitchHead
from ml.rhythm_analysis.model import RhythmHead
from ml.breath_analysis.model import BreathHead


class VoiceCoachModel(nn.Module):
    """Multi-task vocal coaching model combining a shared backbone with three task heads.

    Processes log-mel spectrograms and simultaneously estimates pitch, detects note
    onsets, and classifies breath events from a single forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AudioEncoder()
        self.pitch_head = PitchHead()
        self.rhythm_head = RhythmHead()
        self.breath_head = BreathHead()

    def forward(self, mel_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full multi-task forward pass.

        Args:
            mel_tensor: Log-mel spectrogram of shape (batch, 1, 128, T).

        Returns:
            Dictionary with keys:
                pitch_logits: (batch, 360) raw pitch bin logits.
                pitch_hz:     (batch,) weighted-mean pitch frequency in Hz.
                onset_probs:  (batch, 1, T_out) per-frame onset probabilities.
                breath_prob:  (batch, 1) breath event probability.
                features:     (batch, 256, T_out) shared backbone embeddings.
        """
        features = self.encoder(mel_tensor)          # (batch, 256, T_out)
        pitch_logits = self.pitch_head(features)      # (batch, 360)
        onset_probs = self.rhythm_head(features)      # (batch, 1, T_out)
        breath_prob = self.breath_head(features)      # (batch, 1)

        probs = torch.sigmoid(pitch_logits)           # (batch, 360)
        bins = self.pitch_head.BINS_HZ.to(probs.device)
        top_vals, top_idx = torch.topk(probs, self.pitch_head.top_k, dim=-1)
        pitch_hz = (bins[top_idx] * top_vals).sum(dim=-1) / top_vals.sum(dim=-1).clamp(min=1e-8)

        return {
            "pitch_logits": pitch_logits,
            "pitch_hz": pitch_hz,
            "onset_probs": onset_probs,
            "breath_prob": breath_prob,
            "features": features,
        }

    def summary(self) -> None:
        """Print per-component and total parameter counts."""
        components = [
            ("encoder",     self.encoder),
            ("pitch_head",  self.pitch_head),
            ("rhythm_head", self.rhythm_head),
            ("breath_head", self.breath_head),
        ]
        total = 0
        for name, module in components:
            count = sum(p.numel() for p in module.parameters())
            total += count
            print(f"  {name:<15}: {count:>10,} params")
        print(f"  {'TOTAL':<15}: {total:>10,} params")
