"""VoiceCoach multi-task model: NanoPitchEncoder backbone + pitch, rhythm, and breath heads."""

import torch
import torch.nn as nn

from ml_new.legacy_ml._model.backbone import NanoPitchEncoder
from ml_new.legacy_ml.pitch_detection.model import PitchHead
from ml_new.legacy_ml.rhythm_analysis.model import RhythmHead
from ml_new.legacy_ml.breath_analysis.model import BreathHead


class VoiceCoachModel(nn.Module):
    """Multi-task vocal coaching model combining a shared backbone with three task heads.

    The backbone (``NanoPitchEncoder``) processes ``(B, T, 40)`` log-mel input and
    produces ``(B, T, 384)`` time-major embeddings.  Pitch and VAD heads consume
    these directly; rhythm and breath heads receive a channel-first permutation
    ``(B, 384, T)`` to match their Conv1d expectations.

    Pre-trained backbone and pitch/VAD weights can be loaded from a NanoPitch
    checkpoint via ``from_nanopitch_checkpoint``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = NanoPitchEncoder()
        self.pitch_head = PitchHead()
        self.rhythm_head = RhythmHead()
        self.breath_head = BreathHead()

    def forward(
        self,
        x: torch.Tensor,
        encoder_states: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full multi-task forward pass.

        Args:
            x: Log-mel spectrogram of shape ``(B, T, 40)`` — time-major.
            encoder_states: Optional list of three GRU hidden states for streaming.

        Returns:
            Dictionary with keys:
                pitch_logits:  ``(B, T, 360)`` raw pitch bin logits.
                vad_logits:    ``(B, T, 1)``   raw VAD logits.
                pitch_hz:      ``(B, T)``       sigmoid-weighted Hz estimate per frame.
                onset_probs:   ``(B, 1, T)``    per-frame onset probabilities.
                breath_prob:   ``(B, 1)``       breath event probability.
                features:      ``(B, T, 384)``  shared backbone embeddings.
                encoder_states: list of three updated GRU hidden states.
        """
        features, new_states = self.encoder(x, encoder_states)  # (B, T, 384)

        pitch_logits, vad_logits = self.pitch_head(features)    # (B,T,360), (B,T,1)
        pitch_hz = self.pitch_head.logits_to_hz(pitch_logits)   # (B, T)

        features_chan_first = features.permute(0, 2, 1)         # (B, 384, T)
        onset_probs = self.rhythm_head(features_chan_first)      # (B, 1, T)
        breath_prob = self.breath_head(features_chan_first)      # (B, 1)

        return {
            "pitch_logits": pitch_logits,
            "vad_logits": vad_logits,
            "pitch_hz": pitch_hz,
            "onset_probs": onset_probs,
            "breath_prob": breath_prob,
            "features": features,
            "encoder_states": new_states,
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

    @classmethod
    def from_nanopitch_checkpoint(cls, checkpoint_path: str) -> "VoiceCoachModel":
        """Build a model and initialise backbone + pitch heads from a NanoPitch checkpoint.

        Loads conv1, conv2, all three GRUs, dense_pitch, and dense_vad from the
        checkpoint.  Rhythm and breath heads are randomly initialised.

        Args:
            checkpoint_path: Path to a NanoPitch ``.pth`` checkpoint file.

        Returns:
            ``VoiceCoachModel`` with pre-trained pitch-aware weights.
        """
        model = cls()
        model.encoder = NanoPitchEncoder.from_nanopitch_checkpoint(checkpoint_path)
        model.pitch_head.load_nanopitch_weights(checkpoint_path)
        print("VoiceCoachModel: initialised from NanoPitch checkpoint.")
        return model
