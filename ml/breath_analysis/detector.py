"""Breath detector: combines extractor and model to produce per-frame breath predictions."""

import numpy as np
from typing import Any

from ml.feature_extraction.audio_utils import load_audio, frame_audio


class BreathDetector:
    """Detects breath events using a heuristic (low RMS + high ZCR) or a neural model.

    Heuristic rule: a frame is a breath event when its RMS energy is below
    ``rms_threshold`` AND its zero-crossing rate exceeds ``zcr_threshold``.
    Voiced frames that pass neither condition count toward the support score.

    Args:
        model: Optional PyTorch nn.Module breath classification head.
        sr: Sample rate the detector operates at.
        rms_threshold: RMS threshold below which a frame is a breath candidate.
        zcr_threshold: ZCR fraction a breath candidate must exceed.
    """

    def __init__(
        self,
        model=None,
        sr: int = 16000,
        rms_threshold: float = 0.02,
        zcr_threshold: float = 0.1,
    ) -> None:
        self._model = model
        self._sr = sr
        self._rms_threshold = rms_threshold
        self._zcr_threshold = zcr_threshold

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> dict[str, Any]:
        """Detect breath events and compute support metrics.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            sr: Sample rate of the audio.

        Returns:
            Dict with keys: breath_cycles, support_score, breath_length_variation.
        """
        hop_len = 512
        frames = frame_audio(audio, frame_len=2048, hop_len=hop_len)
        breath_flags = [self._is_breath_frame(f) for f in frames]
        return self._compute_metrics(breath_flags, hop_len=hop_len, sr=sr)

    def _is_breath_frame(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
        return rms < self._rms_threshold and zcr > self._zcr_threshold

    def _compute_metrics(
        self, breath_flags: list[bool], hop_len: int, sr: int
    ) -> dict[str, Any]:
        total = len(breath_flags)
        breath_count = sum(breath_flags)
        voiced_count = total - breath_count

        cycles = sum(
            1
            for i in range(1, len(breath_flags))
            if breath_flags[i] and not breath_flags[i - 1]
        )

        support_score = round(voiced_count / total, 4) if total > 0 else 0.0

        breath_lengths: list[float] = []
        run = 0
        for flag in breath_flags:
            if flag:
                run += 1
            elif run > 0:
                breath_lengths.append(run * hop_len / sr)
                run = 0
        if run > 0:
            breath_lengths.append(run * hop_len / sr)

        breath_length_variation = (
            round(float(np.std(breath_lengths)), 4) if len(breath_lengths) > 1 else 0.0
        )

        return {
            "breath_cycles": cycles,
            "support_score": support_score,
            "breath_length_variation": breath_length_variation,
        }


def analyze_breath(audio_path: str) -> dict[str, Any]:
    """Load audio and detect breath features. Module-level entry point for pipeline.py.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with keys: breath_cycles, support_score, breath_length_variation.
    """
    audio = load_audio(audio_path, sr=16000)
    detector = BreathDetector()
    return detector.analyze(audio, sr=16000)
