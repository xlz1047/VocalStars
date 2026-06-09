"""Breath detector: combines extractor and model to produce per-frame breath predictions."""

import numpy as np
import torch
from typing import Any

from ml_new.legacy_ml.breath_analysis.extractor import BreathExtractor
from ml_new.legacy_ml.breath_analysis.model import BreathHead
from ml_new.legacy_ml.feature_extraction.audio_utils import load_audio, frame_audio


class BreathDetector:
    """Detects breath events in vocal audio using a neural model or DSP heuristics.

    When ``model_path`` is provided, loads a trained ``BreathHead`` and uses it
    for classification. Otherwise operates in heuristic-only mode via
    ``BreathExtractor``.

    Note: ``BreathHead`` requires (batch, 256, T) backbone embeddings. Until the
    shared backbone (ml/_model/backbone.py) is wired in, ``analyze`` falls back
    to the DSP heuristic even when a model is loaded.

    Args:
        model_path: Path to a ``BreathHead`` state-dict checkpoint, or None for
            heuristic-only mode.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._extractor = BreathExtractor()
        self._model: BreathHead | None = None
        if model_path is not None:
            model = BreathHead()
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model = model

    def analyze(self, audio_chunk: np.ndarray, sr: int = 16000) -> tuple[bool, float]:
        """Classify a single audio chunk as breath or non-breath.

        Uses the loaded ``BreathHead`` model when available; falls back to the
        DSP heuristic until backbone embeddings are integrated.

        Args:
            audio_chunk: 1-D float32 numpy array of audio samples.
            sr: Sample rate of the audio in Hz.

        Returns:
            Tuple of (breath_detected, confidence) where confidence is in [0.0, 1.0].
        """
        features = self._extractor.extract_features(audio_chunk, sr=sr)
        label, confidence = self._extractor.classify_heuristic(features)
        return label == "breath", confidence

    def analyze_breath(self, audio_path: str) -> dict[str, Any]:
        """Load a full audio file and run per-frame breath analysis.

        Frames the audio with a 2048-sample window and 512-sample hop, classifies
        each frame, then aggregates into file-level breath metrics.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dict with keys:
                breath_detected (bool): True if any breath frame was found.
                breath_confidence (float): Mean confidence across breath frames.
                breath_cycles (int): Number of breath onset events (non-breath → breath).
                support_score (float): Fraction of frames classified as voiced (non-breath).
                breath_length_variation (float): Std dev of breath run durations in seconds.
        """
        hop_len = 512
        audio = load_audio(audio_path, sr=16000)
        frames = frame_audio(audio, frame_len=2048, hop_len=hop_len)

        results = [self.analyze(f, sr=16000) for f in frames]
        breath_flags = [r[0] for r in results]
        confidences = [r[1] for r in results]

        breath_count = sum(breath_flags)
        total = len(breath_flags)

        breath_detected = breath_count > 0
        breath_confidence = (
            float(np.mean([c for flag, c in zip(breath_flags, confidences) if flag]))
            if breath_count > 0
            else 0.0
        )

        cycles = sum(
            1
            for i in range(1, len(breath_flags))
            if breath_flags[i] and not breath_flags[i - 1]
        )

        voiced_count = total - breath_count
        support_score = round(voiced_count / total, 4) if total > 0 else 0.0

        breath_lengths: list[float] = []
        run = 0
        for flag in breath_flags:
            if flag:
                run += 1
            elif run > 0:
                breath_lengths.append(run * hop_len / 16000)
                run = 0
        if run > 0:
            breath_lengths.append(run * hop_len / 16000)

        breath_length_variation = (
            round(float(np.std(breath_lengths)), 4) if len(breath_lengths) > 1 else 0.0
        )

        return {
            "breath_detected": breath_detected,
            "breath_confidence": round(breath_confidence, 4),
            "breath_cycles": cycles,
            "support_score": support_score,
            "breath_length_variation": breath_length_variation,
        }


def analyze_breath(audio_path: str) -> dict[str, Any]:
    """Load audio and detect breath features. Module-level entry point for pipeline.py.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with keys: breath_detected, breath_confidence, breath_cycles,
        support_score, breath_length_variation.
    """
    return BreathDetector().analyze_breath(audio_path)
