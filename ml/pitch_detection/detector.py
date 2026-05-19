"""Pitch detector: combines DSP extractor and model head to produce pitch features."""

import numpy as np
from typing import Any

from ml.feature_extraction.audio_utils import load_audio, frame_audio


class PitchDetector:
    """Extracts pitch features from audio using a neural head or librosa.yin fallback.

    When a trained PitchHead model is provided, routes audio through it.
    Otherwise uses librosa.yin for fundamental frequency tracking.

    Note: PitchHead expects (batch, 256, T) backbone embeddings. Until the shared
    backbone (ml/_model/backbone.py) is wired in, the model path falls back to
    librosa.yin automatically.

    Args:
        model: Optional PitchHead instance. If None, uses librosa.yin.
        sr: Sample rate the detector operates at.
    """

    def __init__(self, model=None, sr: int = 16000) -> None:
        self._model = model
        self._sr = sr

    def analyze(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict[str, Any]:
        """Extract pitch features from a single audio chunk.

        Args:
            audio_chunk: 1-D float32 numpy array of audio samples.
            sr: Sample rate of the audio chunk.

        Returns:
            Dict with keys: stability_score, estimated_notes, pitch_curve, note_transitions.
        """
        if self._model is not None:
            return self._analyze_with_model(audio_chunk, sr)
        return self._analyze_with_librosa(audio_chunk, sr)

    def _analyze_with_librosa(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        import librosa

        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced = f0[f0 > 0]

        if len(voiced) > 1:
            stability = float(np.clip(1.0 - np.std(voiced) / (np.mean(voiced) + 1e-8), 0.0, 1.0))
        else:
            stability = 0.0

        estimated_notes: list[str] = []
        note_transitions: list[tuple[str, str]] = []
        if len(voiced) > 0:
            try:
                notes = [librosa.hz_to_note(float(hz)) for hz in voiced]
                seen: dict[str, None] = {}
                for n in notes:
                    seen[n] = None
                estimated_notes = list(seen)
                note_transitions = [(a, b) for a, b in zip(notes[:-1], notes[1:]) if a != b]
            except Exception:
                pass

        return {
            "stability_score": stability,
            "estimated_notes": estimated_notes,
            "pitch_curve": f0.tolist(),
            "note_transitions": note_transitions,
        }

    def _analyze_with_model(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        # PitchHead requires backbone embeddings (batch, 256, T) — fall back to librosa
        # until ml/_model/backbone.py is integrated into this detector.
        return self._analyze_with_librosa(audio, sr)


def extract_pitch_features(audio_path: str) -> dict[str, Any]:
    """Load audio and extract pitch features. Module-level entry point for pipeline.py.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with keys: stability_score, estimated_notes, pitch_curve, note_transitions.
    """
    audio = load_audio(audio_path, sr=16000)
    detector = PitchDetector()
    return detector.analyze(audio, sr=16000)
