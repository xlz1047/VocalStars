import logging
from typing import Any

from ml_new.legacy_ml.pitch_detection.detector import extract_pitch_features
from ml_new.legacy_ml.rhythm_analysis.rhythm_detector import analyze_rhythm
from ml_new.legacy_ml.breath_analysis.detector import analyze_breath
from ml_new.legacy_ml.feature_extraction.features import extract_spectral_features
from ml_new.legacy_ml.coaching_engine.feedback import build_coaching_outline

logger = logging.getLogger(__name__)


def analyze_audio_file(audio_path: str) -> dict[str, Any]:
    """Run the core audio analysis pipeline and return a lightweight feature summary."""
    logger.info("Starting audio analysis pipeline for %s", audio_path)

    pitch = extract_pitch_features(audio_path)
    rhythm = analyze_rhythm(audio_path)
    breath = analyze_breath(audio_path)
    spectral = extract_spectral_features(audio_path)
    coaching = build_coaching_outline(pitch, rhythm, breath, spectral)

    return {
        "pitch": pitch,
        "rhythm": rhythm,
        "breath": breath,
        "spectral": spectral,
        "coaching_outline": coaching,
    }
