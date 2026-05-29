"""ML inference service using ml_new models for VocalStars.

This service provides a wrapper around ml_new.coach_inference to analyze
audio recordings and provide coaching feedback using the unified vocal model.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import numpy as np

# Add ml_new to path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_new.inference.coach_inference import (
    analyse_recording,
    CoachingResult,
)

logger = logging.getLogger(__name__)


class MLInferenceService:
    """Service for running ml_new inference and returning coaching analysis."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        """Initialize the ML inference service.
        
        Args:
            checkpoint_path: Path to a trained model checkpoint. If None,
                uses fallback heuristics (librosa.pyin).
        """
        self.checkpoint_path = checkpoint_path
        self.device = "cpu"  # Can be set to "cuda" or "mps" for GPU

    def _debug_info(self) -> Dict[str, Any]:
        """Describe which inference path this service will use."""
        checkpoint_exists = (
            self.checkpoint_path is not None and self.checkpoint_path.exists()
        )
        return {
            "inference_mode": "checkpoint" if checkpoint_exists else "fallback",
            "checkpoint_path_used": str(self.checkpoint_path) if checkpoint_exists else None,
            "device_used": self.device,
            "model_stack_used": "ml_new",
        }

    def analyze_audio(
        self,
        audio_path: str | Path,
        song_title: str = "Unknown Song",
        artist: str = "Unknown Artist",
        task_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a recording and return coaching metrics.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            song_title: Title of the song being sung
            artist: Artist name
            
        Returns:
            Dictionary with coaching analysis including metrics and feedback.
        """
        debug = self._debug_info()
        logger.info(
            "ML inference mode=%s checkpoint_path_used=%s device=%s model_stack=%s",
            debug["inference_mode"],
            debug["checkpoint_path_used"],
            debug["device_used"],
            debug["model_stack_used"],
        )
        try:
            # Run the coaching inference
            result: CoachingResult = analyse_recording(
                audio_path,
                checkpoint=self.checkpoint_path,
                device=self.device,
                task_config=task_config,
            )

            # Convert result to JSON-serializable format
            coaching_data = self._format_coaching_result(result)
            coaching_data["songTitle"] = song_title
            coaching_data["artist"] = artist

            return {
                "status": "success",
                "data": coaching_data,
                "debug": debug,
            }

        except Exception as e:
            logger.exception(
                "ML inference failed mode=%s checkpoint_path_used=%s device=%s model_stack=%s",
                debug["inference_mode"],
                debug["checkpoint_path_used"],
                debug["device_used"],
                debug["model_stack_used"],
            )
            return {
                "status": "error",
                "error": str(e),
                "data": None,
                "debug": debug,
            }

    def _format_coaching_result(self, result: CoachingResult) -> Dict[str, Any]:
        """Convert CoachingResult to frontend-compatible format."""
        return {
            "score": result.score,
            "fullSongScore": result.full_song_score,
            "diagnosticScore": result.diagnostic_score,
            "scoreStatus": result.score_status,
            "scoreCaveat": result.score_caveat,
            "summary": result.summary,
            "issues": result.issues,
            "exercises": result.exercises,
            # Pitch metrics
            "pitchAccuracy": round(result.pitch_accuracy * 100, 1),
            "pitchDrift": round(result.pitch_drift_cents, 2),
            "phraseLengths": [round(p, 2) for p in result.phrase_lengths_s],
            # Breath and timing
            "breathCount": result.breath_count,
            "onsetCount": result.onset_count,
            "onsetClarity": round(result.onset_clarity, 2),
            # Technique classification
            "technique": result.technique,
            "techniqueConfidence": round(result.technique_confidence * 100, 1),
            "allTechniqueScores": {
                k: round(v * 100, 1) for k, v in result.all_technique_scores.items()
            },
            # Notes and voice quality
            "notes": [self._format_note(n) for n in result.notes],
            "voiceQuality": (
                self._format_voice_quality(result.voice_quality)
                if result.voice_quality
                else None
            ),
            "vibrato": result.vibrato_stats,
            "diagnostics": result.diagnostics,
            "analysisValidity": result.analysis_validity,
            "taskConfig": result.task_config,
            "taskAnalysis": result.task_analysis,
            # Frame-level data (sampled for efficiency)
            "frameData": {
                "pitch": self._downsample(result.pitch_hz, 10).tolist(),
                "voiced": self._downsample(
                    result.voiced.astype(float), 10
                ).tolist(),
                "breath": self._downsample(
                    result.breath_frames.astype(float), 10
                ).tolist(),
                "onset": self._downsample(
                    result.onset_frames.astype(float), 10
                ).tolist(),
                "hopLength": result.hop_s * 10,  # adjusted for downsampling
            },
        }

    def _format_note(self, note) -> Dict[str, Any]:
        """Format a NoteSegment for the frontend."""
        return {
            "startSeconds": round(note.start_s, 2),
            "durationSeconds": round(note.duration_s, 2),
            "pitchHz": round(note.pitch_hz, 1),
            "noteName": note.note_name,
            "centsError": round(note.cents_error, 1),
            "stabilityCents": round(note.stability_cents, 1),
            "vibrato": (
                {
                    "rateHz": round(note.vibrato.rate_hz, 1),
                    "depthCents": round(note.vibrato.depth_cents, 1),
                    "regularity": round(note.vibrato.regularity, 2),
                }
                if note.vibrato
                else None
            ),
        }

    def _format_voice_quality(self, vq) -> Dict[str, Any]:
        """Format VoiceQuality metrics for the frontend."""
        return {
            "hnrDb": round(vq.hnr_db, 1),
            "jitterPercent": round(vq.jitter_pct, 2),
            "shimmerPercent": round(vq.shimmer_pct, 2),
            "breathiness": vq.breathiness,
            "isUnstable": vq.is_unstable,
        }

    @staticmethod
    def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
        """Downsample array by taking every nth element."""
        return arr[::factor]


def get_ml_service(checkpoint_path: Optional[Path] = None) -> MLInferenceService:
    """Get an ML inference service instance."""
    return MLInferenceService(checkpoint_path)
