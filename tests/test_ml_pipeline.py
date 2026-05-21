"""Pytest test suite for ML pipeline evaluators.

Run with: pytest tests/test_ml_pipeline.py -v
"""
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import pytest

from tests.evaluate_pitch import evaluate_pitch_on_synth
from tests.evaluate_pipeline import evaluate_rhythm, evaluate_breath, evaluate_spectral
from tests.visualization import plot_pitch_curve, plot_energy_envelope, plot_beat_alignment, plot_note_errors
from tests.synthetic_data import synth_melody, synth_microtonal_notes, synth_vibrato_sweep, synth_noisy_melody


class TestPitchDetection:
    """Test suite for pitch detection accuracy."""

    def test_pitch_pure_tones(self):
        """Test pitch detection on pure sine notes."""
        result = evaluate_pitch_on_synth()
        assert result["rmse_hz"] is not None
        assert result["voiced_true_positive_rate"] > 0.95
        assert result["note_level"]["note_rmse_hz"] is not None

    def test_pitch_note_level_accuracy(self):
        """Verify per-note median pitch accuracy is within 1 Hz."""
        result = evaluate_pitch_on_synth()
        for note in result["note_level"]["notes"]:
            if note.get("error_hz") is not None:
                assert note["error_hz"] < 1.0, f"Note error {note['error_hz']} exceeds tolerance"

    def test_pitch_with_microtonal_offsets(self):
        """Test pitch detection on microtonal notes (±20 cents)."""
        sr = 22050
        freqs = [220.0, 440.0, 329.63]
        cents_offsets = [-20, 0, 20]  # -20 cents, no offset, +20 cents

        y = synth_microtonal_notes(freqs, cents_offsets, duration=1.5, sr=sr)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)

        # Run pitch detection
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        times = librosa.times_like(f0, sr=sr)

        # Check that detection works
        assert np.any(~np.isnan(f0)), "Pitch detection failed on microtonal notes"

    def test_pitch_with_vibrato(self):
        """Test pitch detection robustness to vibrato."""
        sr = 22050
        y = synth_vibrato_sweep(freq=440.0, duration=2.0, sr=sr, vibrato_freq_start=4.0, vibrato_freq_end=6.0, vibrato_cents=50.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        assert np.any(~np.isnan(f0)), "Pitch detection failed on vibrato"


class TestRhythmDetection:
    """Test suite for rhythm and tempo detection."""

    def test_rhythm_tempo_accuracy(self):
        """Test tempo detection on synthetic metronome click train."""
        result = evaluate_rhythm()
        assert result["within_tolerance"] is True, f"Detected tempo {result['detected_tempo']} outside tolerance"
        assert result["detected_tempo"] is not None

    def test_rhythm_beat_timing_variance(self):
        """Verify beat timing variance is low on regular clicks."""
        result = evaluate_rhythm()
        assert result["raw"]["timing_variance"] < 0.01, "Beat timing too irregular"


class TestBreathDetection:
    """Test suite for breath cycle detection."""

    def test_breath_detection_accuracy(self):
        """Test breath detection on synthetic phrase."""
        result = evaluate_breath()
        assert result["within_tolerance"] is True, f"Detected breaths {result['detected_breaths']} outside tolerance"

    def test_breath_support_score(self):
        """Verify support score is computed."""
        result = evaluate_breath()
        assert result["raw"]["support_score"] is not None


class TestSpectralAnalysis:
    """Test suite for spectral feature extraction."""

    def test_spectral_centroid_accuracy(self):
        """Test spectral centroid on pure tone."""
        result = evaluate_spectral()
        assert result["within_tolerance"] is True, f"Centroid {result['centroid_mean']} outside tolerance"

    def test_spectral_mfcc_computation(self):
        """Verify MFCC features are computed."""
        result = evaluate_spectral()
        assert result["raw"]["mfcc"] is not None
        assert isinstance(result["raw"]["mfcc"], list)


class TestNoiseRobustness:
    """Test pipeline robustness to noise."""

    def test_pitch_with_white_noise(self):
        """Test pitch detection in white noise (SNR=10dB)."""
        sr = 22050
        freqs = [220.0, 440.0]
        durs = [1.5, 1.5]

        y = synth_noisy_melody(freqs, durs, sr=sr, snr_db=10.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        voiced_ratio = float(np.sum(~np.isnan(f0)) / max(1, len(f0)))
        assert voiced_ratio > 0.5, "Pitch detection failed in noisy signal"


class TestVisualization:
    """Test visualization utilities."""

    def test_plot_pitch_curve(self, tmp_path):
        """Verify pitch curve plotting works."""
        f0 = np.random.randn(100) * 20 + 440
        times = np.linspace(0, 2, 100)
        save_path = tmp_path / "pitch_test.png"
        plot_pitch_curve(f0, times, save_path=save_path)
        assert save_path.exists(), "Pitch plot not saved"

    def test_plot_energy_envelope(self, tmp_path):
        """Verify energy envelope plotting works."""
        energy = np.abs(np.random.randn(100)) + 0.5
        times = np.linspace(0, 2, 100)
        breaths = [0.5, 1.0, 1.5]
        save_path = tmp_path / "energy_test.png"
        plot_energy_envelope(energy, times, breaths, save_path=save_path)
        assert save_path.exists(), "Energy plot not saved"

    def test_plot_beat_alignment(self, tmp_path):
        """Verify beat alignment plotting works."""
        beat_times = [0.0, 0.6, 1.2, 1.8]
        ibi = np.array([0.6, 0.6, 0.6])
        save_path = tmp_path / "beats_test.png"
        plot_beat_alignment(beat_times, ibi, tempo=100.0, save_path=save_path)
        assert save_path.exists(), "Beat plot not saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
