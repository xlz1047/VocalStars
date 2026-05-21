# ML Pipeline Evaluation and Testing

This document describes the synthetic evaluation framework, test metrics, and interpretation guidelines for VocalStars' audio analysis pipeline.

## Overview

The test suite validates core ML modules (pitch detection, rhythm analysis, breath detection, spectral features) using synthetic audio signals with known ground truth. This enables:

- **Objective accuracy measurement** without requiring labeled recordings
- **Robustness testing** against vibrato, microtonality, and noise
- **Visualization** of analysis outputs (plots saved to `tests/results/`)
- **CI/CD automation** via GitHub Actions on every commit

## Running Tests

### Quick Start

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/test_ml_pipeline.py -v

# Run specific test class
pytest tests/test_ml_pipeline.py::TestPitchDetection -v

# Run with coverage report
pytest tests/test_ml_pipeline.py --cov=ml --cov-report=html
```

### Standalone Evaluators

```bash
# Run pitch evaluator standalone
python -m tests.evaluate_pitch

# Run full pipeline evaluator
python -m tests.evaluate_pipeline

# Both together
python -c "from tests.evaluate_pitch import run as rp; from tests.evaluate_pipeline import run as rpl; print(rp()); print(rpl())"
```

## Test Suites

### 1. Pitch Detection (`TestPitchDetection`)

**Synthetic Test Audio**: Concatenated pure sine waves at known frequencies (220 Hz, 440 Hz, 329.63 Hz).

**Metrics**:
- **Frame-level RMSE (Hz)**: Root mean squared error between estimated and ground-truth F0 per frame. Typically 40-50 Hz for pure tones; higher for noisy recordings.
- **Voiced True Positive Rate**: Fraction of ground-truth voiced frames correctly detected as voiced (target: >0.95).
- **Voiced False Positive Rate**: Fraction of unvoiced frames incorrectly flagged as voiced (target: <0.1).
- **Per-Note Median Error (Hz & cents)**: Median F0 within each note window vs. target frequency. For pure tones, typically <1 Hz or ±1 cent.
- **Note-Level RMSE (Hz)**: RMS of per-note median errors.

**Interpretation**:
- Frame-level RMSE can be large (40-50 Hz) while per-note medians are accurate; this is **expected** for stable notes because F0 varies per frame but the median over the note is stable.
- For coaching, **per-note stability** (low note RMSE) is more relevant than frame RMSE.
- Voiced detection rates reflect the robustness of pitch onset/offset detection.

**Extended Tests**:
- `test_pitch_with_microtonal_offsets`: Detects notes at ±20 cent deviations (tests pitch precision).
- `test_pitch_with_vibrato`: Validates pitch tracking under frequency modulation (4-8 Hz LFO, ±50 cents).
- `test_pitch_with_white_noise`: Assesses robustness in noisy environments (SNR=10dB).

---

### 2. Rhythm Detection (`TestRhythmDetection`)

**Synthetic Test Audio**: Metronome click train at 100 BPM (clicks every 0.6 seconds).

**Metrics**:
- **Detected Tempo (BPM)**: Estimated tempo via `librosa.beat.beat_track` or onset-envelope fallback.
- **Tempo Tolerance**: Detected tempo within ±4 BPM of ground truth (target: `within_tolerance=True`).
- **Timing Variance**: Standard deviation of inter-beat intervals. Lower = more consistent rhythm.
- **Beat Alignment**: Categorical assessment ("aligned" if variance < 0.05s, "loose" otherwise).

**Interpretation**:
- Rhythm detection is used to assess timing consistency in a singing passage.
- Low timing variance indicates steady rhythm; high variance suggests rushed/dragged passages.
- Detected tempo in a range around the true tempo is acceptable; exact detection is sensitive to signal characteristics.

---

### 3. Breath Detection (`TestBreathDetection`)

**Synthetic Test Audio**: Continuous 440 Hz tone with 4 periodic silence gaps (duration: 0.18s, interval: 1.5s).

**Metrics**:
- **Detected Breaths**: Count of silence gaps identified.
- **Breath Detection Tolerance**: Expected breaths ±1 (accounts for edge cases and silence at start/end).
- **Support Score**: Metric inversely related to breath length variability (target: high score = regular breathing).
- **Breath Length Variation**: Standard deviation of inter-breath intervals.

**Interpretation**:
- Breath detection helps identify phrasing and breath control issues.
- Regular breathing (low variation) suggests good phrase planning.
- Irregular breathing may indicate fatigue, tension, or poor phrasing.
- On real recordings, threshold tuning (top_db for silence detection) may be needed.

---

### 4. Spectral Analysis (`TestSpectralAnalysis`)

**Synthetic Test Audio**: 440 Hz pure tone (2 seconds).

**Metrics**:
- **Spectral Centroid Mean (Hz)**: Center-of-mass frequency of the spectrum. For a pure 440 Hz tone, typically 440±20 Hz (depends on windowing).
- **Centroid Tolerance**: Centroid within ±200 Hz of target (conservative for robustness).
- **MFCC Features**: 13-dimensional Mel-frequency Cepstral Coefficients (voice timbre proxy).
- **Energy Contour**: Frame-wise RMS energy.

**Interpretation**:
- Spectral centroid reflects the dominant frequency; useful for vowel identification and formant tracking.
- MFCCs capture voice quality/timbre; used in coaching to suggest vowel shape or resonance improvements.
- Energy contour reveals vocal support; steady energy = good control, variable energy = inconsistent support.

---

## Visualization Outputs

Tests automatically save plots to `tests/results/`:

| Plot | Purpose |
|------|---------|
| `pitch_curve.png` | Estimated vs. ground-truth F0 over time |
| `energy_envelope.png` | RMS energy with detected breath times marked |
| `beat_alignment.png` | Beat timeline and inter-beat interval stability |
| `note_errors.png` | Per-note frequency and cents errors (bar charts) |

View these plots to qualitatively inspect analysis quality.

---

## Synthetic Data Generators (`tests/synthetic_data.py`)

The test suite uses parameterized synthetic generators to create controlled audio:

```python
from tests.synthetic_data import synth_melody, synth_vibrato_sweep, synth_noisy_melody, synth_amplitude_envelope

# Pure melody
y = synth_melody([220, 440, 329.63], [1.5, 1.5, 1.5])

# With vibrato (4 Hz LFO, ±50 cents)
y = synth_vibrato_sweep(440, duration=3.0, vibrato_freq_start=4, vibrato_freq_end=6, vibrato_cents=50)

# With white noise (SNR=10dB)
y = synth_noisy_melody([220, 440], [1.5, 1.5], snr_db=10.0)

# With amplitude envelope (ADSR)
y = synth_amplitude_envelope(440, duration=2.0, envelope="adsr")
```

Use these to extend tests for new scenarios (e.g., lower SNR, different envelopes).

---

## Continuous Integration (GitHub Actions)

The workflow `.github/workflows/tests.yml` runs tests on every commit:

1. **Install dependencies** from `requirements-dev.txt`
2. **Run pytest** suite
3. **Generate coverage report**
4. **Artifact storage** (optional): Save plot artifacts for inspection

View results on the GitHub Actions tab in your repository.

---

## Troubleshooting

### Test Failures

| Issue | Cause | Fix |
|-------|-------|-----|
| Imports fail (librosa, soundfile) | Missing dev dependencies | `pip install -r requirements-dev.txt` |
| `MATPLOTLIB_AVAILABLE = False` | Matplotlib not installed | `pip install matplotlib` |
| Plot files not saved | Missing `tests/results/` dir | Created automatically; check permissions |
| Pitch detection too inaccurate | Bad F0 range or top_db in silence detection | Adjust `fmin`, `fmax` in `ml.pitch_detection.extract_pitch_features` |

### Extending Tests

1. **Add a new synthetic scenario**: Edit `tests/synthetic_data.py` with a new generator.
2. **Add a test method**: Implement in `tests/test_ml_pipeline.py` (e.g., `test_pitch_extreme_vibrato`).
3. **Visualize results**: Call plot utilities from `tests/visualization.py` in your test.

---

## Future Work

- Collect and label real singing recordings for validation.
- Implement perceptual loss metrics (e.g., PESQ for pitch similarity).
- Expand synthetic dataset with consonants, speech, and polyphony.
- Add real-time performance benchmarking (latency, throughput).
- Create a benchmark dashboard tracking trends over time.

---

## References

- Librosa documentation: https://librosa.org/doc/latest/index.html
- F0 estimation with PYIN: https://librosa.org/doc/latest/generated/librosa.pyin.html
- MFCC features: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
- Pytest documentation: https://docs.pytest.org/

---

*Last updated: 2026-05-21*
