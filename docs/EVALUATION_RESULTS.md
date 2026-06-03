# VocalStars — Evaluation Results

Consolidated from evaluation reports N4, M1, M2, and the P4 regression suite.

---

## Evaluation Metric

**Primary metric: False-Voiced Rate (FVR)** — the fraction of audio frames incorrectly classified
as sung voice on non-singing inputs (silence, noise, speaking). A coaching app that hallucinates
singing on background noise produces meaningless feedback, so FVR directly measures whether the
system meets its goal of analyzing only genuine singing.

**Secondary metric: Pitch Jump Rate (PJR)** — octave and semitone discontinuities per recording on
real sung inputs. Jumpy f0 produces incoherent pitch feedback, so PJR measures coaching accuracy.

**Tertiary metric: CPU inference time** — offline, no GPU required.

---

## Model Candidates Evaluated

| Model | Parameters | Role |
|-------|-----------|------|
| **Model A** — `ml_new` UnifiedVocalModel (HCQT+GRU) | ~256 K | Primary: VAD, pitch, onset, breath, technique |
| **NanoPitch** — `ml_3/NanoPitch` (CNN posterior) | 332 K | Guard: conservative silence/noise rejection |
| **pyin** — librosa DSP baseline | — | Reference: stable f0 for diagnostic tasks |

---

## False-Voiced Rate on Non-Singing Inputs

Lower is better (0% = perfect noise rejection).

| Input | Model A | NanoPitch | pyin |
|-------|--------:|----------:|-----:|
| Real fan-noise silence (`00_silence.wav`) | **95.5%** | **0.0%** | 48.7% |
| Synthetic white noise | **49.9%** | **0.0%** | 51.3% |
| Low-frequency hum (80 Hz) | voiced at wrong pitch | rejected | tracks 80 Hz |
| Digital silence | 0.2% | 0.0% | 0.0% |

**Result:** NanoPitch achieves 0% FVR on all silence/noise inputs. Model A and pyin both
false-voice noise at unacceptable rates when used alone. NanoPitch is used as the noise guard
in the hybrid stack.

---

## Pitch Jump Rate on Real Singing

Octave jumps / semitone jumps per recording. Lower is better.

| Recording | Model A | NanoPitch | pyin |
|-----------|--------:|----------:|-----:|
| Sustained vowel (`03_sustained_aaa.wav`) | 27 oct / 31 semi | 0 / 0 | **0 / 0** |
| Pitch slide (`04_pitch_slide.wav`) | 2 / 8 | 0 / 0 | **0 / 0** |
| Free singing — Twinkle (`05_twinkle_twinkle.wav`) | 8 / 15 | 0 / 1 | **0 / 0** |

**Result:** pyin produces the cleanest f0 (zero jumps) for diagnostic tasks. Model A has the
highest jump rate but the broadest voiced coverage (93%+) needed for free-singing heads.
Hybrid strategy: pyin for f0 smoothness, Model A for task/onset/technique heads.

---

## Pitch Accuracy on Synthetic Tones

Ground truth known precisely for pure sine inputs.

| Input | Model A f0 | pyin f0 | Ground truth |
|-------|-----------|---------|-------------|
| 220 Hz sine | 219.98 Hz | 219.98 Hz | 220.00 Hz |
| 440 Hz sine | 439.96 Hz | 439.96 Hz | 440.00 Hz |
| 220→440 Hz sweep | accurate trajectory | accurate trajectory | swept |

Both Model A and pyin track pure sine f0 to within 0.04 Hz. NanoPitch rejects pure sine tones
(not human-voice-like enough), confirming its role as a singing-voice guard, not a general pitch tracker.

---

## Voiced Coverage on Real Singing

Higher is better for sung inputs.

| Recording | Model A | NanoPitch | pyin |
|-----------|--------:|----------:|-----:|
| Sustained vowel | 99.0% | **0.0%** (under-voices) | 88.8% |
| Pitch slide | 92.9% | 79.7% | 86.8% |
| Free singing | 93.3% | 10.5% | 82.1% |

NanoPitch is too conservative for sustained/free singing — it under-voices real sung input. This
confirms the hybrid strategy: Model A carries coverage, NanoPitch provides the noise veto.

---

## Runtime (CPU, offline)

| Model | Input length | Inference time |
|-------|-------------|---------------|
| NanoPitch | 5–8.5 s | 0.08–0.15 s |
| Model A (UnifiedVocalModel) | varies | not yet formally benchmarked |
| pyin (librosa) | 5–8.5 s | ~0.3–0.6 s (DSP) |

Target: <50 ms per 256 ms frame for real-time use. NanoPitch meets this comfortably.

---

## Regression Tests (P4 Behavioral Suite)

Five behavioral checks run against `samples/*.wav`:

| Check | Expected | Result |
|-------|----------|--------|
| Silence blocked from coaching | no coaching output | PASS |
| Speaking voice blocked | no singing feedback | PASS |
| Sustained vowel → diagnostic task | sustained_note evaluator | PASS |
| Pitch slide → diagnostic task | pitch_slide evaluator | PASS |
| Free singing → general feedback | free_singing evaluator (no false reference score) | PASS |

**5/5 P4 behavioral regression checks pass.**

Run: `ml/.venv/bin/python scripts/eval/check_regression_expectations.py`

---

## Hybrid Strategy Outcome

Final recommended stack: **Model A + NanoPitch + pyin**

- NanoPitch vetoes silence/noise (FVR = 0%)
- pyin provides clean f0 for pitch-sensitive diagnostic tasks
- Model A provides broad coverage and all app-specific heads (onset, breath, technique, coaching)

Source selection per task:

| Task | f0 source | VAD source |
|------|-----------|-----------|
| `sustained_note` | pyin | Model A (with NanoPitch veto) |
| `pitch_slide` | pyin | Model A (with NanoPitch veto) |
| `free_singing` | Model A (pyin-guarded) | Model A (with NanoPitch veto) |
| `note_match` | pyin | Model A |
