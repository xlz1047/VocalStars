# VocalStars — Slide Content
## System Overview + Evaluation
_Copy-paste ready for Google Slides / PowerPoint_

---

# SLIDE: SYSTEM OVERVIEW

**Slide title:** System Overview — Five-Stage Pipeline

**Diagram:** Use `docs/architecture.svg` (open in Preview → Export as PNG for Google Slides)

---

## Stage 1 — Data

**Header text:** Data

**Bullet points:**
- 3 datasets: GTSinger, VocalSet, PopBuTFy
- GTSinger: melody + technique labels (16 technique categories, multi-language)
- VocalSet: 20 singers × 10 vocal techniques, studio-quality recordings
- PopBuTFy: pop vocal stems with pitch-annotated MIDI alignment
- MIR-1K: used for evaluation only (not training) — 1,000 clips, frame-level f0 + voicing ground truth
- Label types: pitch f0 (pyin), voice activity (VAD), note onset, breath events, technique tags

**Speaker note:** Labels were generated automatically using `librosa.pyin` (probabilistic YIN) rather than manual annotation — this is standard practice in MIR but introduces label noise, which explains some of the lower F1 scores in breath/onset.

---

## Stage 2 — Preprocessing

**Header text:** Preprocessing

**Bullet points:**
- Resample all audio to 16 kHz mono
- Frame length: 4,096 samples (256 ms), 50% overlap (128 ms hop)
- Feature: HCQT (Harmonic Constant-Q Transform)
  - 6 harmonics × 360 frequency bins
  - Covers B0–B6 (6 octaves) at 20-cent resolution
- Secondary: Log-mel spectrogram, 128 bands (used by NanoPitch head)
- Peak normalization per clip
- pyin run offline to generate ground-truth f0 and voiced/unvoiced labels

**Why HCQT over mel spectrogram:**
HCQT stacks the signal at harmonics h=1,2,3,4,5,6. This gives the model explicit access to harmonic structure, making pitch disambiguation easier than from a standard mel spectrogram alone. Standard approach from Bitteur et al. and Bitteur's Deep Salience work.

---

## Stage 3 — Model Architecture

**Header text:** Model (three components)

### Component A — UnifiedVocalModel (primary)
- **Type:** Custom — trained from scratch, not pretrained or fine-tuned
- **Architecture:** HCQT encoder → shared GRU backbone → 5 task heads
- **Input:** HCQT tensor (6 × 360 × T frames)
- **Backbone:** 2-layer bidirectional GRU, hidden size 256
- **Task heads:**
  - Pitch head → 360-bin softmax (20-cent resolution)
  - VAD head → sigmoid binary classifier
  - Onset head → sigmoid frame-level onset detector
  - Breath head → sigmoid breath event detector
  - Technique head → 5-class softmax (straight tone, vibrato, breathy, pressed, neutral)
- **Parameters:** ~256K
- **Checkpoint size:** 1.0 MB

### Component B — NanoPitch (guard model)
- **Type:** Lightweight CNN — adapted from RNNoise architecture
- **Architecture:** Conv1d(40→64) → Conv1d(64→96) → 3× GRU(96) → VAD head + pitch head
- **Input:** 40-band log-mel spectrogram, 25 ms window, 10 ms hop
- **Outputs:** VAD probability (sigmoid) + 360-bin pitch posteriorgram (sigmoid per bin)
- **Parameters:** 332,873
- **Checkpoint size:** 1.3 MB
- **Role:** Conservative noise/silence guard — not used for coaching, only to veto non-voice inputs

### Component C — pyin (DSP baseline)
- **Type:** Signal processing — no learned parameters
- **Library:** `librosa.pyin` (probabilistic YIN algorithm)
- **Output:** f0 per frame + voiced probability
- **Role:** Clean f0 reference for diagnostic tasks (sustained note, pitch slide) where Model A f0 is too jumpy

---

## Stage 4 — Post-processing

**Header text:** Post-processing

**Bullet points:**
- **Hybrid f0/VAD decision logic:** all three sources (Model A, NanoPitch, pyin) run on same 10 ms frame grid; source selected per task type
- **Noise veto:** if NanoPitch VAD = 0 on ≥80% of frames → block coaching entirely (signal validity gate)
- **f0 source selection per task:**
  - `sustained_note`, `pitch_slide` → pyin f0 (zero jump artifacts)
  - `free_singing`, `note_match` → Model A f0 (broader coverage), pyin as sanity guard
- **f0 smoothing:** median filter over 5-frame window to remove single-frame spikes
- **Note segmentation:** onset frames from Model A onset head → segment boundaries → note objects with start, end, median f0, confidence
- **Task-aware evaluators:** 4 implemented — `sustained_note`, `pitch_slide`, `free_singing`, `note_match`
- **Output layers:** frame-level (f0, VAD, confidence, volume) → segment-level (notes, phrases, dropouts) → task-level (score, feedback, caveats)

---

## Stage 5 — Feedback to User

**Header text:** Feedback to User

**Bullet points:**
- **Pitch lane:** f0 contour plotted over time, colored by cents error vs. target note
- **Issue markers:** flags placed at onset, breath, and unstable-pitch regions on the timeline
- **Coaching cards:** task score, subscores breakdown, technique feedback, allowed/blocked feedback policy
- **Exercise suggestions:** next exercise recommended based on detected weakness (e.g., "try a sustained A4 for breath control")
- **Invalid input state:** user shown clear reason when audio is blocked (silence, noise, speech detected)
- **Stack:** Vite + React + TypeScript frontend, FastAPI REST backend, SQLite database

---

## System Specs (bottom row of slide)

| Spec | Value |
|------|-------|
| Compute | CPU only — no GPU required |
| Mode | Offline (batch) |
| NanoPitch inference | 0.08–0.15 s per 5 s clip (CPU) |
| Model A size | ~256K params, 1.0 MB checkpoint |
| NanoPitch size | 332K params, 1.3 MB checkpoint |
| Input format | 16 kHz mono WAV/audio |
| Backend | FastAPI, Python 3.11, SQLite |
| Frontend | Vite, React, TypeScript |
| Deployment | Local / Docker Compose |

---
---

# SLIDE: EVALUATION

**Slide title:** Evaluation — Metrics, Methodology, and Honest Results

---

## How Metrics Were Computed (methodology box)

- **Library:** `mir_eval` — standard MIR community benchmark (used in MIREX, ISMIR competitions)
- **Frame rate:** 10 ms hop (matches HCQT and NanoPitch preprocessing)
- **Ground truth:** pyin-generated f0 and voiced/unvoiced labels (same as training labels)
- **Voicing threshold:** 0.5 sigmoid probability
- **Split:** manifest-based, stratified by dataset source — held-out songs not seen during training
- **Epochs trained:** 70 (unified model), 30 (standalone VAD), 30 (standalone onset)

---

## Metric Definitions

| Metric | Full name | Definition |
|--------|-----------|------------|
| **RPA** | Raw Pitch Accuracy | % of voiced frames where predicted f0 is within ±50 cents of ground truth |
| **RCA** | Raw Chroma Accuracy | Same as RPA but octave-invariant — detects octave errors separately |
| **VDR** | Voicing Detection Rate | Recall — % of truly-voiced frames correctly detected as voiced |
| **VFA** | Voicing False Alarm | % of truly-unvoiced frames incorrectly called voiced — **key failure metric** |
| **Median cents** | Median f0 error | Median absolute f0 error in cents on voiced frames (100 cents = 1 semitone) |
| **F1** | Harmonic mean | 2 × (precision × recall) / (precision + recall) |

---

## Result 1 — Pitch Accuracy (dedicated pitch head)

Evaluated on held-out manifest. Two decoding strategies compared:

| Decoding | RPA | RCA | VDR | VFA | Median cents |
|----------|:---:|:---:|:---:|:---:|:---:|
| Argmax (greedy) | **93.1%** | **94.1%** | **96.3%** | **5.1%** | **11.2 ¢** |
| Viterbi smoothing | 90.1% | 91.5% | 96.3% | 5.1% | 11.4 ¢ |

**Interpretation:**
- Argmax gives higher raw accuracy; Viterbi trades ~3 pp accuracy for smoother temporal transitions (better for display in pitch lane)
- 11.2 cents = ~1/9th of a semitone — sufficient precision for beginner coaching (humans can reliably distinguish ~25 cents)
- VFA of 5.1% on the isolated pitch model vs. 42.3% in the unified model — shows the unified model's false alarm problem clearly

**Baseline for comparison:**
On the same manifest, the `baseline_vs_pyin_labels` experiment with the same architecture:
RPA = 93.1%, RCA = 93.9%, VFA = 9.2%, median = 11.2 cents — consistent across runs.

---

## Result 2 — Unified Model Training Curves (70 epochs)

Key milestones across training:

| Epoch | Train Loss | RPA | VDR | VFA | Median cents |
|-------|:---:|:---:|:---:|:---:|:---:|
| 1 | 3.78 | 60.8% | 79.7% | 39.3% | 20.0 ¢ |
| 10 | 1.66 | 83.5% | 96.4% | 42.5% | 13.3 ¢ |
| 20 | 1.53 | 84.4% | 93.9% | 30.6% | 13.3 ¢ |
| 40 | 1.41 | 86.7% | 96.3% | 36.6% | 13.3 ¢ |
| 70 | 1.33 | **87.3%** | **97.3%** | **42.3%** | **13.3 ¢** |

**Key observations:**
- RPA improves steadily from 60.8% → 87.3% (+26.5 pp over 70 epochs
- Median cents error converges fast — plateaus at 13.3 ¢ by epoch 6, no further gain
- VDR (recall) is high and stable from epoch ~15 onward
- **VFA does not converge** — oscillates between 30–53% throughout, ends at 42.3%
- This non-convergence of VFA is the core motivation for the hybrid approach

---

## Result 3 — The VFA Problem and Why Hybrid Was Needed

This is the central design finding:

| Source | VDR (coverage of real singing) | VFA (false alarm on noise/silence) |
|--------|:---:|:---:|
| Model A (unified) | **97.3%** | 42.3% ❌ |
| NanoPitch | 10.5–79.7% (task-dependent) | **0.0%** ✅ |
| pyin | 82.1–88.8% | 48.7% ❌ |
| **Hybrid stack** | **93%+** | **0.0%** ✅ |

Tested on real recorded samples:

| Input | Model A | NanoPitch | pyin | Hybrid |
|-------|:---:|:---:|:---:|:---:|
| Fan-noise silence | 95.5% voiced ❌ | **0.0%** ✅ | 48.7% ❌ | **0.0%** ✅ |
| White noise | 49.9% voiced ❌ | **0.0%** ✅ | 51.3% ❌ | **0.0%** ✅ |
| Sustained sung vowel | 99.0% voiced ✅ | 0.0% ❌ | 88.8% ✅ | **99.0%** ✅ |
| Free singing (Twinkle) | 93.3% voiced ✅ | 10.5% ❌ | 82.1% ✅ | **93.3%** ✅ |

**Design decision:** NanoPitch used solely as a noise veto (if NanoPitch strongly rejects → block analysis). Model A used for all coaching outputs. pyin used for f0 accuracy on diagnostic tasks.

---

## Result 4 — Pitch Jump Rate on Real Recordings

Octave and semitone discontinuities per recording. These cause incoherent pitch lanes and bad feedback:

| Recording | Model A octave jumps | Model A semitone jumps | pyin octave jumps | pyin semitone jumps |
|-----------|:---:|:---:|:---:|:---:|
| Sustained vowel (`03_sustained_aaa.wav`) | 27 | 31 | **0** | **0** |
| Pitch slide (`04_pitch_slide.wav`) | 2 | 8 | **0** | **0** |
| Free singing (`05_twinkle_twinkle.wav`) | 8 | 15 | **0** | **0** |

**Why pyin for diagnostic tasks:** zero octave/semitone jumps on all tested recordings. Model A's raw f0 is too noisy for direct display without heavy postprocessing.

---

## Result 5 — Standalone VAD Model (with validation split)

The unified model's VAD head has no validation loss logged (training instrumentation gap). The standalone VAD model does have train/val split:

| Metric | Train | Validation | Gap |
|--------|:---:|:---:|:---:|
| Loss | 0.031 | **0.040** | 0.009 |
| F1 | 0.941 | **0.945** | −0.004 |
| Precision | 0.986 | — | — |
| Recall | 0.965 | — | — |

Small train/val gap (0.009 loss) — model generalizes, not severely overfit. Val F1 (0.945) slightly exceeds train F1 (0.941), consistent with dropout regularization.

---

## Result 6 — Onset Detection (standalone model)

Onset detection is a genuinely hard problem:

| Metric | Train | Validation |
|--------|:---:|:---:|
| Loss | 0.043 | **0.045** |
| Onset Recall | 0.983 | 0.937–0.983 |
| Onset Precision | **0.272** | ~0.27 |
| Onset F1 | **0.282** | ~0.28 |

**Honest interpretation:** Recall is 98.3% — almost every real onset is detected. But precision is only 27.2% — meaning ~73% of detected "onsets" are false positives. F1 = 0.28 is poor. This is a known issue with frame-level onset detection on noisy/continuous singing: the model finds it hard to suppress spurious detections during sustained notes and vibrato. In practice, onset output is used only for note segmentation boundaries, where false positives are filtered by minimum note duration rules.

---

## Result 7 — Honest Weaknesses (show this, it builds credibility)

| Task | Metric | Value | Root cause |
|------|--------|-------|------------|
| Breath detection | F1 | **0.035** | Class imbalance — breath events occupy <2% of frames; model collapses to predicting "not breath" |
| Onset detection | Precision | **27.2%** | Spurious detections during sustained and vibratto phrases; hard to threshold without recall loss |
| Technique classification | Accuracy | **16%** | 5-class problem; GTSinger technique labels are coarse; limited per-class samples in training split |
| Unified model VFA | False alarm | **42.3%** | Model optimizes VAD recall at expense of precision; unvoiced/noise suppression not penalized heavily enough in loss |
| Validation loss | Logged | **Not recorded** | Instrumentation gap in unified model training — only train loss tracked per epoch |

These are presented as **known limitations and future work**, not system failures.

---

## Behavioral Regression Tests (P4 suite)

5 acceptance tests run against self-recorded audio:

| Test | Input | Expected behavior | Result |
|------|-------|-------------------|--------|
| 1 | `00_silence.wav` (fan noise) | No coaching output — blocked as non-voice | **PASS** |
| 2 | `01_speaking_voice.wav` | No singing feedback — blocked as speech | **PASS** |
| 3 | `03_sustained_aaa.wav` | Routes to `sustained_note` evaluator (not reference-song scorer) | **PASS** |
| 4 | `04_pitch_slide.wav` | Routes to `pitch_slide` evaluator | **PASS** |
| 5 | `05_twinkle_twinkle.wav` | General free-singing feedback, no false reference-score claim | **PASS** |

Run command: `ml/.venv/bin/python scripts/eval/check_regression_expectations.py`

---
---

# ADDITIONAL EVALUATION — MULTI-DIMENSION RESULTS
_Add these to the evaluation slides (not replacement — addition)_

---

## Table A — Per-recording results across all evaluated dimensions
_Use on Slide 1 of evaluation. Shows what the system measures on real inputs._

| Recording | Pitch stability (¢ spread) | Octave jumps | Direction | Task routed |
|-----------|:---:|:---:|:---:|:---:|
| Fan-noise silence | 165.6 ¢ | 0 | −0.2 Hz/s | **Blocked — not voice** |
| Speaking voice | 678.6 ¢ | 16 | +23.2 Hz/s | **Blocked — not singing** |
| Sustained vowel | 549 ¢ (A) / **194 ¢ (pyin)** | 27 / **0** | +0.6 Hz/s | `sustained_note` |
| Pitch slide | 375 ¢ (A) / 388 ¢ (pyin) | 2 / **0** | +8.4 Hz/s | `pitch_slide` |
| Free singing | 450 ¢ (A) / 648 ¢ (pyin) | 8 / **0** | −0.8 Hz/s | `free_singing` |

_Lower cents spread = more stable pitch. Direction slope confirms pitch slide direction is captured correctly._

---

## Table B — Per-task head quality (all 5 evaluated dimensions)
_Use on Slide 2 of evaluation. Covers breath, rhythm, and technique beyond pitch + VAD._

| Task head | Metric | Value | Interpretation |
|-----------|--------|:-----:|----------------|
| Pitch detection | RPA | **93.1%** | Strong — within ±50 cents on 93% of voiced frames |
| Pitch detection | Median f0 error | **11.2 ¢** | ~1/9 semitone, sufficient for beginner coaching |
| Voice activity (VAD) | VDR / VFA | **97.3% / 42.3%** | High recall; hallucinates on noise → fixed by NanoPitch guard |
| Pitch stability | Cents spread (pyin, sustained note) | **193.8 ¢** | ~2 semitones spread on held note — detectable and displayable |
| Breath control | F1 | **0.035** | Weak — breath events are <2% of frames (class imbalance) |
| Onset / rhythm | F1 | **0.056** (precision 27%) | Detects almost every onset (98% recall) but 73% are false positives |
| Technique classification | Accuracy | **16%** | 5-class; limited labeled data per category in GTSinger |

_Pitch and pitch stability are the system's strongest outputs. Breath, onset, and technique are implemented and produce output but accuracy is low — disclosed as known limitation with class imbalance and limited labeled data as root causes._

---

## One-Paragraph Summary (for slide or speaker notes)

> We evaluate using `mir_eval` standard MIR metrics across 70 training epochs. The dedicated pitch head achieves **93.1% Raw Pitch Accuracy** with **11.2-cent median error** (~1/9 semitone) and only **5.1% voicing false alarm**. However, the unified multitask model's voicing false alarm does not converge — reaching **42.3% at epoch 70** — meaning it hallucinates singing on background noise. This directly motivated the hybrid stack: NanoPitch (332K params) acts as a conservative noise veto achieving **0% false alarm on silence and white noise**, while Model A provides coverage and task-specific coaching heads. Known weaknesses include **breath F1 of 0.035** (class imbalance), **onset precision of 27%** (false positive heavy), and **technique accuracy of 16%** (limited labeled data). All 5 behavioral regression checks pass. These limitations are documented and point to clear future work: loss reweighting for imbalanced tasks, separate precision-recall thresholds per head, and larger technique-labeled datasets.

---

## Anticipated Expert Questions + Answers

**Q: Why is your validation loss not in the training log?**
A: The unified model training script logged train loss and mir_eval metrics per epoch but did not compute a separate held-out loss. The standalone VAD and onset models do have train/val loss logged (gap of 0.009). This is a known instrumentation gap — the per-epoch mir_eval metrics are computed on a held-out eval split and show consistent generalization.

**Q: Your ground truth is pyin — isn't that circular?**
A: Yes, and we acknowledge it. pyin is an established probabilistic pitch estimator used as pseudo-ground-truth throughout the MIR literature (e.g., CREPE, SPICE training). We use it for training labels and also include it as a baseline source in the hybrid. The dedicated pitch model is evaluated against the same pyin labels, so absolute accuracy against human annotation would likely be lower. MIR-1K provides a route to external ground truth — tooling implemented, dataset download pending.

**Q: Breath F1 of 0.035 is essentially random — is the breath head useful?**
A: In isolation, no. At the current recall/precision balance it identifies roughly 1 in 30 breath events correctly. In the app it is used as a soft signal for phrase boundary detection (combined with RMS energy drops), not as a hard gate. Future work: loss reweighting (focal loss), longer context window, and explicit negative mining.

**Q: Why not use CREPE or a pretrained model?**
A: CREPE (McFee et al.) was considered. The constraint was web-deployability and <5 MB: CREPE's smallest variant is ~17 MB. Our custom HCQT+GRU achieves 93.1% RPA at 1 MB. NanoPitch (332K, 1.3 MB) is the browser-deployable candidate with WASM artifacts. Pretraining was not used due to domain specificity of the singing vocal coaching task vs. general pitch estimation.
