# Model Failure Analysis

This document analyzes the self-recorded WAV rerun described in `EVAL_RUN_REPORT.md`, using the model and pipeline findings from `MODEL_AUDIT.md`.

Scope:

- No model architecture changes.
- No retraining.
- No source code changes in this pass.
- Focus on why the current checkpoint-backed inference path treats invalid or diagnostic audio as normal singing and how to fix that safely.

## Executive Summary

The current checkpoint path can produce pitch, voiced frames, note segments, scores, and coaching text for all five self-recorded inputs, but it does not yet decide whether the input is valid analyzable singing. That missing validity layer is the main product failure.

The most severe result is `00_silence.wav`: fan/background noise was classified as 95.5% voiced, converted into 19 notes, given 100% nearest-semitone pitch accuracy, and received singing coaching. This strongly suggests that the app is treating "any confident-enough periodic or low-frequency signal" as vocal singing.

The failures appear to be a combination of:

- model VAD/pitch outputs that are not robust to noise and speech,
- permissive thresholds,
- note segmentation that is too sensitive to frame-level f0 jumps,
- scoring that rewards nearest equal-tempered pitch instead of task correctness,
- and, most importantly, missing gate logic before score/coaching generation.

Until the app can classify an input as analyzable singing, score and coaching should be withheld or heavily caveated.

## Evidence From WAV Rerun

All five WAV samples ran in checkpoint mode:

- `inference_mode`: `checkpoint`
- `checkpoint_path_used`: `ml_new/checkpoints/unified/best.pt`
- `device_used`: `cpu`
- `model_stack_used`: `ml_new`

Observed results:

| Sample | Expected input | Observed output | Main failure |
| --- | --- | --- | --- |
| `00_silence.wav` | Fan/background noise, no human voice | 95.5% voiced, 19 notes, score 71, singing coaching | Severe false positive |
| `01_speaking_voice.wav` | Speech, not singing | 83.4% voiced, 14 notes, score 71, singing coaching | No speech-vs-singing gate |
| `03_sustained_aaa.wav` | Held sung vowel | 99.0% voiced, 14 notes, 6 onsets, f0 70.6-659.2 Hz | Over-fragmented pitch/onsets |
| `04_pitch_slide.wav` | Sung pitch slide | 92.9% voiced, 3 notes, score 91, no caveats | Over-generous scoring |
| `05_twinkle_twinkle.wav` | Short sung melody | 93.3% voiced, 12 notes, score 78 | No melody/reference scoring |

## 1. Why Silence/Fan Noise Is Being Treated As Voiced Singing

The silence sample is not truly digital silence; it contains fan/background noise. The model appears to treat that noise as voiced because the inference path accepts a frame as voiced whenever the checkpoint's voiced probability crosses a fixed threshold.

In `ml_new/inference/coach_inference.py`, the checkpoint path:

- extracts HCQT and VAD features,
- runs the unified model,
- converts pitch logits to a single argmax f0 bin for every frame,
- thresholds `voiced_prob >= 0.50`,
- keeps the argmax pitch wherever the voiced mask is true,
- then sends the result to note segmentation and coaching.

Fan noise can contain steady low-frequency energy. The rerun's false f0 range for `00_silence.wav` was 65.4-101.8 Hz, which is exactly the kind of low-frequency band that can look pitch-like to a harmonic feature extractor. Once the VAD says "voiced", the pitch head always has a best bin, so background hum becomes a note stream.

The training pipeline described in the audit likely contributes to this. Some training labels appear to derive voicing from energy/RMS-like heuristics rather than from robust human-voice annotations. If a model learns "sustained energy plus harmonic-looking features means voiced", fan noise is a plausible false positive.

The scoring makes the problem worse. Pitch accuracy is nearest-semitone based, not task/reference based. A stable fan hum near any equal-tempered note can receive very high pitch accuracy even though there is no voice.

## 2. Likely Failure Layer

The failure is not isolated to one layer. Based on the current artifacts:

| Layer | Likelihood | Reason |
| --- | --- | --- |
| VAD model output | High | The checkpoint path reported 95.5% voiced for non-human noise. |
| Thresholding | High | `VOICED_THRESH = 0.50` is a single permissive threshold with no hysteresis, confidence calibration, or noise rejection. |
| Postprocessing | High | Voiced frames are used directly for notes, scoring, and coaching without sanity checks. |
| Feature extraction | Medium-high | HCQT and low-level VAD features may interpret tonal fan noise or low-frequency hum as pitch-like. |
| Coaching gate logic | Definite | The app generates singing feedback even when the input should be invalid or non-singing. |
| Backend/frontend behavior | Secondary | The backend now shows checkpoint mode, but the underlying inference result still lacks validity gating. |

Important limitation: the current reports do not expose raw `voiced_prob`, pitch softmax confidence, entropy, or model margins. Without those, we cannot tell whether the VAD head was extremely confident on fan noise or whether many frames were barely above the threshold. That distinction matters for tuning.

## 3. Why Speech Is Treated As Singing

Speech contains voiced phonation, changing f0, formants, syllable-like onsets, and harmonic structure. The current model heads appear to predict:

- voiced/unvoiced,
- pitch,
- breath,
- onset,
- and technique labels.

They do not appear to predict "speech vs singing" or "analyzable singing vs non-singing vocal input".

As a result, `01_speaking_voice.wav` being 83.4% voiced is not surprising. A voice activity detector should often mark speech as voiced. The product failure is that a speech input then proceeds through the singing note, score, and coaching pipeline as if it were a melody.

The system needs a separate singing-validity decision. Voiced speech is not the same thing as analyzable singing.

## 4. Why A Sustained Vowel Fragments Into Many Notes/Onsets

`03_sustained_aaa.wav` should be close to one sustained voiced region. Instead, it produced:

- 99.0% voiced,
- 14 notes,
- 6 onsets,
- f0 range 70.6-659.2 Hz.

The likely causes are:

- frame-level f0 jumps,
- octave errors or subharmonic/overtonal switches,
- onset thresholding that is too sensitive,
- note segmentation that starts a new note when adjacent f0 frames jump by more than 1.5 semitones,
- and a minimum note duration of only 100 ms.

In `ml_new/inference/algorithms.py`, `segment_notes()` creates a new note after silence or when adjacent pitch frames jump by more than `jump_thresh_semitones=1.5`. For a sustained vowel with noisy f0 estimates, this is likely to split one note into many small notes.

The 70.6-659.2 Hz f0 range is especially suspicious for a held vowel. That span is too wide for a normal sustained tone and suggests octave jumps, tracking overtones, low-frequency artifacts, or unstable f0 estimation.

## 5. Are Octave Jumps Or Low-Frequency Noise Causing False Notes?

Likely yes.

Evidence:

- `00_silence.wav` produced f0 mostly between 65.4 and 101.8 Hz, consistent with low-frequency hum/noise being mapped to pitch bins.
- `01_speaking_voice.wav` ranged from 65.4 to 769.0 Hz, which is suspiciously wide for ordinary speech and suggests tracking errors or octave/overtonal jumps.
- `03_sustained_aaa.wav` ranged from 70.6 to 659.2 Hz, which is inconsistent with a single held vowel.
- `04_pitch_slide.wav` ranged from 65.4 to 712.0 Hz. Some movement is expected, but the lower bound and upper bound suggest possible edge artifacts or octave switches.

Because segmentation is jump-based, each octave error can create false notes. Because scoring is nearest-semitone based, those false notes can still look "in tune" if each erroneous f0 lands near some chromatic pitch.

## 6. Should Score/Coaching Be Blocked Unless Validity Checks Pass?

Yes.

The app should not produce a numeric singing score or prescriptive singing exercises unless the input passes basic validity checks. At minimum, the analysis should distinguish:

- non-voice/noise,
- speech,
- too short,
- too quiet/clipped,
- uncertain/low confidence,
- sustained diagnostic tone,
- pitch slide diagnostic,
- analyzable singing melody.

For invalid or uncertain inputs, the response should return diagnostic metrics and a clear status instead of singing coaching. For diagnostic inputs like a sustained vowel or pitch slide, feedback should be constrained to that task. A pitch slide should not receive an "excellent singing" score simply because its frames are near semitone bins.

## 7. Metrics To Add For Analyzable Singing

The system needs explicit validity metrics before note scoring and coaching.

Recommended model-output diagnostics:

- raw voiced probability per frame,
- voiced probability mean, median, percentiles, and histogram,
- pitch softmax max probability per frame,
- pitch entropy or margin between top pitch bins,
- onset and breath raw probabilities,
- fraction of frames near the voiced threshold,
- disagreement between VAD confidence and pitch confidence.

Recommended audio diagnostics:

- RMS/loudness distribution,
- noise floor estimate,
- clipping percentage,
- spectral flatness,
- spectral centroid/bandwidth,
- zero-crossing rate,
- harmonic-to-noise ratio,
- low-frequency energy ratio,
- silence/near-silence duration.

Recommended f0 diagnostics:

- voiced frame percentage,
- longest voiced run,
- number of voiced regions,
- median f0,
- f0 range after outlier trimming,
- low-frequency f0 ratio, especially below likely singing range,
- octave jump rate,
- semitone jump rate,
- f0 stability inside voiced runs,
- pitch confidence-weighted f0 coverage.

Recommended note/onset diagnostics:

- notes per second,
- onset rate,
- median note duration,
- percentage of notes shorter than 200-300 ms,
- fragmentation index for sustained tones,
- gap merge count,
- note confidence aggregated from frame confidence.

Recommended singing-validity metrics:

- analyzable singing probability,
- speech-likeness probability,
- noise/non-voice probability,
- sustained-tone vs melody classification,
- expected task type when known,
- final validity status and reason codes.

## 8. Code Files Likely Needing Changes

Likely immediate inference/postprocessing files:

- `ml_new/inference/coach_inference.py`
  - expose raw model probabilities,
  - compute validity metrics,
  - gate score/coaching,
  - avoid building full singing feedback for invalid inputs.

- `ml_new/inference/algorithms.py`
  - smooth f0 before note segmentation,
  - add octave-jump correction,
  - merge short gaps,
  - raise or contextualize minimum note duration,
  - reduce false note splitting for sustained tones and slides.

- `backend/app/services/ml_inference.py`
  - carry validity/debug fields through API responses.

- `backend/app/services/ml_serialiser.py`
  - serialize validity metrics, confidence summaries, and invalid-input responses.

- `backend/app/schemas/coaching_result.py`
  - add schema fields for validity, confidence, and reason codes if responses are typed there.

Likely frontend display files for a later behavior pass:

- `new_frontend/src/utils/audioAnalysis.ts`
- `new_frontend/src/types.ts`
- `new_frontend/src/components/ResultsView.tsx`
- `new_frontend/src/components/StudioView.tsx`

Likely training/data files for later retraining work:

- `ml_new/data/unified_dataset.py`
- `ml_new/data/extract_all.py`
- `ml_new/training/train_unified.py`
- `ml_new/models/unified_model.py`

Likely test/evaluation additions:

- backend endpoint tests for invalid-input responses,
- inference unit tests around gating and segmentation,
- regression fixtures for the five self-recorded samples,
- an evaluation summary that reports false-positive voice rate, speech rejection, note fragmentation, and confidence calibration.

## 9. Safe Immediate Gating And Postprocessing Fixes

These fixes do not require architecture changes or retraining.

1. Add an `analysis_validity` object

   Include `is_analyzable`, `input_type`, `confidence`, `reason_codes`, and summary metrics. This should be returned even when score/coaching are blocked.

2. Expose raw confidence diagnostics

   Add frame-level or summarized `voiced_prob`, pitch confidence, onset confidence, and breath confidence. This is necessary before threshold tuning.

3. Block score/coaching for invalid or uncertain inputs

   Do not emit singing exercises for likely noise, speech, very low confidence, too-short audio, or unsupported diagnostic tasks. Return a neutral message and metrics instead.

4. Add conservative noise rejection

   Use audio-level checks such as RMS, spectral flatness, HNR, pitch confidence, and low-frequency f0 ratio. The first goal is to stop `00_silence.wav` from receiving any singing score.

5. Add speech/non-singing heuristics

   As a temporary measure, flag likely speech when onset/note density is high, stable sung-note duration is low, pitch movement is speech-like, and no sustained melodic regions are present. This should be treated as a heuristic until trained directly.

6. Smooth f0 before note segmentation

   Apply median filtering or confidence-weighted smoothing to the f0 contour before `segment_notes()`.

7. Add octave correction and outlier rejection

   Suppress implausible frame-to-frame octave jumps unless sustained by nearby frames.

8. Make note segmentation less eager

   Increase minimum note duration for coaching notes, merge tiny gaps, merge very short adjacent notes, and use hysteresis rather than single-frame jumps.

9. Add diagnostic-task caveats

   A sustained vowel and pitch slide should be evaluated with task-specific language. They should not be scored like a complete sung melody.

10. Stop treating nearest-semitone accuracy as global pitch quality

   Keep it as a low-level diagnostic, but do not let it dominate the score without reference melody, exercise type, and validity checks.

## 10. Fixes That Require Retraining Or Dataset Changes

These should not be attempted as quick inference patches.

1. Train with explicit negative examples

   Add fan noise, room noise, silence, appliance hum, speech, breath-only, non-vocal tonal sounds, and microphone handling noise.

2. Add speech-vs-singing labels

   The model needs to learn that voiced speech is not automatically analyzable singing.

3. Add an analyzable-singing head

   A dedicated head for `non_voice`, `speech`, `sustained_voice`, `pitch_slide`, `singing_melody`, and `unknown` would be cleaner than overloading VAD.

4. Improve VAD labels

   Replace or supplement energy-derived labels with human voice annotations, especially for noisy and quiet recordings.

5. Calibrate thresholds on validation data

   Choose VAD, pitch-confidence, onset, and validity thresholds using a held-out validation set that includes real app-like recordings and negative samples.

6. Improve f0 supervision

   Add robust labels and losses for octave errors, unvoiced/noisy frames, and low-confidence pitch regions.

7. Add melody/reference-aware scoring

   `05_twinkle_twinkle.wav` cannot be meaningfully scored as a song without reference melody alignment. Nearest-semitone accuracy is not enough.

8. Train on browser/mobile domain audio

   Since the eventual model should run in browser, training data should include compressed, noisy, and device-recorded examples similar to expected user input.

## Prioritized Fix Plan

### P0: Make Failures Observable

Add summarized raw probabilities and confidence metrics to inference results:

- voiced probability distribution,
- pitch confidence/entropy,
- onset probability distribution,
- f0 jump and octave-jump counts,
- note fragmentation metrics.

Goal: determine whether false positives are high-confidence model errors or threshold/postprocessing errors.

### P1: Add Validity Gate Before Coaching

Add an `analysis_validity` decision before score and coaching text:

- block likely noise/non-voice,
- block or caveat speech,
- block low-confidence analyses,
- classify sustained vowels and slides as diagnostic tasks.

Goal: prevent `00_silence.wav` and `01_speaking_voice.wav` from receiving singing coaching.

### P2: Stabilize F0 And Note Segmentation

Smooth f0, reject octave jumps, merge short notes/gaps, and make onset/note segmentation less frame-reactive.

Goal: stop `03_sustained_aaa.wav` from fragmenting into many notes and false onsets.

### P3: Make Scoring Task-Aware

Separate scoring modes:

- invalid/non-singing: no score,
- sustained vowel: stability/tone-only feedback,
- pitch slide: continuity/range/smoothness feedback,
- melody: note/rhythm scoring only when a reference or intended task exists.

Goal: stop pitch slides and arbitrary inputs from receiving overconfident general singing scores.

### P4: Add Regression Evaluation

Promote the five self-recorded WAV files into a lightweight regression suite with expected behavior:

- silence should be non-voice,
- speech should be speech/non-singing,
- sustained vowel should have low note fragmentation,
- pitch slide should be caveated as a slide,
- Twinkle should require reference-aware scoring for song accuracy.

Goal: prevent these failures from silently returning.

### P5: Retrain With Better Data And Labels

After safe gates and postprocessing are in place, retrain with explicit negative, speech, and browser-like examples plus calibrated validation thresholds.

Goal: move from defensive heuristics to a model that directly predicts whether an input is analyzable singing.

