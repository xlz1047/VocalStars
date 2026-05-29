# Evaluation Run Report

Run date: 2026-05-28

## Commands run

Read the audit:

```bash
sed -n '1,260p' MODEL_AUDIT.md
```

Checked sample locations and formats:

```bash
find samples/self_recorded -maxdepth 2 -type f -print
find samples -maxdepth 5 -type f \( -iname '*.wav' -o -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.webm' \) -print
file samples/*.m4a
```

Checked runtime/dependencies:

```bash
ml/.venv/bin/python - <<'PY'
import importlib.util
for m in ['librosa','soundfile','matplotlib','numpy','torch','scipy','parselmouth']:
    print(m, bool(importlib.util.find_spec(m)))
PY
which afconvert
```

Syntax-checked the harness:

```bash
ml/.venv/bin/python -m py_compile scripts/eval/evaluate_audio.py scripts/eval/evaluate_self_recorded.py
```

Ran all self-recorded samples:

```bash
ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Verified the harness on a known WAV fixture:

```bash
ml/.venv/bin/python scripts/eval/evaluate_audio.py \
  backend/audio_uploads/3fb429c74d3248a2b3b80571cef355e6.wav \
  --output-dir reports/eval/smoke \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

## Entrypoint identified

The correct existing model inference entrypoint is:

```text
ml_new.inference.coach_inference.analyse_recording()
```

The harness calls that entrypoint directly with:

```text
ml_new/checkpoints/unified/best.pt
```

## Sample discovery

The requested `samples/self_recorded/` directory was not present in this checkout. The diagnostic files were present directly under `samples/`:

- `samples/00_silence.m4a`
- `samples/01_speaking_voice.m4a`
- `samples/03_sustained_aaa.m4a`
- `samples/04_pitch_slide.m4a`
- `samples/05_twinkle_twinkle.m4a`

The batch script defaults to `samples/self_recorded/` when it exists and otherwise falls back to `samples/`.

## Dependency observations

- `ml/.venv` has `librosa`, `soundfile`, `numpy`, `torch`, `scipy`, and `parselmouth`.
- `matplotlib` is not installed, so the harness writes dependency-light SVG plots manually.
- `ffmpeg` is not installed.
- macOS `afconvert` is installed, but conversion failed for the current `.m4a` files.

## Errors

All five `.m4a` samples failed before model inference because local conversion failed:

```text
Error: ExtAudioFileSetProperty ('cfmt') failed ('fmt?')
```

Direct `librosa` loading also failed with `NoBackendError` for these `.m4a` files. Because the audio could not be decoded or converted, the model did not run on the five self-recorded samples.

The harness itself was verified on a WAV file and successfully generated JSON, markdown, and SVG plot artifacts.

## Outputs observed

Self-recorded batch output directory:

```text
reports/eval/self_recorded/
```

Generated self-recorded files:

- `reports/eval/self_recorded/summary.json`
- `reports/eval/self_recorded/summary.md`
- `reports/eval/self_recorded/00_silence/00_silence.json`
- `reports/eval/self_recorded/00_silence/00_silence.md`
- `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice.json`
- `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice.md`
- `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa.json`
- `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa.md`
- `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide.json`
- `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide.md`
- `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle.json`
- `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle.md`

Plots were not generated for the five `.m4a` samples because waveform decoding failed before inference. The JSON and markdown files document the conversion error.

Smoke-test output directory:

```text
reports/eval/smoke/
```

The WAV smoke test produced:

- JSON output
- markdown report
- SVG plot with waveform, voiced/breath/onset timelines, and f0 curve

## Expected behavior check by sample

| Sample | Expected behavior | Result |
| --- | --- | --- |
| `00_silence` | Fan/background noise only; should be treated as non-voice. | Not evaluated. M4A conversion failed before inference. |
| `01_speaking_voice` | Speech, not singing. | Not evaluated. M4A conversion failed before inference. |
| `03_sustained_aaa` | Held sung vowel. | Not evaluated. M4A conversion failed before inference. |
| `04_pitch_slide` | Sung pitch slide. | Not evaluated. M4A conversion failed before inference. |
| `05_twinkle_twinkle` | Short sung melody. | Not evaluated. M4A conversion failed before inference. |

## Missing or unusable model outputs

Even when audio is decodable, the current `CoachingResult` does not expose these cleanly:

- raw per-frame pitch confidence
- raw per-frame VAD/voicing probability
- raw per-frame breath probability
- raw per-frame onset probability

The harness therefore plots:

- waveform
- thresholded voiced/unvoiced timeline
- thresholded breath timeline
- thresholded onset timeline
- f0/pitch curve

The confidence plot is documented as unavailable in each output.

Also important from `MODEL_AUDIT.md`:

- technique/timbre outputs are not reliable enough for coaching decisions yet
- breath and onset outputs are thresholded and trained from heuristic labels
- pitch accuracy is nearest-semitone accuracy, not song-reference melody accuracy

## Next step to evaluate these exact samples

Convert each `.m4a` to WAV with a reliable decoder, then rerun the batch script on the WAV files. Example:

```bash
ffmpeg -i samples/03_sustained_aaa.m4a -ac 1 -ar 16000 samples/03_sustained_aaa.wav
ml/.venv/bin/python scripts/eval/evaluate_audio.py samples/03_sustained_aaa.wav
```

## WAV rerun

Run date: 2026-05-28

The five manually converted WAV files were present and decodable:

- `samples/00_silence.wav`
- `samples/01_speaking_voice.wav`
- `samples/03_sustained_aaa.wav`
- `samples/04_pitch_slide.wav`
- `samples/05_twinkle_twinkle.wav`

Commands run:

```bash
file samples/00_silence.wav samples/01_speaking_voice.wav \
  samples/03_sustained_aaa.wav samples/04_pitch_slide.wav \
  samples/05_twinkle_twinkle.wav

ml/.venv/bin/python - <<'PY'
from pathlib import Path
import librosa
for p in [
    Path('samples/00_silence.wav'),
    Path('samples/01_speaking_voice.wav'),
    Path('samples/03_sustained_aaa.wav'),
    Path('samples/04_pitch_slide.wav'),
    Path('samples/05_twinkle_twinkle.wav'),
]:
    y, sr = librosa.load(str(p), sr=16000, mono=True, duration=0.25)
    print(f'{p}: ok sr={sr} frames={len(y)}')
PY

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

All five WAV files loaded successfully as 48 kHz mono PCM WAV and were resampled by `librosa` to 16 kHz for model inference. The batch run ignored `.m4a` files via `--extensions .wav`.

Artifacts generated:

- `reports/eval/self_recorded/summary.json`
- `reports/eval/self_recorded/summary.md`
- `reports/eval/self_recorded/00_silence/00_silence.json`
- `reports/eval/self_recorded/00_silence/00_silence.md`
- `reports/eval/self_recorded/00_silence/00_silence_plots.svg`
- `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice.json`
- `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice.md`
- `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice_plots.svg`
- `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa.json`
- `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa.md`
- `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa_plots.svg`
- `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide.json`
- `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide.md`
- `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide_plots.svg`
- `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle.json`
- `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle.md`
- `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle_plots.svg`

Inference metadata for every WAV sample:

- `inference_mode`: `checkpoint`
- `checkpoint_path_used`: `ml_new/checkpoints/unified/best.pt`
- `device_used`: `cpu`
- `model_stack_used`: `ml_new`

No WAV file failed. There were no tracebacks to classify under audio loading, preprocessing, checkpoint loading, model forward pass, result formatting, or plotting/report generation.

### WAV sample results

| Sample | Inference | Duration | Voiced frames | F0 behavior | Notes / onsets | Score | Expected behavior match |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| `00_silence` | succeeded, checkpoint | 6.933 s | 95.5% | Mean 80.5 Hz, range 65.4-101.8 Hz | 19 notes, 1 onset | 71 | No. This is a severe false positive: fan/background noise was treated as mostly voiced singing. |
| `01_speaking_voice` | succeeded, checkpoint | 7.275 s | 83.4% | Mean 161.3 Hz, range 65.4-769.0 Hz | 14 notes, 11 onsets | 71 | Partially as voiced audio, but not as speech. The model generated singing-style coaching for speech. |
| `03_sustained_aaa` | succeeded, checkpoint | 6.933 s | 99.0% | Mean 120.4 Hz, range 70.6-659.2 Hz | 14 notes, 6 onsets | 71 | Partial. It detected voice, but a held vowel should not fragment into 14 notes / 6 onsets with such a wide f0 range. |
| `04_pitch_slide` | succeeded, checkpoint | 6.891 s | 92.9% | Mean 209.9 Hz, range 65.4-712.0 Hz | 3 notes, 3 onsets | 91 | Mostly yes for moving voiced f0. The high score and no issues may be over-generous for a diagnostic slide. |
| `05_twinkle_twinkle` | succeeded, checkpoint | 8.555 s | 93.3% | Mean 169.4 Hz, range 65.4-423.3 Hz | 12 notes, 10 onsets | 78 | Mostly yes for multi-note singing. It detected multiple events, but scoring/coaching remains nearest-semitone based, not song-reference based. |

### Generated feedback by sample

`00_silence`:

- Summary: Good foundation — pitch 100%, avg phrase 6.8 s. The issues below will make a clear difference.
- Issues:
  - Specific notes are consistently off: E2 (+44 cents, sharp), C2 (+34 cents, sharp).
  - Breathy voice quality detected (HNR -2 dB).
  - Vocal instability detected (jitter 7.6%, shimmer 5.9%).
- Exercises:
  - Relax jaw / breath lower to prevent overshooting.
  - Hum "mmm" into "mah" for cord closure.
  - Rest, hydrate, and warm up with lip trills.
- Assessment: unusable for this sample. Silence/noise should not receive singing feedback.

`01_speaking_voice`:

- Summary: Good foundation — pitch 100%, avg phrase 3.1 s. The issues below will make a clear difference.
- Issues:
  - F2 reported flat.
  - Slightly airy tone.
  - Vocal instability.
  - Phrases average only 3.1 s.
- Exercises:
  - Pitch centering cue.
  - Staccato vowel exercises.
  - Rest/hydration/lip trills.
  - Diaphragmatic breathing.
- Assessment: voiced detection is plausible, but speech is not recognized as non-singing.

`03_sustained_aaa`:

- Summary: Good foundation — pitch 100%, avg phrase 6.9 s. The issues below will make a clear difference.
- Issues:
  - C#3, D3, and D2 reported off.
  - Breathy voice quality.
  - Vocal instability.
  - 4 sustained notes detected but no vibrato found.
- Exercises:
  - Pitch centering cue.
  - Hum "mmm" into "mah".
  - Rest/hydration/lip trills.
  - Vibrato practice.
- Assessment: voice is detected, but note/onset segmentation looks too fragmented for a sustained vowel.

`04_pitch_slide`:

- Summary: Excellent singing! Pitch 100% accurate, voice quality clear.
- Issues: none.
- Exercises: none.
- Assessment: f0 movement was detected. The high score and lack of diagnostic caveats show the current scoring is not designed for pitch-slide exercises.

`05_twinkle_twinkle`:

- Summary: Good foundation — pitch 100%, avg phrase 4.2 s. The issues below will make a clear difference.
- Issues:
  - D3 reported sharp and E2 reported flat.
  - Vocal instability.
  - Phrase length averages 4.2 s.
  - 4 sustained notes detected but no vibrato found.
- Exercises:
  - Pitch centering cue.
  - Rest/hydration/lip trills.
  - Fuller breath before each phrase.
  - Vibrato practice.
- Assessment: the model detected a multi-event sung melody, but it is not comparing against the actual Twinkle Twinkle melody/timing.

### WAV rerun conclusions

- The new inference debug fields confirm the harness used the `ml_new` checkpoint path, not fallback.
- The silence sample is the most important failure: background/fan noise was classified as mostly voiced with high pitch accuracy.
- Speech is treated as analyzable singing and receives singing exercises.
- Sustained vowel segmentation appears unstable.
- Pitch slide and Twinkle produce plausible f0/event plots, but scoring remains generic and not exercise/song aware.
- Confidence curves remain unavailable because `CoachingResult` now exposes summary diagnostics, not frame-level raw probability arrays.

## P0 diagnostics

Run date: 2026-05-28

Implemented P0 observability only. No retraining, model architecture changes, validity gating, scoring changes, coaching changes, or note segmentation changes were made.

Commands run:

```bash
ml/.venv/bin/python -m py_compile \
  ml_new/inference/coach_inference.py \
  scripts/eval/evaluate_audio.py \
  scripts/eval/evaluate_self_recorded.py \
  backend/app/services/ml_inference.py \
  backend/app/schemas/coaching_result.py

ml/.venv/bin/python -m pytest backend/tests/test_ml_endpoint.py

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Verification:

- `py_compile`: passed.
- `backend/tests/test_ml_endpoint.py`: 6 passed, 2 warnings.
- WAV evaluation: all 5 samples succeeded.

Artifacts regenerated:

- `reports/eval/self_recorded/summary.json`
- `reports/eval/self_recorded/summary.md`
- Per-sample JSON, markdown, and SVG plot files under `reports/eval/self_recorded/<sample>/`.

New diagnostics exposed in per-sample JSON/markdown:

- voiced probability summary: mean, median, min, max, percentiles, and near-threshold fraction.
- pitch confidence summary from pitch-logit softmax: max probability, top1/top2 margin, entropy, normalized entropy.
- onset and breath probability summaries.
- f0 median, full range, 5-95% trimmed range, and low-frequency f0 ratio.
- octave/semitone jump counts and rates.
- note fragmentation metrics: notes per second, notes per voiced second, median note duration, and short-note ratios.

P0 diagnostic summary:

| Sample | VAD prob mean | Near VAD threshold | Pitch conf mean | Pitch margin mean | Norm entropy mean | Onset prob mean | Breath prob mean | Low-f0 ratio | Octave jumps | Semitone jumps | Notes/s | Short notes <300ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `00_silence` | 0.565 | 0.307 | 0.128 | 0.013 | 0.653 | 0.417 | 0.167 | 0.268 | 0 | 40 | 2.738 | 0.684 |
| `01_speaking_voice` | 0.673 | 0.297 | 0.197 | 0.038 | 0.565 | 0.331 | 0.349 | 0.135 | 14 | 32 | 1.923 | 0.571 |
| `03_sustained_aaa` | 0.694 | 0.027 | 0.211 | 0.044 | 0.547 | 0.374 | 0.548 | 0.330 | 23 | 31 | 2.017 | 0.500 |
| `04_pitch_slide` | 0.855 | 0.083 | 0.340 | 0.083 | 0.356 | 0.182 | 0.266 | 0.041 | 2 | 11 | 0.435 | 0.667 |
| `05_twinkle_twinkle` | 0.820 | 0.085 | 0.311 | 0.063 | 0.384 | 0.209 | 0.277 | 0.051 | 4 | 15 | 1.402 | 0.417 |

Observations:

- `00_silence` is not a high-confidence pitch case. Its mean pitch max-softmax probability is only 0.128, mean top1/top2 margin is 0.013, and normalized entropy is 0.653. The VAD probability is only moderately above threshold, with 30.7% of frames near the threshold. This supports a near-term confidence/threshold/gating fix once P1 begins.
- `00_silence` still crosses the voiced threshold for 95.5% of frames, so thresholded downstream code continues to treat the noise as singing. P0 intentionally did not change that behavior.
- `01_speaking_voice` and `03_sustained_aaa` both show octave jumps and high semitone-jump counts. This supports the previous suspicion that octave/f0 instability is contributing to false notes and fragmentation.
- `03_sustained_aaa` has 23 octave jumps and 31 semitone jumps despite being a held vowel, which is strong evidence that the f0 contour needs confidence-aware smoothing/outlier handling before note segmentation.
- `04_pitch_slide` and `05_twinkle_twinkle` show higher VAD probability and pitch confidence than silence/speech/sustained-vowel failure cases, but they still have octave jumps and short-note fragmentation.
- Frame-level confidence curves remain unavailable in the public evaluation plots. The current P0 change exposes summary diagnostics first, as requested.

## P1 analysis validity gate

Run date: 2026-05-28

Implemented P1 as a postprocessing/gating layer only. No retraining, model architecture changes, model-output removal, frontend UI changes, or note segmentation changes were made.

Commands run:

```bash
ml/.venv/bin/python -m py_compile \
  ml_new/inference/coach_inference.py \
  scripts/eval/evaluate_audio.py \
  scripts/eval/evaluate_self_recorded.py \
  backend/app/services/ml_inference.py \
  backend/app/schemas/coaching_result.py

ml/.venv/bin/python -m pytest backend/tests/test_ml_endpoint.py

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Verification:

- `py_compile`: passed.
- `backend/tests/test_ml_endpoint.py`: 6 passed, 2 warnings.
- WAV evaluation: all 5 samples succeeded.

New response field:

- `analysis_validity.is_analyzable`
- `analysis_validity.input_type`
- `analysis_validity.confidence`
- `analysis_validity.reason_codes`
- `analysis_validity.summary_metrics`

The gate preserves raw model outputs, frame arrays, notes, diagnostics, and plots for inspection. It blocks normal singing score/coaching for non-analyzable and diagnostic inputs by replacing the public coaching text with a neutral message, empty issues, empty exercises, and `technique=not_applicable`.

P1 rerun summary:

| Sample | Input type | Full-song eligible | Full-song score | Diagnostic score | Score status | Summary | Regression expectation |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `00_silence` | `no_voice_or_noise` | false | null | null | `no_analyzable_singing` | No analyzable singing was detected. | PASS: not analyzable singing. |
| `01_speaking_voice` | `speech_like_or_non_singing` | false | null | null | `speech_or_non_singing_no_score` | This sounds like speech or non-singing voice, so singing coaching was not generated. | PASS: did not receive normal singing coaching. |
| `03_sustained_aaa` | `diagnostic_sustained_tone` | false | null | 54 | `diagnostic_sustained_tone_only` | This sounds like a sustained-tone diagnostic, so full-song singing coaching was not generated. | PASS: not treated as a full melody. |
| `04_pitch_slide` | `diagnostic_pitch_slide` | false | null | 81 | `diagnostic_pitch_slide_only` | This sounds like a pitch-slide diagnostic, so full-song singing coaching was not generated. | PASS: no longer receives "excellent singing" as a normal song. |
| `05_twinkle_twinkle` | `analyzable_singing` | true | 78 | null | `full_song_score_available_no_reference_melody` | Good foundation... not a reference melody. | PASS: analyzable with reference-melody caveat. |

Validity reason codes:

- `00_silence`: `very_low_audio_rms`, `low_pitch_confidence`, `high_pitch_entropy`, `voiced_probabilities_near_threshold`
- `01_speaking_voice`: `speech_like_fragmentation`, `low_pitch_confidence`, `frequent_octave_jumps`
- `03_sustained_aaa`: `continuous_voicing`, `limited_trimmed_f0_range`, `fragmented_f0_tracking`
- `04_pitch_slide`: `continuous_voicing`, `wide_f0_movement`, `few_note_events`
- `05_twinkle_twinkle`: `passes_current_postprocessing_checks`

Important behavior changes observed:

- `00_silence` still has raw model artifacts such as voiced frames, f0, note segments, and raw model technique metadata in the JSON for debugging, but the public coaching output no longer treats it as singing.
- `01_speaking_voice` no longer receives note-specific pitch advice, technique coaching, vibrato feedback, or exercises.
- `03_sustained_aaa` and `04_pitch_slide` are preserved as diagnostic recordings rather than invalid noise, but they do not receive full-song scoring/coaching.
- `05_twinkle_twinkle` remains analyzable singing. Its summary now includes a caveat that the score is based on detected pitch/timing features rather than reference-melody alignment.

### P1 score semantics refinement

Run date: 2026-05-28

The first P1 gate correctly blocked full-song coaching for diagnostic inputs, but returning `score=0` made diagnostic recordings look like poor performances. The result schema/reporting now separates score meanings:

- `full_song_score`: existing full-song score when `analyzable_singing`; otherwise `null`.
- `diagnostic_score`: diagnostic-only score for `diagnostic_sustained_tone` or `diagnostic_pitch_slide`; otherwise `null`.
- `score_status`: reason/status for the score fields.
- `score_caveat`: caveat text when a score is not a full reference-melody score.

Commands rerun:

```bash
ml/.venv/bin/python -m py_compile \
  ml_new/inference/coach_inference.py \
  scripts/eval/evaluate_audio.py \
  scripts/eval/evaluate_self_recorded.py \
  backend/app/services/ml_inference.py \
  backend/app/schemas/coaching_result.py \
  backend/tests/test_ml_endpoint.py

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu

ml/.venv/bin/python -m pytest backend/tests/test_ml_endpoint.py
```

Verification:

- `py_compile`: passed.
- WAV evaluation: all 5 samples succeeded.
- `backend/tests/test_ml_endpoint.py`: 6 passed, 2 warnings.

Updated score semantics after rerun:

| Sample | Input type | Full-song eligible | Full-song score | Diagnostic score | Score status | Summary | Regression expectation |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `00_silence` | `no_voice_or_noise` | false | null | null | `no_analyzable_singing` | No analyzable singing was detected. | PASS |
| `01_speaking_voice` | `speech_like_or_non_singing` | false | null | null | `speech_or_non_singing_no_score` | This sounds like speech or non-singing voice, so singing coaching was not generated. | PASS |
| `03_sustained_aaa` | `diagnostic_sustained_tone` | false | null | 54 | `diagnostic_sustained_tone_only` | This sounds like a sustained-tone diagnostic, so full-song singing coaching was not generated. | PASS |
| `04_pitch_slide` | `diagnostic_pitch_slide` | false | null | 81 | `diagnostic_pitch_slide_only` | This sounds like a pitch-slide diagnostic, so full-song singing coaching was not generated. | PASS |
| `05_twinkle_twinkle` | `analyzable_singing` | true | 78 | null | `full_song_score_available_no_reference_melody` | Good foundation... not a reference melody. | PASS |

## P2 f0 stabilization and note segmentation

Run date: 2026-05-28

Implemented P2 as inference/postprocessing only. No retraining or model architecture changes were made.

Changes:

- Added local f0 cleanup before coaching-note segmentation.
- Rejected isolated f0 spikes when neighboring frames agree.
- Corrected likely octave-like jumps when octave-shifted candidates are closer to the local contour.
- Applied a short 5-frame median filter inside voiced runs to remove burrs without flattening pitch slides.
- Raised coaching-note minimum duration to 200 ms.
- Merged adjacent notes separated by gaps <= 80 ms when pitch distance is <= 1 semitone.
- Preserved raw frame-level `pitch_hz`; only coaching-note segmentation uses the cleaned contour.

Commands run:

```bash
ml/.venv/bin/python -m py_compile \
  ml_new/inference/algorithms.py \
  ml_new/inference/coach_inference.py \
  scripts/eval/evaluate_audio.py \
  scripts/eval/evaluate_self_recorded.py \
  backend/app/services/ml_inference.py \
  backend/app/schemas/coaching_result.py \
  backend/tests/test_ml_endpoint.py

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu

ml/.venv/bin/python -m pytest backend/tests/test_ml_endpoint.py
```

Verification:

- `py_compile`: passed.
- WAV evaluation: all 5 samples succeeded.
- `backend/tests/test_ml_endpoint.py`: 6 passed, 2 warnings.

P2 note segmentation before/after:

| Sample | Input type | Raw notes | Coaching notes | Merges | Octave jumps raw->post | f0 stability cents | Fragmentation index | Score status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `00_silence` | `no_voice_or_noise` | 19 | 8 | 5 | 0->0 | 166 | 1.153 | `no_analyzable_singing` |
| `01_speaking_voice` | `speech_like_or_non_singing` | 14 | 9 | 1 | 14->0 | 982 | 1.236 | `speech_or_non_singing_no_score` |
| `03_sustained_aaa` | `diagnostic_sustained_tone` | 14 | 3 | 1 | 23->0 | 811 | 0.432 | `diagnostic_sustained_tone_only` |
| `04_pitch_slide` | `diagnostic_pitch_slide` | 3 | 2 | 0 | 2->1 | 554 | 0.290 | `diagnostic_pitch_slide_only` |
| `05_twinkle_twinkle` | `analyzable_singing` | 12 | 7 | 0 | 4->0 | 664 | 0.818 | `full_song_score_available_no_reference_melody` |

Acceptance check:

- `03_sustained_aaa.wav` no longer fragments into 14 coaching notes. Raw model note segmentation still reports 14 notes for audit, but the postprocessed coaching-note count is 3.
- `00_silence.wav` remains blocked as `no_voice_or_noise`.
- `01_speaking_voice.wav` remains blocked as `speech_like_or_non_singing`.
- `04_pitch_slide.wav` remains a diagnostic pitch slide; postprocessing reduces 3 raw notes to 2 coaching notes but does not flatten it into a sustained tone.

Notes:

- The sustained vowel still has high f0 instability diagnostics, which indicates the underlying f0 contour remains noisy. P2 prevents that noise from exploding into coaching notes, but it does not claim the raw pitch model is fixed.
- The pitch slide diagnostic score changed from 81 to 87 after smoothing because semitone/octave jump artifacts were reduced while the slide's broad f0 movement was preserved.

## P3 task-aware analysis

Run date: 2026-05-28

Implemented P3 as a task_config and task-aware scoring layer. No retraining, model architecture changes, or checkpoint changes were made.

TaskConfig fields:

- `task_type`
- `skill_focus`
- `target`
- `reference`
- `scoring_mode`
- `strictness`

Supported `task_type` values:

- `free_singing`
- `reference_song`
- `sustained_note`
- `pitch_slide`
- `scale`
- `interval`
- `rhythm`
- `breath_control`
- `tone_consistency`

Behavior:

- If no task_config is provided, inference preserves the current validity classification and infers a safe default task type.
- Invalid/noise/speech-like inputs are still blocked before task scoring.
- `free_singing` allows general feedback and includes a caveat that it is not reference-melody scoring.
- `sustained_note` produces no full-song score and computes a diagnostic score from voicing continuity, pitch stability, pitch drift, dropout, and fragmentation.
- `pitch_slide` produces no full-song score and computes a diagnostic score from slide smoothness, direction, range, continuity, and dropout/fragmentation proxies.
- `reference_song` has a placeholder interface and does not fake reference melody scoring.
- `scale` and `interval` have placeholder/provisional evaluators, but return `insufficient_target_info` when target notes are missing.
- `rhythm` uses onset diagnostics provisionally and caveats that a reference beat/timing grid is needed for proper rhythm scoring.

Commands run:

```bash
ml/.venv/bin/python -m py_compile \
  ml_new/inference/coach_inference.py \
  backend/app/api/routers/audio_processing.py \
  backend/app/services/ml_inference.py \
  backend/app/schemas/coaching_result.py \
  backend/tests/test_ml_endpoint.py \
  scripts/eval/evaluate_audio.py \
  scripts/eval/evaluate_self_recorded.py

ml/.venv/bin/python -m pytest backend/tests/test_ml_endpoint.py

ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --extensions .wav \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Verification:

- `py_compile`: passed.
- `backend/tests/test_ml_endpoint.py`: 6 passed, 2 warnings.
- WAV evaluation: all 5 samples succeeded.

P3 task-aware rerun:

| Sample | Provided task_type | Detected input_type | Score status | Full-song score | Diagnostic score | Task-specific summary | Caveats |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `00_silence` | `free_singing` | `no_voice_or_noise` | `no_analyzable_singing` | null | null | No analyzable singing was detected. | Task scoring skipped because input was not analyzable singing. |
| `01_speaking_voice` | `free_singing` | `speech_like_or_non_singing` | `speech_or_non_singing_no_score` | null | null | This sounds like speech or non-singing voice, so singing coaching was not generated. | Task scoring skipped because input was not analyzable singing. |
| `03_sustained_aaa` | `sustained_note` | `diagnostic_sustained_tone` | `diagnostic_sustained_tone_only` | null | 95 | Sustained-note diagnostic complete; full-song scoring was not generated. | Diagnostic sustained-note score only; no reference melody was evaluated. |
| `04_pitch_slide` | `pitch_slide` | `diagnostic_pitch_slide` | `diagnostic_pitch_slide_only` | null | 89 | Pitch-slide diagnostic complete; full-song scoring was not generated. | Diagnostic pitch-slide score only; no reference melody was evaluated. |
| `05_twinkle_twinkle` | `free_singing` | `analyzable_singing` | `free_singing_general_feedback` | 78 | null | Good foundation... not a reference melody. | Score is based on detected pitch and timing features, not a reference melody. |

New output fields:

- `task_config`
- `task_analysis.task_type`
- `task_analysis.detected_input_type`
- `task_analysis.status`
- `task_analysis.summary`
- `task_analysis.caveats`
- `task_analysis.scoring_components`

Backend/API notes:

- `/api/audio/analyze-with-ml` accepts optional `task_config` as a JSON form field.
- The endpoint test now verifies that `task_config` is parsed and forwarded into `analyse_recording`.
- Frontend UI was not changed.

## P4 regression expectations

Run date: 2026-05-28

Implemented a lightweight behavior-level regression suite for the five self-recorded WAV samples. The suite intentionally does not lock exact numeric scores yet; it checks validity, task/scoring semantics, caveats, and that invalid inputs do not receive user-facing coaching.

Script:

```text
scripts/eval/check_regression_expectations.py
```

Commands run:

```bash
ml/.venv/bin/python -m py_compile scripts/eval/check_regression_expectations.py

ml/.venv/bin/python scripts/eval/check_regression_expectations.py
```

Result: all checks passed.

| Sample | Result | Behavioral expectations checked |
| --- | --- | --- |
| `00_silence` | PASS | Detected `no_voice_or_noise`; no full-song score; no diagnostic score; no issues/exercises; not analyzable singing. |
| `01_speaking_voice` | PASS | Detected `speech_like_or_non_singing`; no full-song score; no singing exercises; not analyzable singing. |
| `03_sustained_aaa` | PASS | Sustained-note diagnostic mode observed; no full-song score; diagnostic score field present; coaching note count is 3; not treated as full-song melody. |
| `04_pitch_slide` | PASS | Pitch-slide diagnostic mode observed; no full-song score; diagnostic score field present; no generic “excellent singing” full-song praise. |
| `05_twinkle_twinkle` | PASS | Free/analyzable singing observed; full-song score is absent or numeric; reference-melody caveat present. |

The checker reads the latest JSON artifacts under `reports/eval/self_recorded/`. If a required successful artifact is missing, it runs the existing evaluation harness for that WAV file with the sample's explicit `task_config`.

## M1 baseline comparison

Run date: 2026-05-29

Created `M1_BASELINE_COMPARISON.md` to compare M0 checkpoint raw VAD/f0 outputs against a `librosa.pyin` DSP baseline on the five self-recorded WAV samples. No retraining, model architecture changes, scoring tuning, app behavior changes, or P4 regression expectation changes were made.

Command run:

```bash
ml/.venv/bin/python -m py_compile scripts/eval/compare_baseline_outputs.py

ml/.venv/bin/python scripts/eval/compare_baseline_outputs.py \
  --samples-dir samples \
  --m0-dir reports/model_output_audit \
  --output-dir reports/baseline_comparison
```

Outputs:

- `M1_BASELINE_COMPARISON.md`
- `reports/baseline_comparison/summary.json`
- Per-sample JSON and SVG plots under `reports/baseline_comparison/<sample>/`

M1 summary:

| Sample | Checkpoint f0 coverage | Baseline f0 coverage | Checkpoint octave/semitone jumps | Baseline octave/semitone jumps | M1 conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `00_silence` | 95.5% | 48.7% | 0 / 40 | 0 / 0 | Baseline rejects noise better, but still detects too much f0 to trust alone. |
| `01_speaking_voice` | 83.4% | 62.0% | 16 / 31 | 0 / 0 | Baseline f0 is less jumpy, but speech still needs singing-validity gating. |
| `03_sustained_aaa` | 99.0% | 88.8% | 27 / 31 | 0 / 0 | Baseline handles sustained f0 much more cleanly. |
| `04_pitch_slide` | 92.9% | 86.8% | 2 / 8 | 0 / 0 | Both preserve directional movement; baseline is cleaner on jumps. |
| `05_twinkle_twinkle` | 93.3% | 82.1% | 8 / 15 | 0 / 0 | Mixed: baseline removes jumps, checkpoint preserves more continuous melody coverage. |

Recommendation from M1:

- Do not use checkpoint-only for raw VAD/f0 yet; it is too permissive on noise and jumpy on f0 argmax.
- Do not use baseline-only as the product path; it lacks the model's learned onset/breath/technique interfaces and is not task-aware.
- Use a hybrid/ensemble near term: checkpoint outputs gated by conservative DSP sanity checks, confidence/consistency metrics, and task-aware validity rules.
