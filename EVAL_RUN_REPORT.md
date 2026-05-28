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
- Confidence curves remain unavailable because `CoachingResult` still does not expose raw model probabilities.
