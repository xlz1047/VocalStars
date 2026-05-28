# VocalStars Evaluation Harness

This harness evaluates local diagnostic audio with the existing model inference entrypoint:

```text
ml_new.inference.coach_inference.analyse_recording()
```

It does not modify model architecture, retrain, or change frontend/backend behavior.

## Runtime

Use the project ML virtualenv:

```bash
ml/.venv/bin/python scripts/eval/evaluate_audio.py samples/03_sustained_aaa.m4a
```

The default checkpoint is:

```text
ml_new/checkpoints/unified/best.pt
```

## Evaluate One File

```bash
ml/.venv/bin/python scripts/eval/evaluate_audio.py \
  samples/03_sustained_aaa.m4a \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Outputs are written to:

```text
reports/eval/self_recorded/<sample_name>/
```

Each sample directory contains:

- `<sample>.json`: raw serialized result plus summary metrics
- `<sample>.md`: human-readable report
- `<sample>_plots.svg`: waveform, voiced/breath/onset timelines, and f0 curve

## Evaluate All Self-Recorded Samples

The batch script prefers `samples/self_recorded/` if it exists. If not, it falls back to `samples/`, which is where the current `.m4a` diagnostic files are located.

```bash
ml/.venv/bin/python scripts/eval/evaluate_self_recorded.py \
  --samples-dir samples \
  --output-dir reports/eval/self_recorded \
  --checkpoint ml_new/checkpoints/unified/best.pt \
  --device cpu
```

Batch outputs also include:

- `reports/eval/self_recorded/summary.json`
- `reports/eval/self_recorded/summary.md`

## M4A Support

The local `librosa`/`audioread` environment could not decode `.m4a` directly during setup. The harness therefore attempts to convert `.m4a`, `.aac`, and `.mp4` files to WAV using macOS `afconvert` when available.

Equivalent manual conversion:

```bash
afconvert -f WAVE -d LEI16@16000 input.m4a output.wav
```

If `afconvert` is unavailable or fails for a particular recording, convert the file to WAV manually and pass the `.wav` to `evaluate_audio.py`. In this workspace, the current `.m4a` samples triggered a CoreAudio conversion error, so a manual conversion tool such as ffmpeg may be needed:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 output.wav
```

## Known Output Limitations

The current `CoachingResult` exposes:

- `pitch_hz`
- thresholded `voiced`
- thresholded `breath_frames`
- thresholded `onset_frames`
- note, vibrato, and voice-quality summaries when available

It does not expose:

- raw per-frame pitch confidence
- raw per-frame VAD/voicing probability
- raw per-frame breath probability
- raw per-frame onset probability

For that reason, the plot includes waveform, voiced/unvoiced timeline, breath/onset timelines, and f0 curve, but the confidence curve is documented as unavailable.
