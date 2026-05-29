# Raw Model Output Audit: 00_silence

- JSON: `reports/model_output_audit/00_silence/00_silence_raw_summary.json`
- Raw arrays NPZ: `reports/model_output_audit/00_silence/00_silence_raw_outputs.npz`
- Plot: `reports/model_output_audit/00_silence/00_silence_raw_outputs.svg`
- Duration: `6.933s`
- Frames: `694`

## Voiced Probability

- Mean/median/min/max: `0.5649` / `0.5700` / `0.2990` / `0.6726`
- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `99.9%` / `95.5%` / `0.0%` / `0.0%`
- Near default 0.5 threshold (+/-0.05): `30.7%`

## Pitch Confidence

- Max softmax mean/median: `0.1284` / `0.1246`
- Top-2 margin mean/median: `0.0125` / `0.0115`
- Normalized entropy mean/median: `0.6530` / `0.6499`

## F0

- Raw voiced f0 median/range: `84.0007` Hz, `65.4000`-`101.8363` Hz
- Raw trimmed range: `20.2337` Hz
- Raw jumps: octave `0`, semitone `40`
- Smoothed f0 median/range: `84.0007` Hz, `65.4000`-`101.8363` Hz
- Smoothed jumps: octave `0`, semitone `30`

## Onset / Breath

- Onset probability mean/median/max: `0.4165` / `0.4162` / `0.5122`
- Breath probability mean/median/max: `0.1674` / `0.1652` / `0.2784`

## Technique

- Top class: `breathy` (`0.2961`)
- Technique probabilities are marked unreliable and should not drive coaching decisions.
