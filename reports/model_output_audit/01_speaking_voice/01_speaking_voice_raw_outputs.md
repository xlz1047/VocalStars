# Raw Model Output Audit: 01_speaking_voice

- JSON: `reports/model_output_audit/01_speaking_voice/01_speaking_voice_raw_summary.json`
- Raw arrays NPZ: `reports/model_output_audit/01_speaking_voice/01_speaking_voice_raw_outputs.npz`
- Plot: `reports/model_output_audit/01_speaking_voice/01_speaking_voice_raw_outputs.svg`
- Duration: `7.275s`
- Frames: `728`

## Voiced Probability

- Mean/median/min/max: `0.6731` / `0.6153` / `0.3657` / `0.9575`
- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `100.0%` / `83.4%` / `39.6%` / `14.6%`
- Near default 0.5 threshold (+/-0.05): `29.7%`

## Pitch Confidence

- Max softmax mean/median: `0.1974` / `0.1308`
- Top-2 margin mean/median: `0.0382` / `0.0172`
- Normalized entropy mean/median: `0.5645` / `0.6768`

## F0

- Raw voiced f0 median/range: `130.8000` Hz, `65.4000`-`768.9650` Hz
- Raw trimmed range: `646.5655` Hz
- Raw jumps: octave `16`, semitone `31`
- Smoothed f0 median/range: `144.0182` Hz, `65.4000`-`754.3010` Hz
- Smoothed jumps: octave `2`, semitone `13`

## Onset / Breath

- Onset probability mean/median/max: `0.3312` / `0.3785` / `0.5570`
- Breath probability mean/median/max: `0.3494` / `0.3046` / `0.7470`

## Technique

- Top class: `spoken` (`0.6495`)
- Technique probabilities are marked unreliable and should not drive coaching decisions.
