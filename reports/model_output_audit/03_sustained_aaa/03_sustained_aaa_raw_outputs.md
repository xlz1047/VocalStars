# Raw Model Output Audit: 03_sustained_aaa

- JSON: `reports/model_output_audit/03_sustained_aaa/03_sustained_aaa_raw_summary.json`
- Raw arrays NPZ: `reports/model_output_audit/03_sustained_aaa/03_sustained_aaa_raw_outputs.npz`
- Plot: `reports/model_output_audit/03_sustained_aaa/03_sustained_aaa_raw_outputs.svg`
- Duration: `6.933s`
- Frames: `694`

## Voiced Probability

- Mean/median/min/max: `0.6936` / `0.6864` / `0.3962` / `0.8915`
- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `100.0%` / `99.0%` / `43.5%` / `0.0%`
- Near default 0.5 threshold (+/-0.05): `2.7%`

## Pitch Confidence

- Max softmax mean/median: `0.2113` / `0.1865`
- Top-2 margin mean/median: `0.0438` / `0.0319`
- Normalized entropy mean/median: `0.5465` / `0.5659`

## F0

- Raw voiced f0 median/range: `138.5778` Hz, `70.6359`-`659.1908` Hz
- Raw trimmed range: `77.6632` Hz
- Raw jumps: octave `27`, semitone `31`
- Smoothed f0 median/range: `288.0364` Hz, `79.2861`-`659.1908` Hz
- Smoothed jumps: octave `1`, semitone `4`

## Onset / Breath

- Onset probability mean/median/max: `0.3743` / `0.3965` / `0.5019`
- Breath probability mean/median/max: `0.5480` / `0.5804` / `0.7417`

## Technique

- Top class: `vocal_fry` (`0.9615`)
- Technique probabilities are marked unreliable and should not drive coaching decisions.
