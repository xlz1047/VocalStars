# Raw Model Output Audit: 05_twinkle_twinkle

- JSON: `reports/model_output_audit/05_twinkle_twinkle/05_twinkle_twinkle_raw_summary.json`
- Raw arrays NPZ: `reports/model_output_audit/05_twinkle_twinkle/05_twinkle_twinkle_raw_outputs.npz`
- Plot: `reports/model_output_audit/05_twinkle_twinkle/05_twinkle_twinkle_raw_outputs.svg`
- Duration: `8.555s`
- Frames: `856`

## Voiced Probability

- Mean/median/min/max: `0.8199` / `0.9059` / `0.2543` / `0.9669`
- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `99.3%` / `93.3%` / `76.4%` / `54.2%`
- Near default 0.5 threshold (+/-0.05): `8.5%`

## Pitch Confidence

- Max softmax mean/median: `0.3110` / `0.3668`
- Top-2 margin mean/median: `0.0625` / `0.0514`
- Normalized entropy mean/median: `0.3840` / `0.2845`

## F0

- Raw voiced f0 median/range: `171.2675` Hz, `65.4000`-`423.3371` Hz
- Raw trimmed range: `153.7730` Hz
- Raw jumps: octave `8`, semitone `15`
- Smoothed f0 median/range: `161.6550` Hz, `65.4000`-`237.5899` Hz
- Smoothed jumps: octave `0`, semitone `8`

## Onset / Breath

- Onset probability mean/median/max: `0.2095` / `0.1449` / `0.5749`
- Breath probability mean/median/max: `0.2774` / `0.2051` / `0.6490`

## Technique

- Top class: `breathy` (`0.3456`)
- Technique probabilities are marked unreliable and should not drive coaching decisions.
