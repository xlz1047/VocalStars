# Raw Model Output Audit: 04_pitch_slide

- JSON: `reports/model_output_audit/04_pitch_slide/04_pitch_slide_raw_summary.json`
- Raw arrays NPZ: `reports/model_output_audit/04_pitch_slide/04_pitch_slide_raw_outputs.npz`
- Plot: `reports/model_output_audit/04_pitch_slide/04_pitch_slide_raw_outputs.svg`
- Duration: `6.891s`
- Frames: `690`

## Voiced Probability

- Mean/median/min/max: `0.8554` / `0.9376` / `0.3020` / `0.9573`
- Frames >= 0.3 / 0.5 / 0.7 / 0.9: `100.0%` / `92.9%` / `81.3%` / `78.6%`
- Near default 0.5 threshold (+/-0.05): `8.3%`

## Pitch Confidence

- Max softmax mean/median: `0.3401` / `0.3914`
- Top-2 margin mean/median: `0.0833` / `0.0841`
- Normalized entropy mean/median: `0.3562` / `0.2722`

## F0

- Raw voiced f0 median/range: `215.7836` Hz, `65.4000`-`711.9655` Hz
- Raw trimmed range: `137.5797` Hz
- Raw jumps: octave `2`, semitone `8`
- Smoothed f0 median/range: `215.7836` Hz, `65.4000`-`219.9785` Hz
- Smoothed jumps: octave `1`, semitone `4`

## Onset / Breath

- Onset probability mean/median/max: `0.1824` / `0.1274` / `0.5539`
- Breath probability mean/median/max: `0.2663` / `0.2797` / `0.6814`

## Technique

- Top class: `straight` (`0.8582`)
- Technique probabilities are marked unreliable and should not drive coaching decisions.
