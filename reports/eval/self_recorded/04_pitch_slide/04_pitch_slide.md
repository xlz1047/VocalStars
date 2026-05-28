# Evaluation: 04_pitch_slide

- Input: `samples/04_pitch_slide.wav`
- Audio used for inference: `samples/04_pitch_slide.wav`
- Converted: `False`
- Score: `91`
- Summary: Excellent singing! Pitch 100% accurate, voice quality clear.

## Artifacts

- JSON: `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide.json`
- Plot: `reports/eval/self_recorded/04_pitch_slide/04_pitch_slide_plots.svg`

## Key Metrics

- `duration_s`: `6.891`
- `audio_rms`: `0.055145636200904846`
- `voiced_frame_ratio`: `0.9289855072463769`
- `voiced_duration_s`: `6.41`
- `mean_f0_hz`: `209.89610290527344`
- `median_f0_hz`: `215.78355407714844`
- `min_f0_hz`: `65.4000015258789`
- `max_f0_hz`: `711.9654541015625`
- `pitch_accuracy`: `1.0`
- `pitch_drift_cents`: `-0.16903279721736908`
- `breath_count`: `3`
- `onset_count`: `3`
- `onset_clarity`: `0.40119418501853943`
- `technique`: `straight`
- `technique_confidence`: `0.31887099146842957`
- `note_count`: `3`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Issues

- None reported.

## Exercises

- None reported.

## Confidence Curve Availability

analyse_recording() returns thresholded voiced/breath/onset arrays, but not raw per-frame confidence probabilities.
