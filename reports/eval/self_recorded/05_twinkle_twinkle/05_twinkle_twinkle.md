# Evaluation: 05_twinkle_twinkle

- Input: `samples/05_twinkle_twinkle.wav`
- Audio used for inference: `samples/05_twinkle_twinkle.wav`
- Converted: `False`
- Score: `78`
- Summary: Good foundation — pitch 100%, avg phrase 4.2 s. The issues below will make a clear difference.

## Artifacts

- JSON: `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle.json`
- Plot: `reports/eval/self_recorded/05_twinkle_twinkle/05_twinkle_twinkle_plots.svg`

## Key Metrics

- `duration_s`: `8.555`
- `audio_rms`: `0.020151298493146896`
- `voiced_frame_ratio`: `0.9334112149532711`
- `voiced_duration_s`: `7.99`
- `mean_f0_hz`: `169.40504455566406`
- `median_f0_hz`: `171.2675018310547`
- `min_f0_hz`: `65.4000015258789`
- `max_f0_hz`: `423.3371276855469`
- `pitch_accuracy`: `1.0`
- `pitch_drift_cents`: `-0.16903279721736908`
- `breath_count`: `5`
- `onset_count`: `10`
- `onset_clarity`: `0.40005308389663696`
- `technique`: `breathy`
- `technique_confidence`: `0.351906418800354`
- `note_count`: `12`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Issues

- Specific notes are consistently off: D3 (+42 ¢, sharp), E2 (-1 ¢, flat).
- Vocal instability detected (jitter 1.0%, shimmer 9.7%). This can indicate tension, fatigue, or insufficient warm-up.
- Phrase length averages 4.2 s. Aim for 5–6 s to improve musical line.
- 4 sustained note(s) detected but no vibrato found. Adding vibrato enriches long notes and shows vocal control.

## Exercises

- For those notes, imagine lifting the back of your tongue slightly and 'thinking' the pitch higher before you sing it.
- Rest the voice for 20 min, hydrate well, then warm up with gentle lip trills before attempting full-voice singing.
- Before each phrase take a fuller breath and see how long you can sustain 'aaah' on a comfortable note. Target: 6 s without strain.
- Practice a gentle hand-on-chest pulse while sustaining a note — this helps initiate the natural 5–6 Hz oscillation of healthy vibrato.

## Confidence Curve Availability

analyse_recording() returns thresholded voiced/breath/onset arrays, but not raw per-frame confidence probabilities.
