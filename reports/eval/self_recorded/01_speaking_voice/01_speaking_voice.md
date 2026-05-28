# Evaluation: 01_speaking_voice

- Input: `samples/01_speaking_voice.wav`
- Audio used for inference: `samples/01_speaking_voice.wav`
- Converted: `False`
- Score: `71`
- Summary: Good foundation — pitch 100%, avg phrase 3.1 s. The issues below will make a clear difference.

## Artifacts

- JSON: `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice.json`
- Plot: `reports/eval/self_recorded/01_speaking_voice/01_speaking_voice_plots.svg`

## Key Metrics

- `duration_s`: `7.275`
- `audio_rms`: `0.011554400436580181`
- `voiced_frame_ratio`: `0.8337912087912088`
- `voiced_duration_s`: `6.07`
- `mean_f0_hz`: `161.2718048095703`
- `median_f0_hz`: `130.8000030517578`
- `min_f0_hz`: `65.4000015258789`
- `max_f0_hz`: `768.9650268554688`
- `pitch_accuracy`: `1.0`
- `pitch_drift_cents`: `-0.16913600265979767`
- `breath_count`: `14`
- `onset_count`: `11`
- `onset_clarity`: `0.4042777121067047`
- `technique`: `breathy`
- `technique_confidence`: `0.3795154094696045`
- `note_count`: `14`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Issues

- Specific notes are consistently off: F2 (-37 ¢, flat).
- Slightly airy tone (HNR 12 dB). More cord engagement will give a clearer, more projected sound.
- Vocal instability detected (jitter 2.9%, shimmer 11.9%). This can indicate tension, fatigue, or insufficient warm-up.
- Phrases average only 3.1 s — breath runs out too quickly.

## Exercises

- For those notes, imagine lifting the back of your tongue slightly and 'thinking' the pitch higher before you sing it.
- Try 'staccato' vowel exercises (short, clear 'ha-ha-ha') to encourage full cord closure at each attack.
- Rest the voice for 20 min, hydrate well, then warm up with gentle lip trills before attempting full-voice singing.
- Diaphragmatic breathing: inhale silently for 4 counts, then sustain 'sss' for 8 counts. Build up to 12 counts over a week.

## Confidence Curve Availability

analyse_recording() returns thresholded voiced/breath/onset arrays, but not raw per-frame confidence probabilities.
