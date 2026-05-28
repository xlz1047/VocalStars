# Evaluation: 00_silence

- Input: `samples/00_silence.wav`
- Audio used for inference: `samples/00_silence.wav`
- Converted: `False`
- Score: `71`
- Summary: Good foundation — pitch 100%, avg phrase 6.8 s. The issues below will make a clear difference.

## Artifacts

- JSON: `reports/eval/self_recorded/00_silence/00_silence.json`
- Plot: `reports/eval/self_recorded/00_silence/00_silence_plots.svg`

## Key Metrics

- `duration_s`: `6.933`
- `audio_rms`: `0.0013413133565336466`
- `voiced_frame_ratio`: `0.9553314121037464`
- `voiced_duration_s`: `6.63`
- `mean_f0_hz`: `80.50320434570312`
- `median_f0_hz`: `84.00072479248047`
- `min_f0_hz`: `65.4000015258789`
- `max_f0_hz`: `101.83626556396484`
- `pitch_accuracy`: `1.0`
- `pitch_drift_cents`: `33.1642951965332`
- `breath_count`: `0`
- `onset_count`: `1`
- `onset_clarity`: `0.4165007770061493`
- `technique`: `breathy`
- `technique_confidence`: `0.26509419083595276`
- `note_count`: `19`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Issues

- Specific notes are consistently off: E2 (+44 ¢, sharp), C2 (+34 ¢, sharp).
- Breathy voice quality detected (HNR -2 dB — air escaping before the cords fully close).
- Vocal instability detected (jitter 7.6%, shimmer 5.9%). This can indicate tension, fatigue, or insufficient warm-up.

## Exercises

- For those notes, relax the jaw and let the breath drop lower in the body before onset to prevent over-shooting.
- Hum 'mmm' with lips lightly closed to build cord closure and forward resonance. Gradually open to 'mah' while keeping the buzz.
- Rest the voice for 20 min, hydrate well, then warm up with gentle lip trills before attempting full-voice singing.

## Confidence Curve Availability

analyse_recording() returns thresholded voiced/breath/onset arrays, but not raw per-frame confidence probabilities.
