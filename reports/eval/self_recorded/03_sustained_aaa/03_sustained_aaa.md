# Evaluation: 03_sustained_aaa

- Input: `samples/03_sustained_aaa.wav`
- Audio used for inference: `samples/03_sustained_aaa.wav`
- Converted: `False`
- Score: `71`
- Summary: Good foundation — pitch 100%, avg phrase 6.9 s. The issues below will make a clear difference.

## Artifacts

- JSON: `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa.json`
- Plot: `reports/eval/self_recorded/03_sustained_aaa/03_sustained_aaa_plots.svg`

## Key Metrics

- `duration_s`: `6.933`
- `audio_rms`: `0.018262263387441635`
- `voiced_frame_ratio`: `0.9899135446685879`
- `voiced_duration_s`: `6.87`
- `mean_f0_hz`: `120.43589782714844`
- `median_f0_hz`: `138.57777404785156`
- `min_f0_hz`: `70.63591003417969`
- `max_f0_hz`: `659.1907958984375`
- `pitch_accuracy`: `1.0`
- `pitch_drift_cents`: `-0.16923919320106506`
- `breath_count`: `1`
- `onset_count`: `6`
- `onset_clarity`: `0.3978387415409088`
- `technique`: `vocal_fry`
- `technique_confidence`: `0.9082742929458618`
- `note_count`: `14`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Issues

- Specific notes are consistently off: C#3 (+43 ¢, sharp), D3 (-40 ¢, flat), D2 (-34 ¢, flat).
- Breathy voice quality detected (HNR 3 dB — air escaping before the cords fully close).
- Vocal instability detected (jitter 6.9%, shimmer 18.3%). This can indicate tension, fatigue, or insufficient warm-up.
- 4 sustained note(s) detected but no vibrato found. Adding vibrato enriches long notes and shows vocal control.

## Exercises

- For those notes, imagine lifting the back of your tongue slightly and 'thinking' the pitch higher before you sing it.
- Hum 'mmm' with lips lightly closed to build cord closure and forward resonance. Gradually open to 'mah' while keeping the buzz.
- Rest the voice for 20 min, hydrate well, then warm up with gentle lip trills before attempting full-voice singing.
- Practice a gentle hand-on-chest pulse while sustaining a note — this helps initiate the natural 5–6 Hz oscillation of healthy vibrato.

## Confidence Curve Availability

analyse_recording() returns thresholded voiced/breath/onset arrays, but not raw per-frame confidence probabilities.
