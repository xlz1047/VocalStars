# Evaluation: 01_speaking_voice

- Input: `samples/01_speaking_voice.wav`
- Audio used for inference: `samples/01_speaking_voice.wav`
- Converted: `False`
- Score: `None`
- Full-song score: `None`
- Diagnostic score: `None`
- Score status: `speech_or_non_singing_no_score`
- Task type: `free_singing`
- Summary: This sounds like speech or non-singing voice, so singing coaching was not generated.

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
- `full_song_score`: `None`
- `diagnostic_score`: `None`
- `score_status`: `speech_or_non_singing_no_score`
- `score_caveat`: `None`
- `breath_count`: `14`
- `onset_count`: `11`
- `onset_clarity`: `0.4042777121067047`
- `technique`: `not_applicable`
- `technique_confidence`: `0.0`
- `note_count`: `9`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Analysis Validity

- `is_analyzable`: `False`
- `input_type`: `speech_like_or_non_singing`
- `confidence`: `0.72`
- `reason_codes`: `['speech_like_fragmentation', 'low_pitch_confidence', 'frequent_octave_jumps']`

### Validity Metrics

| Metric | Value |
| --- | ---: |
| `audio_rms` | `0.0115544` |
| `voiced_frame_ratio` | `0.833791` |
| `voiced_probability_mean` | `0.673063` |
| `voiced_probability_near_threshold_fraction` | `0.296703` |
| `pitch_confidence_mean` | `0.197415` |
| `pitch_confidence_margin_mean` | `0.0382252` |
| `pitch_normalized_entropy_mean` | `0.564517` |
| `f0_trimmed_range_hz` | `509.401` |
| `low_frequency_f0_ratio` | `0.110379` |
| `octave_jump_rate_per_second` | `0` |
| `semitone_jump_rate_per_second` | `1.92308` |
| `notes_per_second` | `1.23626` |
| `short_note_ratio_lt_300ms` | `0.444444` |
| `onsets_per_second` | `1.51099` |

## Task Analysis

- `task_type`: `free_singing`
- `detected_input_type`: `speech_like_or_non_singing`
- `status`: `speech_or_non_singing_no_score`
- `summary`: This sounds like speech or non-singing voice, so singing coaching was not generated.
- `caveats`: `['Task scoring skipped because input was not analyzable singing.']`

## P0 Diagnostics

| Metric | Value |
| --- | ---: |
| `source` | `checkpoint` |
| `voiced_probability.mean` | `0.673063` |
| `voiced_probability.median` | `0.615252` |
| `voiced_probability.min` | `0.365672` |
| `voiced_probability.max` | `0.957466` |
| `voiced_probability.near_threshold_fraction` | `0.296703` |
| `pitch_confidence.max_softmax_probability.mean` | `0.197415` |
| `pitch_confidence.top1_top2_margin.mean` | `0.0382252` |
| `pitch_confidence.normalized_entropy.mean` | `0.564517` |
| `onset_probability.mean` | `0.331173` |
| `breath_probability.mean` | `0.349441` |
| `f0.median_hz` | `144.018` |
| `f0.full_range_hz.min` | `65.4` |
| `f0.full_range_hz.max` | `754.301` |
| `f0.trimmed_range_hz.p05` | `66.6714` |
| `f0.trimmed_range_hz.p95` | `576.073` |
| `f0.low_frequency_f0_ratio` | `0.110379` |
| `f0_jumps.octave_jump_count` | `0` |
| `f0_jumps.octave_jump_rate_per_second` | `0` |
| `f0_jumps.semitone_jump_count` | `14` |
| `f0_jumps.semitone_jump_rate_per_second` | `1.92308` |
| `note_fragmentation.notes_per_second` | `1.23626` |
| `note_fragmentation.notes_per_voiced_second` | `1.4827` |
| `note_fragmentation.median_note_duration_s` | `0.31` |
| `note_fragmentation.short_note_ratio_lt_300ms` | `0.444444` |
| `note_postprocessing.raw_note_count` | `14` |
| `note_postprocessing.postprocessed_note_count` | `9` |
| `note_postprocessing.merge_count` | `1` |
| `note_postprocessing.octave_jump_count` | `14` |
| `note_postprocessing.postprocessed_octave_jump_count` | `0` |
| `note_postprocessing.f0_stability_cents` | `981.617` |
| `note_postprocessing.fragmentation_index` | `1.23626` |

## Issues

- None reported.

## Exercises

- None reported.

## Confidence Curve Availability

analyse_recording() now returns summary diagnostics for raw probabilities/confidence, but not frame-level confidence arrays.
