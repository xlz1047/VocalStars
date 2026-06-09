# Evaluation: 00_silence

- Input: `samples/00_silence.wav`
- Audio used for inference: `samples/00_silence.wav`
- Converted: `False`
- Score: `None`
- Full-song score: `None`
- Diagnostic score: `None`
- Score status: `no_analyzable_singing`
- Task type: `free_singing`
- Summary: No analyzable singing was detected.

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
- `full_song_score`: `None`
- `diagnostic_score`: `None`
- `score_status`: `no_analyzable_singing`
- `score_caveat`: `None`
- `breath_count`: `0`
- `onset_count`: `1`
- `onset_clarity`: `0.4165007770061493`
- `technique`: `not_applicable`
- `technique_confidence`: `0.0`
- `note_count`: `8`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Analysis Validity

- `is_analyzable`: `False`
- `input_type`: `no_voice_or_noise`
- `confidence`: `0.9`
- `reason_codes`: `['very_low_audio_rms', 'low_pitch_confidence', 'high_pitch_entropy', 'voiced_probabilities_near_threshold']`

### Validity Metrics

| Metric | Value |
| --- | ---: |
| `audio_rms` | `0.00134131` |
| `voiced_frame_ratio` | `0.955331` |
| `voiced_probability_mean` | `0.564935` |
| `voiced_probability_near_threshold_fraction` | `0.306916` |
| `pitch_confidence_mean` | `0.128354` |
| `pitch_confidence_margin_mean` | `0.0125362` |
| `pitch_normalized_entropy_mean` | `0.653033` |
| `f0_trimmed_range_hz` | `20.2337` |
| `low_frequency_f0_ratio` | `0.263952` |
| `octave_jump_rate_per_second` | `0` |
| `semitone_jump_rate_per_second` | `4.32277` |
| `notes_per_second` | `1.15274` |
| `short_note_ratio_lt_300ms` | `0.5` |
| `onsets_per_second` | `0.144092` |

## Task Analysis

- `task_type`: `free_singing`
- `detected_input_type`: `no_voice_or_noise`
- `status`: `no_analyzable_singing`
- `summary`: No analyzable singing was detected.
- `caveats`: `['Task scoring skipped because input was not analyzable singing.']`

## P0 Diagnostics

| Metric | Value |
| --- | ---: |
| `source` | `checkpoint` |
| `voiced_probability.mean` | `0.564935` |
| `voiced_probability.median` | `0.569966` |
| `voiced_probability.min` | `0.298973` |
| `voiced_probability.max` | `0.672553` |
| `voiced_probability.near_threshold_fraction` | `0.306916` |
| `pitch_confidence.max_softmax_probability.mean` | `0.128354` |
| `pitch_confidence.top1_top2_margin.mean` | `0.0125362` |
| `pitch_confidence.normalized_entropy.mean` | `0.653033` |
| `onset_probability.mean` | `0.416501` |
| `breath_probability.mean` | `0.167441` |
| `f0.median_hz` | `84.0007` |
| `f0.full_range_hz.min` | `65.4` |
| `f0.full_range_hz.max` | `101.836` |
| `f0.trimmed_range_hz.p05` | `65.4` |
| `f0.trimmed_range_hz.p95` | `85.6338` |
| `f0.low_frequency_f0_ratio` | `0.263952` |
| `f0_jumps.octave_jump_count` | `0` |
| `f0_jumps.octave_jump_rate_per_second` | `0` |
| `f0_jumps.semitone_jump_count` | `30` |
| `f0_jumps.semitone_jump_rate_per_second` | `4.32277` |
| `note_fragmentation.notes_per_second` | `1.15274` |
| `note_fragmentation.notes_per_voiced_second` | `1.20664` |
| `note_fragmentation.median_note_duration_s` | `0.295` |
| `note_fragmentation.short_note_ratio_lt_300ms` | `0.5` |
| `note_postprocessing.raw_note_count` | `19` |
| `note_postprocessing.postprocessed_note_count` | `8` |
| `note_postprocessing.merge_count` | `5` |
| `note_postprocessing.octave_jump_count` | `0` |
| `note_postprocessing.postprocessed_octave_jump_count` | `0` |
| `note_postprocessing.f0_stability_cents` | `166.102` |
| `note_postprocessing.fragmentation_index` | `1.15274` |

## Issues

- None reported.

## Exercises

- None reported.

## Confidence Curve Availability

analyse_recording() now returns summary diagnostics for raw probabilities/confidence, but not frame-level confidence arrays.
