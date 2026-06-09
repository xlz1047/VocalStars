# Evaluation: 04_pitch_slide

- Input: `samples/04_pitch_slide.wav`
- Audio used for inference: `samples/04_pitch_slide.wav`
- Converted: `False`
- Score: `89`
- Full-song score: `None`
- Diagnostic score: `89`
- Score status: `diagnostic_pitch_slide_only`
- Task type: `pitch_slide`
- Summary: Pitch-slide diagnostic complete; full-song scoring was not generated.

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
- `full_song_score`: `None`
- `diagnostic_score`: `89`
- `score_status`: `diagnostic_pitch_slide_only`
- `score_caveat`: `Diagnostic pitch-slide score only; no reference melody was evaluated.`
- `breath_count`: `3`
- `onset_count`: `3`
- `onset_clarity`: `0.40119418501853943`
- `technique`: `not_applicable`
- `technique_confidence`: `0.0`
- `note_count`: `2`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Analysis Validity

- `is_analyzable`: `False`
- `input_type`: `diagnostic_pitch_slide`
- `confidence`: `0.72`
- `reason_codes`: `['continuous_voicing', 'wide_f0_movement', 'few_note_events']`

### Validity Metrics

| Metric | Value |
| --- | ---: |
| `audio_rms` | `0.0551456` |
| `voiced_frame_ratio` | `0.928986` |
| `voiced_probability_mean` | `0.855435` |
| `voiced_probability_near_threshold_fraction` | `0.0826087` |
| `pitch_confidence_mean` | `0.340149` |
| `pitch_confidence_margin_mean` | `0.083285` |
| `pitch_normalized_entropy_mean` | `0.356167` |
| `f0_trimmed_range_hz` | `137.58` |
| `low_frequency_f0_ratio` | `0.0405616` |
| `octave_jump_rate_per_second` | `0.144928` |
| `semitone_jump_rate_per_second` | `0.724638` |
| `notes_per_second` | `0.289855` |
| `short_note_ratio_lt_300ms` | `0.5` |
| `onsets_per_second` | `0.434783` |

## Task Analysis

- `task_type`: `pitch_slide`
- `detected_input_type`: `diagnostic_pitch_slide`
- `status`: `diagnostic_pitch_slide_only`
- `summary`: Pitch-slide diagnostic complete; full-song scoring was not generated.
- `caveats`: `['Diagnostic pitch-slide score only; no reference melody was evaluated.']`

## P0 Diagnostics

| Metric | Value |
| --- | ---: |
| `source` | `checkpoint` |
| `voiced_probability.mean` | `0.855435` |
| `voiced_probability.median` | `0.937582` |
| `voiced_probability.min` | `0.302032` |
| `voiced_probability.max` | `0.957305` |
| `voiced_probability.near_threshold_fraction` | `0.0826087` |
| `pitch_confidence.max_softmax_probability.mean` | `0.340149` |
| `pitch_confidence.top1_top2_margin.mean` | `0.083285` |
| `pitch_confidence.normalized_entropy.mean` | `0.356167` |
| `onset_probability.mean` | `0.182427` |
| `breath_probability.mean` | `0.266324` |
| `f0.median_hz` | `215.784` |
| `f0.full_range_hz.min` | `65.4` |
| `f0.full_range_hz.max` | `219.979` |
| `f0.trimmed_range_hz.p05` | `82.3988` |
| `f0.trimmed_range_hz.p95` | `219.979` |
| `f0.low_frequency_f0_ratio` | `0.0405616` |
| `f0_jumps.octave_jump_count` | `1` |
| `f0_jumps.octave_jump_rate_per_second` | `0.144928` |
| `f0_jumps.semitone_jump_count` | `5` |
| `f0_jumps.semitone_jump_rate_per_second` | `0.724638` |
| `note_fragmentation.notes_per_second` | `0.289855` |
| `note_fragmentation.notes_per_voiced_second` | `0.312012` |
| `note_fragmentation.median_note_duration_s` | `3.005` |
| `note_fragmentation.short_note_ratio_lt_300ms` | `0.5` |
| `note_postprocessing.raw_note_count` | `3` |
| `note_postprocessing.postprocessed_note_count` | `2` |
| `note_postprocessing.merge_count` | `0` |
| `note_postprocessing.octave_jump_count` | `2` |
| `note_postprocessing.postprocessed_octave_jump_count` | `1` |
| `note_postprocessing.f0_stability_cents` | `553.569` |
| `note_postprocessing.fragmentation_index` | `0.289855` |

## Issues

- None reported.

## Exercises

- None reported.

## Confidence Curve Availability

analyse_recording() now returns summary diagnostics for raw probabilities/confidence, but not frame-level confidence arrays.
