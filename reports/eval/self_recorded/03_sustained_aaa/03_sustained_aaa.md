# Evaluation: 03_sustained_aaa

- Input: `samples/03_sustained_aaa.wav`
- Audio used for inference: `samples/03_sustained_aaa.wav`
- Converted: `False`
- Score: `95`
- Full-song score: `None`
- Diagnostic score: `95`
- Score status: `diagnostic_sustained_tone_only`
- Task type: `sustained_note`
- Summary: Sustained-note diagnostic complete; full-song scoring was not generated.

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
- `full_song_score`: `None`
- `diagnostic_score`: `95`
- `score_status`: `diagnostic_sustained_tone_only`
- `score_caveat`: `Diagnostic sustained-note score only; no reference melody was evaluated.`
- `breath_count`: `1`
- `onset_count`: `6`
- `onset_clarity`: `0.3978387415409088`
- `technique`: `not_applicable`
- `technique_confidence`: `0.0`
- `note_count`: `3`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Analysis Validity

- `is_analyzable`: `False`
- `input_type`: `diagnostic_sustained_tone`
- `confidence`: `0.7`
- `reason_codes`: `['continuous_voicing', 'fragmented_f0_tracking']`

### Validity Metrics

| Metric | Value |
| --- | ---: |
| `audio_rms` | `0.0182623` |
| `voiced_frame_ratio` | `0.989914` |
| `voiced_probability_mean` | `0.693577` |
| `voiced_probability_near_threshold_fraction` | `0.0273775` |
| `pitch_confidence_mean` | `0.211325` |
| `pitch_confidence_margin_mean` | `0.0438307` |
| `pitch_normalized_entropy_mean` | `0.546539` |
| `f0_trimmed_range_hz` | `470.31` |
| `low_frequency_f0_ratio` | `0.00291121` |
| `octave_jump_rate_per_second` | `0` |
| `semitone_jump_rate_per_second` | `0.864553` |
| `notes_per_second` | `0.432277` |
| `short_note_ratio_lt_300ms` | `0` |
| `onsets_per_second` | `0.864553` |

## Task Analysis

- `task_type`: `sustained_note`
- `detected_input_type`: `diagnostic_sustained_tone`
- `status`: `diagnostic_sustained_tone_only`
- `summary`: Sustained-note diagnostic complete; full-song scoring was not generated.
- `caveats`: `['Diagnostic sustained-note score only; no reference melody was evaluated.']`

## P0 Diagnostics

| Metric | Value |
| --- | ---: |
| `source` | `checkpoint` |
| `voiced_probability.mean` | `0.693577` |
| `voiced_probability.median` | `0.686374` |
| `voiced_probability.min` | `0.396194` |
| `voiced_probability.max` | `0.891488` |
| `voiced_probability.near_threshold_fraction` | `0.0273775` |
| `pitch_confidence.max_softmax_probability.mean` | `0.211325` |
| `pitch_confidence.top1_top2_margin.mean` | `0.0438307` |
| `pitch_confidence.normalized_entropy.mean` | `0.546539` |
| `onset_probability.mean` | `0.374316` |
| `breath_probability.mean` | `0.547965` |
| `f0.median_hz` | `288.036` |
| `f0.full_range_hz.min` | `79.2861` |
| `f0.full_range_hz.max` | `659.191` |
| `f0.trimmed_range_hz.p05` | `84.0007` |
| `f0.trimmed_range_hz.p95` | `554.311` |
| `f0.low_frequency_f0_ratio` | `0.00291121` |
| `f0_jumps.octave_jump_count` | `0` |
| `f0_jumps.octave_jump_rate_per_second` | `0` |
| `f0_jumps.semitone_jump_count` | `6` |
| `f0_jumps.semitone_jump_rate_per_second` | `0.864553` |
| `note_fragmentation.notes_per_second` | `0.432277` |
| `note_fragmentation.notes_per_voiced_second` | `0.436681` |
| `note_fragmentation.median_note_duration_s` | `1.08` |
| `note_fragmentation.short_note_ratio_lt_300ms` | `0` |
| `note_postprocessing.raw_note_count` | `14` |
| `note_postprocessing.postprocessed_note_count` | `3` |
| `note_postprocessing.merge_count` | `1` |
| `note_postprocessing.octave_jump_count` | `23` |
| `note_postprocessing.postprocessed_octave_jump_count` | `0` |
| `note_postprocessing.f0_stability_cents` | `810.873` |
| `note_postprocessing.fragmentation_index` | `0.432277` |

## Issues

- None reported.

## Exercises

- None reported.

## Confidence Curve Availability

analyse_recording() now returns summary diagnostics for raw probabilities/confidence, but not frame-level confidence arrays.
