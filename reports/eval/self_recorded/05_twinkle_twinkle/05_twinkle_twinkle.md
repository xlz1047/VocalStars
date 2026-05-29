# Evaluation: 05_twinkle_twinkle

- Input: `samples/05_twinkle_twinkle.wav`
- Audio used for inference: `samples/05_twinkle_twinkle.wav`
- Converted: `False`
- Score: `78`
- Full-song score: `78`
- Diagnostic score: `None`
- Score status: `free_singing_general_feedback`
- Task type: `free_singing`
- Summary: Good foundation — pitch 100%, avg phrase 4.2 s. The issues below will make a clear difference. Note: Score is based on detected pitch and timing features, not a reference melody.

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
- `full_song_score`: `78`
- `diagnostic_score`: `None`
- `score_status`: `free_singing_general_feedback`
- `score_caveat`: `Score is based on detected pitch and timing features, not a reference melody.`
- `breath_count`: `5`
- `onset_count`: `10`
- `onset_clarity`: `0.40005308389663696`
- `technique`: `breathy`
- `technique_confidence`: `0.351906418800354`
- `note_count`: `7`
- `voice_quality_available`: `True`
- `confidence_curve_available`: `False`

## Analysis Validity

- `is_analyzable`: `True`
- `input_type`: `analyzable_singing`
- `confidence`: `0.75`
- `reason_codes`: `['passes_current_postprocessing_checks']`

### Validity Metrics

| Metric | Value |
| --- | ---: |
| `audio_rms` | `0.0201513` |
| `voiced_frame_ratio` | `0.933411` |
| `voiced_probability_mean` | `0.819921` |
| `voiced_probability_near_threshold_fraction` | `0.0852804` |
| `pitch_confidence_mean` | `0.31103` |
| `pitch_confidence_margin_mean` | `0.0625399` |
| `pitch_normalized_entropy_mean` | `0.384041` |
| `f0_trimmed_range_hz` | `155.206` |
| `low_frequency_f0_ratio` | `0.157697` |
| `octave_jump_rate_per_second` | `0` |
| `semitone_jump_rate_per_second` | `0.934579` |
| `notes_per_second` | `0.817757` |
| `short_note_ratio_lt_300ms` | `0.142857` |
| `onsets_per_second` | `1.16822` |

## Task Analysis

- `task_type`: `free_singing`
- `detected_input_type`: `analyzable_singing`
- `status`: `free_singing_general_feedback`
- `summary`: Good foundation — pitch 100%, avg phrase 4.2 s. The issues below will make a clear difference. Note: Score is based on detected pitch and timing features, not a reference melody.
- `caveats`: `['Score is based on detected pitch and timing features, not a reference melody.']`

## P0 Diagnostics

| Metric | Value |
| --- | ---: |
| `source` | `checkpoint` |
| `voiced_probability.mean` | `0.819921` |
| `voiced_probability.median` | `0.905944` |
| `voiced_probability.min` | `0.254326` |
| `voiced_probability.max` | `0.966917` |
| `voiced_probability.near_threshold_fraction` | `0.0852804` |
| `pitch_confidence.max_softmax_probability.mean` | `0.31103` |
| `pitch_confidence.top1_top2_margin.mean` | `0.0625399` |
| `pitch_confidence.normalized_entropy.mean` | `0.384041` |
| `onset_probability.mean` | `0.209482` |
| `breath_probability.mean` | `0.277446` |
| `f0.median_hz` | `161.655` |
| `f0.full_range_hz.min` | `65.4` |
| `f0.full_range_hz.max` | `237.59` |
| `f0.trimmed_range_hz.p05` | `73.409` |
| `f0.trimmed_range_hz.p95` | `228.615` |
| `f0.low_frequency_f0_ratio` | `0.157697` |
| `f0_jumps.octave_jump_count` | `0` |
| `f0_jumps.octave_jump_rate_per_second` | `0` |
| `f0_jumps.semitone_jump_count` | `8` |
| `f0_jumps.semitone_jump_rate_per_second` | `0.934579` |
| `note_fragmentation.notes_per_second` | `0.817757` |
| `note_fragmentation.notes_per_voiced_second` | `0.876095` |
| `note_fragmentation.median_note_duration_s` | `0.6` |
| `note_fragmentation.short_note_ratio_lt_300ms` | `0.142857` |
| `note_postprocessing.raw_note_count` | `12` |
| `note_postprocessing.postprocessed_note_count` | `7` |
| `note_postprocessing.merge_count` | `0` |
| `note_postprocessing.octave_jump_count` | `4` |
| `note_postprocessing.postprocessed_octave_jump_count` | `0` |
| `note_postprocessing.f0_stability_cents` | `664.21` |
| `note_postprocessing.fragmentation_index` | `0.817757` |

## Issues

- Vocal instability detected (jitter 1.0%, shimmer 9.7%). This can indicate tension, fatigue, or insufficient warm-up.
- Phrase length averages 4.2 s. Aim for 5–6 s to improve musical line.
- 4 sustained note(s) detected but no vibrato found. Adding vibrato enriches long notes and shows vocal control.

## Exercises

- Rest the voice for 20 min, hydrate well, then warm up with gentle lip trills before attempting full-voice singing.
- Before each phrase take a fuller breath and see how long you can sustain 'aaah' on a comfortable note. Target: 6 s without strain.
- Practice a gentle hand-on-chest pulse while sustaining a note — this helps initiate the natural 5–6 Hz oscillation of healthy vibrato.

## Confidence Curve Availability

analyse_recording() now returns summary diagnostics for raw probabilities/confidence, but not frame-level confidence arrays.
