# Self-Recorded Evaluation Summary

- Samples evaluated: `5`
- JSON summary: `reports/eval/self_recorded/summary.json`

| Sample | Provided task | Detected input | Score status | Full-song score | Diagnostic score | Task summary | Caveats | Regression expectation |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 00_silence | free_singing | no_voice_or_noise | no_analyzable_singing |  |  | No analyzable singing was detected. | Task scoring skipped because input was not analyzable singing. | PASS |
| 01_speaking_voice | free_singing | speech_like_or_non_singing | speech_or_non_singing_no_score |  |  | This sounds like speech or non-singing voice, so singing coaching was not generated. | Task scoring skipped because input was not analyzable singing. | PASS |
| 03_sustained_aaa | sustained_note | diagnostic_sustained_tone | diagnostic_sustained_tone_only |  | 95 | Sustained-note diagnostic complete; full-song scoring was not generated. | Diagnostic sustained-note score only; no reference melody was evaluated. | PASS |
| 04_pitch_slide | pitch_slide | diagnostic_pitch_slide | diagnostic_pitch_slide_only |  | 89 | Pitch-slide diagnostic complete; full-song scoring was not generated. | Diagnostic pitch-slide score only; no reference melody was evaluated. | PASS |
| 05_twinkle_twinkle | free_singing | analyzable_singing | free_singing_general_feedback | 78 |  | Good foundation — pitch 100%, avg phrase 4.2 s. The issues below will make a clear difference. Note: Score is based on detected pitch and timing features, not a reference melody. | Score is based on detected pitch and timing features, not a reference melody. | PASS |

## Notes

- Analysis validity is a postprocessing gate; raw frame outputs and notes remain present for inspection.
- Each sample is evaluated with an explicit task_config.
- Full-song score and diagnostic score are reported separately.
- Normal singing coaching is blocked for non-analyzable and diagnostic inputs.
- `.m4a` files are converted with macOS `afconvert` when direct decoding is unavailable.
- Checks are heuristics for diagnostic sanity, not formal model accuracy metrics.
