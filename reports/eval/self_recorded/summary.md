# Self-Recorded Evaluation Summary

- Samples evaluated: `5`
- JSON summary: `reports/eval/self_recorded/summary.json`

| Sample | Status | Expected behavior check | Score | Voiced ratio | Mean f0 | Onsets | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 00_silence | success | QUESTIONABLE: silence produced substantial voiced frames | 71 | 0.955 | 80.5 | 1 | 19 |
| 01_speaking_voice | success | OBSERVE: speech may be voiced; singing-specific outputs are not validated | 71 | 0.834 | 161.3 | 11 | 14 |
| 03_sustained_aaa | success | QUESTIONABLE: sustained vowel was not cleanly represented | 71 | 0.990 | 120.4 | 6 | 14 |
| 04_pitch_slide | success | PASS-ish | 91 | 0.929 | 209.9 | 3 | 3 |
| 05_twinkle_twinkle | success | PASS-ish | 78 | 0.933 | 169.4 | 10 | 12 |

## Notes

- The existing inference entrypoint exposes thresholded voiced/breath/onset arrays, not raw confidence curves.
- `.m4a` files are converted with macOS `afconvert` when direct decoding is unavailable.
- Checks are heuristics for diagnostic sanity, not formal model accuracy metrics.
