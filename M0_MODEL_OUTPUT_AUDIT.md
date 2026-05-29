# M0 Model Output Audit

Goal: inspect whether the checkpoint model itself is producing reliable raw outputs, independent of scoring, coaching, validity gates, and task-specific evaluators.

No retraining, model architecture changes, scoring tuning, app behavior changes, or P4 regression expectation changes were made.

## Checkpoint Inference Path

The checkpoint path in `ml_new/inference/coach_inference.py` loads audio at 16 kHz, computes HCQT features and handcrafted VAD features, loads `UnifiedVocalModel`, and runs one forward pass:

```text
pitch_logits, voiced_prob, breath_prob, onset_prob, tech_logits_base, _ = model(hcqt_t, vad_t)
```

This audit bypasses validity gating, note segmentation, scoring, and coaching text. It saves raw/minimally processed arrays directly after the model forward pass.

## Raw Outputs Produced

- `pitch_logits`: `(T, 180)` unnormalized pitch-bin logits.
- `voiced_prob`: `(T,)` per-frame voiced probability.
- `breath_prob`: `(T,)` per-frame breath probability.
- `onset_prob`: `(T,)` per-frame onset probability.
- `tech_logits`: `(20,)` clip-level technique logits from the unified model technique head.

Minimally processed audit fields derived from those outputs:

- pitch softmax confidence, top-2 margin, and entropy from `pitch_logits`.
- raw f0 from pitch-logit argmax, thresholded by `voiced_prob >= 0.5`.
- smoothed f0 using the existing P2 `stabilize_f0_for_notes()` helper, for comparison only.
- technique probabilities from softmax over `tech_logits`; marked unreliable.

## Outputs

- Output directory: `reports/model_output_audit`
- Checkpoint: `ml_new/checkpoints/unified/best.pt`

| Sample | VAD mean | VAD >=0.3 | VAD >=0.5 | VAD >=0.7 | VAD >=0.9 | Pitch conf mean | Pitch margin mean | Norm entropy mean | Raw f0 trimmed range | Raw octave jumps | Smoothed octave jumps | Onset mean | Breath mean | Top technique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `00_silence` | 0.5649 | 99.9% | 95.5% | 0.0% | 0.0% | 0.1284 | 0.0125 | 0.6530 | 20.2337 | 0 | 0 | 0.4165 | 0.1674 | `breathy` |
| `01_speaking_voice` | 0.6731 | 100.0% | 83.4% | 39.6% | 14.6% | 0.1974 | 0.0382 | 0.5645 | 646.5655 | 16 | 2 | 0.3312 | 0.3494 | `spoken` |
| `03_sustained_aaa` | 0.6936 | 100.0% | 99.0% | 43.5% | 0.0% | 0.2113 | 0.0438 | 0.5465 | 77.6632 | 27 | 1 | 0.3743 | 0.5480 | `vocal_fry` |
| `04_pitch_slide` | 0.8554 | 100.0% | 92.9% | 81.3% | 78.6% | 0.3401 | 0.0833 | 0.3562 | 137.5797 | 2 | 1 | 0.1824 | 0.2663 | `straight` |
| `05_twinkle_twinkle` | 0.8199 | 99.3% | 93.3% | 76.4% | 54.2% | 0.3110 | 0.0625 | 0.3840 | 153.7730 | 8 | 0 | 0.2095 | 0.2774 | `breathy` |

## Direct Answers

### Was `00_silence` a high-confidence VAD false positive or barely above threshold?

`00_silence` is a VAD false positive, but not a high-confidence one. Mean voiced probability is `0.5649`, `95.5%` of frames are above 0.5, only `0.0%` are above 0.7, and `0.0%` are above 0.9. `30.7%` of frames are within +/-0.05 of the 0.5 threshold. Pitch confidence is weak: mean max-softmax `0.1284` and mean top-2 margin `0.0125`.

### Was `03_sustained_aaa` raw f0 actually unstable, or did segmentation cause most of the instability?

The raw f0 is already unstable before note segmentation. Raw thresholded f0 spans `70.6359`-`659.1908` Hz with trimmed range `77.6632` Hz, `27` octave-scale jumps, and `31` >=2-semitone adjacent jumps. Smoothing reduces octave jumps to `1` and semitone jumps to `4`, so segmentation amplified the problem, but did not create it from a stable contour.

### Does `04_pitch_slide` preserve directional movement in raw f0?

Yes. The raw f0 preserves broad directional movement: the fitted raw-f0 slope is `7.8591` Hz/s (upward), with trimmed range `137.5797` Hz. There are still `8` large adjacent semitone jumps, so the movement is useful but not artifact-free.

### Does `05_twinkle_twinkle` have usable f0 structure?

Yes, with caveats. `05_twinkle_twinkle` has usable multi-note f0 structure: raw trimmed f0 range is `153.7730` Hz, median f0 is `171.2675` Hz, and voiced f0 covers `93.3%` of frames. Pitch confidence is moderate rather than strong: mean max-softmax `0.3110` and mean top-2 margin `0.0625`.

## Interpretation

- The raw VAD head is not calibrated enough to be trusted alone on fan/noise input; postprocessing gates remain necessary.
- Pitch confidence is generally modest, and low top-2 margins indicate that argmax f0 can jump even when the voiced probability is high.
- Technique logits/probabilities are included for audit completeness only. Existing notes about technique unreliability still apply.
- This audit does not assert model accuracy against ground truth because these five samples do not include frame-level labels.
