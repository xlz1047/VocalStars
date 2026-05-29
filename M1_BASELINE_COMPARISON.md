# M1 Baseline Comparison

Goal: compare checkpoint raw VAD/f0 behavior against a baseline DSP method on the same five WAV samples.

No retraining, model architecture changes, scoring tuning, app behavior changes, or P4 regression expectation changes were made.

## Methods

- Checkpoint outputs: loaded from M0 raw arrays in `reports/model_output_audit`.
- Baseline: `librosa.pyin()` at 16 kHz with 10 ms hop, `fmin=32.7 Hz`, `fmax=2100 Hz`.
- Checkpoint f0: pitch-logit argmax, masked by checkpoint `voiced_prob >= 0.5`.
- Baseline f0: `pyin` f0 masked by `pyin` voiced flag.
- These are raw-output comparisons, not user-facing scoring/coaching evaluations.

## Outputs

- Output directory: `reports/baseline_comparison`
- Per-sample JSON and SVG comparison plots are saved under `reports/baseline_comparison/<sample>/`.

## Summary Table

| Sample | Method | Voiced frames | Median voiced prob | F0 coverage | Median f0 | Full range | Trimmed range | Octave jumps/rate | Semitone jumps/rate | Stability cents | Pitch confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `00_silence` | checkpoint | 95.5% | 0.5700 | 95.5% | 84.0007 | 65.4000-101.8363 | 20.2337 | 0/0.0000 | 40/5.7637 | 165.6179 | 0.1284 |
| `00_silence` | baseline pyin | 48.7% | 0.0100 | 48.7% | 54.6779 | 32.7000-65.0233 | 18.0791 | 0/0.0000 | 0/0.0000 | 130.0906 | null |
| `01_speaking_voice` | checkpoint | 83.4% | 0.6153 | 83.4% | 130.8000 | 65.4000-768.9650 | 646.5655 | 16/2.1978 | 31/4.2582 | 678.5964 | 0.1974 |
| `01_speaking_voice` | baseline pyin | 62.0% | 0.0101 | 62.0% | 131.5577 | 33.2716-161.9666 | 110.9134 | 0/0.0000 | 0/0.0000 | 803.9249 | null |
| `03_sustained_aaa` | checkpoint | 99.0% | 0.6864 | 99.0% | 138.5778 | 70.6359-659.1908 | 77.6632 | 27/3.8905 | 31/4.4669 | 549.3202 | 0.2113 |
| `03_sustained_aaa` | baseline pyin | 88.8% | 0.2796 | 88.8% | 71.3192 | 40.0266-145.1317 | 25.1283 | 0/0.0000 | 0/0.0000 | 193.7593 | null |
| `04_pitch_slide` | checkpoint | 92.9% | 0.9376 | 92.9% | 215.7836 | 65.4000-711.9655 | 137.5797 | 2/0.2899 | 8/1.1594 | 375.0838 | 0.3401 |
| `04_pitch_slide` | baseline pyin | 86.8% | 0.9459 | 86.8% | 218.7115 | 44.4122-221.2528 | 170.5596 | 0/0.0000 | 0/0.0000 | 388.0485 | null |
| `05_twinkle_twinkle` | checkpoint | 93.3% | 0.9059 | 93.3% | 171.2675 | 65.4000-423.3371 | 153.7730 | 8/0.9346 | 15/1.7523 | 449.8915 | 0.3110 |
| `05_twinkle_twinkle` | baseline pyin | 82.1% | 0.6159 | 82.1% | 172.5916 | 32.7000-237.1329 | 181.6697 | 0/0.0000 | 0/0.0000 | 648.1783 | null |

## Direct Answers

### Does baseline reject `00_silence` better than checkpoint?

Yes. Checkpoint marks `95.5%` of frames as voiced/f0-covered, while pyin covers `48.7%`. Checkpoint median voiced probability is `0.5700`; pyin median voiced probability is `0.0100`. For this noise sample, lower f0 coverage is better, but pyin still produces too much f0 coverage to be trusted alone as a silence/noise rejector.

### Does baseline handle `03_sustained_aaa` with fewer octave/f0 jumps?

Yes. Checkpoint has `27` octave jumps and `31` large semitone jumps; pyin has `0` octave jumps and `0` large semitone jumps. Checkpoint trimmed range is `77.6632` Hz and stability is `549.3202` cents; pyin trimmed range is `25.1283` Hz and stability is `193.7593` cents.

### Does checkpoint or baseline better preserve `04_pitch_slide` direction?

Both methods preserve directional movement. Checkpoint slope is `7.8591` Hz/s with trimmed range `137.5797` Hz; pyin slope is `14.1668` Hz/s with trimmed range `170.5596` Hz. Pyin has fewer large jumps (`0` vs checkpoint `8`), so it is cleaner if the direction agrees.

### Does checkpoint or baseline produce cleaner structure for `05_twinkle_twinkle`?

The result is mixed. Pyin is cleaner on discontinuities, with `0` octave jumps and `0` large semitone jumps versus checkpoint `8` and `15`. Checkpoint has lower trimmed stability spread (`449.8915` cents vs pyin `648.1783` cents) and higher f0 coverage (`93.3%` vs `82.1%`). So pyin is cleaner as an artifact filter, while checkpoint may preserve more continuous melody coverage.

### Should the product use checkpoint-only, baseline-only, or a hybrid/ensemble for now?

Use a hybrid/ensemble for now. The checkpoint provides task-head outputs needed by the app, but its raw VAD is too permissive on noise and its f0 argmax can jump. The pyin baseline is better as a conservative sanity check for no-voice/noise and f0 stability, but baseline-only would discard the model's learned onset/breath/technique interfaces and can be less task-aware. Near-term product behavior should use checkpoint outputs only when they agree with conservative DSP sanity checks or pass confidence/consistency gates.

## Notes

- `librosa.pyin` voiced probability is not the same calibration target as the neural VAD probability; compare it qualitatively.
- There is no frame-level ground truth for these five samples, so this report compares plausibility and stability, not formal accuracy.
- The plots show waveform, checkpoint f0 vs baseline f0, and checkpoint voiced mask vs baseline voiced mask.
