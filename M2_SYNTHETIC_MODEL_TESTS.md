# M2 Synthetic Model Tests

Goal: test checkpoint raw VAD/f0 behavior against a DSP baseline on controlled synthetic audio.

No retraining, model architecture changes, scoring tuning, or P4 regression expectation changes were made.

## Generated WAV Files

- Directory: `samples/synthetic_model_tests`
- `samples/synthetic_model_tests/digital_silence_5s.wav`
- `samples/synthetic_model_tests/white_noise_5s.wav`
- `samples/synthetic_model_tests/low_hum_80hz_5s.wav`
- `samples/synthetic_model_tests/sine_220hz_5s.wav`
- `samples/synthetic_model_tests/sine_440hz_5s.wav`
- `samples/synthetic_model_tests/sine_sweep_220_to_440_5s.wav`
- `samples/synthetic_model_tests/pulsed_220hz_voiced_unvoiced.wav`

## Outputs

- JSON summary: `reports/synthetic_model_tests/summary.json`

## Results

| Sample | Method | Expected f0 | Predicted median f0 | Error cents | Voiced % | False voiced % | Jumps | Confidence | More reliable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `digital_silence_5s` | checkpoint | null | 610.328 | null | 0.2% | 0.2% | 0 | 0.076 | baseline |
| `digital_silence_5s` | baseline pyin | null | null | null | 0.0% | 0.0% | 0 | 0.000 |  |
| `white_noise_5s` | checkpoint | null | 288.036 | null | 49.9% | 49.9% | 53 | 0.100 | neither |
| `white_noise_5s` | baseline pyin | null | 34.247 | null | 51.3% | 51.3% | 0 | 0.010 |  |
| `low_hum_80hz_5s` | checkpoint | 80.000 | 391.957 | 2751.149 | 99.8% |  | 2 | 0.257 | baseline |
| `low_hum_80hz_5s` | baseline pyin | 80.000 | 80.053 | 1.149 | 100.0% |  | 0 | 0.827 |  |
| `sine_220hz_5s` | checkpoint | 220.000 | 219.979 | -0.169 | 100.0% |  | 0 | 0.390 | mixed |
| `sine_220hz_5s` | baseline pyin | 220.000 | 219.979 | -0.169 | 100.0% |  | 0 | 0.985 |  |
| `sine_440hz_5s` | checkpoint | 440.000 | 439.957 | -0.169 | 100.0% |  | 0 | 0.344 | mixed |
| `sine_440hz_5s` | baseline pyin | 440.000 | 439.957 | -0.169 | 100.0% |  | 0 | 0.985 |  |
| `sine_sweep_220_to_440_5s` | checkpoint | 330.000 | 329.595 | -2.124 | 100.0% |  | 0 | 0.344 | mixed |
| `sine_sweep_220_to_440_5s` | baseline pyin | 330.000 | 329.595 | -2.124 | 100.0% |  | 0 | 0.985 |  |
| `pulsed_220hz_voiced_unvoiced` | checkpoint | 220.000 | 219.979 | -0.169 | 64.3% |  | 4 | 0.255 | baseline |
| `pulsed_220hz_voiced_unvoiced` | baseline pyin | 220.000 | 219.979 | -0.169 | 59.9% |  | 0 | 0.757 |  |

## Interpretation

- `digital_silence_5s` and `white_noise_5s` should have near-zero voiced output. Any voiced percentage there is false voiced behavior.
- Pure sine waves are not human singing, but they are useful for measuring f0 binning, octave errors, and confidence.
- `sine_sweep_220_to_440_5s` uses an expected median f0 of 330 Hz for the cents-error column; direction and continuity matter more than that single number.
- `pulsed_220hz_voiced_unvoiced.wav` should show roughly half voiced coverage if the detector respects the silent gaps.

## Overall Takeaway

Baseline is more reliable on `3` synthetic cases, checkpoint on `0`, with `4` mixed/neither cases. These tests reinforce the M1 recommendation: use a hybrid/ensemble for now, with DSP checks guarding checkpoint VAD/f0 before product-facing scoring or coaching.
