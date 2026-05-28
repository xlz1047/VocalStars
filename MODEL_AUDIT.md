# VocalStars Model Audit

Audit date: 2026-05-28

Scope: repository inspection only. No training was run. No source files were modified while preparing this audit.

## Executive summary

VocalStars currently has two ML stacks:

- `ml_new/` is the active neural singing-analysis stack. It contains feature extraction, pretrained checkpoints, training scripts, evaluation scripts, and the main `analyse_recording()` inference API.
- `ml/` is an older stack. It still contains a `VoiceCoachModel`, dataset loaders, training code, export stubs, and a backend-upload path, but several modules are placeholders or heuristic-only.

The app can analyze uploaded/recorded audio through FastAPI and `ml_new`, but the default `analyze-with-ml` endpoint does not use `settings.ML_CHECKPOINT`; unless the caller supplies `checkpoint_path`, it falls back to `librosa.pyin` plus heuristics. The newer `/api/coaching/analyse` endpoint does use `settings.ML_CHECKPOINT`.

The best-developed task is pitch/f0. Current logs show high-resolution pitch checkpoints around 93% validation RPA, and VAD around 96% F1. Breath and onset are much weaker by F1 despite high recall/accuracy numbers. Technique/timbre classification is not reliable enough for coaching; unified technique accuracy is around 13-16%, and the acoustic fine-tune log reaches roughly 39% validation accuracy at best in the visible log. There is no clear evaluation against human coaching labels, accompaniment/noise conditions, browser/WebAssembly deployment, or real-time chunked inference.

## 1. Repo structure

Top-level structure relevant to the model and app:

- `new_frontend/`: Vite + React frontend. This is the active-looking frontend for recording, live pitch display, and results.
- `old_frontend/`: older Next.js frontend.
- `frontend/`: another Next.js app/build tree, including `.next/` and `node_modules/`; not listed by `rg --files`, likely not tracked fully.
- `backend/`: FastAPI app, schemas, services, routers, tests, and dependency files.
- `ml_new/`: newer ML pipeline with HCQT features, unified model, inference, checkpoints, training, and evaluation.
- `ml/`: older ML pipeline with log-mel/NanoPitch-style model, heuristic analysis modules, training, export, and mostly empty tests.
- `shared/`: shared TypeScript docs/types.
- `docs/`: architecture and integration docs.
- `tests/`: contains only Python bytecode cache files from evaluation scripts, not real source tests.

Important generated/heavy-looking directories present in the repo:

- `ml_new/checkpoints/`: trained `.pt` files and training logs.
- `ml_new/data/extracted_*`: manifests and likely extracted feature files.
- `ml/data/raw/`: raw singing datasets.
- `ml/.venv/`, `frontend/node_modules/`, `new_frontend/node_modules/`: local dependency trees.
- `backend/audio_uploads/`: uploaded/test audio.

## 2. Frontend/backend/model/training/evaluation files

Frontend:

- `new_frontend/src/components/StudioView.tsx`: session UI, microphone recording flow, live pitch scoring, ML submission, and fallback mock result behavior.
- `new_frontend/src/utils/audioAnalysis.ts`: `MediaRecorder` WebM capture, upload to `/api/audio/analyze-with-ml`, and conversion from ML result to `PerformanceResult`.
- `new_frontend/src/utils/pitchDetector.ts`: simple autocorrelation pitch detector for live browser feedback.
- `new_frontend/src/components/ResultsView.tsx`: displays final results and ML analysis when present.
- `new_frontend/src/types.ts`: frontend `MLAnalysisResult`, `PerformanceResult`, note, frame, and voice-quality types.

Backend:

- `backend/app/api/routers/audio_processing.py`: `/api/audio/upload` and `/api/audio/analyze-with-ml`.
- `backend/app/api/routers/coaching.py`: `/api/coaching/analyse`, which runs `ml_new.inference.coach_inference.analyse_recording()` using configured checkpoint/device.
- `backend/app/services/ml_inference.py`: wrapper around `ml_new` for frontend-shaped JSON.
- `backend/app/services/ml_serialiser.py`: serializer for raw `CoachingResult` into Pydantic response shape.
- `backend/app/services/audio_processing.py`: older upload path calling `ml.pipeline.analyze_audio_file()`.
- `backend/app/schemas/coaching_result.py`: raw `CoachingResult`-style API schema.
- `backend/app/core/config.py`: defines `ML_CHECKPOINT` and `ML_DEVICE`, but these are not used by `MLInferenceService`.

Current model and inference:

- `ml_new/inference/coach_inference.py`: main public inference API, fallback path, model path, scoring, and rule-based coaching text.
- `ml_new/inference/algorithms.py`: note segmentation, vibrato, and voice-quality helpers.
- `ml_new/models/unified_model.py`: main multi-task `UnifiedVocalModel`.
- `ml_new/models/pitch_model.py`, `vad_model.py`, `breath_model.py`, `onset_model.py`: separate task-specific models.
- `ml_new/models/acoustic_technique.py`: frozen-backbone plus acoustic-feature technique classifier.
- `ml_new/models/viterbi.py`: pitch decoding helper used by evaluation.

Training/data:

- `ml_new/data/extract_all.py`: dataset walker and HCQT/VAD/f0/VAD-label extraction.
- `ml_new/data/feature_dataset.py`: windowed dataset for separate pitch/VAD/breath/onset models.
- `ml_new/data/unified_dataset.py`: windowed multi-task dataset for unified training.
- `ml_new/data/merge_manifests.py`, `build_manifest_from_dir.py`: manifest utilities.
- `ml_new/training/train_unified.py`: unified multi-task training.
- `ml_new/training/train_pitch.py`, `train_vad.py`, `train_breath.py`, `train_onset.py`: separate task training.
- `ml_new/training/finetune_technique.py`: acoustic technique classifier fine-tune.
- `ml_new/training/run_all.sh`: partial chained training helper.

Evaluation:

- `ml_new/training/evaluate_pitch.py`: pitch checkpoint evaluation with argmax, weighted mean, and Viterbi decoding.
- `ml_new/training/evaluate_coaching.py`: NPZ-based coaching evaluation, but currently not a full raw-audio end-to-end evaluation.
- Checkpoint logs/results under `ml_new/checkpoints/*`.

Older/legacy ML:

- `ml/_model/voice_coach.py`, `ml/_model/backbone.py`: log-mel NanoPitch-style multi-task model.
- `ml/pipeline.py`: older public analysis API.
- `ml/pitch_detection/`, `ml/rhythm_analysis/`, `ml/breath_analysis/`, `ml/feature_extraction/`, `ml/training/`, `ml/export/`: mixed implemented, placeholder, and legacy code.

## 3. Current model architecture

Active `ml_new` architecture:

- Input audio is loaded mono at 16 kHz.
- Frame hop is 160 samples, so inference frame rate is 10 ms.
- HCQT feature input is shaped `(B, 6, 180, T)` in active model inference and unified training.
- Handcrafted VAD features are shaped `(B, 3, T)`:
  - RMS energy
  - spectral flatness
  - zero crossing rate

`UnifiedVocalModel`:

- Harmonic fusion over HCQT:
  - Conv1d `6 -> 16`, ReLU
  - Conv1d `16 -> 8`, kernel 5, ReLU
  - Conv1d `8 -> 1`
- LayerNorm over frequency bins.
- Two-layer GRU, hidden size 128, causal in the sense that the GRU consumes time left-to-right.
- Task heads:
  - pitch head: per-frame linear `128 -> 180`
  - voiced/VAD head: linear over `128 + 3` features to sigmoid probability
  - breath head: linear over `128 + 3` features to sigmoid probability
  - onset head: `128 -> 64 -> 1` MLP with sigmoid
  - technique head: mean-pool GRU output over time, then `128 -> 64 -> 20`
- Pitch bins use C1-ish `FMIN = 32.7 Hz`, 36 bins per octave, 180 bins total.

Technique classifier extension:

- `AcousticTechniqueClassifier` concatenates the frozen backbone clip representation `(128)` with 10 clip-level acoustic features:
  - RMS mean/std
  - flatness mean/std
  - ZCR mean/std
  - voiced ratio
  - normalized f0 mean/std/range
- It predicts the same 20 technique classes. It stores feature normalization buffers in the checkpoint.

Separate task models:

- `PitchModel`: similar HCQT harmonic fusion plus smaller GRU and pitch/VAD heads.
- `VADModel`: HCQT mean over frequency plus VAD features, two-layer GRU, sigmoid voicing.
- `BreathModel`: VAD features only, two-layer GRU, sigmoid breath probability.
- `OnsetModel`: first HCQT harmonic only, Conv1d projection, GRU, sigmoid onset probability.

Older `ml/` architecture:

- `VoiceCoachModel` uses 40-band log-mel input `(B, T, 40)`.
- `NanoPitchEncoder` uses causal Conv1d layers and three GRUs, concatenating features to `(B, T, 384)`.
- Heads predict pitch bins/VAD, rhythm onset probabilities, and a clip-level breath probability.
- This stack is not the main path for the new frontend ML endpoint.

## 4. Current training pipeline

Feature extraction:

- `ml_new/data/extract_all.py` walks GTSinger, VocalSet, and PopBuTFy under `ml/data/raw`.
- It resamples to 16 kHz mono.
- It extracts HCQT, VAD features, f0, VAD labels, and `voiced_probs`.
- F0 label extraction can use:
  - `fast`: `librosa.yin` plus RMS-based voicing.
  - `accurate`: `librosa.pyin` with voicing probabilities.
- GTSinger VAD can come from phoneme JSON boundaries.
- VocalSet/PopBuTFy VAD falls back to RMS thresholding.
- Manifests include `npz_path`, original `audio_path`, `dataset`, `singer_id`, `technique`, and `n_frames`.

Unified training:

- `UnifiedDataset` reads extracted `.npz` files and samples random fixed windows, default 200 frames, about 2 seconds.
- Splits are singer-level by default for train/val/test.
- Training applies optional pitch-shift augmentation and median f0 smoothing on train windows.
- Breath labels are not manually annotated; they are derived from VAD plus RMS/ZCR heuristics.
- Onset labels are not manually annotated; they are derived from f0 starts and pitch jumps.
- Technique labels are read from manifest strings and mapped into a 20-class vocabulary; unknown labels are masked out.

Unified losses:

- Pitch: Gaussian soft-label cross-entropy on voiced frames.
- Voiced: focal BCE.
- Breath: focal BCE with high positive weight.
- Onset: focal BCE with high positive weight.
- Technique: weighted cross-entropy over known technique labels.
- Total loss weights: pitch 1.0, voiced 0.6, breath 0.9, onset 0.9, technique 0.4.

Separate training:

- Pitch/VAD/breath/onset scripts train task-specific models from the same extracted features.
- `finetune_technique.py` trains only the acoustic technique classifier on top of a frozen unified backbone using a technique-stratified split, not a singer split.

Observed training/evaluation logs:

- `pitch_hires`: last visible epoch has validation RPA around 0.9305, RCA around 0.9396, VDR around 0.963, VFA around 0.056, median error around 11.2 cents.
- `pitch_hires/results_baseline_vs_pyin_labels.json`: argmax RPA around 0.931 vs pyin labels.
- `vad`: last visible epoch has validation F1 around 0.9656.
- `breath`: last visible epoch has recall around 0.879 but precision around 0.217 and F1 around 0.348.
- `onset`: last visible epoch has precision around 0.272, recall around 0.292, and F1 around 0.282.
- `unified`: last visible epoch has RPA around 0.873, VAD F1 around 0.959, breath recall around 0.990 but breath F1 around 0.035, onset recall around 0.936 but onset F1 around 0.056, technique accuracy around 0.160.
- `unified_tech/finetune_acoustic_log.csv`: visible best validation accuracy around 0.3905.

## 5. Current inference pipeline

Frontend final analysis flow:

1. `StudioView.tsx` records microphone audio through `MediaRecorder`.
2. `audioAnalysis.ts` creates a WebM/Opus blob and POSTs multipart form data to `/api/audio/analyze-with-ml`.
3. `backend/app/api/routers/audio_processing.py` writes the upload to a temporary file.
4. It creates `MLInferenceService(checkpoint_path=Path(checkpoint_path) if checkpoint_path else None)`.
5. `MLInferenceService.analyze_audio()` calls `ml_new.inference.coach_inference.analyse_recording()`.
6. `analyse_recording()` loads audio with `librosa.load(..., sr=16000, mono=True)`.
7. If a checkpoint path exists, `_run_model()` computes HCQT/VAD features and runs `UnifiedVocalModel`.
8. If no checkpoint is passed or the path is missing, `_run_fallback()` runs `librosa.pyin` and derives breath/onset labels heuristically.
9. `_build_result()` computes note segmentation, pitch accuracy, pitch drift, phrase lengths, breath count, onset count, onset clarity, voice quality, vibrato stats, score, summary, issues, and exercises.
10. Backend formats the result to frontend camelCase fields and downsamples frame data by a factor of 10.

Important wiring issue:

- `backend/app/core/config.py` defines `ML_CHECKPOINT`, defaulting to `ml_new/checkpoints/unified/best.pt`.
- `/api/coaching/analyse` uses this setting.
- `/api/audio/analyze-with-ml` does not use this setting. It only uses a query parameter `checkpoint_path`. The frontend does not pass that parameter. Therefore the normal new-frontend path likely runs fallback pyin/heuristics, not the trained unified checkpoint.

Real-time frontend flow:

- `new_frontend/src/utils/pitchDetector.ts` performs simple autocorrelation in the browser for live pitch.
- This browser live detector is not the same preprocessing/model as backend final inference.
- There is no checked-in browser deployment of `UnifiedVocalModel`, ONNX, TFLite, WebGPU, WebNN, or WASM inference for `ml_new`.

## 6. Input/output formats

Raw audio input:

- Frontend recording: `audio/webm;codecs=opus` from `MediaRecorder`.
- Backend accepts any `content_type` starting with `audio/` for `/api/audio/analyze-with-ml`.
- Backend temporary suffix comes from the uploaded filename, usually `.webm`.
- Inference loads via `librosa`, so actual WebM/Opus support depends on the installed audio backend/ffmpeg stack.

Feature input:

- Audio is resampled to 16 kHz mono.
- HCQT:
  - model path: `(6, 180, T)` float32 log-magnitude HCQT.
  - extraction script defaults are currently `bins_per_octave=12`, `n_bins=60`, but active high-resolution/unified training and inference use 36/180.
- VAD features:
  - `(3, T)` float32: RMS, spectral flatness, ZCR.
- F0 labels:
  - `(T,)` float32 Hz, 0.0 for unvoiced.
- VAD labels:
  - `(T,)` uint8/float, 1 voiced, 0 unvoiced.
- Breath/onset labels:
  - `(T,)` derived binary float arrays.

Backend `/api/audio/analyze-with-ml` response:

- Top-level:
  - `status`: `"success"` or `"error"`
  - `data`: frontend-shaped object or `null`
  - `error`: only on error
- Main data fields:
  - `score`, `summary`, `issues`, `exercises`
  - `songTitle`, `artist`
  - `pitchAccuracy` as percent
  - `pitchDrift` in cents
  - `phraseLengths` in seconds
  - `breathCount`, `onsetCount`, `onsetClarity`
  - `technique`, `techniqueConfidence` as percent, `allTechniqueScores` as percents
  - `notes` with `startSeconds`, `durationSeconds`, `pitchHz`, `noteName`, `centsError`, `stabilityCents`, optional vibrato
  - `voiceQuality` with HNR/jitter/shimmer/breathiness/instability when available
  - `vibrato`
  - `frameData` with downsampled `pitch`, `voiced`, `breath`, `onset`, and adjusted `hopLength`

Backend `/api/coaching/analyse` response:

- Uses raw snake_case `CoachingResult` fields as declared in `backend/app/schemas/coaching_result.py`.
- Includes full per-frame arrays rather than the downsampled frontend shape.

Frontend final type:

- `MLAnalysisResult` in `new_frontend/src/types.ts` expects camelCase fields from `/api/audio/analyze-with-ml`.
- `PerformanceResult` maps:
  - overall score from `score`
  - intonation from `pitchAccuracy`
  - rhythm from `onsetClarity * 100`
  - timbre from `techniqueConfidence`
  - dynamics from `voiceQuality.hnrDb * 2.5`

## 7. Do training and inference preprocessing match?

Partially, but there are serious caveats.

Matches:

- Both active unified training and model inference use 16 kHz mono audio and 10 ms hop.
- Both use `HCQTExtractor` and `VADFeatureExtractor`.
- Both use HCQT and VAD features with 180 bins and 36 bins/octave in the active unified model path.
- The acoustic technique helper is reused in training and inference.

Mismatches / risks:

- `extract_all.py` defaults to `bins_per_octave=12`, `n_bins=60`, while `UnifiedVocalModel` and `_run_model()` expect 36/180. Existing manifests appear to point at `extracted_pyin`/merged data that likely used high-resolution settings, but the default extraction command in the script can produce features incompatible with the unified model.
- Training windows are random 2-second crops. Inference runs the full clip and mean-pools the entire clip for technique. This is acceptable for sequence models, but technique training/fine-tuning uses random HCQT windows while acoustic features come from the full clip.
- Training f0 labels often come from YIN/pyin pseudo-labels. Inference pitch is judged partly by nearest semitone, not against song reference pitch or human ground truth.
- Training has optional pitch-shift augmentation; inference does not need augmentation, but no documented normalization/calibration step ensures consistent audio level/noise conditions.
- Browser live pitch preprocessing is completely different from backend inference.
- Breath and onset targets are derived heuristics, not independently annotated labels. The model can learn the heuristic rather than real breath/timing events.
- `evaluate_coaching.py` is NPZ-based and uses stored features, so it does not verify raw upload decoding, resampling, WebM handling, temporary-file path behavior, or backend formatting.

## 8. Does the model actually predict what the app needs?

Pitch/f0:

- Mostly yes for absolute pitch tracking. The separate `pitch_hires` checkpoint appears strong against pyin labels.
- However, the app goal is not just pitch tracking; it needs singing accuracy relative to the song. Current `pitch_accuracy` measures closeness to the nearest equal-tempered semitone, not closeness to a reference melody, lyric timing, key, or song arrangement. A singer can sing the wrong melody in tune and still score well.

Voiced frames/VAD:

- Mostly yes. VAD is well represented in `ml_new`, and logs show strong F1.
- Browser live feedback also has a rough silence threshold, but that is not the trained VAD model.

Rhythm/timing:

- Not yet in the way the app needs. `ml_new` detects note onsets, but the current onset F1 logs are weak. It does not align user onsets against a song reference timeline.
- Older `ml/rhythm_analysis/rhythm_detector.py` is a placeholder returning unknown tempo/timing values.

Breath:

- Partially. The unified and separate breath models output breath frames, but labels are heuristic and current precision/F1 are poor. Breath count can be noisy.
- The coaching text uses phrase length as a proxy for breath support. This can be useful, but it is not the same as diagnosing breath technique.

Timbre/technique:

- Not reliably. The technique vocabulary is broad and dataset/style-derived, not directly mapped to beginner technique issues.
- Unified technique accuracy is very low in logs. The acoustic fine-tune is better but still not strong enough for confident coaching.
- Voice quality metrics from parselmouth, if available, provide HNR/jitter/shimmer and may be more directly useful for breathiness/instability than the 20-class technique head.

Real-time feedback:

- The active neural model is designed causally, and GRU hidden-state helpers exist, but app inference is currently full-file backend inference.
- The in-browser real-time feedback uses a simple autocorrelation detector and simulated/scored UI logic, not the trained model.
- There is no active browser model export path for `ml_new`.

Overall:

- Current implementation can provide a final overview with pitch, voicing, rough phrase/onset/breath, voice quality, and rule-based feedback.
- It does not yet provide reliable song-relative intonation/rhythm grading, robust technique diagnosis, or trained in-browser inference.

## 9. Missing tests/evaluation

Tests present:

- `backend/tests/test_ml_endpoint.py` checks non-audio rejection and, if a local WAV fixture exists, response shape/form fields/range for `/api/audio/analyze-with-ml`.
- `ml/tests/test_pipeline.py`, `ml/tests/test_model.py`, and `ml/tests/test_features.py` contain only docstrings.

Major missing tests:

- Unit tests for `HCQTExtractor`, `VADFeatureExtractor`, label extraction, breath/onset label derivation, and shape alignment.
- Tests that `UnifiedVocalModel` accepts expected feature shapes and rejects mismatched 60-bin features.
- Tests for `analyse_recording()` fallback vs checkpoint path behavior.
- Tests that `/api/audio/analyze-with-ml` uses the configured checkpoint by default, or intentionally documents fallback mode.
- Tests for WebM/Opus upload decoding in the backend environment.
- Tests for `MLInferenceService._format_coaching_result()` and frontend `MLAnalysisResult` compatibility.
- Frontend tests for fallback behavior, ML result rendering, and type/schema drift.
- Regression tests for score calculation, issue selection, note segmentation, vibrato, and voice-quality serialization.
- Tests for temporary file cleanup on success and error.

Major missing evaluation:

- Raw-audio end-to-end test set evaluation, not only NPZ feature evaluation.
- Evaluation against human-annotated f0/voicing/onset/breath labels.
- Song-relative pitch/rhythm evaluation against target melodies.
- Calibration curves/threshold selection for breath and onset probabilities.
- Dataset breakdown by singer, dataset, voice type, gender, register, technique, genre, and noise condition.
- Robustness to microphone recordings, WebM compression, room noise, accompaniment, reverb, clipping, phones/laptops, and very short clips.
- Human coaching validation: whether generated issues/exercises are correct, helpful, and safe.
- Browser performance/export evaluation.

## 10. Redundant/dead/suspicious code

Suspicious or redundant:

- `ml/` and `ml_new/` are parallel ML stacks. The app currently uses both depending on endpoint:
  - `/api/audio/upload` uses old `ml.pipeline`.
  - `/api/audio/analyze-with-ml` uses `ml_new` wrapper but likely fallback by default.
  - `/api/coaching/analyse` uses `ml_new` with configured checkpoint.
- `backend/app/core/config.py` has `ML_CHECKPOINT` and `ML_DEVICE`, but `MLInferenceService` hardcodes CPU and ignores settings unless a checkpoint is manually passed to the service.
- `ml_new/data/extract_all.py` docstring says HCQT `(6, 60, T)` and defaults to 60 bins, while active unified model/inference expects `(6, 180, T)`.
- `evaluate_coaching.py` calls `_build_coaching_text()` with missing newer arguments (`notes`, `voice_quality`, `vib_stats`) in the inspected code path, which appears stale relative to the current function signature.
- `ml_new/training/evaluate_coaching.py` imports `_build_result` but evaluates from NPZ without raw audio; it is not actually "what a backend call will eventually run" despite its docstring.
- `ml_new/presentation.html` contains audit-like statements that technique is only around 16% and not presentation-grade, but the app still surfaces technique in frontend scoring.
- `new_frontend/src/components/StudioView.tsx` falls back to mock coaching with random metrics if ML analysis fails or no recording exists. This can mask backend/model failures and make demos look successful when no model ran.
- `new_frontend/src/utils/audioAnalysis.ts` maps `techniqueConfidence` directly to `timbre`, and maps HNR to `dynamics`; those are not validated measures of timbre or dynamics.
- `ml/rhythm_analysis/rhythm_detector.py` is a placeholder.
- `ml/feature_extraction/features.py` is a placeholder.
- `ml/breath_analysis/detector.py` says neural model loading exists, but analysis always uses DSP heuristic because backbone embeddings are not wired.
- Several legacy dataset loaders in `ml/data/` raise `NotImplementedError`.
- `ml/tests/*` are effectively empty.
- Local dependency/build/cache directories are present in the workspace and create noise for repository inspection.

Potential bugs:

- Normal frontend endpoint likely never uses the trained checkpoint.
- Temporary file cleanup in `/api/audio/analyze-with-ml` happens after inference, but if inference raises before cleanup inside the inner try, the temp file may remain.
- WebM decoding depends on environment support; there is no explicit conversion step.
- `FeatureDataset` only loads `f0_hz` for task `"pitch"` or `"both"`, so task `"onset"` directly reads `data["f0_hz"]` later and works, but the docstring is easier to misread than the code.
- Singer-level splits protect pitch/VAD evaluation from singer leakage, but `finetune_technique.py` uses technique-stratified split and may allow singer overlap across splits.
- Technique labels from folder names/dataset metadata may encode dataset/source artifacts as much as vocal technique.

## 11. Recommended next steps

1. Fix backend checkpoint wiring first.
   - Make `/api/audio/analyze-with-ml` use `settings.ML_CHECKPOINT` and `settings.ML_DEVICE` by default.
   - Keep explicit fallback mode only when intentionally requested.
   - Add a response/debug flag indicating whether inference used `"checkpoint"` or `"fallback"`.

2. Define the app's target outputs as contracts.
   - Separate "measured acoustic facts" from "coaching interpretation".
   - Do not call technique/timbre/dynamics scores reliable until they are measured against appropriate labels.
   - Add a song-relative representation: reference f0 contour, note events, timing windows, lyrics/sections, and tolerance rules.

3. Build a real evaluation set.
   - Include raw microphone/WebM recordings.
   - Include reference melody and timing.
   - Include human labels for at least f0/voicing/onset and a small set of coaching issue categories.
   - Track metrics per dataset/voice type/noise condition.

4. Improve onset/rhythm before using rhythm scores.
   - Use annotated note onsets or align to score/MIDI/reference audio.
   - Calibrate thresholds for precision/recall tradeoff.
   - Report rhythm as timing error against target events, not onset clarity alone.

5. Treat breath and technique conservatively.
   - Keep breath output as "possible breath/noise/phrase support" until labels improve.
   - Prefer HNR/jitter/shimmer and phrase metrics for cautious feedback.
   - Hide or down-rank 20-class technique labels in the user UI until validation is strong.

6. Unify or retire ML stacks.
   - Decide whether `ml_new` replaces `ml/`.
   - Remove or quarantine legacy placeholders from production paths.
   - Update docs so architecture, default commands, active endpoints, and model dimensions agree.

7. Add focused tests.
   - Backend: checkpoint/default behavior, WebM fixture, error cleanup, schema compatibility.
   - ML: extractor shape tests, model forward shape tests, fallback/model parity smoke tests, label derivation tests.
   - Frontend: result rendering and no-silent-mock-success behavior.

8. Plan browser deployment explicitly.
   - Choose ONNX Runtime Web, WebGPU/WebNN, TensorFlow.js, or a smaller custom WASM path.
   - Export and benchmark the actual `ml_new` model or a distilled streaming pitch/VAD model.
   - Match browser preprocessing to training/inference preprocessing, or train a model for the browser feature path.

9. Make failures visible in the product.
   - Avoid random polished fallback coaching when ML fails.
   - Surface "analysis unavailable" or "basic live pitch only" states.
   - Log model path, inference mode, audio duration, frame count, and processing time.

10. Recalibrate scoring.
    - Current score is a weighted heuristic over pitch-to-nearest-semitone, voice quality, onset clarity, and phrase support.
    - Replace or supplement it with song-relative accuracy and confidence intervals.
    - Keep issue/exercise generation deterministic and traceable to measured signals.

