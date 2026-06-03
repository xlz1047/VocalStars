# CLAUDE.md

## Project
AI voice coaching app — ml/ module.
Target users: casual singers (shower/car). Mobile-first inference.
I am responsible for the entire ml/ directory only.

## Existing Structure (DO NOT MODIFY without reading first)
- ml/rhythm_analysis/rhythm_detector.py — HAS EXISTING CODE, extend carefully
- ml/pipeline.py — main interface, backend imports from here
- ml/__init__.py — exists

## My Branch
Working on branch: Ush

## Tech Stack
Python 3.11, PyTorch, torchaudio, librosa
Export: ONNX → TFLite (int8 quantized)
Experiment tracking: Weights & Biases

## Model Constraints
- Final exported model < 5MB after int8 quantization
- Inference latency < 50ms on mid-range Android
- Input: 16kHz mono audio, 256ms frames (4096 samples), 50% overlap
- Output: pitch_hz, intonation_cents, rhythm_score, breath_bool, feedback strings

## Folder Structure (feature-based — follow this convention)
ml/
  breath_analysis/      — extractor.py, model.py, detector.py
  coaching_engine/      — scorer.py, feedback.py
  feature_extraction/   — mel.py, audio_utils.py, label_utils.py
  pitch_detection/      — model.py, detector.py
  rhythm_analysis/      — rhythm_detector.py (EXISTS), model.py, extractor.py
  _model/               — backbone.py, voice_coach.py  (shared backbone)
  data/                 — dataset loaders
  training/             — train.py, losses.py, evaluate.py, config/
  export/               — to_onnx.py, to_tflite.py, benchmark.py
  pipeline.py           — EXISTS — only interface backend uses

## Pattern for Every Feature Folder
extractor.py  → pure DSP, no PyTorch, librosa/numpy only
model.py      → PyTorch nn.Module head
detector.py   → combines both, what pipeline.py calls

## Datasets (will be placed in ml/data/raw/)
- VocalSet
- MIR-1K
- GTSinger
- MedleyDB
- CSD (Children's Song Dataset)
- DSing
- NUS-48E

## Code Rules
- No comments attributing code to any tool or external source
- Docstrings on all public classes and functions
- Type hints throughout
- PEP8, max line length 100
- Tests go in ml/tests/ for every new module