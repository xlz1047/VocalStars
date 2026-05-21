# VocalStars: Team Setup & Project Summary

## Project Overview

**VocalStars** is an AI-powered beginner vocal coaching application with a modular architecture:
- **Frontend**: Next.js 14 + React 18 + TypeScript (UI at `frontend/`)
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL (API at `backend/`)
- **ML Pipeline**: Audio analysis (pitch, rhythm, breath, spectral features) at `ml/`
- **Database**: SQLite (dev) / PostgreSQL (production)
- **Tests**: Pytest suite with synthetic audio evaluation (`tests/`)
- **Deployment**: Docker + Docker Compose (dev-ready, prod tuning needed)

---

## Quick Start (5 minutes)

### 1. Clone & Setup Environment

```bash
# Clone the repo
git clone <repo-url>
cd VocalStars

# Create Python virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux bash)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Backend runtime
pip install -r backend/requirements.txt

# Development + testing (optional, recommended for testing/CI work)
pip install -r requirements-dev.txt

# Frontend
cd frontend
npm install
cd ..
```

### 3. Start Services Locally

**Terminal 1 — Backend (FastAPI):**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Server: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Terminal 2 — Frontend (Next.js):**
```bash
cd frontend
npm run dev
# App: http://localhost:3000
```

### 4. Test ML Pipeline (Optional)

```bash
# Run synthetic evaluation suite
python -m tests.evaluate_pitch
python -m tests.evaluate_pipeline

# Run pytest suite (requires requirements-dev.txt)
pytest tests/test_ml_pipeline.py -v
```

---

## Project Structure

```
VocalStars/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── main.py         # FastAPI app + router mounting
│   │   ├── api/routers/    # Endpoint definitions
│   │   ├── services/       # Business logic (audio processing, analysis)
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── core/           # Config, database setup
│   ├── requirements.txt    # Runtime dependencies
│   └── requirements-ml.txt # Heavy ML deps (optional)
│
├── frontend/               # Next.js React app
│   ├── app/                # App router + pages
│   ├── components/         # React components
│   ├── lib/                # Utilities (API client, helpers)
│   ├── package.json
│   └── tsconfig.json
│
├── ml/                     # ML pipeline modules
│   ├── pitch_detection/    # Pitch/F0 estimation
│   ├── rhythm_analysis/    # Tempo & beat tracking
│   ├── breath_analysis/    # Breath cycle detection
│   ├── feature_extraction/ # Spectral features (MFCC, centroid)
│   ├── coaching_engine/    # Maps features → beginner coaching
│   └── pipeline.py         # Orchestrates all modules
│
├── tests/                  # Test suite
│   ├── test_ml_pipeline.py       # Pytest test classes
│   ├── evaluate_pitch.py          # Standalone pitch evaluator
│   ├── evaluate_pipeline.py       # Full-pipeline evaluator
│   ├── visualization.py           # Plot generation utilities
│   ├── synthetic_data.py          # Synthetic audio generators
│   └── results/                   # Generated plots (auto-created)
│
├── .github/workflows/      # CI/CD
│   └── tests.yml           # GitHub Actions test workflow
│
├── docs/                   # Documentation
│   ├── setup.md           # Detailed setup guide
│   └── architecture.md    # System architecture
│
├── EVALUATION.md           # ML evaluation metrics & interpretation
├── requirements-dev.txt    # Dev + test dependencies
└── README.md               # Project overview (start here!)
```

---

## Key Features & Workflow

### 1. Audio Upload & Analysis

**User flow:**
1. User uploads a singing audio file via frontend (`AudioRecorder.tsx`)
2. Frontend POSTs to `POST /api/audio/upload`
3. Backend:
   - Saves file to `./audio_uploads/`
   - Runs ML pipeline (`ml.pipeline.analyze_audio_file()`)
   - Persists `SingingSession` + `AnalysisResult` to database
   - Returns session ID to frontend
4. Frontend fetches analysis + coaching recommendations
5. Frontend displays results

**Key files:**
- Frontend: `frontend/components/AudioRecorder.tsx`, `frontend/lib/api.ts`
- Backend: `backend/app/api/routers/audio_processing.py`, `backend/app/services/audio_processing.py`
- ML: `ml/pipeline.py` (orchestrator)

### 2. ML Pipeline Modules

Each module analyzes a different aspect:

| Module | Purpose | Output |
|--------|---------|--------|
| `pitch_detection` | F0 contour + stability | `stability_score`, `voiced_ratio`, `pitch_curve` |
| `rhythm_analysis` | Tempo + timing variance | `tempo`, `timing_variance`, `beat_alignment` |
| `breath_analysis` | Breath cycles + support | `breath_cycles`, `support_score`, `breath_length_variation` |
| `feature_extraction` | Spectral features | MFCC, spectral centroid, energy contour |
| `coaching_engine` | Maps features → recommendations | Beginner-friendly exercise suggestions |

**Key file:** `ml/pipeline.py` — entry point for all analysis

### 3. Database Models

Core entities:
- `User` — Basic user (email, display_name)
- `SingingSession` — Recording metadata (user_id, file_name, created_at)
- `AnalysisResult` — JSON summary from ML pipeline (session_id, summary)

**For local dev:** Uses SQLite (`sqlite:///vocalstars.db`)  
**For production:** Use PostgreSQL (configured via `DATABASE_URL` in `.env`)

### 4. Testing & Validation

**Synthetic evaluation suite** (`tests/`):
- Generates controlled audio with known ground truth
- Validates accuracy of pitch, rhythm, breath, spectral detection
- Plots results to PNG for inspection

**Run tests:**
```bash
pytest tests/test_ml_pipeline.py -v              # Pytest suite
python -m tests.evaluate_pitch                   # Pitch evaluator
python -m tests.evaluate_pipeline               # Full pipeline
```

**CI/CD:** GitHub Actions (`.github/workflows/tests.yml`) runs tests on every push to `main`/`develop`.

---

## Configuration

### Environment Variables (`.env` at repo root)

```bash
# Database
DATABASE_URL=sqlite:///vocalstars.db  # or postgresql://user:pass@localhost/dbname

# Frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Optional ML features (commented out by default)
# ENABLE_TORCHCREPE=1
# ENABLE_PARSELMOUTH=1
```

**For local dev:** Defaults are fine (SQLite, localhost:8000).  
**For production:** Update `DATABASE_URL` to PostgreSQL, set `NEXT_PUBLIC_API_BASE_URL` to your domain.

---

## Common Tasks

### Run Backend Only
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### Run Frontend Only
```bash
cd frontend
npm run dev
```

### Test ML Modules
```bash
pytest tests/test_ml_pipeline.py::TestPitchDetection -v
pytest tests/test_ml_pipeline.py::TestRhythmDetection -v
```

### View Test Plots
Plots auto-save to `tests/results/`:
- `pitch_curve.png` — Estimated vs ground-truth F0
- `energy_envelope.png` — RMS energy + detected breaths
- `beat_alignment.png` — Beat timeline + inter-beat intervals
- `note_errors.png` — Per-note estimation errors

### Generate Test Coverage Report
```bash
pytest tests/test_ml_pipeline.py --cov=ml --cov-report=html
# View: open htmlcov/index.html
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'librosa'` | `pip install -r backend/requirements.txt` |
| `NEXT_PUBLIC_API_BASE_URL not set` | Add to `.env` or `.env.local` in `frontend/` |
| `Database locked (SQLite)` | Stop backend, delete `vocalstars.db`, restart |
| Pitch detection very inaccurate | Check `fmin`/`fmax` in `ml/pitch_detection/pitch_detector.py` |
| Tests fail with matplotlib error | `pip install -r requirements-dev.txt` |

---

## Architecture Decisions

### Why Separate Runtime & ML Dependencies?
- `requirements.txt` — lightweight, fast installation for dev/demo
- `requirements-ml.txt` — heavy deps (torch, parselmouth) separated; only install if needed
- Benefit: Quick local iteration without waiting for ML library builds

### Why Synthetic Test Audio?
- No need for labeled real recordings
- Controlled ground truth for objective metrics
- Fast iteration on algorithm improvements
- CI/CD friendly (no external data downloads)

### Why SQLite for Local Dev?
- Zero setup, no separate database server needed
- Files check into version control easily (dev.db)
- Production uses PostgreSQL (configured in Docker Compose)

---

## Next Steps for Contributors

1. **Familiarize yourself** with the directory structure (see above)
2. **Run locally** following "Quick Start"
3. **Read** `EVALUATION.md` to understand ML testing approach
4. **Pick a task** from the issues/project board
5. **Create a feature branch**: `git checkout -b feature/your-feature-name`
6. **Commit & push**: Follow git flow (see below)
7. **Create a Pull Request** with description of changes

---

## Git Workflow & PR Guidelines

### Before Creating a PR

- [ ] Run `pytest tests/test_ml_pipeline.py -v` — all tests pass
- [ ] Run `npm run lint` (frontend) if available
- [ ] Test manually on localhost
- [ ] Update relevant docs (EVALUATION.md, README, etc.) if code changes
- [ ] Commit message is clear: `feat: add vibrato detection` or `fix: pitch RMSE calculation`

### PR Checklist

- [ ] Branch name: `feature/xyz`, `fix/xyz`, or `docs/xyz`
- [ ] Commits are logical and atomic (not one giant commit for multiple features)
- [ ] No secrets (API keys, passwords) in code or commits
- [ ] If ML changes: include test coverage or evaluation results
- [ ] If DB schema changes: document migration steps

---

## Questions?

- **Architecture**: See `docs/architecture.md`
- **ML Metrics**: See `EVALUATION.md`
- **Setup issues**: Check `docs/setup.md`
- **Code style**: Follow PEP 8 (Python), Prettier (JS/TS)

---

**Last Updated:** 2026-05-21  
**Maintainer:** [Your Name]
