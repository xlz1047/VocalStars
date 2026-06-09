# VocalStars New Frontend + ML Integration Setup

## Overview

This guide covers setting up the new VocalStars frontend (built with Vite + React) integrated with the ML models from `ml_new`.

### Key Components
- **Frontend**: `new_frontend/` - Vite + React + TypeScript application
- **Backend API**: `backend/app/api/routers/audio_processing.py` - FastAPI audio analysis endpoint
- **ML Inference**: `ml_new/inference/coach_inference.py` - Unified vocal model analysis
- **ML Models**: `ml_new/checkpoints/` - Pre-trained models (pitch, breath, onset, VAD)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       New Frontend                           │
│           (React + Vite + TypeScript)                       │
│                                                              │
│  StudioView                  ResultsView                     │
│  ├─ Audio Recording    ├─ Performance Metrics               │
│  ├─ Live Pitch Tracking      ├─ ML Analysis Display         │
│  └─ Submit Recording    └─ Coaching Notes                  │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ HTTP POST /api/audio/analyze-with-ml
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│          (app/api/routers/audio_processing.py)              │
│                                                              │
│  Endpoint: /api/audio/analyze-with-ml                       │
│  └─ Receives: WebM audio blob, song_title, artist          │
│     Returns: MLAnalysisResult (JSON)                        │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Calls MLInferenceService
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  ML Inference Service                        │
│      (backend/app/services/ml_inference.py)                │
│                                                              │
│  analyze_audio()                                            │
│  └─ Uses: ml_new.inference.coach_inference                 │
│     Returns: Formatted coaching results                     │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Uses trained models
               │
┌──────────────▼──────────────────────────────────────────────┐
│                   ML Pipeline (ml_new)                       │
│                                                              │
│  coach_inference.analyse_recording()                        │
│  ├─ Feature Extraction (HCQT, VAD)                         │
│  ├─ Model Inference:                                        │
│  │  ├─ Unified Vocal Model (pitch, voicing, breath, onset) │
│  │  ├─ Acoustic Technique Classifier                       │
│  │  └─ Voice Quality Analysis (HNR, jitter, shimmer)      │
│  └─ Returns: CoachingResult                                │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Backend Dependencies

Ensure the backend has all ml_new dependencies installed:

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-ml.txt  # If separate ML requirements exist
```

### 2. Frontend Dependencies

```bash
cd new_frontend
npm install
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and update with your configuration:

```bash
# Backend
DATABASE_URL=postgresql+psycopg://vocalstars:vocalstars@db:5432/vocalstars
API_HOST=http://localhost:8000

# Frontend
REACT_APP_API_URL=http://localhost:8000

# ML Configuration
ML_CHECKPOINT_PATH=ml_new/checkpoints/unified/best.pt
ML_DEVICE=cpu  # or 'cuda' if using GPU

# Optional: Gemini API (for legacy coaching)
GEMINI_API_KEY=your_key_here
```

### 4. Available ML Checkpoints

The following pre-trained models are available in `ml_new/checkpoints/`:

- **`unified/`** - Unified vocal model (recommended)
  - Outputs: pitch, voicing, breath detection, onset detection
  - Model: Combined backbone with multiple task heads

- **`pitch_hires/`** - High-resolution pitch detection
  - Specialized pitch tracking model

- **`pitch_allpyin/`** - Librosa Pyin fallback
  - Heuristic-based pitch detection

- **`breath/`** - Breath detection model
  - Specialized breath support analysis

- **`onset/`** - Note onset detection
  - Detects vocal note attacks

- **`vad/`** - Voice Activity Detection
  - Separates voiced from unvoiced regions

### 5. Running the Application

#### Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd new_frontend
npm run dev
```

Frontend will be available at: `http://localhost:3000`
Backend API at: `http://localhost:8000`

#### Production Build

**Frontend:**
```bash
cd new_frontend
npm run build
npm run start
```

**Backend:**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. Docker Setup

Use docker-compose for integrated deployment:

```bash
docker-compose up
```

This will start:
- PostgreSQL database
- FastAPI backend (port 8000)
- React frontend (port 3000)

## API Reference

### Audio Analysis Endpoint

**POST** `/api/audio/analyze-with-ml`

**Request:**
```
Content-Type: multipart/form-data

Parameters:
- file (audio/webm): Audio recording blob
- song_title (string): Title of the song
- artist (string): Artist name
- checkpoint_path (string, optional): Path to custom checkpoint
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "score": 85,
    "summary": "Excellent pitch control with strong breath support.",
    "issues": ["Minor pitch drift in high range", "Could improve dynamics"],
    "exercises": ["Practice sustains on high notes"],
    "songTitle": "Example Song",
    "artist": "Example Artist",
    "pitchAccuracy": 92.5,
    "pitchDrift": -0.5,
    "phraseLengths": [2.3, 2.1, 1.9],
    "breathCount": 4,
    "onsetCount": 12,
    "onsetClarity": 0.78,
    "technique": "bel canto",
    "techniqueConfidence": 0.89,
    "allTechniqueScores": {
      "bel_canto": 0.89,
      "contemporary": 0.45,
      "classical": 0.67
    },
    "notes": [...],
    "voiceQuality": {
      "hnrDb": 22.5,
      "jitterPercent": 0.85,
      "shimmerPercent": 3.2,
      "breathiness": "clear",
      "isUnstable": false
    },
    "vibrato": {...},
    "frameData": {...}
  }
}
```

## Frontend Integration Details

### Audio Recording Flow

1. **StudioView.tsx**
   - User starts recording with microphone access
   - `startAudioRecording()` initializes WebM recording
   - Live pitch is tracked from AudioContext
   - User can view real-time feedback

2. **Session Completion**
   - `stopAudioRecording()` returns audio blob
   - `analyzeAudioWithML()` sends to backend
   - MLAnalysisResult is converted to PerformanceResult

3. **ResultsView.tsx**
   - Displays traditional metrics (intonation, rhythm, etc.)
   - Shows ML-specific analysis when available:
     - Pitch accuracy and drift
     - Breath/onset detection
     - Technique classification
     - Voice quality metrics (HNR, jitter, shimmer)
     - Detected notes with vibrato info
     - Recommended exercises based on issues

### Type System

Key types in `src/types.ts`:

```typescript
interface MLAnalysisResult {
  score: number;
  summary: string;
  issues: string[];
  exercises: string[];
  pitchAccuracy: number;
  pitchDrift: number;
  // ... (see types.ts for full schema)
}

interface PerformanceResult {
  songId: string;
  songTitle: string;
  // ... traditional fields
  mlAnalysis?: MLAnalysisResult;  // Optional ML data
}
```

## Troubleshooting

### Issue: "Audio recording failed"
- Check browser microphone permissions
- Ensure HTTPS in production (required for mediaDevices API)
- Check console for specific error messages

### Issue: "Backend analysis endpoint 404"
- Verify backend is running on `http://localhost:8000`
- Check CORS configuration in `backend/app/main.py`
- Ensure `audio_processing.py` router is registered

### Issue: "ML model not found"
- Verify checkpoint exists at path specified in `.env`
- Check `ML_CHECKPOINT_PATH` is correct
- Models will fall back to librosa.pyin if checkpoint unavailable

### Issue: "Out of memory during inference"
- Set `ML_DEVICE=cpu` in `.env`
- Reduce audio length
- Check system resources

## Performance Notes

- **Audio Processing**: ~2-5 seconds for typical 3-minute recording
- **File Size**: WebM recordings typically 500KB-2MB
- **Model Size**: Unified model ~100MB
- **GPU Acceleration**: Available with `ML_DEVICE=cuda` (requires CUDA-capable GPU)

## Development Workflow

1. **Adding new features to frontend**:
   - Update components in `new_frontend/src/components/`
   - Update types in `new_frontend/src/types.ts`
   - Test with local backend

2. **Modifying ML analysis**:
   - Changes in `ml_new/inference/coach_inference.py` automatically reflected
   - Update `backend/app/services/ml_inference.py` formatting if needed
   - Test with various audio samples

3. **Extending analysis metrics**:
   - Add new fields to `CoachingResult` in ml_new
   - Update `_format_coaching_result()` in `MLInferenceService`
   - Add new display components in `ResultsView.tsx`

## Next Steps

- [ ] Set up authentication system
- [ ] Implement user session history storage
- [ ] Add more sophisticated coaching recommendations
- [ ] Integrate progress tracking
- [ ] Deploy to production
- [ ] Add real-time collaboration features
- [ ] Expand model support for different vocal styles

## Support

For issues or questions:
1. Check error messages in browser console and backend logs
2. Review this documentation
3. Check ml_new documentation for model-specific questions
4. Review backend FastAPI logs for API errors

## References

- [ml_new Documentation](../ml_new/inference/coach_inference.py)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
