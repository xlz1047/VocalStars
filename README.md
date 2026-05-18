# VocalStars

VocalStars is an AI-powered beginner vocal coaching application designed for supportive, technical improvement rather than scoring or judging. It helps beginner singers identify measurable voice patterns, understand actionable coaching feedback, and track progress over time.

## Architecture

- `frontend/` — Next.js + TypeScript + Tailwind user interface
- `backend/` — FastAPI REST API, modular services, database integration
- `ml/` — Python audio and ML pipeline with feature extraction placeholders
- `shared/` — Common schemas and cross-platform types
- `docs/` — Architecture, API contract, setup guidance

## Key features

- Audio recording and file upload flow
- Analysis routing for pitch, rhythm, breath, and vocal stability
- Coaching recommendations engine skeleton
- Progress tracking starter schema
- Docker-based local development

## Getting Started

1. Install dependencies for frontend and backend.
2. Copy `.env.example` to `.env` and configure your PostgreSQL connection.
3. Start services with Docker Compose.

## Local development

- Frontend: `cd frontend && npm install && npm run dev`
- Backend: `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`

## Roadmap

- Add end-to-end audio feature extraction and model training
- Implement user authentication and session history
- Build interactive progress charts and coaching timelines
- Extend coaching engine with sample exercises
- Add tests and CI linting workflows
