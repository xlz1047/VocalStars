# VocalStars Local Setup

## Prerequisites

- Node.js 20+
- Python 3.12+
- Docker and Docker Compose
- PostgreSQL (optional when using Docker Compose)

## Frontend

1. `cd frontend`
2. `npm install`
3. `npm run dev`

## Backend

1. `cd backend`
2. `poetry install`
3. `cp ../.env.example ../.env`
4. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

## Docker

1. `docker compose up --build`
2. Frontend available at `http://localhost:3000`
3. Backend available at `http://localhost:8000`

## Notes

- The backend is designed to support a modular ML package under `ml/`.
- Audio upload files are stored temporarily in `backend/audio_uploads/`.
