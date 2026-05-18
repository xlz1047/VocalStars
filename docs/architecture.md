# VocalStars Architecture

## Overview

VocalStars is organized as a monorepo with clear separation between UI, backend services, and AI/audio modeling.

- `frontend/` contains the Next.js app and Tailwind-based UI.
- `backend/` contains the FastAPI REST API, database configuration, and service layers.
- `ml/` contains the audio analysis pipeline and feature extraction modules.
- `shared/` contains reusable types and cross-project documentation.

## Backend layers

- `app/api/routers` — route definitions for auth, audio upload, analysis, coaching, and progress.
- `app/services` — business logic and orchestration of analysis and coaching flows.
- `app/core` — database connection, configuration, and shared infrastructure.
- `models` / `schemas` — ORM and data contract definitions.

## ML pipeline modules

- `pitch_detection/` — pitch contour and stability placeholders
- `rhythm_analysis/` — tempo and timing analysis placeholders
- `breath_analysis/` — breath support and phrasing placeholders
- `feature_extraction/` — spectral and audio feature placeholders
- `coaching_engine/` — coaching recommendation outline generator
