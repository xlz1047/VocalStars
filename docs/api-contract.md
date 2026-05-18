# VocalStars API Contract

## Auth

- `GET /api/auth/status`
  - Response: `{ status: string, message: string }`

## Audio Processing

- `POST /api/audio/upload`
  - Request: `multipart/form-data` with audio file field `file`
  - Response: `{ status: string, analysis_session: object }`

## Analysis

- `POST /api/analysis/run`
  - Request: `{ session_id: string }`
  - Response: `{ session_id: string, report: object }`

## Coaching

- `GET /api/coaching/recommendations/{session_id}`
  - Response: `{ session_id: string, recommendations: array }`

## Progress

- `GET /api/progress/user/{user_id}`
  - Response: `{ user_id: int, progress: array }`
