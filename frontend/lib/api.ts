export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export async function uploadAudioFile(file: File) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/audio/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    return { status: 'error' }
  }

  return response.json()
}

export async function fetchAnalysis(sessionId: number) {
  const res = await fetch(`${API_BASE_URL}/api/analysis/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  return res.ok ? res.json() : null
}

export async function fetchRecommendations(sessionId: number) {
  const res = await fetch(`${API_BASE_URL}/api/coaching/recommendations/${sessionId}`)
  return res.ok ? res.json() : null
}
