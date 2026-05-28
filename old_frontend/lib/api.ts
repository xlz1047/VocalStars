import type { CoachingResult } from './types'

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'

export async function analyseAudio(
  blob: Blob,
  filename: string,
): Promise<CoachingResult> {
  const form = new FormData()
  form.append('file', blob, filename)

  const res = await fetch(`${API_BASE_URL}/api/coaching/analyse`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {}
    throw new Error(detail)
  }

  return res.json() as Promise<CoachingResult>
}
