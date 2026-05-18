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
