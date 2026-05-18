"use client"

import { useState, type ChangeEvent } from 'react'
import { uploadAudioFile } from '../lib/api'

export default function AudioRecorder() {
  const [status, setStatus] = useState('Ready to upload your first clip.')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null
    setSelectedFile(file)
    setStatus(file ? `Selected ${file.name}` : 'Ready to upload your first clip.')
  }

  async function handleUpload() {
    if (!selectedFile) return
    setStatus('Uploading session...')
    const response = await uploadAudioFile(selectedFile)
    setStatus(response.status === 'accepted' ? 'Upload successful — analysis started.' : 'Upload failed.')
  }

  return (
    <section className="rounded-3xl border border-slate-200 bg-slate-50 p-8 shadow-sm">
      <h2 className="text-2xl font-semibold text-slate-900">Record or upload a session</h2>
      <p className="mt-2 text-slate-600">Share a short clip and get beginner-friendly feedback on pitch, breath, and timing.</p>

      <div className="mt-6 space-y-4">
        <input type="file" accept="audio/*" onChange={handleFileChange} />
        <button
          type="button"
          onClick={handleUpload}
          disabled={!selectedFile}
          className="inline-flex items-center justify-center rounded-2xl bg-brand-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-brand-500 disabled:opacity-50"
        >
          Upload audio
        </button>
        <p className="text-sm text-slate-500">{status}</p>
      </div>
    </section>
  )
}
