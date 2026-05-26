"use client"

import { type ChangeEvent } from 'react'
import MicRecorder from './MicRecorder'

interface AudioRecorderProps {
  phase: 'idle' | 'recording' | 'processing' | 'results' | 'error'
  onAnalyse: (blob: Blob, filename: string) => void
  onRecordingStateChange: (p: 'idle' | 'recording') => void
}

export default function AudioRecorder({ phase, onAnalyse, onRecordingStateChange }: AudioRecorderProps) {
  function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    onAnalyse(file, file.name)
    // Reset input so the same file can be re-selected
    e.target.value = ''
  }

  if (phase === 'recording') {
    return (
      <section className="rounded-3xl border border-slate-200 bg-slate-50 p-8 shadow-sm">
        <h2 className="text-2xl font-semibold text-slate-900">Recording…</h2>
        <p className="mt-2 text-slate-600">Sing freely — click Stop when you&rsquo;re done.</p>
        <div className="mt-6">
          <MicRecorder onResult={(b) => onAnalyse(b, 'recording.webm')} onStateChange={onRecordingStateChange} />
        </div>
      </section>
    )
  }

  if (phase === 'processing') {
    return (
      <section className="rounded-3xl border border-slate-200 bg-slate-50 p-8 shadow-sm">
        <h2 className="text-2xl font-semibold text-slate-900">Analysing…</h2>
        <p className="mt-2 text-slate-600">The model is processing your recording. This usually takes 10–30 s.</p>
        <div className="mt-6 flex items-center gap-3 text-brand-700">
          <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
          <span className="text-sm font-medium">Processing your recording…</span>
        </div>
      </section>
    )
  }

  return (
    <section className="rounded-3xl border border-slate-200 bg-slate-50 p-8 shadow-sm">
      <h2 className="text-2xl font-semibold text-slate-900">
        {phase === 'results' ? 'Record another session' : 'Record or upload a session'}
      </h2>
      <p className="mt-2 text-slate-600">
        Sing a short clip and get beginner-friendly feedback on pitch, breath, and timing.
      </p>

      <div className="mt-6 space-y-4">
        <MicRecorder
          onResult={(b) => onAnalyse(b, 'recording.webm')}
          onStateChange={onRecordingStateChange}
        />

        <div className="flex items-center gap-3">
          <hr className="flex-1 border-slate-200" />
          <span className="text-xs text-slate-400">or</span>
          <hr className="flex-1 border-slate-200" />
        </div>

        <label className="flex cursor-pointer items-center gap-3 rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-slate-50">
          <svg className="h-4 w-4 text-slate-500" viewBox="0 0 20 20" fill="currentColor">
            <path d="M4 3a1 1 0 000 2h12a1 1 0 100-2H4zm-1 6a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h6a1 1 0 110 2H4a1 1 0 01-1-1z" />
          </svg>
          Upload an audio file
          <input
            type="file"
            accept="audio/*"
            className="sr-only"
            onChange={handleFile}
            disabled={false}
          />
        </label>
      </div>
    </section>
  )
}
