"use client"

import { useEffect, useRef, useState } from 'react'
import type { CoachingResult } from '../lib/types'
import { analyseAudio } from '../lib/api'
import AudioRecorder from '../components/AudioRecorder'
import { AnalysisCardResults, AnalysisCardPlaceholder } from '../components/AnalysisCard'
import CoachingPanel from '../components/CoachingPanel'

type AppPhase = 'idle' | 'recording' | 'processing' | 'results' | 'error'

interface AppState {
  phase: AppPhase
  data?: CoachingResult
  blob?: Blob
  error?: string
  elapsedS?: number
}

export default function Home() {
  const [state, setState] = useState<AppState>({ phase: 'idle' })
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  function clearTimer() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
  }

  async function handleAnalyse(blob: Blob, filename: string) {
    setState({ phase: 'processing', blob, elapsedS: 0 })
    timerRef.current = setInterval(() => {
      setState(s => ({ ...s, elapsedS: (s.elapsedS ?? 0) + 1 }))
    }, 1000)

    try {
      const data = await analyseAudio(blob, filename)
      clearTimer()
      setState({ phase: 'results', data, blob })
    } catch (err: unknown) {
      clearTimer()
      const message = err instanceof Error ? err.message : 'Unknown error'
      setState({ phase: 'error', error: message })
    }
  }

  useEffect(() => () => clearTimer(), [])

  const { phase, data, blob, error, elapsedS } = state

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      {/* Header */}
      <section className="mb-10 rounded-3xl bg-white p-8 shadow-lg">
        <div className="flex items-start justify-between">
          <div className="max-w-3xl">
            <p className="text-sm uppercase tracking-[0.24em] text-brand-700">VocalStars</p>
            <h1 className="mt-4 text-4xl font-semibold tracking-tight text-slate-900">
              Beginner-friendly vocal coaching
            </h1>
            <p className="mt-4 text-lg leading-8 text-slate-600">
              Record a short vocal exercise or upload a clip. VocalStars analyses your pitch,
              breath, and timing and gives you specific exercises to improve.
            </p>
          </div>
          {phase === 'results' && (
            <button
              type="button"
              onClick={() => setState({ phase: 'idle' })}
              className="ml-6 shrink-0 rounded-2xl border border-slate-200 bg-white px-5 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
            >
              Start over
            </button>
          )}
        </div>
      </section>

      {/* Error banner */}
      {phase === 'error' && (
        <div className="mb-6 flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-5">
          <span className="mt-0.5 text-red-500">✕</span>
          <div>
            <p className="font-medium text-red-800">Analysis failed</p>
            <p className="mt-0.5 text-sm text-red-700">{error}</p>
            <button
              type="button"
              onClick={() => setState({ phase: 'idle' })}
              className="mt-2 text-sm font-medium text-red-700 underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Processing elapsed counter */}
      {phase === 'processing' && (
        <p className="mb-4 text-sm text-slate-500 text-center">
          {elapsedS ?? 0} s elapsed — large files can take up to 30 s
        </p>
      )}

      <div className="grid gap-8 lg:grid-cols-[1.1fr_0.85fr]">
        {/* Left column */}
        <div className="space-y-8">
          <AudioRecorder
            phase={phase}
            onAnalyse={handleAnalyse}
            onRecordingStateChange={(p) =>
              setState(s => ({ ...s, phase: p }))
            }
          />

          {phase === 'results' && data && blob
            ? <AnalysisCardResults result={data} audioBlob={blob} />
            : phase !== 'results' && <AnalysisCardPlaceholder />
          }
        </div>

        {/* Right column */}
        <div>
          {phase === 'results' && data ? (
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <CoachingPanel result={data} />
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
              <p className="text-sm text-slate-400">
                {phase === 'processing'
                  ? 'Results will appear here once analysis is complete.'
                  : 'Your coaching report will appear here after you record or upload a clip.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
