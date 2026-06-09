"use client"

import { useState } from 'react'
import type { CoachingResult } from '../lib/types'
import ScoreGauge from './ScoreGauge'

interface CoachingPanelProps {
  result: CoachingResult
}

function breathinessBadge(b: string) {
  const map: Record<string, string> = {
    clear: 'bg-green-100 text-green-700',
    mild: 'bg-amber-100 text-amber-700',
    breathy: 'bg-red-100 text-red-700',
  }
  return map[b] ?? 'bg-slate-100 text-slate-700'
}

export default function CoachingPanel({ result }: CoachingPanelProps) {
  const [openIdx, setOpenIdx] = useState<number | null>(0)

  const vb = result.vibrato_stats
  const hasVibrato = (vb?.n_vibrato_notes ?? 0) > 0

  return (
    <div className="space-y-6">
      {/* Score gauge */}
      <div className="flex justify-center">
        <ScoreGauge
          score={result.score}
          technique={result.technique}
          confidence={result.technique_confidence}
        />
      </div>

      {/* Summary */}
      <p className="text-sm leading-6 text-slate-700">{result.summary}</p>

      {/* Issues accordion */}
      {result.issues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Issues &amp; Exercises
          </h3>
          {result.issues.map((issue, i) => (
            <div key={i} className="rounded-xl border border-slate-200 bg-white">
              <button
                type="button"
                onClick={() => setOpenIdx(openIdx === i ? null : i)}
                className="flex w-full items-start justify-between gap-2 px-4 py-3 text-left text-sm font-medium text-slate-800"
              >
                <span>{issue}</span>
                <span className="mt-0.5 shrink-0 text-slate-400">{openIdx === i ? '▲' : '▼'}</span>
              </button>
              {openIdx === i && result.exercises[i] && (
                <div className="border-t border-slate-100 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                  <span className="font-medium text-brand-700">Exercise: </span>
                  {result.exercises[i]}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Voice quality */}
      {result.voice_quality && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Voice Quality</h3>
          <div className="flex flex-wrap gap-2">
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${breathinessBadge(result.voice_quality.breathiness)}`}>
              {result.voice_quality.breathiness}
            </span>
            {result.voice_quality.is_unstable && (
              <span className="rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-700">unstable</span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs text-slate-600">
            <div>
              <div className="font-semibold text-slate-800">{result.voice_quality.hnr_db.toFixed(1)} dB</div>
              <div className="text-slate-400">HNR</div>
            </div>
            <div>
              <div className="font-semibold text-slate-800">{result.voice_quality.jitter_pct.toFixed(1)}%</div>
              <div className="text-slate-400">Jitter</div>
            </div>
            <div>
              <div className="font-semibold text-slate-800">{result.voice_quality.shimmer_pct.toFixed(1)}%</div>
              <div className="text-slate-400">Shimmer</div>
            </div>
          </div>
        </div>
      )}

      {/* Vibrato stats */}
      {hasVibrato && (
        <div className="rounded-xl border border-green-200 bg-green-50 p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-green-700">Vibrato Detected</h3>
          <p className="mt-1 text-sm text-green-800">
            {vb.n_vibrato_notes} of {vb.n_long_notes} sustained notes —&nbsp;
            avg {vb.mean_rate_hz?.toFixed(1)} Hz, {Math.round(vb.mean_depth_cents ?? 0)} ¢ depth
          </p>
        </div>
      )}

      {/* Pitch drift */}
      {Math.abs(result.pitch_drift_cents) >= 10 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
          <span className="font-medium">Overall drift: </span>
          {result.pitch_drift_cents > 0
            ? `+${Math.round(result.pitch_drift_cents)} ¢ (tends sharp)`
            : `${Math.round(result.pitch_drift_cents)} ¢ (tends flat)`}
        </div>
      )}
    </div>
  )
}
