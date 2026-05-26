"use client"

import { useState } from 'react'
import type { NoteSegment } from '../lib/types'

interface NoteTableProps {
  notes: NoteSegment[]
}

function centsColour(error: number): string {
  const abs = Math.abs(error)
  if (abs < 25) return 'text-green-700 bg-green-50'
  if (abs < 50) return 'text-amber-700 bg-amber-50'
  return 'text-red-700 bg-red-50'
}

export default function NoteTable({ notes }: NoteTableProps) {
  const [showAll, setShowAll] = useState(false)

  if (notes.length === 0) {
    return <p className="text-sm text-slate-400 italic">No notes detected.</p>
  }

  const displayed = showAll ? notes : notes.slice(0, 20)

  return (
    <div>
      <div className="overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-2 text-left">Time</th>
              <th className="px-4 py-2 text-left">Note</th>
              <th className="px-4 py-2 text-left">Duration</th>
              <th className="px-4 py-2 text-left">Cents off</th>
              <th className="px-4 py-2 text-left">Stability</th>
              <th className="px-4 py-2 text-left">Vibrato</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {displayed.map((n, i) => (
              <tr key={i} className="bg-white hover:bg-slate-50">
                <td className="px-4 py-2 tabular-nums text-slate-600">
                  {n.start_s.toFixed(2)} s
                </td>
                <td className="px-4 py-2 font-semibold text-slate-800">{n.note_name}</td>
                <td className="px-4 py-2 tabular-nums text-slate-600">
                  {n.duration_s.toFixed(2)} s
                </td>
                <td className="px-4 py-2">
                  <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${centsColour(n.cents_error)}`}>
                    {n.cents_error > 0 ? '+' : ''}{Math.round(n.cents_error)} ¢
                  </span>
                </td>
                <td className="px-4 py-2 tabular-nums text-slate-600">
                  ±{Math.round(n.stability_cents)} ¢
                </td>
                <td className="px-4 py-2 text-slate-600">
                  {n.vibrato
                    ? `${n.vibrato.rate_hz.toFixed(1)} Hz`
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {notes.length > 20 && (
        <button
          type="button"
          onClick={() => setShowAll(v => !v)}
          className="mt-2 text-xs text-brand-700 hover:underline"
        >
          {showAll ? 'Show fewer' : `Show all ${notes.length} notes`}
        </button>
      )}
    </div>
  )
}
