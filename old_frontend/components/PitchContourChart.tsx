"use client"

import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

interface PitchContourChartProps {
  pitchHz: number[]
  voiced: boolean[]
  hop_s: number
}

const A_NOTE_HZ: Record<string, number> = {
  'A2': 110, 'A3': 220, 'A4': 440, 'A5': 880,
}

function nearestSemitoneHz(hz: number): number {
  const semitones = Math.round(12 * Math.log2(hz / 440)) // semitones from A4
  return 440 * Math.pow(2, semitones / 12)
}

function centsError(hz: number): number {
  const ref = nearestSemitoneHz(hz)
  return 1200 * Math.log2(hz / ref)
}

function pointColour(error: number): string {
  const abs = Math.abs(error)
  if (abs < 50) return '#22c55e'
  if (abs < 100) return '#f59e0b'
  return '#ef4444'
}

interface ChartPoint {
  t: number
  hz: number
  colour: string
}

export default function PitchContourChart({ pitchHz, voiced, hop_s }: PitchContourChartProps) {
  const segments = useMemo(() => {
    // Downsample to ≤600 points
    const maxPoints = 600
    const factor = Math.max(1, Math.ceil(pitchHz.length / maxPoints))

    const points: ChartPoint[] = []
    for (let i = 0; i < pitchHz.length; i += factor) {
      // Average voiced Hz in this bucket
      let sum = 0, count = 0
      for (let j = i; j < Math.min(i + factor, pitchHz.length); j++) {
        if (voiced[j] && pitchHz[j] > 0) { sum += pitchHz[j]; count++ }
      }
      if (count === 0) continue
      const hz = sum / count
      const err = centsError(hz)
      points.push({ t: parseFloat((i * hop_s).toFixed(2)), hz, colour: pointColour(err) })
    }

    // Split into contiguous colour groups for separate <Line> segments
    if (points.length === 0) return []

    const segs: ChartPoint[][] = []
    let cur: ChartPoint[] = [points[0]]
    for (let i = 1; i < points.length; i++) {
      // Gap check: if there's a >0.5 s jump in time it's a silence gap
      if (points[i].t - points[i - 1].t > 0.5 || points[i].colour !== points[i - 1].colour) {
        segs.push(cur)
        cur = []
      }
      cur.push(points[i])
    }
    segs.push(cur)
    return segs
  }, [pitchHz, voiced, hop_s])

  if (segments.length === 0) {
    return <p className="text-sm text-slate-400 italic py-6 text-center">No voiced pitch detected.</p>
  }

  // Merge all points for a single responsive chart
  const allPoints = segments.flat()
  const colours = ['#22c55e', '#f59e0b', '#ef4444']

  return (
    <div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart margin={{ top: 8, right: 8, bottom: 8, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="t"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v: number) => `${v.toFixed(1)}s`}
            tick={{ fontSize: 10 }}
            allowDuplicatedCategory={false}
          />
          <YAxis
            scale="log"
            domain={[60, 1400]}
            tickFormatter={(v: number) => {
              const label = Object.entries(A_NOTE_HZ).find(([, hz]) => Math.abs(hz - v) < 20)
              return label ? label[0] : ''
            }}
            tick={{ fontSize: 10 }}
          />
          {Object.entries(A_NOTE_HZ).map(([name, hz]) => (
            <ReferenceLine key={name} y={hz} stroke="#e2e8f0" strokeDasharray="4 2" label={{ value: name, fontSize: 9, fill: '#94a3b8' }} />
          ))}
          <Tooltip
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [`${Number(v).toFixed(1)} Hz`, 'Pitch']}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            labelFormatter={(l: any) => `${Number(l).toFixed(2)} s`}
          />
          {segments.map((seg, idx) => (
            <Line
              key={idx}
              data={seg}
              dataKey="hz"
              dot={false}
              stroke={seg[0]?.colour ?? '#22c55e'}
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-full bg-green-500 inline-block" /> In tune (&lt;50 ¢)</span>
        <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-full bg-amber-400 inline-block" /> Slightly off (50–100 ¢)</span>
        <span className="flex items-center gap-1"><span className="h-2 w-4 rounded-full bg-red-500 inline-block" /> Off (&gt;100 ¢)</span>
      </div>
    </div>
  )
}
