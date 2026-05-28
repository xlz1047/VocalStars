import type { CoachingResult } from '../lib/types'
import WaveformCanvas from './WaveformCanvas'
import PitchContourChart from './PitchContourChart'
import NoteTable from './NoteTable'

interface AnalysisCardResultsProps {
  result: CoachingResult
  audioBlob: Blob
}

export function AnalysisCardResults({ result, audioBlob }: AnalysisCardResultsProps) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Waveform</h2>
        <p className="mt-1 text-xs text-slate-500">Teal regions = breath events detected by the model</p>
        <div className="mt-3">
          <WaveformCanvas
            audioBlob={audioBlob}
            breathFrames={result.breath_frames}
            hop_s={result.hop_s}
          />
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-900">Pitch contour</h2>
        <p className="mt-1 text-xs text-slate-500">
          Accuracy: {Math.round(result.pitch_accuracy * 100)}% in-tune&nbsp;·&nbsp;
          Drift: {result.pitch_drift_cents > 0 ? '+' : ''}{Math.round(result.pitch_drift_cents)} ¢
        </p>
        <div className="mt-3">
          <PitchContourChart
            pitchHz={result.pitch_hz}
            voiced={result.voiced}
            hop_s={result.hop_s}
          />
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-900">Note breakdown</h2>
        <p className="mt-1 text-xs text-slate-500">
          {result.notes.length} notes · {result.breath_count} breath{result.breath_count !== 1 ? 's' : ''} ·&nbsp;
          avg phrase {result.phrase_lengths_s.length > 0
            ? (result.phrase_lengths_s.reduce((a, b) => a + b, 0) / result.phrase_lengths_s.length).toFixed(1)
            : '—'} s
        </p>
        <div className="mt-3">
          <NoteTable notes={result.notes} />
        </div>
      </div>
    </section>
  )
}

export function AnalysisCardPlaceholder() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
      <h2 className="text-2xl font-semibold text-slate-900">Analysis results</h2>
      <p className="mt-2 text-slate-600">
        Record a clip or upload an audio file to see pitch contour, waveform, and per-note breakdown.
      </p>
      <div className="mt-8 space-y-4">
        {['Waveform + breath markers', 'Pitch contour', 'Per-note breakdown'].map(label => (
          <div key={label} className="rounded-3xl bg-slate-50 p-5">
            <h3 className="text-base font-semibold text-slate-400">{label}</h3>
            <div className="mt-2 h-4 w-3/4 rounded bg-slate-200" />
          </div>
        ))}
      </div>
    </section>
  )
}
