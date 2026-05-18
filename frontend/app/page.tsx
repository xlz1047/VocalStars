import AudioRecorder from '../components/AudioRecorder'
import AnalysisCard from '../components/AnalysisCard'
import ProgressCard from '../components/ProgressCard'

export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <section className="mb-10 rounded-3xl bg-white p-8 shadow-lg">
        <div className="max-w-3xl">
          <p className="text-sm uppercase tracking-[0.24em] text-brand-700">VocalStars</p>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight text-slate-900">
            Beginner-friendly vocal analysis and coaching support
          </h1>
          <p className="mt-4 text-lg leading-8 text-slate-600">
            Upload a singing session or record a short vocal exercise. VocalStars focuses on stability,
            breath support, rhythm, and smooth note transitions — not judgment.
          </p>
        </div>
      </section>

      <div className="grid gap-8 lg:grid-cols-[0.9fr_0.7fr]">
        <div className="space-y-8">
          <AudioRecorder />
          <AnalysisCard />
        </div>
        <ProgressCard />
      </div>
    </main>
  )
}
