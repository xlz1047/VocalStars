export default function AnalysisCard() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
      <h2 className="text-2xl font-semibold text-slate-900">Analysis results</h2>
      <p className="mt-2 text-slate-600">Your session will be reviewed for core singing patterns and technical coaching notes.</p>

      <div className="mt-8 space-y-4">
        <div className="rounded-3xl bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Pitch Stability</h3>
          <p className="mt-2 text-slate-600">Learn how steady your pitch stayed across sustained notes.</p>
        </div>
        <div className="rounded-3xl bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Rhythm & timing</h3>
          <p className="mt-2 text-slate-600">Understand whether rhythm and timing matched your intended phrase.</p>
        </div>
        <div className="rounded-3xl bg-slate-50 p-5">
          <h3 className="text-base font-semibold text-slate-900">Breath support</h3>
          <p className="mt-2 text-slate-600">See if your breath approach helped keep your sound stable.</p>
        </div>
      </div>
    </section>
  )
}
