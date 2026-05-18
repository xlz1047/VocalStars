export default function ProgressCard() {
  return (
    <section className="rounded-3xl border border-slate-200 bg-slate-50 p-8 shadow-sm">
      <h2 className="text-2xl font-semibold text-slate-900">Practice recommendations</h2>
      <p className="mt-2 text-slate-600">Build your next practice session from beginner-focused vocal exercises.</p>

      <div className="mt-8 space-y-4">
        <div className="rounded-3xl bg-white p-5 shadow-sm">
          <h3 className="text-base font-semibold text-slate-900">Warm-up support</h3>
          <p className="mt-2 text-slate-600">Start with breath control and gentle pitch shapes before singing.</p>
        </div>
        <div className="rounded-3xl bg-white p-5 shadow-sm">
          <h3 className="text-base font-semibold text-slate-900">Stability drill</h3>
          <p className="mt-2 text-slate-600">Focus on sustaining a connected tone through small melodic steps.</p>
        </div>
      </div>
    </section>
  )
}
