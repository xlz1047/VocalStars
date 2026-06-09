"use client"

import { useEffect, useRef } from 'react'

interface WaveformCanvasProps {
  audioBlob: Blob
  breathFrames: boolean[]
  hop_s: number
}

function findBreathRuns(frames: boolean[]): Array<[number, number]> {
  const runs: Array<[number, number]> = []
  let start: number | null = null
  for (let i = 0; i <= frames.length; i++) {
    if (frames[i] && start === null) start = i
    if (!frames[i] && start !== null) {
      runs.push([start, i - 1])
      start = null
    }
  }
  return runs
}

export default function WaveformCanvas({ audioBlob, breathFrames, hop_s }: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false

    async function draw() {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return

      const dpr = window.devicePixelRatio || 1
      const w = container.clientWidth
      const h = 96
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`

      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      let samples: Float32Array
      let totalSamples: number

      try {
        const arrayBuf = await audioBlob.arrayBuffer()
        if (cancelled) return
        const audioCtx = new AudioContext()
        const audioBuf = await audioCtx.decodeAudioData(arrayBuf)
        audioCtx.close()
        if (cancelled) return
        samples = audioBuf.getChannelData(0)
        totalSamples = samples.length
      } catch {
        ctx.fillStyle = '#94a3b8'
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('Waveform unavailable', w / 2, h / 2)
        return
      }

      // Draw breath overlays first (behind the waveform)
      const totalS = totalSamples / 44100  // approx; precise enough for overlay
      const breathRuns = findBreathRuns(breathFrames)
      ctx.fillStyle = 'rgba(20, 184, 166, 0.18)'  // teal-500 @ 18%
      for (const [s, e] of breathRuns) {
        const x0 = (s * hop_s / totalS) * w
        const x1 = ((e + 1) * hop_s / totalS) * w
        ctx.fillRect(x0, 0, Math.max(x1 - x0, 2), h)
      }

      // Draw waveform
      const samplesPerPixel = Math.max(1, Math.floor(totalSamples / w))
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let px = 0; px < w; px++) {
        const start = px * samplesPerPixel
        let min = 1, max = -1
        for (let j = 0; j < samplesPerPixel && start + j < totalSamples; j++) {
          const v = samples[start + j]
          if (v < min) min = v
          if (v > max) max = v
        }
        const y0 = (1 - (max + 1) / 2) * h
        const y1 = (1 - (min + 1) / 2) * h
        ctx.moveTo(px + 0.5, y0)
        ctx.lineTo(px + 0.5, y1)
      }
      ctx.stroke()

      // Centreline
      ctx.strokeStyle = '#cbd5e1'
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.moveTo(0, h / 2)
      ctx.lineTo(w, h / 2)
      ctx.stroke()
    }

    draw()
    return () => { cancelled = true }
  }, [audioBlob, breathFrames, hop_s])

  return (
    <div ref={containerRef} className="w-full">
      <canvas ref={canvasRef} className="w-full rounded-lg" />
      <p className="mt-1 text-xs text-slate-400">
        <span className="inline-block h-2 w-4 rounded-sm bg-teal-400 opacity-60 mr-1" />
        Teal shading = breath events detected
      </p>
    </div>
  )
}
