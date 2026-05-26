"use client"

import { useEffect, useRef, useState } from 'react'

interface MicRecorderProps {
  onResult: (blob: Blob, durationMs: number) => void
  onStateChange: (phase: 'idle' | 'recording') => void
}

function pickMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',
    '',
  ]
  if (typeof MediaRecorder === 'undefined') return ''
  return candidates.find(t => !t || MediaRecorder.isTypeSupported(t)) ?? ''
}

export default function MicRecorder({ onResult, onStateChange }: MicRecorderProps) {
  const [recording, setRecording] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [level, setLevel] = useState(0)          // 0–1 RMS level
  const [micError, setMicError] = useState<string | null>(null)

  const streamRef    = useRef<MediaStream | null>(null)
  const recorderRef  = useRef<MediaRecorder | null>(null)
  const chunksRef    = useRef<BlobPart[]>([])
  const startTimeRef = useRef<number>(0)
  const rafRef       = useRef<number>(0)
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null)
  const analyserRef  = useRef<AnalyserNode | null>(null)
  const audioCtxRef  = useRef<AudioContext | null>(null)

  // Unavailable on old iOS — hide the whole component
  if (typeof window !== 'undefined' && typeof MediaRecorder === 'undefined') {
    return null
  }

  function stopStream() {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (timerRef.current) clearInterval(timerRef.current)
    cancelAnimationFrame(rafRef.current)
    audioCtxRef.current?.close()
    audioCtxRef.current = null
    analyserRef.current = null
  }

  function drawLevel() {
    const analyser = analyserRef.current
    if (!analyser) return
    const buf = new Uint8Array(analyser.fftSize)
    analyser.getByteTimeDomainData(buf)
    let sum = 0
    for (const v of buf) {
      const s = (v - 128) / 128
      sum += s * s
    }
    setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 6))
    rafRef.current = requestAnimationFrame(drawLevel)
  }

  async function startRecording() {
    setMicError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Level meter
      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser
      rafRef.current = requestAnimationFrame(drawLevel)

      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' })
        const durationMs = Date.now() - startTimeRef.current
        onResult(blob, durationMs)
        stopStream()
        setRecording(false)
        setLevel(0)
        onStateChange('idle')
      }

      recorder.start(100)
      startTimeRef.current = Date.now()
      setElapsedMs(0)
      timerRef.current = setInterval(() => setElapsedMs(Date.now() - startTimeRef.current), 200)
      setRecording(true)
      onStateChange('recording')
    } catch (err: unknown) {
      const name = err instanceof Error ? err.name : ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setMicError('Microphone access denied — use the file upload below instead.')
      } else if (name === 'NotFoundError') {
        setMicError('No microphone found — use the file upload below instead.')
      } else {
        setMicError('Could not access microphone. Please use file upload.')
      }
    }
  }

  function stopRecording() {
    recorderRef.current?.stop()
    if (timerRef.current) clearInterval(timerRef.current)
  }

  useEffect(() => () => stopStream(), [])

  const elapsed = Math.floor(elapsedMs / 1000)
  const elapsedLabel = `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')}`

  if (!recording) {
    return (
      <div className="space-y-3">
        <button
          type="button"
          onClick={startRecording}
          className="flex items-center gap-3 rounded-2xl bg-brand-700 px-6 py-4 text-sm font-semibold text-white transition hover:bg-brand-500"
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-white">
            <span className="h-2.5 w-2.5 rounded-full bg-white" />
          </span>
          Record with microphone
        </button>
        {micError && (
          <p className="text-sm text-red-600">{micError}</p>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Level meter */}
      <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-brand-500 transition-all duration-75"
          style={{ width: `${Math.round(level * 100)}%` }}
        />
      </div>

      <div className="flex items-center gap-4">
        <span className="font-mono text-lg tabular-nums text-slate-700">{elapsedLabel}</span>
        <button
          type="button"
          onClick={stopRecording}
          className="rounded-2xl bg-red-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-red-500"
        >
          Stop recording
        </button>
      </div>

      <p className="flex items-center gap-2 text-sm text-slate-500">
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />
        Recording in progress…
      </p>
    </div>
  )
}
