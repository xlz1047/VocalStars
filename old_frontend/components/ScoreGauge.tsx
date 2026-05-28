"use client"

import { useEffect, useRef } from 'react'

interface ScoreGaugeProps {
  score: number
  technique: string
  confidence: number
}

export default function ScoreGauge({ score, technique, confidence }: ScoreGaugeProps) {
  const arcRef = useRef<SVGCircleElement>(null)

  const R = 50
  const circumference = 2 * Math.PI * R
  const colour =
    score >= 80 ? '#22c55e' :
    score >= 60 ? '#f59e0b' :
                  '#ef4444'

  useEffect(() => {
    const el = arcRef.current
    if (!el) return
    // Start fully hidden, then animate to final value
    el.style.strokeDashoffset = String(circumference)
    el.style.transition = 'none'
    const id = setTimeout(() => {
      el.style.transition = 'stroke-dashoffset 1.2s ease-out'
      el.style.strokeDashoffset = String(circumference * (1 - score / 100))
    }, 50)
    return () => clearTimeout(id)
  }, [score, circumference])

  return (
    <div className="flex flex-col items-center gap-3">
      <svg viewBox="0 0 120 120" className="w-40 h-40" aria-label={`Score: ${score} out of 100`}>
        {/* Track */}
        <circle
          cx="60" cy="60" r={R}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="10"
        />
        {/* Animated arc */}
        <circle
          ref={arcRef}
          cx="60" cy="60" r={R}
          fill="none"
          stroke={colour}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
          transform="rotate(-90 60 60)"
        />
        {/* Score text */}
        <text
          x="60" y="56"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="28"
          fontWeight="700"
          fill="#0f172a"
        >
          {score}
        </text>
        <text
          x="60" y="75"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="11"
          fill="#64748b"
        >
          / 100
        </text>
      </svg>

      <div className="text-center">
        <span
          className="inline-block rounded-full px-3 py-1 text-xs font-semibold text-white"
          style={{ backgroundColor: colour }}
        >
          {technique.replace(/_/g, ' ')}
        </span>
        <p className="mt-1 text-xs text-slate-500">
          {Math.round(confidence * 100)}% confident
        </p>
      </div>
    </div>
  )
}
