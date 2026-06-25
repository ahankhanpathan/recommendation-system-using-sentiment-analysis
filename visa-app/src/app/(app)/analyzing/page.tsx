"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

const STEPS = [
  "Reading academic profile",
  "Parsing degree & CGPA data",
  "Comparing 500+ universities",
  "Checking admission requirements",
  "Analyzing visa policies",
  "Calculating tuition ROI",
  "Matching scholarship eligibility",
  "Predicting salary trajectories",
  "Comparing living costs across 6 countries",
  "Ranking destination countries",
  "Assessing PR pathways",
  "Building your personalized roadmap",
  "Formatting your report",
]

const STATS = [
  { label: "Universities Compared", target: 500, suffix: "+" },
  { label: "Data Points", target: 2400000, suffix: "" },
  { label: "Countries", target: 180, suffix: "" },
  { label: "Scholarships", target: 230, suffix: "" },
]

export default function AnalyzingPage() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(0)
  const [doneSteps, setDoneSteps] = useState<number[]>([])
  const [progress, setProgress] = useState(0)
  const [statValues, setStatValues] = useState(STATS.map(() => 0))
  const [done, setDone] = useState(false)

  useEffect(() => {
    const total = STEPS.length
    const stepDuration = 2500
    const timers: ReturnType<typeof setTimeout>[] = []

    STEPS.forEach((_, i) => {
      timers.push(setTimeout(() => {
        setCurrentStep(i)
        setProgress(Math.round(((i + 1) / total) * 88))
        setTimeout(() => {
          setDoneSteps(prev => [...prev, i])
        }, 1800)
      }, i * stepDuration))
    })

    timers.push(setTimeout(() => {
      setProgress(100)
      setDone(true)
    }, total * stepDuration + 2000))

    // Count-up for stats
    const countDuration = total * stepDuration - 1000
    STATS.forEach((stat, si) => {
      const steps = 60
      const interval = countDuration / steps
      for (let s = 1; s <= steps; s++) {
        timers.push(setTimeout(() => {
          setStatValues(prev => {
            const next = [...prev]
            next[si] = Math.round(stat.target * (s / steps))
            return next
          })
        }, s * interval))
      }
    })

    return () => timers.forEach(clearTimeout)
  }, [])

  const formatStat = (val: number, stat: typeof STATS[0]) => {
    if (val >= 1000000) return (val / 1000000).toFixed(1) + "M"
    if (val >= 1000) return (val / 1000).toFixed(0) + "K"
    return val.toString() + stat.suffix
  }

  return (
    <div style={{ minHeight: "100vh", background: "#09090B", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
      {/* Background orbs */}
      <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        <div style={{ position: "absolute", width: 600, height: 600, borderRadius: "50%", background: "#6366F1", filter: "blur(120px)", opacity: 0.06, top: "50%", left: "50%", transform: "translate(-50%,-50%)" }} />
        <div style={{ position: "absolute", width: 300, height: 300, borderRadius: "50%", background: "#8B5CF6", filter: "blur(80px)", opacity: 0.08, top: "20%", right: "20%" }} />
      </div>

      <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", alignItems: "center", width: "100%", maxWidth: 640, padding: "0 24px" }}>
        {/* Logo */}
        <div style={{ fontWeight: 800, fontSize: 20, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 48 }}>Pathora</div>

        {!done ? (
          <>
            {/* Animated ring */}
            <div style={{ position: "relative", marginBottom: 40 }}>
              <svg width={120} height={120} style={{ transform: "rotate(-90deg)" }}>
                <circle cx={60} cy={60} r={52} fill="none" stroke="#27272A" strokeWidth={6} />
                <circle cx={60} cy={60} r={52} fill="none" stroke="url(#grad)" strokeWidth={6}
                  strokeDasharray={`${2 * Math.PI * 52}`}
                  strokeDashoffset={`${2 * Math.PI * 52 * (1 - progress / 100)}`}
                  strokeLinecap="round" style={{ transition: "stroke-dashoffset 0.5s ease" }} />
                <defs>
                  <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#6366F1" />
                    <stop offset="100%" stopColor="#8B5CF6" />
                  </linearGradient>
                </defs>
              </svg>
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#6366F1" }}>{progress}%</div>
              </div>
            </div>

            <h1 style={{ fontSize: 26, fontWeight: 700, color: "#FAFAFA", textAlign: "center", marginBottom: 8 }}>Pathora AI is analyzing your profile</h1>
            <p style={{ fontSize: 14, color: "#71717A", textAlign: "center", marginBottom: 40 }}>Processing 2.4M data points across 180 countries</p>

            {/* Steps list */}
            <div style={{ width: "100%", background: "#111113", border: "1px solid #27272A", borderRadius: 20, padding: "24px 28px", display: "flex", flexDirection: "column", gap: 14 }}>
              {STEPS.map((step, i) => {
                const isDone = doneSteps.includes(i)
                const isActive = currentStep === i && !isDone
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 14, opacity: i > currentStep + 1 ? 0.3 : 1, transition: "opacity 0.3s" }}>
                    <div style={{ width: 22, height: 22, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.3s",
                      background: isDone ? "#10B981" : isActive ? "#6366F1" : "#27272A",
                    }}>
                      {isDone
                        ? <span style={{ fontSize: 12, color: "white" }}>✓</span>
                        : isActive
                        ? <span style={{ width: 8, height: 8, borderRadius: "50%", background: "white", display: "block", animation: "pulse-ring 1s ease infinite" }} />
                        : <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#52525B", display: "block" }} />
                      }
                    </div>
                    <span style={{ fontSize: 14, fontWeight: isDone ? 500 : isActive ? 600 : 400, color: isDone ? "#10B981" : isActive ? "#FAFAFA" : "#52525B", transition: "all 0.3s" }}>
                      {step}
                    </span>
                    {isDone && <span style={{ marginLeft: "auto", fontSize: 11, color: "#10B98160" }}>done</span>}
                    {isActive && <span style={{ marginLeft: "auto", fontSize: 11, color: "#6366F1" }}>analyzing...</span>}
                  </div>
                )
              })}
            </div>
          </>
        ) : (
          <div style={{ textAlign: "center", animation: "slideUp 0.6s ease forwards" }}>
            <div style={{ fontSize: 64, marginBottom: 24 }}>✨</div>
            <h1 style={{ fontSize: 36, fontWeight: 800, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 12 }}>Your Report is Ready!</h1>
            <p style={{ fontSize: 16, color: "#A1A1AA", marginBottom: 40 }}>We analyzed 2.4M data points to build your personalized relocation roadmap</p>
            <button onClick={() => router.push("/report")}
              style={{ padding: "16px 40px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", borderRadius: 14, color: "white", fontSize: 17, fontWeight: 700, cursor: "pointer", boxShadow: "0 8px 32px rgba(99,102,241,0.4)", transition: "transform 0.2s" }}
              onMouseEnter={e => (e.currentTarget.style.transform = "translateY(-2px)")}
              onMouseLeave={e => (e.currentTarget.style.transform = "none")}>
              View My Report →
            </button>
          </div>
        )}

        {/* Stats bar */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, width: "100%", marginTop: 40 }}>
          {STATS.map((stat, i) => (
            <div key={i} style={{ textAlign: "center", padding: "16px", background: "#111113", border: "1px solid #27272A", borderRadius: 14 }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: "#6366F1", fontVariantNumeric: "tabular-nums" }}>{formatStat(statValues[i], stat)}</div>
              <div style={{ fontSize: 11, color: "#52525B", marginTop: 4 }}>{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Progress bar */}
        {!done && (
          <div style={{ width: "100%", marginTop: 24 }}>
            <div style={{ height: 3, background: "#27272A", borderRadius: 99, overflow: "hidden" }}>
              <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#8B5CF6)", width: `${progress}%`, transition: "width 0.5s ease", borderRadius: 99 }} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
