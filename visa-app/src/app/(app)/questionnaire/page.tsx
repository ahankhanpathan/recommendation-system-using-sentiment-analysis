"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

const STEPS = [
  {
    id: 1, title: "Where are you right now?", subtitle: "Academic & career stage",
    type: "card-select",
    options: ["Final year undergrad", "Just graduated", "1–2 years work exp", "3+ years work exp"],
  },
  {
    id: 2, title: "Target field of study?", subtitle: "Select all that apply",
    type: "multi-chip",
    options: ["AI / Machine Learning", "Data Science", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Finance / FinTech", "Business / MBA", "Biotech / Life Sciences", "Architecture / Design", "Law / Policy", "Medicine", "Other"],
  },
  {
    id: 3, title: "Your academic profile", subtitle: "Be accurate for best matches",
    type: "sliders",
    fields: [
      { key: "cgpa", label: "CGPA", min: 5.0, max: 10.0, step: 0.1, default: 8.4, suffix: "/10" },
      { key: "ielts", label: "IELTS Score", min: 4.0, max: 9.0, step: 0.5, default: 7.5, suffix: "/9" },
    ],
    extras: [
      { key: "gre", label: "GRE Status", options: ["Taken (320+)", "Taken (<320)", "Planning", "Not taking"] },
      { key: "pubs", label: "Research Publications", options: ["0", "1", "2", "3+"] },
    ],
  },
  {
    id: 4, title: "Budget & Timeline", subtitle: "This shapes everything",
    type: "budget",
  },
  {
    id: 5, title: "What's your #1 priority?", subtitle: "Be honest — it changes your matches",
    type: "card-select-icon",
    options: [
      { label: "Best University Brand", emoji: "🏆" },
      { label: "Lowest Total Cost", emoji: "💰" },
      { label: "Fastest PR Pathway", emoji: "🛂" },
      { label: "Highest Salary", emoji: "💼" },
      { label: "Best Quality of Life", emoji: "🌍" },
    ],
  },
  {
    id: 6, title: "Any preferences?", subtitle: "Select all that matter to you",
    type: "checkbox",
    options: [
      "English-only programs", "Part-time work allowed", "Scholarship opportunities",
      "Warm climate", "Large Indian community", "Strong tech job market",
      "Low cost of living", "Easy visa process",
    ],
  },
  {
    id: 7, title: "Work experience", subtitle: "Helps universities and visa officers",
    type: "work-exp",
  },
  {
    id: 8, title: "Almost there!", subtitle: "Any last details for your AI report?",
    type: "final",
  },
]

export default function QuestionnairePage() {
  const router = useRouter()
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState<Record<string, any>>({})
  const [saved, setSaved] = useState(false)
  const [sliders, setSliders] = useState<Record<string, number>>({ cgpa: 8.4, ielts: 7.5, budget: 25000 })
  const [multiSelect, setMultiSelect] = useState<string[]>([])
  const [checkboxes, setCheckboxes] = useState<string[]>([])

  const current = STEPS[step]
  const progress = ((step + 1) / STEPS.length) * 100

  const triggerSave = () => { setSaved(true); setTimeout(() => setSaved(false), 2000) }

  const next = () => {
    triggerSave()
    if (step < STEPS.length - 1) setStep(s => s + 1)
    else router.push("/analyzing")
  }

  const toggleMulti = (opt: string) => {
    setMultiSelect(prev => prev.includes(opt) ? prev.filter(x => x !== opt) : [...prev, opt])
    triggerSave()
  }

  const toggleCheck = (opt: string) => {
    setCheckboxes(prev => prev.includes(opt) ? prev.filter(x => x !== opt) : [...prev, opt])
    triggerSave()
  }

  return (
    <div style={{ minHeight: "100vh", background: "#09090B", display: "flex", flexDirection: "column" }}>
      {/* Top bar */}
      <div style={{ padding: "20px 40px", borderBottom: "1px solid #27272A", display: "flex", alignItems: "center", gap: 20 }}>
        <div style={{ fontWeight: 700, fontSize: 16, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Pathora</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: "#71717A" }}>Step {step + 1} of {STEPS.length}</span>
            <span style={{ fontSize: 12, color: "#6366F1", fontWeight: 600 }}>{Math.round(progress)}% complete</span>
          </div>
          <div style={{ height: 4, background: "#27272A", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg,#6366F1,#8B5CF6)", borderRadius: 99, transition: "width 0.5s cubic-bezier(0.34,1.56,0.64,1)" }} />
          </div>
        </div>
        {saved && <div style={{ fontSize: 12, color: "#10B981", display: "flex", alignItems: "center", gap: 4, whiteSpace: "nowrap" }}>✓ Saved</div>}
      </div>

      {/* Content */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "48px 24px" }}>
        <div style={{ width: "100%", maxWidth: 640 }} key={step} className="animate-slide-up">
          {/* Label */}
          <div style={{ fontSize: 12, color: "#6366F1", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>
            {current.title.split(" ").slice(0, 2).join(" ")} — Step {step + 1}/{STEPS.length}
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>{current.title}</h1>
          <p style={{ fontSize: 16, color: "#71717A", marginBottom: 40 }}>{current.subtitle}</p>

          {/* CARD SELECT */}
          {current.type === "card-select" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {(current.options as string[]).map(opt => {
                const sel = answers[current.id] === opt
                return (
                  <button key={opt} onClick={() => { setAnswers(a => ({ ...a, [current.id]: opt })); triggerSave() }}
                    style={{ padding: "20px", borderRadius: 14, border: `2px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "#111113", color: sel ? "#FAFAFA" : "#A1A1AA", fontSize: 15, fontWeight: 500, cursor: "pointer", textAlign: "left", transition: "all 0.15s" }}>
                    {opt}
                  </button>
                )
              })}
            </div>
          )}

          {/* CARD SELECT WITH ICON */}
          {current.type === "card-select-icon" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {(current.options as { label: string; emoji: string }[]).map(opt => {
                const sel = answers[current.id] === opt.label
                return (
                  <button key={opt.label} onClick={() => { setAnswers(a => ({ ...a, [current.id]: opt.label })); triggerSave() }}
                    style={{ padding: "20px", borderRadius: 14, border: `2px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "#111113", color: sel ? "#FAFAFA" : "#A1A1AA", fontSize: 15, fontWeight: 500, cursor: "pointer", textAlign: "left", transition: "all 0.15s", display: "flex", flexDirection: "column", gap: 8 }}>
                    <span style={{ fontSize: 28 }}>{opt.emoji}</span>
                    <span>{opt.label}</span>
                  </button>
                )
              })}
            </div>
          )}

          {/* MULTI CHIP */}
          {current.type === "multi-chip" && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {(current.options as string[]).map(opt => {
                const sel = multiSelect.includes(opt)
                return (
                  <button key={opt} onClick={() => toggleMulti(opt)}
                    style={{ padding: "10px 18px", borderRadius: 99, border: `2px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "#111113", color: sel ? "#818CF8" : "#71717A", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all 0.15s" }}>
                    {sel && "✓ "}{opt}
                  </button>
                )
              })}
            </div>
          )}

          {/* SLIDERS */}
          {current.type === "sliders" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
              {current.fields!.map(f => (
                <div key={f.key}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                    <label style={{ fontSize: 15, fontWeight: 500, color: "#FAFAFA" }}>{f.label}</label>
                    <span style={{ fontSize: 20, fontWeight: 700, color: "#6366F1" }}>{sliders[f.key]?.toFixed(1)}{f.suffix}</span>
                  </div>
                  <input type="range" min={f.min} max={f.max} step={f.step} value={sliders[f.key] ?? f.default}
                    onChange={e => { setSliders(s => ({ ...s, [f.key]: parseFloat(e.target.value) })); triggerSave() }}
                    style={{ width: "100%", accentColor: "#6366F1", height: 6 }} />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span style={{ fontSize: 11, color: "#52525B" }}>{f.min}</span>
                    <span style={{ fontSize: 11, color: "#52525B" }}>{f.max}</span>
                  </div>
                </div>
              ))}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 8 }}>
                {current.extras!.map(ex => (
                  <div key={ex.key}>
                    <label style={{ fontSize: 13, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 8 }}>{ex.label}</label>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {ex.options.map(o => {
                        const sel = answers[`${current.id}-${ex.key}`] === o
                        return (
                          <button key={o} onClick={() => { setAnswers(a => ({ ...a, [`${current.id}-${ex.key}`]: o })); triggerSave() }}
                            style={{ padding: "6px 12px", borderRadius: 8, border: `1px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "transparent", color: sel ? "#818CF8" : "#71717A", fontSize: 13, cursor: "pointer" }}>
                            {o}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* BUDGET */}
          {current.type === "budget" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                  <label style={{ fontSize: 15, fontWeight: 500, color: "#FAFAFA" }}>Annual Budget (USD)</label>
                  <div style={{ textAlign: "right" }}>
                    <span style={{ fontSize: 22, fontWeight: 700, color: "#6366F1" }}>${sliders.budget?.toLocaleString()}</span>
                    <div style={{ fontSize: 12, color: "#71717A" }}>≈ ₹{Math.round((sliders.budget || 25000) * 83.5 / 100000).toFixed(1)}L / year</div>
                  </div>
                </div>
                <input type="range" min={5000} max={60000} step={1000} value={sliders.budget ?? 25000}
                  onChange={e => { setSliders(s => ({ ...s, budget: parseInt(e.target.value) })); triggerSave() }}
                  style={{ width: "100%", accentColor: "#6366F1" }} />
              </div>
              <div>
                <label style={{ fontSize: 14, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 10 }}>Target intake</label>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {["Sep 2025", "Jan 2026", "Sep 2026", "Flexible"].map(opt => {
                    const sel = answers["intake"] === opt
                    return (
                      <button key={opt} onClick={() => setAnswers(a => ({ ...a, intake: opt }))}
                        style={{ padding: "12px", borderRadius: 12, border: `2px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "#111113", color: sel ? "#FAFAFA" : "#A1A1AA", fontSize: 14, cursor: "pointer" }}>
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {/* CHECKBOX */}
          {current.type === "checkbox" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {(current.options as string[]).map(opt => {
                const sel = checkboxes.includes(opt)
                return (
                  <button key={opt} onClick={() => toggleCheck(opt)}
                    style={{ display: "flex", alignItems: "center", gap: 12, padding: "14px 16px", borderRadius: 12, border: `2px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "#111113", color: sel ? "#FAFAFA" : "#A1A1AA", fontSize: 14, cursor: "pointer", textAlign: "left" }}>
                    <div style={{ width: 20, height: 20, borderRadius: 6, border: `2px solid ${sel ? "#6366F1" : "#3F3F46"}`, background: sel ? "#6366F1" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all 0.15s" }}>
                      {sel && <span style={{ fontSize: 12, color: "white" }}>✓</span>}
                    </div>
                    {opt}
                  </button>
                )
              })}
            </div>
          )}

          {/* WORK EXP */}
          {current.type === "work-exp" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {[
                { key: "company", label: "Company name (optional)", placeholder: "e.g. Infosys, TCS, startup..." },
                { key: "role", label: "Your role / title", placeholder: "e.g. Software Engineer" },
              ].map(f => (
                <div key={f.key}>
                  <label style={{ fontSize: 13, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 6 }}>{f.label}</label>
                  <input placeholder={f.placeholder} onChange={triggerSave} style={{ width: "100%", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, padding: "12px 16px", color: "#FAFAFA", fontSize: 14, outline: "none" }} />
                </div>
              ))}
              <div>
                <label style={{ fontSize: 13, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 8 }}>Years of experience</label>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {["0", "< 1 yr", "1 yr", "2 yrs", "3 yrs", "4 yrs", "5+ yrs"].map(y => {
                    const sel = answers["exp"] === y
                    return (
                      <button key={y} onClick={() => { setAnswers(a => ({ ...a, exp: y })); triggerSave() }}
                        style={{ padding: "8px 16px", borderRadius: 99, border: `1px solid ${sel ? "#6366F1" : "#27272A"}`, background: sel ? "#6366F115" : "transparent", color: sel ? "#818CF8" : "#71717A", fontSize: 14, cursor: "pointer" }}>
                        {y}
                      </button>
                    )
                  })}
                </div>
              </div>
              <div>
                <label style={{ fontSize: 13, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 6 }}>Describe your role (2 sentences)</label>
                <textarea placeholder="I built AI models at..." onChange={triggerSave} rows={3} style={{ width: "100%", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, padding: "12px 16px", color: "#FAFAFA", fontSize: 14, outline: "none", resize: "vertical" }} />
              </div>
            </div>
          )}

          {/* FINAL */}
          {current.type === "final" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div>
                <label style={{ fontSize: 13, fontWeight: 500, color: "#A1A1AA", display: "block", marginBottom: 6 }}>Anything else AI should know about you?</label>
                <textarea placeholder="e.g. I have a specific university in mind, I need scholarship, I have family in Germany..." rows={5} style={{ width: "100%", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, padding: "14px 16px", color: "#FAFAFA", fontSize: 14, outline: "none", resize: "vertical" }} />
              </div>
              <div style={{ padding: "20px", background: "#6366F108", border: "1px solid #6366F130", borderRadius: 14 }}>
                <div style={{ fontSize: 14, color: "#A1A1AA", marginBottom: 8 }}>🤖 Pathora AI will analyze:</div>
                {["500+ universities across 6 countries", "230+ scholarships matching your profile", "Visa requirements & processing times", "10-year ROI and salary projections", "Living costs and part-time earning potential"].map(item => (
                  <div key={item} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0" }}>
                    <span style={{ color: "#10B981", fontSize: 13 }}>✓</span>
                    <span style={{ fontSize: 13, color: "#71717A" }}>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 48 }}>
            {step > 0
              ? <button onClick={() => setStep(s => s - 1)} style={{ padding: "12px 24px", borderRadius: 12, border: "1px solid #27272A", background: "transparent", color: "#A1A1AA", fontSize: 15, fontWeight: 500, cursor: "pointer" }}>← Back</button>
              : <div />
            }
            <button onClick={next}
              style={{ padding: "14px 32px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", borderRadius: 12, color: "white", fontSize: 15, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 8, boxShadow: "0 4px 20px rgba(99,102,241,0.3)" }}>
              {step === STEPS.length - 1 ? "🚀 Generate My Report" : "Continue →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
