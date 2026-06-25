"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"

const GOALS = [
  { id: "masters", label: "Masters Degree", icon: "🎓", desc: "MS / MEng / MBA" },
  { id: "phd", label: "PhD", icon: "🔬", desc: "Research Doctorate" },
  { id: "bachelor", label: "Bachelors", icon: "📚", desc: "Undergraduate" },
  { id: "job", label: "Job Search", icon: "💼", desc: "Career relocation" },
]

const FIELDS = ["AI/ML", "Computer Science", "Data Science", "Engineering", "Business", "Finance", "Healthcare", "Design", "Law", "Biology"]

const COUNTRIES = [
  { id: "germany", name: "Germany", flag: "🇩🇪" },
  { id: "canada", name: "Canada", flag: "🇨🇦" },
  { id: "uk", name: "United Kingdom", flag: "🇬🇧" },
  { id: "australia", name: "Australia", flag: "🇦🇺" },
  { id: "netherlands", name: "Netherlands", flag: "🇳🇱" },
  { id: "sweden", name: "Sweden", flag: "🇸🇪" },
  { id: "usa", name: "USA", flag: "🇺🇸" },
  { id: "singapore", name: "Singapore", flag: "🇸🇬" },
]

export default function OnboardingPage() {
  const router = useRouter()
  const [step, setStep] = useState(0)
  const [data, setData] = useState({
    name: "",
    goal: "",
    fields: [] as string[],
    countries: [] as string[],
    cgpa: 8.0,
    budget: 20,
  })

  const steps = [
    { title: "What's your name?", sub: "Let's personalize your experience" },
    { title: "What's your goal?", sub: "We'll tailor recommendations to your objective" },
    { title: "What field interests you?", sub: "Select all that apply" },
    { title: "Which countries interest you?", sub: "Select up to 4 countries to compare" },
    { title: "Your academic profile", sub: "This helps us match you with the right programs" },
  ]

  const progress = ((step + 1) / steps.length) * 100

  const toggleField = (f: string) => {
    setData(d => ({ ...d, fields: d.fields.includes(f) ? d.fields.filter(x => x !== f) : [...d.fields, f] }))
  }

  const toggleCountry = (id: string) => {
    setData(d => {
      if (d.countries.includes(id)) return { ...d, countries: d.countries.filter(x => x !== id) }
      if (d.countries.length >= 4) return d
      return { ...d, countries: [...d.countries, id] }
    })
  }

  const canNext = () => {
    if (step === 0) return data.name.trim().length > 1
    if (step === 1) return !!data.goal
    if (step === 2) return data.fields.length > 0
    if (step === 3) return data.countries.length > 0
    return true
  }

  const handleNext = () => {
    if (step < steps.length - 1) setStep(s => s + 1)
    else router.push("/questionnaire")
  }

  return (
    <div style={{ minHeight: "100vh", background: "#09090B", display: "flex", alignItems: "center", justifyContent: "center", padding: "24px" }}>
      {/* Background */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none" }}>
        <div style={{ position: "absolute", width: 500, height: 500, borderRadius: "50%", background: "#6366F1", filter: "blur(120px)", opacity: 0.05, top: "30%", left: "50%", transform: "translate(-50%,-50%)" }} />
      </div>

      <div style={{ width: "100%", maxWidth: 560, position: "relative", zIndex: 1 }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 18, color: "white" }}>P</div>
            <span style={{ fontSize: 22, fontWeight: 800, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Pathora</span>
          </div>
        </div>

        {/* Progress */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: "#71717A" }}>Step {step + 1} of {steps.length}</span>
            <span style={{ fontSize: 13, color: "#6366F1", fontWeight: 600 }}>{Math.round(progress)}%</span>
          </div>
          <div style={{ height: 4, background: "#27272A", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#8B5CF6)", width: `${progress}%`, transition: "width 0.4s ease", borderRadius: 99 }} />
          </div>
        </div>

        {/* Card */}
        <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 24, padding: "40px 36px" }}>
          <h2 style={{ fontSize: 26, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>{steps[step].title}</h2>
          <p style={{ fontSize: 14, color: "#71717A", marginBottom: 32 }}>{steps[step].sub}</p>

          {/* Step 0: Name */}
          {step === 0 && (
            <div>
              <input
                autoFocus
                value={data.name}
                onChange={e => setData(d => ({ ...d, name: e.target.value }))}
                onKeyDown={e => e.key === "Enter" && canNext() && handleNext()}
                placeholder="Your full name..."
                style={{ width: "100%", background: "#18181B", border: "1px solid #3F3F46", borderRadius: 14, padding: "16px 20px", color: "#FAFAFA", fontSize: 18, fontWeight: 500, outline: "none", transition: "border-color 0.2s" }}
                onFocus={e => (e.target.style.borderColor = "#6366F1")}
                onBlur={e => (e.target.style.borderColor = "#3F3F46")}
              />
              {data.name && (
                <p style={{ marginTop: 16, fontSize: 14, color: "#6366F1" }}>
                  Welcome, {data.name.split(" ")[0]}! Let's find your perfect destination. ✨
                </p>
              )}
            </div>
          )}

          {/* Step 1: Goal */}
          {step === 1 && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {GOALS.map(g => (
                <button key={g.id} onClick={() => setData(d => ({ ...d, goal: g.id }))}
                  style={{ padding: "20px", background: data.goal === g.id ? "#6366F115" : "#18181B", border: `2px solid ${data.goal === g.id ? "#6366F1" : "#27272A"}`, borderRadius: 16, cursor: "pointer", textAlign: "left", transition: "all 0.2s" }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>{g.icon}</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: data.goal === g.id ? "#6366F1" : "#FAFAFA" }}>{g.label}</div>
                  <div style={{ fontSize: 12, color: "#71717A", marginTop: 4 }}>{g.desc}</div>
                </button>
              ))}
            </div>
          )}

          {/* Step 2: Fields */}
          {step === 2 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {FIELDS.map(f => (
                <button key={f} onClick={() => toggleField(f)}
                  style={{ padding: "10px 18px", background: data.fields.includes(f) ? "#6366F120" : "#18181B", border: `1.5px solid ${data.fields.includes(f) ? "#6366F1" : "#27272A"}`, borderRadius: 99, color: data.fields.includes(f) ? "#6366F1" : "#A1A1AA", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all 0.2s" }}>
                  {data.fields.includes(f) && "✓ "}{f}
                </button>
              ))}
            </div>
          )}

          {/* Step 3: Countries */}
          {step === 3 && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {COUNTRIES.map(c => (
                  <button key={c.id} onClick={() => toggleCountry(c.id)}
                    style={{ padding: "14px 18px", background: data.countries.includes(c.id) ? "#6366F115" : "#18181B", border: `2px solid ${data.countries.includes(c.id) ? "#6366F1" : "#27272A"}`, borderRadius: 14, display: "flex", alignItems: "center", gap: 10, cursor: "pointer", transition: "all 0.2s" }}>
                    <span style={{ fontSize: 22 }}>{c.flag}</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: data.countries.includes(c.id) ? "#6366F1" : "#FAFAFA" }}>{c.name}</span>
                    {data.countries.includes(c.id) && <span style={{ marginLeft: "auto", fontSize: 16, color: "#6366F1" }}>✓</span>}
                  </button>
                ))}
              </div>
              <p style={{ marginTop: 12, fontSize: 12, color: "#52525B" }}>{data.countries.length}/4 selected</p>
            </div>
          )}

          {/* Step 4: CGPA + Budget */}
          {step === 4 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                  <label style={{ fontSize: 14, fontWeight: 600, color: "#A1A1AA" }}>Your CGPA</label>
                  <span style={{ fontSize: 18, fontWeight: 700, color: "#6366F1" }}>{data.cgpa.toFixed(1)}/10</span>
                </div>
                <input type="range" min={4} max={10} step={0.1} value={data.cgpa}
                  onChange={e => setData(d => ({ ...d, cgpa: parseFloat(e.target.value) }))}
                  style={{ width: "100%", accentColor: "#6366F1" }} />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                  <span style={{ fontSize: 11, color: "#52525B" }}>4.0</span>
                  <span style={{ fontSize: 11, color: "#52525B" }}>10.0</span>
                </div>
                <div style={{ marginTop: 12, padding: "10px 14px", background: "#18181B", borderRadius: 10, fontSize: 13, color: data.cgpa >= 8 ? "#10B981" : data.cgpa >= 7 ? "#F59E0B" : "#EF4444" }}>
                  {data.cgpa >= 8.5 ? "🌟 Excellent — Qualifies for top universities globally" :
                    data.cgpa >= 8 ? "✅ Strong — Great for most programs" :
                    data.cgpa >= 7 ? "👍 Good — Many solid options available" :
                    "💡 Average — Strong SOP can make a difference"}
                </div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                  <label style={{ fontSize: 14, fontWeight: 600, color: "#A1A1AA" }}>Annual Budget</label>
                  <span style={{ fontSize: 18, fontWeight: 700, color: "#6366F1" }}>₹{data.budget}L</span>
                </div>
                <input type="range" min={5} max={60} step={1} value={data.budget}
                  onChange={e => setData(d => ({ ...d, budget: parseInt(e.target.value) }))}
                  style={{ width: "100%", accentColor: "#6366F1" }} />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                  <span style={{ fontSize: 11, color: "#52525B" }}>₹5L</span>
                  <span style={{ fontSize: 11, color: "#52525B" }}>₹60L</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 24 }}>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            style={{ padding: "12px 24px", background: "transparent", border: "1px solid #27272A", borderRadius: 12, color: step === 0 ? "#3F3F46" : "#A1A1AA", fontSize: 14, fontWeight: 500, cursor: step === 0 ? "default" : "pointer", transition: "all 0.2s" }}>
            ← Back
          </button>
          <div style={{ display: "flex", gap: 6 }}>
            {steps.map((_, i) => (
              <div key={i} style={{ width: i === step ? 20 : 6, height: 6, borderRadius: 99, background: i <= step ? "#6366F1" : "#27272A", transition: "all 0.3s" }} />
            ))}
          </div>
          <button onClick={handleNext} disabled={!canNext()}
            style={{ padding: "12px 28px", background: canNext() ? "linear-gradient(135deg,#6366F1,#8B5CF6)" : "#27272A", border: "none", borderRadius: 12, color: canNext() ? "white" : "#52525B", fontSize: 14, fontWeight: 600, cursor: canNext() ? "pointer" : "default", transition: "all 0.2s" }}>
            {step === steps.length - 1 ? "Start AI Analysis →" : "Continue →"}
          </button>
        </div>
      </div>
    </div>
  )
}
