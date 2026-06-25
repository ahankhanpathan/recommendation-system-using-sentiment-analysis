"use client"

import { useState } from "react"
import Link from "next/link"
import { countries } from "@/lib/mock-data"

const SORT_OPTIONS = ["Match Score", "Avg Salary", "Tuition Cost", "PR Ease"]
const SCORE_KEYS = ["prScore", "safetyScore", "healthcareScore", "educationScore"] as const

export default function CountriesPage() {
  const [sort, setSort] = useState("Match Score")
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [view, setView] = useState<"grid" | "compare">("grid")

  const toggleCompare = (id: string) => {
    setCompareIds(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id)
      if (prev.length >= 3) return prev
      return [...prev, id]
    })
  }

  const sorted = [...countries].sort((a, b) => {
    if (sort === "Match Score") return b.score - a.score
    if (sort === "Tuition Cost") return a.visaDifficultyScore - b.visaDifficultyScore
    if (sort === "PR Ease") return b.prScore - a.prScore
    return b.score - a.score
  })

  const compareCountries = countries.filter(c => compareIds.includes(c.id))

  return (
    <div style={{ padding: "32px", maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>🌍 Country Explorer</h1>
        <p style={{ fontSize: 14, color: "#71717A" }}>Compare 6 top destinations ranked by AI match score for your profile</p>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 8 }}>
          {SORT_OPTIONS.map(o => (
            <button key={o} onClick={() => setSort(o)}
              style={{ padding: "8px 16px", borderRadius: 10, border: `1px solid ${sort === o ? "#6366F1" : "#27272A"}`, background: sort === o ? "#6366F115" : "transparent", color: sort === o ? "#6366F1" : "#71717A", fontSize: 13, fontWeight: 500, cursor: "pointer" }}>
              {o}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {compareIds.length > 0 && (
            <button onClick={() => setView(view === "compare" ? "grid" : "compare")}
              style={{ padding: "9px 18px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
              Compare {compareIds.length} Countries →
            </button>
          )}
          <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 10, padding: 4 }}>
            {(["grid", "compare"] as const).map(v => (
              <button key={v} onClick={() => setView(v)}
                style={{ padding: "6px 14px", borderRadius: 8, border: "none", background: view === v ? "#27272A" : "transparent", color: view === v ? "#FAFAFA" : "#71717A", fontSize: 13, cursor: "pointer", textTransform: "capitalize" }}>
                {v}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid view */}
      {view === "grid" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 20 }}>
          {sorted.map((c, i) => (
            <div key={c.id} style={{ background: "#111113", border: `1px solid ${i === 0 ? "#6366F140" : "#27272A"}`, borderRadius: 20, overflow: "hidden", transition: "transform 0.2s, border-color 0.2s", cursor: "pointer" }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = "translateY(-3px)"; (e.currentTarget as HTMLElement).style.borderColor = i === 0 ? "#6366F170" : "#3F3F46" }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = "none"; (e.currentTarget as HTMLElement).style.borderColor = i === 0 ? "#6366F140" : "#27272A" }}>
              {/* Top section */}
              <div style={{ padding: "24px 24px 20px", background: i === 0 ? "linear-gradient(135deg,#6366F110,#18181B)" : "#18181B" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 32 }}>{c.flag}</span>
                      {i === 0 && <span style={{ fontSize: 10, padding: "2px 8px", background: "#6366F1", color: "white", borderRadius: 99, fontWeight: 700 }}>YOUR #1</span>}
                    </div>
                    <h3 style={{ fontSize: 18, fontWeight: 700, color: "#FAFAFA" }}>{c.name}</h3>
                    <p style={{ fontSize: 12, color: "#71717A", marginTop: 2 }}>{c.tagline}</p>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 28, fontWeight: 800, color: i === 0 ? "#6366F1" : "#10B981" }}>{c.score}</div>
                    <div style={{ fontSize: 11, color: "#52525B" }}>/ 100</div>
                  </div>
                </div>

                {/* Score bars */}
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { label: "PR Score", value: c.prScore },
                    { label: "Job Market", value: c.score },
                    { label: "Safety", value: c.safetyScore },
                  ].map(({ label, value }) => (
                    <div key={label}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: "#71717A" }}>{label}</span>
                        <span style={{ fontSize: 11, fontWeight: 600, color: "#A1A1AA" }}>{value}%</span>
                      </div>
                      <div style={{ height: 3, background: "#27272A", borderRadius: 99 }}>
                        <div style={{ height: "100%", background: i === 0 ? "#6366F1" : "#10B981", width: `${value}%`, borderRadius: 99, transition: "width 0.8s" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Stats grid */}
              <div style={{ padding: "16px 24px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[
                  { label: "Avg Salary", value: c.avgSalaryINR },
                  { label: "Living Cost", value: c.livingCost },
                  { label: "Tuition", value: c.avgTuition },
                  { label: "PR Path", value: `${c.pr.years} yrs` },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "10px 12px", background: "#18181B", borderRadius: 10 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                    <div style={{ fontSize: 10, color: "#52525B", marginTop: 2 }}>{label}</div>
                  </div>
                ))}
              </div>

              {/* Pros/Cons */}
              <div style={{ padding: "0 24px 16px" }}>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {c.pros.slice(0, 2).map(p => (
                    <span key={p} style={{ fontSize: 11, padding: "3px 8px", background: "#10B98110", color: "#10B981", borderRadius: 99 }}>✓ {p}</span>
                  ))}
                  {c.cons.slice(0, 1).map(p => (
                    <span key={p} style={{ fontSize: 11, padding: "3px 8px", background: "#EF444410", color: "#EF4444", borderRadius: 99 }}>✗ {p}</span>
                  ))}
                </div>
              </div>

              {/* Footer */}
              <div style={{ padding: "14px 24px", borderTop: "1px solid #27272A", display: "flex", gap: 8 }}>
                <Link href={`/countries/${c.id}`} style={{ flex: 1, padding: "9px", background: i === 0 ? "linear-gradient(135deg,#6366F1,#8B5CF6)" : "#18181B", border: i === 0 ? "none" : "1px solid #3F3F46", borderRadius: 10, color: i === 0 ? "white" : "#A1A1AA", fontSize: 13, fontWeight: 600, textDecoration: "none", textAlign: "center" }}>
                  Explore →
                </Link>
                <button onClick={() => toggleCompare(c.id)}
                  style={{ padding: "9px 14px", background: compareIds.includes(c.id) ? "#6366F120" : "transparent", border: `1px solid ${compareIds.includes(c.id) ? "#6366F1" : "#27272A"}`, borderRadius: 10, color: compareIds.includes(c.id) ? "#6366F1" : "#71717A", fontSize: 13, cursor: "pointer" }}>
                  {compareIds.includes(c.id) ? "✓" : "+"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Compare view */}
      {view === "compare" && compareCountries.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0 }}>
            <thead>
              <tr>
                <th style={{ padding: "16px", textAlign: "left", color: "#71717A", fontSize: 13, fontWeight: 600, width: 180 }}>Metric</th>
                {compareCountries.map(c => (
                  <th key={c.id} style={{ padding: "16px", textAlign: "center", background: "#111113", border: "1px solid #27272A", borderBottom: "none" }}>
                    <div style={{ fontSize: 28, marginBottom: 4 }}>{c.flag}</div>
                    <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>{c.name}</div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: "#6366F1" }}>{c.score}%</div>
                    <div style={{ fontSize: 11, color: "#71717A" }}>match</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Avg Salary (INR)", key: "avgSalaryINR" },
                { label: "Living Cost", key: "livingCost" },
                { label: "Avg Tuition", key: "avgTuition" },
                { label: "PR Timeline", key: null, fn: (c: typeof countries[0]) => `${c.pr.years} years` },
                { label: "Part-time Work", key: "partTimeRights" },
                { label: "Language", key: "language" },
                { label: "Visa Difficulty", key: "visaDifficulty" },
                { label: "Job Growth", key: "jobGrowth" },
                { label: "Safety Score", key: "safetyScore" },
                { label: "PR Score", key: "prScore" },
                { label: "Scholarships", key: "scholarships" },
                { label: "Universities", key: "universities" },
              ].map(({ label, key, fn }, ri) => (
                <tr key={label}>
                  <td style={{ padding: "14px 16px", fontSize: 13, color: "#71717A", fontWeight: 500, borderBottom: "1px solid #1F1F22" }}>{label}</td>
                  {compareCountries.map(c => (
                    <td key={c.id} style={{ padding: "14px 16px", textAlign: "center", background: ri % 2 === 0 ? "#111113" : "#0D0D10", border: "1px solid #1F1F22", fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>
                      {fn ? fn(c) : String((c as any)[key as string])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {view === "compare" && compareCountries.length === 0 && (
        <div style={{ textAlign: "center", padding: "60px 20px" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>⚖️</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "#FAFAFA", marginBottom: 8 }}>Select countries to compare</div>
          <div style={{ fontSize: 14, color: "#71717A" }}>Switch to Grid view and click the + button on countries to add them here.</div>
          <button onClick={() => setView("grid")} style={{ marginTop: 20, padding: "10px 24px", background: "#6366F1", border: "none", borderRadius: 10, color: "white", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
            Browse Countries →
          </button>
        </div>
      )}
    </div>
  )
}
