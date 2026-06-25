"use client"

import { useState } from "react"
import { scholarships } from "@/lib/mock-data"

const FILTERS = ["All", "Fully Funded", "Government", "India-Specific", "Merit-Based"]

export default function ScholarshipsPage() {
  const [filter, setFilter] = useState("All")
  const [sort, setSort] = useState<"match" | "amount" | "deadline">("match")
  const [search, setSearch] = useState("")
  const [expanded, setExpanded] = useState<string | null>(null)

  const filtered = scholarships
    .filter(s => {
      if (search && !s.name.toLowerCase().includes(search.toLowerCase()) && !s.country.toLowerCase().includes(search.toLowerCase())) return false
      if (filter === "All") return true
      return s.tags.some(t => t.includes(filter.replace("-", " ")))
    })
    .sort((a, b) => {
      if (sort === "match") return b.matchScore - a.matchScore
      if (sort === "deadline") return new Date(a.deadline).getTime() - new Date(b.deadline).getTime()
      return b.matchScore - a.matchScore
    })

  return (
    <div style={{ padding: "32px", maxWidth: 1000, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>🏆 Scholarship Explorer</h1>
        <p style={{ fontSize: 14, color: "#71717A" }}>
          Showing {filtered.length} scholarships matched to your profile — sorted by eligibility score
        </p>
      </div>

      {/* AI Match Banner */}
      <div style={{ background: "linear-gradient(135deg,#6366F110,#10B98110)", border: "1px solid #6366F130", borderRadius: 16, padding: "16px 24px", display: "flex", alignItems: "center", gap: 14, marginBottom: 24 }}>
        <div style={{ fontSize: 28 }}>✨</div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>Pathora AI matched you with 3 scholarships</div>
          <div style={{ fontSize: 13, color: "#71717A" }}>Based on your CGPA 8.4, Indian nationality, and AI/ML focus. Deutschlandstipendium is your top pick with 94% match!</div>
        </div>
        <div style={{ marginLeft: "auto", fontSize: 20, fontWeight: 700, color: "#10B981" }}>₹83K/mo</div>
      </div>

      {/* Filters */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, gap: 16 }}>
        <div style={{ display: "flex", gap: 8, flex: 1, overflowX: "auto" }}>
          {FILTERS.map(f => (
            <button key={f} onClick={() => setFilter(f)}
              style={{ padding: "8px 16px", borderRadius: 99, border: `1px solid ${filter === f ? "#6366F1" : "#27272A"}`, background: filter === f ? "#6366F120" : "transparent", color: filter === f ? "#6366F1" : "#71717A", fontSize: 13, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap" }}>
              {f}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search scholarships..." style={{ padding: "8px 14px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#FAFAFA", fontSize: 13, outline: "none", width: 200 }} />
          <select value={sort} onChange={e => setSort(e.target.value as "match" | "amount" | "deadline")}
            style={{ padding: "8px 12px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#A1A1AA", fontSize: 13, outline: "none", cursor: "pointer" }}>
            <option value="match">Sort: Match Score</option>
            <option value="deadline">Sort: Deadline</option>
          </select>
        </div>
      </div>

      {/* Scholarship cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {filtered.map(s => {
          const isExpanded = expanded === s.id
          const matchColor = s.matchScore >= 90 ? "#10B981" : s.matchScore >= 75 ? "#6366F1" : "#F59E0B"

          return (
            <div key={s.id} style={{ background: "#111113", border: `1px solid ${s.matchScore >= 90 ? "#10B98130" : "#27272A"}`, borderRadius: 20, overflow: "hidden", transition: "border-color 0.2s" }}>
              {/* Main row */}
              <div style={{ padding: "24px 28px", cursor: "pointer" }} onClick={() => setExpanded(isExpanded ? null : s.id)}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                  {/* Flag + icon */}
                  <div style={{ width: 56, height: 56, borderRadius: 16, background: "#18181B", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 30, flexShrink: 0 }}>
                    {s.flag}
                  </div>

                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                      <h3 style={{ fontSize: 17, fontWeight: 700, color: "#FAFAFA" }}>{s.name}</h3>
                      <span style={{ fontSize: 11, padding: "2px 8px", background: "#F59E0B20", color: "#F59E0B", borderRadius: 99, fontWeight: 600 }}>{s.type}</span>
                      {s.matchScore >= 90 && <span style={{ fontSize: 11, padding: "2px 8px", background: "#10B98120", color: "#10B981", borderRadius: 99, fontWeight: 600 }}>✨ Top Pick</span>}
                    </div>
                    <p style={{ fontSize: 13, color: "#71717A", marginBottom: 12 }}>{s.description}</p>
                    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: "#10B981" }}>{s.amount}</span>
                      <span style={{ fontSize: 13, color: "#71717A" }}>·</span>
                      <span style={{ fontSize: 13, color: "#71717A" }}>{s.duration}</span>
                      <span style={{ fontSize: 13, color: "#71717A" }}>·</span>
                      <span style={{ fontSize: 13, color: "#EF4444", fontWeight: 500 }}>Deadline: {s.deadline}</span>
                    </div>
                    <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
                      {s.tags.map(tag => (
                        <span key={tag} style={{ fontSize: 11, padding: "3px 10px", background: "#27272A", color: "#A1A1AA", borderRadius: 99 }}>{tag}</span>
                      ))}
                    </div>
                  </div>

                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                    <div style={{ fontSize: 26, fontWeight: 800, color: matchColor }}>{s.matchScore}%</div>
                    <div style={{ fontSize: 11, color: "#71717A" }}>eligible</div>
                    <div style={{ marginTop: 8, width: 60, height: 4, background: "#27272A", borderRadius: 99, marginLeft: "auto" }}>
                      <div style={{ height: "100%", background: matchColor, width: `${s.matchScore}%`, borderRadius: 99 }} />
                    </div>
                    <div style={{ fontSize: 12, color: "#52525B", marginTop: 8 }}>{isExpanded ? "▲ Less" : "▼ More"}</div>
                  </div>
                </div>
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div style={{ borderTop: "1px solid #1F1F22", padding: "24px 28px", background: "#0D0D10" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 20 }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#52525B", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>Eligibility</div>
                      <div style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.6 }}>{s.eligibility}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#52525B", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>Coverage Includes</div>
                      {s.coverage.map(c => (
                        <div key={c} style={{ fontSize: 13, color: "#10B981", marginBottom: 4 }}>✓ {c}</div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#52525B", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>Competition</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>{s.competitiveness}</div>
                      <div style={{ fontSize: 13, color: "#71717A" }}>Success rate: {s.successRate}</div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 10 }}>
                    <button style={{ padding: "10px 20px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", borderRadius: 10, color: "white", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
                      Apply Now →
                    </button>
                    <button style={{ padding: "10px 20px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#A1A1AA", fontSize: 14, cursor: "pointer" }}>
                      Save for Later
                    </button>
                    <button onClick={() => window.open(s.link, "_blank")} style={{ padding: "10px 20px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#A1A1AA", fontSize: 14, cursor: "pointer" }}>
                      Official Website ↗
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {filtered.length === 0 && (
        <div style={{ textAlign: "center", padding: "60px 20px" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "#FAFAFA", marginBottom: 8 }}>No scholarships found</div>
          <div style={{ fontSize: 14, color: "#71717A" }}>Try adjusting your filters or search term</div>
          <button onClick={() => { setFilter("All"); setSearch("") }} style={{ marginTop: 16, padding: "10px 20px", background: "#6366F1", border: "none", borderRadius: 10, color: "white", fontSize: 14, cursor: "pointer" }}>
            Clear Filters
          </button>
        </div>
      )}
    </div>
  )
}
