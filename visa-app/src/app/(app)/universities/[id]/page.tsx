"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { universities } from "@/lib/mock-data"

const TABS = ["Overview", "Programs", "Scholarships", "Campus Life", "Reviews", "Apply"]

export default function UniversityDetailPage() {
  const params = useParams()
  const [tab, setTab] = useState("Overview")

  const uni = universities.find(u => u.id === params.id)
  if (!uni) return (
    <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>
      University not found. <Link href="/countries" style={{ color: "#6366F1" }}>Go back</Link>
    </div>
  )

  const matchColor = uni.matchScore >= 90 ? "#10B981" : uni.matchScore >= 80 ? "#6366F1" : "#F59E0B"

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      {/* Hero */}
      <div style={{ position: "relative", height: 200, background: "#111113", overflow: "hidden" }}>
        <div style={{ position: "absolute", inset: 0, background: "linear-gradient(135deg,#0D0D18,#1a1a2e)" }}>
          <div style={{ position: "absolute", width: 500, height: 500, borderRadius: "50%", background: "#6366F1", filter: "blur(100px)", opacity: 0.07, top: "50%", left: "50%", transform: "translate(-50%,-50%)" }} />
        </div>
        <div style={{ position: "absolute", bottom: 24, left: 32, display: "flex", alignItems: "flex-end", gap: 20 }}>
          <div style={{ width: 72, height: 72, borderRadius: 16, background: "#18181B", border: "1px solid #27272A", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36, fontWeight: 800, color: "#6366F1" }}>
            {uni.shortName[0]}
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <h1 style={{ fontSize: 28, fontWeight: 800, color: "#FAFAFA" }}>{uni.name}</h1>
              <span style={{ fontSize: 11, padding: "2px 8px", background: "#6366F120", color: "#6366F1", borderRadius: 99 }}>{uni.qsRank}</span>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <span style={{ fontSize: 13, color: "#71717A" }}>📍 {uni.location}</span>
              <span style={{ fontSize: 12, color: "#52525B" }}>·</span>
              <span style={{ fontSize: 13, color: "#71717A" }}>Founded {uni.founded}</span>
              <span style={{ fontSize: 12, color: "#52525B" }}>·</span>
              <span style={{ fontSize: 13, color: "#71717A" }}>{uni.type}</span>
            </div>
          </div>
        </div>
        <div style={{ position: "absolute", bottom: 24, right: 32, textAlign: "right" }}>
          <div style={{ fontSize: 32, fontWeight: 800, color: matchColor }}>{uni.matchScore}%</div>
          <div style={{ fontSize: 12, color: "#71717A" }}>Your Match Score</div>
          <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "flex-end", marginTop: 4 }}>
            {"★★★★★".split("").slice(0, Math.round(uni.rating)).map((_, i) => (
              <span key={i} style={{ color: "#F59E0B", fontSize: 12 }}>★</span>
            ))}
            <span style={{ fontSize: 12, color: "#71717A" }}>{uni.rating}</span>
          </div>
        </div>
      </div>

      <div style={{ padding: "0 32px 40px" }}>
        {/* Quick stats */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, margin: "24px 0 28px" }}>
          {[
            { label: "Tuition", value: uni.tuitionFee, icon: "💰" },
            { label: "Acceptance Rate", value: `${uni.acceptanceRate}%`, icon: "📊" },
            { label: "Employment Rate", value: `${uni.employmentRate}%`, icon: "💼" },
            { label: "Avg Starting Salary", value: uni.avgSalary, icon: "📈" },
            { label: "Intl Students", value: `${uni.intlStudents}%`, icon: "🌍" },
          ].map(({ label, value, icon }) => (
            <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "16px", textAlign: "center" }}>
              <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 11, color: "#52525B" }}>{label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 12, padding: 4, marginBottom: 24 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "8px 16px", borderRadius: 8, border: "none", background: tab === t ? "#1C1C1F" : "transparent", color: tab === t ? "#FAFAFA" : "#71717A", fontSize: 14, fontWeight: 500, cursor: "pointer" }}>
              {t}
            </button>
          ))}
        </div>

        {/* Overview */}
        {tab === "Overview" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ background: "#111113", border: "1px solid #10B98130", borderRadius: 16, padding: "20px 24px" }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#10B981", marginBottom: 14 }}>✅ Why Choose {uni.shortName}</h3>
                {uni.pros.map(p => (
                  <div key={p} style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                    <span style={{ color: "#10B981" }}>✓</span>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{p}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: "#111113", border: "1px solid #EF444430", borderRadius: 16, padding: "20px 24px" }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#EF4444", marginBottom: 14 }}>⚠️ Things to Consider</h3>
                {uni.cons.map(p => (
                  <div key={p} style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                    <span style={{ color: "#EF4444" }}>!</span>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{p}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📋 Key Details</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10 }}>
                {[
                  { label: "Total Students", value: uni.students.toLocaleString() },
                  { label: "International %", value: `${uni.intlStudents}%` },
                  { label: "Research Output", value: uni.researchOutput },
                  { label: "Nobel Laureates", value: uni.nobelLaureates },
                  { label: "Campus Life", value: uni.campusLife },
                  { label: "Industry Partners", value: uni.industry },
                  { label: "Student Housing", value: uni.housing },
                  { label: "Avg Time to Job", value: uni.avgTimeToJob },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "12px 14px", background: "#18181B", borderRadius: 10 }}>
                    <div style={{ fontSize: 11, color: "#52525B", marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{String(value)}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {uni.tags.map(tag => (
                <span key={tag} style={{ padding: "6px 14px", background: "#6366F115", border: "1px solid #6366F130", color: "#6366F1", borderRadius: 99, fontSize: 13, fontWeight: 500 }}>{tag}</span>
              ))}
            </div>
          </div>
        )}

        {/* Programs */}
        {tab === "Programs" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📚 Available Programs</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {uni.programs.map((p, i) => (
                  <div key={p} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 16px", background: "#18181B", borderRadius: 12, border: i === 0 ? "1px solid #6366F130" : "1px solid transparent" }}>
                    <div style={{ width: 36, height: 36, borderRadius: 10, background: i === 0 ? "#6366F120" : "#27272A", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🎓</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{p}</div>
                      <div style={{ fontSize: 12, color: "#71717A", marginTop: 2 }}>
                        {uni.tuitionFee} · IELTS {uni.ielts}+ · GRE: {uni.gre}
                      </div>
                    </div>
                    {i === 0 && <span style={{ fontSize: 11, padding: "3px 10px", background: "#6366F120", color: "#6366F1", borderRadius: 99, fontWeight: 600 }}>Best Match</span>}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>📅 Application Deadlines</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {Object.entries(uni.deadlines).map(([term, date]) => (
                  <div key={term} style={{ padding: "14px", background: "#18181B", borderRadius: 12 }}>
                    <div style={{ fontSize: 11, color: "#52525B", textTransform: "capitalize", marginBottom: 6 }}>{term} Intake</div>
                    <div style={{ fontSize: 15, fontWeight: 700, color: "#F59E0B" }}>{date}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Scholarships */}
        {tab === "Scholarships" && (
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>🏆 Available Scholarships</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {uni.scholarships.map((s, i) => (
                <div key={s} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 16px", background: "#18181B", borderRadius: 12 }}>
                  <div style={{ width: 36, height: 36, borderRadius: 10, background: "#F59E0B20", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🏆</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{s}</div>
                    <div style={{ fontSize: 12, color: "#71717A" }}>Check eligibility at the scholarship page</div>
                  </div>
                  <Link href="/scholarships" style={{ padding: "7px 14px", background: "#6366F115", border: "1px solid #6366F130", borderRadius: 8, color: "#6366F1", fontSize: 12, fontWeight: 600, textDecoration: "none" }}>
                    Details →
                  </Link>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Campus Life */}
        {tab === "Campus Life" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>🎉 Campus Experience</h3>
              <p style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.7, marginBottom: 20 }}>{uni.campusLife}</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 12 }}>
                {[
                  { label: "Student Housing", value: uni.housing, icon: "🏠" },
                  { label: "Industry Partners", value: uni.industry, icon: "🤝" },
                  { label: "Research Output", value: uni.researchOutput, icon: "🔬" },
                  { label: "Nobel Laureates", value: `${uni.nobelLaureates} alumni`, icon: "🏅" },
                ].map(({ label, value, icon }) => (
                  <div key={label} style={{ padding: "16px", background: "#18181B", borderRadius: 12 }}>
                    <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA", marginBottom: 4 }}>{value}</div>
                    <div style={{ fontSize: 11, color: "#52525B" }}>{label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Reviews */}
        {tab === "Reviews" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {uni.reviews.map((r, i) => (
              <div key={i} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 36, height: 36, borderRadius: "50%", background: `hsl(${220 + i * 40},60%,50%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "white" }}>
                      {r.name[0]}
                    </div>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{r.name}</div>
                      <div style={{ fontSize: 12, color: "#71717A" }}>Alumni · {uni.name}</div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 2 }}>
                    {"★★★★★".split("").map((_, si) => (
                      <span key={si} style={{ color: si < r.rating ? "#F59E0B" : "#27272A", fontSize: 14 }}>★</span>
                    ))}
                  </div>
                </div>
                <p style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.6, fontStyle: "italic" }}>"{r.text}"</p>
              </div>
            ))}
          </div>
        )}

        {/* Apply */}
        {tab === "Apply" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "linear-gradient(135deg,#6366F110,#8B5CF610)", border: "1px solid #6366F130", borderRadius: 16, padding: "32px", textAlign: "center" }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>🎓</div>
              <h3 style={{ fontSize: 22, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>Apply to {uni.shortName}</h3>
              <p style={{ fontSize: 14, color: "#71717A", marginBottom: 24 }}>
                Deadlines: Winter — <strong style={{ color: "#F59E0B" }}>{uni.deadlines.winter}</strong>
              </p>
              <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
                <a href={uni.applyUrl} style={{ padding: "12px 28px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 12, color: "white", fontSize: 15, fontWeight: 700, textDecoration: "none" }}>
                  Start Application →
                </a>
                <Link href="/chat" style={{ padding: "12px 24px", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, color: "#A1A1AA", fontSize: 15, fontWeight: 500, textDecoration: "none" }}>
                  Ask AI for Help
                </Link>
              </div>
            </div>

            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>📋 Requirements Checklist</h3>
              {[
                `CGPA above 7.5/10 (yours: 8.4 ✅)`,
                `IELTS ${uni.ielts}+ (yours: 7.5 ✅)`,
                `GRE: ${uni.gre}`,
                "Statement of Purpose",
                "2-3 Letters of Recommendation",
                "Academic Transcripts (notarized)",
                "CV/Resume",
              ].map((req, i) => (
                <div key={i} style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                  <span style={{ color: req.includes("✅") ? "#10B981" : "#6366F1", fontWeight: 700 }}>{req.includes("✅") ? "✓" : "○"}</span>
                  <span style={{ fontSize: 13, color: req.includes("✅") ? "#10B981" : "#A1A1AA" }}>{req}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
