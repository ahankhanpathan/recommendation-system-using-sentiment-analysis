"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { countries, universities, scholarships } from "@/lib/mock-data"

const TABS = ["Overview", "Universities", "Scholarships", "Visa", "Cost of Living", "Jobs"]

export default function CountryDetailPage() {
  const params = useParams()
  const [tab, setTab] = useState("Overview")

  const country = countries.find(c => c.id === params.id)
  if (!country) return (
    <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>
      Country not found. <Link href="/countries" style={{ color: "#6366F1" }}>Go back</Link>
    </div>
  )

  const countryUniversities = universities.filter(u => u.countryId === country.id)
  const countryScholarships = scholarships.filter(s => s.country === country.name)

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      {/* Hero */}
      <div style={{ position: "relative", height: 240, background: "#18181B", overflow: "hidden" }}>
        <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, transparent 0%, #09090B 100%)" }} />
        <div style={{ position: "absolute", inset: 0, background: "#1a1a2e", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ position: "absolute", width: 400, height: 400, borderRadius: "50%", background: "#6366F1", filter: "blur(120px)", opacity: 0.1 }} />
          <span style={{ fontSize: 100, filter: "drop-shadow(0 20px 40px rgba(0,0,0,0.5))" }}>{country.flag}</span>
        </div>
        <div style={{ position: "absolute", bottom: 24, left: 32 }}>
          <Link href="/countries" style={{ fontSize: 13, color: "#71717A", textDecoration: "none", display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
            ← Countries
          </Link>
          <h1 style={{ fontSize: 36, fontWeight: 800, color: "#FAFAFA" }}>{country.flag} {country.name}</h1>
          <p style={{ fontSize: 14, color: "#A1A1AA", marginTop: 4 }}>{country.tagline} · #{country.rank} Ranked Destination</p>
        </div>
        <div style={{ position: "absolute", bottom: 24, right: 32, textAlign: "right" }}>
          <div style={{ fontSize: 42, fontWeight: 800, color: "#6366F1" }}>{country.score}%</div>
          <div style={{ fontSize: 12, color: "#71717A" }}>Match Score for You</div>
        </div>
      </div>

      <div style={{ padding: "0 32px 40px" }}>
        {/* Quick stats bar */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, margin: "24px 0 28px" }}>
          {[
            { label: "Avg Salary", value: country.avgSalaryINR, icon: "💰" },
            { label: "Living Cost", value: country.livingCost, icon: "🏠" },
            { label: "PR Path", value: `${country.pr.years} years`, icon: "🛂" },
            { label: "Tuition", value: country.avgTuition, icon: "🎓" },
            { label: "Job Growth", value: country.jobGrowth, icon: "📈" },
          ].map(({ label, value, icon }) => (
            <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "16px", textAlign: "center" }}>
              <div style={{ fontSize: 20, marginBottom: 8 }}>{icon}</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 11, color: "#52525B" }}>{label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 12, padding: 4, marginBottom: 24, overflowX: "auto" }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "8px 16px", borderRadius: 8, border: "none", background: tab === t ? "#1C1C1F" : "transparent", color: tab === t ? "#FAFAFA" : "#71717A", fontSize: 14, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap" }}>
              {t}
            </button>
          ))}
        </div>

        {/* Overview */}
        {tab === "Overview" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* About */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 12 }}>About {country.name}</h3>
              <p style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.7 }}>{country.overview}</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginTop: 20 }}>
                {[
                  { label: "Population", value: country.population },
                  { label: "Currency", value: country.currency },
                  { label: "Language", value: country.language },
                  { label: "Timezone", value: country.timezone },
                  { label: "Weather", value: country.weather },
                  { label: "GDP", value: country.gdp },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "12px 14px", background: "#18181B", borderRadius: 10 }}>
                    <div style={{ fontSize: 11, color: "#52525B", marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Score breakdown */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>📊 Score Breakdown</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {[
                  { label: "PR Pathway", value: country.prScore, color: "#6366F1" },
                  { label: "Safety", value: country.safetyScore, color: "#10B981" },
                  { label: "Healthcare", value: country.healthcareScore, color: "#F59E0B" },
                  { label: "Education Quality", value: country.educationScore, color: "#8B5CF6" },
                ].map(({ label, value, color }) => (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 13, color: "#A1A1AA" }}>{label}</span>
                      <span style={{ fontSize: 13, fontWeight: 700, color }}>{value}/100</span>
                    </div>
                    <div style={{ height: 6, background: "#27272A", borderRadius: 99 }}>
                      <div style={{ height: "100%", background: color, width: `${value}%`, borderRadius: 99, transition: "width 1s ease" }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Pros/Cons */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ background: "#111113", border: "1px solid #10B98130", borderRadius: 16, padding: "20px 24px" }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#10B981", marginBottom: 14 }}>✅ Pros</h3>
                {country.pros.map(p => (
                  <div key={p} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <span style={{ color: "#10B981", fontSize: 14 }}>✓</span>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{p}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: "#111113", border: "1px solid #EF444430", borderRadius: 16, padding: "20px 24px" }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#EF4444", marginBottom: 14 }}>❌ Cons</h3>
                {country.cons.map(p => (
                  <div key={p} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <span style={{ color: "#EF4444", fontSize: 14 }}>✗</span>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{p}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Top cities */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>🏙️ Top Cities</h3>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {country.topCities.map(city => (
                  <span key={city} style={{ padding: "8px 16px", background: "#18181B", border: "1px solid #27272A", borderRadius: 99, fontSize: 13, color: "#A1A1AA" }}>📍 {city}</span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Universities */}
        {tab === "Universities" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {countryUniversities.length > 0 ? countryUniversities.map(u => (
              <div key={u.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                      <h3 style={{ fontSize: 17, fontWeight: 700, color: "#FAFAFA" }}>{u.name}</h3>
                      <span style={{ fontSize: 11, padding: "2px 8px", background: "#6366F115", color: "#6366F1", borderRadius: 99 }}>{u.qsRank}</span>
                    </div>
                    <p style={{ fontSize: 13, color: "#71717A" }}>📍 {u.location}</p>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#10B981" }}>{u.matchScore}%</div>
                    <div style={{ fontSize: 11, color: "#71717A" }}>match</div>
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 14 }}>
                  {[
                    { label: "Tuition", value: u.tuitionFee },
                    { label: "Acceptance", value: `${u.acceptanceRate}%` },
                    { label: "Avg Salary", value: u.avgSalary },
                  ].map(({ label, value }) => (
                    <div key={label} style={{ padding: "10px 14px", background: "#18181B", borderRadius: 10 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                      <div style={{ fontSize: 11, color: "#52525B", marginTop: 3 }}>{label}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                  {u.tags.map(tag => (
                    <span key={tag} style={{ fontSize: 11, padding: "3px 10px", background: "#27272A", color: "#A1A1AA", borderRadius: 99 }}>{tag}</span>
                  ))}
                </div>
                <Link href={`/universities/${u.id}`} style={{ padding: "9px 18px", background: "#6366F115", border: "1px solid #6366F130", borderRadius: 10, color: "#6366F1", fontSize: 13, fontWeight: 600, textDecoration: "none" }}>
                  View Full Profile →
                </Link>
              </div>
            )) : (
              <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>
                No universities listed for {country.name} yet. More coming soon!
              </div>
            )}
          </div>
        )}

        {/* Scholarships */}
        {tab === "Scholarships" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {countryScholarships.length > 0 ? countryScholarships.map(s => (
              <div key={s.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>{s.name}</h3>
                    <p style={{ fontSize: 13, color: "#71717A", marginBottom: 12 }}>{s.description}</p>
                    <div style={{ display: "flex", gap: 10 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: "#10B981" }}>{s.amount}</span>
                      <span style={{ fontSize: 12, color: "#71717A" }}>·</span>
                      <span style={{ fontSize: 12, color: "#71717A" }}>Deadline: {s.deadline}</span>
                    </div>
                  </div>
                  <Link href="/scholarships" style={{ padding: "9px 16px", background: "#6366F1", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, textDecoration: "none" }}>
                    Apply →
                  </Link>
                </div>
              </div>
            )) : (
              <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>No scholarships listed for {country.name}.</div>
            )}
          </div>
        )}

        {/* Visa */}
        {tab === "Visa" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>🛂 Visa Overview</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 20 }}>
                {[
                  { label: "Visa Types", value: country.visaTypes[0] },
                  { label: "Difficulty", value: country.visaDifficulty },
                  { label: "Work Rights", value: country.partTimeRights },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "14px", background: "#18181B", borderRadius: 12 }}>
                    <div style={{ fontSize: 11, color: "#52525B", marginBottom: 6 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                  </div>
                ))}
              </div>
              <Link href={`/visa/${country.id}`} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "10px 20px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 10, color: "white", fontSize: 14, fontWeight: 600, textDecoration: "none" }}>
                View Complete Visa Guide →
              </Link>
            </div>
          </div>
        )}

        {/* Cost of Living */}
        {tab === "Cost of Living" && (
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>💰 Cost of Living — {country.topCities[0]}</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { item: "🏠 Rent (shared room)", cost: country.id === "germany" ? "€350-600" : country.id === "canada" ? "CAD $800-1200" : country.id === "uk" ? "£600-900" : "$800-1200", pct: 45 },
                { item: "🛒 Groceries & Food", cost: country.id === "germany" ? "€150-250" : "CAD $300-500", pct: 20 },
                { item: "🚌 Transport", cost: country.id === "germany" ? "€80-100 (semester ticket)" : "CAD $120-180", pct: 10 },
                { item: "📡 Internet & Phone", cost: "€20-40", pct: 5 },
                { item: "📚 Books & Supplies", cost: "€30-60", pct: 5 },
                { item: "🎉 Entertainment", cost: "€100-200", pct: 15 },
              ].map(({ item, cost, pct }) => (
                <div key={item} style={{ display: "flex", alignItems: "center", gap: 14, padding: "12px 14px", background: "#18181B", borderRadius: 10 }}>
                  <span style={{ fontSize: 14, color: "#A1A1AA", flex: 1 }}>{item}</span>
                  <span style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA", width: 160, textAlign: "right" }}>{cost}</span>
                  <div style={{ width: 80, height: 4, background: "#27272A", borderRadius: 99 }}>
                    <div style={{ height: "100%", background: "#6366F1", width: `${pct}%`, borderRadius: 99 }} />
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 20, padding: "16px", background: "#10B98110", border: "1px solid #10B98120", borderRadius: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#10B981" }}>💡 Total Monthly: {country.livingCost}</div>
              <div style={{ fontSize: 12, color: "#71717A", marginTop: 4 }}>Students typically cover 40-60% of expenses through part-time work</div>
            </div>
          </div>
        )}

        {/* Jobs */}
        {tab === "Jobs" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>💼 Job Market Overview</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 20 }}>
                {[
                  { label: "Job Market", value: country.jobMarket, color: "#10B981" },
                  { label: "Growth Rate", value: country.jobGrowth, color: "#6366F1" },
                  { label: "Avg Salary", value: country.avgSalary, color: "#F59E0B" },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ padding: "20px", background: "#18181B", borderRadius: 14, textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color, marginBottom: 4 }}>{value}</div>
                    <div style={{ fontSize: 12, color: "#71717A" }}>{label}</div>
                  </div>
                ))}
              </div>
              <h4 style={{ fontSize: 14, fontWeight: 600, color: "#A1A1AA", marginBottom: 12 }}>Top Sectors:</h4>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {country.topSectors.map(s => (
                  <span key={s} style={{ padding: "6px 14px", background: "#27272A", color: "#A1A1AA", borderRadius: 99, fontSize: 13 }}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
