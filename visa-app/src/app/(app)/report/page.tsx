"use client"

import { useState } from "react"
import Link from "next/link"
import { aiReport, demoUser, countries, universities, scholarships } from "@/lib/mock-data"

const TABS = ["Overview", "Countries", "Universities", "Scholarships", "Career", "Financial", "Risks", "Roadmap"]

export default function ReportPage() {
  const [tab, setTab] = useState("Overview")

  const topCountry = countries.find(c => c.id === "germany")!
  const recUniversities = universities.filter(u => aiReport.universities.includes(u.id))
  const recScholarships = scholarships.filter(s => aiReport.scholarships.includes(s.id))

  return (
    <div style={{ padding: "32px", maxWidth: 1100, margin: "0 auto" }}>
      {/* Report Header */}
      <div style={{ background: "linear-gradient(135deg,#111118,#18181B)", border: "1px solid #27272A", borderRadius: 24, padding: "36px 40px", marginBottom: 28, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -60, right: -60, width: 300, height: 300, borderRadius: "50%", background: "#6366F1", filter: "blur(100px)", opacity: 0.08 }} />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", background: "#6366F115", border: "1px solid #6366F130", borderRadius: 99, marginBottom: 16 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#10B981" }} />
                <span style={{ fontSize: 12, color: "#6366F1", fontWeight: 600 }}>AI Report · Generated {aiReport.generatedAt}</span>
              </div>
              <h1 style={{ fontSize: 32, fontWeight: 800, color: "#FAFAFA", marginBottom: 8 }}>
                {demoUser.name}'s Relocation Roadmap
              </h1>
              <p style={{ fontSize: 15, color: "#71717A", maxWidth: 600, lineHeight: 1.6 }}>{aiReport.summary}</p>
            </div>
            <div style={{ textAlign: "center", flexShrink: 0 }}>
              <div style={{ width: 100, height: 100, borderRadius: "50%", background: "conic-gradient(#6366F1 0% 92%, #27272A 92% 100%)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 8px" }}>
                <div style={{ width: 80, height: 80, borderRadius: "50%", background: "#18181B", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#6366F1" }}>{aiReport.overallScore}</div>
                  <div style={{ fontSize: 10, color: "#71717A" }}>/ 100</div>
                </div>
              </div>
              <div style={{ fontSize: 12, color: "#71717A" }}>Profile Score</div>
              <div style={{ fontSize: 11, padding: "3px 10px", background: "#10B98120", color: "#10B981", borderRadius: 99, marginTop: 6, fontWeight: 600 }}>{aiReport.readinessLevel} Readiness</div>
            </div>
          </div>

          {/* Quick stats */}
          <div style={{ display: "flex", gap: 24, marginTop: 28, paddingTop: 24, borderTop: "1px solid #27272A" }}>
            {[
              { label: "Top Destination", value: `${topCountry.flag} ${topCountry.name}`, color: "#6366F1" },
              { label: "Confidence Level", value: aiReport.confidenceLevel, color: "#10B981" },
              { label: "Visa Timeline", value: aiReport.visaPathway.timeline, color: "#F59E0B" },
              { label: "10-Year ROI", value: aiReport.roi.rateOfReturn, color: "#8B5CF6" },
              { label: "Payback Period", value: `${aiReport.roi.paybackYears} years`, color: "#A1A1AA" },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <div style={{ fontSize: 11, color: "#52525B", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 16, fontWeight: 700, color }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: 6, marginBottom: 24, overflowX: "auto" }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: "8px 16px", borderRadius: 10, border: "none", background: tab === t ? "#1C1C1F" : "transparent", color: tab === t ? "#FAFAFA" : "#71717A", fontSize: 14, fontWeight: 500, cursor: "pointer", whiteSpace: "nowrap", transition: "all 0.2s" }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === "Overview" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Strengths & Improvements */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div style={{ background: "#111113", border: "1px solid #10B98130", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#10B981", marginBottom: 16 }}>✅ Profile Strengths</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {aiReport.strengths.map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                    <div style={{ width: 20, height: 20, borderRadius: 6, background: "#10B98120", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: "#10B981", flexShrink: 0, marginTop: 2 }}>✓</div>
                    <span style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.5 }}>{s}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "#111113", border: "1px solid #F59E0B30", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#F59E0B", marginBottom: 16 }}>💡 Improvements</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {aiReport.improvements.map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                    <div style={{ width: 20, height: 20, borderRadius: 6, background: "#F59E0B20", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: "#F59E0B", flexShrink: 0, marginTop: 2 }}>!</div>
                    <span style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.5 }}>{s}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* AI Recommendations */}
          <div style={{ background: "#111113", border: "1px solid #6366F130", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>🎯 Pathora's Recommendations</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {aiReport.recommendations.map((r, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 14, padding: "14px 16px", background: "#18181B", borderRadius: 12, border: "1px solid #27272A" }}>
                  <div style={{ width: 28, height: 28, borderRadius: 8, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, color: "white", flexShrink: 0 }}>{i + 1}</div>
                  <span style={{ fontSize: 14, color: "#D4D4D8", lineHeight: 1.5 }}>{r}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ROI Summary */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>💰 Financial Overview</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16 }}>
              {[
                { label: "Total Investment", value: aiReport.roi.totalCost, color: "#EF4444", icon: "📤" },
                { label: "Year 1 Income", value: aiReport.roi.yearOneIncome, color: "#10B981", icon: "📈" },
                { label: "Payback Period", value: `${aiReport.roi.paybackYears} years`, color: "#F59E0B", icon: "⏱️" },
                { label: "10-Year Wealth", value: aiReport.roi.tenYearWealth, color: "#6366F1", icon: "🏦" },
              ].map(({ label, value, color, icon }) => (
                <div key={label} style={{ textAlign: "center", padding: "20px 16px", background: "#18181B", borderRadius: 14 }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color, marginBottom: 4 }}>{value}</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Countries Tab */}
      {tab === "Countries" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {countries.map((c, i) => (
            <div key={c.id} style={{ background: "#111113", border: `1px solid ${i === 0 ? "#6366F140" : "#27272A"}`, borderRadius: 16, padding: "24px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
                <span style={{ fontSize: 36 }}>{c.flag}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                    <h3 style={{ fontSize: 18, fontWeight: 700, color: "#FAFAFA" }}>{c.name}</h3>
                    {i === 0 && <span style={{ fontSize: 11, padding: "2px 8px", background: "#6366F120", color: "#6366F1", borderRadius: 99, fontWeight: 600 }}>✨ Top Pick</span>}
                  </div>
                  <p style={{ fontSize: 13, color: "#71717A" }}>{c.tagline}</p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 28, fontWeight: 800, color: i === 0 ? "#6366F1" : "#10B981" }}>{c.score}%</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>match score</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 12, marginBottom: 16 }}>
                {[
                  { label: "Avg Salary", value: c.avgSalary },
                  { label: "Living Cost", value: c.livingCost },
                  { label: "Tuition", value: c.avgTuition },
                  { label: "PR Path", value: `${c.pr.years} years` },
                  { label: "ROI", value: c.roi },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "12px", background: "#18181B", borderRadius: 10, textAlign: "center" }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA", marginBottom: 4 }}>{value}</div>
                    <div style={{ fontSize: 11, color: "#52525B" }}>{label}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 12 }}>
                <Link href={`/countries/${c.id}`} style={{ padding: "10px 20px", background: i === 0 ? "linear-gradient(135deg,#6366F1,#8B5CF6)" : "#18181B", border: i === 0 ? "none" : "1px solid #27272A", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, textDecoration: "none" }}>
                  Explore {c.name} →
                </Link>
                <Link href={`/visa/${c.id}`} style={{ padding: "10px 20px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#A1A1AA", fontSize: 13, fontWeight: 500, textDecoration: "none" }}>
                  Visa Guide
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Universities Tab */}
      {tab === "Universities" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {recUniversities.map((u, i) => (
            <div key={u.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <h3 style={{ fontSize: 18, fontWeight: 700, color: "#FAFAFA" }}>{u.name}</h3>
                    <span style={{ fontSize: 11, padding: "2px 8px", background: "#6366F115", color: "#6366F1", borderRadius: 99 }}>{u.qsRank}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "#71717A" }}>📍 {u.location} · Founded {u.founded}</p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: "#10B981" }}>{u.matchScore}%</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>your match</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 16 }}>
                {[
                  { label: "Tuition", value: u.tuitionFee },
                  { label: "Acceptance", value: `${u.acceptanceRate}%` },
                  { label: "Avg Salary", value: u.avgSalary },
                  { label: "Employment", value: `${u.employmentRate}%` },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "12px", background: "#18181B", borderRadius: 10, textAlign: "center" }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                    <div style={{ fontSize: 11, color: "#52525B", marginTop: 3 }}>{label}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {u.tags.map(tag => (
                  <span key={tag} style={{ fontSize: 11, padding: "4px 10px", background: "#27272A", color: "#A1A1AA", borderRadius: 99 }}>{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Scholarships Tab */}
      {tab === "Scholarships" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {recScholarships.map(s => (
            <div key={s.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <span style={{ fontSize: 24 }}>{s.flag}</span>
                    <h3 style={{ fontSize: 17, fontWeight: 700, color: "#FAFAFA" }}>{s.name}</h3>
                    <span style={{ fontSize: 11, padding: "2px 8px", background: "#F59E0B15", color: "#F59E0B", borderRadius: 99, fontWeight: 600 }}>{s.type}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "#71717A" }}>{s.description}</p>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color: "#10B981" }}>{s.matchScore}%</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>eligible</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 14 }}>
                {[
                  { label: "Amount", value: s.amount },
                  { label: "Duration", value: s.duration },
                  { label: "Success Rate", value: s.successRate },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "10px 14px", background: "#18181B", borderRadius: 10 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                    <div style={{ fontSize: 11, color: "#52525B", marginTop: 3 }}>{label}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 13, color: "#EF4444", fontWeight: 500 }}>Deadline: {s.deadline}</span>
                <Link href="/scholarships" style={{ padding: "8px 18px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, textDecoration: "none" }}>
                  Apply Now →
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Career Tab */}
      {tab === "Career" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>🚀 Career Trajectories — {topCountry.name}</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {aiReport.careerFit.roles.map((r, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 16, padding: "16px", background: "#18181B", borderRadius: 12 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 15, fontWeight: 600, color: "#FAFAFA", marginBottom: 4 }}>{r.title}</div>
                    <div style={{ display: "flex", gap: 12 }}>
                      <span style={{ fontSize: 12, color: "#71717A" }}>Avg: {r.avgSalary}/yr</span>
                      <span style={{ fontSize: 12, color: "#10B981" }}>Growth: {r.growth}</span>
                    </div>
                  </div>
                  <span style={{ fontSize: 12, padding: "4px 10px", background: r.demand === "Very High" ? "#10B98120" : "#6366F115", color: r.demand === "Very High" ? "#10B981" : "#6366F1", borderRadius: 99, fontWeight: 600 }}>{r.demand}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>💼 Salary Progression</h3>
              {[
                { level: "Entry Level (0-2 yrs)", salary: aiReport.expectedSalary.entry, color: "#6366F1" },
                { level: "Mid Level (3-5 yrs)", salary: aiReport.expectedSalary.mid, color: "#10B981" },
                { level: "Senior (6+ yrs)", salary: aiReport.expectedSalary.senior, color: "#F59E0B" },
              ].map(({ level, salary, color }, i, arr) => (
                <div key={level} style={{ marginBottom: i < arr.length - 1 ? 16 : 0 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: "#71717A" }}>{level}</span>
                    <span style={{ fontSize: 14, fontWeight: 700, color }}>{salary}/yr</span>
                  </div>
                  <div style={{ height: 6, background: "#27272A", borderRadius: 99 }}>
                    <div style={{ height: "100%", background: color, width: `${(i + 1) * 33}%`, borderRadius: 99 }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>🌐 PR Pathway</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ padding: "12px", background: "#18181B", borderRadius: 10 }}>
                  <div style={{ fontSize: 13, color: "#71717A", marginBottom: 4 }}>Route</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{topCountry.pr.path}</div>
                </div>
                <div style={{ padding: "12px", background: "#18181B", borderRadius: 10 }}>
                  <div style={{ fontSize: 13, color: "#71717A", marginBottom: 4 }}>Timeline</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#10B981" }}>{topCountry.pr.years} years to PR</div>
                </div>
                <div style={{ padding: "12px", background: "#18181B", borderRadius: 10 }}>
                  <div style={{ fontSize: 13, color: "#71717A", marginBottom: 4 }}>Difficulty</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#F59E0B" }}>{topCountry.pr.difficulty}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Financial Tab */}
      {tab === "Financial" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📊 Monthly Budget — Munich</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {Object.entries({
                  Rent: aiReport.livingCosts.rent,
                  Food: aiReport.livingCosts.food,
                  Transport: aiReport.livingCosts.transport,
                  Utilities: aiReport.livingCosts.utilities,
                  Entertainment: aiReport.livingCosts.entertainment,
                }).map(([key, val]) => (
                  <div key={key} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: "1px solid #1F1F22" }}>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{key}</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{val}/mo</span>
                  </div>
                ))}
                <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 0 0" }}>
                  <span style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA" }}>Total</span>
                  <span style={{ fontSize: 16, fontWeight: 700, color: "#6366F1" }}>{aiReport.livingCosts.total}/mo</span>
                </div>
                <div style={{ padding: "12px 14px", background: "#10B98110", borderRadius: 10, border: "1px solid #10B98120" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 13, color: "#10B981" }}>Part-time income</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#10B981" }}>+{aiReport.livingCosts.partTimeEarning}/mo</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                    <span style={{ fontSize: 13, color: "#10B981" }}>Net monthly</span>
                    <span style={{ fontSize: 14, fontWeight: 700, color: "#10B981" }}>-{aiReport.livingCosts.netMonthly}/mo</span>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📈 Return on Investment</h3>
              <div style={{ textAlign: "center", padding: "24px 0" }}>
                <div style={{ fontSize: 52, fontWeight: 800, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                  {aiReport.roi.rateOfReturn}
                </div>
                <div style={{ fontSize: 14, color: "#71717A", marginBottom: 24 }}>10-year rate of return</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[
                    { label: "Total Investment", value: aiReport.roi.totalCost, color: "#EF4444" },
                    { label: "Year 1 Salary", value: aiReport.roi.yearOneIncome, color: "#10B981" },
                    { label: "Payback in", value: `${aiReport.roi.paybackYears} years`, color: "#F59E0B" },
                    { label: "10-Year Wealth", value: aiReport.roi.tenYearWealth, color: "#6366F1" },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "10px 14px", background: "#18181B", borderRadius: 10 }}>
                      <span style={{ fontSize: 13, color: "#71717A" }}>{label}</span>
                      <span style={{ fontSize: 14, fontWeight: 700, color }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Risks Tab */}
      {tab === "Risks" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ background: "#10B98110", border: "1px solid #10B98130", borderRadius: 14, padding: "16px 20px", display: "flex", gap: 12, alignItems: "center" }}>
            <span style={{ fontSize: 20 }}>🛡️</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#10B981" }}>Overall Risk Score: Low (23/100)</div>
              <div style={{ fontSize: 12, color: "#71717A" }}>Your profile mitigates most common application risks. Here's what to watch for:</div>
            </div>
          </div>
          {aiReport.risks.map((r, i) => (
            <div key={i} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>⚠️ {r.risk}</h3>
                <span style={{ fontSize: 12, padding: "4px 10px", background: r.probability.includes("Low") ? "#10B98120" : "#F59E0B20", color: r.probability.includes("Low") ? "#10B981" : "#F59E0B", borderRadius: 99, fontWeight: 600 }}>{r.probability}</span>
              </div>
              <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span style={{ fontSize: 16, color: "#6366F1" }}>→</span>
                <p style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.6 }}><strong style={{ color: "#FAFAFA" }}>Mitigation:</strong> {r.mitigation}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Roadmap Tab */}
      {tab === "Roadmap" && (
        <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "28px" }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 24 }}>🗓️ Your Month-by-Month Roadmap</h3>
          <div style={{ position: "relative" }}>
            <div style={{ position: "absolute", left: 20, top: 0, bottom: 0, width: 2, background: "#27272A" }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {aiReport.timeline.map((item, i) => (
                <div key={i} style={{ display: "flex", gap: 20, paddingLeft: 52, position: "relative", paddingBottom: 24 }}>
                  <div style={{ position: "absolute", left: 12, top: 2, width: 18, height: 18, borderRadius: "50%", background: item.type === "milestone" ? "#6366F1" : item.type === "visa" ? "#F59E0B" : item.type === "financial" ? "#10B981" : "#8B5CF6", border: "2px solid #09090B", zIndex: 1 }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, color: "#52525B", fontWeight: 600, marginBottom: 4 }}>{item.month}</div>
                    <div style={{ fontSize: 14, color: "#D4D4D8", fontWeight: item.type === "milestone" ? 600 : 400 }}>
                      {item.type === "milestone" ? "🎯 " : ""}{item.task}
                    </div>
                  </div>
                  <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 99, height: "fit-content", background: item.type === "milestone" ? "#6366F120" : item.type === "visa" ? "#F59E0B20" : item.type === "financial" ? "#10B98120" : "#8B5CF620", color: item.type === "milestone" ? "#6366F1" : item.type === "visa" ? "#F59E0B" : item.type === "financial" ? "#10B981" : "#8B5CF6", fontWeight: 600, whiteSpace: "nowrap" }}>{item.type}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
