"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { visaData } from "@/lib/mock-data"

const TABS = ["Overview", "Documents", "Timeline", "Interview Tips", "FAQ"]

export default function VisaGuidePage() {
  const params = useParams()
  const [tab, setTab] = useState("Overview")
  const [checkedDocs, setCheckedDocs] = useState<number[]>([])
  const [timelineStep, setTimelineStep] = useState<number | null>(null)

  const countryKey = params.country as string
  const visa = visaData[countryKey as keyof typeof visaData]

  if (!visa) return (
    <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>
      Visa guide not available for this country yet.{" "}
      <Link href="/countries" style={{ color: "#6366F1" }}>Browse countries</Link>
    </div>
  )

  const toggleDoc = (i: number) => {
    setCheckedDocs(prev => prev.includes(i) ? prev.filter(x => x !== i) : [...prev, i])
  }

  const requiredDocs = visa.documents.filter(d => d.required)
  const readinessScore = checkedDocs.length > 0 ? Math.round((checkedDocs.length / requiredDocs.length) * 100) : 0

  return (
    <div style={{ padding: "32px", maxWidth: 1000, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 20, padding: "32px", marginBottom: 28, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -40, right: -40, width: 200, height: 200, borderRadius: "50%", background: "#6366F1", filter: "blur(80px)", opacity: 0.06 }} />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
              <span style={{ fontSize: 40 }}>{visa.flag}</span>
              <div>
                <h1 style={{ fontSize: 26, fontWeight: 700, color: "#FAFAFA" }}>{visa.visaName}</h1>
                <p style={{ fontSize: 13, color: "#71717A" }}>{visa.country} Study Visa Guide for Indian Students</p>
              </div>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "#10B981" }}>{visa.successRate}</div>
            <div style={{ fontSize: 12, color: "#71717A" }}>approval rate</div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginTop: 24 }}>
          {[
            { label: "Processing Time", value: visa.processingTime, icon: "⏱️" },
            { label: "Visa Fee", value: visa.fee, icon: "💳" },
            { label: "Work Rights", value: visa.workRights, icon: "💼" },
            { label: "Post-Grad Stay", value: visa.postGrad, icon: "🎓" },
          ].map(({ label, value, icon }) => (
            <div key={label} style={{ padding: "14px", background: "#18181B", borderRadius: 12, textAlign: "center" }}>
              <div style={{ fontSize: 18, marginBottom: 6 }}>{icon}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA", marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 11, color: "#52525B" }}>{label}</div>
            </div>
          ))}
        </div>
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
          {/* Risk meter */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>🛡️ Visa Risk Assessment — Your Profile</h3>
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 20 }}>
              <div style={{ position: "relative", width: 100, height: 100 }}>
                <svg width={100} height={100} style={{ transform: "rotate(-90deg)" }}>
                  <circle cx={50} cy={50} r={42} fill="none" stroke="#27272A" strokeWidth={8} />
                  <circle cx={50} cy={50} r={42} fill="none" stroke={visa.riskScore < 30 ? "#10B981" : "#F59E0B"} strokeWidth={8}
                    strokeDasharray={`${2 * Math.PI * 42}`}
                    strokeDashoffset={`${2 * Math.PI * 42 * (1 - visa.riskScore / 100)}`}
                    strokeLinecap="round" />
                </svg>
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: visa.riskScore < 30 ? "#10B981" : "#F59E0B" }}>{visa.riskScore}</div>
                  <div style={{ fontSize: 9, color: "#52525B" }}>risk</div>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: visa.riskScore < 30 ? "#10B981" : "#F59E0B", marginBottom: 4 }}>
                  {visa.riskScore < 30 ? "Low Risk" : "Medium Risk"}
                </div>
                <div style={{ fontSize: 14, color: "#71717A" }}>
                  {visa.riskScore < 30
                    ? "Your profile is well-suited for this visa. Follow the checklist and you'll be fine."
                    : "Take extra care with documentation to maximize your approval chances."}
                </div>
              </div>
            </div>
            <div>
              <h4 style={{ fontSize: 13, fontWeight: 600, color: "#A1A1AA", marginBottom: 10 }}>Action Checklist:</h4>
              {visa.checklist.map((item, i) => (
                <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8 }}>
                  <span style={{ color: "#6366F1", fontSize: 14 }}>→</span>
                  <span style={{ fontSize: 13, color: "#A1A1AA" }}>{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Embassy Info */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>🏛️ Embassy Contact</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {[
                { label: "Name", value: visa.embassy.name },
                { label: "Phone", value: visa.embassy.phone },
                { label: "Appointment Wait", value: visa.embassy.appointmentWait },
                { label: "Website", value: "Visit Official Site" },
              ].map(({ label, value }) => (
                <div key={label} style={{ padding: "12px 14px", background: "#18181B", borderRadius: 10 }}>
                  <div style={{ fontSize: 11, color: "#52525B", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Documents */}
      {tab === "Documents" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Progress */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>Document Readiness</h3>
              <span style={{ fontSize: 16, fontWeight: 700, color: readinessScore >= 80 ? "#10B981" : "#6366F1" }}>{readinessScore}%</span>
            </div>
            <div style={{ height: 6, background: "#27272A", borderRadius: 99, overflow: "hidden" }}>
              <div style={{ height: "100%", background: readinessScore >= 80 ? "#10B981" : "linear-gradient(90deg,#6366F1,#8B5CF6)", width: `${readinessScore}%`, borderRadius: 99, transition: "width 0.4s" }} />
            </div>
            <div style={{ fontSize: 12, color: "#71717A", marginTop: 8 }}>{checkedDocs.length} of {requiredDocs.length} required documents ready</div>
          </div>

          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📋 Document Checklist</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {visa.documents.map((doc, i) => {
                const isChecked = checkedDocs.includes(i)
                return (
                  <div key={i} onClick={() => toggleDoc(i)}
                    style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "14px 16px", background: isChecked ? "#10B98108" : "#18181B", borderRadius: 12, cursor: "pointer", border: `1px solid ${isChecked ? "#10B98120" : "transparent"}`, transition: "all 0.2s" }}>
                    <div style={{ width: 22, height: 22, borderRadius: 6, border: `2px solid ${isChecked ? "#10B981" : "#3F3F46"}`, background: isChecked ? "#10B981" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2, transition: "all 0.2s" }}>
                      {isChecked && <span style={{ fontSize: 12, color: "white" }}>✓</span>}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <span style={{ fontSize: 14, fontWeight: 600, color: isChecked ? "#10B981" : "#FAFAFA" }}>{doc.name}</span>
                        {doc.required && <span style={{ fontSize: 10, padding: "2px 6px", background: "#EF444415", color: "#EF4444", borderRadius: 99, fontWeight: 600 }}>Required</span>}
                      </div>
                      {doc.tip && <span style={{ fontSize: 12, color: "#71717A" }}>💡 {doc.tip}</span>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Timeline */}
      {tab === "Timeline" && (
        <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "28px" }}>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 24 }}>🗓️ Step-by-Step Visa Timeline</h3>
          <div style={{ position: "relative" }}>
            <div style={{ position: "absolute", left: 20, top: 0, bottom: 0, width: 2, background: "#27272A" }} />
            {visa.timeline.map((step, i) => (
              <div key={i} style={{ display: "flex", gap: 20, paddingLeft: 52, paddingBottom: 28, position: "relative", cursor: "pointer" }} onClick={() => setTimelineStep(timelineStep === i ? null : i)}>
                <div style={{ position: "absolute", left: 12, top: 2, width: 18, height: 18, borderRadius: "50%", background: "#6366F1", border: "2px solid #09090B", zIndex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: "white", fontWeight: 700 }}>
                  {i + 1}
                </div>
                <div style={{ flex: 1, background: "#18181B", borderRadius: 14, padding: "16px", border: `1px solid ${timelineStep === i ? "#6366F140" : "transparent"}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
                    <h4 style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{step.phase}</h4>
                    <span style={{ fontSize: 12, padding: "3px 8px", background: "#27272A", color: "#71717A", borderRadius: 99 }}>{step.timing}</span>
                  </div>
                  <div style={{ display: "flex", gap: 12, marginBottom: timelineStep === i ? 12 : 0 }}>
                    <span style={{ fontSize: 13, color: "#6366F1", fontWeight: 500 }}>⏱ {step.duration}</span>
                  </div>
                  {timelineStep === i && (
                    <p style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.6, marginTop: 8, paddingTop: 10, borderTop: "1px solid #27272A" }}>{step.description}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, padding: "14px 16px", background: "#6366F110", border: "1px solid #6366F130", borderRadius: 12, fontSize: 13, color: "#A1A1AA" }}>
            💡 Click any step to see detailed instructions. Total timeline: ~{visa.country === "Germany" ? "10-12" : "6-8"} months before departure.
          </div>
        </div>
      )}

      {/* Interview Tips */}
      {tab === "Interview Tips" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {visa.interviewTips.map((tip, i) => (
            <div key={i} style={{ display: "flex", gap: 14, padding: "18px 20px", background: "#111113", border: "1px solid #27272A", borderRadius: 14 }}>
              <div style={{ width: 32, height: 32, borderRadius: 10, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "white", flexShrink: 0 }}>{i + 1}</div>
              <span style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.6 }}>{tip}</span>
            </div>
          ))}

          <div style={{ background: "#F59E0B10", border: "1px solid #F59E0B30", borderRadius: 14, padding: "16px 20px" }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#F59E0B", marginBottom: 6 }}>⚠️ Common Mistakes to Avoid</div>
            <div style={{ fontSize: 13, color: "#71717A", lineHeight: 1.7 }}>
              Missing documents · Incomplete application form · Wrong document order · Unexplained gaps in study history · Insufficient financial proof
            </div>
          </div>
        </div>
      )}

      {/* FAQ */}
      {tab === "FAQ" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {visa.faq.length > 0 ? visa.faq.map((item, i) => (
            <div key={i} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "20px 24px" }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#FAFAFA", marginBottom: 10 }}>Q: {item.q}</div>
              <div style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.7 }}>A: {item.a}</div>
            </div>
          )) : (
            <div style={{ padding: 40, textAlign: "center", color: "#71717A" }}>FAQ coming soon for this country.</div>
          )}
          <div style={{ background: "#6366F110", border: "1px solid #6366F130", borderRadius: 14, padding: "16px 20px", display: "flex", gap: 14, alignItems: "center" }}>
            <span style={{ fontSize: 24 }}>🤖</span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA", marginBottom: 4 }}>Have more questions?</div>
              <div style={{ fontSize: 13, color: "#71717A" }}>Our AI advisor has answered 10,000+ visa questions from Indian students.</div>
            </div>
            <Link href="/chat" style={{ marginLeft: "auto", padding: "10px 18px", background: "#6366F1", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, textDecoration: "none", flexShrink: 0 }}>
              Ask AI →
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
