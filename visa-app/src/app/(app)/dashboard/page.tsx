"use client"

import { useState } from "react"
import Link from "next/link"
import { demoUser, countries, aiReport, scholarships } from "@/lib/mock-data"

const TASKS = [
  { id: 1, label: "Complete APS appointment booking", done: false, priority: "High", tag: "Visa", link: "/visa/germany" },
  { id: 2, label: "Write Statement of Purpose draft", done: false, priority: "High", tag: "Application", link: "/report" },
  { id: 3, label: "Open Deutsche Bank blocked account", done: false, priority: "Medium", tag: "Financial", link: "#" },
  { id: 4, label: "Apply for DAAD Scholarship", done: false, priority: "High", tag: "Scholarship", link: "/scholarships" },
  { id: 5, label: "Register on Uni-Assist portal", done: true, priority: "Medium", tag: "Application", link: "#" },
  { id: 6, label: "Get transcripts notarized & apostilled", done: true, priority: "Medium", tag: "Document", link: "#" },
]

const METRICS = [
  { label: "Profile Score", value: "92%", change: "+4%", icon: "⚡", color: "#6366F1", bg: "#6366F110" },
  { label: "Application Readiness", value: "67%", change: "+12%", icon: "📋", color: "#10B981", bg: "#10B98110" },
  { label: "Scholarships Eligible", value: "3", change: "of 6", icon: "🏆", color: "#F59E0B", bg: "#F59E0B10" },
  { label: "Days to Deadline", value: "127", change: "DAAD Oct 15", icon: "📅", color: "#EF4444", bg: "#EF444410" },
]

export default function DashboardPage() {
  const [tasks, setTasks] = useState(TASKS)

  const toggleTask = (id: number) => {
    setTasks(t => t.map(task => task.id === id ? { ...task, done: !task.done } : task))
  }

  const topCountries = countries.slice(0, 3)
  const topScholarships = scholarships.slice(0, 3)
  const completedTasks = tasks.filter(t => t.done).length
  const totalTasks = tasks.length

  return (
    <div style={{ padding: "32px", maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>
            Good morning, {demoUser.name.split(" ")[0]} 👋
          </h1>
          <p style={{ fontSize: 15, color: "#71717A" }}>
            You're {completedTasks}/{totalTasks} tasks complete — keep going!
          </p>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <Link href="/questionnaire" style={{ padding: "10px 18px", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, color: "#A1A1AA", fontSize: 14, fontWeight: 500, textDecoration: "none", display: "flex", alignItems: "center", gap: 6 }}>
            ✏️ Update Profile
          </Link>
          <Link href="/report" style={{ padding: "10px 18px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 12, color: "white", fontSize: 14, fontWeight: 600, textDecoration: "none", display: "flex", alignItems: "center", gap: 6 }}>
            📊 View Full Report →
          </Link>
        </div>
      </div>

      {/* Key Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
        {METRICS.map((m, i) => (
          <div key={i} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div style={{ width: 40, height: 40, borderRadius: 12, background: m.bg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>{m.icon}</div>
              <span style={{ fontSize: 12, color: m.color, background: m.bg, padding: "3px 8px", borderRadius: 99, fontWeight: 600 }}>{m.change}</span>
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: m.color, marginBottom: 4 }}>{m.value}</div>
            <div style={{ fontSize: 13, color: "#71717A" }}>{m.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20, marginBottom: 20 }}>
        {/* Main content */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* AI Insight banner */}
          <div style={{ background: "linear-gradient(135deg,#6366F110,#8B5CF610)", border: "1px solid #6366F130", borderRadius: 16, padding: "20px 24px", display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ width: 48, height: 48, borderRadius: 14, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, flexShrink: 0 }}>✨</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 4 }}>Pathora AI Insight</div>
              <div style={{ fontSize: 13, color: "#A1A1AA", lineHeight: 1.5 }}>
                {aiReport.summary.slice(0, 140)}...
              </div>
            </div>
            <Link href="/report" style={{ padding: "10px 18px", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 10, color: "white", fontSize: 13, fontWeight: 600, textDecoration: "none", flexShrink: 0 }}>
              Full Report →
            </Link>
          </div>

          {/* Top Countries */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA" }}>🌍 Top Country Matches</h3>
              <Link href="/countries" style={{ fontSize: 13, color: "#6366F1", textDecoration: "none", fontWeight: 500 }}>See all →</Link>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {topCountries.map((c, i) => (
                <Link key={c.id} href={`/countries/${c.id}`} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px", background: "#18181B", borderRadius: 12, textDecoration: "none", border: "1px solid transparent", transition: "border-color 0.2s" }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = "#3F3F46")}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = "transparent")}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: i === 0 ? "#6366F120" : "#27272A", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: i === 0 ? "#6366F1" : "#71717A" }}>#{c.rank}</div>
                  <span style={{ fontSize: 24 }}>{c.flag}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{c.name}</div>
                    <div style={{ fontSize: 12, color: "#71717A" }}>{c.tagline}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: i === 0 ? "#6366F1" : "#10B981" }}>{c.score}%</div>
                    <div style={{ fontSize: 11, color: "#52525B" }}>match</div>
                  </div>
                  <div style={{ width: 60 }}>
                    <div style={{ height: 4, background: "#27272A", borderRadius: 99 }}>
                      <div style={{ height: "100%", background: i === 0 ? "#6366F1" : "#10B981", width: `${c.score}%`, borderRadius: 99 }} />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Checklist */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA" }}>✅ Action Checklist</h3>
              <span style={{ fontSize: 12, color: "#71717A" }}>{completedTasks}/{totalTasks} done</span>
            </div>
            <div style={{ height: 4, background: "#27272A", borderRadius: 99, marginBottom: 20, overflow: "hidden" }}>
              <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#10B981)", width: `${(completedTasks / totalTasks) * 100}%`, borderRadius: 99, transition: "width 0.4s" }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {tasks.map(t => (
                <div key={t.id} onClick={() => toggleTask(t.id)}
                  style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 14px", background: t.done ? "#10B98108" : "#18181B", borderRadius: 12, cursor: "pointer", border: `1px solid ${t.done ? "#10B98120" : "transparent"}`, transition: "all 0.2s", opacity: t.done ? 0.7 : 1 }}>
                  <div style={{ width: 22, height: 22, borderRadius: 6, border: `2px solid ${t.done ? "#10B981" : "#3F3F46"}`, background: t.done ? "#10B981" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all 0.2s" }}>
                    {t.done && <span style={{ fontSize: 12, color: "white" }}>✓</span>}
                  </div>
                  <span style={{ flex: 1, fontSize: 14, color: t.done ? "#71717A" : "#FAFAFA", textDecoration: t.done ? "line-through" : "none" }}>{t.label}</span>
                  <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 99, background: t.priority === "High" ? "#EF444415" : "#F59E0B15", color: t.priority === "High" ? "#EF4444" : "#F59E0B", fontWeight: 500 }}>{t.priority}</span>
                  <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 99, background: "#27272A", color: "#71717A" }}>{t.tag}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Profile card */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
              <div style={{ width: 48, height: 48, borderRadius: "50%", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, fontWeight: 700, color: "white" }}>
                {demoUser.name[0]}
              </div>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>{demoUser.name}</div>
                <div style={{ fontSize: 12, color: "#71717A" }}>{demoUser.degree} · {demoUser.nationality}</div>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { label: "CGPA", value: `${demoUser.cgpa}/10`, color: "#6366F1" },
                { label: "IELTS", value: `${demoUser.ielts}/9`, color: "#10B981" },
                { label: "Work Exp", value: `${demoUser.workExp} year`, color: "#F59E0B" },
                { label: "Target", value: demoUser.targetDegree, color: "#8B5CF6" },
                { label: "Budget", value: `₹${demoUser.budget}L/yr`, color: "#A1A1AA" },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid #1F1F22" }}>
                  <span style={{ fontSize: 13, color: "#71717A" }}>{label}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Scholarships */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>🏆 Scholarships For You</h3>
              <Link href="/scholarships" style={{ fontSize: 12, color: "#6366F1", textDecoration: "none" }}>All →</Link>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {topScholarships.map(s => (
                <div key={s.id} style={{ padding: "12px", background: "#18181B", borderRadius: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 18 }}>{s.flag}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{s.name}</span>
                    </div>
                    <span style={{ fontSize: 12, fontWeight: 700, color: "#10B981" }}>{s.matchScore}%</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 12, color: "#71717A" }}>{s.amount}</span>
                    <span style={{ fontSize: 11, color: "#EF4444" }}>Due {s.deadline.split(",")[0]}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick links */}
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>⚡ Quick Actions</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                { label: "View My AI Report", href: "/report", icon: "📊" },
                { label: "Compare Countries", href: "/countries", icon: "🌍" },
                { label: "Visa Timeline", href: "/visa/germany", icon: "📋" },
                { label: "Ask AI Advisor", href: "/chat", icon: "💬" },
                { label: "ROI Calculator", href: "/calculator", icon: "🧮" },
              ].map(link => (
                <Link key={link.href} href={link.href}
                  style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", background: "#18181B", borderRadius: 10, textDecoration: "none", color: "#A1A1AA", fontSize: 14, transition: "all 0.2s" }}
                  onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = "#FAFAFA"; (e.currentTarget as HTMLElement).style.background = "#27272A" }}
                  onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = "#A1A1AA"; (e.currentTarget as HTMLElement).style.background = "#18181B" }}>
                  <span>{link.icon}</span>
                  <span>{link.label}</span>
                  <span style={{ marginLeft: "auto" }}>→</span>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom banner */}
      <div style={{ background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 16, padding: "24px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "white", marginBottom: 4 }}>Ready to take the next step?</div>
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.8)" }}>Your AI advisor is available 24/7 to answer questions about your relocation journey.</div>
        </div>
        <Link href="/chat" style={{ padding: "12px 24px", background: "rgba(255,255,255,0.15)", border: "1px solid rgba(255,255,255,0.3)", borderRadius: 12, color: "white", fontSize: 14, fontWeight: 600, textDecoration: "none", backdropFilter: "blur(8px)", whiteSpace: "nowrap" }}>
          Chat with AI →
        </Link>
      </div>
    </div>
  )
}
