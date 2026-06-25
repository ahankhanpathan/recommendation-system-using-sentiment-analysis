"use client"

import { useState } from "react"
import { adminStats } from "@/lib/mock-data"

const TABS = ["Overview", "Users", "Reports", "Countries", "Settings"]

export default function AdminPage() {
  const [tab, setTab] = useState("Overview")
  const [timeframe, setTimeframe] = useState("30d")

  const maxUsers = Math.max(...adminStats.userGrowth.map(d => d.users))

  return (
    <div style={{ padding: "32px", maxWidth: 1200, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
        <div>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "4px 12px", background: "#EF444415", border: "1px solid #EF444430", borderRadius: 99, marginBottom: 8 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#EF4444" }} />
            <span style={{ fontSize: 11, color: "#EF4444", fontWeight: 600 }}>ADMIN PANEL</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA" }}>Pathora Command Center</h1>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["7d", "30d", "90d"].map(tf => (
            <button key={tf} onClick={() => setTimeframe(tf)}
              style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${timeframe === tf ? "#6366F1" : "#27272A"}`, background: timeframe === tf ? "#6366F120" : "transparent", color: timeframe === tf ? "#6366F1" : "#71717A", fontSize: 13, cursor: "pointer" }}>
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
        {[
          { label: "Total Users", value: adminStats.totalUsers.toLocaleString(), change: `+${adminStats.monthlyGrowth}%`, color: "#6366F1", icon: "👥" },
          { label: "Active Users", value: adminStats.activeUsers.toLocaleString(), change: "+8.2%", color: "#10B981", icon: "⚡" },
          { label: "Reports Generated", value: adminStats.reportsGenerated.toLocaleString(), change: "+15.4%", color: "#F59E0B", icon: "📊" },
          { label: "NPS Score", value: adminStats.npsScore, change: "+3 pts", color: "#8B5CF6", icon: "⭐" },
        ].map(({ label, value, change, color, icon }) => (
          <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div style={{ width: 40, height: 40, borderRadius: 12, background: `${color}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>{icon}</div>
              <span style={{ fontSize: 12, color: "#10B981", background: "#10B98115", padding: "3px 8px", borderRadius: 99, fontWeight: 600 }}>{change}</span>
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color, marginBottom: 4 }}>{value}</div>
            <div style={{ fontSize: 13, color: "#71717A" }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Secondary metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
        {[
          { label: "Avg Session", value: adminStats.avgSessionTime, color: "#A1A1AA" },
          { label: "Conversion Rate", value: adminStats.conversionRate, color: "#10B981" },
          { label: "Countries", value: adminStats.countriesAnalyzed, color: "#6366F1" },
          { label: "Scholarships Listed", value: adminStats.scholarshipsListed, color: "#F59E0B" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "#71717A" }}>{label}</span>
            <span style={{ fontSize: 18, fontWeight: 700, color }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 12, padding: 4, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ flex: 1, padding: "9px", borderRadius: 8, border: "none", background: tab === t ? "#1C1C1F" : "transparent", color: tab === t ? "#FAFAFA" : "#71717A", fontSize: 14, fontWeight: 500, cursor: "pointer" }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab === "Overview" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20 }}>
            {/* User growth chart */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>📈 User Growth</h3>
                <span style={{ fontSize: 13, color: "#10B981" }}>+{adminStats.monthlyGrowth}% MoM</span>
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 150 }}>
                {adminStats.userGrowth.map(d => (
                  <div key={d.month} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
                    <div style={{ fontSize: 11, color: "#71717A" }}>{d.users > 1000 ? `${(d.users / 1000).toFixed(0)}K` : d.users}</div>
                    <div style={{ width: "100%", background: "#6366F1", borderRadius: "4px 4px 0 0", transition: "height 0.8s", height: `${(d.users / maxUsers) * 120}px` }} />
                    <div style={{ fontSize: 11, color: "#52525B" }}>{d.month}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Country interest */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>🌍 Country Interest</h3>
              {adminStats.topCountries.map(({ country, interest, flag }) => (
                <div key={country} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: "#A1A1AA" }}>{flag} {country}</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{interest}%</span>
                  </div>
                  <div style={{ height: 6, background: "#27272A", borderRadius: 99 }}>
                    <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#8B5CF6)", width: `${interest}%`, borderRadius: 99 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Health metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
            {[
              { title: "API Health", status: "99.9% uptime", color: "#10B981", icon: "🟢" },
              { title: "AI Response Time", status: "1.2s avg", color: "#6366F1", icon: "⚡" },
              { title: "Error Rate", status: "0.03%", color: "#10B981", icon: "🟢" },
            ].map(({ title, status, color, icon }) => (
              <div key={title} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "16px 20px", display: "flex", gap: 12, alignItems: "center" }}>
                <span style={{ fontSize: 20 }}>{icon}</span>
                <div>
                  <div style={{ fontSize: 13, color: "#71717A" }}>{title}</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color }}>{status}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Users */}
      {tab === "Users" && (
        <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, overflow: "hidden" }}>
          <div style={{ padding: "16px 24px", borderBottom: "1px solid #27272A", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>Recent Users</h3>
            <span style={{ fontSize: 13, color: "#71717A" }}>Last 30 minutes</span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#18181B" }}>
                {["User", "Email", "Interest", "Status", "Joined"].map(h => (
                  <th key={h} style={{ padding: "12px 20px", textAlign: "left", fontSize: 11, color: "#52525B", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {adminStats.recentUsers.map((user, i) => (
                <tr key={i} style={{ borderTop: "1px solid #1F1F22" }}
                  onMouseEnter={e => (e.currentTarget.style.background = "#18181B")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                  <td style={{ padding: "14px 20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 32, height: 32, borderRadius: "50%", background: `hsl(${200 + i * 30},60%,50%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: "white" }}>
                        {user.name[0]}
                      </div>
                      <span style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA" }}>{user.name}</span>
                    </div>
                  </td>
                  <td style={{ padding: "14px 20px", fontSize: 13, color: "#71717A" }}>{user.email}</td>
                  <td style={{ padding: "14px 20px" }}>
                    <span style={{ fontSize: 13, color: "#A1A1AA" }}>{user.country}</span>
                  </td>
                  <td style={{ padding: "14px 20px" }}>
                    <span style={{ fontSize: 12, padding: "3px 10px", borderRadius: 99, fontWeight: 600, background: user.status === "Report Ready" ? "#10B98120" : "#6366F115", color: user.status === "Report Ready" ? "#10B981" : "#6366F1" }}>
                      {user.status}
                    </span>
                  </td>
                  <td style={{ padding: "14px 20px", fontSize: 13, color: "#52525B" }}>{user.joined}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Reports */}
      {tab === "Reports" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
            {[
              { label: "Reports Today", value: "247", trend: "+12%", color: "#6366F1" },
              { label: "Avg Generation Time", value: "38.2s", trend: "-3s", color: "#10B981" },
              { label: "Download Rate", value: "67%", trend: "+4%", color: "#F59E0B" },
            ].map(({ label, value, trend, color }) => (
              <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "20px" }}>
                <div style={{ fontSize: 26, fontWeight: 700, color, marginBottom: 4 }}>{value}</div>
                <div style={{ fontSize: 13, color: "#71717A" }}>{label}</div>
                <div style={{ fontSize: 12, color: "#10B981", marginTop: 6 }}>{trend} vs last week</div>
              </div>
            ))}
          </div>

          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>Report Generation Funnel</h3>
            {[
              { stage: "Visited Questionnaire", users: 12847, pct: 100 },
              { stage: "Completed Questionnaire", users: 9240, pct: 72 },
              { stage: "Reached Analyzing Screen", users: 8800, pct: 69 },
              { stage: "Report Generated", users: 7600, pct: 59 },
              { stage: "Report Downloaded", users: 5100, pct: 40 },
            ].map(({ stage, users, pct }, i) => (
              <div key={stage} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 13, color: "#A1A1AA" }}>{stage}</span>
                  <div style={{ display: "flex", gap: 12 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA" }}>{users.toLocaleString()}</span>
                    <span style={{ fontSize: 13, color: "#6366F1", width: 40, textAlign: "right" }}>{pct}%</span>
                  </div>
                </div>
                <div style={{ height: 6, background: "#27272A", borderRadius: 99 }}>
                  <div style={{ height: "100%", background: `hsl(${240 - i * 15},70%,65%)`, width: `${pct}%`, borderRadius: 99 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Countries */}
      {tab === "Countries" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
          {adminStats.topCountries.map(({ country, interest, flag }) => (
            <div key={country} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                <span style={{ fontSize: 32 }}>{flag}</span>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>{country}</div>
                  <div style={{ fontSize: 13, color: "#6366F1", fontWeight: 600 }}>{interest}% user interest</div>
                </div>
              </div>
              <div style={{ height: 8, background: "#27272A", borderRadius: 99, marginBottom: 12 }}>
                <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#8B5CF6)", width: `${interest}%`, borderRadius: 99 }} />
              </div>
              <div style={{ fontSize: 12, color: "#71717A" }}>
                ~{Math.round(adminStats.reportsGenerated * interest / 100).toLocaleString()} reports generated
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Settings */}
      {tab === "Settings" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {[
            { section: "Platform", settings: [
              { label: "AI Report Generation", enabled: true },
              { label: "User Registration", enabled: true },
              { label: "Maintenance Mode", enabled: false },
              { label: "Beta Features", enabled: false },
            ]},
            { section: "Notifications", settings: [
              { label: "Email Alerts", enabled: true },
              { label: "Slack Integration", enabled: true },
              { label: "Weekly Reports", enabled: true },
              { label: "Real-time Monitoring", enabled: false },
            ]},
          ].map(({ section, settings }) => (
            <div key={section} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>{section} Settings</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {settings.map(({ label, enabled }) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 14px", background: "#18181B", borderRadius: 10 }}>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{label}</span>
                    <div style={{ width: 44, height: 24, borderRadius: 99, background: enabled ? "#6366F1" : "#27272A", position: "relative", cursor: "pointer", transition: "background 0.2s" }}>
                      <div style={{ position: "absolute", width: 18, height: 18, borderRadius: "50%", background: "white", top: 3, left: enabled ? 23 : 3, transition: "left 0.2s" }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
