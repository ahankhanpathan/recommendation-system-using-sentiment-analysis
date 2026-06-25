"use client"

import { useState } from "react"
import { referralData, demoUser } from "@/lib/mock-data"

const TIER_CONFIG = {
  Bronze: { color: "#CD7F32", bg: "#CD7F3215", min: 0, max: 3 },
  Silver: { color: "#C0C0C0", bg: "#C0C0C015", min: 3, max: 7 },
  Gold: { color: "#F59E0B", bg: "#F59E0B15", min: 7, max: 15 },
  Platinum: { color: "#8B5CF6", bg: "#8B5CF615", min: 15, max: 999 },
}

const HOW_IT_WORKS = [
  { step: "1", title: "Share Your Code", desc: "Share your unique referral code or link with friends planning to study abroad.", icon: "🔗" },
  { step: "2", title: "Friend Signs Up", desc: "When your friend creates a Pathora account using your code, we track the referral.", icon: "👤" },
  { step: "3", title: "They Generate Report", desc: "Once your friend generates their AI Relocation Report, the referral is confirmed.", icon: "📊" },
  { step: "4", title: "You Earn Rewards", desc: "₹2,000 credited to your Pathora wallet instantly. Use for premium features!", icon: "💰" },
]

export default function ReferralPage() {
  const [copied, setCopied] = useState(false)
  const [copiedLink, setCopiedLink] = useState(false)
  const tier = referralData.tier as keyof typeof TIER_CONFIG
  const tierConfig = TIER_CONFIG[tier]

  const copyCode = () => {
    navigator.clipboard.writeText(referralData.code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const copyLink = () => {
    navigator.clipboard.writeText(`https://pathora.ai/join?ref=${referralData.code}`)
    setCopiedLink(true)
    setTimeout(() => setCopiedLink(false), 2000)
  }

  const progressToNext = (referralData.totalReferrals - TIER_CONFIG[tier].min) / (TIER_CONFIG[tier].max - TIER_CONFIG[tier].min)

  return (
    <div style={{ padding: "32px", maxWidth: 900, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>🎁 Referral Program</h1>
        <p style={{ fontSize: 14, color: "#71717A" }}>Earn ₹2,000 for every friend who joins Pathora</p>
      </div>

      {/* Hero card */}
      <div style={{ background: "linear-gradient(135deg,#6366F1,#8B5CF6)", borderRadius: 24, padding: "40px", marginBottom: 24, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -40, right: -40, width: 200, height: 200, borderRadius: "50%", background: "rgba(255,255,255,0.05)" }} />
        <div style={{ position: "absolute", bottom: -60, left: -30, width: 250, height: 250, borderRadius: "50%", background: "rgba(255,255,255,0.03)" }} />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ fontSize: 13, color: "rgba(255,255,255,0.7)", marginBottom: 8, fontWeight: 500 }}>Your Total Earnings</div>
          <div style={{ fontSize: 48, fontWeight: 800, color: "white", marginBottom: 4 }}>{referralData.totalEarnings}</div>
          <div style={{ fontSize: 14, color: "rgba(255,255,255,0.8)" }}>from {referralData.totalReferrals} referrals · {referralData.successfulReferrals} confirmed</div>

          <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
            {/* Code */}
            <div style={{ display: "flex", flex: 1, alignItems: "center", background: "rgba(255,255,255,0.12)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.2)", overflow: "hidden" }}>
              <span style={{ padding: "12px 16px", fontSize: 16, fontWeight: 800, color: "white", letterSpacing: "0.1em", flex: 1 }}>{referralData.code}</span>
              <button onClick={copyCode} style={{ padding: "12px 18px", background: "rgba(255,255,255,0.2)", border: "none", cursor: "pointer", color: "white", fontSize: 13, fontWeight: 600 }}>
                {copied ? "✓ Copied!" : "Copy Code"}
              </button>
            </div>

            <button onClick={copyLink} style={{ padding: "12px 20px", background: "rgba(255,255,255,0.15)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 12, color: "white", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
              {copiedLink ? "✓ Copied!" : "🔗 Copy Link"}
            </button>
          </div>
        </div>
      </div>

      {/* Stats + Tier */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
        {/* Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {[
            { label: "Total Referrals", value: referralData.totalReferrals, color: "#6366F1" },
            { label: "Confirmed", value: referralData.successfulReferrals, color: "#10B981" },
            { label: "Pending", value: referralData.pendingReferrals, color: "#F59E0B" },
            { label: "Per Referral", value: referralData.rewardPerReferral, color: "#8B5CF6" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: "18px" }}>
              <div style={{ fontSize: 24, fontWeight: 800, color, marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 12, color: "#71717A" }}>{label}</div>
            </div>
          ))}
        </div>

        {/* Tier progress */}
        <div style={{ background: "#111113", border: `1px solid ${tierConfig.color}30`, borderRadius: 16, padding: "24px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 13, color: "#71717A", marginBottom: 4 }}>Current Tier</div>
              <div style={{ fontSize: 24, fontWeight: 800, color: tierConfig.color }}>{tier} ⭐</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 13, color: "#71717A", marginBottom: 4 }}>Next Tier</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: TIER_CONFIG["Platinum"].color }}>Platinum</div>
            </div>
          </div>

          <div style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: "#71717A" }}>{referralData.totalReferrals} referrals</span>
              <span style={{ fontSize: 12, color: "#71717A" }}>{TIER_CONFIG[tier].max} needed</span>
            </div>
            <div style={{ height: 8, background: "#27272A", borderRadius: 99 }}>
              <div style={{ height: "100%", background: tierConfig.color, width: `${Math.min(100, progressToNext * 100)}%`, borderRadius: 99, transition: "width 0.8s" }} />
            </div>
          </div>
          <div style={{ fontSize: 13, color: "#A1A1AA" }}>
            {referralData.referralsToNextTier} more referrals to reach Platinum
          </div>

          <div style={{ marginTop: 16, padding: "12px", background: "#18181B", borderRadius: 10 }}>
            <div style={{ fontSize: 12, color: "#71717A" }}>Platinum Benefits:</div>
            <div style={{ fontSize: 13, color: "#8B5CF6", marginTop: 4 }}>₹3,000/referral · Priority support · Exclusive badge</div>
          </div>
        </div>
      </div>

      {/* Leaderboard */}
      <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px", marginBottom: 24 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>🏆 Top Referrers This Month</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {referralData.leaderboard.map(entry => (
            <div key={entry.rank} style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 16px", background: entry.isYou ? "#6366F110" : "#18181B", borderRadius: 12, border: `1px solid ${entry.isYou ? "#6366F140" : "transparent"}` }}>
              <div style={{ width: 36, height: 36, borderRadius: 10, background: entry.rank <= 3 ? `${["#FFD700","#C0C0C0","#CD7F32"][entry.rank - 1]}20` : "#27272A", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>
                {entry.badge}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: entry.isYou ? "#6366F1" : "#FAFAFA" }}>
                  {entry.name} {entry.isYou && "(You)"}
                </div>
                <div style={{ fontSize: 12, color: "#71717A" }}>{entry.referrals} referrals</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#10B981" }}>{entry.earnings}</div>
                <div style={{ fontSize: 11, color: "#52525B" }}>earned</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px", marginBottom: 24 }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>📖 How It Works</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16 }}>
          {HOW_IT_WORKS.map(({ step, title, desc, icon }) => (
            <div key={step} style={{ textAlign: "center" }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "#18181B", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, margin: "0 auto 12px" }}>{icon}</div>
              <div style={{ fontSize: 11, color: "#6366F1", fontWeight: 700, marginBottom: 6 }}>STEP {step}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#FAFAFA", marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 12, color: "#71717A", lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Share options */}
      <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📱 Share via</h3>
        <div style={{ display: "flex", gap: 10 }}>
          {[
            { label: "WhatsApp", color: "#25D366", emoji: "💬" },
            { label: "LinkedIn", color: "#0A66C2", emoji: "🔗" },
            { label: "Twitter/X", color: "#1DA1F2", emoji: "🐦" },
            { label: "Email", color: "#EA4335", emoji: "📧" },
          ].map(({ label, color, emoji }) => (
            <button key={label}
              style={{ flex: 1, padding: "12px", background: `${color}20`, border: `1px solid ${color}40`, borderRadius: 12, color, fontSize: 13, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
              {emoji} {label}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 14, padding: "14px", background: "#18181B", borderRadius: 12, fontSize: 13, color: "#A1A1AA", lineHeight: 1.6 }}>
          📝 <strong style={{ color: "#FAFAFA" }}>Suggested message:</strong> "Hey! I'm using Pathora to plan my MS abroad — it gave me a personalized AI report for Germany. Use code {referralData.code} to get started free! pathora.ai/join?ref={referralData.code}"
        </div>
      </div>
    </div>
  )
}
