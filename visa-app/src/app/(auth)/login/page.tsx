"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPass, setShowPass] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSignIn = async () => {
    setLoading(true)
    await new Promise(r => setTimeout(r, 1000))
    router.push("/onboarding")
  }

  const handleDemo = () => router.push("/dashboard")

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#09090B" }}>
      {/* Left Panel */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px", position: "relative" }}>
        <div style={{ width: "100%", maxWidth: 400 }} className="animate-fade-in">
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 40 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 20, color: "white" }}>P</div>
            <div>
              <div style={{ fontWeight: 800, fontSize: 22, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Pathora</div>
              <div style={{ fontSize: 11, color: "#6366F1", fontWeight: 500 }}>AI Operating System</div>
            </div>
          </div>

          <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>Welcome back</h1>
          <p style={{ color: "#71717A", fontSize: 15, marginBottom: 32 }}>Sign in to your relocation command center</p>

          {/* Google Button */}
          <button onClick={handleSignIn} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 12, padding: "13px 20px", background: "white", border: "none", borderRadius: 12, fontSize: 15, fontWeight: 600, color: "#09090B", cursor: "pointer", marginBottom: 16, transition: "all 0.2s" }}
            onMouseEnter={e => (e.currentTarget.style.background = "#F4F4F5")}
            onMouseLeave={e => (e.currentTarget.style.background = "white")}>
            <svg width={20} height={20} viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            Continue with Google
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <div style={{ flex: 1, height: 1, background: "#27272A" }} />
            <span style={{ fontSize: 12, color: "#52525B" }}>or</span>
            <div style={{ flex: 1, height: 1, background: "#27272A" }} />
          </div>

          {/* Email */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: "block", fontSize: 13, fontWeight: 500, color: "#A1A1AA", marginBottom: 6 }}>Email</label>
            <input value={email} onChange={e => setEmail(e.target.value)} placeholder="aryan@gmail.com" style={{ width: "100%", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, padding: "12px 16px", color: "#FAFAFA", fontSize: 14, outline: "none" }} />
          </div>

          {/* Password */}
          <div style={{ marginBottom: 8 }}>
            <label style={{ display: "block", fontSize: 13, fontWeight: 500, color: "#A1A1AA", marginBottom: 6 }}>Password</label>
            <div style={{ position: "relative" }}>
              <input type={showPass ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="········" style={{ width: "100%", background: "#18181B", border: "1px solid #27272A", borderRadius: 12, padding: "12px 48px 12px 16px", color: "#FAFAFA", fontSize: 14, outline: "none" }} />
              <button onClick={() => setShowPass(!showPass)} style={{ position: "absolute", right: 14, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", color: "#71717A", cursor: "pointer", fontSize: 12 }}>{showPass ? "Hide" : "Show"}</button>
            </div>
          </div>

          <div style={{ textAlign: "right", marginBottom: 24 }}>
            <span style={{ fontSize: 13, color: "#6366F1", cursor: "pointer" }}>Forgot password?</span>
          </div>

          <button onClick={handleSignIn} disabled={loading} style={{ width: "100%", padding: "13px", background: loading ? "#4338CA" : "#6366F1", border: "none", borderRadius: 12, color: "white", fontSize: 15, fontWeight: 600, cursor: loading ? "wait" : "pointer", transition: "all 0.2s", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
            {loading ? "Signing in..." : "Sign In →"}
          </button>

          <p style={{ textAlign: "center", fontSize: 14, color: "#71717A", marginTop: 20 }}>
            Don't have an account?{" "}
            <Link href="/onboarding" style={{ color: "#6366F1", textDecoration: "none", fontWeight: 500 }}>Get started →</Link>
          </p>

          {/* Demo Mode */}
          <div style={{ marginTop: 32, padding: "16px", background: "#18181B", border: "1px solid #6366F130", borderRadius: 12, textAlign: "center" }}>
            <div style={{ fontSize: 13, color: "#A1A1AA", marginBottom: 10 }}>✨ Try the full experience instantly</div>
            <button onClick={handleDemo} style={{ background: "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", borderRadius: 10, padding: "10px 24px", color: "white", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
              Enter Demo Mode →
            </button>
          </div>

          <div style={{ marginTop: 24, display: "flex", alignItems: "center", gap: 8 }}>
            {["A","R","P","S","V"].map((l, i) => (
              <div key={i} style={{ width: 28, height: 28, borderRadius: "50%", background: `hsl(${220+i*20},70%,50%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "white", marginLeft: i > 0 ? -8 : 0, border: "2px solid #09090B" }}>{l}</div>
            ))}
            <span style={{ fontSize: 13, color: "#71717A", marginLeft: 8 }}>Joined by <strong style={{ color: "#A1A1AA" }}>12,847</strong> students</span>
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", background: "#0D0D10" }}>
        {[
          { size: 400, x: "60%", y: "30%", color: "#6366F1", delay: "0s" },
          { size: 300, x: "30%", y: "65%", color: "#8B5CF6", delay: "2s" },
          { size: 250, x: "75%", y: "70%", color: "#A78BFA", delay: "4s" },
        ].map((orb, i) => (
          <div key={i} style={{
            position: "absolute", width: orb.size, height: orb.size, borderRadius: "50%",
            background: orb.color, filter: "blur(80px)", opacity: 0.15,
            left: orb.x, top: orb.y, transform: "translate(-50%,-50%)",
            animation: `float 8s ease-in-out infinite`, animationDelay: orb.delay,
          }} />
        ))}

        <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 20 }}>
          {[
            { emoji: "🇩🇪", title: "Germany Match", value: "94%", sub: "Your #1 recommendation", color: "#6366F1", delay: "0s" },
            { emoji: "💰", title: "10-Year Wealth", value: "₹4.2 Cr", sub: "Projected in Germany", color: "#10B981", delay: "1s" },
            { emoji: "🎓", title: "DAAD Scholarship", value: "€934/mo", sub: "You're eligible!", color: "#F59E0B", delay: "2s" },
            { emoji: "⏱️", title: "Time to Offer", value: "87 days", sub: "Average for your profile", color: "#8B5CF6", delay: "0.5s" },
          ].map((card, i) => (
            <div key={i} className="glass animate-slide-right" style={{
              padding: "16px 20px", borderRadius: 16, display: "flex", alignItems: "center", gap: 14, minWidth: 280,
              animation: "slideInRight 0.6s ease forwards", animationDelay: card.delay, opacity: 0,
              boxShadow: `0 0 0 1px ${card.color}20, 0 8px 32px rgba(0,0,0,0.3)`,
              transform: i % 2 === 0 ? "rotate(-1deg)" : "rotate(1deg)",
            }}>
              <div style={{ fontSize: 32 }}>{card.emoji}</div>
              <div>
                <div style={{ fontSize: 12, color: "#71717A", fontWeight: 500 }}>{card.title}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: card.color }}>{card.value}</div>
                <div style={{ fontSize: 12, color: "#A1A1AA" }}>{card.sub}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
