"use client"

import { useState, useRef, useEffect } from "react"
import { chatMessages, demoUser } from "@/lib/mock-data"

type Message = {
  id: number
  role: "user" | "assistant"
  content: string
  timestamp: string
}

const SUGGESTED_QUESTIONS = [
  "What's the best country for AI/ML with my profile?",
  "How do I get the DAAD scholarship?",
  "What is the APS certificate process?",
  "Compare Germany vs Canada for CS students",
  "What IELTS score do I need for TUM?",
  "How much money do I need for a German blocked account?",
]

const AI_RESPONSES: Record<string, string> = {
  "aps": "The **APS (Akademische Prüfstelle)** is mandatory for Indian students applying to German universities.\n\n**Process:**\n1. Book appointment at APS India (Delhi/Chennai/Mumbai)\n2. Submit all your academic documents\n3. Attend the interview (~30 min)\n4. Wait 4-8 weeks for the APS certificate\n\n**Cost:** ~₹15,000\n**Timeline:** Book 4-5 months before your target visa date.\n\nHere's the APS India website: aps-india.de",
  "daad": "**DAAD Scholarship** — Your 88% match score means you're a strong candidate!\n\n**Key details:**\n- Amount: €934/month + travel + health insurance\n- Duration: Full master's program (2 years)\n- Deadline: **October 15, 2025** ⚠️\n\n**You need:**\n✓ CGPA above 3.0 (you have 8.4 — excellent)\n✓ Under 32 years old\n✓ Recommendation from a professor\n✓ Research proposal / SOP\n\n**Your next step:** Start the DAAD online portal account today at daad.de",
  "germany": "Based on your profile (CGPA 8.4, IELTS 7.5, AI/ML focus), **Germany is your #1 pick**. Here's why:\n\n**Why Germany beats the alternatives for you:**\n| | Germany | Canada |\n|---|---|---|\n| Tuition | ✅ FREE | ❌ CAD $30K/yr |\n| AI Market | Strong | Very Strong |\n| PR | 5 years | 3 years |\n| Budget fit | ✅ Perfect | ❌ Tight |\n\n**Top programs for AI/ML:**\n- TUM Data Science (your #1 match)\n- RWTH Aachen AI\n- Saarland University CS\n\nWould you like me to generate your TUM application checklist?",
  "default": "That's a great question! Based on your profile analysis:\n\n**Your Key Stats:**\n- CGPA: 8.4/10 (Top 20% applicant)\n- IELTS: 7.5 (Excellent)\n- Field: AI/ML (High demand everywhere)\n- Budget: ₹25L/yr\n\nBased on this, **Germany remains your strongest option** — free tuition saves you ~₹18L over 2 years vs Canada.\n\nWould you like me to dig deeper into any specific aspect? I can help with:\n- University shortlisting\n- Scholarship eligibility\n- Visa step-by-step\n- Cost comparison across cities\n- SOP structure advice",
}

function getAIResponse(input: string): string {
  const lower = input.toLowerCase()
  if (lower.includes("aps") || lower.includes("certificate")) return AI_RESPONSES["aps"]
  if (lower.includes("daad") || lower.includes("scholarship")) return AI_RESPONSES["daad"]
  if (lower.includes("germany") || lower.includes("canada") || lower.includes("compare") || lower.includes("best country")) return AI_RESPONSES["germany"]
  return AI_RESPONSES["default"]
}

function formatMessage(text: string) {
  const lines = text.split("\n")
  return lines.map((line, i) => {
    if (line.startsWith("**") && line.endsWith("**")) {
      return <div key={i} style={{ fontWeight: 700, color: "#FAFAFA", marginBottom: 6 }}>{line.replace(/\*\*/g, "")}</div>
    }
    if (line.startsWith("- ") || line.startsWith("✓ ") || line.startsWith("✅ ")) {
      return <div key={i} style={{ paddingLeft: 12, marginBottom: 4 }}>{line}</div>
    }
    if (line.startsWith("| ")) {
      return <div key={i} style={{ fontFamily: "monospace", fontSize: 12, color: "#A1A1AA", paddingLeft: 8 }}>{line}</div>
    }
    if (line.trim() === "") return <br key={i} />
    const boldified = line.replace(/\*\*([^*]+)\*\*/g, (_: string, m: string) => `<strong style="color:#FAFAFA">${m}</strong>`)
    return <div key={i} dangerouslySetInnerHTML={{ __html: boldified }} style={{ marginBottom: 4 }} />
  })
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>(chatMessages)
  const [input, setInput] = useState("")
  const [typing, setTyping] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, typing])

  const sendMessage = async (text?: string) => {
    const content = (text || input).trim()
    if (!content) return

    const userMsg: Message = {
      id: Date.now(),
      role: "user",
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setTyping(true)

    await new Promise(r => setTimeout(r, 1200 + Math.random() * 800))

    const aiMsg: Message = {
      id: Date.now() + 1,
      role: "assistant",
      content: getAIResponse(content),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }
    setMessages(prev => [...prev, aiMsg])
    setTyping(false)
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px", borderBottom: "1px solid #27272A", background: "#111113", display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ width: 44, height: 44, borderRadius: 14, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>✨</div>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA" }}>Pathora AI Advisor</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10B981" }} />
            <span style={{ fontSize: 13, color: "#71717A" }}>Online · Knows your profile · 2.4M data points</span>
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button style={{ padding: "8px 14px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#A1A1AA", fontSize: 13, cursor: "pointer" }}>
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: 20 }}>
        {messages.map(msg => (
          <div key={msg.id} style={{ display: "flex", gap: 12, flexDirection: msg.role === "user" ? "row-reverse" : "row", alignItems: "flex-end" }}>
            {/* Avatar */}
            <div style={{ width: 32, height: 32, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "white", background: msg.role === "assistant" ? "linear-gradient(135deg,#6366F1,#8B5CF6)" : "linear-gradient(135deg,#10B981,#059669)" }}>
              {msg.role === "assistant" ? "✨" : demoUser.name[0]}
            </div>

            <div style={{ maxWidth: "72%" }}>
              <div style={{ padding: "14px 18px", borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px", background: msg.role === "user" ? "linear-gradient(135deg,#6366F1,#8B5CF6)" : "#111113", border: msg.role === "assistant" ? "1px solid #27272A" : "none" }}>
                <div style={{ fontSize: 14, color: msg.role === "user" ? "white" : "#D4D4D8", lineHeight: 1.6 }}>
                  {msg.role === "assistant" ? formatMessage(msg.content) : msg.content}
                </div>
              </div>
              <div style={{ fontSize: 11, color: "#52525B", marginTop: 4, textAlign: msg.role === "user" ? "right" : "left" }}>{msg.timestamp}</div>
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {typing && (
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
            <div style={{ width: 32, height: 32, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", color: "white" }}>✨</div>
            <div style={{ padding: "14px 18px", background: "#111113", border: "1px solid #27272A", borderRadius: "18px 18px 18px 4px" }}>
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                {[0, 1, 2].map(i => (
                  <div key={i} className="typing-dot" style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366F1", animation: `bounce-dot 1.4s ease-in-out ${i * 0.16}s infinite` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggested questions */}
      {messages.length <= chatMessages.length && (
        <div style={{ padding: "0 24px 12px", overflowX: "auto" }}>
          <div style={{ display: "flex", gap: 8, paddingBottom: 4 }}>
            {SUGGESTED_QUESTIONS.map(q => (
              <button key={q} onClick={() => sendMessage(q)}
                style={{ padding: "8px 14px", background: "#111113", border: "1px solid #27272A", borderRadius: 99, color: "#A1A1AA", fontSize: 12, cursor: "pointer", whiteSpace: "nowrap", transition: "all 0.2s" }}
                onMouseEnter={e => { (e.currentTarget).style.borderColor = "#6366F1"; (e.currentTarget).style.color = "#6366F1" }}
                onMouseLeave={e => { (e.currentTarget).style.borderColor = "#27272A"; (e.currentTarget).style.color = "#A1A1AA" }}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div style={{ padding: "16px 24px", borderTop: "1px solid #27272A", background: "#111113" }}>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
              placeholder="Ask anything about your relocation — visa, universities, scholarships, costs..."
              rows={1}
              style={{ width: "100%", padding: "14px 16px", background: "#18181B", border: "1px solid #27272A", borderRadius: 14, color: "#FAFAFA", fontSize: 14, outline: "none", resize: "none", fontFamily: "inherit", lineHeight: 1.5 }}
              onFocus={e => (e.target.style.borderColor = "#6366F1")}
              onBlur={e => (e.target.style.borderColor = "#27272A")}
            />
          </div>
          <button onClick={() => sendMessage()} disabled={!input.trim() || typing}
            style={{ width: 48, height: 48, borderRadius: 14, background: !input.trim() || typing ? "#27272A" : "linear-gradient(135deg,#6366F1,#8B5CF6)", border: "none", cursor: !input.trim() || typing ? "default" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, transition: "all 0.2s", flexShrink: 0 }}>
            ↑
          </button>
        </div>
        <div style={{ fontSize: 11, color: "#3F3F46", marginTop: 8, textAlign: "center" }}>
          AI responses are for guidance only. Always verify with official sources.
        </div>
      </div>
    </div>
  )
}
