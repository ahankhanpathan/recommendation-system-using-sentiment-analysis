"use client"

import { usePathname } from "next/navigation"
import Link from "next/link"

const navLinks = [
  { href: "/dashboard", label: "Dashboard", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" },
  { href: "/report", label: "My Report", icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" },
  { href: "/countries", label: "Countries", icon: "M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064" },
  { href: "/scholarships", label: "Scholarships", icon: "M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" },
  { href: "/visa/germany", label: "Visa Guide", icon: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" },
  { href: "/chat", label: "AI Chat", icon: "M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z", badge: "AI" },
  { href: "/calculator", label: "Calculator", icon: "M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" },
  { href: "/referral", label: "Referral", icon: "M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7" },
  { href: "/admin", label: "Admin", icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z" },
]

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#09090B" }}>
      <aside style={{
        width: 256,
        minHeight: "100vh",
        background: "#111113",
        borderRight: "1px solid #27272A",
        display: "flex",
        flexDirection: "column",
        position: "fixed",
        top: 0,
        left: 0,
        bottom: 0,
        zIndex: 40,
        overflowY: "auto",
      }}>
        <div style={{ padding: "24px 20px 20px", borderBottom: "1px solid #27272A" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg, #6366F1, #8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 16, color: "white" }}>P</div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 18, background: "linear-gradient(135deg,#6366F1,#8B5CF6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Pathora</div>
              <div style={{ fontSize: 10, color: "#6366F1", fontWeight: 500, letterSpacing: "0.05em" }}>AI OS</div>
            </div>
          </div>
        </div>

        <nav style={{ flex: 1, padding: "16px 12px", display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ fontSize: 10, color: "#52525B", fontWeight: 600, letterSpacing: "0.1em", padding: "8px 12px 4px", textTransform: "uppercase" }}>Main</div>
          {navLinks.slice(0, 4).map(link => (
            <NavItem key={link.href} link={link} active={pathname === link.href} />
          ))}
          <div style={{ fontSize: 10, color: "#52525B", fontWeight: 600, letterSpacing: "0.1em", padding: "16px 12px 4px", textTransform: "uppercase" }}>Tools</div>
          {navLinks.slice(4, 8).map(link => (
            <NavItem key={link.href} link={link} active={pathname === link.href || pathname.startsWith(link.href.split("/")[1] ? "/" + link.href.split("/")[1] : link.href)} />
          ))}
          <div style={{ fontSize: 10, color: "#52525B", fontWeight: 600, letterSpacing: "0.1em", padding: "16px 12px 4px", textTransform: "uppercase" }}>Settings</div>
          {navLinks.slice(8).map(link => (
            <NavItem key={link.href} link={link} active={pathname === link.href} />
          ))}
        </nav>

        <div style={{ padding: "16px", borderTop: "1px solid #27272A" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 12, background: "#18181B" }}>
            <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg,#6366F1,#8B5CF6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, color: "white", flexShrink: 0 }}>A</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#FAFAFA", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>Aryan Mehta</div>
              <div style={{ fontSize: 11, color: "#71717A", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>aryan@gmail.com</div>
            </div>
          </div>
        </div>
      </aside>

      <main style={{ marginLeft: 256, flex: 1, minHeight: "100vh" }}>
        {children}
      </main>
    </div>
  )
}

function NavItem({ link, active }: { link: typeof navLinks[0]; active: boolean }) {
  return (
    <Link href={link.href} style={{
      display: "flex", alignItems: "center", gap: 10, padding: "9px 12px",
      borderRadius: 10, textDecoration: "none",
      color: active ? "#FAFAFA" : "#71717A",
      background: active ? "#1C1C1F" : "transparent",
      fontSize: 14, fontWeight: 500,
      transition: "all 0.15s ease",
    }}
    onMouseEnter={e => { if (!active) { (e.currentTarget as HTMLElement).style.color = "#A1A1AA"; (e.currentTarget as HTMLElement).style.background = "#18181B"; } }}
    onMouseLeave={e => { if (!active) { (e.currentTarget as HTMLElement).style.color = "#71717A"; (e.currentTarget as HTMLElement).style.background = "transparent"; } }}>
      <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
        <path d={link.icon} />
      </svg>
      <span style={{ flex: 1 }}>{link.label}</span>
      {link.badge && (
        <span style={{ fontSize: 9, fontWeight: 700, color: "#6366F1", background: "#6366F115", padding: "2px 6px", borderRadius: 99, letterSpacing: "0.05em" }}>{link.badge}</span>
      )}
    </Link>
  )
}
