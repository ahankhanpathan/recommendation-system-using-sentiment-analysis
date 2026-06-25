"use client"

import { useState } from "react"

const COUNTRIES_DATA = [
  { id: "germany", name: "Germany", flag: "🇩🇪", currency: "EUR", rate: 90, tuition: 1500, living: 13200, avgSalary: 58000, taxRate: 0.35, prYears: 5 },
  { id: "canada", name: "Canada", flag: "🇨🇦", currency: "CAD", rate: 60, tuition: 28000, living: 26400, avgSalary: 85000, taxRate: 0.30, prYears: 3 },
  { id: "uk", name: "United Kingdom", flag: "🇬🇧", currency: "GBP", rate: 107, tuition: 25000, living: 21600, avgSalary: 52000, taxRate: 0.32, prYears: 5 },
  { id: "australia", name: "Australia", flag: "🇦🇺", currency: "AUD", rate: 53, tuition: 35000, living: 30000, avgSalary: 90000, taxRate: 0.325, prYears: 4 },
  { id: "netherlands", name: "Netherlands", flag: "🇳🇱", currency: "EUR", rate: 90, tuition: 10000, living: 16800, avgSalary: 62000, taxRate: 0.33, prYears: 5 },
]

const MODES = ["ROI Calculator", "Cost of Living", "Salary Comparison", "Loan vs Scholarship"]

export default function CalculatorPage() {
  const [mode, setMode] = useState("ROI Calculator")
  const [country1, setCountry1] = useState("germany")
  const [country2, setCountry2] = useState("canada")
  const [programLength, setProgramLength] = useState(2)
  const [cgpa, setCgpa] = useState(8.4)
  const [partTimeHours, setPartTimeHours] = useState(10)
  const [loanAmount, setLoanAmount] = useState(15)
  const [scholarshipAmount, setScholarshipAmount] = useState(5)
  const [salaryGrowth, setSalaryGrowth] = useState(8)

  const c1 = COUNTRIES_DATA.find(c => c.id === country1)!
  const c2 = COUNTRIES_DATA.find(c => c.id === country2)!

  const calcROI = (country: typeof COUNTRIES_DATA[0]) => {
    const totalCost = (country.tuition + country.living) * programLength
    const partTimeIncome = partTimeHours * 12 * 12 * programLength * country.rate
    const netCost = Math.max(0, totalCost * country.rate - partTimeIncome)
    const yearOneSalary = country.avgSalary * country.rate * (1 - country.taxRate)
    const payback = netCost / yearOneSalary
    let tenYrWealth = 0
    for (let i = 0; i < 10; i++) {
      tenYrWealth += country.avgSalary * country.rate * (1 - country.taxRate) * Math.pow(1 + salaryGrowth / 100, i)
    }
    return { totalCost, netCost, yearOneSalary, payback, tenYrWealth }
  }

  const r1 = calcROI(c1)
  const r2 = calcROI(c2)

  const formatINR = (n: number) => {
    if (n >= 10000000) return `₹${(n / 10000000).toFixed(1)}Cr`
    if (n >= 100000) return `₹${(n / 100000).toFixed(1)}L`
    return `₹${Math.round(n).toLocaleString()}`
  }

  return (
    <div style={{ padding: "32px", maxWidth: 1100, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "#FAFAFA", marginBottom: 8 }}>🧮 Financial Calculator</h1>
        <p style={{ fontSize: 14, color: "#71717A" }}>Compare the real financial impact of studying abroad across countries</p>
      </div>

      {/* Mode tabs */}
      <div style={{ display: "flex", gap: 4, background: "#111113", border: "1px solid #27272A", borderRadius: 14, padding: 5, marginBottom: 28 }}>
        {MODES.map(m => (
          <button key={m} onClick={() => setMode(m)}
            style={{ flex: 1, padding: "10px 14px", borderRadius: 10, border: "none", background: mode === m ? "#1C1C1F" : "transparent", color: mode === m ? "#FAFAFA" : "#71717A", fontSize: 13, fontWeight: 500, cursor: "pointer" }}>
            {m}
          </button>
        ))}
      </div>

      {/* ROI Calculator */}
      {mode === "ROI Calculator" && (
        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24 }}>
          {/* Controls */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Country selectors */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>Compare Countries</h3>
              {[{ label: "Country 1", val: country1, set: setCountry1 }, { label: "Country 2", val: country2, set: setCountry2 }].map(({ label, val, set }) => (
                <div key={label} style={{ marginBottom: 12 }}>
                  <label style={{ display: "block", fontSize: 12, color: "#71717A", marginBottom: 6 }}>{label}</label>
                  <select value={val} onChange={e => set(e.target.value)}
                    style={{ width: "100%", padding: "10px 12px", background: "#18181B", border: "1px solid #27272A", borderRadius: 10, color: "#FAFAFA", fontSize: 14, outline: "none", cursor: "pointer" }}>
                    {COUNTRIES_DATA.map(c => (
                      <option key={c.id} value={c.id}>{c.flag} {c.name}</option>
                    ))}
                  </select>
                </div>
              ))}
            </div>

            {/* Sliders */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>Your Parameters</h3>
              {[
                { label: "Program Length", value: programLength, min: 1, max: 4, step: 1, unit: " years", set: setProgramLength },
                { label: "Part-Time Hours/Week", value: partTimeHours, min: 0, max: 20, step: 1, unit: " hrs", set: setPartTimeHours },
                { label: "Annual Salary Growth", value: salaryGrowth, min: 3, max: 15, step: 1, unit: "%", set: setSalaryGrowth },
              ].map(({ label, value, min, max, step, unit, set }) => (
                <div key={label} style={{ marginBottom: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <label style={{ fontSize: 13, color: "#A1A1AA" }}>{label}</label>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#6366F1" }}>{value}{unit}</span>
                  </div>
                  <input type="range" min={min} max={max} step={step} value={value}
                    onChange={e => set(parseFloat(e.target.value))}
                    style={{ width: "100%", accentColor: "#6366F1" }} />
                </div>
              ))}
            </div>
          </div>

          {/* Results */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Country cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[{ country: c1, result: r1 }, { country: c2, result: r2 }].map(({ country, result }) => (
                <div key={country.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
                  <div style={{ display: "flex", align: "center", gap: 10, marginBottom: 16 }}>
                    <span style={{ fontSize: 24 }}>{country.flag}</span>
                    <div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA" }}>{country.name}</div>
                      <div style={{ fontSize: 12, color: "#71717A" }}>{country.currency}</div>
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {[
                      { label: "Total Cost", value: formatINR(result.netCost), color: "#EF4444" },
                      { label: "Year 1 Net Income", value: formatINR(result.yearOneSalary), color: "#10B981" },
                      { label: "Payback Period", value: `${result.payback.toFixed(1)} yrs`, color: "#F59E0B" },
                      { label: "10-Year Wealth", value: formatINR(result.tenYrWealth), color: "#6366F1" },
                    ].map(({ label, value, color }) => (
                      <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "10px 12px", background: "#18181B", borderRadius: 10 }}>
                        <span style={{ fontSize: 12, color: "#71717A" }}>{label}</span>
                        <span style={{ fontSize: 14, fontWeight: 700, color }}>{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Winner */}
            <div style={{ background: "linear-gradient(135deg,#6366F110,#10B98110)", border: "1px solid #6366F130", borderRadius: 16, padding: "20px 24px" }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "#FAFAFA", marginBottom: 10 }}>🏆 AI Verdict</h3>
              <div style={{ fontSize: 14, color: "#A1A1AA", lineHeight: 1.7 }}>
                {r1.tenYrWealth > r2.tenYrWealth
                  ? `${c1.flag} <strong style="color:#FAFAFA">${c1.name}</strong> wins financially — ${formatINR(r1.tenYrWealth - r2.tenYrWealth)} more wealth over 10 years. ${c1.tuition < c2.tuition ? "Lower tuition" : "Higher salaries"} make the difference.`
                  : `${c2.flag} <strong style="color:#FAFAFA">${c2.name}</strong> wins financially — ${formatINR(r2.tenYrWealth - r1.tenYrWealth)} more wealth over 10 years.`
                }
              </div>
              <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#6366F1" }}>{formatINR(Math.abs(r1.tenYrWealth - r2.tenYrWealth))}</div>
                  <div style={{ fontSize: 11, color: "#71717A" }}>Difference in 10yr wealth</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#10B981" }}>
                    {r1.payback < r2.payback ? c1.flag : c2.flag} {Math.min(r1.payback, r2.payback).toFixed(1)} yrs
                  </div>
                  <div style={{ fontSize: 11, color: "#71717A" }}>Faster investment payback</div>
                </div>
              </div>
            </div>

            {/* 10-year wealth chart */}
            <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px 24px" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>📈 Wealth Accumulation (10 Years)</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[c1, c2].map((country, ci) => {
                  const result = ci === 0 ? r1 : r2
                  const maxWealth = Math.max(r1.tenYrWealth, r2.tenYrWealth)
                  const pct = (result.tenYrWealth / maxWealth) * 100
                  return (
                    <div key={country.id}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 13, color: "#A1A1AA" }}>{country.flag} {country.name}</span>
                        <span style={{ fontSize: 14, fontWeight: 700, color: ci === 0 ? "#6366F1" : "#10B981" }}>{formatINR(result.tenYrWealth)}</span>
                      </div>
                      <div style={{ height: 10, background: "#27272A", borderRadius: 99 }}>
                        <div style={{ height: "100%", background: ci === 0 ? "#6366F1" : "#10B981", width: `${pct}%`, borderRadius: 99, transition: "width 0.8s ease" }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cost of Living */}
      {mode === "Cost of Living" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
            {COUNTRIES_DATA.map(c => {
              const monthly = (c.living / 12) * c.rate
              return (
                <div key={c.id} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
                  <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 14 }}>
                    <span style={{ fontSize: 24 }}>{c.flag}</span>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA" }}>{c.name}</div>
                      <div style={{ fontSize: 11, color: "#71717A" }}>Student lifestyle</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#6366F1", marginBottom: 4 }}>{formatINR(monthly)}</div>
                  <div style={{ fontSize: 12, color: "#71717A", marginBottom: 14 }}>per month (INR equivalent)</div>
                  <div style={{ height: 6, background: "#27272A", borderRadius: 99 }}>
                    <div style={{ height: "100%", background: monthly > 100000 ? "#EF4444" : monthly > 70000 ? "#F59E0B" : "#10B981", width: `${Math.min(100, (monthly / 150000) * 100)}%`, borderRadius: 99 }} />
                  </div>
                  <div style={{ fontSize: 11, color: monthly > 100000 ? "#EF4444" : monthly > 70000 ? "#F59E0B" : "#10B981", marginTop: 6 }}>
                    {monthly > 100000 ? "High cost" : monthly > 70000 ? "Medium cost" : "Budget-friendly"}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Salary Comparison */}
      {mode === "Salary Comparison" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "24px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#FAFAFA", marginBottom: 20 }}>💰 Salary Comparison — AI/ML Field</h3>
            {COUNTRIES_DATA.map(c => {
              const netSalary = c.avgSalary * c.rate * (1 - c.taxRate)
              const maxNet = Math.max(...COUNTRIES_DATA.map(x => x.avgSalary * x.rate * (1 - x.taxRate)))
              return (
                <div key={c.id} style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 14, color: "#A1A1AA" }}>{c.flag} {c.name}</span>
                    <div style={{ textAlign: "right" }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: "#10B981" }}>{formatINR(netSalary)}</span>
                      <span style={{ fontSize: 11, color: "#52525B", marginLeft: 6 }}>net/yr</span>
                    </div>
                  </div>
                  <div style={{ height: 10, background: "#27272A", borderRadius: 99 }}>
                    <div style={{ height: "100%", background: "linear-gradient(90deg,#6366F1,#10B981)", width: `${(netSalary / maxNet) * 100}%`, borderRadius: 99, transition: "width 0.8s" }} />
                  </div>
                  <div style={{ fontSize: 11, color: "#52525B", marginTop: 4 }}>
                    Gross: {c.avgSalary.toLocaleString()} {c.currency}/yr · Tax: {Math.round(c.taxRate * 100)}%
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Loan vs Scholarship */}
      {mode === "Loan vs Scholarship" && (
        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 24 }}>
          <div style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 16 }}>Parameters</h3>
            {[
              { label: "Education Loan (₹L)", value: loanAmount, min: 5, max: 50, set: setLoanAmount },
              { label: "Scholarship Value (₹L/yr)", value: scholarshipAmount, min: 0, max: 30, set: setScholarshipAmount },
              { label: "Program Length", value: programLength, min: 1, max: 4, set: setProgramLength },
            ].map(({ label, value, min, max, set }) => (
              <div key={label} style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <label style={{ fontSize: 12, color: "#A1A1AA" }}>{label}</label>
                  <span style={{ fontSize: 13, fontWeight: 700, color: "#6366F1" }}>₹{value}L</span>
                </div>
                <input type="range" min={min} max={max} value={value} onChange={e => set(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#6366F1" }} />
              </div>
            ))}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                {
                  title: "📋 With Education Loan",
                  items: [
                    { label: "Loan Amount", value: `₹${loanAmount}L`, color: "#EF4444" },
                    { label: "Interest (9%/yr)", value: `₹${(loanAmount * 9 * programLength / 100).toFixed(1)}L`, color: "#EF4444" },
                    { label: "EMI (10yr repay)", value: `₹${Math.round(loanAmount * 100000 / 120).toLocaleString()}/mo`, color: "#F59E0B" },
                    { label: "Total Payable", value: `₹${(loanAmount * 1.9).toFixed(1)}L`, color: "#EF4444" },
                  ]
                },
                {
                  title: "🏆 With Scholarship",
                  items: [
                    { label: "Scholarship Value", value: `₹${scholarshipAmount * programLength}L`, color: "#10B981" },
                    { label: "Out-of-pocket", value: `₹${Math.max(0, loanAmount - scholarshipAmount * programLength).toFixed(1)}L`, color: "#6366F1" },
                    { label: "No EMI burden", value: "✓ Debt free", color: "#10B981" },
                    { label: "Net Savings vs Loan", value: `₹${(loanAmount * 1.9 - Math.max(0, loanAmount - scholarshipAmount * programLength)).toFixed(1)}L`, color: "#10B981" },
                  ]
                }
              ].map(({ title, items }) => (
                <div key={title} style={{ background: "#111113", border: "1px solid #27272A", borderRadius: 16, padding: "20px" }}>
                  <h3 style={{ fontSize: 14, fontWeight: 700, color: "#FAFAFA", marginBottom: 14 }}>{title}</h3>
                  {items.map(({ label, value, color }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #1F1F22" }}>
                      <span style={{ fontSize: 12, color: "#71717A" }}>{label}</span>
                      <span style={{ fontSize: 13, fontWeight: 700, color }}>{value}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ background: "#10B98110", border: "1px solid #10B98130", borderRadius: 14, padding: "16px 20px" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#10B981", marginBottom: 6 }}>💡 Pathora Recommendation</div>
              <div style={{ fontSize: 13, color: "#A1A1AA" }}>
                Your DAAD scholarship eligibility (88% match) could cover ₹83K/month — applying for it first before taking any loan is strongly recommended. Apply by October 15!
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
