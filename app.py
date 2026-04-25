"""
app.py
Personal Cybersecurity Guardian Agent — Streamlit Frontend

Calls the FastAPI backend at BACKEND_URL via HTTP.
Professional, clean UI with risk gauge, evidence breakdown,
contextual intervention panel, and source attribution.

Usage:
    # Start backend first
    python api.py

    # Then start frontend (separate terminal)
    streamlit run app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "Cybersecurity Guardian",
    page_icon  = "🛡",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #0f1923;
  }

  .stApp {
    background: #f7f8fa;
  }

  /* ── Header ── */
  .guardian-header {
    background: #0f1923;
    color: #ffffff;
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
  }
  .guardian-header h1 {
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
    color: #ffffff;
  }
  .guardian-header p {
    font-size: 0.88rem;
    color: #8a9bb0;
    margin: 0;
    font-weight: 300;
  }

  /* ── Input area ── */
  .stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    border: 1.5px solid #dde3ec !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    color: #0f1923 !important;
    padding: 0.9rem 1rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: #1a56db !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    background: #1648c4 !important;
  }

  /* ── Verdict badge ── */
  .verdict-card {
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
  }
  .verdict-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.4rem;
  }
  .verdict-value {
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .verdict-confidence {
    font-size: 0.85rem;
    margin-top: 0.3rem;
    opacity: 0.75;
  }

  /* Verdict colors */
  .verdict-MALICIOUS {
    background: #fff1f1;
    border: 1.5px solid #f5c2c2;
    color: #8b1a1a;
  }
  .verdict-VULNERABLE {
    background: #fff8ed;
    border: 1.5px solid #f5d9a0;
    color: #7a4a00;
  }
  .verdict-UNCERTAIN {
    background: #fffbeb;
    border: 1.5px solid #e9d97c;
    color: #6b5a00;
  }
  .verdict-SAFE {
    background: #f0faf3;
    border: 1.5px solid #a8d9b5;
    color: #0f5c2a;
  }
  .verdict-INSUFFICIENT_EVIDENCE {
    background: #f4f6f9;
    border: 1.5px solid #c8d0dc;
    color: #3a4a5a;
  }
  .verdict-CLARIFY {
    background: #f0f5ff;
    border: 1.5px solid #b8ccf5;
    color: #1a3a7a;
  }

  /* ── Panel cards ── */
  .panel-card {
    background: #ffffff;
    border: 1px solid #e4e9f0;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
  }
  .panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7a8d;
    margin-bottom: 0.9rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #eef0f4;
  }

  /* ── Evidence items ── */
  .evidence-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.45rem 0;
    font-size: 0.875rem;
    color: #2a3545;
    border-bottom: 1px solid #f4f5f7;
  }
  .evidence-item:last-child { border-bottom: none; }
  .evidence-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #1a56db;
    margin-top: 0.45rem;
    flex-shrink: 0;
  }

  /* ── CVE table ── */
  .cve-row {
    display: grid;
    grid-template-columns: 1fr 80px 90px 1fr;
    gap: 0.5rem;
    padding: 0.55rem 0;
    font-size: 0.82rem;
    border-bottom: 1px solid #f4f5f7;
    align-items: center;
  }
  .cve-row:last-child { border-bottom: none; }
  .cve-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8a9bb0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #eef0f4;
    margin-bottom: 0.3rem;
  }
  .cve-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #1a56db;
  }
  .badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .badge-CRITICAL { background: #fee2e2; color: #991b1b; }
  .badge-HIGH     { background: #fff3e0; color: #92400e; }
  .badge-MEDIUM   { background: #fefce8; color: #854d0e; }
  .badge-LOW      { background: #f0fdf4; color: #166534; }
  .badge-KEV      { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }

  /* ── Intervention panel ── */
  .intervention-card {
    background: #0f1923;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    color: #ffffff;
  }
  .intervention-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8a9bb0;
    margin-bottom: 0.5rem;
  }
  .intervention-situation {
    font-size: 1rem;
    font-weight: 500;
    color: #ffffff;
    margin-bottom: 0.3rem;
  }
  .intervention-urgency {
    font-size: 0.8rem;
    color: #f59e0b;
    font-weight: 500;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
  }
  .intervention-step {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.5rem 0;
    font-size: 0.875rem;
    color: #c8d6e8;
    border-bottom: 1px solid #1e2d3e;
  }
  .intervention-step:last-child { border-bottom: none; }
  .step-number {
    background: #1a56db;
    color: #ffffff;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    flex-shrink: 0;
    margin-top: 0.05rem;
  }
  .intervention-reassurance {
    font-size: 0.82rem;
    color: #6b9ab8;
    margin-top: 0.8rem;
    font-style: italic;
  }

  /* ── Source badges ── */
  .source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }
  .source-badge {
    background: #eef2f9;
    border: 1px solid #d0daea;
    border-radius: 5px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    color: #3a4a5a;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Clarification card ── */
  .clarify-card {
    background: #f0f5ff;
    border: 1.5px solid #b8ccf5;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
  }
  .clarify-card p {
    font-size: 0.9rem;
    color: #1a3a7a;
    margin: 0;
    line-height: 1.6;
  }

  /* ── Mock mode notice ── */
  .mock-notice {
    background: #fffbeb;
    border: 1px solid #f5d9a0;
    border-radius: 7px;
    padding: 0.5rem 0.9rem;
    font-size: 0.78rem;
    color: #7a4a00;
    margin-bottom: 1rem;
  }

  /* ── Timing footer ── */
  .timing-footer {
    font-size: 0.75rem;
    color: #8a9bb0;
    text-align: right;
    margin-top: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

VERDICT_LABELS = {
    "MALICIOUS":            "Malicious",
    "VULNERABLE":           "Vulnerable",
    "UNCERTAIN":            "Uncertain",
    "SAFE":                 "Safe",
    "INSUFFICIENT_EVIDENCE":"Insufficient Evidence",
    "CLARIFY":              "More Information Needed",
}

VERDICT_ICONS = {
    "MALICIOUS":            "●",
    "VULNERABLE":           "▲",
    "UNCERTAIN":            "◆",
    "SAFE":                 "✓",
    "INSUFFICIENT_EVIDENCE":"○",
    "CLARIFY":              "?",
}

RISK_COLORS = {
    "MALICIOUS":            "#dc2626",
    "VULNERABLE":           "#d97706",
    "UNCERTAIN":            "#ca8a04",
    "SAFE":                 "#16a34a",
    "INSUFFICIENT_EVIDENCE":"#6b7280",
    "CLARIFY":              "#2563eb",
}


def risk_gauge_html(risk_score: int, verdict: str) -> str:
    import math
    color  = RISK_COLORS.get(verdict, "#6b7280")
    pct    = risk_score / 100
    angle  = -150 + pct * 300
    cx, cy = 80, 75
    nx     = cx + 48 * math.cos((angle - 90) * math.pi / 180)
    ny     = cy + 48 * math.sin((angle - 90) * math.pi / 180)
    return f"""
    <svg viewBox="0 0 160 100" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:200px;display:block;margin:0 auto">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#16a34a"/>
          <stop offset="40%"  stop-color="#ca8a04"/>
          <stop offset="70%"  stop-color="#d97706"/>
          <stop offset="100%" stop-color="#dc2626"/>
        </linearGradient>
      </defs>
      <!-- Track -->
      <path d="M 22 75 A 58 58 0 0 1 138 75"
            fill="none" stroke="#e4e9f0" stroke-width="8" stroke-linecap="round"/>
      <!-- Colored arc -->
      <path d="M 22 75 A 58 58 0 0 1 138 75"
            fill="none" stroke="url(#arcGrad)" stroke-width="8"
            stroke-linecap="round" opacity="0.35"/>
      <!-- Active needle -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
      <!-- Center dot -->
      <circle cx="{cx}" cy="{cy}" r="4" fill="{color}"/>
      <!-- Score -->
      <text x="{cx}" y="95" text-anchor="middle"
            font-family="IBM Plex Mono" font-size="13" font-weight="500"
            fill="{color}">{risk_score}</text>
      <text x="{cx}" y="103" text-anchor="middle"
            font-family="IBM Plex Sans" font-size="7" fill="#8a9bb0">RISK SCORE</text>
    </svg>"""


def verdict_card_html(verdict: str, confidence: float) -> str:
    icon  = VERDICT_ICONS.get(verdict, "?")
    label = VERDICT_LABELS.get(verdict, verdict)
    conf  = f"{confidence:.0%} confidence" if confidence > 0 else ""
    return f"""
    <div class="verdict-card verdict-{verdict}">
      <div class="verdict-label">Verdict</div>
      <div class="verdict-value">{icon}&nbsp; {label}</div>
      {f'<div class="verdict-confidence">{conf}</div>' if conf else ''}
    </div>"""


def evidence_panel_html(items: list, title: str = "Evidence") -> str:
    if not items:
        return ""
    rows = "".join(
        f'<div class="evidence-item"><div class="evidence-dot"></div><span>{item}</span></div>'
        for item in items
    )
    return f"""
    <div class="panel-card">
      <div class="panel-title">{title}</div>
      {rows}
    </div>"""


def cve_table_html(cves: list) -> str:
    if not cves:
        return ""
    rows = ""
    for cve in cves:
        sev     = cve.get("cvss_severity", "UNKNOWN")
        score   = cve.get("cvss_score", 0)
        kev     = '<span class="badge badge-KEV">KEV</span>' if cve.get("actively_exploited") else ""
        desc    = cve.get("description", "")[:80] + "..." if len(cve.get("description","")) > 80 else cve.get("description","")
        rows   += f"""
        <div class="cve-row">
          <span class="cve-id">{cve.get("cve_id","")}</span>
          <span><span class="badge badge-{sev}">{score}</span></span>
          <span>{kev if kev else f'<span style="color:#8a9bb0;font-size:0.78rem">{sev}</span>'}</span>
          <span style="color:#5a6a7a;font-size:0.8rem">{desc}</span>
        </div>"""

    return f"""
    <div class="panel-card">
      <div class="panel-title">CVE Details</div>
      <div class="cve-row cve-header">
        <span>CVE ID</span><span>CVSS</span><span>Status</span><span>Description</span>
      </div>
      {rows}
    </div>"""


def intervention_html(data: dict) -> str:
    if not data:
        return ""
    situation   = data.get("situation", "")
    urgency     = data.get("urgency", "")
    steps       = data.get("steps", [])
    reassurance = data.get("reassurance", "")

    steps_html = "".join(
        f'<div class="intervention-step"><div class="step-number">{i+1}</div><span>{step}</span></div>'
        for i, step in enumerate(steps)
    )
    reassurance_html = f'<div class="intervention-reassurance">{reassurance}</div>' if reassurance else ""

    return f"""
    <div class="intervention-card">
      <div class="intervention-title">Recommended Actions</div>
      <div class="intervention-situation">{situation}</div>
      {f'<div class="intervention-urgency">{urgency}</div>' if urgency else ''}
      {steps_html}
      {reassurance_html}
    </div>"""


def sources_html(citations: list, mock_mode: bool) -> str:
    sources = []
    if any("VirusTotal" in c for c in citations):
        sources.append("VirusTotal")
    if any("PhishTank" in c for c in citations):
        sources.append("PhishTank")
    if any("CISA" in c for c in citations):
        sources.append("CISA KEV")
    if any("NVD" in c or "National Vulnerability" in c for c in citations):
        sources.append("NIST NVD")

    if not sources:
        return ""

    badges = "".join(f'<span class="source-badge">{s}</span>' for s in sources)
    mock   = '<span class="source-badge" style="color:#7a4a00;background:#fff8ed;border-color:#f5d9a0">Mock Mode</span>' if mock_mode else ""

    return f"""
    <div class="panel-card">
      <div class="panel-title">Data Sources</div>
      <div class="source-row">{badges}{mock}</div>
    </div>"""


def call_api(user_input: str, already_clicked: bool) -> dict:
    """Call the pipeline directly — no HTTP needed on HF Spaces."""
    try:
        from main import analyze
        result = analyze(
            user_input           = user_input,
            user_already_clicked = already_clicked,
        )
        return result.to_dict()
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {}


# ============================================================
# MAIN UI
# ============================================================

def main():
    # Header
    st.markdown("""
    <div class="guardian-header">
      <h1>Cybersecurity Guardian</h1>
      <p>AI-powered threat analysis for URLs, emails, and software vulnerabilities</p>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    col_input, col_btn = st.columns([4, 1])

    with col_input:
            if "user_input" not in st.session_state:
                st.session_state.user_input = ""

            user_input = st.text_area(
                label            = "Input",
                placeholder      = "Paste a suspicious URL, email text, or enter software + version (e.g. Zoom 5.15.2)",
                height           = 100,
                label_visibility = "collapsed",
                value            = st.session_state.user_input,
                key              = "input_box",
            )
            st.session_state.user_input = user_input

    with col_btn:
        already_clicked = st.checkbox("Already clicked?", value=False)
        analyze_clicked = st.button("Analyze", use_container_width=True)

    # Run analysis
    if analyze_clicked and user_input.strip():
        with st.spinner("Analyzing..."):
            data = call_api(user_input.strip(), already_clicked)

        if not data:
            return

        verdict    = data.get("verdict", "INSUFFICIENT_EVIDENCE")
        confidence = data.get("confidence", 0.0)
        risk_score = data.get("risk_score", 0)
        mock_mode  = data.get("mock_mode", True)

        # Mock mode notice
        if mock_mode:
            st.markdown(
                '<div class="mock-notice">Development mode — VirusTotal responses are simulated. '
                'Set <code>MOCK_VIRUSTOTAL=false</code> for live scanning.</div>',
                unsafe_allow_html=True
            )

        # ── Clarification ────────────────────────────────────
        if verdict == "CLARIFY":
            clarification = data.get("clarification", "")
            st.markdown(f"""
            <div class="clarify-card">
              <div class="panel-title">More Information Needed</div>
              <p>{clarification}</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Main result layout ────────────────────────────────
        col_gauge, col_verdict, col_summary = st.columns([1, 1.2, 2.8])

        with col_gauge:
            st.markdown(risk_gauge_html(risk_score, verdict), unsafe_allow_html=True)

        with col_verdict:
            st.markdown(verdict_card_html(verdict, confidence), unsafe_allow_html=True)

            # Agents activated
            agents = data.get("agents_activated", [])
            if agents:
                agent_text = " + ".join(a.capitalize() for a in agents)
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#8a9bb0;margin-top:-0.5rem">'
                    f'Analyzed by: {agent_text} agent{"s" if len(agents)>1 else ""}</div>',
                    unsafe_allow_html=True
                )

        with col_summary:
            summary = data.get("summary", "")
            if summary:
                st.markdown(f"""
                <div class="panel-card" style="height:100%;box-sizing:border-box">
                  <div class="panel-title">Analysis Summary</div>
                  <p style="font-size:0.9rem;color:#2a3545;line-height:1.65;margin:0">{summary}</p>
                </div>
                """, unsafe_allow_html=True)

        # ── TABS ─────────────────────────────────────────────
        ph_evidence   = data.get("phishing_evidence",   [])
        vuln_evidence = data.get("vuln_evidence",        [])
        vuln_cves     = data.get("vuln_cves",            [])
        actions       = data.get("actions",              [])
        intervention  = data.get("intervention",         {})
        citations     = data.get("citations",            [])
        ms            = data.get("processing_time_ms",   0)

        tab_actions, tab_analysis = st.tabs(["  Actions  ", "  Analysis  "])

        with tab_analysis:
            col_ev1, col_ev2 = st.columns(2)

            with col_ev1:
                if ph_evidence:
                    st.markdown(
                        evidence_panel_html(ph_evidence, "Phishing Analysis"),
                        unsafe_allow_html=True
                    )
                if vuln_evidence:
                    st.markdown(
                        evidence_panel_html(vuln_evidence, "Vulnerability Analysis"),
                        unsafe_allow_html=True
                    )

            with col_ev2:
                if vuln_cves:
                    import pandas as pd
                    st.markdown(
                        '<div class="panel-card"><div class="panel-title">CVE Details</div>',
                        unsafe_allow_html=True
                    )
                    cve_df = pd.DataFrame([{
                        "CVE ID":    c.get("cve_id", ""),
                        "CVSS":      c.get("cvss_score", ""),
                        "Severity":  c.get("cvss_severity", ""),
                        "Exploited": "Yes — CISA KEV" if c.get("actively_exploited") else "No",
                    } for c in vuln_cves])
                    st.dataframe(cve_df, hide_index=True, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        with tab_actions:
            if actions:
                steps_html = "".join(
                    f'<div class="intervention-step"><div class="step-number">{i+1}</div><span style="color:#c8d6e8">{a}</span></div>'
                    for i, a in enumerate(actions)
                )
                st.markdown(f"""
                <div class="intervention-card">
                  <div class="intervention-title">Prioritized Actions</div>
                  {steps_html}
                </div>
                """, unsafe_allow_html=True)

            if intervention and intervention.get("steps"):
                st.markdown(intervention_html(intervention), unsafe_allow_html=True)

            col_src, col_time = st.columns([3, 1])
            with col_src:
                st.markdown(sources_html(citations, mock_mode), unsafe_allow_html=True)
            with col_time:
                st.markdown(
                    f'<div class="timing-footer">Analysis completed in {ms:,}ms</div>',
                    unsafe_allow_html=True
                )

    elif analyze_clicked and not user_input.strip():
        st.warning("Please enter a URL, email text, or software version to analyze.")

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #e4e9f0;
                font-size:0.75rem;color:#8a9bb0;text-align:center">
      Cybersecurity Guardian &nbsp;·&nbsp;
      Data: CISA KEV · NIST NVD · PhishTank · VirusTotal &nbsp;·&nbsp;
      INFO7375 — Prompt Engineering & AI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()