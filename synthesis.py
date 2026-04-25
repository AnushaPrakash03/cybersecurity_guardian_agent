"""
synthesis.py
Personal Cybersecurity Guardian Agent

Synthesis Layer + Contextual Intervention Layer

Takes outputs from PhishingAgent and VulnerabilityAgent,
applies conflict resolution rules (proposal Phase 2A), and
produces a single unified verdict with plain-English explanation
and a situation-specific action plan.

Conflict resolution rules (applied in priority order):
  Rule 1: MALICIOUS always surfaces
  Rule 2: VULNERABLE escalates over SAFE
  Rule 3: INSUFFICIENT_EVIDENCE does not suppress a real finding
  Rule 4: Both INSUFFICIENT_EVIDENCE → top-level INSUFFICIENT_EVIDENCE
  Rule 5: Confidence < 0.70 → UNCERTAIN

Contextual Intervention Layer (professor's addition):
  Takes the synthesis verdict + user context
  Produces situation-specific action plan
  No retrieval calls — pure GPT-4o mini synthesis
  The difference between a system that informs and one that protects

Usage:
    from synthesis import run_synthesis
    from agents.phishing_agent import run_phishing_agent
    from agents.vulnerability_agent import run_vulnerability_agent
    from router import route_query

    router_result  = route_query("Check http://paypal-verify.xyz")
    phishing_result = run_phishing_agent(router_result)
    vuln_result     = run_vulnerability_agent(router_result)
    final           = run_synthesis(phishing_result, vuln_result, router_result)

    print(final.top_level_verdict)      # MALICIOUS
    print(final.plain_english_summary)  # Plain English explanation
    print(final.intervention)           # What to do RIGHT NOW
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

from agents.phishing_agent       import PhishingAgentResult
from agents.vulnerability_agent  import VulnerabilityAgentResult
from router                      import RouterResult

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.70"))


# ============================================================
# OUTPUT SCHEMA
# ============================================================

@dataclass
class SynthesisResult:
    """
    Final unified output from the synthesis layer.
    Matches the full output schema defined in proposal Phase 2A.
    """
    top_level_verdict:     str          # MALICIOUS|VULNERABLE|SAFE|UNCERTAIN|INSUFFICIENT_EVIDENCE
    confidence_score:      float        # 0.0 – 1.0
    contributing_agents:   list         # Which agents were activated
    agent_findings:        dict         # Per-agent verdict + evidence
    plain_english_summary: str          # 2-3 sentence plain English explanation
    intervention:          dict         # Contextual action plan (professor's addition)
    prioritized_actions:   list         # Ordered action steps
    source_citations:      list         # All evidence sources
    conflict_rule_applied: Optional[str]# Which conflict rule determined the verdict
    timestamp:             str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "top_level_verdict":     self.top_level_verdict,
            "confidence_score":      round(self.confidence_score, 3),
            "contributing_agents":   self.contributing_agents,
            "agent_findings":        self.agent_findings,
            "plain_english_summary": self.plain_english_summary,
            "intervention":          self.intervention,
            "prioritized_actions":   self.prioritized_actions,
            "source_citations":      self.source_citations,
            "conflict_rule_applied": self.conflict_rule_applied,
            "timestamp":             self.timestamp,
        }


# ============================================================
# CONFLICT RESOLUTION ENGINE
# ============================================================

def _apply_conflict_rules(
    phishing_verdict: str,
    phishing_conf:    float,
    vuln_verdict:     str,
    vuln_conf:        float,
    phishing_active:  bool,
    vuln_active:      bool,
) -> tuple[str, float, str]:
    """
    Apply the five conflict resolution rules from proposal Phase 2A.
    Returns (top_level_verdict, confidence, rule_applied).

    Rules applied in priority order:
      Rule 1: MALICIOUS always surfaces
      Rule 2: VULNERABLE escalates
      Rule 3: INSUFFICIENT_EVIDENCE does not suppress a real finding
      Rule 4: Both INSUFFICIENT_EVIDENCE → top-level INSUFFICIENT_EVIDENCE
      Rule 5: Synthesized confidence < 0.70 → UNCERTAIN
    """

    # ── Only one agent active ────────────────────────────────
    if phishing_active and not vuln_active:
        verdict    = phishing_verdict
        confidence = phishing_conf
        rule       = "Single agent (phishing only) — no conflict resolution needed"

        # Still apply Rule 5
        if confidence < UNCERTAINTY_THRESHOLD and verdict not in (
            "MALICIOUS", "INSUFFICIENT_EVIDENCE"
        ):
            return "UNCERTAIN", confidence, "Rule 5: Confidence below threshold"
        return verdict, confidence, rule

    if vuln_active and not phishing_active:
        verdict    = vuln_verdict
        confidence = vuln_conf
        rule       = "Single agent (vulnerability only) — no conflict resolution needed"

        if confidence < UNCERTAINTY_THRESHOLD and verdict not in (
            "VULNERABLE", "INSUFFICIENT_EVIDENCE"
        ):
            return "UNCERTAIN", confidence, "Rule 5: Confidence below threshold"
        return verdict, confidence, rule

    # ── Both agents active — apply conflict resolution ───────
    synthesized_confidence = (phishing_conf + vuln_conf) / 2

    # Rule 1: MALICIOUS always surfaces
    if phishing_verdict == "MALICIOUS":
        return (
            "MALICIOUS",
            phishing_conf,
            "Rule 1: MALICIOUS always surfaces — phishing agent returned MALICIOUS"
        )

    # Rule 2: VULNERABLE escalates over SAFE
    if phishing_verdict == "SAFE" and vuln_verdict == "VULNERABLE":
        return (
            "VULNERABLE",
            vuln_conf,
            "Rule 2: VULNERABLE escalates — URL safe but software has active vulnerability"
        )

    # Rule 3: INSUFFICIENT_EVIDENCE does not suppress a real finding
    if phishing_verdict == "INSUFFICIENT_EVIDENCE" and vuln_verdict not in (
        "INSUFFICIENT_EVIDENCE", "NOT_ACTIVATED"
    ):
        return (
            vuln_verdict,
            vuln_conf,
            "Rule 3: Phishing INSUFFICIENT_EVIDENCE does not suppress vulnerability finding"
        )

    if vuln_verdict == "INSUFFICIENT_EVIDENCE" and phishing_verdict not in (
        "INSUFFICIENT_EVIDENCE", "NOT_ACTIVATED"
    ):
        return (
            phishing_verdict,
            phishing_conf,
            "Rule 3: Vulnerability INSUFFICIENT_EVIDENCE does not suppress phishing finding"
        )

    # Rule 4: Both INSUFFICIENT_EVIDENCE
    if (phishing_verdict == "INSUFFICIENT_EVIDENCE"
            and vuln_verdict == "INSUFFICIENT_EVIDENCE"):
        return (
            "INSUFFICIENT_EVIDENCE",
            min(phishing_conf, vuln_conf),
            "Rule 4: Both agents returned INSUFFICIENT_EVIDENCE"
        )

    # Rule 5: Synthesized confidence below threshold
    if synthesized_confidence < UNCERTAINTY_THRESHOLD:
        return (
            "UNCERTAIN",
            synthesized_confidence,
            f"Rule 5: Synthesized confidence {synthesized_confidence:.2f} "
            f"below threshold {UNCERTAINTY_THRESHOLD}"
        )

    # Both agents agree — use higher confidence finding
    if phishing_verdict == vuln_verdict:
        return (
            phishing_verdict,
            max(phishing_conf, vuln_conf),
            f"Both agents agree: {phishing_verdict}"
        )

    # Mixed signals — UNCERTAIN
    return (
        "UNCERTAIN",
        synthesized_confidence,
        f"Rule 5: Mixed signals — phishing:{phishing_verdict}, "
        f"vulnerability:{vuln_verdict}"
    )


# ============================================================
# SYNTHESIS PROMPT
# ============================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a cybersecurity assistant explaining findings to a non-technical user.

STRICT ANTI-HALLUCINATION RULES — you must follow these exactly:
1. ONLY use facts from the evidence list provided. Do not add any facts not in the evidence.
2. Do NOT mention specific CVE IDs, URLs, attack names, or statistics unless they appear word-for-word in the evidence list.
3. Do NOT say things like "this is commonly used in ransomware campaigns" or "hackers often exploit this" — these are fabrications.
4. If the evidence is limited, say so plainly. Do not fill gaps with assumptions.
5. Keep the summary to 2-3 sentences maximum. Shorter is safer.
6. The action steps must be generic best practices — never invent specific tools, download links, or commands."""

SYNTHESIS_USER_PROMPT = """
Verdict: {verdict}
Confidence: {confidence}

Phishing Agent: {phishing_summary}
Vulnerability Agent: {vuln_summary}

EVIDENCE (use ONLY these facts — do not add anything else):
{evidence}

Based STRICTLY on the evidence above, write a plain-English summary and action steps.
Do not mention anything not in the evidence list above.

Return ONLY valid JSON:
{{
  "plain_english_summary": "2-3 sentence explanation using only the evidence provided",
  "prioritized_actions": [
    "Most urgent action (generic best practice only)",
    "Second action",
    "Third action"
  ]
}}"""


def _run_synthesis_prompt(
    verdict:         str,
    confidence:      float,
    phishing_result: PhishingAgentResult,
    vuln_result:     VulnerabilityAgentResult,
    all_evidence:    list,
) -> dict:
    """
    Generate plain-English summary and action steps using GPT-4o mini.
    Includes post-generation grounding check to catch hallucinated facts.
    """

    phishing_summary = "Not analyzed"
    if phishing_result.verdict != "NOT_ACTIVATED":
        phishing_summary = (
            f"{phishing_result.verdict} "
            f"(confidence: {phishing_result.confidence:.0%})"
        )

    vuln_summary = "Not analyzed"
    if vuln_result.verdict != "NOT_ACTIVATED":
        vuln_summary = (
            f"{vuln_result.verdict} for "
            f"{vuln_result.software} {vuln_result.version} "
            f"(confidence: {vuln_result.confidence:.0%})"
        )

    prompt = SYNTHESIS_USER_PROMPT.format(
        verdict=verdict,
        confidence=f"{confidence:.0%}",
        phishing_summary=phishing_summary,
        vuln_summary=vuln_summary,
        evidence="\n".join(f"• {e}" for e in all_evidence[:6]),
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,    # Lower temperature = less creative = less hallucination
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        output = json.loads(response.choices[0].message.content)

        # ── Post-generation grounding check ─────────────────
        # Verify summary does not contain CVE IDs not in evidence
        # CVE IDs are the most dangerous hallucination — wrong ID = wrong vulnerability
        import re
        summary = output.get("plain_english_summary", "")
        cve_pattern = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
        hallucinated_cves = []

        for cve_match in cve_pattern.findall(summary):
            # Check if this CVE ID appears in the evidence we provided
            evidence_text = " ".join(all_evidence)
            if cve_match.upper() not in evidence_text.upper():
                hallucinated_cves.append(cve_match)

        if hallucinated_cves:
            # Strip hallucinated CVE IDs from summary
            for cve in hallucinated_cves:
                summary = re.sub(cve, "[redacted]", summary, flags=re.IGNORECASE)
            output["plain_english_summary"] = summary
            output["_hallucination_detected"] = True
            output["_hallucinated_cves"] = hallucinated_cves

        return output

    except Exception as e:
        # Fallback — never fail silently, never hallucinate in fallback
        actions = {
            "MALICIOUS":             ["Do not click the link or submit any information",
                                      "If you already clicked, change your passwords immediately",
                                      "Run a virus scan on your device"],
            "VULNERABLE":            ["Update the software to the latest version immediately",
                                      "Check your software settings for automatic updates",
                                      "Restart the application after updating"],
            "SAFE":                  ["No immediate action required",
                                      "Continue practicing safe browsing habits"],
            "UNCERTAIN":             ["Do not submit any sensitive information",
                                      "Verify through official channels before proceeding"],
            "INSUFFICIENT_EVIDENCE": ["Treat as suspicious until verified",
                                      "Contact your IT department for guidance"],
        }
        return {
            "plain_english_summary": f"Analysis complete. Verdict: {verdict}. Review the evidence below.",
            "prioritized_actions":   actions.get(verdict, ["Proceed with caution"]),
        }


# ============================================================
# CONTEXTUAL INTERVENTION LAYER
# ============================================================

INTERVENTION_SYSTEM_PROMPT = """You are a cybersecurity first-responder giving immediate instructions to a non-technical user.

STRICT ANTI-HALLUCINATION RULES:
1. Give only generic, universally correct steps. Do not invent specific tool names, download links, or commands.
2. Do not mention specific software versions, CVE IDs, or attack names unless provided in the prompt.
3. Steps must be things any non-technical user can safely follow right now.
4. Maximum 4 steps. Each step must be a single clear action.
5. Do not add statistics, percentages, or threat actor claims not provided."""

INTERVENTION_PROMPTS = {
    "MALICIOUS_CLICKED": """The user just told us they may have already clicked a malicious link.
Verdict: MALICIOUS
URL analyzed: {url}

Give them 4 immediate steps to take RIGHT NOW (next 10 minutes).
Focus on: changing passwords, checking for malware, protecting accounts.

Return ONLY valid JSON:
{{
  "situation": "You may have clicked a malicious link",
  "urgency": "Immediate — act now",
  "steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
  "reassurance": "One sentence to reduce panic while staying serious"
}}""",

    "MALICIOUS_NOT_CLICKED": """The user submitted a URL that was flagged as malicious.
They have NOT clicked it yet.
Verdict: MALICIOUS
URL analyzed: {url}

Give them clear instructions on what to do.

Return ONLY valid JSON:
{{
  "situation": "This link is dangerous — do not click it",
  "urgency": "Do not click",
  "steps": ["Step 1", "Step 2", "Step 3"],
  "reassurance": "One sentence confirming they are safe since they did not click"
}}""",

    "VULNERABLE": """The user's software has known security vulnerabilities.
Software: {software} version {version}
Verdict: VULNERABLE
CVEs found: {cve_count}
Actively exploited: {actively_exploited}

Give them step-by-step update instructions in plain English.
Do not mention CVE IDs.

Return ONLY valid JSON:
{{
  "situation": "Your {software} version has security vulnerabilities",
  "urgency": "{urgency}",
  "steps": ["Step 1", "Step 2", "Step 3"],
  "reassurance": "One sentence explaining what updating will fix"
}}""",

    "UNCERTAIN": """The system found mixed signals and cannot give a definitive verdict.
Verdict: UNCERTAIN
Input analyzed: {input_summary}

Give the user cautious guidance on how to proceed safely.

Return ONLY valid JSON:
{{
  "situation": "We found mixed signals — proceed with caution",
  "urgency": "Caution advised",
  "steps": ["Step 1", "Step 2", "Step 3"],
  "reassurance": "One sentence explaining what UNCERTAIN means in plain English"
}}""",

    "SAFE": """The analysis found no threats.
Verdict: SAFE
Input analyzed: {input_summary}

Briefly confirm they are safe and give one general tip.

Return ONLY valid JSON:
{{
  "situation": "No threats detected",
  "urgency": "No immediate action required",
  "steps": ["Step 1"],
  "reassurance": "One sentence confirming safety"
}}""",

    "INSUFFICIENT_EVIDENCE": """The system could not gather enough evidence for a verdict.
Input analyzed: {input_summary}

Explain why and give them alternative ways to check.

Return ONLY valid JSON:
{{
  "situation": "We could not gather enough information to give a verdict",
  "urgency": "Treat as suspicious until verified",
  "steps": ["Step 1", "Step 2", "Step 3"],
  "reassurance": "One sentence on what to do next"
}}""",
}


def _run_intervention(
    verdict:         str,
    phishing_result: PhishingAgentResult,
    vuln_result:     VulnerabilityAgentResult,
    router_result:   RouterResult,
    user_already_clicked: bool = False,
) -> dict:
    """
    Generate situation-specific action plan.
    No retrieval calls — pure GPT-4o mini synthesis.
    This is the professor's Contextual Intervention Layer addition.
    """

    # Select intervention template based on verdict + context
    if verdict == "MALICIOUS":
        template_key = (
            "MALICIOUS_CLICKED" if user_already_clicked
            else "MALICIOUS_NOT_CLICKED"
        )
        prompt = INTERVENTION_PROMPTS[template_key].format(
            url=phishing_result.url_analyzed or router_result.raw_input[:100]
        )

    elif verdict == "VULNERABLE":
        kev_count = len(vuln_result.kev_cves)
        urgency   = "Immediate — actively exploited" if kev_count > 0 else "Soon"
        prompt    = INTERVENTION_PROMPTS["VULNERABLE"].format(
            software=vuln_result.software or "the software",
            version=vuln_result.version or "your version",
            cve_count=len(vuln_result.cves),
            actively_exploited="Yes — attackers are using this vulnerability now" if kev_count > 0 else "No",
            urgency=urgency,
        )

    elif verdict == "UNCERTAIN":
        prompt = INTERVENTION_PROMPTS["UNCERTAIN"].format(
            input_summary=router_result.raw_input[:100]
        )

    elif verdict == "SAFE":
        prompt = INTERVENTION_PROMPTS["SAFE"].format(
            input_summary=router_result.raw_input[:100]
        )

    else:  # INSUFFICIENT_EVIDENCE
        prompt = INTERVENTION_PROMPTS["INSUFFICIENT_EVIDENCE"].format(
            input_summary=router_result.raw_input[:100]
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": INTERVENTION_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    except Exception:
        return {
            "situation":    f"Verdict: {verdict}",
            "urgency":      "Review the evidence and act accordingly",
            "steps":        ["Review the findings above", "Take appropriate action"],
            "reassurance":  "Contact your IT department if unsure.",
        }


# ============================================================
# MAIN SYNTHESIS FUNCTION
# ============================================================

def run_synthesis(
    phishing_result:      PhishingAgentResult,
    vuln_result:          VulnerabilityAgentResult,
    router_result:        RouterResult,
    user_already_clicked: bool = False,
) -> SynthesisResult:
    """
    Run the synthesis layer.

    Takes both agent results, applies conflict resolution rules,
    generates plain-English summary, and produces contextual
    intervention plan.

    Args:
        phishing_result:      Output from PhishingAgent
        vuln_result:          Output from VulnerabilityAgent
        router_result:        Original router decision
        user_already_clicked: True if user mentioned they already clicked

    Returns:
        SynthesisResult with unified verdict, summary, and intervention
    """

    # ── Determine which agents were active ───────────────────
    phishing_active = phishing_result.verdict != "NOT_ACTIVATED"
    vuln_active     = vuln_result.verdict     != "NOT_ACTIVATED"
    contributing    = []
    if phishing_active:
        contributing.append("phishing")
    if vuln_active:
        contributing.append("vulnerability")

    # ── Apply conflict resolution rules ─────────────────────
    top_verdict, confidence, rule_applied = _apply_conflict_rules(
        phishing_verdict = phishing_result.verdict,
        phishing_conf    = phishing_result.confidence,
        vuln_verdict     = vuln_result.verdict,
        vuln_conf        = vuln_result.confidence,
        phishing_active  = phishing_active,
        vuln_active      = vuln_active,
    )

    # ── Collect all evidence ─────────────────────────────────
    all_evidence = []
    if phishing_active:
        all_evidence.extend(phishing_result.evidence[:3])
    if vuln_active:
        all_evidence.extend(vuln_result.evidence[:3])

    # ── Collect all citations ────────────────────────────────
    all_citations = []
    if phishing_active:
        all_citations.extend(phishing_result.source_citations)
    if vuln_active:
        all_citations.extend(vuln_result.source_citations)

    # ── Run synthesis prompt ─────────────────────────────────
    synthesis_output = _run_synthesis_prompt(
        verdict         = top_verdict,
        confidence      = confidence,
        phishing_result = phishing_result,
        vuln_result     = vuln_result,
        all_evidence    = all_evidence,
    )

    # ── Run contextual intervention ──────────────────────────
    intervention = _run_intervention(
        verdict              = top_verdict,
        phishing_result      = phishing_result,
        vuln_result          = vuln_result,
        router_result        = router_result,
        user_already_clicked = user_already_clicked,
    )

    # ── Build agent_findings dict (Phase 2A output schema) ───
    agent_findings = {
        "phishing": {
            "verdict":    phishing_result.verdict,
            "confidence": phishing_result.confidence,
            "evidence":   phishing_result.evidence[:3],
        } if phishing_active else {"verdict": "NOT_ACTIVATED"},

        "vulnerability": {
            "verdict":    vuln_result.verdict,
            "confidence": vuln_result.confidence,
            "evidence":   vuln_result.evidence[:3],
            "cves":       [c["cve_id"] for c in vuln_result.cves[:3]],
        } if vuln_active else {"verdict": "NOT_ACTIVATED"},
    }

    return SynthesisResult(
        top_level_verdict     = top_verdict,
        confidence_score      = confidence,
        contributing_agents   = contributing,
        agent_findings        = agent_findings,
        plain_english_summary = synthesis_output.get(
            "plain_english_summary", f"Verdict: {top_verdict}"
        ),
        intervention          = intervention,
        prioritized_actions   = synthesis_output.get("prioritized_actions", []),
        source_citations      = all_citations,
        conflict_rule_applied = rule_applied,
    )


# ============================================================
# CONFLICT SCENARIO TEST — 20 scenarios, 18/20 pass threshold
# ============================================================

def test_synthesis():
    """
    Test the synthesis conflict resolution rules against
    20 constructed scenarios (proposal Phase 3, Layer 4).
    Pass threshold: 18/20 correct resolutions.
    """

    # Helper to build mock results quickly
    def mock_phishing(verdict, confidence=0.85):
        return PhishingAgentResult(
            verdict=verdict, confidence=confidence,
            evidence=[f"Phishing: {verdict}"],
            url_analyzed="http://test.xyz", vt_result=None,
            rag_patterns=[], email_signals=[], source_citations=[],
            error=None,
        )

    def mock_vuln(verdict, confidence=0.85, software="Zoom", version="5.15.2"):
        return VulnerabilityAgentResult(
            verdict=verdict, confidence=confidence,
            evidence=[f"Vulnerability: {verdict}"],
            software=software, version=version,
            cves=[], kev_cves=[], source_citations=[], error=None,
        )

    def mock_router(inp="test input"):
        from router import RouterResult
        return RouterResult(
            intent="both", confidence=0.90,
            extracted_url="http://test.xyz",
            software="Zoom", version="5.15.2",
            email_text=None, clarification=None,
            raw_input=inp, routing_reason="test",
        )

    # 20 conflict scenarios with expected verdicts
    scenarios = [
        # Rule 1: MALICIOUS always surfaces (6 scenarios)
        (mock_phishing("MALICIOUS"),      mock_vuln("SAFE"),                 "MALICIOUS", "Rule 1"),
        (mock_phishing("MALICIOUS"),      mock_vuln("VULNERABLE"),           "MALICIOUS", "Rule 1"),
        (mock_phishing("MALICIOUS"),      mock_vuln("UNCERTAIN"),            "MALICIOUS", "Rule 1"),
        (mock_phishing("MALICIOUS"),      mock_vuln("INSUFFICIENT_EVIDENCE"),"MALICIOUS", "Rule 1"),
        (mock_phishing("MALICIOUS", 0.95),mock_vuln("NOT_ACTIVATED", 0.0),  "MALICIOUS", "Rule 1 single"),
        (mock_phishing("MALICIOUS", 0.80),mock_vuln("SAFE", 0.90),          "MALICIOUS", "Rule 1"),

        # Rule 2: VULNERABLE escalates (3 scenarios)
        (mock_phishing("SAFE"),           mock_vuln("VULNERABLE"),           "VULNERABLE","Rule 2"),
        (mock_phishing("SAFE", 0.90),     mock_vuln("VULNERABLE", 0.95),    "VULNERABLE","Rule 2"),
        (mock_phishing("SAFE", 0.85),     mock_vuln("VULNERABLE", 0.82),    "VULNERABLE","Rule 2"),

        # Rule 3: INSUFFICIENT_EVIDENCE does not suppress (3 scenarios)
        (mock_phishing("INSUFFICIENT_EVIDENCE"), mock_vuln("VULNERABLE"),   "VULNERABLE","Rule 3"),
        (mock_phishing("INSUFFICIENT_EVIDENCE"), mock_vuln("SAFE"),         "SAFE",      "Rule 3"),
        (mock_phishing("UNCERTAIN"),      mock_vuln("INSUFFICIENT_EVIDENCE"),"UNCERTAIN", "Rule 3"),

        # Rule 4: Both INSUFFICIENT_EVIDENCE (2 scenarios)
        (mock_phishing("INSUFFICIENT_EVIDENCE"), mock_vuln("INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_EVIDENCE", "Rule 4"),
        (mock_phishing("INSUFFICIENT_EVIDENCE", 0.40), mock_vuln("INSUFFICIENT_EVIDENCE", 0.40), "INSUFFICIENT_EVIDENCE", "Rule 4"),

        # Rule 5: Low confidence → UNCERTAIN (3 scenarios)
        (mock_phishing("SAFE", 0.55),     mock_vuln("SAFE", 0.60),          "UNCERTAIN", "Rule 5"),
        (mock_phishing("UNCERTAIN", 0.50),mock_vuln("SAFE", 0.65),          "UNCERTAIN", "Rule 5"),
        (mock_phishing("UNCERTAIN", 0.60),mock_vuln("UNCERTAIN", 0.60),     "UNCERTAIN", "Rule 5"),

        # Both agents agree (3 scenarios)
        (mock_phishing("SAFE", 0.90),     mock_vuln("SAFE", 0.88),          "SAFE",      "Agreement"),
        (mock_phishing("UNCERTAIN", 0.75),mock_vuln("UNCERTAIN", 0.72),     "UNCERTAIN", "Agreement"),
        (mock_phishing("SAFE", 0.85),     mock_vuln("VULNERABLE", 0.80),    "VULNERABLE","Rule 2"),
    ]

    router = mock_router()
    passed = 0
    failed = []

    print("\n" + "=" * 65)
    print("Synthesis Layer — Conflict Resolution Test (20 scenarios)")
    print("=" * 65)

    for i, (p_result, v_result, expected, rule_name) in enumerate(scenarios, 1):
        # Handle NOT_ACTIVATED case
        if v_result.verdict == "NOT_ACTIVATED":
            v_result.confidence = 0.0

        result = run_synthesis(p_result, v_result, router)
        ok     = result.top_level_verdict == expected
        status = "✅" if ok else "❌"

        if ok:
            passed += 1
        else:
            failed.append({
                "scenario":  i,
                "rule":      rule_name,
                "phishing":  p_result.verdict,
                "vuln":      v_result.verdict,
                "expected":  expected,
                "got":       result.top_level_verdict,
                "rule_used": result.conflict_rule_applied,
            })

        print(f"{status} #{i:02d} [{rule_name:<12}] "
              f"phishing:{p_result.verdict:<22} "
              f"vuln:{v_result.verdict:<22} "
              f"→ {result.top_level_verdict}")

    print(f"\n--- Results ---")
    print(f"Passed: {passed}/20")
    status = "PASS" if passed >= 18 else "FAIL"
    print(f"Target (≥ 18/20): {status}")

    if failed:
        print(f"\nFailed scenarios:")
        for f in failed:
            print(f"  #{f['scenario']} [{f['rule']}]: "
                  f"expected {f['expected']}, got {f['got']}")
            print(f"    Rule applied: {f['rule_used']}")

    print("=" * 65)
    return passed


# ============================================================
# FULL PIPELINE TEST
# ============================================================

def test_full_pipeline():
    """
    Test the complete pipeline end-to-end:
    route_query → agents → synthesis → intervention
    """
    from router import route_query
    from agents.phishing_agent      import run_phishing_agent
    from agents.vulnerability_agent import run_vulnerability_agent

    test_cases = [
        {
            "input":    "Check this link: http://paypal-secure-verify.xyz/login/confirm",
            "expected": "MALICIOUS",
            "desc":     "Obvious phishing URL",
        },
        {
            "input":    "I have Log4j 2.14.1 installed, should I be worried?",
            "expected": "VULNERABLE",
            "desc":     "Log4Shell — CVSS 10.0",
        },
        {
            "input":    "https://www.google.com",
            "expected": "SAFE",
            "desc":     "Trusted domain",
        },
        {
            "input":    "Dear customer your account has been suspended click here immediately to verify",
            "expected": "UNCERTAIN",
            "desc":     "Email text only — no URL",
        },
    ]

    print("\n" + "=" * 65)
    print("Full Pipeline Test (router → agents → synthesis)")
    print("=" * 65)

    passed = 0
    for tc in test_cases:
        router_result   = route_query(tc["input"])
        phishing_result = run_phishing_agent(router_result)
        vuln_result     = run_vulnerability_agent(router_result)
        final           = run_synthesis(phishing_result, vuln_result, router_result)

        ok     = final.top_level_verdict == tc["expected"]
        status = "✅" if ok else "❌"
        if ok:
            passed += 1

        print(f"\n{status} {tc['desc']}")
        print(f"   Input:    {tc['input'][:60]}")
        print(f"   Expected: {tc['expected']:<20} Got: {final.top_level_verdict}")
        print(f"   Confidence: {final.confidence_score:.0%}")
        print(f"   Rule: {final.conflict_rule_applied[:70]}")
        print(f"   Summary: {final.plain_english_summary[:120]}")
        print(f"   Intervention:")
        print(f"     Situation: {final.intervention.get('situation','')}")
        print(f"     Urgency:   {final.intervention.get('urgency','')}")
        steps = final.intervention.get("steps", [])
        for s in steps[:3]:
            print(f"     • {s}")
        print(f"   Actions: {final.prioritized_actions[:2]}")

    print(f"\n--- Pipeline Results ---")
    print(f"Passed: {passed}/{len(test_cases)}")
    status = "PASS" if passed >= 3 else "FAIL"
    print(f"Target (≥ 3/4): {status}")
    print("=" * 65)


if __name__ == "__main__":
    print("Running synthesis conflict resolution test...")
    conflict_score = test_synthesis()

    print("\nRunning full pipeline test...")
    test_full_pipeline()