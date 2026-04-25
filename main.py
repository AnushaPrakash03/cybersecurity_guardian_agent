"""
main.py
Personal Cybersecurity Guardian Agent

Single entry point for the complete pipeline.
Called by both the FastAPI backend and directly for testing.

Pipeline:
    User Input
        │
        ▼
    router.py          ← Intent classification
        │
        ├──────────────────────┐
        ▼                      ▼
    phishing_agent.py   vulnerability_agent.py
        │                      │
        └──────────┬───────────┘
                   ▼
            synthesis.py       ← Conflict resolution + intervention
                   │
                   ▼
            GuardianResult     ← Final unified output

Usage:
    # Direct
    from main import analyze
    result = analyze("Check http://paypal-verify.xyz/login")

    # With context
    result = analyze(
        user_input="I have Log4j 2.14.1",
        user_already_clicked=False
    )

    print(result.verdict)       # VULNERABLE
    print(result.summary)       # Plain English
    print(result.intervention)  # What to do now
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

from router                      import route_query, RouterResult
from agents.phishing_agent       import run_phishing_agent, PhishingAgentResult
from agents.vulnerability_agent  import run_vulnerability_agent, VulnerabilityAgentResult
from synthesis                   import run_synthesis, SynthesisResult


# ============================================================
# FINAL OUTPUT SCHEMA
# ============================================================

@dataclass
class GuardianResult:
    """
    Top-level output from the Guardian Agent pipeline.
    This is what the API and Streamlit UI consume.
    """
    # Core verdict
    verdict:            str             # MALICIOUS|VULNERABLE|SAFE|UNCERTAIN|INSUFFICIENT_EVIDENCE
    confidence:         float           # 0.0 – 1.0
    risk_score:         int             # 0–100 for the UI gauge

    # Agents involved
    agents_activated:   list            # ["phishing"] | ["vulnerability"] | ["phishing","vulnerability"]

    # Per-agent findings (for evidence breakdown panel)
    phishing_verdict:   Optional[str]
    phishing_confidence:Optional[float]
    phishing_evidence:  list

    vuln_verdict:       Optional[str]
    vuln_confidence:    Optional[float]
    vuln_evidence:      list
    vuln_cves:          list            # CVE dicts for display

    # Synthesis outputs
    summary:            str             # Plain English 2-3 sentences
    actions:            list            # Prioritized action steps

    # Contextual intervention (professor's addition)
    intervention:       dict            # situation, urgency, steps, reassurance

    # Source attribution
    citations:          list

    # Routing info (for debugging/display)
    intent:             str
    extracted_url:      Optional[str]
    software:           Optional[str]
    version:            Optional[str]
    clarification:      Optional[str]   # If router needs more info

    # Metadata
    processing_time_ms: int
    timestamp:          str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "verdict":             self.verdict,
            "confidence":          round(self.confidence, 3),
            "risk_score":          self.risk_score,
            "agents_activated":    self.agents_activated,
            "phishing_verdict":    self.phishing_verdict,
            "phishing_confidence": self.phishing_confidence,
            "phishing_evidence":   self.phishing_evidence,
            "vuln_verdict":        self.vuln_verdict,
            "vuln_confidence":     self.vuln_confidence,
            "vuln_evidence":       self.vuln_evidence,
            "vuln_cves":           self.vuln_cves,
            "summary":             self.summary,
            "actions":             self.actions,
            "intervention":        self.intervention,
            "citations":           self.citations,
            "intent":              self.intent,
            "extracted_url":       self.extracted_url,
            "software":            self.software,
            "version":             self.version,
            "clarification":       self.clarification,
            "processing_time_ms":  self.processing_time_ms,
            "timestamp":           self.timestamp,
        }


# ============================================================
# RISK SCORE MAPPING
# ============================================================

VERDICT_RISK_SCORES = {
    "MALICIOUS":            95,
    "VULNERABLE":           80,
    "UNCERTAIN":            55,
    "SAFE":                 5,
    "INSUFFICIENT_EVIDENCE":40,
}

def _compute_risk_score(verdict: str, confidence: float) -> int:
    """
    Convert verdict + confidence to a 0-100 risk score for the gauge.

    Base score from verdict, adjusted by confidence:
    - High confidence amplifies the base score
    - Low confidence pulls score toward 50 (uncertain midpoint)
    """
    base  = VERDICT_RISK_SCORES.get(verdict, 40)

    # Blend base score with confidence
    # Full confidence → pure base score
    # Zero confidence → pulled toward 40 (uncertain midpoint)
    score = int(base * confidence + 40 * (1 - confidence))
    return max(0, min(100, score))


# ============================================================
# CLARIFICATION RESULT
# ============================================================

def _clarification_result(router_result: RouterResult) -> GuardianResult:
    """
    Return a result asking the user for more information.
    No agents are run.
    """
    return GuardianResult(
        verdict="CLARIFY",
        confidence=0.0,
        risk_score=0,
        agents_activated=[],
        phishing_verdict=None,
        phishing_confidence=None,
        phishing_evidence=[],
        vuln_verdict=None,
        vuln_confidence=None,
        vuln_evidence=[],
        vuln_cves=[],
        summary="",
        actions=[],
        intervention={},
        citations=[],
        intent="clarify",
        extracted_url=None,
        software=router_result.software,
        version=router_result.version,
        clarification=router_result.clarification,
        processing_time_ms=0,
    )


# ============================================================
# MAIN ANALYZE FUNCTION
# ============================================================

def analyze(
    user_input:           str,
    user_already_clicked: bool = False,
) -> GuardianResult:
    """
    Run the complete Guardian Agent pipeline.

    Args:
        user_input:           Raw text from the user (URL, email, software query)
        user_already_clicked: True if user mentions they already clicked the link

    Returns:
        GuardianResult with verdict, evidence, summary, and intervention plan
    """
    start_time = time.time()

    # ── Step 1: Route the query ──────────────────────────────
    router_result = route_query(user_input)

    # If router needs clarification — return immediately, no agents run
    if router_result.needs_clarification():
        result = _clarification_result(router_result)
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    # ── Step 2: Run agents ───────────────────────────────────
    phishing_result = run_phishing_agent(router_result)
    vuln_result     = run_vulnerability_agent(router_result)

    # ── Step 3: Synthesis + intervention ────────────────────
    synthesis = run_synthesis(
        phishing_result      = phishing_result,
        vuln_result          = vuln_result,
        router_result        = router_result,
        user_already_clicked = user_already_clicked,
    )

    # ── Step 4: Build final result ───────────────────────────
    risk_score = _compute_risk_score(
        synthesis.top_level_verdict,
        synthesis.confidence_score,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    return GuardianResult(
        verdict             = synthesis.top_level_verdict,
        confidence          = synthesis.confidence_score,
        risk_score          = risk_score,
        agents_activated    = synthesis.contributing_agents,
        phishing_verdict    = (
            phishing_result.verdict
            if phishing_result.verdict != "NOT_ACTIVATED" else None
        ),
        phishing_confidence = (
            phishing_result.confidence
            if phishing_result.verdict != "NOT_ACTIVATED" else None
        ),
        phishing_evidence   = phishing_result.evidence[:4],
        vuln_verdict        = (
            vuln_result.verdict
            if vuln_result.verdict != "NOT_ACTIVATED" else None
        ),
        vuln_confidence     = (
            vuln_result.confidence
            if vuln_result.verdict != "NOT_ACTIVATED" else None
        ),
        vuln_evidence       = vuln_result.evidence[:4],
        vuln_cves           = vuln_result.cves[:3],
        summary             = synthesis.plain_english_summary,
        actions             = synthesis.prioritized_actions,
        intervention        = synthesis.intervention,
        citations           = synthesis.source_citations,
        intent              = router_result.intent,
        extracted_url       = router_result.extracted_url,
        software            = router_result.software,
        version             = router_result.version,
        clarification       = None,
        processing_time_ms  = processing_time_ms,
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    test_cases = [
        {
            "input":    "http://paypal-secure-verify.xyz/login/confirm",
            "expected": "MALICIOUS",
            "desc":     "Phishing URL",
        },
        {
            "input":    "I have Log4j 2.14.1 installed",
            "expected": "VULNERABLE",
            "desc":     "Log4Shell",
        },
        {
            "input":    "https://www.google.com",
            "expected": "SAFE",
            "desc":     "Safe URL",
        },
        {
            "input":    "check my zoom",
            "expected": "CLARIFY",
            "desc":     "Ambiguous — needs version",
        },
        {
            "input":    "Check Apache HTTP Server 2.4.49",
            "expected": "VULNERABLE",
            "desc":     "Apache path traversal",
        },
    ]

    print("=" * 65)
    print("Guardian Agent — main.py Pipeline Test")
    print("=" * 65)

    passed = 0
    for tc in test_cases:
        result = analyze(tc["input"])
        ok     = result.verdict == tc["expected"]
        status = "✅" if ok else "❌"
        if ok:
            passed += 1

        print(f"\n{status} {tc['desc']}")
        print(f"   Input:      {tc['input'][:60]}")
        print(f"   Expected:   {tc['expected']:<25} Got: {result.verdict}")
        print(f"   Risk Score: {result.risk_score}/100")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Agents:     {result.agents_activated}")

        if result.clarification:
            print(f"   Clarification: {result.clarification[:80]}")
        elif result.summary:
            print(f"   Summary: {result.summary[:100]}")

        if result.intervention:
            print(f"   Intervention — {result.intervention.get('situation','')}")
            steps = result.intervention.get("steps", [])
            for s in steps[:2]:
                print(f"     • {s}")

        print(f"   Time: {result.processing_time_ms}ms")
        print("-" * 65)

    print(f"\nResults: {passed}/{len(test_cases)}")
    status = "PASS" if passed >= 4 else "FAIL"
    print(f"Target (≥ 4/5): {status}")
    print("=" * 65)