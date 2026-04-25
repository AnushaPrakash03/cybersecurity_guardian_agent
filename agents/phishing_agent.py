"""
agents/phishing_agent.py
Personal Cybersecurity Guardian Agent

Phishing Agent — investigates URLs and email text for phishing indicators.

Inputs:  RouterResult with extracted_url and/or email_text
Outputs: PhishingAgentResult (structured JSON matching proposal Phase 2A schema)

Two evidence sources (combined for verdict):
  1. VirusTotal API — live URL reputation across 70+ scanning engines
  2. Pinecone RAG  — semantically similar verified phishing patterns

Usage:
    from agents.phishing_agent import run_phishing_agent
    from router import route_query

    router_result = route_query("check http://paypal-verify.xyz/login")
    result = run_phishing_agent(router_result)
    print(result.verdict)       # MALICIOUS
    print(result.confidence)    # 0.91
    print(result.evidence)      # ["VirusTotal: 25/72 engines flagged", ...]
"""

import os
import sys
import json
import asyncio


from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from tools.virustotal import scan_url, VirusTotalResult
from router import RouterResult

load_dotenv()

client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PC        = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX     = PC.Index(os.getenv("PINECONE_INDEX_NAME", "cybersecurity-guardian"))
NAMESPACE = "phishing-patterns"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ============================================================
# OUTPUT SCHEMA
# ============================================================

@dataclass
class PhishingAgentResult:
    """
    Structured output from the Phishing Agent.
    Matches the agent_findings.phishing schema in proposal Phase 2A.
    """
    verdict:          str            # MALICIOUS | SAFE | UNCERTAIN | INSUFFICIENT_EVIDENCE | NOT_ACTIVATED
    confidence:       float          # 0.0 – 1.0
    evidence:         list           # List of evidence strings
    url_analyzed:     Optional[str]  # URL that was analyzed
    vt_result:        Optional[dict] # VirusTotal raw result
    rag_patterns:     list           # Similar phishing patterns retrieved
    email_signals:    list           # Signals found in email text
    source_citations: list           # For output attribution
    error:            Optional[str]  # Error message if analysis failed
    timestamp:        str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "verdict":          self.verdict,
            "confidence":       round(self.confidence, 3),
            "evidence":         self.evidence,
            "url_analyzed":     self.url_analyzed,
            "vt_result":        self.vt_result,
            "rag_patterns":     self.rag_patterns,
            "email_signals":    self.email_signals,
            "source_citations": self.source_citations,
            "error":            self.error,
            "timestamp":        self.timestamp,
        }


# ============================================================
# NOT ACTIVATED RESULT
# ============================================================

def _not_activated() -> PhishingAgentResult:
    return PhishingAgentResult(
        verdict="NOT_ACTIVATED",
        confidence=0.0,
        evidence=[],
        url_analyzed=None,
        vt_result=None,
        rag_patterns=[],
        email_signals=[],
        source_citations=[],
        error=None,
    )


# ============================================================
# EMAIL SIGNAL EXTRACTION
# ============================================================

URGENCY_PATTERNS = [
    "urgent", "immediately", "within 24 hours", "expires",
    "suspended", "limited", "verify now", "act now",
    "account will be", "final notice", "last chance",
]

CREDENTIAL_PATTERNS = [
    "password", "username", "login", "sign in", "credentials",
    "verify your identity", "confirm your information",
    "update your billing", "payment method",
]

SENDER_SPOOF_PATTERNS = [
    "it department", "security team", "support team",
    "account team", "billing department", "customer service",
    "helpdesk", "no-reply", "noreply",
]


def _extract_email_signals(text: str) -> list:
    """Extract social engineering signals from email text."""
    text_lower = text.lower()
    signals    = []

    urgency_hits = [p for p in URGENCY_PATTERNS if p in text_lower]
    if urgency_hits:
        signals.append(f"Urgency language detected: {', '.join(urgency_hits[:3])}")

    cred_hits = [p for p in CREDENTIAL_PATTERNS if p in text_lower]
    if cred_hits:
        signals.append(f"Credential harvesting patterns: {', '.join(cred_hits[:3])}")

    spoof_hits = [p for p in SENDER_SPOOF_PATTERNS if p in text_lower]
    if spoof_hits:
        signals.append(f"Sender spoofing signals: {', '.join(spoof_hits[:2])}")

    return signals


# ============================================================
# PINECONE RAG — PHISHING PATTERN RETRIEVAL
# ============================================================

def _retrieve_phishing_patterns(query_text: str, top_k: int = 5) -> list:
    """
    Retrieve semantically similar phishing patterns from Pinecone.
    Returns list of dicts with url, target, score.
    """
    embedder  = _get_embedder()
    embedding = embedder.encode(query_text).tolist()

    results = INDEX.query(
        vector=embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )

    patterns = []
    for match in results.matches:
        meta = match.metadata or {}
        patterns.append({
            "url":    meta.get("url", "")[:100],
            "target": meta.get("target", "Other"),
            "score":  round(match.score, 3),
            "tld":    meta.get("tld", ""),
        })

    return patterns


# ============================================================
# VERDICT COMPUTATION
# ============================================================

def _compute_verdict(
    vt:             Optional[VirusTotalResult],
    email_signals:  list,
    rag_patterns:   list,
    has_url:        bool,
) -> tuple[str, float, list]:
    """
    Compute verdict, confidence, and evidence list.

    Priority rules (from proposal Phase 2A):
    1. VirusTotal API unavailable → INSUFFICIENT_EVIDENCE
    2. High VT engine count (≥ 5) → MALICIOUS
    3. Moderate VT flags (2–4) or suspicious signals → UNCERTAIN
    4. Zero VT flags + no email signals → SAFE
    5. Email only (no URL) + strong signals → UNCERTAIN (cannot verify URL)
    """
    evidence = []

    # ── Case: VirusTotal unavailable ────────────────────────
    if vt and vt.total_engines == 0:
        evidence.append(
            "Live URL scanning temporarily unavailable — "
            "verdict based on pattern matching only"
        )
        if email_signals:
            evidence.extend(email_signals)
        # If Pinecone found strong phishing patterns, escalate to UNCERTAIN
        strong_rag = any(p.get("score", 0) >= 0.35 for p in rag_patterns)
        if strong_rag or len(email_signals) >= 2:
            evidence.append(
                f"Pattern analysis found {len(rag_patterns)} similar phishing patterns"
            )
            return "UNCERTAIN", 0.45, evidence
        return "INSUFFICIENT_EVIDENCE", 0.35, evidence

    # ── URL analysis path ────────────────────────────────────
    if vt:
        # Build VT evidence
        vt_summary = (
            f"VirusTotal: {vt.malicious_count}/{vt.total_engines} engines flagged"
        )
        evidence.append(vt_summary)

        if vt.domain_age_days is not None:
            if vt.domain_age_days <= 7:
                evidence.append(f"Domain registered {vt.domain_age_days} day(s) ago — very new")
            elif vt.domain_age_days <= 30:
                evidence.append(f"Domain registered {vt.domain_age_days} days ago — recently created")

        if vt.tld:
            evidence.append(f"TLD: .{vt.tld}")

    # Add email signals if present
    evidence.extend(email_signals)

    # Add RAG pattern context
    if rag_patterns:
        top_pattern = rag_patterns[0]
        if top_pattern["score"] >= 0.35:
            evidence.append(
                f"Similar phishing pattern found in database "
                f"(target: {top_pattern['target']}, similarity: {top_pattern['score']})"
            )

    # ── Verdict logic ────────────────────────────────────────
    if vt:
        if vt.malicious_count >= 5:
            # Strong malicious signal
            confidence = min(0.70 + (vt.malicious_count / vt.total_engines) * 0.25, 0.98)
            return "MALICIOUS", confidence, evidence

        elif vt.malicious_count >= 2 or vt.suspicious_count >= 3:
            # Moderate signal — uncertain
            confidence = 0.50 + (vt.malicious_count / 20)
            return "UNCERTAIN", min(confidence, 0.69), evidence

        elif vt.malicious_count == 0 and vt.suspicious_count == 0:
            # Clean — but check email signals
            if len(email_signals) >= 2:
                evidence.append("URL appears clean but email contains multiple phishing signals")
                return "UNCERTAIN", 0.55, evidence
            confidence = 0.85 - (len(email_signals) * 0.05)
            return "SAFE", max(confidence, 0.70), evidence

        else:
            # Mixed (1 malicious or 1-2 suspicious)
            return "UNCERTAIN", 0.55, evidence

    return "INSUFFICIENT_EVIDENCE", 0.30, evidence


# ============================================================
# SYNTHESIS PROMPT
# ============================================================

PHISHING_SYNTHESIS_PROMPT = """You are a cybersecurity analyst producing a plain-English phishing verdict.

Given the analysis data below, produce a clear explanation for a non-technical user.

Verdict: {verdict}
Confidence: {confidence}
Evidence: {evidence}
URL analyzed: {url}
VirusTotal: {vt_summary}
Email signals found: {email_signals}
Similar phishing patterns: {rag_count} found in database

Write 2-3 sentences in plain English that:
1. State clearly whether the URL/email is dangerous or safe
2. Explain the key evidence in simple terms (no jargon)
3. For MALICIOUS/UNCERTAIN: warn the user not to click or submit information

Return ONLY a JSON object:
{{
  "plain_english_summary": "your explanation here",
  "key_evidence_plain": ["evidence point 1", "evidence point 2"],
  "immediate_action": "single most important thing the user should do right now"
}}"""


def _run_synthesis_prompt(
    verdict:       str,
    confidence:    float,
    evidence:      list,
    url:           Optional[str],
    vt:            Optional[VirusTotalResult],
    rag_patterns:  list,
    email_signals: list,
) -> dict:
    """Generate plain-English explanation using GPT-4o mini."""
    vt_summary = "Not available"
    if vt and vt.total_engines > 0:
        vt_summary = (
            f"{vt.malicious_count} of {vt.total_engines} engines flagged"
        )

    prompt = PHISHING_SYNTHESIS_PROMPT.format(
        verdict=verdict,
        confidence=f"{confidence:.0%}",
        evidence="\n".join(f"- {e}" for e in evidence),
        url=url or "No URL — email text analysis",
        vt_summary=vt_summary,
        email_signals=", ".join(email_signals) if email_signals else "None",
        rag_count=len(rag_patterns),
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "plain_english_summary": f"Analysis complete. Verdict: {verdict}.",
            "key_evidence_plain": evidence[:3],
            "immediate_action": "Do not click any links until further verification." if verdict in ("MALICIOUS", "UNCERTAIN") else "This appears safe.",
        }


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def run_phishing_agent(router_result: RouterResult) -> PhishingAgentResult:
    """
    Run the phishing agent on a router result.

    Args:
        router_result: Output from route_query()

    Returns:
        PhishingAgentResult with verdict, confidence, evidence, and synthesis
    """
    # Not activated if router didn't dispatch to phishing
    if not router_result.needs_phishing_agent():
        return _not_activated()

    url        = router_result.extracted_url
    email_text = router_result.email_text or router_result.raw_input
    has_url    = url is not None

    # ── Step 1: VirusTotal scan ──────────────────────────────
    vt_result = None
    if has_url:
        import concurrent.futures

        def _run_vt_in_thread(url: str):
            """Run async VT scan in isolated thread with its own event loop."""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(scan_url(url))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future    = executor.submit(_run_vt_in_thread, url)
            vt_result = future.result(timeout=30)

    # ── Step 2: Email signal extraction ─────────────────────
    email_signals = _extract_email_signals(email_text) if email_text else []

    # ── Step 3: Pinecone RAG retrieval ──────────────────────
    query_text   = url if has_url else email_text[:200]
    rag_patterns = _retrieve_phishing_patterns(query_text)

    # ── Step 4: Compute verdict ──────────────────────────────
    verdict, confidence, evidence = _compute_verdict(
        vt=vt_result,
        email_signals=email_signals,
        rag_patterns=rag_patterns,
        has_url=has_url,
    )

    # ── Step 5: Synthesis prompt ─────────────────────────────
    synthesis = _run_synthesis_prompt(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        url=url,
        vt=vt_result,
        rag_patterns=rag_patterns,
        email_signals=email_signals,
    )

    # ── Step 6: Build source citations ──────────────────────
    citations = []
    if vt_result and vt_result.total_engines > 0:
        citations.append(
            f"VirusTotal scan: {vt_result.malicious_count}/{vt_result.total_engines} engines"
        )
    if rag_patterns:
        citations.append(f"PhishTank database: {len(rag_patterns)} similar patterns retrieved")

    # ── Step 7: Attach synthesis to result ───────────────────
    evidence_with_synthesis = evidence.copy()
    if synthesis.get("plain_english_summary"):
        evidence_with_synthesis.append(
            f"Summary: {synthesis['plain_english_summary']}"
        )

    return PhishingAgentResult(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        url_analyzed=url,
        vt_result=vt_result.to_dict() if vt_result else None,
        rag_patterns=rag_patterns[:3],
        email_signals=email_signals,
        source_citations=citations,
        error=None,
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    from router import route_query

    test_inputs = [
        # Should be MALICIOUS
        "Check this link: http://paypal-secure-verify.xyz/login/confirm",
        # Should be MALICIOUS
        "http://apple-id-verify.tk/account/suspended",
        # Should be SAFE
        "https://www.google.com",
        # Should be UNCERTAIN (email only, no URL)
        "Dear customer, your account has been suspended. Click here immediately to verify your identity or lose access within 24 hours.",
        # Should be SAFE
        "https://northeastern.edu",
    ]

    print("=" * 65)
    print("Phishing Agent Test")
    print("=" * 65)

    for inp in test_inputs:
        router_result = route_query(inp)
        result        = run_phishing_agent(router_result)

        print(f"\nInput:    {inp[:60]}...")
        print(f"Verdict:  {result.verdict}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Evidence:")
        for e in result.evidence:
            print(f"  • {e}")
        if result.rag_patterns:
            print(f"RAG patterns: {len(result.rag_patterns)} similar phishing URLs found")
        print(f"Citations: {result.source_citations}")
        print("-" * 65)