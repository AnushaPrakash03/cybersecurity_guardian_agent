"""
router.py
Personal Cybersecurity Guardian Agent

LangGraph intent classification router.
Classifies user input and dispatches to the correct agent(s).

Routing table (from proposal Phase 2A):
  URL only                    → Phishing Agent only
  Email text (± embedded URL) → Phishing Agent only (URL extracted)
  Software name + version     → Vulnerability Agent only
  URL + software mention      → Both agents
  Ambiguous                   → Clarification prompt, no dispatch

Fast-path version detection:
  If regex finds a complete version string (X.Y.Z) AND the LLM
  confirms vulnerability intent, the router bypasses the confidence
  threshold to avoid false CLARIFY returns on casual phrasing like
  "I have Log4j 2.14.1 installed".

Usage:
    from router import route_query
    result = route_query("Is Zoom 5.15.2 vulnerable?")
    print(result.intent)        # "vulnerability"
    print(result.software)      # "Zoom"
    print(result.version)       # "5.15.2"
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.75"))


# ============================================================
# OUTPUT SCHEMA
# ============================================================

@dataclass
class RouterResult:
    """
    Result of the intent classification router.
    Consumed by the LangGraph state machine to decide
    which agents to activate.
    """
    intent:          str
    confidence:      float
    extracted_url:   Optional[str]
    software:        Optional[str]
    version:         Optional[str]
    email_text:      Optional[str]
    clarification:   Optional[str]
    raw_input:       str
    routing_reason:  str

    def needs_phishing_agent(self) -> bool:
        return self.intent in ("phishing", "both")

    def needs_vulnerability_agent(self) -> bool:
        return self.intent in ("vulnerability", "both")

    def needs_clarification(self) -> bool:
        return self.intent == "clarify"

    def has_complete_version(self) -> bool:
        if not self.version:
            return False
        parts = self.version.strip().split(".")
        return len(parts) >= 3

    def to_dict(self) -> dict:
        return {
            "intent":         self.intent,
            "confidence":     self.confidence,
            "extracted_url":  self.extracted_url,
            "software":       self.software,
            "version":        self.version,
            "email_text":     self.email_text,
            "clarification":  self.clarification,
            "raw_input":      self.raw_input,
            "routing_reason": self.routing_reason,
        }


# ============================================================
# URL EXTRACTION
# ============================================================

URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+'
    r'|(?:www\.)[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

def extract_url(text: str) -> Optional[str]:
    matches = URL_PATTERN.findall(text)
    if not matches:
        return None
    url = matches[0].rstrip(".,;:!?)")
    if not url.startswith("http"):
        url = "http://" + url
    return url


# ============================================================
# VERSION EXTRACTION
# ============================================================

VERSION_PATTERN = re.compile(
    r'\b(\d+\.\d+(?:\.\d+)*(?:[._-]\w+)?)\b'
)

def extract_version(text: str) -> Optional[str]:
    matches = VERSION_PATTERN.findall(text)
    if not matches:
        return None
    return max(matches, key=len)


# ============================================================
# STOP WORDS FOR FAST-PATH SOFTWARE DETECTION
# ============================================================

STOP_WORDS = {
    "the", "and", "for", "have", "with", "installed", "running",
    "using", "check", "is", "my", "i", "it", "this", "any", "are",
    "was", "were", "has", "had", "not", "but", "from", "that",
    "which", "what", "when", "where", "who", "how", "why", "can",
    "could", "should", "would", "will", "may", "might", "must",
    "does", "did", "do", "be", "been", "being", "get", "got",
    "there", "their", "they", "them", "then", "than", "also",
    "just", "more", "some", "such", "into", "over", "after",
    "about", "want", "know", "need", "help", "please", "think",
    "version", "update", "patch", "fix", "bug", "issue", "problem",
    "vulnerable", "vulnerability", "exploit", "attack", "security",
    "software", "application", "app", "program", "system", "tool",
    "worried", "concern", "safe", "dangerous", "risk",
}


# ============================================================
# INTENT CLASSIFICATION (LLM)
# ============================================================

ROUTER_SYSTEM_PROMPT = """You are a cybersecurity query intent classifier.

Classify the user's input into one of these intents:
- "phishing": User submitted a URL to check, email text to analyze, or is asking if a link/email is safe
- "vulnerability": User submitted a software name and version to check for known vulnerabilities
- "both": User submitted both a URL AND mentioned specific software with a version number
- "clarify": Input is too ambiguous to classify confidently

Extract any structured data present:
- url: any URL or link in the input (include full URL with http/https if present)
- software: software/application name if vulnerability query
- version: version number if vulnerability query (exact string as provided)
- is_email: true if the input looks like email text being analyzed for phishing

IMPORTANT: These phrasings all indicate a vulnerability query — classify as "vulnerability":
- "I have [software] [version] installed"
- "I'm running [software] [version]"
- "using [software] [version]"
- "[software] [version] — any CVEs?"
- "is [software] [version] patched?"
- "should I be worried about [software] [version]?"
- "I have [software] [version]" (even without a question)

Return ONLY valid JSON with this exact schema:
{
  "intent": "phishing" | "vulnerability" | "both" | "clarify",
  "confidence": 0.0-1.0,
  "url": "string or null",
  "software": "string or null",
  "version": "string or null",
  "is_email": true | false,
  "reason": "one sentence explaining routing decision"
}

Rules:
- If URL present AND software+version present → "both"
- If only URL or email text → "phishing"
- If software mentioned but NO version → "clarify" (version required)
- If completely unclear → "clarify"
- Confidence < 0.75 → consider "clarify"
- Never fabricate a version number if not explicitly provided
- If a version number is clearly visible in the input (e.g. 2.14.1), confidence should be ≥ 0.85"""


def _classify_with_llm(user_input: str) -> dict:
    """Use GPT-4o mini to classify intent and extract structured data."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_input}
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {
            "intent": "clarify", "confidence": 0.0,
            "url": None, "software": None, "version": None,
            "is_email": False, "reason": f"Parse error: {e}"
        }
    except Exception as e:
        return {
            "intent": "clarify", "confidence": 0.0,
            "url": None, "software": None, "version": None,
            "is_email": False, "reason": f"LLM error: {e}"
        }


# ============================================================
# CLARIFICATION MESSAGES
# ============================================================

def _build_clarification(llm_result: dict, user_input: str) -> str:
    software = llm_result.get("software")
    version  = llm_result.get("version")

    if software and not version:
        return (
            f"I can check {software} for known vulnerabilities, but I need "
            f"the exact version number including the patch level "
            f"(for example: {software} 5.15.2, not just 5.15). "
            f"What version are you running?"
        )

    if software and version and len(version.split(".")) < 3:
        return (
            f"I found version {version} for {software}, but I need the full "
            f"version including the patch number (e.g. {version}.0 or {version}.x) "
            f"to give you an accurate vulnerability check. "
            f"What is the complete version?"
        )

    return (
        "I can help you with two things: "
        "(1) Check if a URL or email is a phishing attempt — paste the link or email text, "
        "(2) Check if your software has known vulnerabilities — tell me the software name "
        "and exact version (e.g. 'Zoom 5.15.2'). "
        "What would you like to check?"
    )


# ============================================================
# RESULT BUILDER
# ============================================================

def _build_router_result(
    user_input:    str,
    llm_result:    dict,
    regex_url:     Optional[str],
    regex_version: Optional[str],
) -> RouterResult:
    """Build RouterResult from LLM output with confidence and version checks."""
    intent     = llm_result.get("intent",     "clarify")
    confidence = float(llm_result.get("confidence", 0.0))
    llm_url    = llm_result.get("url")
    software   = llm_result.get("software")
    version    = llm_result.get("version") or regex_version
    is_email   = llm_result.get("is_email",   False)
    reason     = llm_result.get("reason",     "")

    extracted_url = regex_url or llm_url

    # Confidence threshold check
    if confidence < CONFIDENCE_THRESHOLD and intent != "clarify":
        intent = "clarify"
        reason = f"Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"

    # Version completeness enforcement
    if intent in ("vulnerability", "both") and software and version:
        parts = version.strip().split(".")
        if len(parts) < 3:
            intent = "clarify"
            reason = f"Incomplete version '{version}' — patch level required"
    elif intent in ("vulnerability", "both") and software and not version:
        intent = "clarify"
        reason = f"Software '{software}' found but no version provided"

    clarification = None
    if intent == "clarify":
        clarification = _build_clarification(llm_result, user_input)

    email_text = None
    if is_email or (
        intent == "phishing"
        and len(user_input) > 100
        and not extracted_url
    ):
        email_text = user_input

    return RouterResult(
        intent=intent,
        confidence=confidence,
        extracted_url=extracted_url,
        software=software,
        version=version,
        email_text=email_text,
        clarification=clarification,
        raw_input=user_input,
        routing_reason=reason,
    )


# ============================================================
# MAIN ROUTER
# ============================================================

def route_query(user_input: str) -> RouterResult:
    """
    Classify user input and return routing decision.

    Pipeline:
    1. Regex URL + version extraction
    2. Fast-path: if complete version detected + no URL →
       run LLM and boost confidence if it agrees (vulnerability intent)
       This handles casual phrasing like "I have Log4j 2.14.1 installed"
    3. Standard LLM classification for all other cases
    4. Confidence + version completeness checks
    5. Build RouterResult
    """
    user_input = user_input.strip()

    if not user_input:
        return RouterResult(
            intent="clarify", confidence=0.0,
            extracted_url=None, software=None, version=None,
            email_text=None,
            clarification=(
                "I'm ready to help. Paste a suspicious URL or email, "
                "or ask me to check a specific software version."
            ),
            raw_input=user_input,
            routing_reason="Empty input",
        )

    # ── Step 1: Regex extraction ─────────────────────────────
    regex_url     = extract_url(user_input)
    regex_version = extract_version(user_input)

    # ── Step 2: Fast-path for explicit version + no URL ──────
    if (regex_version
            and len(regex_version.split(".")) >= 3
            and not regex_url):

        # Find non-stop-word tokens as software name candidates
        tokens = user_input.lower().split()
        software_candidates = [
            t.strip(".,;:!?()")
            for t in tokens
            if t.strip(".,;:!?()") not in STOP_WORDS
            and len(t.strip(".,;:!?()")) > 1
            and not re.match(r'^\d+[\.\d]*$', t)
        ]

        if software_candidates:
            llm_result = _classify_with_llm(user_input)
            llm_intent = llm_result.get("intent", "clarify")
            llm_conf   = float(llm_result.get("confidence", 0.0))
            llm_sw     = llm_result.get("software")
            llm_ver    = llm_result.get("version") or regex_version

            if llm_intent == "vulnerability" and llm_sw:
                # Boost confidence when version is explicitly present
                boosted = max(llm_conf, 0.80)
                return RouterResult(
                    intent="vulnerability",
                    confidence=boosted,
                    extracted_url=None,
                    software=llm_sw,
                    version=llm_ver,
                    email_text=None,
                    clarification=None,
                    raw_input=user_input,
                    routing_reason=(
                        f"Fast-path: version '{regex_version}' detected by regex, "
                        f"LLM confirmed vulnerability intent "
                        f"(confidence boosted from {llm_conf:.2f} to {boosted:.2f})"
                    ),
                )

            # LLM disagrees — use its result with normal flow
            return _build_router_result(
                user_input=user_input,
                llm_result=llm_result,
                regex_url=regex_url,
                regex_version=regex_version,
            )

    # ── Step 3: Standard LLM classification ─────────────────
    llm_result = _classify_with_llm(user_input)

    return _build_router_result(
        user_input=user_input,
        llm_result=llm_result,
        regex_url=regex_url,
        regex_version=regex_version,
    )


# ============================================================
# ROUTER TEST
# ============================================================

def test_router():
    test_cases = [
        # URL only → phishing
        {"input": "http://paypal-secure-verify.xyz/login",                                                   "expected_intent": "phishing",       "category": "URL only"},
        {"input": "can you check this link: https://apple-id-verify.tk/account/suspended",                  "expected_intent": "phishing",       "category": "URL only"},
        {"input": "is this safe to click? www.microsoft-verify.online/password-reset",                      "expected_intent": "phishing",       "category": "URL only"},
        {"input": "https://netflix-update-payment-details.vercel.app/",                                     "expected_intent": "phishing",       "category": "URL only"},
        {"input": "check this URL for me: http://bank-secure-login.xyz/verify",                             "expected_intent": "phishing",       "category": "URL only"},
        # Email text → phishing
        {"input": "Dear customer, your account has been suspended. Click here to verify your identity immediately or lose access within 24 hours. This is urgent.", "expected_intent": "phishing", "category": "Email text"},
        {"input": "Your PayPal account has been limited. We noticed unusual activity. Please confirm your information at the link below to restore access.",         "expected_intent": "phishing", "category": "Email text"},
        {"input": "IT Department: Your password expires in 24 hours. Click here: http://it-support-verify.online/reset",                                            "expected_intent": "phishing", "category": "Email text"},
        {"input": "URGENT: IRS Notice - You have an unclaimed tax refund of $1,247. Submit your information at the link to claim it before it expires.",            "expected_intent": "phishing", "category": "Email text"},
        {"input": "Your Amazon order #114-2847562 has a problem. Verify your payment method to avoid cancellation.",                                                 "expected_intent": "phishing", "category": "Email text"},
        # Software + version → vulnerability
        {"input": "Is Zoom 5.15.2 vulnerable to any known exploits?",      "expected_intent": "vulnerability", "category": "Software + version"},
        {"input": "I have Log4j 2.14.1 installed, should I be worried?",   "expected_intent": "vulnerability", "category": "Software + version"},
        {"input": "I have Log4j 2.14.1 installed",                         "expected_intent": "vulnerability", "category": "Software + version"},
        {"input": "Apache HTTP Server 2.4.49 - any CVEs?",                 "expected_intent": "vulnerability", "category": "Software + version"},
        {"input": "running VMware vCenter 7.0.3, is it patched?",          "expected_intent": "vulnerability", "category": "Software + version"},
        # URL + software → both
        {"input": "I got this suspicious link http://paypal-verify.xyz/login and I'm also running Zoom 5.15.2",                           "expected_intent": "both",    "category": "URL + software"},
        {"input": "Check this URL: https://apple-id.tk/verify and also check if my Chrome 119.0.6045.105 is safe",                       "expected_intent": "both",    "category": "URL + software"},
        # Ambiguous → clarify
        {"input": "check my zoom",        "expected_intent": "clarify", "category": "Ambiguous"},
        {"input": "is this safe",         "expected_intent": "clarify", "category": "Ambiguous"},
        {"input": "I think I got phished","expected_intent": "clarify", "category": "Ambiguous"},
    ]

    print("\n" + "=" * 65)
    print("LangGraph Router Test")
    print("=" * 65)

    results_by_category: dict = {}
    total_passed = 0

    for tc in test_cases:
        result   = route_query(tc["input"])
        passed   = result.intent == tc["expected_intent"]
        category = tc["category"]
        status   = "✅" if passed else "❌"
        if passed:
            total_passed += 1

        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "total": 0}
        results_by_category[category]["total"]  += 1
        results_by_category[category]["passed"] += 1 if passed else 0

        short_input = tc["input"][:55] + "..." if len(tc["input"]) > 55 else tc["input"]
        print(f"\n{status} [{category}]")
        print(f"   Input:    {short_input}")
        print(f"   Expected: {tc['expected_intent']:<15} Got: {result.intent}")
        print(f"   Confidence: {result.confidence:.2f} | Reason: {result.routing_reason[:60]}")
        if result.software:
            print(f"   Software: {result.software} | Version: {result.version}")
        if result.clarification:
            print(f"   Clarification: {result.clarification[:80]}...")

    print("\n" + "=" * 65)
    print("RESULTS BY CATEGORY:")
    for category, counts in results_by_category.items():
        p   = counts["passed"]
        t   = counts["total"]
        bar = "✅" * p + "❌" * (t - p)
        print(f"  {category:<20} {p}/{t}  {bar}")

    print(f"\nOverall: {total_passed}/{len(test_cases)}")
    target_pct = 0.85
    status = "PASS" if total_passed / len(test_cases) >= target_pct else "FAIL"
    print(f"Target (≥ {target_pct:.0%}): {status}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    test_router()