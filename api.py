"""
api.py
Personal Cybersecurity Guardian Agent

FastAPI backend — wraps the main.py pipeline as a REST API.
Called by the Streamlit frontend via HTTP requests.

Endpoints:
    POST /analyze        ← Main analysis endpoint
    GET  /health         ← Health check for Render.com
    GET  /               ← Root info

Runtime fallback (proposal Phase 5, Risk 2):
    If VirusTotal API is unavailable, the pipeline continues
    and returns INSUFFICIENT_EVIDENCE rather than failing.
    No silent degradation — the response always includes
    a status field indicating mock vs live mode.

Usage:
    # Local development
    uvicorn api:app --reload --port 8000

    # Production (Render.com)
    uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from main import analyze, GuardianResult

load_dotenv()

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title       = "Personal Cybersecurity Guardian Agent",
    description = "Multi-agent cybersecurity analysis API",
    version     = "1.0.0",
)

# CORS — allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

MOCK_MODE = os.getenv("MOCK_VIRUSTOTAL", "true").lower() == "true"


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""
    user_input: str = Field(
        ...,
        min_length  = 1,
        max_length  = 5000,
        description = "URL, email text, or software version query",
        json_schema_extra = {"example": "Is Zoom 5.15.2 vulnerable?"}
    )
    user_already_clicked: bool = Field(
        default     = False,
        description = "True if user already clicked the suspicious link"
    )


class AgentFinding(BaseModel):
    verdict:    Optional[str]
    confidence: Optional[float]
    evidence:   list


class CVEItem(BaseModel):
    cve_id:             str
    cvss_score:         float
    cvss_severity:      str
    description:        str
    actively_exploited: bool
    source:             str


class InterventionModel(BaseModel):
    situation:    Optional[str]
    urgency:      Optional[str]
    steps:        list
    reassurance:  Optional[str]


class AnalyzeResponse(BaseModel):
    """Response body from /analyze endpoint."""
    # Core verdict
    verdict:            str
    confidence:         float
    risk_score:         int

    # Agents
    agents_activated:   list

    # Per-agent findings
    phishing_verdict:    Optional[str]
    phishing_confidence: Optional[float]
    phishing_evidence:   list

    vuln_verdict:        Optional[str]
    vuln_confidence:     Optional[float]
    vuln_evidence:       list
    vuln_cves:           list

    # Synthesis outputs
    summary:            str
    actions:            list
    intervention:       dict
    citations:          list

    # Routing
    intent:             str
    extracted_url:      Optional[str]
    software:           Optional[str]
    version:            Optional[str]
    clarification:      Optional[str]

    # Metadata
    processing_time_ms: int
    timestamp:          str
    mock_mode:          bool


class HealthResponse(BaseModel):
    status:     str
    mock_mode:  bool
    version:    str
    timestamp:  str


class ErrorResponse(BaseModel):
    error:      str
    detail:     Optional[str]
    timestamp:  str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint — API info."""
    return {
        "name":        "Personal Cybersecurity Guardian Agent",
        "version":     "1.0.0",
        "description": "Multi-agent cybersecurity analysis for non-technical users",
        "endpoints": {
            "analyze": "POST /analyze",
            "health":  "GET /health",
        },
        "mock_mode": MOCK_MODE,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    Called by Render.com to verify the service is running.
    Returns mock_mode status so frontend can show appropriate notice.
    """
    from datetime import datetime, timezone
    return HealthResponse(
        status    = "ok",
        mock_mode = MOCK_MODE,
        version   = "1.0.0",
        timestamp = datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/analyze",
    response_model = AnalyzeResponse,
    tags           = ["Analysis"],
    summary        = "Analyze a URL, email, or software version",
    description    = """
Run the full Guardian Agent pipeline on user input.

**Input types handled:**
- URL: `http://suspicious-domain.xyz/login`
- Email text: paste the full email body
- Software version: `Is Zoom 5.15.2 vulnerable?`
- Combined: URL + software version

**Verdicts returned:**
- `MALICIOUS` — URL confirmed phishing by VirusTotal + pattern matching
- `VULNERABLE` — Software version has known CVEs (CISA KEV or NVD)
- `SAFE` — No threats detected
- `UNCERTAIN` — Mixed signals, proceed with caution
- `INSUFFICIENT_EVIDENCE` — Not enough data for a verdict
- `CLARIFY` — Need more information (e.g. full version number)
    """,
)
async def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Main analysis endpoint.
    Runs the complete pipeline: router → agents → synthesis → intervention.
    """
    try:
        result: GuardianResult = analyze(
            user_input           = request.user_input.strip(),
            user_already_clicked = request.user_already_clicked,
        )

        return AnalyzeResponse(
            verdict             = result.verdict,
            confidence          = result.confidence,
            risk_score          = result.risk_score,
            agents_activated    = result.agents_activated,
            phishing_verdict    = result.phishing_verdict,
            phishing_confidence = result.phishing_confidence,
            phishing_evidence   = result.phishing_evidence,
            vuln_verdict        = result.vuln_verdict,
            vuln_confidence     = result.vuln_confidence,
            vuln_evidence       = result.vuln_evidence,
            vuln_cves           = result.vuln_cves,
            summary             = result.summary,
            actions             = result.actions,
            intervention        = result.intervention,
            citations           = result.citations,
            intent              = result.intent,
            extracted_url       = result.extracted_url,
            software            = result.software,
            version             = result.version,
            clarification       = result.clarification,
            processing_time_ms  = result.processing_time_ms,
            timestamp           = result.timestamp,
            mock_mode           = MOCK_MODE,
        )

    except ValueError as e:
        # Input validation error
        raise HTTPException(
            status_code = 422,
            detail      = str(e)
        )

    except Exception as e:
        # Unexpected error — log and return safe error response
        print(f"[API ERROR] {type(e).__name__}: {e}")
        print(traceback.format_exc())

        # Runtime fallback per proposal Phase 5 Risk 2:
        # Never crash silently — return structured error
        raise HTTPException(
            status_code = 500,
            detail      = (
                "Analysis temporarily unavailable. "
                "Please try again in a moment."
            )
        )


# ============================================================
# SESSION RATE LIMITING (proposal Phase 4 abuse prevention)
# ============================================================

from collections import defaultdict
from datetime import datetime, timezone

_session_counts: dict = defaultdict(lambda: {"count": 0, "reset_time": None})
MAX_QUERIES = int(os.getenv("MAX_QUERIES_PER_SESSION", "10"))


def _check_rate_limit(session_id: str) -> bool:
    """
    Simple in-memory rate limiter.
    Max 10 URL queries per session (proposal Phase 4).
    Resets every hour.
    """
    now = datetime.now(timezone.utc)
    session = _session_counts[session_id]

    if session["reset_time"] is None or now > session["reset_time"]:
        session["count"]      = 0
        session["reset_time"] = datetime(
            now.year, now.month, now.day,
            now.hour + 1, 0, 0,
            tzinfo=timezone.utc
        )

    session["count"] += 1
    return session["count"] <= MAX_QUERIES


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting Cybersecurity Guardian API...")
    print(f"Mock mode: {MOCK_MODE}")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")

    uvicorn.run(
        "api:app",
        host     = os.getenv("BACKEND_HOST", "0.0.0.0"),
        port     = int(os.getenv("BACKEND_PORT", "8000")),
        reload   = True,
        log_level= "info",
    )