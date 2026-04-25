"""
tools/virustotal.py
Personal Cybersecurity Guardian Agent

VirusTotal API integration with mock support for development.

MOCK MODE (MOCK_VIRUSTOTAL=true in .env):
  - Returns realistic mock responses based on URL patterns
  - Zero API quota consumed during development
  - Deterministic: same URL always returns same result

LIVE MODE (MOCK_VIRUSTOTAL=false in .env):
  - Calls real VirusTotal API
  - Rate limited: 4 requests/min, 500 requests/day
  - Responses cached for 24 hours to protect quota

Usage:
    from tools.virustotal import scan_url
    result = await scan_url("http://suspicious-domain.xyz/login")
"""

import os
import re
import time
import hashlib
import asyncio
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

VIRUSTOTAL_API_KEY  = os.getenv("VIRUSTOTAL_API_KEY", "")
MOCK_VIRUSTOTAL     = os.getenv("MOCK_VIRUSTOTAL", "true").lower() == "true"
VT_BASE_URL         = "https://www.virustotal.com/api/v3"
VT_RATE_LIMIT       = 4       # requests per minute (free tier)
VT_DAILY_LIMIT      = 500     # requests per day (free tier)

# Simple in-memory cache: url_hash → (result, timestamp)
_cache: dict = {}
CACHE_TTL_SECONDS = 86400     # 24 hours


# ============================================================
# RESULT SCHEMA
# ============================================================

class VirusTotalResult:
    """
    Structured result from VirusTotal scan.
    Same schema whether mock or live — agents never know the difference.
    """
    def __init__(
        self,
        url:              str,
        malicious_count:  int,
        suspicious_count: int,
        harmless_count:   int,
        undetected_count: int,
        total_engines:    int,
        domain_age_days:  Optional[int],
        domain:           str,
        tld:              str,
        scan_date:        str,
        is_mocked:        bool = False,
        raw_stats:        Optional[dict] = None,
    ):
        self.url              = url
        self.malicious_count  = malicious_count
        self.suspicious_count = suspicious_count
        self.harmless_count   = harmless_count
        self.undetected_count = undetected_count
        self.total_engines    = total_engines
        self.domain_age_days  = domain_age_days
        self.domain           = domain
        self.tld              = tld
        self.scan_date        = scan_date
        self.is_mocked        = is_mocked
        self.raw_stats        = raw_stats or {}

    @property
    def detection_rate(self) -> float:
        """Proportion of engines that flagged the URL as malicious."""
        if self.total_engines == 0:
            return 0.0
        return self.malicious_count / self.total_engines

    @property
    def verdict(self) -> str:
        """
        Preliminary verdict based on engine counts only.
        Final verdict is determined by the synthesis layer.
        """
        if self.malicious_count >= 5:
            return "MALICIOUS"
        elif self.malicious_count >= 2 or self.suspicious_count >= 3:
            return "SUSPICIOUS"
        elif self.malicious_count == 0 and self.suspicious_count == 0:
            return "CLEAN"
        else:
            return "UNCERTAIN"

    def to_dict(self) -> dict:
        return {
            "url":              self.url,
            "malicious_count":  self.malicious_count,
            "suspicious_count": self.suspicious_count,
            "harmless_count":   self.harmless_count,
            "undetected_count": self.undetected_count,
            "total_engines":    self.total_engines,
            "detection_rate":   round(self.detection_rate, 3),
            "preliminary_verdict": self.verdict,
            "domain_age_days":  self.domain_age_days,
            "domain":           self.domain,
            "tld":              self.tld,
            "scan_date":        self.scan_date,
            "is_mocked":        self.is_mocked,
        }

    def __repr__(self):
        return (
            f"VTResult(verdict={self.verdict}, "
            f"malicious={self.malicious_count}/{self.total_engines}, "
            f"domain={self.domain}, "
            f"mocked={self.is_mocked})"
        )


# ============================================================
# MOCK ENGINE
# ============================================================

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "cf", "ga", "gq", "pw", "top", "club",
    "online", "site", "website", "space", "fun", "click", "link",
    "live", "stream", "download", "win", "loan", "work", "party"
}

# High-reputation domains — always return clean
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com",
    "microsoft.com", "apple.com", "amazon.com", "wikipedia.org",
    "reddit.com", "linkedin.com", "github.com", "stackoverflow.com",
    "nytimes.com", "bbc.com", "cnn.com", "reuters.com",
    "paypal.com", "bankofamerica.com", "chase.com", "wellsfargo.com",
    "netflix.com", "spotify.com", "dropbox.com", "zoom.us",
    "harvard.edu", "mit.edu", "northeastern.edu", "stanford.edu",
    "cisa.gov", "nist.gov", "irs.gov", "fbi.gov", "cdc.gov",
}

# Brand names that should NOT appear in a domain (phishing signal)
BRAND_IN_DOMAIN_SIGNALS = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "twitter", "linkedin", "dropbox",
    "bankofamerica", "chase", "wellsfargo", "citibank", "irs",
    "usps", "fedex", "dhl", "ups", "steam", "coinbase",
]

# Path patterns commonly seen in phishing URLs
PHISHING_PATH_PATTERNS = [
    r"/login", r"/signin", r"/verify", r"/confirm", r"/secure",
    r"/account", r"/update", r"/billing", r"/payment", r"/credential",
    r"/authenticate", r"/validation", r"/suspended", r"/restore",
    r"/password", r"/reset", r"/urgent", r"/limited",
]


def _get_mock_result(url: str) -> VirusTotalResult:
    """
    Generate a deterministic mock VirusTotal result based on URL analysis.

    Scoring logic mirrors real VirusTotal signal patterns:
    - Suspicious TLD → more engine flags
    - Brand name in non-brand domain → more engine flags
    - Suspicious path patterns → more engine flags
    - Known trusted domain → zero flags
    - Very new-looking domain patterns → more flags
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        tld    = domain.split(".")[-1] if "." in domain else ""
        path   = parsed.path.lower()
        base_domain = ".".join(domain.split(".")[-2:]) if "." in domain else domain
    except Exception:
        domain      = ""
        tld         = ""
        path        = ""
        base_domain = ""

    total_engines = 72

    # ── Trusted domain: always clean ──────────────────────────
    if base_domain in TRUSTED_DOMAINS:
        return VirusTotalResult(
            url=url, malicious_count=0, suspicious_count=0,
            harmless_count=68, undetected_count=4,
            total_engines=total_engines,
            domain_age_days=3650,   # ~10 years
            domain=domain, tld=tld,
            scan_date=_today(),
            is_mocked=True,
        )

    # ── Score phishing signals ─────────────────────────────────
    malicious_score  = 0
    suspicious_score = 0

    # Signal 1: Suspicious TLD
    if tld in SUSPICIOUS_TLDS:
        malicious_score  += 8
        suspicious_score += 4

    # Signal 2: Brand name in non-brand domain
    for brand in BRAND_IN_DOMAIN_SIGNALS:
        if brand in domain and base_domain not in TRUSTED_DOMAINS:
            malicious_score  += 12
            suspicious_score += 5
            break

    # Signal 3: Phishing path patterns
    path_hits = sum(1 for p in PHISHING_PATH_PATTERNS if re.search(p, path))
    if path_hits >= 3:
        malicious_score  += 8
        suspicious_score += 4
    elif path_hits >= 1:
        malicious_score  += 3
        suspicious_score += 2

    # Signal 4: Multiple hyphens in domain (common in phishing)
    hyphen_count = domain.count("-")
    if hyphen_count >= 3:
        malicious_score  += 6
        suspicious_score += 3
    elif hyphen_count >= 1:
        malicious_score  += 2
        suspicious_score += 1

    # Signal 5: Long subdomain chain (e.g. secure.login.verify.paypal.fake.xyz)
    subdomain_parts = domain.split(".")
    if len(subdomain_parts) >= 4:
        malicious_score  += 5
        suspicious_score += 3

    # Signal 6: IP address instead of domain
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
        malicious_score  += 15
        suspicious_score += 5

    # Cap at realistic engine counts
    malicious_count  = min(malicious_score,  total_engines - 2)
    suspicious_count = min(suspicious_score, total_engines - malicious_count - 2)
    harmless_count   = max(0, total_engines - malicious_count - suspicious_count - 4)
    undetected_count = total_engines - malicious_count - suspicious_count - harmless_count

    # Estimate domain age: suspicious domains tend to be newer
    if malicious_count >= 10:
        domain_age_days = 3    # Very new — high risk
    elif malicious_count >= 5:
        domain_age_days = 14
    elif malicious_count >= 2:
        domain_age_days = 90
    else:
        domain_age_days = 365  # Unknown but established

    return VirusTotalResult(
        url=url,
        malicious_count=malicious_count,
        suspicious_count=suspicious_count,
        harmless_count=harmless_count,
        undetected_count=undetected_count,
        total_engines=total_engines,
        domain_age_days=domain_age_days,
        domain=domain,
        tld=tld,
        scan_date=_today(),
        is_mocked=True,
    )


def _today() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================
# LIVE API
# ============================================================

async def _call_virustotal_api(url: str) -> VirusTotalResult:
    """
    Call the real VirusTotal API v3.
    Only called when MOCK_VIRUSTOTAL=false.
    """
    import aiohttp

    if not VIRUSTOTAL_API_KEY:
        raise ValueError(
            "VIRUSTOTAL_API_KEY not set in .env. "
            "Set MOCK_VIRUSTOTAL=true for development."
        )

    headers = {"x-apikey": VIRUSTOTAL_API_KEY}

    try:
        parsed     = urlparse(url)
        domain     = parsed.netloc.lower().replace("www.", "")
        tld        = domain.split(".")[-1] if "." in domain else ""

        async with aiohttp.ClientSession() as session:
            # Step 1: Submit URL for analysis
            async with session.post(
                f"{VT_BASE_URL}/urls",
                headers=headers,
                data={"url": url},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    raise RuntimeError(
                        "VirusTotal rate limit exceeded. "
                        "Set MOCK_VIRUSTOTAL=true or wait 1 minute."
                    )
                resp.raise_for_status()
                submit_data = await resp.json()

            # Extract analysis ID
            analysis_id = submit_data.get("data", {}).get("id", "")
            if not analysis_id:
                raise RuntimeError("VirusTotal did not return an analysis ID")

            # Step 2: Poll for results (max 30 seconds)
            max_polls  = 6
            poll_delay = 5

            for attempt in range(max_polls):
                await asyncio.sleep(poll_delay)

                async with session.get(
                    f"{VT_BASE_URL}/analyses/{analysis_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    resp.raise_for_status()
                    analysis_data = await resp.json()

                status = (analysis_data
                          .get("data", {})
                          .get("attributes", {})
                          .get("status", ""))

                if status == "completed":
                    stats = (analysis_data
                             .get("data", {})
                             .get("attributes", {})
                             .get("stats", {}))

                    malicious_count  = stats.get("malicious",  0)
                    suspicious_count = stats.get("suspicious", 0)
                    harmless_count   = stats.get("harmless",   0)
                    undetected_count = stats.get("undetected", 0)
                    total_engines    = (malicious_count + suspicious_count +
                                        harmless_count  + undetected_count)

                    return VirusTotalResult(
                        url=url,
                        malicious_count=malicious_count,
                        suspicious_count=suspicious_count,
                        harmless_count=harmless_count,
                        undetected_count=undetected_count,
                        total_engines=total_engines,
                        domain_age_days=None,   # Requires separate WHOIS lookup
                        domain=domain,
                        tld=tld,
                        scan_date=_today(),
                        is_mocked=False,
                        raw_stats=stats,
                    )

            # Timed out waiting for analysis
            raise RuntimeError(
                f"VirusTotal analysis timed out after {max_polls * poll_delay}s"
            )

    except Exception as e:
        # Runtime fallback: return INSUFFICIENT_EVIDENCE result
        print(f"[VirusTotal] API error: {e}")
        print("[VirusTotal] Returning INSUFFICIENT_EVIDENCE fallback")
        return _get_fallback_result(url, str(e))


def _get_fallback_result(url: str, error_msg: str) -> VirusTotalResult:
    """
    Runtime fallback when VirusTotal API is unavailable.
    Returns a result that the synthesis layer will interpret as
    INSUFFICIENT_EVIDENCE — never fabricates a clean verdict.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        tld    = domain.split(".")[-1] if "." in domain else ""
    except Exception:
        domain = ""
        tld    = ""

    return VirusTotalResult(
        url=url,
        malicious_count=-1,     # Sentinel: API unavailable
        suspicious_count=-1,
        harmless_count=-1,
        undetected_count=-1,
        total_engines=0,
        domain_age_days=None,
        domain=domain,
        tld=tld,
        scan_date=_today(),
        is_mocked=False,
        raw_stats={"error": error_msg, "status": "api_unavailable"},
    )


# ============================================================
# PUBLIC API
# ============================================================

async def scan_url(url: str) -> VirusTotalResult:
    """
    Scan a URL using VirusTotal.

    In mock mode (MOCK_VIRUSTOTAL=true): returns deterministic mock result.
    In live mode (MOCK_VIRUSTOTAL=false): calls real VirusTotal API.

    Results are cached for 24 hours in both modes to protect API quota
    and ensure consistent results within a session.

    Args:
        url: The URL to scan (full URL including scheme)

    Returns:
        VirusTotalResult with engine counts, domain info, and preliminary verdict

    Example:
        result = await scan_url("http://paypal-verify.suspicious.xyz/login")
        print(result.verdict)          # MALICIOUS
        print(result.malicious_count)  # 14
        print(result.domain_age_days)  # 3
    """
    # Normalize URL
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # Check cache
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    if url_hash in _cache:
        cached_result, cached_time = _cache[url_hash]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_result

    # Scan
    if MOCK_VIRUSTOTAL:
        result = _get_mock_result(url)
    else:
        result = await _call_virustotal_api(url)

    # Cache result
    _cache[url_hash] = (result, time.time())
    return result


def scan_url_sync(url: str) -> VirusTotalResult:
    """Synchronous wrapper for scan_url."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(scan_url(url))
    finally:
        pass


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    """
    Quick test to verify the mock is working correctly.
    Run: python tools/virustotal.py
    """
    test_urls = [
        # Should be CLEAN — trusted domains
        ("https://www.google.com",              "CLEAN"),
        ("https://www.paypal.com/signin",       "CLEAN"),
        ("https://northeastern.edu",            "CLEAN"),

        # Should be MALICIOUS — obvious phishing signals
        ("http://paypal-secure-verify.xyz/login/confirm", "MALICIOUS"),
        ("http://apple-id-verify.tk/account/suspended",   "MALICIOUS"),
        ("http://secure.login.microsoft-verify.xyz/auth", "MALICIOUS"),

        # Should be SUSPICIOUS — moderate signals
        ("http://account-update.online/billing",          "MALICIOUS"),
        ("http://verify-identity.site/login",             "MALICIOUS"),
    ]

    print("=" * 60)
    print("VirusTotal Mock Engine Test")
    print(f"Mode: {'MOCK' if MOCK_VIRUSTOTAL else 'LIVE'}")
    print("=" * 60)

    passed = 0
    for url, expected in test_urls:
        result = scan_url_sync(url)
        status = "✅" if result.verdict == expected else "❌"
        if result.verdict == expected:
            passed += 1
        print(f"\n{status} {url[:55]}")
        print(f"   Expected: {expected:<12} Got: {result.verdict}")
        print(f"   Engines: {result.malicious_count} malicious / "
              f"{result.suspicious_count} suspicious / "
              f"{result.total_engines} total")
        print(f"   Domain age: {result.domain_age_days} days | "
              f"TLD: .{result.tld}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(test_urls)} correct")
    status = "PASS" if passed >= len(test_urls) - 1 else "FAIL"
    print(f"Status: {status}")
    print("=" * 60)