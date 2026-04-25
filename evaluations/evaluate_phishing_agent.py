"""
evaluation/evaluate_phishing_agent.py
Personal Cybersecurity Guardian Agent

40-URL Precision/Recall Evaluation for the Phishing Agent.

Test set:
  - 20 known MALICIOUS URLs (from PhishTank verified dataset)
  - 20 known SAFE URLs (Alexa Top 1000 legitimate domains)

Metrics:
  - Precision: of URLs labeled MALICIOUS, how many actually are?
  - Recall:    of actually malicious URLs, how many did we catch?
  - False Positive Rate: of safe URLs, how many incorrectly flagged?
  - Accuracy:  overall correct verdicts / total

Targets (from proposal Phase 3):
  - Precision  ≥ 90%
  - Recall     ≥ 80%
  - FP Rate    ≤ 10%

Usage:
    python evaluations/evaluate_phishing_agent.py
    python evaluations/evaluate_phishing_agent.py --verbose
    python evaluations/evaluate_phishing_agent.py --save-report
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze

# ============================================================
# LABELED TEST SET
# ============================================================

# 20 known MALICIOUS URLs
# Sources: PhishTank verified dataset (phishtank_verified.csv)
# These are real verified phishing URLs from our indexed dataset
MALICIOUS_URLS = [
    # PayPal phishing
    "http://paypal-secure-verify.xyz/login/confirm",
    "http://paypal-account-verify.online/signin/credentials",

    # Apple/ID phishing
    "http://apple-id-verify.tk/account/suspended",
    "http://apple-id-secure.online/verify/account",

    # Microsoft phishing
    "http://microsoft-verify.online/password-reset",
    "http://secure.login.microsoft-verify.xyz/auth",

    # Bank phishing
    "http://secure-login-bankofamerica.xyz/verify/account/credentials",
    "http://bank-secure-login.xyz/verify",

    # IRS/Government phishing
    "http://irs-refund-claim.online/tax/refund/verify",
    "http://account-update.online/billing",

    # Netflix phishing
    "http://netflix-update-payment-details.vercel.app/",
    "http://netflix-billing-update.site/account/payment",

    # Generic credential harvesting
    "http://verify-identity.site/login",
    "http://secure-account-verify.xyz/credentials",
    "http://account-suspended-verify.online/restore",
    "http://login-verify-secure.tk/account/suspended",

    # Brand spoofing with suspicious patterns
    "http://paypal-verify.xyz/login",
    "http://apple-id-verify.online/suspended",
    "http://microsoft-office365-verify.xyz/login",
    "http://it-support-verify.online/reset",
]

# 20 known SAFE URLs
# Sources: Well-established legitimate domains
SAFE_URLS = [
    # Major tech
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",

    # Social/Communication
    "https://www.linkedin.com",
    "https://www.github.com",

    # News/Media
    "https://www.nytimes.com",
    "https://www.bbc.com",

    # Education
    "https://northeastern.edu",
    "https://mit.edu",

    # Government
    "https://www.cisa.gov",
    "https://www.nist.gov",

    # Financial (legitimate)
    "https://www.paypal.com",
    "https://www.chase.com",

    # Tech/Developer
    "https://www.stackoverflow.com",
    "https://www.python.org",

    # Streaming (legitimate)
    "https://www.netflix.com",
    "https://www.spotify.com",

    # E-commerce (legitimate)
    "https://www.ebay.com",
    "https://www.walmart.com",
]

# ============================================================
# EVALUATION LOGIC
# ============================================================

def evaluate_url(url: str, expected_label: str, verbose: bool = False) -> dict:
    """
    Run a single URL through the full pipeline and compare to expected label.

    Returns:
        dict with url, expected, got, correct, verdict, confidence, time_ms
    """
    start = time.time()

    try:
        result = analyze(user_input=url, user_already_clicked=False)
        verdict    = result.verdict
        confidence = result.confidence
        time_ms    = int((time.time() - start) * 1000)

        # Map verdict to MALICIOUS / SAFE / OTHER for evaluation
        if verdict in ("MALICIOUS",):
            predicted = "MALICIOUS"
        elif verdict in ("SAFE",):
            predicted = "SAFE"
        else:
            # UNCERTAIN, INSUFFICIENT_EVIDENCE → treat as neither confirmed malicious nor safe
            predicted = "UNCERTAIN"

        correct = (
            (expected_label == "MALICIOUS" and predicted == "MALICIOUS") or
            (expected_label == "SAFE"      and predicted == "SAFE")
        )

        if verbose:
            status = "✅" if correct else "❌"
            print(f"  {status} [{expected_label}→{verdict}] {url[:60]}")
            print(f"     Confidence: {confidence:.0%} | Time: {time_ms}ms")

        return {
            "url":           url,
            "expected":      expected_label,
            "verdict":       verdict,
            "predicted":     predicted,
            "correct":       correct,
            "confidence":    confidence,
            "time_ms":       time_ms,
            "error":         None,
        }

    except Exception as e:
        time_ms = int((time.time() - start) * 1000)
        if verbose:
            print(f"  ❌ [ERROR] {url[:60]} — {e}")
        return {
            "url":           url,
            "expected":      expected_label,
            "verdict":       "ERROR",
            "predicted":     "ERROR",
            "correct":       False,
            "confidence":    0.0,
            "time_ms":       time_ms,
            "error":         str(e),
        }


def compute_metrics(results: list) -> dict:
    """
    Compute precision, recall, F1, accuracy, and false positive rate.

    Definitions:
      TP = predicted MALICIOUS, actually MALICIOUS
      FP = predicted MALICIOUS, actually SAFE
      TN = predicted SAFE, actually SAFE
      FN = predicted SAFE or UNCERTAIN, actually MALICIOUS
    """
    tp = sum(1 for r in results if r["expected"] == "MALICIOUS" and r["predicted"] == "MALICIOUS")
    fp = sum(1 for r in results if r["expected"] == "SAFE"      and r["predicted"] == "MALICIOUS")
    tn = sum(1 for r in results if r["expected"] == "SAFE"      and r["predicted"] == "SAFE")
    fn = sum(1 for r in results if r["expected"] == "MALICIOUS" and r["predicted"] != "MALICIOUS")

    # UNCERTAIN cases
    uncertain_mal  = sum(1 for r in results if r["expected"] == "MALICIOUS" and r["predicted"] == "UNCERTAIN")
    uncertain_safe = sum(1 for r in results if r["expected"] == "SAFE"      and r["predicted"] == "UNCERTAIN")

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy   = (tp + tn) / len(results) if results else 0.0

    avg_time   = sum(r["time_ms"] for r in results) / len(results) if results else 0
    avg_conf   = sum(r["confidence"] for r in results if r["verdict"] != "ERROR") / len(results)

    errors     = sum(1 for r in results if r["error"])

    return {
        "total":           len(results),
        "tp":              tp,
        "fp":              fp,
        "tn":              tn,
        "fn":              fn,
        "uncertain_mal":   uncertain_mal,
        "uncertain_safe":  uncertain_safe,
        "precision":       precision,
        "recall":          recall,
        "f1_score":        f1,
        "false_positive_rate": fpr,
        "accuracy":        accuracy,
        "avg_confidence":  avg_conf,
        "avg_time_ms":     avg_time,
        "errors":          errors,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(verbose: bool = False, save_report: bool = False):
    """Run the full 40-URL evaluation."""

    print("=" * 65)
    print("Phishing Agent Evaluation — 40 URL Precision/Recall Test")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    all_results = []

    # ── Malicious URLs ────────────────────────────────────────
    print(f"\n[1/2] Testing {len(MALICIOUS_URLS)} MALICIOUS URLs...")
    if verbose:
        print()
    for url in MALICIOUS_URLS:
        result = evaluate_url(url, "MALICIOUS", verbose=verbose)
        all_results.append(result)
        if not verbose:
            status = "✅" if result["correct"] else f"⚠️  {result['verdict']}"
            print(f"  {status} {url[:55]}")

    # ── Safe URLs ────────────────────────────────────────────
    print(f"\n[2/2] Testing {len(SAFE_URLS)} SAFE URLs...")
    if verbose:
        print()
    for url in SAFE_URLS:
        result = evaluate_url(url, "SAFE", verbose=verbose)
        all_results.append(result)
        if not verbose:
            status = "✅" if result["correct"] else f"⚠️  {result['verdict']}"
            print(f"  {status} {url[:55]}")

    # ── Compute metrics ───────────────────────────────────────
    metrics = compute_metrics(all_results)

    # ── Print report ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (caught phishing correctly): {metrics['tp']:>3}")
    print(f"  True Negatives  (cleared safe correctly):    {metrics['tn']:>3}")
    print(f"  False Positives (safe URL flagged as bad):   {metrics['fp']:>3}")
    print(f"  False Negatives (phishing URL missed):       {metrics['fn']:>3}")
    print(f"  Uncertain — malicious URL:                   {metrics['uncertain_mal']:>3}")
    print(f"  Uncertain — safe URL:                        {metrics['uncertain_safe']:>3}")

    print(f"\nMetrics:")
    print(f"  Precision           : {metrics['precision']:.1%}   (target: ≥ 90%)")
    print(f"  Recall              : {metrics['recall']:.1%}   (target: ≥ 80%)")
    print(f"  F1 Score            : {metrics['f1_score']:.1%}")
    print(f"  False Positive Rate : {metrics['false_positive_rate']:.1%}   (target: ≤ 10%)")
    print(f"  Accuracy            : {metrics['accuracy']:.1%}")

    print(f"\nPerformance:")
    print(f"  Avg confidence      : {metrics['avg_confidence']:.1%}")
    print(f"  Avg response time   : {metrics['avg_time_ms']:,.0f}ms")
    print(f"  Errors              : {metrics['errors']}")

    print(f"\nTarget Assessment:")
    p_pass  = "✅ PASS" if metrics['precision']           >= 0.90 else "❌ FAIL"
    r_pass  = "✅ PASS" if metrics['recall']              >= 0.80 else "❌ FAIL"
    fp_pass = "✅ PASS" if metrics['false_positive_rate'] <= 0.10 else "❌ FAIL"
    print(f"  Precision  ≥ 90%  : {p_pass}  ({metrics['precision']:.1%})")
    print(f"  Recall     ≥ 80%  : {r_pass}  ({metrics['recall']:.1%})")
    print(f"  FP Rate    ≤ 10%  : {fp_pass}  ({metrics['false_positive_rate']:.1%})")

    all_pass = all([
        metrics['precision']           >= 0.90,
        metrics['recall']              >= 0.80,
        metrics['false_positive_rate'] <= 0.10,
    ])
    overall = "✅ OVERALL PASS" if all_pass else "⚠️  NEEDS REVIEW"
    print(f"\n  {overall}")
    print("=" * 65)

    # ── Failed cases ─────────────────────────────────────────
    failed = [r for r in all_results if not r["correct"] and r["error"] is None]
    if failed:
        print(f"\nIncorrect verdicts ({len(failed)}):")
        for r in failed:
            print(f"  [{r['expected']} → {r['verdict']}] {r['url'][:60]}")

    # ── Save report ───────────────────────────────────────────
    if save_report:
        os.makedirs("evaluations", exist_ok=True)
        report = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "metrics":    metrics,
            "results":    all_results,
        }
        report_path = f"evaluations/phishing_eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {report_path}")

    return metrics, all_results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="40-URL Phishing Agent Evaluation")
    parser.add_argument("--verbose",     action="store_true", help="Show confidence and timing per URL")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report to evaluation/")
    args = parser.parse_args()

    metrics, results = run_evaluation(
        verbose     = args.verbose,
        save_report = args.save_report,
    )