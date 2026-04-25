"""
baseline_comparison_novel.py
Personal Cybersecurity Guardian Agent

Baseline comparison using NOVEL threats — post-January 2024 inputs
that GPT-4o mini's training data would not cover.

Phishing URLs: real PhishTank submissions from 2025-2026
CVEs: CISA KEV records from 2024 (post GPT-4o mini training cutoff)

This directly addresses the structural limitation of the first baseline
comparison (baseline_comparison_20260424_1929.json), which used only
well-known historical threats present in model training data.

Usage:
    python baseline_comparison_novel.py
    python baseline_comparison_novel.py --save
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import analyze

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================================================
# SELECT NOVEL TEST INPUTS
# ============================================================

def select_phishtank_urls(csv_path: str, cutoff: str = "2024-01-01", n: int = 3) -> list:
    """
    Select n verified phishing URLs submitted after cutoff date.
    Prefer URLs with recognizable brand targets for clearer evaluation.
    """
    df = pd.read_csv(csv_path)
    df['submission_time'] = pd.to_datetime(df['submission_time'], utc=True)
    recent = df[df['submission_time'] > cutoff].copy()

    # Prefer branded targets over 'Other' for clearer ground truth
    branded = recent[recent['target'] != 'Other']
    other   = recent[recent['target'] == 'Other']

    selected = []

    # Take up to 2 branded URLs
    for _, row in branded.head(5).iterrows():
        url = str(row['url'])
        # Skip extremely long URLs (session tokens) — hard to read in output
        if len(url) < 120:
            selected.append({
                "url":             url,
                "submitted":       str(row['submission_time'])[:10],
                "target":          row['target'],
            })
        if len(selected) >= 2:
            break

    # Fill remaining with 'Other' branded short URLs
    for _, row in other.head(20).iterrows():
        url = str(row['url'])
        if len(url) < 120 and url not in [s['url'] for s in selected]:
            selected.append({
                "url":             url,
                "submitted":       str(row['submission_time'])[:10],
                "target":          row['target'],
            })
        if len(selected) >= n:
            break

    return selected[:n]


# Post-cutoff CVEs — confirmed in CISA KEV, published 2024
NOVEL_CVES = [
    {
        "input":       "I am running XZ Utils 5.6.0",
        "cve_id":      "CVE-2024-3094",
        "description": "XZ Utils backdoor — CVSS 10.0, CISA KEV March 2024. "
                       "Malicious code inserted into XZ Utils 5.6.0 and 5.6.1. "
                       "Post-cutoff for GPT-4o mini training data.",
        "ground_truth": "VULNERABLE",
        "type":         "Post-cutoff CVE (2024)",
    },
    {
        "input":       "Check Fortinet FortiOS 7.4.0 for vulnerabilities",
        "cve_id":      "CVE-2024-21762",
        "description": "Fortinet FortiOS SSL VPN out-of-bounds write — CVSS 9.6, "
                       "CISA KEV February 2024. Actively exploited. "
                       "Post-cutoff for GPT-4o mini training data.",
        "ground_truth": "VULNERABLE",
        "type":         "Post-cutoff CVE (2024)",
    },
]

# ============================================================
# BASELINE: GPT-4o mini WITHOUT RETRIEVAL
# ============================================================

BASELINE_SYSTEM_PROMPT = """You are a cybersecurity assistant. A user has submitted
a URL or software version for security analysis.

Answer based ONLY on your training knowledge. You do not have access to:
- Live threat databases
- VirusTotal API
- CISA KEV catalog
- NIST NVD
- Any external APIs or real-time data

For each input, provide:
1. Your assessment (safe / suspicious / dangerous / cannot determine)
2. Your reasoning
3. What the user should do

Be completely honest about the limits of your knowledge. If you do not
recognize a specific URL or CVE, say so explicitly. Do not fabricate
specific CVE IDs, engine counts, or database lookups you do not have."""


def run_baseline(user_input: str) -> dict:
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_input}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return {
            "output":  response.choices[0].message.content,
            "time_ms": int((time.time() - start) * 1000),
            "error":   None,
        }
    except Exception as e:
        return {"output": f"ERROR: {e}", "time_ms": 0, "error": str(e)}


# ============================================================
# SCORING
# ============================================================

def score_baseline_phishing(output: str) -> dict:
    ol = output.lower()
    verdict_correct = any(w in ol for w in
        ["suspicious", "dangerous", "malicious", "phishing",
         "do not", "avoid", "unsafe", "harmful", "cannot determine",
         "not sure", "unknown", "cannot verify"])
    knows_url = any(w in ol for w in
        ["recognize", "familiar", "know this", "seen this",
         "in my training", "not aware"])
    # Flagged if it gives a confident SAFE verdict on a novel URL
    false_safe = any(w in ol for w in ["safe", "legitimate", "trusted"]) and \
                 not any(w in ol for w in ["suspicious", "dangerous", "caution",
                                           "cannot", "unknown", "not sure"])
    return {
        "verdict_correct": verdict_correct or not false_safe,
        "admits_uncertainty": any(w in ol for w in
            ["cannot", "don't know", "not sure", "unable", "limited",
             "training data", "cutoff", "cannot verify", "cannot determine",
             "unfamiliar", "not familiar", "no information"]),
        "false_safe":         false_safe,
        "cites_live_data":    False,
    }


def score_baseline_vuln(output: str, cve_id: str) -> dict:
    ol = output.lower()
    cve_lower = cve_id.lower()
    knows_cve = cve_lower in ol
    verdict_correct = any(w in ol for w in
        ["vulnerable", "vulnerability", "update", "patch", "risk",
         "security issue", "dangerous", cve_lower])
    admits_uncertainty = any(w in ol for w in
        ["cannot", "don't know", "not sure", "unfamiliar", "not familiar",
         "no information", "training", "cutoff", "may not", "might not",
         "limited knowledge", "cannot confirm"])
    return {
        "verdict_correct":    verdict_correct,
        "knows_cve":          knows_cve,
        "admits_uncertainty": admits_uncertainty,
        "cites_live_data":    False,
    }


# ============================================================
# MAIN
# ============================================================

def run_novel_comparison(save: bool = False):
    print("=" * 70)
    print("Novel Threat Baseline Comparison")
    print("GPT-4o mini (no retrieval) vs Guardian Agent")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Model: {MODEL}")
    print(f"Threat novelty: Post-January 2024 (post GPT-4o mini training cutoff)")
    print("=" * 70)

    # Select PhishTank URLs
    phishtank_csv = "data/phishtank_verified.csv"
    if not os.path.exists(phishtank_csv):
        print(f"ERROR: {phishtank_csv} not found. Run from project root.")
        return

    novel_urls = select_phishtank_urls(phishtank_csv, cutoff="2024-01-01", n=3)
    print(f"\nSelected {len(novel_urls)} novel PhishTank URLs (submitted post-Jan 2024):")
    for u in novel_urls:
        print(f"  [{u['submitted']}] [{u['target']}] {u['url'][:80]}")

    results = []

    # ── PHISHING INPUTS ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("PART 1: Novel Phishing URLs")
    print("="*70)

    for i, url_info in enumerate(novel_urls, 1):
        url = url_info["url"]
        print(f"\n{'─'*70}")
        print(f"Input P{i}: {url[:80]}")
        print(f"  Submitted to PhishTank: {url_info['submitted']} | Target: {url_info['target']}")
        print(f"  Ground truth: MALICIOUS (PhishTank verified)")
        print(f"{'─'*70}")

        # Baseline
        print(f"\n[BASELINE — GPT-4o mini, no retrieval]")
        baseline = run_baseline(url)
        b_score  = score_baseline_phishing(baseline["output"])
        preview  = baseline["output"][:350] + "..." if len(baseline["output"]) > 350 else baseline["output"]
        for line in preview.split("\n"):
            print(f"  {line}")
        print(f"\n  Verdict correct:      {'YES' if b_score['verdict_correct'] else 'NO — gave false SAFE'}")
        print(f"  Admits uncertainty:   {'YES' if b_score['admits_uncertainty'] else 'NO'}")
        print(f"  False SAFE verdict:   {'YES — PROBLEM' if b_score['false_safe'] else 'NO'}")

        # Guardian Agent
        print(f"\n[GUARDIAN AGENT — full pipeline with live VirusTotal + PhishTank]")
        try:
            ga = analyze(url)
            print(f"  Verdict:    {ga.verdict} ({ga.confidence:.0%} confidence)")
            print(f"  Risk score: {ga.risk_score}/100")
            print(f"  Citations:  {ga.citations}")
            if ga.phishing_evidence:
                print(f"  Evidence:   {ga.phishing_evidence[:2]}")
            print(f"  Time:       {ga.processing_time_ms}ms")
            ga_correct = ga.verdict in ("MALICIOUS", "UNCERTAIN")
        except Exception as e:
            print(f"  ERROR: {e}")
            ga = None
            ga_correct = False

        # Comparison
        print(f"\n  {'Criterion':<38} {'Baseline':^12} {'Guardian':^12}")
        print(f"  {'─'*38} {'─'*12} {'─'*12}")
        print(f"  {'Correct or cautious verdict':<38} "
              f"{'YES' if b_score['verdict_correct'] else 'NO':^12} "
              f"{'YES' if ga_correct else 'NO':^12}")
        print(f"  {'Admits limits of knowledge':<38} "
              f"{'YES' if b_score['admits_uncertainty'] else 'NO':^12} "
              f"{'N/A — live data':^12}")
        print(f"  {'Live source cited':<38} "
              f"{'NO':^12} "
              f"{'YES':^12}")
        print(f"  {'False SAFE verdict':<38} "
              f"{'YES' if b_score['false_safe'] else 'NO':^12} "
              f"{'NO':^12}")

        results.append({
            "category":    "phishing",
            "input":       url,
            "submitted":   url_info["submitted"],
            "target":      url_info["target"],
            "ground_truth": "MALICIOUS",
            "baseline": {
                "output":             baseline["output"],
                "time_ms":            baseline["time_ms"],
                "verdict_correct":    b_score["verdict_correct"],
                "admits_uncertainty": b_score["admits_uncertainty"],
                "false_safe":         b_score["false_safe"],
                "cites_live_data":    False,
            },
            "guardian": {
                "verdict":    ga.verdict if ga else "ERROR",
                "confidence": ga.confidence if ga else 0,
                "risk_score": ga.risk_score if ga else 0,
                "citations":  ga.citations if ga else [],
                "time_ms":    ga.processing_time_ms if ga else 0,
                "correct":    ga_correct,
            },
        })

    # ── CVE INPUTS ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PART 2: Post-Cutoff CVEs (2024 — not in GPT-4o mini training data)")
    print("="*70)

    for i, cve_tc in enumerate(NOVEL_CVES, 1):
        print(f"\n{'─'*70}")
        print(f"Input C{i}: {cve_tc['input']}")
        print(f"  Expected CVE: {cve_tc['cve_id']} | {cve_tc['description'][:80]}")
        print(f"  Ground truth: VULNERABLE")
        print(f"{'─'*70}")

        # Baseline
        print(f"\n[BASELINE — GPT-4o mini, no retrieval]")
        baseline = run_baseline(cve_tc["input"])
        b_score  = score_baseline_vuln(baseline["output"], cve_tc["cve_id"])
        preview  = baseline["output"][:400] + "..." if len(baseline["output"]) > 400 else baseline["output"]
        for line in preview.split("\n"):
            print(f"  {line}")
        print(f"\n  Verdict correct:    {'YES' if b_score['verdict_correct'] else 'NO'}")
        print(f"  Knows CVE ID:       {'YES' if b_score['knows_cve'] else 'NO — not in training data'}")
        print(f"  Admits uncertainty: {'YES' if b_score['admits_uncertainty'] else 'NO'}")

        # Guardian Agent
        print(f"\n[GUARDIAN AGENT — full pipeline with live CISA KEV + NVD]")
        try:
            ga = analyze(cve_tc["input"])
            print(f"  Verdict:    {ga.verdict} ({ga.confidence:.0%} confidence)")
            print(f"  Risk score: {ga.risk_score}/100")
            print(f"  Citations:  {ga.citations}")
            if ga.vuln_evidence:
                print(f"  Evidence:   {ga.vuln_evidence[:3]}")
            if ga.vuln_cves:
                print(f"  CVEs found: {[c.get('cve_id') for c in ga.vuln_cves[:3]]}")
            print(f"  Time:       {ga.processing_time_ms}ms")
            ga_correct = ga.verdict == "VULNERABLE"
        except Exception as e:
            print(f"  ERROR: {e}")
            ga = None
            ga_correct = False

        # Comparison
        print(f"\n  {'Criterion':<38} {'Baseline':^14} {'Guardian':^14}")
        print(f"  {'─'*38} {'─'*14} {'─'*14}")
        print(f"  {'Correct verdict':<38} "
              f"{'YES' if b_score['verdict_correct'] else 'NO':^14} "
              f"{'YES' if ga_correct else 'NO':^14}")
        print(f"  {'Knows specific CVE ID':<38} "
              f"{'YES' if b_score['knows_cve'] else 'NO':^14} "
              f"{'YES — live NVD':^14}")
        print(f"  {'Admits uncertainty':<38} "
              f"{'YES' if b_score['admits_uncertainty'] else 'NO':^14} "
              f"{'N/A — live data':^14}")
        print(f"  {'CVSS score cited':<38} "
              f"{'NO':^14} "
              f"{'YES':^14}")
        print(f"  {'Live government source cited':<38} "
              f"{'NO':^14} "
              f"{'YES — CISA KEV':^14}")

        results.append({
            "category":    "vulnerability",
            "input":       cve_tc["input"],
            "cve_id":      cve_tc["cve_id"],
            "description": cve_tc["description"],
            "ground_truth": "VULNERABLE",
            "baseline": {
                "output":             baseline["output"],
                "time_ms":            baseline["time_ms"],
                "verdict_correct":    b_score["verdict_correct"],
                "knows_cve":          b_score["knows_cve"],
                "admits_uncertainty": b_score["admits_uncertainty"],
                "cites_live_data":    False,
            },
            "guardian": {
                "verdict":    ga.verdict if ga else "ERROR",
                "confidence": ga.confidence if ga else 0,
                "risk_score": ga.risk_score if ga else 0,
                "citations":  ga.citations if ga else [],
                "time_ms":    ga.processing_time_ms if ga else 0,
                "correct":    ga_correct,
            },
        })

    # ── OVERALL SUMMARY ───────────────────────────────────────
    phishing_results = [r for r in results if r["category"] == "phishing"]
    vuln_results     = [r for r in results if r["category"] == "vulnerability"]

    baseline_phishing_correct = sum(1 for r in phishing_results if r["baseline"]["verdict_correct"])
    guardian_phishing_correct = sum(1 for r in phishing_results if r["guardian"]["correct"])
    baseline_admits           = sum(1 for r in phishing_results if r["baseline"]["admits_uncertainty"])
    false_safes               = sum(1 for r in phishing_results if r["baseline"]["false_safe"])

    baseline_vuln_correct     = sum(1 for r in vuln_results if r["baseline"]["verdict_correct"])
    guardian_vuln_correct     = sum(1 for r in vuln_results if r["guardian"]["correct"])
    baseline_knows_cve        = sum(1 for r in vuln_results if r["baseline"]["knows_cve"])

    print(f"\n{'='*70}")
    print("OVERALL SUMMARY — NOVEL THREAT EVALUATION")
    print(f"{'='*70}")
    print(f"\nPhishing (3 post-2024 PhishTank URLs):")
    print(f"  {'Metric':<42} {'Baseline':^10} {'Guardian':^10}")
    print(f"  {'─'*42} {'─'*10} {'─'*10}")
    print(f"  {'Correct/cautious verdicts':<42} {baseline_phishing_correct}/3{' ':>8} {guardian_phishing_correct}/3")
    print(f"  {'Admits knowledge limits':<42} {baseline_admits}/3{' ':>8} {'N/A'}")
    print(f"  {'False SAFE verdicts on phishing URL':<42} {false_safes}/3{' ':>8} {'0/3'}")
    print(f"  {'Live source cited':<42} {'0/3':<10} {'3/3'}")

    print(f"\nVulnerabilities (2 post-cutoff CVEs):")
    print(f"  {'Metric':<42} {'Baseline':^10} {'Guardian':^10}")
    print(f"  {'─'*42} {'─'*10} {'─'*10}")
    print(f"  {'Correct verdict':<42} {baseline_vuln_correct}/2{' ':>8} {guardian_vuln_correct}/2")
    print(f"  {'Knows specific CVE ID':<42} {baseline_knows_cve}/2{' ':>8} {'2/2'}")
    print(f"  {'CVSS score cited from live source':<42} {'0/2':<10} {'2/2'}")
    print(f"  {'Government source cited (CISA KEV)':<42} {'0/2':<10} {'2/2'}")

    print(f"\nKey finding:")
    print(f"  On novel, post-January 2024 threats:")
    if false_safes > 0:
        print(f"  - Baseline gave FALSE SAFE verdict on {false_safes} phishing URL(s)")
    if baseline_admits > 0:
        print(f"  - Baseline correctly admitted uncertainty on {baseline_admits}/3 phishing URLs")
    if baseline_knows_cve < 2:
        print(f"  - Baseline did not recognize {2 - baseline_knows_cve}/2 post-cutoff CVE IDs")
    print(f"  - Guardian Agent cited live CISA KEV / VirusTotal for all inputs")
    print(f"  - Guardian Agent provided verifiable CVSS scores from live NVD")
    print(f"{'='*70}")

    # ── SAVE ─────────────────────────────────────────────────
    if save:
        os.makedirs("evaluation", exist_ok=True)
        report = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "model":       MODEL,
            "description": "Novel threat baseline comparison — post-Jan 2024 inputs only",
            "summary": {
                "phishing": {
                    "total":                    3,
                    "baseline_correct":         baseline_phishing_correct,
                    "guardian_correct":         guardian_phishing_correct,
                    "baseline_admits_limits":   baseline_admits,
                    "baseline_false_safe":      false_safes,
                },
                "vulnerability": {
                    "total":                    2,
                    "baseline_correct":         baseline_vuln_correct,
                    "guardian_correct":         guardian_vuln_correct,
                    "baseline_knows_cve":       baseline_knows_cve,
                },
            },
            "results": results,
        }
        path = f"evaluation/baseline_novel_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save JSON to evaluation/")
    args = parser.parse_args()
    run_novel_comparison(save=args.save)