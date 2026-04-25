# Personal Cybersecurity Guardian Agent

**Live Demo:** https://huggingface.co/spaces/prakashanusha/cybersecurity-guardian

**Project Web Page:** https://anushaprakash03.github.io/cybersecurity_guardian_agent/

A multi-agent AI system that provides real-time cybersecurity threat analysis for non-technical users. Paste a suspicious URL, email, or software version — the system returns a plain-English verdict with evidence and a situation-specific action plan.

---

## What It Does

Most cybersecurity tools are built for security professionals. This system is built for everyone else.

| Input | What the system checks | Example verdict |
|-------|------------------------|-----------------|
| Suspicious URL | VirusTotal (70+ engines) + PhishTank pattern matching | MALICIOUS — 25/72 engines flagged, domain registered 3 days ago |
| Email text | Social engineering signals, urgency language, credential harvesting patterns | UNCERTAIN — urgency language detected, no URL to verify |
| Software version | CISA Known Exploited Vulnerabilities + NIST NVD | VULNERABLE — CVE-2021-44228 CVSS 10.0, actively exploited |
| Combined (URL + software) | Both agents, conflict resolution applied | MALICIOUS — phishing agent verdict surfaces over vulnerability finding |

**Verdicts:** `MALICIOUS` / `VULNERABLE` / `SAFE` / `UNCERTAIN` / `INSUFFICIENT_EVIDENCE` / `CLARIFY`

---

## Architecture

```
User Input (URL / Email / Software + Version)
        │
        ▼
LangGraph Router
  Intent classification  ·  Version completeness enforcement
  GPT-4o mini  ·  Confidence threshold 0.75
        │
        ├─────────────────────────┬─────────────────────────┐
        ▼                         ▼                         ▼
Phishing Agent            Vulnerability Agent           CLARIFY
  VirusTotal API v3          CISA KEV (Pass 1)        Ask user for
  PhishTank RAG              NVD semantic (Pass 2)    missing info
  Email signal extraction    Two-pass search
        │                         │
        └──────────┬──────────────┘
                   ▼
          Synthesis Layer
            5 conflict resolution rules
            GPT-4o mini  ·  Anti-hallucination guards
            Plain-English summary  ·  Source citations
                   │
                   ▼
     Contextual Intervention Layer
       Situation-specific action plan
       5 prompt templates (not-clicked / clicked / vulnerable / uncertain / safe)
                   │
                   ▼
            Final Output
     Verdict  ·  Risk Score  ·  Evidence  ·  Action Plan
```

---

## Data Sources

| Source | Records Indexed | Purpose |
|--------|----------------|---------|
| CISA Known Exploited Vulnerabilities | 1,566 | Actively exploited CVEs — highest priority |
| NIST National Vulnerability Database | 90,560 | HIGH/CRITICAL CVEs (CVSS >= 7.0), 1988–2022 |
| PhishTank | 57,705 | Verified phishing URL patterns |
| VirusTotal API v3 | Live | Real-time URL scanning across 70+ engines |
| **Total indexed** | **149,831 vectors** | Pinecone Serverless, aws us-east-1, 384 dims |

---

## Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Agent orchestration | LangGraph (2024) | Graph-based multi-agent state |
| Vector database | Pinecone Serverless (2024) | Persistent across restarts |
| Embeddings | all-MiniLM-L6-v2 | 384-dim cosine similarity |
| Language model | GPT-4o mini (July 2024) | Synthesis + routing + intervention |
| Threat scanning | VirusTotal API v3 | Live, 70+ engines |
| Frontend | Streamlit | Deployed on Hugging Face Spaces |
| Async resolution | concurrent.futures | Isolated event loop per VT call |

---

## Evaluation Results

| Test | Score | Target | Status |
|------|-------|--------|--------|
| CVE Recall@5 (20 KEV queries) | 85% (17/20) | >= 75% | PASS |
| Phishing evaluation — mock mode (40 URLs) | Precision 100%, Recall 100% | Precision >= 90% | PASS |
| Phishing evaluation — live VirusTotal (40 URLs) | Precision 100%, FP Rate 0% | FP Rate <= 10% | PASS |
| LangGraph router (20 inputs, 5 categories) | 20/20 | >= 85% | PASS |
| Synthesis conflict resolution (20 scenarios) | 20/20 | >= 18/20 | PASS |
| Full pipeline end-to-end (5 inputs) | 5/5 | — | PASS |

---

## Project Structure

```
cybersecurity-guardian/
├── app.py                        # Streamlit UI
├── main.py                       # Full pipeline entry point
├── router.py                     # LangGraph intent classifier
├── synthesis.py                  # Conflict resolution + intervention layer
├── requirements.txt
├── README.md
├── agents/
│   ├── __init__.py
│   ├── phishing_agent.py         # VirusTotal + PhishTank RAG
│   └── vulnerability_agent.py   # CISA KEV + NVD two-pass search
└── tools/
    └── virustotal.py             # Mock + live VirusTotal API
```

**Not in this repository (data too large for HF Spaces):**
- `data/cisa_kev.json` — 1,566 records (indexing only)
- `data/phishtank_verified.csv` — 57,705 records (indexing only)
- `data/nvd/nvd_filtered.json` — 90,560 records (indexing only)
- `index_pinecone.py` — CVE indexing script
- `index_phishtank.py` — PhishTank indexing script
- `evaluation/` — evaluation scripts and JSON results

---

## Running Locally

### Prerequisites

```bash
python 3.10+
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
VIRUSTOTAL_API_KEY=your_virustotal_key
NVD_API_KEY=your_nvd_key
PINECONE_INDEX_NAME=cybersecurity-guardian
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
MOCK_VIRUSTOTAL=true
UNCERTAINTY_THRESHOLD=0.70
ROUTER_CONFIDENCE_THRESHOLD=0.75
MAX_QUERIES_PER_SESSION=10
```

Set `MOCK_VIRUSTOTAL=true` during development to avoid consuming your 500 requests/day free tier quota. Switch to `false` for live scanning.

### Option A — Streamlit only (recommended for HF Spaces)

```bash
streamlit run app.py
```

The app imports `main.py` directly — no separate backend process needed.

### Option B — FastAPI + Streamlit (local production setup)

Terminal 1:
```bash
python api.py
# FastAPI backend starts on http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

Terminal 2:
```bash
streamlit run app.py
# Streamlit frontend starts on http://localhost:8501
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at huggingface.co → SDK: Streamlit
2. Add secrets in Space Settings → Variables and secrets:
   - `PINECONE_API_KEY`, `OPENAI_API_KEY`, `VIRUSTOTAL_API_KEY`, `NVD_API_KEY`
   - `PINECONE_INDEX_NAME=cybersecurity-guardian`
   - `OPENAI_MODEL=gpt-4o-mini`
   - `EMBEDDING_MODEL=all-MiniLM-L6-v2`
   - `MOCK_VIRUSTOTAL=true`
   - `UNCERTAINTY_THRESHOLD=0.70`
   - `ROUTER_CONFIDENCE_THRESHOLD=0.75`
   - `MAX_QUERIES_PER_SESSION=10`
3. Push files:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/cybersecurity-guardian
cd cybersecurity-guardian
# copy your files here
git add .
git commit -m "Initial deployment"
git push https://YOUR_USERNAME:hf_YOUR_TOKEN@huggingface.co/spaces/YOUR_USERNAME/cybersecurity-guardian
```

---

## Sample Inputs

**Phishing URL:**
```
http://paypal-secure-verify.xyz/login/confirm
```

**Known vulnerability:**
```
I have Log4j 2.14.1 installed
```

**Safe URL:**
```
https://www.google.com
```

**Email phishing (no URL):**
```
Dear customer, your account has been suspended. Click here immediately to verify your identity or lose access within 24 hours.
```

**Combined — URL + software:**
```
I got this suspicious link http://paypal-verify.xyz/login and I'm also running Apache HTTP Server 2.4.49
```

**Clarification trigger — incomplete version:**
```
Is Zoom 5.15 vulnerable?
```

---

## Key Design Decisions

**Why not just use ChatGPT?**
A base LLM cannot call the VirusTotal API, query the current CISA KEV catalog, or search 57,000 verified phishing URLs. Its training data has a cutoff date. In cybersecurity, last week is where the danger lives.

**Why UNCERTAIN instead of SAFE for unknown URLs?**
When VirusTotal returns zero engine flags (URL not in their database), the system returns UNCERTAIN with Pinecone pattern evidence rather than SAFE. Calibrated uncertainty is safer than unwarranted confidence.

**Why CISA KEV first in the two-pass search?**
Actively exploited CVEs require immediate action. Surfacing KEV records before the broader NVD search ensures the most urgent findings are presented first.

**Why drop FLAN-T5 fine-tuning?**
GPT-4o mini's synthesis prompt already meets the faithfulness requirement at lower complexity and zero training cost. Fine-tuning was evaluated as unnecessary.

---

## Known Limitations

- **NVD product metadata gap:** NVD records in Pinecone do not carry product/vendor fields. The vulnerability agent uses semantic search rather than CPE product filtering for NVD records, which may return related but not version-specific CVEs for obscure software names.
- **VirusTotal free tier:** 500 requests/day. Unknown URLs (not in VT database) return zero engine flags — the system correctly returns UNCERTAIN, not SAFE.
- **Static index:** Pinecone snapshot as of April 2026. CVEs and phishing URLs after this date are not reflected. Production deployment requires continuous sync.
- **Test URL recall:** The 40-URL live evaluation used constructed test URLs not present in the VirusTotal database. Precision (100%) and false positive rate (0%) are valid; recall in live mode reflects test design rather than real-world performance.

---

## Academic Context

This project was built for INFO7375 — Prompt Engineering & AI at Northeastern University (Spring 2026).

**Instructor:** Professor Nik Bear Brown

**Professor's tools used during development:**
- **Gru** (Software Design Document Expert) — identified synthesis layer gaps, router underspecification, and ChromaDB/Pinecone inconsistency in the v2 proposal
- **Bacon** (Academic Writing Coach) — structured the evidence-first proposal format
- **CRITIQ** (Peer Review Protocol) — evaluation framework rigor

The Contextual Intervention Layer (the fourth system component) was introduced at the recommendation of the course professor: *"the difference between a system that informs and a system that protects."*

---

## License

MIT License. Copyright (c) 2026 Anusha Prakash.

See LICENSE for full terms.

---

## Author
```
**Anusha Prakash**
MS Information Systems, Northeastern University (May 2026)
Substack: @anushaprakash1
Hugging Face: prakashanusha

```
