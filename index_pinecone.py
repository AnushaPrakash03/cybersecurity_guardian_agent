"""
Week 1: Pinecone Indexing Script
Personal Cybersecurity Guardian Agent

Indexes CISA KEV and NVD data into Pinecone using two namespaces:
  - namespace: "cve-vulnerabilities"  ← CISA KEV + NVD
  - namespace: "phishing-patterns"    ← PhishTank (Week 2)

Usage:
    python index_pinecone.py --kev-only              # Index CISA KEV only
    python index_pinecone.py --nvd-from-file         # Filter local NVD JSON (CVSS >= 7.0) and upload
    python index_pinecone.py --nvd-from-file --test  # Upload NVD + run Recall@5 test
    python index_pinecone.py --test-only             # Run Recall@5 test without re-indexing
    python index_pinecone.py --clear                 # Clear cve-vulnerabilities namespace
"""

import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cybersecurity-guardian"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384 dimensions, fast, free
EMBEDDING_DIM = 384
NAMESPACE_CVE = "cve-vulnerabilities"
BATCH_SIZE = 100                         # Pinecone upsert batch size

# Data sources
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
NVD_DATA_DIR = "data/nvd/"              # NVD JSON files go here
NVD_FILTERED_PATH = "data/nvd/nvd_filtered.json"  # Already-downloaded local file
NVD_CVSS_THRESHOLD = 7.0               # Only index HIGH and CRITICAL severity CVEs

# ============================================================
# SETUP
# ============================================================

def setup_pinecone() -> Pinecone:
    """Initialize Pinecone client and create index if it doesn't exist."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"      # Free tier region
            )
        )
        # Wait for index to be ready
        print("Waiting for index to initialize...")
        time.sleep(10)
    else:
        print(f"Index '{INDEX_NAME}' already exists — skipping creation")

    return pc


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model for generating embeddings."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# CISA KEV INDEXING
# ============================================================

def download_cisa_kev() -> list[dict]:
    """Download CISA KEV catalog from official feed."""
    print(f"Downloading CISA KEV from {CISA_KEV_URL}")
    response = requests.get(CISA_KEV_URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    vulnerabilities = data.get("vulnerabilities", [])
    print(f"Downloaded {len(vulnerabilities)} KEV records")

    # Save locally as backup
    os.makedirs("data", exist_ok=True)
    with open("data/cisa_kev.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved to data/cisa_kev.json")

    return vulnerabilities


def build_kev_text(record: dict) -> str:
    """
    Build the text string that will be embedded for a KEV record.
    Combines the most semantically meaningful fields.
    """
    return (
        f"CVE: {record.get('cveID', 'Unknown')} | "
        f"Product: {record.get('product', 'Unknown')} | "
        f"Vendor: {record.get('vendorProject', 'Unknown')} | "
        f"Vulnerability: {record.get('vulnerabilityName', 'Unknown')} | "
        f"Description: {record.get('shortDescription', '')} | "
        f"Required Action: {record.get('requiredAction', '')}"
    )


def index_cisa_kev(pc: Pinecone, model: SentenceTransformer, records: list[dict]):
    """Index CISA KEV records into Pinecone cve-vulnerabilities namespace."""
    print(f"\nIndexing {len(records)} CISA KEV records into Pinecone...")
    index = pc.Index(INDEX_NAME)

    vectors = []
    for record in tqdm(records, desc="Building KEV vectors"):
        cve_id = record.get("cveID", "unknown")
        text = build_kev_text(record)
        embedding = model.encode(text).tolist()

        vectors.append({
            "id": f"kev-{cve_id}",
            "values": embedding,
            "metadata": {
                # CPE metadata fields — used for strict pre-filter before semantic search
                "cve_id": cve_id,
                "product": record.get("product", "").lower(),
                "vendor": record.get("vendorProject", "").lower(),
                "vulnerability_name": record.get("vulnerabilityName", ""),
                "description": record.get("shortDescription", ""),
                "required_action": record.get("requiredAction", ""),
                "due_date": record.get("dueDate", ""),
                "date_added": record.get("dateAdded", ""),
                "known_exploited": True,        # All KEV records are actively exploited
                "source": "CISA_KEV",
                # Text field for retrieval display
                "text": text
            }
        })

    # Upsert in batches
    _upsert_batches(index, vectors, NAMESPACE_CVE, "KEV")
    print(f"CISA KEV indexing complete — {len(vectors)} records indexed")


# ============================================================
# NVD INDEXING
# ============================================================

def download_nvd_data() -> list[dict]:
    """
    Download NVD data using the NVD API 2.0.
    Fetches records in paginated batches of 2000.
    Filters out 'Awaiting Analysis' entries with no CVSS score.

    Note: NVD API has a rate limit of 5 requests per 30 seconds without an API key.
    Get a free API key at https://nvd.nist.gov/developers/request-an-api-key
    to increase to 50 requests per 30 seconds.
    """
    NVD_API_KEY = os.getenv("NVD_API_KEY", "")
    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}

    os.makedirs(NVD_DATA_DIR, exist_ok=True)

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    results_per_page = 2000
    start_index = 0
    all_records = []
    skipped = 0

    print("Downloading NVD data (this will take several minutes)...")
    print("Tip: Get a free NVD API key to speed this up significantly")

    while True:
        params = {
            "resultsPerPage": results_per_page,
            "startIndex": start_index
        }

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"NVD API error at startIndex {start_index}: {e}")
            break

        vulnerabilities = data.get("vulnerabilities", [])
        total = data.get("totalResults", 0)

        for item in vulnerabilities:
            cve = item.get("cve", {})
            metrics = cve.get("metrics", {})

            # Filter: skip entries with no CVSS score (Awaiting Analysis)
            cvss_score = None
            cvss_severity = None

            # Try CVSSv3.1 first, then v3.0, then v2
            for metric_key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if metric_key in metrics and metrics[metric_key]:
                    cvss_data = metrics[metric_key][0].get("cvssData", {})
                    cvss_score = cvss_data.get("baseScore")
                    cvss_severity = cvss_data.get("baseSeverity") or metrics[metric_key][0].get("baseSeverity")
                    break

            if cvss_score is None:
                skipped += 1
                continue  # Skip INCOMPLETE_RECORD as per proposal

            # Extract description
            descriptions = cve.get("descriptions", [])
            description = next(
                (d["value"] for d in descriptions if d.get("lang") == "en"),
                ""
            )

            # Extract affected products (CPE data)
            configurations = cve.get("configurations", [])
            affected_products = []
            for config in configurations:
                for node in config.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        if cpe_match.get("vulnerable"):
                            affected_products.append(cpe_match.get("criteria", ""))

            record = {
                "cve_id": cve.get("id", ""),
                "description": description,
                "cvss_score": cvss_score,
                "cvss_severity": cvss_severity or "UNKNOWN",
                "published": cve.get("published", ""),
                "last_modified": cve.get("lastModified", ""),
                "affected_products": affected_products,
                "source": "NVD"
            }
            all_records.append(record)

        start_index += results_per_page
        print(f"  Downloaded {min(start_index, total)}/{total} records | "
              f"Valid: {len(all_records)} | Skipped (no CVSS): {skipped}")

        if start_index >= total:
            break

        # Respect NVD rate limit
        sleep_time = 0.6 if NVD_API_KEY else 6
        time.sleep(sleep_time)

    print(f"\nNVD download complete: {len(all_records)} valid records, {skipped} skipped")

    # Save locally
    with open(f"{NVD_DATA_DIR}nvd_filtered.json", "w") as f:
        json.dump(all_records, f)
    print(f"Saved to {NVD_DATA_DIR}nvd_filtered.json")

    return all_records


def build_nvd_text(record: dict) -> str:
    """Build the text string for embedding an NVD record."""
    products_str = " | ".join(record.get("affected_products", [])[:5])  # Cap at 5 CPEs
    return (
        f"CVE: {record.get('cve_id', 'Unknown')} | "
        f"Severity: {record.get('cvss_severity', 'Unknown')} | "
        f"CVSS Score: {record.get('cvss_score', 'Unknown')} | "
        f"Description: {record.get('description', '')} | "
        f"Affected Products: {products_str}"
    )


def index_nvd(pc: Pinecone, model: SentenceTransformer, records: list[dict]):
    """Index NVD records into Pinecone cve-vulnerabilities namespace."""
    print(f"\nIndexing {len(records)} NVD records into Pinecone...")
    index = pc.Index(INDEX_NAME)

    vectors = []
    for record in tqdm(records, desc="Building NVD vectors"):
        cve_id = record.get("cve_id", "unknown")
        text = build_nvd_text(record)
        embedding = model.encode(text).tolist()

        vectors.append({
            "id": f"nvd-{cve_id}",
            "values": embedding,
            "metadata": {
                "cve_id": cve_id,
                "description": record.get("description", ""),
                "cvss_score": float(record.get("cvss_score", 0)),
                "cvss_severity": record.get("cvss_severity", "UNKNOWN"),
                "published": record.get("published", ""),
                "last_modified": record.get("last_modified", ""),
                "affected_products": record.get("affected_products", [])[:10],
                "known_exploited": False,   # Will be True only for KEV records
                "source": "NVD",
                "text": text
            }
        })

    _upsert_batches(index, vectors, NAMESPACE_CVE, "NVD")
    print(f"NVD indexing complete — {len(vectors)} records indexed")


# ============================================================
# SHARED UTILITIES
# ============================================================

def _upsert_batches(index, vectors: list, namespace: str, label: str):
    """Upsert vectors into Pinecone in batches."""
    total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(vectors), BATCH_SIZE),
                  total=total_batches,
                  desc=f"Uploading {label} to Pinecone"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        time.sleep(0.1)  # Small delay to avoid rate limits


def clear_namespace(pc: Pinecone):
    """Clear the cve-vulnerabilities namespace to remove partial uploads."""
    print(f"Clearing namespace '{NAMESPACE_CVE}' from Pinecone index...")
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True, namespace=NAMESPACE_CVE)
    print("Namespace cleared successfully.")
    time.sleep(3)  # Wait for deletion to propagate

def verify_index(pc: Pinecone):
    """Print index stats to confirm data was indexed correctly."""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print("\n--- Pinecone Index Stats ---")
    print(f"Total vectors: {stats.total_vector_count}")
    for ns, ns_stats in stats.namespaces.items():
        print(f"  Namespace '{ns}': {ns_stats.vector_count} vectors")
    print("----------------------------\n")

def load_nvd_from_file(filepath: str) -> list[dict]:
    """
    Load already-downloaded NVD data from local JSON file.
    Filters to CVSS >= 7.0 (HIGH and CRITICAL only).
    Prints a summary of what was filtered.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"NVD file not found at {filepath}\n"
            f"Run without --nvd-from-file first to download NVD data."
        )

    print(f"Loading NVD data from {filepath}...")
    with open(filepath) as f:
        records = json.load(f)

    total = len(records)

    # Filter to CVSS >= 7.0 only
    filtered = [
        r for r in records
        if r.get("cvss_score") is not None
        and float(r.get("cvss_score", 0)) >= NVD_CVSS_THRESHOLD
    ]

    skipped = total - len(filtered)

    # Show severity breakdown of what we're keeping
    critical = sum(1 for r in filtered if float(r.get("cvss_score", 0)) >= 9.0)
    high = sum(1 for r in filtered if 7.0 <= float(r.get("cvss_score", 0)) < 9.0)

    print(f"\n--- NVD Filter Summary ---")
    print(f"Total records in file : {total:,}")
    print(f"Skipped (CVSS < 7.0)  : {skipped:,}")
    print(f"Kept for indexing     : {len(filtered):,}")
    print(f"  Critical (CVSS 9-10): {critical:,}")
    print(f"  High     (CVSS 7-9) : {high:,}")
    print(f"--------------------------\n")

    return filtered
    """Print index stats to confirm data was indexed correctly."""
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print("\n--- Pinecone Index Stats ---")
    print(f"Total vectors: {stats.total_vector_count}")
    for ns, ns_stats in stats.namespaces.items():
        print(f"  Namespace '{ns}': {ns_stats.vector_count} vectors")
    print("----------------------------\n")


# ============================================================
# RECALL@5 TEST — Week 1 Validation
# ============================================================

def test_recall_at_5(pc: Pinecone, model: SentenceTransformer):
    """
    Test Recall@5 on 20 known CVE queries from the CISA KEV catalog.
    A query passes if the correct CVE appears in the top 5 results.
    """
    # 20 known CVEs from CISA KEV for retrieval testing
    test_queries = [
        {"query": "Microsoft Exchange Server remote code execution", "expected_cve": "CVE-2021-26855"},
        {"query": "Log4j remote code execution vulnerability", "expected_cve": "CVE-2021-44228"},
        {"query": "Apache HTTP Server path traversal", "expected_cve": "CVE-2021-41773"},
        {"query": "VMware vCenter Server remote code execution", "expected_cve": "CVE-2021-21985"},
        {"query": "Fortinet FortiOS SSL VPN authentication bypass", "expected_cve": "CVE-2018-13379"},
        {"query": "Pulse Connect Secure arbitrary file reading", "expected_cve": "CVE-2019-11510"},
        {"query": "SolarWinds Orion supply chain attack", "expected_cve": "CVE-2020-10148"},
        {"query": "Zoho ManageEngine authentication bypass", "expected_cve": "CVE-2021-40539"},
        {"query": "Atlassian Confluence remote code execution", "expected_cve": "CVE-2022-26134"},
        {"query": "Spring Framework remote code execution", "expected_cve": "CVE-2022-22965"},
        {"query": "Chrome V8 type confusion vulnerability", "expected_cve": "CVE-2021-21148"},
        {"query": "Microsoft MSHTML remote code execution", "expected_cve": "CVE-2021-40444"},
        {"query": "Apple iOS kernel privilege escalation", "expected_cve": "CVE-2021-30883"},
        {"query": "F5 BIG-IP iControl REST unauthenticated RCE", "expected_cve": "CVE-2021-22986"},
        {"query": "GitLab CE EE arbitrary file read", "expected_cve": "CVE-2021-22205"},
        {"query": "ProxyLogon Exchange Server SSRF", "expected_cve": "CVE-2021-26857"},
        {"query": "Windows Print Spooler privilege escalation PrintNightmare", "expected_cve": "CVE-2021-34527"},
        {"query": "Cisco IOS XR software memory exhaustion", "expected_cve": "CVE-2020-3566"},
        {"query": "Adobe Acrobat Reader use after free", "expected_cve": "CVE-2021-28550"},
        {"query": "QNAP NAS QTS command injection", "expected_cve": "CVE-2020-2509"},
    ]

    index = pc.Index(INDEX_NAME)
    passed = 0
    failed = []

    print("\nRunning Recall@5 test on 20 known CVE queries...")
    for test in tqdm(test_queries, desc="Testing retrieval"):
        query_embedding = model.encode(test["query"]).tolist()

        results = index.query(
            vector=query_embedding,
            top_k=5,
            namespace=NAMESPACE_CVE,
            filter={"known_exploited": {"$eq": True}},
            include_metadata=True
        )

        retrieved_cves = [
            match.metadata.get("cve_id", "")
            for match in results.matches
        ]

        if test["expected_cve"] in retrieved_cves:
            passed += 1
        else:
            failed.append({
                "query": test["query"],
                "expected": test["expected_cve"],
                "retrieved": retrieved_cves
            })

    recall_at_5 = passed / len(test_queries)
    print(f"\n--- Recall@5 Results ---")
    print(f"Passed: {passed}/{len(test_queries)}")
    print(f"Recall@5 Score: {recall_at_5:.2%}")

    if failed:
        print(f"\nFailed queries ({len(failed)}):")
        for f in failed:
            print(f"  Query: {f['query']}")
            print(f"  Expected: {f['expected']} | Retrieved: {f['retrieved']}")
    else:
        print("All 20 queries passed!")

    # Target is >= 75% for Week 1
    target = 0.75
    status = "PASS" if recall_at_5 >= target else "FAIL"
    print(f"\nTarget (≥{target:.0%}): {status}")
    print("------------------------\n")

    return recall_at_5


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Index cybersecurity data into Pinecone")
    parser.add_argument("--kev-only", action="store_true",
                        help="Index CISA KEV only (skip NVD)")
    parser.add_argument("--nvd-from-file", action="store_true",
                        help="Load NVD from local file, filter CVSS >= 7.0, upload to Pinecone")
    parser.add_argument("--test", action="store_true",
                        help="Run Recall@5 test after indexing")
    parser.add_argument("--test-only", action="store_true",
                        help="Run Recall@5 test without re-indexing")
    parser.add_argument("--clear", action="store_true",
                        help="Clear cve-vulnerabilities namespace before indexing")
    args = parser.parse_args()

    # Setup
    pc = setup_pinecone()
    model = load_embedding_model()

    # Clear namespace if requested
    if args.clear:
        clear_namespace(pc)

    if not args.test_only:
        # Step 1: CISA KEV (always, unless test-only)
        kev_records = download_cisa_kev()
        index_cisa_kev(pc, model, kev_records)

        # Step 2: NVD — from local file (filtered) or fresh download
        if args.nvd_from_file:
            nvd_records = load_nvd_from_file(NVD_FILTERED_PATH)
            index_nvd(pc, model, nvd_records)
        elif not args.kev_only:
            print("\nNVD download will take 15-30 minutes for full dataset.")
            print("Use --nvd-from-file to load from your already-downloaded local file.\n")
            nvd_records = download_nvd_data()
            index_nvd(pc, model, nvd_records)

    # Verify index stats
    verify_index(pc)

    # Run Recall@5 test if requested
    if args.test or args.test_only:
        test_recall_at_5(pc, model)


if __name__ == "__main__":
    main()