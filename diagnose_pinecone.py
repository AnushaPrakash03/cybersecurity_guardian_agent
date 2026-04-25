"""
diagnose_pinecone.py
Run this to inspect what metadata fields are actually stored in Pinecone.
This tells us why the CPE filter is returning no results.
"""

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cybersecurity-guardian")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("=" * 65)
print("Pinecone Metadata Diagnostic")
print("=" * 65)

# Test 1: Raw semantic search — no filter
print("\n=== Test 1: Raw semantic search for Log4j (no filter) ===")
embedding = model.encode("Log4j 2.14.1 remote code execution vulnerability").tolist()
results   = index.query(
    vector=embedding,
    top_k=5,
    namespace="cve-vulnerabilities",
    include_metadata=True
)
for r in results.matches:
    meta = r.metadata or {}
    print(f"  {meta.get('cve_id','?')} | score: {r.score:.3f}")
    print(f"    product='{meta.get('product','MISSING')}'")
    print(f"    vendor='{meta.get('vendor','MISSING')}'")
    print(f"    known_exploited={meta.get('known_exploited','MISSING')}")
    print(f"    source='{meta.get('source','MISSING')}'")

# Test 2: Raw semantic search for Apache
print("\n=== Test 2: Raw semantic search for Apache 2.4.49 (no filter) ===")
embedding2 = model.encode("Apache HTTP Server 2.4.49 path traversal vulnerability").tolist()
results2   = index.query(
    vector=embedding2,
    top_k=5,
    namespace="cve-vulnerabilities",
    include_metadata=True
)
for r in results2.matches:
    meta = r.metadata or {}
    print(f"  {meta.get('cve_id','?')} | score: {r.score:.3f}")
    print(f"    product='{meta.get('product','MISSING')}'")
    print(f"    vendor='{meta.get('vendor','MISSING')}'")
    print(f"    known_exploited={meta.get('known_exploited','MISSING')}")

# Test 3: known_exploited=True filter with no product filter
print("\n=== Test 3: known_exploited=True filter only (no product filter) ===")
results3 = index.query(
    vector=embedding,
    top_k=5,
    namespace="cve-vulnerabilities",
    filter={"known_exploited": {"$eq": True}},
    include_metadata=True
)
print(f"Results with known_exploited=True: {len(results3.matches)}")
for r in results3.matches:
    meta = r.metadata or {}
    print(f"  {meta.get('cve_id','?')} | product='{meta.get('product','?')}' | known_exploited={meta.get('known_exploited','?')}")

# Test 4: Check what ALL metadata keys exist on a sample record
print("\n=== Test 4: All metadata keys on first result ===")
if results.matches:
    meta = results.matches[0].metadata or {}
    print(f"CVE: {meta.get('cve_id','?')}")
    print("All metadata keys and values:")
    for k, v in meta.items():
        val_preview = str(v)[:80] if v else "None"
        print(f"  {k}: {val_preview}")