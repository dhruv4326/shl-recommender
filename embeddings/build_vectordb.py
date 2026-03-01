"""
Build Pinecone Index using Gemini Embeddings
=============================================
Uses Gemini embedding-001 model (correct endpoint).

Run ONCE:
    cd embeddings
    python build_pinecone.py
"""

import json, re, sys, time, os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "shl-assessments")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

# ── Correct Gemini embedding endpoint ─────────────────────────────────────────
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
EMBEDDING_DIM    = 3072
BATCH_SIZE       = 20
DELAY            = 1.2

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
JSON_PATH = DATA_DIR / "shl_assessments.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
}

NAME_KEYWORDS = {
    "java":        "Java programming language software development backend",
    "python":      "Python programming language data science scripting",
    "javascript":  "JavaScript web frontend development React Node",
    "sql":         "SQL database queries data management relational",
    "c++":         "C++ systems programming performance engineering",
    "c#":          "C# .NET Microsoft software development",
    ".net":        ".NET C# Microsoft enterprise development",
    "php":         "PHP web development backend scripting",
    "swift":       "Swift iOS mobile Apple development",
    "kotlin":      "Kotlin Android mobile development",
    "personality": "personality traits behaviour soft skills character",
    "verbal":      "verbal reasoning reading comprehension language",
    "numerical":   "numerical reasoning mathematics quantitative data",
    "inductive":   "inductive reasoning abstract pattern recognition",
    "deductive":   "deductive logical critical thinking",
    "situational": "situational judgement workplace decisions behaviour",
    "opq":         "occupational personality questionnaire character traits",
    "motivation":  "motivation values engagement drive work style",
    "competency":  "competencies leadership collaboration skills",
    "simulation":  "simulation practical hands-on job exercise",
    "360":         "360 feedback development review performance",
    "cognitive":   "cognitive ability general intelligence reasoning",
    "sales":       "sales business development customer revenue",
    "manager":     "management leadership team people performance",
    "graduate":    "graduate entry level early career campus",
    "clerical":    "clerical administrative office data entry",
    "customer":    "customer service support communication",
}


def load_assessments():
    if not JSON_PATH.exists():
        print(f"❌ {JSON_PATH} not found. Run scraper first!")
        sys.exit(1)
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} assessments.")
    return data


def get_name_keywords(name):
    name_lower = name.lower()
    return " ".join(kw for key, kw in NAME_KEYWORDS.items() if key in name_lower)


def fetch_detail(session, url):
    detail = {"description": "", "duration": None}
    try:
        time.sleep(DELAY)
        resp = session.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for sel in [
            {"class": re.compile(r"pdp__body", re.I)},
            {"class": re.compile(r"product.*description", re.I)},
            {"class": re.compile(r"rich.text", re.I)},
        ]:
            el = soup.find("div", sel)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 40:
                    detail["description"] = text[:700]
                    break
        if not detail["description"]:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    detail["description"] = text[:700]
                    break
        page_text = soup.get_text()
        m = re.search(r"(?:duration|time\s*limit)[^\d]*(\d{1,3})\s*(?:minutes?|mins?)", page_text, re.IGNORECASE)
        if m:
            detail["duration"] = int(m.group(1))
    except Exception as e:
        print(f"  ⚠ {url}: {e}")
    return detail


def enrich_assessments(assessments):
    missing = [a for a in assessments if not a.get("description")]
    if not missing:
        print("✅ All assessments have descriptions.")
        return assessments
    print(f"\n📥 Fetching descriptions for {len(missing)} assessments...")
    session = requests.Session()
    for item in tqdm(missing, desc="Detail pages"):
        detail = fetch_detail(session, item["url"])
        item["description"] = detail["description"]
        item["duration"]    = detail["duration"]
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved enriched data → {JSON_PATH}")
    return assessments


def build_document_text(a):
    parts = [f"Assessment: {a.get('name', '')}"]
    types = a.get("test_type", [])
    if types:
        parts.append(f"Test Types: {', '.join(types)}")
    desc = (a.get("description") or "").strip()
    if desc:
        parts.append(f"Description: {desc}")
    hints = get_name_keywords(a.get("name", ""))
    if hints:
        parts.append(f"Keywords: {hints}")
    if a.get("duration"):
        parts.append(f"Duration: {a['duration']} minutes")
    parts.append(f"Remote Testing: {a.get('remote_support', 'No')}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Gemini Embeddings — CORRECT endpoint
# ══════════════════════════════════════════════════════════════════════════════

def get_gemini_embedding(text: str) -> list[float]:
    """Get embedding from Gemini embedding-001 model."""
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text[:2000]}]},
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                GEMINI_EMBED_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]
        except Exception as e:
            print(f"  Embed attempt {attempt+1} failed: {e}")
            time.sleep(3)
    print(f"  ⚠ Using zero vector as fallback")
    return [0.0] * EMBEDDING_DIM


# ══════════════════════════════════════════════════════════════════════════════
# Pinecone
# ══════════════════════════════════════════════════════════════════════════════

def init_pinecone_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        print(f"Deleting old index '{PINECONE_INDEX}'...")
        pc.delete_index(PINECONE_INDEX)
        time.sleep(8)

    print(f"Creating index '{PINECONE_INDEX}' (dim={EMBEDDING_DIM}, cosine)...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready...")
    while not pc.describe_index(PINECONE_INDEX).status["ready"]:
        time.sleep(2)
    print("✅ Index ready!")
    return pc.Index(PINECONE_INDEX)


def upload_to_pinecone(assessments, index):
    print(f"\n🚀 Embedding + uploading {len(assessments)} assessments to Pinecone...")

    for i in tqdm(range(0, len(assessments), BATCH_SIZE), desc="Batches"):
        batch   = assessments[i : i + BATCH_SIZE]
        vectors = []
        for j, a in enumerate(batch):
            text = build_document_text(a)
            emb  = get_gemini_embedding(text)
            time.sleep(0.5)
            vectors.append({
                "id": f"shl_{i + j}",
                "values": emb,
                "metadata": {
                    "name":             a.get("name", ""),
                    "url":              a.get("url", ""),
                    "description":      (a.get("description") or "")[:500],
                    "duration":         a.get("duration") or 0,
                    "remote_support":   a.get("remote_support", "No"),
                    "adaptive_support": a.get("adaptive_support", "No"),
                    "test_type_str":    " | ".join(a.get("test_type", [])),
                },
            })
        index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print(f"\n✅ Upload complete! {stats['total_vector_count']} vectors stored.")


def run_test_queries(index):
    queries = [
        "Python developer programming skills",
        "personality behaviour soft skills leadership",
        "numerical reasoning cognitive ability",
    ]
    print("\n── Validation queries ──────────────────────────────────────")
    for q in queries:
        emb     = get_gemini_embedding(q)
        results = index.query(vector=emb, top_k=3, include_metadata=True)
        print(f"\nQuery: '{q}'")
        for match in results["matches"]:
            m = match["metadata"]
            print(f"  [{match['score']:.3f}] {m['name']}  [{m['test_type_str']}]")


if __name__ == "__main__":
    print("=" * 60)
    print("SHL Recommender — Build Pinecone with Gemini Embeddings")
    print("=" * 60)

    if not PINECONE_API_KEY or "your_pinecone" in PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not set!"); sys.exit(1)
    if not GEMINI_API_KEY or "your_gemini" in GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set!"); sys.exit(1)

    # Quick test to verify embedding works before processing all 377
    print("\n🔍 Testing Gemini embedding API first...")
    test_emb = get_gemini_embedding("test query")
    if all(v == 0.0 for v in test_emb):
        print("❌ Embedding API failed! Check your GEMINI_API_KEY.")
        sys.exit(1)
    print(f"✅ Embedding API works! Vector dim: {len(test_emb)}")

    assessments = load_assessments()
    assessments = enrich_assessments(assessments)

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = init_pinecone_index(pc)

    upload_to_pinecone(assessments, index)
    run_test_queries(index)

    print("\n" + "=" * 60)
    print("🎉 Done! Vectors live in Pinecone cloud.")
    print("   Push to GitHub and redeploy on Render.")
    print("=" * 60)