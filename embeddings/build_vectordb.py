"""
Build Pinecone Vector Index
=============================
1. Reads shl_assessments.json
2. Fetches descriptions from detail pages (if missing)
3. Generates embeddings locally using sentence-transformers
4. Uploads everything to Pinecone cloud

Run ONCE after scraper.py:
    cd embeddings
    python build_pinecone.py

After this runs, your vectors live in Pinecone cloud permanently.
You never need to run this again unless you rescrape.
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os

# ── Load env ───────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "shl-assessments")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # 384-dimensional, fast, free
EMBEDDING_DIM      = 384
BATCH_SIZE         = 50
DELAY              = 1.2

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
JSON_PATH = DATA_DIR / "shl_assessments.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
}

# Keyword hints for name-based semantic enrichment
NAME_KEYWORDS = {
    "java":        "Java programming language software development backend",
    "python":      "Python programming language data science scripting",
    "javascript":  "JavaScript web frontend development React Node",
    "sql":         "SQL database queries data management relational",
    "c++":         "C++ systems programming performance engineering",
    "c#":          "C# .NET Microsoft software development",
    ".net":        ".NET C# Microsoft enterprise development",
    "php":         "PHP web development backend scripting",
    "ruby":        "Ruby on Rails web development scripting",
    "scala":       "Scala functional programming big data",
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
    "mechanical":  "mechanical engineering technical aptitude",
    "customer":    "customer service support communication",
    "contact":     "contact center call center customer support",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_assessments() -> list[dict]:
    if not JSON_PATH.exists():
        print(f"❌ {JSON_PATH} not found. Run scraper/scraper.py first!")
        sys.exit(1)
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} assessments from JSON.")
    return data


def get_name_keywords(name: str) -> str:
    name_lower = name.lower()
    hints = []
    for key, kw in NAME_KEYWORDS.items():
        if key in name_lower:
            hints.append(kw)
    return " ".join(hints)


def fetch_detail(session: requests.Session, url: str) -> dict:
    """Scrape description + duration from individual assessment page."""
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
            {"class": re.compile(r"content__body", re.I)},
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
        for pat in [
            r"(?:duration|time\s*limit|approx)[^\d]*(\d{1,3})\s*(?:minutes?|mins?)",
            r"(\d{1,3})\s*(?:minutes?|mins?)\s*(?:to complete|test|assessment)",
        ]:
            m = re.search(pat, page_text, re.IGNORECASE)
            if m:
                detail["duration"] = int(m.group(1))
                break

    except Exception as e:
        print(f"  ⚠ Detail failed for {url}: {e}")
    return detail


def enrich_assessments(assessments: list[dict]) -> list[dict]:
    """Fetch descriptions for assessments that don't have them."""
    missing = [a for a in assessments if not a.get("description")]
    if not missing:
        print("✅ All assessments already have descriptions.")
        return assessments

    print(f"\n📥 Fetching descriptions for {len(missing)} assessments (~10-15 min)...")
    session = requests.Session()
    for item in tqdm(missing, desc="Detail pages"):
        detail = fetch_detail(session, item["url"])
        item["description"] = detail["description"]
        item["duration"]    = detail["duration"]

    # Save enriched JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"✅ Enriched data saved → {JSON_PATH}")
    return assessments


def build_document_text(a: dict) -> str:
    """Rich text representation for embedding."""
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
    parts.append(f"Adaptive Testing: {a.get('adaptive_support', 'No')}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Pinecone
# ══════════════════════════════════════════════════════════════════════════════

def init_pinecone_index(pc: Pinecone) -> object:
    """Create Pinecone index if it doesn't exist, return it."""
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free tier
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(2)
        print("✅ Index created and ready!")
    else:
        print(f"✅ Index '{PINECONE_INDEX}' already exists.")

    return pc.Index(PINECONE_INDEX)


def upload_to_pinecone(assessments: list[dict], model: SentenceTransformer, index):
    """Generate embeddings and upsert to Pinecone in batches."""
    print(f"\n🚀 Uploading {len(assessments)} assessments to Pinecone...")

    for i in tqdm(range(0, len(assessments), BATCH_SIZE), desc="Uploading batches"):
        batch = assessments[i : i + BATCH_SIZE]

        # Build document texts
        texts = [build_document_text(a) for a in batch]

        # Generate embeddings locally
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Build Pinecone vectors
        vectors = []
        for j, (a, emb) in enumerate(zip(batch, embeddings)):
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

    # Verify count
    stats = index.describe_index_stats()
    total = stats["total_vector_count"]
    print(f"\n✅ Pinecone upload complete! {total} vectors stored.")
    return total


def run_test_queries(model: SentenceTransformer, index):
    """Run a few test queries to validate the index."""
    queries = [
        "Python developer programming skills",
        "personality behaviour soft skills leadership collaboration",
        "numerical reasoning cognitive ability quantitative",
    ]
    print("\n── Validation queries ──────────────────────────────────────")
    for q in queries:
        emb = model.encode([q]).tolist()
        results = index.query(vector=emb[0], top_k=3, include_metadata=True)
        print(f"\nQuery: '{q}'")
        for match in results["matches"]:
            meta = match["metadata"]
            score = match["score"]
            print(f"  [{score:.3f}] {meta['name']}  [{meta['test_type_str']}]")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SHL Recommender — Build Pinecone Vector Index")
    print("=" * 60)

    # Validate keys
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        print("❌ PINECONE_API_KEY not set in .env file!")
        print("   Get your free key at: https://app.pinecone.io/")
        sys.exit(1)

    # Step 1: Load assessments
    assessments = load_assessments()

    # Step 2: Enrich with descriptions (fetches detail pages if missing)
    assessments = enrich_assessments(assessments)

    # Step 3: Load embedding model
    print(f"\n📦 Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")

    # Step 4: Init Pinecone
    print(f"\n🌲 Connecting to Pinecone...")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = init_pinecone_index(pc)

    # Step 5: Upload
    upload_to_pinecone(assessments, model, index)

    # Step 6: Validate
    run_test_queries(model, index)

    print("\n" + "=" * 60)
    print("🎉 All done! Your vectors are now live in Pinecone cloud.")
    print("   Next: cd ../api && uvicorn main:app --reload --port 8000")
    print("=" * 60)