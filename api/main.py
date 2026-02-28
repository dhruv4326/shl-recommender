"""
SHL Assessment Recommendation API
====================================
FastAPI + Pinecone + Gemini

Endpoints:
  GET  /health    → {"status": "healthy"}
  POST /recommend → list of 5-10 relevant assessments

Run locally:
    uvicorn main:app --reload --port 8000

Deploy on Render:
    Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import re
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Env & Logging ──────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX    = os.getenv("PINECONE_INDEX_NAME", "shl-assessments")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
GEMINI_URL        = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# ── Global state (loaded once at startup) ─────────────────────────────────────
_model: SentenceTransformer = None
_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup."""
    global _model, _index

    log.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
    _model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("✅ Embedding model ready.")

    log.info("Connecting to Pinecone index '%s'...", PINECONE_INDEX)
    pc     = Pinecone(api_key=PINECONE_API_KEY)
    _index = pc.Index(PINECONE_INDEX)
    stats  = _index.describe_index_stats()
    log.info("✅ Pinecone ready. %d vectors indexed.", stats["total_vector_count"])

    yield   # app is running

    log.info("Shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL assessments for a job description or natural language query.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url:              str
    name:             str
    adaptive_support: str
    description:      str
    duration:         Optional[int] = None
    remote_support:   str
    test_type:        list[str]

class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ══════════════════════════════════════════════════════════════════════════════

def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


async def fetch_url_text(url: str) -> str:
    """Fetch a URL and return plain text (strips HTML)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000]
    except Exception as e:
        log.warning("URL fetch failed: %s", e)
        return ""


async def expand_query(raw: str) -> str:
    """If query is a URL, fetch its content first."""
    if is_url(raw.strip()):
        log.info("Query is a URL — fetching content...")
        content = await fetch_url_text(raw.strip())
        return content if content else raw
    return raw


def retrieve_from_pinecone(query_text: str, top_k: int = 20) -> list[dict]:
    """Embed query → search Pinecone → return top_k candidates."""
    embedding = _model.encode([query_text]).tolist()[0]
    results   = _index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
    )
    candidates = []
    for match in results["matches"]:
        meta      = match["metadata"]
        test_type = [t.strip() for t in meta.get("test_type_str", "").split("|") if t.strip()]
        candidates.append({
            "name":             meta.get("name", ""),
            "url":              meta.get("url", ""),
            "description":      meta.get("description", ""),
            "duration":         meta.get("duration") or None,
            "remote_support":   meta.get("remote_support", "No"),
            "adaptive_support": meta.get("adaptive_support", "No"),
            "test_type":        test_type,
            "_score":           match["score"],
        })
    return candidates


async def rerank_with_gemini(query: str, candidates: list[dict]) -> list[dict]:
    """
    Ask Gemini to pick the best 5-10 assessments with balanced skill coverage.
    Falls back to top-10 by vector score if Gemini fails.
    """
    if not GEMINI_API_KEY:
        log.warning("No GEMINI_API_KEY — skipping reranking, returning top 10.")
        return candidates[:10]

    # Build candidate summary for the prompt
    summary = "\n".join([
        f"{i+1}. [{', '.join(c['test_type'])}] {c['name']} — {c['description'][:150]}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are an expert HR assessment consultant at SHL.

Given the job requirement below, select the BEST 5 to 10 assessments from the list.

STRICT RULES:
1. Return between 5 and 10 assessments (never fewer than 5).
2. BALANCE technical and behavioral assessments:
   - If the role needs technical skills → include Knowledge & Skills (K) type assessments.
   - If the role needs soft skills/personality → include Personality & Behavior (P) or Competencies (C).
   - For mixed roles → include BOTH types.
3. Prioritize assessments most directly relevant to the stated role and skills.
4. Return ONLY a JSON array of 1-based index numbers. Example: [1, 3, 5, 7, 9]
5. No explanation. No markdown. Just the JSON array.

JOB REQUIREMENT:
{query[:1500]}

CANDIDATE ASSESSMENTS:
{summary}

Selected numbers (JSON array only):"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 150},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
            )
            resp.raise_for_status()
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

            match = re.search(r"\[[\d,\s]+\]", raw_text)
            if not match:
                raise ValueError(f"No JSON array in response: {raw_text}")

            indices  = json.loads(match.group())
            selected = []
            seen     = set()

            for idx in indices:
                i = idx - 1
                if 0 <= i < len(candidates) and i not in seen:
                    selected.append(candidates[i])
                    seen.add(i)
                if len(selected) == 10:
                    break

            # Pad to 5 if Gemini returned too few
            if len(selected) < 5:
                for i, c in enumerate(candidates):
                    if i not in seen:
                        selected.append(c)
                        seen.add(i)
                    if len(selected) == 5:
                        break

            log.info("Gemini selected %d assessments.", len(selected))
            return selected

    except Exception as e:
        log.error("Gemini failed: %s — falling back to top 10.", e)
        return candidates[:10]


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info("Query: %s", query[:100])

    # Step 1: Expand URL queries
    query_text = await expand_query(query)

    # Step 2: Vector search — top 20 candidates
    candidates = retrieve_from_pinecone(query_text, top_k=20)
    log.info("Retrieved %d candidates from Pinecone.", len(candidates))

    if not candidates:
        raise HTTPException(status_code=404, detail="No assessments found.")

    # Step 3: Gemini reranking — pick best 5-10
    final = await rerank_with_gemini(query_text, candidates)

    # Step 4: Clean internal fields and return
    output = [
        {
            "url":              a["url"],
            "name":             a["name"],
            "adaptive_support": a.get("adaptive_support", "No"),
            "description":      a.get("description", ""),
            "duration":         a.get("duration"),
            "remote_support":   a.get("remote_support", "No"),
            "test_type":        a.get("test_type", []),
        }
        for a in final
    ]

    return {"recommended_assessments": output}


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)