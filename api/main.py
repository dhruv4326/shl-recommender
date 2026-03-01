"""
SHL Assessment Recommendation API
====================================
FastAPI + Pinecone + Gemini Embeddings + Gemini Reranking
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

# ── Env & Logging ──────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "shl-assessments")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
GEMINI_GEN_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    log.info("Connecting to Pinecone index '%s'...", PINECONE_INDEX)
    log.info("PINECONE_API_KEY set: %s", bool(PINECONE_API_KEY))
    log.info("GEMINI_API_KEY set: %s", bool(GEMINI_API_KEY))
    pc     = Pinecone(api_key=PINECONE_API_KEY)
    _index = pc.Index(PINECONE_INDEX)
    stats  = _index.describe_index_stats()
    log.info("✅ Pinecone ready. %d vectors indexed.", stats["total_vector_count"])
    log.info("Index dimension: %s", stats.get("dimension", "unknown"))
    yield
    log.info("Shutting down.")


app = FastAPI(title="SHL Assessment Recommendation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# Gemini Embedding
# ══════════════════════════════════════════════════════════════════════════════

async def get_gemini_embedding(text: str) -> list[float]:
    """Get embedding from Gemini gemini-embedding-001 (3072 dim)."""
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text[:2000]}]},
    }
    log.info("Calling Gemini embedding API...")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GEMINI_EMBED_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )
        log.info("Gemini embed response status: %d", resp.status_code)
        if resp.status_code != 200:
            log.error("Gemini embed error: %s", resp.text[:300])
            raise HTTPException(status_code=500, detail=f"Gemini embedding failed: {resp.text[:200]}")
        result = resp.json()
        values = result["embedding"]["values"]
        log.info("Embedding received, dim=%d", len(values))
        return values


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))


async def fetch_url_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", resp.text)
            return re.sub(r"\s+", " ", text).strip()[:3000]
    except Exception as e:
        log.warning("URL fetch failed: %s", e)
        return ""


async def expand_query(raw: str) -> str:
    if is_url(raw.strip()):
        content = await fetch_url_text(raw.strip())
        return content if content else raw
    return raw


async def retrieve_from_pinecone(query_text: str, top_k: int = 20) -> list[dict]:
    embedding = await get_gemini_embedding(query_text)
    log.info("Querying Pinecone with top_k=%d...", top_k)
    results   = _index.query(vector=embedding, top_k=top_k, include_metadata=True)
    log.info("Pinecone returned %d matches.", len(results["matches"]))
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
    if not GEMINI_API_KEY:
        return candidates[:10]

    summary = "\n".join([
        f"{i+1}. [{', '.join(c['test_type'])}] {c['name']} — {c['description'][:150]}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""You are an expert HR assessment consultant at SHL.

Given the job requirement below, select the BEST 5 to 10 assessments from the list.

STRICT RULES:
1. Return between 5 and 10 assessments (never fewer than 5).
2. BALANCE technical and behavioral assessments:
   - Technical skills needed → include Knowledge & Skills (K) assessments.
   - Soft skills/personality needed → include Personality & Behavior (P) or Competencies (C).
   - Mixed roles → include BOTH types.
3. Return ONLY a JSON array of 1-based index numbers like: [1, 3, 5, 7, 9]
4. No explanation. No markdown. Just the JSON array.

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
        log.info("Calling Gemini reranking...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GEMINI_GEN_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
            )
            log.info("Gemini rerank status: %d", resp.status_code)
            resp.raise_for_status()
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            log.info("Gemini rerank response: %s", raw_text[:100])

        match = re.search(r"\[[\d,\s]+\]", raw_text)
        if not match:
            raise ValueError(f"No JSON array in: {raw_text}")

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

        if len(selected) < 5:
            for i, c in enumerate(candidates):
                if i not in seen:
                    selected.append(c)
                    seen.add(i)
                if len(selected) == 5:
                    break

        log.info("Final selection: %d assessments.", len(selected))
        return selected

    except Exception as e:
        log.error("Gemini rerank failed: %s — using top 10.", e)
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

    log.info("=== New request: %s", query[:100])

    try:
        query_text = await expand_query(query)
        candidates = await retrieve_from_pinecone(query_text, top_k=20)

        if not candidates:
            raise HTTPException(status_code=404, detail="No assessments found.")

        final = await rerank_with_gemini(query_text, candidates)

        output = [{
            "url":              a["url"],
            "name":             a["name"],
            "adaptive_support": a.get("adaptive_support", "No"),
            "description":      a.get("description", ""),
            "duration":         a.get("duration"),
            "remote_support":   a.get("remote_support", "No"),
            "test_type":        a.get("test_type", []),
        } for a in final]

        log.info("=== Returning %d assessments.", len(output))
        return {"recommended_assessments": output}

    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)