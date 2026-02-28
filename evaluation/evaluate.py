"""
Evaluation — Mean Recall@10
=============================
Measures how well the recommendation system performs on the labeled train set.

The train set CSV should have columns:
    query | Assessment_url

Run:
    python evaluate.py --train_csv path/to/train.csv

Or edit TRAIN_CSV below and run:
    python evaluate.py
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL   = "http://localhost:8000/recommend"
TRAIN_CSV = Path(__file__).parent.parent / "data" / "train.csv"   # edit path if needed
K         = 10


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_train_data(csv_path: Path) -> dict[str, list[str]]:
    """
    Load the labeled train CSV.
    Expected columns: query, Assessment_url  (case-insensitive)
    Returns: {query_str: [relevant_url1, relevant_url2, ...]}
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize column names
    query_col = next(c for c in df.columns if "query" in c)
    url_col   = next(c for c in df.columns if "url" in c or "assessment" in c)

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        q   = str(row[query_col]).strip()
        url = str(row[url_col]).strip()
        grouped[q].append(url)

    print(f"Loaded {len(grouped)} unique queries from {csv_path}")
    return dict(grouped)


def normalize_url(url: str) -> str:
    """Normalize URLs for comparison (strip trailing slashes, lowercase)."""
    return url.strip().rstrip("/").lower()


async def get_recommendations(query: str) -> list[str]:
    """Call the /recommend API and return list of recommended URLs."""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(API_URL, json={"query": query})
            resp.raise_for_status()
            data = resp.json()
            return [a["url"] for a in data.get("recommended_assessments", [])]
        except Exception as exc:
            print(f"  API error for query '{query[:50]}': {exc}")
            return []


def recall_at_k(relevant: list[str], predicted: list[str], k: int = 10) -> float:
    """Compute Recall@K for a single query."""
    if not relevant:
        return 0.0
    relevant_norm  = {normalize_url(u) for u in relevant}
    predicted_norm = [normalize_url(u) for u in predicted[:k]]
    hits = sum(1 for u in predicted_norm if u in relevant_norm)
    return hits / len(relevant_norm)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

async def evaluate(csv_path: Path):
    print(f"\n{'='*60}")
    print("SHL Assessment Recommendation — Evaluation")
    print(f"Train CSV : {csv_path}")
    print(f"API URL   : {API_URL}")
    print(f"K         : {K}")
    print(f"{'='*60}\n")

    # Load ground truth
    ground_truth = load_train_data(csv_path)

    results = []
    for i, (query, relevant_urls) in enumerate(ground_truth.items()):
        print(f"[{i+1}/{len(ground_truth)}] Query: {query[:70]}…")
        print(f"          Relevant assessments: {len(relevant_urls)}")

        predicted_urls = await get_recommendations(query)
        print(f"          Predicted assessments: {len(predicted_urls)}")

        r_at_k = recall_at_k(relevant_urls, predicted_urls, k=K)
        print(f"          Recall@{K}: {r_at_k:.4f}\n")

        results.append({
            "query":           query,
            "num_relevant":    len(relevant_urls),
            "num_predicted":   len(predicted_urls),
            f"recall@{K}":     r_at_k,
            "predicted_urls":  predicted_urls,
            "relevant_urls":   relevant_urls,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    recalls = [r[f"recall@{K}"] for r in results]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

    print("=" * 60)
    print(f"RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  • [{r[f'recall@{K}']:.3f}] {r['query'][:60]}")
    print(f"\n  Mean Recall@{K}: {mean_recall:.4f}  ({mean_recall*100:.1f}%)")
    print("=" * 60)

    # Save detailed results
    out_path = Path(__file__).parent / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump({"mean_recall_at_k": mean_recall, "k": K, "details": results}, f, indent=2)
    print(f"\nDetailed results saved → {out_path}")

    return mean_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=str(TRAIN_CSV), help="Path to labeled train CSV")
    args = parser.parse_args()

    csv_path = Path(args.train_csv)
    if not csv_path.exists():
        print(f"ERROR: Train CSV not found at {csv_path}")
        print("Download it from the assignment link and place it at data/train.csv")
        sys.exit(1)

    asyncio.run(evaluate(csv_path))