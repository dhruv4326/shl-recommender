"""
Generate Predictions CSV for Test Set
========================================
Runs the recommendation API on all test queries and outputs predictions.csv
in the exact format required for submission.

Output format:
    query,Assessment_url
    Query 1,https://...
    Query 1,https://...
    Query 2,https://...
    ...

Run:
    python generate_predictions.py --test_csv path/to/test.csv
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000/recommend"
TEST_CSV = Path(__file__).parent.parent / "data" / "test.csv"   # edit if needed
OUT_CSV  = Path(__file__).parent.parent / "predictions.csv"


async def get_recommendations(query: str) -> list[str]:
    """Call the /recommend API and return list of recommended URLs."""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(API_URL, json={"query": query})
            resp.raise_for_status()
            data = resp.json()
            return [a["url"] for a in data.get("recommended_assessments", [])]
        except Exception as exc:
            print(f"  ERROR for query '{query[:50]}': {exc}")
            return []


async def generate(test_csv: Path):
    # Load test queries
    df = pd.read_csv(test_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    query_col  = next(c for c in df.columns if "query" in c)

    # Deduplicate queries (maintain order)
    queries = list(dict.fromkeys(df[query_col].dropna().astype(str).str.strip().tolist()))
    print(f"Found {len(queries)} unique test queries")

    # Generate predictions
    rows = []
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {query[:70]}")
        urls = await get_recommendations(query)
        print(f"          → {len(urls)} recommendations")
        for url in urls:
            rows.append({"query": query, "Assessment_url": url})

    # Save
    out_df = pd.DataFrame(rows, columns=["query", "Assessment_url"])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Predictions saved → {OUT_CSV}")
    print(f"   Total rows: {len(out_df)}")
    print(out_df.head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default=str(TEST_CSV))
    args = parser.parse_args()

    csv_path = Path(args.test_csv)
    if not csv_path.exists():
        print(f"ERROR: Test CSV not found at {csv_path}")
        print("Download it from the assignment link and place at data/test.csv")
        sys.exit(1)

    asyncio.run(generate(csv_path))