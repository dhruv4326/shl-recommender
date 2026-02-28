"""
SHL Assessment Catalog Scraper — FIXED VERSION
================================================
Robust parser with multiple fallback strategies.

Run:
    python scraper.py
"""

import os, re, json, time, logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_URL        = "https://www.shl.com"
CATALOG_URL     = f"{BASE_URL}/products/product-catalog/"
PAGE_SIZE       = 12
INDIVIDUAL_TYPE = 1
DELAY           = 1.5

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

PREPACKAGED_KEYWORDS = ["pre-packaged", "pre packaged", "job solution", "packaged job"]
INDIVIDUAL_KEYWORDS  = ["individual test", "individual"]


def fetch_soup(session, url, params=None):
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as exc:
            log.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(4)
    raise RuntimeError(f"Failed to fetch {url}")


def get_last_page_start(soup):
    max_start = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"type={INDIVIDUAL_TYPE}" in href and "start=" in href:
            try:
                val = int(href.split("start=")[1].split("&")[0])
                max_start = max(max_start, val)
            except ValueError:
                pass
    return max_start


def get_heading_before(table):
    for tag in ["h2", "h3", "h4", "h1", "p"]:
        el = table.find_previous(tag)
        if el:
            return el.get_text(separator=" ", strip=True).lower()
    return ""


def table_has_shl_links(table):
    return bool(table.find("a", href=re.compile(r"/product-catalog/view/")))


def _has_indicator(td):
    if td.find(["img", "svg"]):
        return "Yes"
    for span in td.find_all("span"):
        cls = " ".join(span.get("class", []))
        text = span.get_text(strip=True)
        if any(kw in cls.lower() for kw in ["circle", "tick", "check", "yes", "icon"]):
            return "Yes"
        if text and cls:
            return "Yes"
    for el in td.find_all(True):
        if el.get("role") == "img":
            return "Yes"
        if "yes" in el.get("aria-label", "").lower():
            return "Yes"
    return "No"


def parse_row(tr):
    tds = tr.find_all("td")
    if len(tds) < 2:
        return None
    if tr.find("th"):
        return None

    a_tag = tds[0].find("a", href=True)
    if not a_tag:
        return None

    name = a_tag.get_text(strip=True)
    if not name:
        return None

    href = a_tag["href"]
    url  = href if href.startswith("http") else BASE_URL + href

    remote   = _has_indicator(tds[1]) if len(tds) > 1 else "No"
    adaptive = _has_indicator(tds[2]) if len(tds) > 2 else "No"

    test_types_raw = []
    if len(tds) > 3:
        for span in tds[3].find_all("span"):
            letter = span.get_text(strip=True).upper()
            if letter in TEST_TYPE_MAP:
                test_types_raw.append(letter)
        if not test_types_raw:
            cell_text = tds[3].get_text(strip=True).upper()
            for letter in TEST_TYPE_MAP:
                if letter in cell_text:
                    test_types_raw.append(letter)

    return {
        "name": name, "url": url,
        "remote_testing": remote, "adaptive_irt": adaptive,
        "test_types_raw": test_types_raw,
    }


def parse_individual_table(soup, page_num=0):
    all_tables = soup.find_all("table")
    if not all_tables:
        log.warning("Page %d: No <table> found!", page_num)
        return []

    target = None

    # Strategy 1: heading contains "individual"
    for t in all_tables:
        h = get_heading_before(t)
        if any(kw in h for kw in INDIVIDUAL_KEYWORDS) and table_has_shl_links(t):
            target = t
            break

    # Strategy 2: not pre-packaged, has SHL links
    if target is None:
        for t in all_tables:
            h = get_heading_before(t)
            if not any(kw in h for kw in PREPACKAGED_KEYWORDS) and table_has_shl_links(t):
                target = t
                break

    # Strategy 3: any table with SHL links (last resort)
    if target is None:
        for t in all_tables:
            if table_has_shl_links(t):
                target = t
                break

    # Strategy 4: pick the SECOND table (SHL page layout: prepackaged first, individual second)
    if target is None and len(all_tables) >= 2:
        for t in all_tables:
            if table_has_shl_links(t):
                # skip first, take second
                if target is None:
                    target = t  # set first
                else:
                    target = t  # override with second
                    break

    if target is None:
        log.warning("Page %d: Could not find individual table.", page_num)
        debug_path = OUTPUT_DIR / f"debug_page_{page_num}.html"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(str(soup))
        log.warning("Debug HTML saved → %s", debug_path)
        return []

    records = []
    for tr in target.find_all("tr"):
        record = parse_row(tr)
        if record:
            records.append(record)
    return records


def scrape_detail_page(session, url):
    detail = {"description": "", "duration": None}
    try:
        time.sleep(DELAY)
        soup = fetch_soup(session, url)

        for sel in [
            {"class": re.compile(r"pdp__body", re.I)},
            {"class": re.compile(r"product.*description", re.I)},
            {"class": re.compile(r"rich.text", re.I)},
            {"class": re.compile(r"description", re.I)},
        ]:
            el = soup.find("div", sel)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 40:
                    detail["description"] = text[:600]
                    break

        if not detail["description"]:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    detail["description"] = text[:600]
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
    except Exception as exc:
        log.warning("Detail failed for %s: %s", url, exc)
    return detail


def build_record(raw):
    return {
        "name":             raw["name"],
        "url":              raw["url"],
        "description":      raw.get("description", ""),
        "duration":         raw.get("duration"),
        "remote_support":   raw.get("remote_testing", "No"),
        "adaptive_support": raw.get("adaptive_irt", "No"),
        "test_type":        [TEST_TYPE_MAP.get(l, l) for l in raw.get("test_types_raw", [])],
    }


def main(fetch_details=True):
    session = requests.Session()
    all_raw = []

    log.info("Phase 1: Scraping listing pages …")
    first_soup  = fetch_soup(session, CATALOG_URL, params={"start": 0, "type": INDIVIDUAL_TYPE})
    max_start   = get_last_page_start(first_soup)
    total_pages = (max_start // PAGE_SIZE) + 1
    log.info("Detected %d pages (max_start=%d)", total_pages, max_start)

    rows = parse_individual_table(first_soup, page_num=1)
    log.info("Page 1/%d: %d items", total_pages, len(rows))
    all_raw.extend(rows)

    for idx in range(1, total_pages):
        start = idx * PAGE_SIZE
        time.sleep(DELAY)
        soup = fetch_soup(session, CATALOG_URL, params={"start": start, "type": INDIVIDUAL_TYPE})
        rows = parse_individual_table(soup, page_num=idx + 1)
        log.info("Page %d/%d: %d items", idx + 1, total_pages, len(rows))
        all_raw.extend(rows)

    log.info("Total from listing: %d assessments", len(all_raw))

    if len(all_raw) == 0:
        log.error("ZERO items found! Run debug_html.py first and check the saved HTML.")
        return []

    if fetch_details:
        log.info("Phase 2: Fetching %d detail pages …", len(all_raw))
        for item in tqdm(all_raw, desc="Detail pages"):
            item.update(scrape_detail_page(session, item["url"]))
    else:
        for item in all_raw:
            item["description"] = ""
            item["duration"]    = None

    final_records = [build_record(r) for r in all_raw]

    json_path = OUTPUT_DIR / "shl_assessments.json"
    csv_path  = OUTPUT_DIR / "shl_assessments.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
    log.info("Saved JSON → %s", json_path)

    df = pd.DataFrame(final_records)
    df["test_type"] = df["test_type"].apply(lambda x: " | ".join(x) if isinstance(x, list) else (x or ""))
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV  → %s", csv_path)

    count = len(final_records)
    log.info("=" * 50)
    log.info("TOTAL SCRAPED: %d", count)
    log.info("✅ Meets 377+ requirement!" if count >= 377 else f"⚠️  Only {count} — check debug HTML.")

    if final_records:
        print("\n── Sample record ──")
        print(json.dumps(final_records[0], indent=2))

    return final_records


if __name__ == "__main__":
    # Change to True after confirming items are found
    main(fetch_details=True)