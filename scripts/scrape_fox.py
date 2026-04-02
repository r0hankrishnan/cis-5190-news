import csv
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

# ─── Config ───────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
FAILED_DIR = RAW_DIR / "fox_failed"
SCRAPED_DIR = RAW_DIR / "fox_scraped"

INPUT_PATH = Path(__file__).parent.parent / "data" / "external" / "fox_news_urls.csv" # Set this to overall fox links file or most recent failed file

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = SCRAPED_DIR/ f"fox_scraped_{TIMESTAMP}.csv"
FAILED_FILE = FAILED_DIR / f"fox_failed_urls_{TIMESTAMP}.csv"

MAX_WORKERS = 5
FIELDNAMES = ["url", "topic", "title", "subtitle", "author", "datetime_posted", "label"]

SEEN_FILE = None # Set this to the most recent successful output file to avoid re-scraping already scraped links

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Session ──────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
})

# ─── Scraper ──────────────────────────────────────────────────────────────────
class BadStatusError(Exception):
    pass

@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, BadStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def scrape(link):
    time.sleep(random.uniform(2, 4))

    response = session.get(link, timeout=10)

    if response.status_code != 200:
        raise BadStatusError(f"Failed to load page: Status code {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    if "/live-news/" in link:
        return scrape_live(soup, link)
    
    topic_span = soup.find("span", class_="eyebrow")
    topic = topic_span.get_text(strip=True) if topic_span else None

    title_h1 = soup.find("h1", class_="headline speakable")
    title = title_h1.get_text(strip=True) if title_h1 else None

    subtitle_h2 = soup.find("h2", class_="sub-headline speakable")
    subtitle = subtitle_h2.get_text(strip=True) if subtitle_h2 else None

    author_div = soup.find("div", class_="author-byline")
    a_tags = author_div.find_all("a") if isinstance(author_div, Tag) else []
    author = a_tags[0].get_text(strip = True) if a_tags else None
    
    date_posted_time = soup.find("time")
    datetime_posted = date_posted_time.get("datetime") if isinstance(date_posted_time, Tag) else None

    return {
        "url": link,
        "topic": topic,
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "datetime_posted": datetime_posted,
        "label": "Fox News",
    }
    
# Alternative scraping function for live news pages which have a diff structure

def scrape_live(soup, link):
    # Title from og:title meta tag
    title_meta = soup.find("meta", property="og:title")
    title = title_meta.get("content") if isinstance(title_meta, Tag) else None

    # Subtitle from og:description meta tag
    subtitle_meta = soup.find("meta", property="og:description")
    subtitle = subtitle_meta.get("content") if isinstance(subtitle_meta, Tag) else None

    # Topic from prism.section meta tag
    topic_meta = soup.find("meta", attrs={"name": "prism.section"})
    topic = topic_meta.get("content") if isinstance(topic_meta, Tag) else None
    
    # Date and authors from JSON-LD script tag
    script = soup.find("script", attrs={"type": "application/ld+json"})
    datetime_posted = None
    author = None
    if isinstance(script, Tag) and script.string:
        data = json.loads(script.string)
        datetime_posted = data.get("datePublished")
        authors = data.get("author", [])
        if isinstance(authors, list):
            author = ", ".join(a.get("name", "") for a in authors)
        elif isinstance(authors, dict):
            author = authors.get("name")

    return {
        "url": link,
        "topic": topic,
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "datetime_posted": datetime_posted,
        "label": "Fox News",
    }


# ─── Load Links ───────────────────────────────────────────────────────────────

def load_links(filepath):
    # Check if file is csv
    if not filepath.suffix.lower() == ".csv":
        raise ValueError(f"Input file must be a CSV: {filepath}")
    
    # Load with pandas and check that 'url' column exists
    df = pd.read_csv(filepath)
    if "url" not in df.columns:
        raise ValueError(f"CSV file must contain 'url' column: {filepath}")
    
    urls = df["url"].str.strip().str.replace(r"\.print$", "", regex=True).tolist() # remove .print suffixes

    return urls

# ─── Load Seen URLs ───────────────────────────────────────────────────────────────

def load_seen_urls(filepath: str | None) -> set:
    if filepath is None or not Path(filepath).exists():
        return set()
    
    df = pd.read_csv(filepath)
    if "url" not in df.columns:
        return set()
    seen = set(df["url"].str.strip())
    log.info(f"Loaded {len(seen)} seen URLs from {filepath}")
    return seen

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Make sure RAW_DIR and FAILED_DIR exist
    RAW_DIR.mkdir(parents =True, exist_ok =True)
    FAILED_DIR.mkdir(parents =True, exist_ok =True)
    SCRAPED_DIR.mkdir(parents = True, exist_ok = True)
    
    # Get unseen links
    seen = load_seen_urls(str(SEEN_FILE))
    links = [link for link in load_links(INPUT_PATH) if link not in seen] # Select only unseen links
    total = len(links)
    log.info(f"Loaded {total} links from {INPUT_PATH}")
    log.info(f"Output: {OUTPUT_FILE} | Failed: {FAILED_FILE}")

    success_count = 0
    fail_count = 0

    # Start scraping
    with (
        open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f,
        open(FAILED_FILE, "w", newline="", encoding="utf-8") as fail_f,
    ):
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        failed_writer = csv.DictWriter(fail_f, fieldnames=["url", "error"])
        writer.writeheader()
        failed_writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(scrape, link): link for link in links}

            for future in as_completed(futures):
                link = futures[future]
                try:
                    result = future.result()
                    writer.writerow(result)
                    out_f.flush()
                    success_count += 1
                    log.info(f"[{success_count + fail_count}/{total}] OK: {link}")
                except Exception as e:
                    failed_writer.writerow({"url": link, "error": str(e)})
                    fail_f.flush()
                    fail_count += 1
                    log.warning(f"[{success_count + fail_count}/{total}] FAILED: {link} — {e}")

    log.info(f"Done. {success_count} succeeded, {fail_count} failed.")
    if fail_count:
        log.info(f"Failed links saved to {FAILED_FILE} — rerun against that file to retry.")

if __name__ == "__main__":
    main()