from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from scrape_nbc import (
    _jsonld_canonical_url,
    _jsonld_objects_from_parsed,
    _nbc_author_to_string,
    _schema_types,
    urls_jsonld_compatible,
)

DEFAULT_CSV = ROOT / "data" / "raw" / "nbc_scraped_all.csv"
MIN_DELAY = 0.8
MAX_DELAY = 1.8


def http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=7,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return session


def value_is_missing(series_row: object) -> bool:
    if pd.isna(series_row):
        return True
    return str(series_row).strip() == ""


def branch_deeplink_authors_from_soup(soup: BeautifulSoup) -> str | None:
    """Collect branch:deeplink:authorName1, authorName2, ... until the first missing index."""
    parts: list[str] = []
    x = 1
    while x <= 500:
        tag = soup.find("meta", attrs={"name": f"branch:deeplink:authorName{x}"})
        if tag is None or not isinstance(tag, Tag):
            break
        raw = tag.get("content")
        if raw is None:
            break
        text = str(raw).strip()
        if text:
            parts.append(text)
        x += 1
    return ", ".join(parts) if parts else None


def _meta_text(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    attrs: dict[str, str] = {}
    if name is not None:
        attrs["name"] = name
    if prop is not None:
        attrs["property"] = prop
    tag = soup.find("meta", attrs=attrs)
    if not isinstance(tag, Tag):
        return None
    c = tag.get("content")
    if not c:
        return None
    s = str(c).strip()
    return s or None


def jsonld_date_published_for_page(soup: BeautifulSoup, page_url: str) -> str | None:
    """
    datePublished from LiveBlogPosting / NewsArticle / BlogPosting whose JSON-LD url matches
    the page (exact or parent shell for /rcrd… /ncrd… / ?canonicalCard= URLs).
    """
    live_dp: str | None = None
    news_dp: str | None = None
    blog_dp: str | None = None

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string if isinstance(script, Tag) else None
        if raw is None and isinstance(script, Tag):
            raw = script.get_text()
        if not raw or not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for root in _jsonld_objects_from_parsed(data):
            types = _schema_types(root)
            cu = _jsonld_canonical_url(root)
            if cu is None or not urls_jsonld_compatible(page_url, cu):
                continue
            dp = root.get("datePublished")
            if not isinstance(dp, str) or not dp.strip():
                continue
            dp = dp.strip()
            if "LiveBlogPosting" in types:
                live_dp = live_dp or dp
            if "NewsArticle" in types:
                news_dp = news_dp or dp
            if "BlogPosting" in types:
                blog_dp = blog_dp or dp

    return live_dp or news_dp or blog_dp


def jsonld_author_for_page(soup: BeautifulSoup, page_url: str) -> str | None:
    """Person / Organization author from JSON-LD blocks whose url matches the page."""
    live_au: str | None = None
    news_au: str | None = None
    blog_au: str | None = None

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string if isinstance(script, Tag) else None
        if raw is None and isinstance(script, Tag):
            raw = script.get_text()
        if not raw or not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for root in _jsonld_objects_from_parsed(data):
            types = _schema_types(root)
            cu = _jsonld_canonical_url(root)
            if cu is None or not urls_jsonld_compatible(page_url, cu):
                continue
            au = _nbc_author_to_string(root.get("author"))
            if not au:
                continue
            if "LiveBlogPosting" in types:
                live_au = live_au or au
            if "NewsArticle" in types:
                news_au = news_au or au
            if "BlogPosting" in types:
                blog_au = blog_au or au

    return live_au or news_au or blog_au


def backfill_fields_from_html(soup: BeautifulSoup, page_url: str) -> dict[str, str | None]:
    """Author + publish time from HTML (branch meta, JSON-LD, then cXense / Open Graph)."""
    authors = branch_deeplink_authors_from_soup(soup)
    datetime_posted = jsonld_date_published_for_page(soup, page_url)

    if not datetime_posted:
        datetime_posted = _meta_text(soup, name="cXenseParse:publishtime")
    if not datetime_posted:
        datetime_posted = _meta_text(soup, prop="article:published_time")

    if not authors:
        authors = jsonld_author_for_page(soup, page_url)
    if not authors:
        authors = _meta_text(soup, name="cXenseParse:author")

    return {"author": authors, "datePublished": datetime_posted}


def is_live_blog_url(url: str) -> bool:
    return "/live-blog/" in str(url).lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refetch pages and backfill missing author and/or datetime_posted. "
            "Default: all rows with gaps (JSON-LD loose URL match for live shell vs card URLs, "
            "Branch metas, BlogPosting, cXense). "
            "Use --live-only to restrict to /live-blog/ as before."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite --input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List rows that would trigger fetches without HTTP or writing",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Only rows whose URL contains /live-blog/ (legacy scope)",
    )
    args = parser.parse_args()
    inp = args.input.resolve()
    out = args.output.resolve() if args.output else inp

    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    df = pd.read_csv(inp)
    required = {"url", "author", "datetime_posted"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns: {sorted(required)}", file=sys.stderr)
        return 1

    live = df["url"].map(is_live_blog_url)
    miss_author = df["author"].map(value_is_missing)
    miss_dt = df["datetime_posted"].map(value_is_missing)

    mask = miss_author | miss_dt
    if args.live_only:
        mask = mask & live
    targets = df.loc[mask]
    unique_urls = targets["url"].drop_duplicates().tolist()

    scope = "live-blog URLs only" if args.live_only else "all URLs with gaps"
    print(f"Rows needing author and/or datetime_posted ({scope}): {len(targets)}")
    print(f"  of those, live-blog: {int((live & mask).sum())}; non-live: {int((~live & mask).sum())}")
    print(f"  CSV-wide missing author: {int(miss_author.sum())}; missing datetime_posted: {int(miss_dt.sum())}")
    print(f"Unique URLs to fetch: {len(unique_urls)}")

    if args.dry_run:
        for u in unique_urls[:50]:
            print(" ", u)
        if len(unique_urls) > 50:
            print(f"  ... and {len(unique_urls) - 50} more")
        return 0

    session = http_session()
    cache: dict[str, dict[str, str | None]] = {}
    filled_author = 0
    filled_dt = 0
    errors = 0

    for url in unique_urls:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        ukey = str(url).strip()
        try:
            r = session.get(ukey, timeout=(10, 45))
            if r.status_code != 200:
                print(f"HTTP {r.status_code}: {url}")
                errors += 1
                cache[ukey] = {"author": None, "datePublished": None}
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            extracted = backfill_fields_from_html(soup, ukey)
            authors = extracted["author"]
            date_pub = extracted["datePublished"]
            cache[ukey] = {"author": authors, "datePublished": date_pub}
            if authors:
                filled_author += 1
            if date_pub:
                filled_dt += 1
            ap = (authors[:80] + "…") if authors and len(authors) > 80 else authors
            print(
                f"author_meta={filled_author}/{len(unique_urls)} "
                f"jsonld_date={filled_dt}/{len(unique_urls)} | {ap or '-'} | {date_pub or '-'} | {ukey[:72]}"
            )
        except requests.RequestException as exc:
            print(f"Request error {ukey[:72]}: {exc}")
            errors += 1
            cache[ukey] = {"author": None, "datePublished": None}

    rows_author = 0
    rows_dt = 0
    for idx in targets.index:
        ukey = str(df.at[idx, "url"]).strip()
        blob = cache.get(ukey) or {}
        if value_is_missing(df.at[idx, "author"]) and blob.get("author"):
            df.at[idx, "author"] = blob["author"]
            rows_author += 1
        if value_is_missing(df.at[idx, "datetime_posted"]) and blob.get("datePublished"):
            df.at[idx, "datetime_posted"] = blob["datePublished"]
            rows_dt += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(
        f"Unique URLs with any author extracted: {filled_author} / {len(unique_urls)}; "
        f"with datetime extracted: {filled_dt} / {len(unique_urls)}"
    )
    print(f"CSV rows author filled: {rows_author}; datetime_posted filled: {rows_dt}; errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())