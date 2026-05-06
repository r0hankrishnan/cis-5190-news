from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
URL_CSV = DATA_DIR / "url_only_data.csv"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "nbc_scraped_all.csv"
MIN_REQUEST_DELAY_SECONDS = 0.8
MAX_REQUEST_DELAY_SECONDS = 1.8


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
    # Mimic a regular browser to reduce anti-bot blocking.
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


def load_nbc_urls() -> list[str]:
    with URL_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["url"].strip() for row in reader if "nbcnews.com" in row["url"]]


def normalized_url(url: str) -> str:
    """Strip fragments (#anchors); server response is the same but storage/training stays consistent."""
    return urldefrag(url.strip())[0]


def is_live_blog_url(url: str) -> bool:
    """Rolling live coverage: one CSV row per page, using NBC's JSON-LD (LiveBlogPosting / NewsArticle)."""
    path = urlparse(url).path.lower()
    return "/live-blog/" in path


def _url_match_key(url: str) -> str:
    base = urldefrag(url.strip())[0].rstrip("/")
    return base.lower()


def urls_jsonld_compatible(page_url: str, entity_url: str | None) -> bool:
    """
    True if JSON-LD `url` / @id refers to the same story as the requested page.
    NBC card URLs add path segments (e.g. /rcrd…, /ncrd…) and ?canonicalCard= while
    LiveBlogPosting / NewsArticle JSON-LD still points at the shell URL.
    """
    if not entity_url or not isinstance(entity_url, str):
        return False
    pu = urldefrag(page_url.strip())[0]
    eu = urldefrag(entity_url.strip())[0]
    if pu.lower().rstrip("/") == eu.lower().rstrip("/"):
        return True
    pl = urlparse(pu.lower())
    el = urlparse(eu.lower())
    if pl.netloc != el.netloc:
        return False
    ep = el.path.rstrip("/")
    pp = pl.path.rstrip("/")
    if not ep:
        return False
    return pp == ep or pp.startswith(ep + "/")


def _schema_types(obj: dict) -> frozenset[str]:
    t = obj.get("@type")
    if isinstance(t, str):
        return frozenset({t})
    if isinstance(t, list):
        return frozenset(x for x in t if isinstance(x, str))
    return frozenset()


def _jsonld_canonical_url(obj: dict) -> str | None:
    u = obj.get("url")
    if isinstance(u, str) and u.strip():
        return u.strip()
    mep = obj.get("mainEntityOfPage")
    if isinstance(mep, dict):
        mid = mep.get("@id")
        if isinstance(mid, str) and mid.strip():
            return urldefrag(mid.strip())[0]
    return None


def _jsonld_objects_from_parsed(data: object) -> list[dict]:
    out: list[dict] = []
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict):
                    out.append(item)
            if not out:
                out.append(data)
        else:
            out.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out.append(item)
    return out


def _nbc_author_to_string(author_field: object) -> str | None:
    """Schema.org author: Person / Organization / NewsMediaOrganization, list or single."""
    if author_field is None:
        return None
    if isinstance(author_field, list):
        parts: list[str] = []
        for entry in author_field:
            s = _nbc_author_to_string(entry)
            if s:
                parts.append(s)
        return ", ".join(parts) if parts else None
    if isinstance(author_field, dict):
        name = author_field.get("name")
        if name:
            return str(name).strip() or None
        return None
    if isinstance(author_field, str):
        s = author_field.strip()
        return s or None
    return None


def _nbc_datetime_from_article(obj: dict) -> str | None:
    for key in ("datePublished", "dateModified", "coverageStartTime"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _nbc_headline_description(obj: dict) -> tuple[str | None, str | None]:
    headline = obj.get("headline") or obj.get("alternativeHeadline")
    if not isinstance(headline, str):
        headline = None
    else:
        headline = headline.strip() or None
    desc = obj.get("description")
    if not isinstance(desc, str):
        desc = None
    else:
        desc = desc.strip() or None
    return headline, desc


def _find_nbc_live_blog_entities(soup: BeautifulSoup, page_url: str) -> tuple[dict | None, dict | None]:
    """
    Locate NBC page-level LiveBlogPosting and NewsArticle blocks whose URL matches this page.
    Ignores VideoObject, Dataset, and embedded liveBlogUpdate BlogPostings.
    """
    live_posting: dict | None = None
    news_article: dict | None = None

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
            if "LiveBlogPosting" in types and live_posting is None:
                live_posting = root
            if "NewsArticle" in types and news_article is None:
                news_article = root

    return live_posting, news_article


def parse_live_blog_page_nbc(soup: BeautifulSoup, url: str) -> dict[str, str | None]:
    """
    One row per live-blog URL using NBC's schema.org JSON-LD (LiveBlogPosting + NewsArticle),
    with Open Graph / meta fallbacks. Does not expand liveBlogUpdate into separate rows.
    """
    topic: str | None = None
    topic_meta = soup.find("meta", attrs={"name": "prism.section"})
    if isinstance(topic_meta, Tag):
        c = topic_meta.get("content")
        topic = str(c).strip() if c else None
    if not topic:
        topic = parse_topic_from_url(url)

    live_posting, news_article = _find_nbc_live_blog_entities(soup, url)

    ld_title: str | None = None
    ld_subtitle: str | None = None
    ld_dt: str | None = None
    ld_author: str | None = None

    for block in (live_posting, news_article):
        if not isinstance(block, dict):
            continue
        h, d = _nbc_headline_description(block)
        ld_title = ld_title or h
        ld_subtitle = ld_subtitle or d
        if ld_dt is None:
            ld_dt = _nbc_datetime_from_article(block)
        if ld_author is None:
            ld_author = _nbc_author_to_string(block.get("author"))

    og_title: str | None = None
    title_meta = soup.find("meta", property="og:title")
    if isinstance(title_meta, Tag):
        c = title_meta.get("content")
        og_title = str(c).strip() if c else None

    og_desc: str | None = None
    subtitle_meta = soup.find("meta", property="og:description")
    if isinstance(subtitle_meta, Tag):
        c = subtitle_meta.get("content")
        og_desc = str(c).strip() if c else None

    title = ld_title or og_title or parse_title(soup)
    subtitle = ld_subtitle or og_desc or parse_meta_item(soup, "description")
    datetime_posted = ld_dt
    author = ld_author

    return {
        "url": url,
        "topic": topic,
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "datetime_posted": datetime_posted,
        "label": "NBC News",
    }


def parse_topic_from_url(url: str) -> str | None:
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    topic = path.split("/", 1)[0].strip().lower()
    return topic or None


def parse_title(soup: BeautifulSoup) -> str | None:
    if soup.title is None or soup.title.string is None:
        return None
    return soup.title.string.strip()


def parse_meta_item(soup: BeautifulSoup, meta_name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": meta_name})
    if tag is None:
        tag = soup.find("meta", attrs={"property": meta_name})
    if tag is None:
        return None
    return tag.get("content")


def parse_first_paragraph(soup: BeautifulSoup) -> str | None:
    root = soup.find("article") or soup.body or soup
    p = root.find("p")
    if p is None:
        return None
    text = p.get_text(separator=" ", strip=True)
    return text if text else None


def _preview(value: str | None, max_len: int = 200) -> str:
    if value is None or not str(value).strip():
        return "(none)"
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _print_article_block(
    url: str,
    title: str | None,
    author: str | None,
    publish_time: str | None,
    description: str | None,
    first_paragraph: str | None,
    *,
    is_live_blog: bool = False,
) -> None:
    bar = "-" * 72
    kind = "live blog (rolling updates)" if is_live_blog else "article"
    print(
        f"\n{bar}\n"
        f"URL\n  {url}\n"
        f"Page kind\n  {kind}\n"
        f"Title\n  {_preview(title, 500)}\n"
        f"Author\n  {_preview(author, 200)}\n"
        f"Published\n  {_preview(publish_time, 120)}\n"
        f"Description\n  {_preview(description, 320)}\n"
        f"First paragraph\n  {_preview(first_paragraph, 400)}\n"
        f"{bar}"
    )


if __name__ == "__main__":
    session = http_session()
    rows: list[dict[str, str | None]] = []
    for raw_url in load_nbc_urls():
        time.sleep(random.uniform(MIN_REQUEST_DELAY_SECONDS, MAX_REQUEST_DELAY_SECONDS))
        url = normalized_url(raw_url)
        is_live = is_live_blog_url(url)
        topic = parse_topic_from_url(url)
        try:
            response = session.get(url, timeout=(10, 45))
        except requests.exceptions.RequestException as exc:
            print(f"\n{'-' * 70}\nSKIP (REQUEST ERROR)\n  {url}\n  {exc}\n{'-' * 70}")
            rows.append(
                {
                    "url": url,
                    "topic": topic,
                    "title": None,
                    "subtitle": None,
                    "author": None,
                    "datetime_posted": None,
                    "label": "NBC News",
                }
            )
            continue
        if response.status_code != 200:
            print(f"\n{'-' * 70}\nSKIP (HTTP {response.status_code})\n  {url}\n{'-' * 70}")
            rows.append(
                {
                    "url": url,
                    "topic": topic,
                    "title": None,
                    "subtitle": None,
                    "author": None,
                    "datetime_posted": None,
                    "label": "NBC News",
                }
            )
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        if is_live:
            live_row = parse_live_blog_page_nbc(soup, url)
            title = live_row["title"]
            author = live_row["author"]
            publish_time = live_row["datetime_posted"]
            description = live_row["subtitle"]
            if live_row.get("topic"):
                topic = live_row["topic"]
            first_paragraph = parse_first_paragraph(soup)
        else:
            title = parse_title(soup)
            author = parse_meta_item(soup, "cXenseParse:author")
            publish_time = parse_meta_item(soup, "cXenseParse:publishtime")
            description = parse_meta_item(soup, "description")
            first_paragraph = parse_first_paragraph(soup)
        _print_article_block(
            url, title, author, publish_time, description, first_paragraph, is_live_blog=is_live
        )
        rows.append(
            {
                "url": url,
                "topic": topic,
                "title": title,
                "subtitle": description,
                "author": author,
                "datetime_posted": publish_time,
                "label": "NBC News",
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "url",
            "topic",
            "title",
            "subtitle",
            "author",
            "datetime_posted",
            "label",
        ],
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(df.head())
