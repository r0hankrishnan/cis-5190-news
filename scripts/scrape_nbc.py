import csv
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
URL_CSV = DATA_DIR / "url_only_data.csv"


def http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def load_nbc_urls() -> list[str]:
    with URL_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["url"].strip() for row in reader if "nbcnews.com" in row["url"]]


def normalized_url(url: str) -> str:
    """Strip fragments (#anchors); server response is the same but storage/training stays consistent."""
    return urldefrag(url.strip())[0]


def is_live_blog_url(url: str) -> bool:
    """Rolling live coverage: many updates on one page — poor fit for single-article ML labels."""
    path = urlparse(url).path.lower()
    return "/live-blog/" in path


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
    rows: list[dict[str, str | bool | None]] = []
    for raw_url in load_nbc_urls():
        url = normalized_url(raw_url)
        is_live = is_live_blog_url(url)
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            print(f"\n{'-' * 70}\nSKIP (HTTP {response.status_code})\n  {url}\n{'-' * 70}")
            rows.append(
                {
                    "URL": url,
                    "Is Live Blog": is_live,
                    "Title": None,
                    "Author": None,
                    "Publish Time": None,
                    "Description": None,
                    "First Paragraph": None,
                }
            )
            continue
        soup = BeautifulSoup(response.text, "html.parser")
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
                "URL": url,
                "Is Live Blog": is_live,
                "Title": title,
                "Author": author,
                "Publish Time": publish_time,
                "Description": description,
                "First Paragraph": first_paragraph,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "URL",
            "Is Live Blog",
            "Title",
            "Author",
            "Publish Time",
            "Description",
            "First Paragraph",
        ],
    )
    print(df.head())
