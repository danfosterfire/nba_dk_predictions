"""Scrape last-day ESPN Fantasy Basketball articles and persist as markdown.

Fetches articles from ESPN's public JSON API (the same network layer that
powers the fantasy page), filters to the past N hours, optionally fetches
full article bodies via HTML, then writes a single timestamped markdown
digest to data/raw/espn_fantasy/.

Usage:
    python -m src.data.espn_fantasy              # last 24h
    python -m src.data.espn_fantasy --hours 48
    python -m src.data.espn_fantasy --output-dir path/to/dir
    python -m src.data.espn_fantasy --no-fetch-bodies
"""

import argparse
import re
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ESPN_NEWS_API = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news?limit=50"
)
ESPN_FANTASY_URL = "https://www.espn.com/fantasy/mens-basketball/"
ESPN_BASE = "https://www.espn.com"

# No User-Agent override, for the reason given in injuries.py: ESPN's edge 403s
# browser-impersonating agents and serves `python-requests/x.y`. That restores the JSON
# APIs; ESPN_FANTASY_URL is an HTML page and stays 403 either way.
_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch_articles_api(url: str = ESPN_NEWS_API, timeout: int = 30) -> list[dict]:
    """Call ESPN's NBA news JSON API and return a list of normalized article dicts."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return [_normalize_api_article(item) for item in resp.json().get("articles", [])]


def _normalize_api_article(item: dict) -> dict:
    headline = item.get("headline") or item.get("title", "")
    description = item.get("description") or item.get("subHeadline", "")
    published = item.get("published") or item.get("lastModified") or ""
    story = item.get("story") or ""
    byline = item.get("byline") or ""

    links = item.get("links", {})
    url = links.get("web", {}).get("href", "") or item.get("url", "")

    categories = [c.get("description", "") for c in item.get("categories", [])]

    return {
        "headline": headline,
        "description": description,
        "byline": byline,
        "published": published,
        "url": url,
        "story": story,
        "categories": categories,
    }


# ── HTML scraping fallback ────────────────────────────────────────────────────

def fetch_articles_html(url: str = ESPN_FANTASY_URL, timeout: int = 30) -> list[dict]:
    """Scrape the ESPN Fantasy Basketball landing page for article links.

    ESPN is JS-rendered so this only captures statically embedded links.
    Used as a fallback when the API returns nothing useful.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen_urls: set[str] = set()
    articles: list[dict] = []

    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]
        if not any(seg in href for seg in ("/story/", "/blog/", "/fantasy/")):
            continue
        headline = a_tag.get_text(strip=True)
        if not headline or len(headline) < 15:
            continue
        full_url = href if href.startswith("http") else urljoin(ESPN_BASE, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        articles.append({
            "headline": headline,
            "description": "",
            "byline": "",
            "published": "",
            "url": full_url,
            "story": "",
            "categories": [],
        })

    return articles


def fetch_article_body(url: str, timeout: int = 30) -> str:
    """Fetch the body text of an individual ESPN article page."""
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    for selector in [
        "div.article-body",
        "div[class*='article__body']",
        "div[class*='story-body']",
        "div.story",
        "section[class*='article']",
    ]:
        container = soup.select_one(selector)
        if container:
            return _extract_paragraphs(container)

    # Fallback: any long <p> tag
    paras = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 60]
    return "\n\n".join(paras)


def _extract_paragraphs(element) -> str:
    for tag in element(["script", "style", "aside", "figure"]):
        tag.decompose()
    paras = [p.get_text(strip=True) for p in element.find_all("p")]
    return "\n\n".join(p for p in paras if p)


# ── Date filtering ────────────────────────────────────────────────────────────

def _parse_published(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def filter_recent(articles: list[dict], hours: float = 24.0) -> list[dict]:
    """Return articles published within the last `hours` hours.

    Articles with no parseable date are included rather than silently dropped.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = []
    for art in articles:
        pub = _parse_published(art.get("published", ""))
        if pub is None or pub >= cutoff:
            recent.append(art)
    return recent


# ── Markdown rendering ────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(
        textwrap.fill(line, width) if line else ""
        for line in text.splitlines()
    )


def _format_article(art: dict, index: int) -> str:
    parts = [f"## {index}. {art['headline']}"]

    meta: list[str] = []
    pub = _parse_published(art.get("published", ""))
    if pub:
        meta.append(pub.strftime("%Y-%m-%d %H:%M %Z"))
    if art.get("byline"):
        meta.append(f"by {art['byline']}")
    if meta:
        parts.append("*" + " · ".join(meta) + "*")

    if art.get("url"):
        parts.append(f"[Full article]({art['url']})")

    if art.get("description"):
        parts.append(f"\n**Summary:** {art['description']}")

    if art.get("story"):
        parts.append("\n" + _wrap(art["story"]))

    tags = ", ".join(c for c in art.get("categories", []) if c)
    if tags:
        parts.append(f"\n*Tags: {tags}*")

    return "\n\n".join(parts)


def format_digest(articles: list[dict], scraped_at: datetime, hours: float) -> str:
    header = "\n".join([
        f"# ESPN Fantasy Basketball — {hours:.0f}h Digest",
        "",
        f"*Scraped: {scraped_at.strftime('%Y-%m-%d %H:%M %Z')}*",
        f"*Source: {ESPN_FANTASY_URL}*",
        f"*Articles: {len(articles)}*",
        "",
        "---",
        "",
    ])
    sections = [_format_article(art, i + 1) for i, art in enumerate(articles)]
    return header + "\n\n---\n\n".join(sections)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    output_dir: str | Path = "data/raw/espn_fantasy",
    hours: float = 24.0,
    timeout: int = 30,
    fetch_bodies: bool = True,
) -> Path | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scraped_at = datetime.now(timezone.utc).astimezone()

    print("Fetching ESPN Fantasy Basketball articles from API ...")
    try:
        articles = fetch_articles_api(timeout=timeout)
        print(f"  API returned {len(articles)} articles.")
    except Exception as exc:
        print(f"  API fetch failed ({exc}); falling back to HTML scraping ...")
        articles = fetch_articles_html(timeout=timeout)
        print(f"  HTML scrape found {len(articles)} article links.")
        fetch_bodies = True

    recent = filter_recent(articles, hours=hours)
    print(f"  {len(recent)} article(s) within the last {hours:.0f}h.")

    if not recent:
        print("  Nothing to write.")
        return None

    if fetch_bodies:
        for i, art in enumerate(recent, 1):
            if not art.get("story") and art.get("url"):
                print(f"  [{i}/{len(recent)}] Fetching body: {art['url'][:80]} ...")
                art["story"] = fetch_article_body(art["url"], timeout=timeout)

    md = format_digest(recent, scraped_at, hours)
    ts = scraped_at.strftime("%Y-%m-%d_%H-%M-%S")
    dest = output_dir / f"espn_fantasy_{ts}.md"
    dest.write_text(md, encoding="utf-8")
    print(f"  Digest saved → {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape ESPN Fantasy Basketball articles from the last N hours."
    )
    parser.add_argument(
        "--hours", type=float, default=24.0,
        help="How far back to look (default: 24)",
    )
    parser.add_argument(
        "--output-dir", default="data/raw/espn_fantasy",
        help="Output directory (default: data/raw/espn_fantasy)",
    )
    parser.add_argument(
        "--no-fetch-bodies", action="store_true",
        help="Skip fetching full article bodies (headline + description only).",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        hours=args.hours,
        timeout=args.timeout,
        fetch_bodies=not args.no_fetch_bodies,
    )
