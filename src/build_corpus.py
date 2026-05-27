from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
QUERIES_PATH = Path("data/queries.jsonl")
OUTPUT_PATH = Path("data/docs.jsonl")
MAX_DOCS = 500
MIN_DOCS = 200


@dataclass
class SectionDoc:
    doc_id: str
    source_url: str
    page_title: str
    section_title: str
    text: str


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_doc_urls() -> list[str]:
    resp = requests.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    urls = [loc.text.strip() for loc in soup.find_all("loc")]
    filtered = []
    for url in urls:
        if "fastapi.tiangolo.com" not in url:
            continue
        if "/img/" in url or url.endswith(('.jpg', '.png', '.svg', '.ico')):
            continue
        filtered.append(url)
    return filtered


def extract_sections(url: str) -> Iterable[SectionDoc]:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("article") or soup.find("main")
    if not article:
        return []

    page_title = clean_text((soup.find("h1") or soup.find("title")).get_text(" "))
    headings = article.find_all(["h2", "h3"])

    if not headings:
        txt = clean_text(article.get_text(" "))
        if len(txt) < 120:
            return []
        yield SectionDoc(
            doc_id=f"{url}#whole",
            source_url=url,
            page_title=page_title,
            section_title="Whole page",
            text=txt[:5000],
        )
        return

    for idx, head in enumerate(headings):
        section_title = clean_text(head.get_text(" "))
        section_chunks: list[str] = []
        node = head.next_sibling
        while node is not None:
            if getattr(node, "name", None) in {"h2", "h3"}:
                break
            if hasattr(node, "get_text"):
                section_chunks.append(node.get_text(" "))
            node = node.next_sibling
        section_text = clean_text(" ".join(section_chunks))
        if len(section_text) < 120:
            continue
        yield SectionDoc(
            doc_id=f"{url}#sec-{idx}",
            source_url=url,
            page_title=page_title,
            section_title=section_title,
            text=section_text[:5000],
        )


def get_required_substrings() -> list[str]:
    substrings: list[str] = []
    for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        subs = q.get("relevant_url_substrings") or [q.get("relevant_url_substring", "")]
        substrings.extend(s for s in subs if s)
    return list(dict.fromkeys(substrings))


def find_priority_urls(all_urls: list[str], required_substrings: list[str]) -> list[str]:
    priority: list[str] = []
    seen: set[str] = set()
    for sub in required_substrings:
        best: str | None = None
        # Prefer the URL whose path ends exactly with the required substring
        for url in all_urls:
            if url.rstrip("/").endswith(sub.rstrip("/")):
                best = url
                break
        # Fall back to first URL that contains the substring
        if best is None:
            for url in all_urls:
                if sub in url:
                    best = url
                    break
        if best and best not in seen:
            priority.append(best)
            seen.add(best)
    return priority


def validate_coverage(docs: list[SectionDoc], required_substrings: list[str]) -> None:
    doc_urls = {d.source_url for d in docs}
    missing = [
        sub for sub in required_substrings
        if not any(sub in url for url in doc_urls)
    ]
    if missing:
        msg = "Required URL substrings not covered by corpus after build:\n" + "\n".join(f"  {s}" for s in missing)
        raise RuntimeError(msg)


def build_corpus() -> list[SectionDoc]:
    all_urls = get_doc_urls()
    required_substrings = get_required_substrings()
    priority_urls = find_priority_urls(all_urls, required_substrings)
    remaining_urls = [u for u in all_urls if u not in set(priority_urls)]

    docs: list[SectionDoc] = []

    # Fetch required pages first to guarantee every query label is satisfiable
    for url in priority_urls:
        for sec in extract_sections(url):
            docs.append(sec)
        time.sleep(0.05)

    # Fill from sitemap order until corpus cap
    for url in remaining_urls:
        if len(docs) >= MAX_DOCS:
            break
        for sec in extract_sections(url):
            docs.append(sec)
            if len(docs) >= MAX_DOCS:
                break
        time.sleep(0.05)

    return docs


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = build_corpus()

    required_substrings = get_required_substrings()
    validate_coverage(docs, required_substrings)

    if len(docs) < MIN_DOCS:
        raise RuntimeError(f"Only collected {len(docs)} docs; need at least {MIN_DOCS}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} section docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
