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
OUTPUT_PATH = Path("data/docs.jsonl")
MAX_DOCS = 450
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


def build_corpus() -> list[SectionDoc]:
    docs: list[SectionDoc] = []
    for url in get_doc_urls():
        for sec in extract_sections(url):
            docs.append(sec)
            if len(docs) >= MAX_DOCS:
                return docs
        time.sleep(0.05)
    return docs


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = build_corpus()
    if len(docs) < MIN_DOCS:
        raise RuntimeError(f"Only collected {len(docs)} docs; need at least {MIN_DOCS}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} section docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
