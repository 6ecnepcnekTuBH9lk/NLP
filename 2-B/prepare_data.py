from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_authors"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NLP-course-bot/1.0"
}

ITEMS = [

    {
        "url": "https://ru.wikisource.org/wiki/%D0%9F%D0%B8%D0%BA%D0%BE%D0%B2%D0%B0%D1%8F_%D0%B4%D0%B0%D0%BC%D0%B0_%28%D0%9F%D1%83%D1%88%D0%BA%D0%B8%D0%BD%29",
        "path": OUT / "pushkin" / "prose" / "pikovaya_dama.txt",
    },
    {
        "url": "https://ru.wikisource.org/wiki/%D0%A0%D1%83%D1%81%D0%BB%D0%B0%D0%BD_%D0%B8_%D0%9B%D1%8E%D0%B4%D0%BC%D0%B8%D0%BB%D0%B0_%28%D0%9F%D1%83%D1%88%D0%BA%D0%B8%D0%BD%29",
        "path": OUT / "pushkin" / "poetry" / "ruslan_i_lyudmila.txt",
    },

    {
        "url": "https://ru.wikisource.org/wiki/%D0%93%D0%B5%D1%80%D0%BE%D0%B9_%D0%BD%D0%B0%D1%88%D0%B5%D0%B3%D0%BE_%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%B8_%28%D0%9B%D0%B5%D1%80%D0%BC%D0%BE%D0%BD%D1%82%D0%BE%D0%B2%29/1969_%28%D0%A1%D0%9E%29",
        "path": OUT / "lermontov" / "prose" / "geroy_nashego_vremeni.txt",
    },
    {
        "url": "https://ru.wikisource.org/wiki/%D0%9C%D1%86%D1%8B%D1%80%D0%B8_%28%D0%9B%D0%B5%D1%80%D0%BC%D0%BE%D0%BD%D1%82%D0%BE%D0%B2%29",
        "path": OUT / "lermontov" / "poetry" / "mtsyri.txt",
    },
]


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        content = soup.select_one("#mw-content-text")
    if content is None:
        raise RuntimeError("Не нашёл блок с текстом (#mw-content-text).")

    for sel in [
        "div#toc",
        "div.mw-editsection",
        "span.mw-editsection",
        "sup.reference",
        "ol.references",
        "div.reflist",
        "table",
        "div.navbox",
        "div.printfooter",
    ]:
        for tag in content.select(sel):
            tag.decompose()

    text = content.get_text("\n")

    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)

    stop_markers = [
        "Источник —",
        "Категории",
        "Скрытые категории",
        "Политика конфиденциальности",
        "Описание Викитеки",
    ]
    cleaned = []
    for ln in lines:
        if any(ln.startswith(m) for m in stop_markers):
            break
        cleaned.append(ln)

    return "\n".join(cleaned).strip() + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(ITEMS, start=1):
        url = item["url"]
        path = item["path"]

        print(f"[{i}/{len(ITEMS)}] Download: {url}")
        html = fetch_html(url)
        text = html_to_clean_text(html)
        save_text(path, text)

        print(f"   saved -> {path} (chars={len(text):,})")
        time.sleep(1.0)  # чуть-чуть “вежливо” к сайту

    print("\nDONE. Dataset folder:", OUT)


if __name__ == "__main__":
    main()
