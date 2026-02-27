from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import Matcher

import pymorphy3
from nltk.stem.snowball import SnowballStemmer


def load_text_auto_encoding(path: str | Path) -> str:
    path = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # если совсем не вышло — пусть упадёт с ошибкой, чтобы было понятно
    return path.read_text(encoding="utf-8")


def strip_gutenberg_if_present(text: str) -> str:

    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    start_idx = None
    for m in start_markers:
        i = text.find(m)
        if i != -1:
            start_idx = i
            break

    end_idx = None
    for m in end_markers:
        i = text.find(m)
        if i != -1:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return text

    start_line_end = text.find("\n", start_idx)
    if start_line_end == -1:
        start_line_end = start_idx
    return text[start_line_end:end_idx].strip()


def strip_header_until_chapter_1(text: str) -> str:

    m = re.search(r"\n\s*I\s*\n", text)
    return text[m.start():].strip() if m else text.strip()


_RU_WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?")


def normalize_text_ru(text: str, *, lower: bool = True, yo_to_e: bool = False) -> str:
    if lower:
        text = text.lower()
    if yo_to_e:
        text = text.replace("ё", "е")
    return text


def tokenize_ru(text: str) -> list[str]:
    return _RU_WORD_RE.findall(text)


@dataclass
class TailAndTop:
    n_tokens: int
    n_types: int
    tail_types_share: float
    top20: list[tuple[str, int]]


def compute_tail_types_share_and_top20(units: list[str], *, tail_max_freq: int = 3) -> TailAndTop:
    cnt = Counter(units)
    n_tokens = sum(cnt.values())
    n_types = len(cnt)
    tail_types = sum(1 for _, c in cnt.items() if c <= tail_max_freq)
    tail_types_share = (tail_types / n_types) if n_types else 0.0
    return TailAndTop(
        n_tokens=n_tokens,
        n_types=n_types,
        tail_types_share=tail_types_share,
        top20=cnt.most_common(20),
    )


def build_stemmer() -> SnowballStemmer:
    return SnowballStemmer("russian")


def build_pymorphy() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


def attach_pymorphy_cache(morph: pymorphy3.MorphAnalyzer):
    @lru_cache(maxsize=200_000)
    def normal_form(word: str) -> str:
        parses = morph.parse(word)
        if not parses:
            return word
        return parses[0].normal_form

    return normal_form


def make_units(text: str, *, mode: str, yo_to_e: bool = False,
               stemmer: SnowballStemmer | None = None,
               normal_form=None) -> list[str]:

    norm = normalize_text_ru(text, lower=True, yo_to_e=yo_to_e)
    toks = tokenize_ru(norm)

    if mode == "token":
        return toks

    if mode == "stem":
        stemmer = stemmer or build_stemmer()
        return [stemmer.stem(t) for t in toks]

    if mode == "lemma":
        if normal_form is None:
            morph = build_pymorphy()
            normal_form = attach_pymorphy_cache(morph)
        return [normal_form(t) for t in toks]

    raise ValueError(f"Unknown mode={mode}")


def run_freq_comparison(fragment: str):
    stemmer = build_stemmer()
    morph = build_pymorphy()
    normal_form = attach_pymorphy_cache(morph)

    variants = [
        ("TOKENS", dict(mode="token", yo_to_e=False)),
        ("STEMS", dict(mode="stem", yo_to_e=False, stemmer=stemmer)),
        ("LEMMAS(pymorphy3)", dict(mode="lemma", yo_to_e=False, normal_form=normal_form)),
    ]

    print("\n" + "=" * 90)
    print("Частоты: сравнение TOKENS vs STEMS vs LEMMAS")
    print("Long tail: доля types с частотой ≤3")
    for name, kwargs in variants:
        units = make_units(fragment, **kwargs)
        res = compute_tail_types_share_and_top20(units, tail_max_freq=3)
        print("\n" + "-" * 90)
        print(name)
        print(f"N tokens = {res.n_tokens:,} | V types = {res.n_types:,} | tail(types<=3) = {res.tail_types_share:.2%}")
        print("TOP-20:")
        print(res.top20)


def pick_5_sentences(doc, *, min_tokens: int = 6) -> list[str]:
    out = []
    for sent in doc.sents:

        toks = [t for t in sent if not t.is_space]
        if len(toks) >= min_tokens:
            out.append(sent.text.strip())
        if len(out) == 5:
            break
    return out


def print_sentence_comparison(nlp, morph, sentences: list[str]):
    normal_form = attach_pymorphy_cache(morph)

    def fmt_row(cols, widths):
        return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    headers = ["TOK", "POS", "DEP", "HEAD", "spaCy_lemma", "pymorphy_lemma"]
    widths = [18, 6, 10, 18, 16, 16]

    print("\n" + "=" * 90)
    print("5 предложений: сравнение spaCy POS/DEP/NER и лемм pymorphy3")
    for i, s in enumerate(sentences, 1):
        doc = nlp(s)
        ents = [(e.text, e.label_) for e in doc.ents]

        print("\n" + "-" * 90)
        print(f"[{i}] {s}")
        if ents:
            print("NER:", ents)
        else:
            print("NER: (нет сущностей)")

        print(fmt_row(headers, widths))
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))

        for t in doc:
            if t.is_space:
                continue
            tok = t.text
            sp_lemma = t.lemma_
            pm_lemma = normal_form(tok.lower()) if tok.isalpha() else ""
            row = [
                tok[:18],
                t.pos_,
                t.dep_,
                t.head.text[:18],
                sp_lemma[:16],
                pm_lemma[:16],
            ]
            print(fmt_row(row, widths))


def extract_character_names(doc) -> Counter:

    names = Counter()

    for ent in doc.ents:
        if ent.label_ in {"PER", "PERSON"}:
            names[ent.text.strip()] += 1

    i = 0
    while i < len(doc):
        t = doc[i]
        if t.pos_ == "PROPN" and t.text[:1].isupper():
            j = i + 1
            while j < len(doc) and doc[j].pos_ == "PROPN" and doc[j].text[:1].isupper():
                j += 1
            span = doc[i:j].text.strip()

            if 2 <= len(span) <= 40:
                names[span] += 1
            i = j
        else:
            i += 1

    return names


GREET_PATTERNS = [

    r"\bпривет\b",
    r"\bздравств(уй|уйте)\b",
    r"\bдобрый\s+(день|вечер)\b",
    r"\bдоброе\s+утро\b",
    r"\bмо(ё|е)\s+почтение\b",

    r"\bрад(а|ы)?\b.*\b(видеть|увидеть|увидеться)\b",
    r"\bочень\s+рад(а|ы)?\b",
    r"\b(рад|рада|рады)\s+встрече\b",

    r"\bвойдите\b",
    r"\bпроходите\b",
    r"\b(милости\s+прошу|прошу)\b",
    r"\bпрошу\s+(в\s+дом|сюда|войти)\b",
]
GREET_RE = re.compile("|".join(GREET_PATTERNS), flags=re.IGNORECASE)


def find_greeting_sentences(doc, *, limit: int | None = None) -> list[str]:
    res = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue
        if GREET_RE.search(s):
            res.append(s)
            if limit and len(res) >= limit:
                break
    return res


def main():
    base = Path(__file__).resolve().parent  # .../NLP/1-2
    project_root = base.parent  # .../NLP
    text_path = project_root / "data" / "NLP.txt"

    raw = load_text_auto_encoding(text_path)
    raw = strip_gutenberg_if_present(raw)
    raw = strip_header_until_chapter_1(raw)

    print(f"Текст: {len(raw):,} символов")

    run_freq_comparison(raw)

    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(raw)

    morph = build_pymorphy()
    sents5 = pick_5_sentences(doc, min_tokens=6)
    print_sentence_comparison(nlp, morph, sents5)

    names = extract_character_names(doc)
    print("\n" + "=" * 90)
    print("Имена персонажей (топ-30 по частоте, NER+PROPN):")
    for name, c in names.most_common(30):
        print(f"{name} — {c}")

    greets = find_greeting_sentences(doc)
    print("\n" + "=" * 90)
    print(f"Предложения, где здороваются: {len(greets)}")
    for s in greets[:50]:
        print("-", s)

    if len(greets) > 50:
        print(f"... (ещё {len(greets) - 50} предложений не показано)")


if __name__ == "__main__":
    main()
