from __future__ import annotations

import re
import math
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from functools import lru_cache
from typing import Iterable

import pymorphy3
from spacy.lang.ru.stop_words import STOP_WORDS as RU_STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
CHUNK_SIZE = 200
STRIDE = 200
MIN_TOKENS_PER_DOC = 80
MAX_TOKENS_PER_DOC = 300

# Жанры для domain shift
TRAIN_GENRE = "prose"
SHIFT_GENRE = "poetry"

# Папка с авторами
DATA_ROOT = Path(__file__).resolve().parent / "data_authors"

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яa-яЁё]+)?")

morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=400_000)
def lemma_cached(w: str) -> str:
    parses = morph.parse(w)
    return parses[0].normal_form if parses else w


def normalize(text: str) -> str:

    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def read_all_txt(folder: Path) -> str:
    parts = []
    for p in sorted(folder.rglob("*.txt")):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            parts.append(p.read_text(encoding="cp1251"))
    return "\n".join(parts)


def load_author_genre(author_dir: Path, genre: str) -> str:
    folder = author_dir / genre
    if not folder.exists():
        raise FileNotFoundError(f"Не найдена папка жанра: {folder}")
    text = read_all_txt(folder)
    if not text.strip():
        raise ValueError(f"Папка пуста: {folder}")
    return text


def preprocess_tokens(tokens: list[str], *, use_lemmas: bool, remove_stopwords: bool) -> list[str]:
    out = tokens
    if remove_stopwords:
        out = [t for t in out if t not in RU_STOPWORDS]
    if use_lemmas:

        def safe_lemma(t: str) -> str:
            if len(t) <= 3:
                return t
            if any(ch.isdigit() for ch in t):
                return t
            return lemma_cached(t)
        out = [safe_lemma(t) for t in out]
    return out


def chunk_tokens(tokens: list[str], chunk_size: int, stride: int,
                 min_tokens: int = 80, max_tokens: int = 300) -> list[list[str]]:
    chunks = []
    n = len(tokens)
    i = 0
    while i < n:
        chunk = tokens[i:i + chunk_size]
        if len(chunk) >= min_tokens:
            chunk = chunk[:max_tokens]
            chunks.append(chunk)
        i += stride
    return chunks


@dataclass
class DatasetXY:
    X: list[str]
    y: list[int]
    label_names: list[str]


def build_xy_for_genre(author1_dir: Path, author2_dir: Path, genre: str,
                       *, use_lemmas: bool, remove_stopwords: bool,
                       chunk_size: int, stride: int) -> DatasetXY:
    t1 = load_author_genre(author1_dir, genre)
    t2 = load_author_genre(author2_dir, genre)

    toks1 = preprocess_tokens(tokenize(t1), use_lemmas=use_lemmas, remove_stopwords=remove_stopwords)
    toks2 = preprocess_tokens(tokenize(t2), use_lemmas=use_lemmas, remove_stopwords=remove_stopwords)

    chunks1 = chunk_tokens(toks1, chunk_size=chunk_size, stride=stride,
                           min_tokens=MIN_TOKENS_PER_DOC, max_tokens=MAX_TOKENS_PER_DOC)
    chunks2 = chunk_tokens(toks2, chunk_size=chunk_size, stride=stride,
                           min_tokens=MIN_TOKENS_PER_DOC, max_tokens=MAX_TOKENS_PER_DOC)

    docs1 = [" ".join(ch) for ch in chunks1]
    docs2 = [" ".join(ch) for ch in chunks2]

    n = min(len(docs1), len(docs2))
    rnd = random.Random(SEED)
    docs1 = rnd.sample(docs1, n)
    docs2 = rnd.sample(docs2, n)

    X = docs1 + docs2
    y = [0] * len(docs1) + [1] * len(docs2)
    label_names = [author1_dir.name, author2_dir.name]
    return DatasetXY(X=X, y=y, label_names=label_names)


def fit_and_report(X_train, y_train, X_test, y_test, label_names: list[str],
                   *, ngram_range=(1, 1), alpha: float = 0.1, use_complement_nb: bool = False):

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        ngram_range=ngram_range,
        min_df=2,
        token_pattern=None,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = ComplementNB(alpha=alpha) if use_complement_nb else MultinomialNB(alpha=alpha)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=label_names,
        digits=4,
        zero_division=0
    ))

    feature_names = vectorizer.get_feature_names_out()
    logp = clf.feature_log_prob_
    delta = logp[0] - logp[1]

    top0_idx = delta.argsort()[::-1][:20]
    top1_idx = delta.argsort()[:20]

    print("\nTop-20 tokens for", label_names[0], "(highest Δ):")
    for i in top0_idx:
        print(f"{feature_names[i]:30s}  Δ={float(delta[i]):.4f}")

    print("\nTop-20 tokens for", label_names[1], "(lowest Δ):")
    for i in top1_idx:
        print(f"{feature_names[i]:30s}  Δ={float(delta[i]):.4f}")

    return vectorizer, clf


@dataclass
class Variant:
    name: str
    use_lemmas: bool
    remove_stopwords: bool
    ngram_range: tuple[int, int]
    use_complement_nb: bool = False


def run_variants(author1_dir: Path, author2_dir: Path):
    random.seed(SEED)

    variants = [
        Variant("Baseline (tokens, unigrams, MultinomialNB)", use_lemmas=False, remove_stopwords=False, ngram_range=(1, 1)),
        Variant("Stopwords removed (tokens, unigrams, MultinomialNB)", use_lemmas=False, remove_stopwords=True, ngram_range=(1, 1)),
        Variant("Lemmas (pymorphy3, unigrams, MultinomialNB)", use_lemmas=True, remove_stopwords=False, ngram_range=(1, 1)),
        Variant("Lemmas + stopwords removed (unigrams, MultinomialNB)", use_lemmas=True, remove_stopwords=True, ngram_range=(1, 1)),
        Variant("Lemmas + stopwords + bigrams (1–2), MultinomialNB", use_lemmas=True, remove_stopwords=True, ngram_range=(1, 2)),
        Variant("Lemmas + stopwords + bigrams (1–2), ComplementNB", use_lemmas=True, remove_stopwords=True, ngram_range=(1, 2), use_complement_nb=True),
    ]

    for v in variants:
        print("\n" + "=" * 110)
        print("VARIANT:", v.name)
        print(f"use_lemmas={v.use_lemmas}, stopwords={v.remove_stopwords}, ngrams={v.ngram_range}, complementNB={v.use_complement_nb}")

        ds_train = build_xy_for_genre(
            author1_dir, author2_dir, TRAIN_GENRE,
            use_lemmas=v.use_lemmas,
            remove_stopwords=v.remove_stopwords,
            chunk_size=CHUNK_SIZE,
            stride=STRIDE,
        )
        Xtr, Xte, ytr, yte = train_test_split(ds_train.X, ds_train.y, test_size=0.25, random_state=SEED, stratify=ds_train.y)

        print(f"\nIN-DOMAIN: train/test on '{TRAIN_GENRE}' | docs={len(ds_train.X)}")
        vectorizer, clf = fit_and_report(
            Xtr, ytr, Xte, yte, ds_train.label_names,
            ngram_range=v.ngram_range,
            alpha=0.1,
            use_complement_nb=v.use_complement_nb
        )

        ds_shift = build_xy_for_genre(
            author1_dir, author2_dir, SHIFT_GENRE,
            use_lemmas=v.use_lemmas,
            remove_stopwords=v.remove_stopwords,
            chunk_size=CHUNK_SIZE,
            stride=STRIDE,
        )

        print(f"\nDOMAIN SHIFT: train on '{TRAIN_GENRE}' -> test on '{SHIFT_GENRE}' | test_docs={len(ds_shift.X)}")

        Xtr_full = vectorizer.fit_transform(ds_train.X)
        Xsh = vectorizer.transform(ds_shift.X)
        clf.fit(Xtr_full, ds_train.y)
        y_pred_shift = clf.predict(Xsh)

        print("\nConfusion matrix (shift):")
        print(confusion_matrix(ds_shift.y, y_pred_shift, labels=[0, 1]))

        print("\nClassification report (shift):")
        print(classification_report(
            ds_shift.y, y_pred_shift,
            labels=[0, 1],
            target_names=ds_train.label_names,
            digits=4,
            zero_division=0
        ))


def main():
    author1_dir = DATA_ROOT / "pushkin"
    author2_dir = DATA_ROOT / "lermontov"

    if not author1_dir.exists() or not author2_dir.exists():
        raise FileNotFoundError(
            f"Ожидаю папки:\n  {author1_dir}\n  {author2_dir}\n"
            f"Сделай структуру data_authors/author1/... и data_authors/author2/..."
        )

    print("Author1:", author1_dir.name)
    print("Author2:", author2_dir.name)
    print("Train genre:", TRAIN_GENRE, "| Shift genre:", SHIFT_GENRE)
    print("Chunk size:", CHUNK_SIZE, "| Stride:", STRIDE)

    run_variants(author1_dir, author2_dir)


if __name__ == "__main__":
    main()
