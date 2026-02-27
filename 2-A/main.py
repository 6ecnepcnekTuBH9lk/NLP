import re
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
from datasets import load_dataset

from spacy.lang.ru.stop_words import STOP_WORDS as RU_STOPWORDS
import pymorphy3


CATEGORIES_TRANSLATOR = {
    "climate": 0,
    "conflicts": 1,
    "culture": 2,
    "economy": 3,
    "gloss": 4,
    "health": 5,
    "politics": 6,
    "science": 7,
    "society": 8,
    "sports": 9,
    "travel": 10,
}

SEED = 42
ALPHA = 0.1
MAX_DOCS_PER_DOMAIN = 8000

# Выбор доменов по имени категории:
DOMAIN_A_NAME = "politics"
DOMAIN_B_NAME = "sports"

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)?")


def normalize(text: str, yo_to_e: bool = True) -> str:
    t = text.lower()
    return t.replace("ё", "е") if yo_to_e else t


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def load_news_dataset():
    ds = load_dataset("data-silence/rus_news_classifier", split="train")
    return ds


def label_id_by_name(name: str) -> int:
    if name not in CATEGORIES_TRANSLATOR:
        raise ValueError(f"Категория '{name}' не найдена. Доступно: {sorted(CATEGORIES_TRANSLATOR.keys())}")
    return CATEGORIES_TRANSLATOR[name]


def select_two_domains(ds, a_name: str, b_name: str, max_docs: int, seed: int):
    a_id = label_id_by_name(a_name)
    b_id = label_id_by_name(b_name)

    A = ds.filter(lambda x: x["labels"] == a_id)
    B = ds.filter(lambda x: x["labels"] == b_id)

    n = min(len(A), len(B), max_docs)
    rnd = random.Random(seed)

    idxA = list(range(len(A)))
    idxB = list(range(len(B)))
    rnd.shuffle(idxA)
    rnd.shuffle(idxB)

    A = A.select(idxA[:n])
    B = B.select(idxB[:n])

    return A, B, a_id, b_id


morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=300_000)
def pymorphy_lemma(word: str) -> str:
    parses = morph.parse(word)
    return parses[0].normal_form if parses else word


def make_units(text: str, *, mode: str,
               remove_stopwords: bool = False,
               add_bigrams: bool = False) -> list[str]:

    toks = tokenize(text)

    if remove_stopwords:
        toks = [t for t in toks if t not in RU_STOPWORDS]

    if mode == "lemma":
        toks = [pymorphy_lemma(t) for t in toks]

    if add_bigrams and len(toks) >= 2:
        bigrams = [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
        toks = toks + bigrams

    return toks


def count_domain_units(docs, *, mode: str, remove_stopwords: bool, add_bigrams: bool) -> tuple[Counter, list[int]]:

    cnt = Counter()
    lengths = []
    for row in docs:
        txt = row["news"]
        base_toks = tokenize(txt)
        if remove_stopwords:
            base_toks = [t for t in base_toks if t not in RU_STOPWORDS]
        if mode == "lemma":
            base_toks = [pymorphy_lemma(t) for t in base_toks]

        lengths.append(len(base_toks))

        units = base_toks
        if add_bigrams and len(units) >= 2:
            units = units + [f"{units[i]}_{units[i+1]}" for i in range(len(units)-1)]

        cnt.update(units)
    return cnt, lengths


def log_odds_delta(cntA: Counter, cntB: Counter, alpha: float = 0.1) -> list[tuple[str, float]]:

    V = set(cntA.keys()) | set(cntB.keys())
    V_size = len(V)
    NA = sum(cntA.values())
    NB = sum(cntB.values())

    denomA = NA + alpha * V_size
    denomB = NB + alpha * V_size

    deltas = []
    for w in V:
        cA = cntA.get(w, 0)
        cB = cntB.get(w, 0)
        pA = (cA + alpha) / denomA
        pB = (cB + alpha) / denomB
        deltas.append((w, math.log(pA) - math.log(pB)))

    deltas.sort(key=lambda x: x[1], reverse=True)
    return deltas


def print_top_logodds(deltas: list[tuple[str, float]], k: int = 30):
    topA = deltas[:k]
    topB = deltas[-k:][::-1]

    print("\nTOP tokens for domain A (highest Δ):")
    for w, d in topA:
        print(f"{w:25s}  Δ={d:.4f}")

    print("\nTOP tokens for domain B (lowest Δ):")
    for w, d in topB:
        print(f"{w:25s}  Δ={d:.4f}")


def plot_length_distributions(lenA: list[int], lenB: list[int], a_name: str, b_name: str):
    plt.figure(figsize=(9, 5))
    plt.hist(lenA, bins=40, alpha=0.6, label=a_name)
    plt.hist(lenB, bins=40, alpha=0.6, label=b_name)
    plt.title("Document length distribution (tokens)")
    plt.xlabel("tokens per document")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top20_bar(cnt: Counter, title: str):
    top = cnt.most_common(20)
    words = [w for w, _ in top]
    freqs = [c for _, c in top]

    plt.figure(figsize=(10, 4.8))
    plt.bar(words, freqs)
    plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.xlabel("token")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


@dataclass
class Variant:
    name: str
    mode: str
    remove_stopwords: bool
    add_bigrams: bool


def run_variant(A, B, a_name: str, b_name: str, variant: Variant):
    print("\n" + "=" * 100)
    print(f"VARIANT: {variant.name}")
    print(f"mode={variant.mode}, stopwords={variant.remove_stopwords}, bigrams={variant.add_bigrams}")

    cntA, lenA = count_domain_units(A, mode=variant.mode,
                                   remove_stopwords=variant.remove_stopwords,
                                   add_bigrams=variant.add_bigrams)
    cntB, lenB = count_domain_units(B, mode=variant.mode,
                                   remove_stopwords=variant.remove_stopwords,
                                   add_bigrams=variant.add_bigrams)

    print(f"A docs={len(A):,} | A tokens(total units)={sum(cntA.values()):,} | A vocab={len(cntA):,} | avg_len={sum(lenA)/len(lenA):.1f}")
    print(f"B docs={len(B):,} | B tokens(total units)={sum(cntB.values()):,} | B vocab={len(cntB):,} | avg_len={sum(lenB)/len(lenB):.1f}")

    if variant.name.lower().startswith("baseline"):
        plot_length_distributions(lenA, lenB, a_name, b_name)
        plot_top20_bar(cntA, f"Top-20 in {a_name} ({variant.name})")
        plot_top20_bar(cntB, f"Top-20 in {b_name} ({variant.name})")

    deltas = log_odds_delta(cntA, cntB, alpha=ALPHA)
    print_top_logodds(deltas, k=30)


def main():
    ds = load_news_dataset()
    print("Available categories:", sorted(CATEGORIES_TRANSLATOR.keys()))

    A, B, a_id, b_id = select_two_domains(
        ds,
        DOMAIN_A_NAME, DOMAIN_B_NAME,
        max_docs=MAX_DOCS_PER_DOMAIN,
        seed=SEED
    )
    print(f"\nChosen domains: A='{DOMAIN_A_NAME}'(id={a_id}) vs B='{DOMAIN_B_NAME}'(id={b_id})")
    print(f"Balanced docs per domain: {len(A):,}")

    variants = [
        Variant("BASELINE_tokens", mode="token", remove_stopwords=False, add_bigrams=False),
        Variant("Tokens_no_stopwords", mode="token", remove_stopwords=True, add_bigrams=False),
        Variant("Lemmas_pymorphy3", mode="lemma", remove_stopwords=False, add_bigrams=False),
        Variant("Lemmas_no_stopwords", mode="lemma", remove_stopwords=True, add_bigrams=False),
        Variant("Lemmas_no_stopwords + bigrams", mode="lemma", remove_stopwords=True, add_bigrams=True),
    ]

    for v in variants:
        run_variant(A, B, DOMAIN_A_NAME, DOMAIN_B_NAME, v)


if __name__ == "__main__":
    main()