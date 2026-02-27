# =========================
# Мини-задача 1 (русский роман):
# Zipf + длинный хвост + сравнение предобработок + 2 интента (rule-based)
# =========================

import re
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt

# ---------- 1) Загрузка текста ----------

def load_text_from_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def strip_gutenberg_if_present(text: str) -> str:

    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    lo = text
    start_idx = None
    for m in start_markers:
        i = lo.find(m)
        if i != -1:
            start_idx = i
            break

    end_idx = None
    for m in end_markers:
        i = lo.find(m)
        if i != -1:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return text

    start_line_end = text.find("\n", start_idx)
    if start_line_end == -1:
        start_line_end = start_idx
    return text[start_line_end:end_idx].strip()


# ---------- 2) Нормализация и токенизация ----------

_RU_WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?")

def normalize_text_ru(text: str, *, lower: bool = True, yo_to_e: bool = False) -> str:
    if lower:
        text = text.lower()
    if yo_to_e:
        text = text.replace("ё", "е")
    return text

def tokenize_ru(text: str) -> list[str]:

    return _RU_WORD_RE.findall(text)


# ---------- 3) Стемминг и лемматизация ----------

def build_russian_stemmer():
    try:
        from nltk.stem.snowball import SnowballStemmer
    except Exception as e:
        raise RuntimeError("Не найден nltk/SnowballStemmer. Установи nltk.") from e
    return SnowballStemmer("russian")

def build_pymorphy():
    try:
        import pymorphy3
    except Exception as e:
        raise RuntimeError("Не найден pymorphy3. Установи pymorphy3 и словари.") from e
    return pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=200_000)
def _pymorphy_normal_form_cached(word: str) -> str:

    return word

def attach_pymorphy_cache(morph):

    @lru_cache(maxsize=200_000)
    def normal_form(word: str) -> str:
        p = morph.parse(word)
        if not p:
            return word
        return p[0].normal_form
    return normal_form


# ---------- 4) Частоты, Zipf, длинный хвост ----------

@dataclass
class FreqStats:
    n_tokens: int
    n_types: int
    top20: list[tuple[str, int]]
    tail_types: int
    tail_tokens: int
    tail_types_share: float
    tail_tokens_share: float


def compute_freq_stats(units: list[str], *, tail_max_freq: int = 3) -> FreqStats:
    cnt = Counter(units)
    n_tokens = sum(cnt.values())
    n_types = len(cnt)

    tail_words = [w for w, c in cnt.items() if c <= tail_max_freq]
    tail_types = len(tail_words)
    tail_tokens = sum(cnt[w] for w in tail_words)

    tail_types_share = (tail_types / n_types) if n_types else 0.0
    tail_tokens_share = (tail_tokens / n_tokens) if n_tokens else 0.0

    return FreqStats(
        n_tokens=n_tokens,
        n_types=n_types,
        top20=cnt.most_common(20),
        tail_types=tail_types,
        tail_tokens=tail_tokens,
        tail_types_share=tail_types_share,
        tail_tokens_share=tail_tokens_share,
    )

def plot_zipf(units: list[str], title: str = "Zipf (rank–frequency)"):

    cnt = Counter(units)
    freqs = sorted(cnt.values(), reverse=True)
    ranks = range(1, len(freqs) + 1)

    plt.figure(figsize=(8, 5))
    plt.loglog(list(ranks), freqs)
    plt.xlabel("rank (log)")
    plt.ylabel("frequency (log)")
    plt.title(title)
    plt.show()

def plot_top20(units: list[str], title: str = "Top-20"):
    cnt = Counter(units)
    top = cnt.most_common(20)
    words = [w for w, _ in top]
    freqs = [c for _, c in top]

    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs)
    plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.xlabel("unit")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


# ---------- 5) Сбор “юнитов” для разных предобработок ----------

def make_units(
    text: str,
    *,
    mode: str = "token",
    lower: bool = True,
    yo_to_e: bool = False,
    stemmer=None,
    pymorphy_normal_form=None,
) -> list[str]:

    norm = normalize_text_ru(text, lower=lower, yo_to_e=yo_to_e)
    toks = tokenize_ru(norm)

    if mode == "token":
        return toks

    if mode == "stem":
        if stemmer is None:
            stemmer = build_russian_stemmer()
        return [stemmer.stem(t) for t in toks]

    if mode == "lemma_pymorphy":
        if pymorphy_normal_form is None:
            morph = build_pymorphy()
            pymorphy_normal_form = attach_pymorphy_cache(morph)
        return [pymorphy_normal_form(t) for t in toks]

    raise ValueError(f"Unknown mode: {mode}")


# ---------- 6) Rule-based интенты (2 новых) на spaCy Matcher ----------

def build_ru_intent_matcher():

    try:
        import spacy
        from spacy.matcher import Matcher
    except Exception as e:
        raise RuntimeError("Не найден spaCy. Установи spacy и модель ru_core_news_sm.") from e

    nlp = spacy.load("ru_core_news_sm")
    matcher = Matcher(nlp.vocab)

    # greeting
    matcher.add("greeting", [
        [{"LOWER": {"IN": ["привет", "здравствуйте", "здравствуй"]}}],
        [{"LOWER": "добрый"}, {"LOWER": {"IN": ["день", "вечер", "утро"]}}],
    ])

    # goodbye
    matcher.add("goodbye", [
        [{"LOWER": {"IN": ["пока", "до", "прощай", "свидания"]}}],
        [{"LOWER": "до"}, {"LOWER": "свидания"}],
    ])

    # NEW: ask_name
    matcher.add("ask_name", [
        [{"LOWER": "как"}, {"LOWER": {"IN": ["тебя", "вас"]}}, {"LOWER": "зовут"}],
        [{"LOWER": "кто"}, {"LOWER": "ты"}],
        [{"LOWER": "как"}, {"LOWER": "тебя"}, {"LOWER": "звать"}],
    ])

    # NEW: thanks
    matcher.add("thanks", [
        [{"LOWER": {"IN": ["спасибо", "благодарю"]}}],
        [{"LOWER": "спасиб"}],
    ])

    def detect_intent(text: str) -> str:
        doc = nlp(text)
        matches = matcher(doc)

        if not matches:
            return "other"

        best = None
        best_len = -1
        for match_id, start, end in matches:
            ln = end - start
            if ln > best_len:
                best_len = ln
                best = match_id

        return nlp.vocab.strings[best]

    return nlp, matcher, detect_intent


def run_zipf_experiments(raw_text: str):
    stemmer = build_russian_stemmer()
    morph = build_pymorphy()
    normal_form = attach_pymorphy_cache(morph)

    experiments = [
        ("TOKENS (lower)", dict(mode="token", lower=True, yo_to_e=False)),
        ("TOKENS (lower + ё→е)", dict(mode="token", lower=True, yo_to_e=True)),
        ("STEMS (lower)", dict(mode="stem", lower=True, yo_to_e=False, stemmer=stemmer)),
        ("LEMMAS pymorphy3 (lower)", dict(mode="lemma_pymorphy", lower=True, yo_to_e=False, pymorphy_normal_form=normal_form)),
        ("LEMMAS pymorphy3 (lower + ё→е)", dict(mode="lemma_pymorphy", lower=True, yo_to_e=True, pymorphy_normal_form=normal_form)),
    ]

    for title, kwargs in experiments:
        units = make_units(raw_text, **kwargs)
        st = compute_freq_stats(units, tail_max_freq=3)

        print("\n" + "=" * 80)
        print(title)
        print(f"tokens (N) = {st.n_tokens:,}")
        print(f"types  (V) = {st.n_types:,}")
        print(f"tail types (<=3) = {st.tail_types:,}  ({st.tail_types_share:.2%} of V)")
        print(f"tail tokens(<=3) = {st.tail_tokens:,} ({st.tail_tokens_share:.2%} of N)")
        print("top-10:", st.top20[:10])

        plot_zipf(units, title=f"Zipf: {title}")
        plot_top20(units, title=f"Top-20: {title}")


def run_intent_tests():
    nlp, matcher, detect_intent = build_ru_intent_matcher()
    samples = [
        "Привет!",
        "Добрый вечер, скажи пожалуйста...",
        "Как тебя зовут?",
        "Кто ты?",
        "Спасибо большое!",
        "Благодарю!",
        "До свидания.",
        "Мне нужна помощь с заданием.",
    ]
    for s in samples:
        print(f"{s!r} -> {detect_intent(s)}")


if __name__ == "__main__":
    TEXT_PATH = "data/NLP.txt"
    raw = load_text_from_file(TEXT_PATH, encoding="utf-8")
    raw = strip_gutenberg_if_present(raw)

    run_zipf_experiments(raw)
    run_intent_tests()
