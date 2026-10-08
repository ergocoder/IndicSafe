"""English swap units for code-mixing: which source words may go back to English, with POS.

A unit is one whitespace token of the English seed, or several consecutive ones
treated as a whole (a multi-word name: "Lok Sabha", "Mark Twain"). Code-mixing
swaps a unit completely or not at all.

- `SpacyAnalyzer` (default; spaCy `en_core_web_sm`): POS from spaCy, mapped
  from its sub-word tokens to whitespace tokens. Hyphenated words
  ("drug-induced") are one token, POS `COMPOUND`. Consecutive PROPN tokens, or
  PROPN/NOUN tokens inside one named entity, form one unit. Verbs carry their
  lemma ("celebrate"), used for the "English verb + do-verb" construction.
- `ClosedClassAnalyzer` (fallback when spaCy is not installed): no POS. Every
  word outside the closed-class list (stopwords_en) is a candidate with POS
  `X`, so verbs cannot be recognised and are swapped like nouns. Consecutive
  capitalised words (not sentence-initial) form one unit. The tagger name is
  part of the code-mixer's provider version, so which one ran is recorded.

Words in `never_swap_en` (preposition-like verb forms spaCy tags VERB:
regarding, including, ...) and in `stopwords_en` are never units.
"""

from __future__ import annotations

import importlib.metadata
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from generator import code_mix_metrics as cmm

_WS_TOKEN = re.compile(r"\S+")
_POSSESSIVE = re.compile(r"['’]s?$")


@dataclass(frozen=True)
class EnUnit:
    indices: tuple[int, ...]     # whitespace-token positions in the English source
    words: tuple[str, ...]       # surface forms, edge punctuation and possessive 's removed
    pos: str                     # NOUN | PROPN | ADJ | VERB | COMPOUND | X
    lemma: str                   # base form (verbs); otherwise the lowercased text

    @property
    def text(self) -> str:
        return " ".join(self.words)


def _core(token: str) -> str:
    return _POSSESSIVE.sub("", cmm.split_token(token)[1])


def _eligible(word: str, stop: set[str], never: set[str], min_chars: int) -> bool:
    w = word.lower()
    return (len(word) >= min_chars and word.replace("-", "").isalpha() and word.isascii()
            and w not in stop and w not in never)


class EnglishAnalyzer(ABC):
    name: str
    version: str

    @abstractmethod
    def units(self, source: str, *, stopwords: set[str], never_swap: set[str], min_chars: int,
              swap_pos: set[str]) -> list[EnUnit]: ...


def _joins(prev: dict, nxt: dict) -> bool:
    """PROPN PROPN, or PROPN/NOUN tokens inside the same named entity."""
    if nxt["pos"] not in ("PROPN", "NOUN"):
        return False
    return (prev["pos"] == nxt["pos"] == "PROPN") or (prev["ent"] is not None and prev["ent"] == nxt["ent"])


class SpacyAnalyzer(EnglishAnalyzer):
    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy  # noqa: PLC0415 - optional dependency

        self._nlp = spacy.load(model, disable=["parser"])
        self.version = f"{spacy.__version__}+{model}-{self._nlp.meta.get('version', '?')}"
        self._cache: dict[str, list] = {}

    def _tokens(self, source: str) -> list[dict]:
        """Per whitespace token: core word, POS, lemma, entity index."""
        if source not in self._cache:
            doc = self._nlp(source)
            ent_of = {t.i: k for k, e in enumerate(doc.ents) for t in e}
            out = []
            for m in _WS_TOKEN.finditer(source):
                sub = [t for t in doc if m.start() <= t.idx < m.end() and t.is_alpha]
                core = _core(m.group())
                if not sub or not core:
                    out.append({"core": core, "pos": None, "lemma": "", "ent": None})
                    continue
                if "-" in core and len(sub) > 1:
                    pos = "COMPOUND" if any(t.pos_ in ("NOUN", "PROPN", "ADJ", "VERB") for t in sub) else None
                    out.append({"core": core, "pos": pos, "lemma": core.lower(), "ent": None})
                    continue
                main = sub[0]
                out.append({"core": core, "pos": main.pos_, "lemma": main.lemma_.lower(), "ent": ent_of.get(main.i)})
            self._cache[source] = out
        return self._cache[source]

    def units(self, source, *, stopwords, never_swap, min_chars, swap_pos):
        toks = self._tokens(source)
        units: list[EnUnit] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            j = i + 1
            if t["pos"] in ("PROPN", "NOUN"):     # extend a multi-word name
                while j < len(toks) and _joins(toks[j - 1], toks[j]):
                    j += 1
            span = toks[i:j]
            if j - i > 1:
                words = tuple(s["core"] for s in span)
                if all(w.replace("-", "").isalpha() and w.isascii() for w in words) \
                        and not any(w.lower() in never_swap for w in words):
                    pos = "PROPN" if any(s["pos"] == "PROPN" for s in span) else "NOUN"
                    units.append(EnUnit(tuple(range(i, j)), words, pos, " ".join(words).lower()))
            elif t["pos"] in swap_pos and _eligible(t["core"], stopwords, never_swap, min_chars):
                units.append(EnUnit((i,), (t["core"],), t["pos"], t["lemma"] or t["core"].lower()))
            i = j
        return units


class ClosedClassAnalyzer(EnglishAnalyzer):
    name = "closed_class"
    version = "1.0"

    def units(self, source, *, stopwords, never_swap, min_chars, swap_pos):
        cores = [_core(t) for t in cmm.tokens(source)]
        units: list[EnUnit] = []
        i = 0
        while i < len(cores):
            j = i + 1
            if i > 0 and cores[i][:1].isupper():
                while j < len(cores) and cores[j][:1].isupper() and cores[j].isalpha():
                    j += 1
            if j - i > 1:
                words = tuple(cores[i:j])
                units.append(EnUnit(tuple(range(i, j)), words, "PROPN", " ".join(words).lower()))
            elif _eligible(cores[i], stopwords, never_swap, min_chars):
                units.append(EnUnit((i,), (cores[i],), "X", cores[i].lower()))
            i = j
        return units


def build_analyzer(name: str, model: str) -> EnglishAnalyzer:
    """spaCy when requested and importable; otherwise the closed-class fallback."""
    if name == "spacy":
        try:
            importlib.metadata.version("spacy")
            return SpacyAnalyzer(model)
        except (ImportError, OSError, importlib.metadata.PackageNotFoundError):
            return ClosedClassAnalyzer()
    if name == "closed_class":
        return ClosedClassAnalyzer()
    raise ValueError(f"unknown english_tagger {name!r}")
