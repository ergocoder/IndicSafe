"""Text normalisation and script detection shared by the seed manager.

Script detection is configuration-driven (Unicode ranges in languages.yaml).
It answers "which writing system are the letters in", not "which language is
this": Hindi and Marathi both use Devanagari, so language checks need a
separate lexical detector (Phase 2).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Canonical stored form: NFC, whitespace collapsed, trimmed.

    Keeps case and punctuation — only representation noise is removed.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection.

    NFKC + casefold, punctuation/symbols dropped, whitespace collapsed, so
    "How can I X?" and "how can i  x" collide but different words never do.
    """
    t = unicodedata.normalize("NFKC", text).casefold()
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in t)
    return _WS.sub(" ", t).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(dedup_key(text).encode("utf-8")).hexdigest()


def has_control_chars(text: str) -> bool:
    """True for C0/C1 control chars other than tab/newline (ZWJ/ZWNJ are allowed:
    they are legitimate in Indic scripts)."""
    return any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in text)


def _is_letterlike(ch: str) -> bool:
    # Letters plus combining marks (Devanagari vowel signs are Mn/Mc).
    return unicodedata.category(ch)[0] in "LM"


def script_profile(text: str, scripts: Mapping[str, Sequence[tuple[int, int]]]) -> Counter:
    """Count letter-like characters per configured script; 'Other' for the rest."""
    counts: Counter = Counter()
    for ch in text:
        if not _is_letterlike(ch):
            continue
        cp = ord(ch)
        for name, ranges in scripts.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break
        else:
            counts["Other"] += 1
    return counts


def dominant_script(
    text: str, scripts: Mapping[str, Sequence[tuple[int, int]]]
) -> tuple[str | None, float]:
    """(script, share of letters in that script). (None, 0.0) if no letters."""
    profile = script_profile(text, scripts)
    total = sum(profile.values())
    if total == 0:
        return None, 0.0
    name, n = max(profile.items(), key=lambda kv: (kv[1], kv[0]))
    return name, round(n / total, 4)


def word_set(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(dedup_key(text)))


def word_jaccard(a: str | frozenset[str], b: str | frozenset[str]) -> float:
    sa = a if isinstance(a, frozenset) else word_set(a)
    sb = b if isinstance(b, frozenset) else word_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
