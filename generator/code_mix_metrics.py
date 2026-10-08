"""Code-mix measurement: word-level language tags, code_mix_ratio, CMI, band check.

Used by the engine's `code_mix_band` hook (generation time) and by the QC
report (Phase 5). No model is needed.

Word-level tagging (tokens = whitespace-separated words, edge punctuation
stripped):

- Native-script text (hi/mr Deva, gu Gujr, mixed with English in Latin):
  the tag follows the token's letters. Native-script letters -> primary
  language, Latin letters -> partner language (English), no letters (numbers,
  punctuation, emoji) -> language-independent. This is exact for our variants,
  which only ever write English in Latin and the target language in its own
  script. It cannot tell a named entity from a code-mixed word (no NER).
- Romanised text (everything in Latin): letters say nothing. The romanised
  variant is the token-by-token transliteration of its native-script parent,
  so each token takes the tag of the parent token at the same position; a
  partner-language token must be unchanged by romanisation (English is passed
  through). If the token counts differ, or an English token was altered, the
  alignment is broken and the variant is unmeasurable (None), which the band
  check reports as a failure. There is no independent romanised-text tagger:
  data/raw has no romanised hi/mr/gu with word-level tags (the LID files are
  Devanagari + English).

Measures (design §, languages.yaml):

    code_mix_ratio = secondary / (primary + secondary)
    CMI            = 100 * (1 - max(primary, secondary) / (primary + secondary))
                     (Das & Gambäck 2014, two languages; 0 when no tagged words)

Language-independent tokens are excluded from both.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from generator.text_utils import script_profile

Tag = Literal["primary", "secondary", "other"]
METRICS_VERSION = "1.0"


def split_token(token: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation)."""
    def edge(ch: str) -> bool:
        return unicodedata.category(ch)[0] in "PSZ" or ch in "।॥"
    i, j = 0, len(token)
    while i < j and edge(token[i]):
        i += 1
    while j > i and edge(token[j - 1]):
        j -= 1
    return token[:i], token[i:j], token[j:]


def tokens(text: str) -> list[str]:
    return text.split()


def tag_native(text: str, native_script: str, partner_script: str,
               scripts: Mapping[str, Sequence[tuple[int, int]]]) -> list[tuple[str, Tag]]:
    out: list[tuple[str, Tag]] = []
    for tok in tokens(text):
        prof = script_profile(split_token(tok)[1], scripts)
        nat, par = prof.get(native_script, 0), prof.get(partner_script, 0)
        if nat == 0 and par == 0:
            out.append((tok, "other"))
        else:
            out.append((tok, "primary" if nat >= par else "secondary"))
    return out


def tag_romanized(text: str, parent_tags: list[tuple[str, Tag]]) -> list[tuple[str, Tag]] | None:
    """Transfer the parent's tags position by position; None if the alignment is broken."""
    toks = tokens(text)
    if len(toks) != len(parent_tags):
        return None
    out: list[tuple[str, Tag]] = []
    for tok, (ptok, tag) in zip(toks, parent_tags, strict=True):
        if tag == "secondary" and split_token(tok)[1] != split_token(ptok)[1]:
            return None
        out.append((tok, tag))
    return out


@dataclass(frozen=True)
class CodeMixMeasure:
    n_primary: int
    n_secondary: int
    n_other: int
    ratio: float | None           # None when there are no tagged words
    cmi: float
    method: str                   # "script_tags" | "aligned_to_native_parent"
    secondary_tokens: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"n_primary": self.n_primary, "n_secondary": self.n_secondary, "n_other": self.n_other,
                "code_mix_ratio": self.ratio, "cmi": self.cmi, "tagging": self.method,
                "secondary_tokens": list(self.secondary_tokens), "metrics_version": METRICS_VERSION}


def summarize(tags: list[tuple[str, Tag]], method: str) -> CodeMixMeasure:
    p = sum(t == "primary" for _, t in tags)
    s = sum(t == "secondary" for _, t in tags)
    o = len(tags) - p - s
    n = p + s
    ratio = round(s / n, 4) if n else None
    cmi = round(100 * (1 - max(p, s) / n), 2) if n else 0.0
    return CodeMixMeasure(p, s, o, ratio, cmi, method,
                          tuple(split_token(tok)[1] for tok, t in tags if t == "secondary"))


def measure(text: str, *, native_script: str, partner_script: str,
            scripts: Mapping[str, Sequence[tuple[int, int]]],
            is_transliterated: bool, parent_text: str | None = None) -> CodeMixMeasure | None:
    """Measure a native-script text directly, a romanised one via its native parent text."""
    if not is_transliterated:
        return summarize(tag_native(text, native_script, partner_script, scripts), "script_tags")
    if parent_text is None:
        return None
    tags = tag_romanized(text, tag_native(parent_text, native_script, partner_script, scripts))
    return None if tags is None else summarize(tags, "aligned_to_native_parent")


def band_check(ratio: float | None, level: str, bands: Mapping[str, tuple[float, float]],
               tolerance: float) -> tuple[Literal["PASS", "WARN", "FAIL"], str | None, dict]:
    """PASS inside [min, max) (L3: max inclusive), WARN within `tolerance` outside, else FAIL."""
    lo, hi = bands[level]
    top = max(b for _, b in bands.values())
    details = {"level": level, "band": [lo, hi], "tolerance": tolerance, "measured_ratio": ratio}
    if ratio is None:
        return "FAIL", "code_mix_unmeasurable", details
    inside = lo <= ratio < hi or (hi == top and ratio == hi)
    if inside:
        return "PASS", None, details
    dist = lo - ratio if ratio < lo else ratio - hi
    details["distance"] = round(dist, 4)
    if dist <= tolerance:
        return "WARN", "code_mix_near_band_edge", details
    return "FAIL", "code_mix_out_of_band", details


def level_bands(settings) -> dict[str, tuple[float, float]]:
    return {k: (v.min_ratio, v.max_ratio) for k, v in settings.languages.code_mix_levels.items()}


def level_for_ratio(ratio: float, bands: Mapping[str, tuple[float, float]]) -> str | None:
    """The level whose band contains `ratio` (reporting only; never used to relabel a variant)."""
    top = max(b for _, b in bands.values())
    for name, (lo, hi) in sorted(bands.items(), key=lambda kv: kv[1][0]):
        if lo <= ratio < hi or (hi == top and ratio == hi):
            return name
    return None
