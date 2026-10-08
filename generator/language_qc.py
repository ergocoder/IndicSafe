"""Language / script QC (Phase 3): one LanguageQCRecord per variant.

Checks, using the thresholds in generation.yaml → qc:

1. Script: the measured dominant script must be the expected one (the
   language's native script, or its romanized script when the variant is
   transliterated) with share >= qc.script.native_min_share /
   romanized_min_share. Same rule as the engine's `expected_script` hook.
2. Language ID (native-script variants only):
   - Lingua (offline, prebuilt wheels on Windows) scores the configured
     candidate languages. It separates en / gu / Devanagari reliably but is
     close to a coin flip between hi and mr on short sentences.
   - hi vs mr is therefore decided by function-word markers (है, में, का …
     vs आहे, मध्ये, च्या …) when at least `min_marker_tokens` markers are
     present: the Devanagari probability mass from Lingua is split by the
     smoothed marker ratio. With fewer markers Lingua's scores stand and the
     result can at best be REVIEW.
   - Romanised text is NOT_APPLICABLE: neither tool identifies romanised
     hi / mr / gu (Lingua calls it English). Its language follows from the
     lineage (parent native variant, checked here) plus the script check.

Status: PASS / REVIEW / FAIL per check; the record's `language_qc_status` is
the worst of them. Nothing here changes a variant; later phases join on
prompt_id.
"""

from __future__ import annotations

import importlib.metadata
import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.config import Settings
from generator.schemas import VariantRecord
from generator.text_utils import dominant_script

LANGUAGE_QC_VERSION = "1.0"
CheckStatus = Literal["PASS", "REVIEW", "FAIL", "NOT_APPLICABLE"]

# Frequent function words that are (nearly) exclusive to one language as
# standalone tokens. Words shared by both (e.g. या, ही, ना, तो) are left out.
HI_MARKERS = frozenset(
    "है हैं था थी थे का की के में से को और नहीं क्या कैसे कौन कौनसा कौनसी मैं हम यह वह इस उस इसे उसे "
    "लिए करना करने करें सकते सकता सकती होता होती रहा रही रहे भी पर किसी कोई अपने अपना अपनी जा जाता "
    "जाती जाए बारे बीच हूँ हूं".split()
)
MR_MARKERS = frozenset(
    "आहे आहेत आणि नाही काय कसे कसा कशी कोण कोणती कोणता कोणते मी आम्ही आपण हे ते त्या साठी मध्ये "
    "होते होता होती करू करणे केला केले केली कोणत्या पासून म्हणून शकतो शकते शकता शकतात असे तर पण मला तुम्ही कसं याचा याची याचे".split()
)
# Marathi case/postposition endings glued to the word (शत्रूकडून, चोरण्यासाठी, हल्ल्यांचा).
MR_SUFFIXES = ("च्या", "साठी", "मध्ये", "कडून", "कडे", "तून", "ांचा", "ांची", "ांचे", "ाचा", "ाची", "ाचे")
_DEVA_WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def hi_mr_markers(text: str) -> dict[str, int]:
    hi = mr = 0
    for w in _DEVA_WORD.findall(text):
        if w in HI_MARKERS:
            hi += 1
        elif w in MR_MARKERS or (len(w) > 4 and w.endswith(MR_SUFFIXES)):
            mr += 1
    return {"hi": hi, "mr": mr}


class LanguageIdentifier(ABC):
    """Scores a text over a fixed set of candidate languages (ISO 639-1)."""

    name: str
    version: str

    @abstractmethod
    def scores(self, text: str) -> dict[str, float]:
        """Probability per candidate language; sums to ~1."""


class LinguaIdentifier(LanguageIdentifier):
    name = "lingua"

    def __init__(self, candidates: list[str]):
        from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

        langs = [Language.from_iso_code_639_1(getattr(IsoCode639_1, c.upper())) for c in candidates]
        self._detector = LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()
        self.version = importlib.metadata.version("lingua-language-detector")

    def scores(self, text: str) -> dict[str, float]:
        return {c.language.iso_code_639_1.name.lower(): round(c.value, 4)
                for c in self._detector.compute_language_confidence_values(text)}


def build_language_identifier(settings: Settings) -> LanguageIdentifier:
    cfg = settings.generation.qc.language
    if cfg.detector != "lingua_markers":
        raise ValueError(f"unknown language detector {cfg.detector!r}")
    return LinguaIdentifier(cfg.candidates)


class LanguageQCRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    seed_id: str
    parent_prompt_id: str | None
    language_qc_version: str = LANGUAGE_QC_VERSION

    expected_language: str
    expected_script: str
    is_transliterated: bool

    measured_script: str | None
    script_share: float
    script_min_share: float
    script_status: CheckStatus
    script_reason: str | None = None

    lid_detector: str | None = None          # e.g. "lingua-2.2.0+hi_mr_markers-1.0"
    lid_method: Literal["lingua", "lingua+markers"] | None = None
    lid_language: str | None = None
    lid_confidence: float | None = None       # probability of lid_language
    expected_language_confidence: float | None = None
    lid_scores: dict[str, float] = {}
    hi_mr_markers: dict[str, int] | None = None
    lid_status: CheckStatus
    lid_reason: str | None = None

    language_qc_status: Literal["PASS", "REVIEW", "FAIL"]
    reasons: list[str] = []


def _expected_script(settings: Settings, v: VariantRecord) -> str:
    lang = settings.languages.languages[v.language]
    if v.is_transliterated:
        return lang.romanized_script or "Latn"
    return lang.native_script


def check_variant(settings: Settings, v: VariantRecord, lid: LanguageIdentifier) -> LanguageQCRecord:
    qc = settings.generation.qc
    scripts = {k: s.ranges for k, s in settings.languages.scripts.items()}
    exp_script = _expected_script(settings, v)
    native = not v.is_transliterated
    min_share = qc.script.native_min_share if native else qc.script.romanized_min_share
    measured, share = dominant_script(v.prompt, scripts)
    if measured != exp_script:
        script_status, script_reason = "FAIL", "script_mismatch"
    elif share < min_share:
        script_status, script_reason = "FAIL", "low_script_share"
    else:
        script_status, script_reason = "PASS", None

    out: dict = dict(
        prompt_id=v.prompt_id, seed_id=v.seed_id, parent_prompt_id=v.parent_prompt_id,
        expected_language=v.language, expected_script=exp_script, is_transliterated=v.is_transliterated,
        measured_script=measured, script_share=share, script_min_share=min_share,
        script_status=script_status, script_reason=script_reason,
    )
    if not native or v.secondary_language is not None:
        out.update(lid_status="NOT_APPLICABLE", lid_reason="lid_unreliable_on_romanised_or_code_mixed_text")
    else:
        out.update(_identify(v.prompt, v.language, lid, settings))

    statuses = [out["script_status"], out["lid_status"]]
    reasons = [r for r in (out.get("script_reason"), out.get("lid_reason"))
               if r and r != "lid_unreliable_on_romanised_or_code_mixed_text"]
    overall = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
    return LanguageQCRecord(**out, language_qc_status=overall, reasons=reasons)


def _identify(text: str, expected: str, lid: LanguageIdentifier, settings: Settings) -> dict:
    cfg = settings.generation.qc.language
    scores = dict(lid.scores(text))
    markers = None
    method = "lingua"
    if "hi" in scores and "mr" in scores and max(sorted(scores), key=lambda k: scores[k]) in ("hi", "mr"):
        markers = hi_mr_markers(text)
        n = markers["hi"] + markers["mr"]
        if n >= cfg.min_marker_tokens:
            deva = scores["hi"] + scores["mr"]
            p_hi = (markers["hi"] + 0.5) / (n + 1)
            scores["hi"], scores["mr"] = round(deva * p_hi, 4), round(deva * (1 - p_hi), 4)
            method = "lingua+markers"
    top = max(sorted(scores), key=lambda k: scores[k])
    conf = scores[top]
    if top == expected:
        if conf >= cfg.pass_confidence:
            status, reason = "PASS", None
        else:
            status, reason = "REVIEW", "low_language_confidence"
    elif conf >= cfg.pass_confidence:
        status, reason = "FAIL", "language_mismatch"
    else:
        status, reason = "REVIEW", "possible_language_mismatch"
    return dict(
        lid_detector=f"{lid.name}-{lid.version}+hi_mr_markers-{LANGUAGE_QC_VERSION}",
        lid_method=method, lid_language=top, lid_confidence=conf,
        expected_language_confidence=scores.get(expected, 0.0), lid_scores=scores,
        hi_mr_markers=markers, lid_status=status, lid_reason=reason,
    )
