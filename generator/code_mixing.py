r"""Code-mixing (Phase 4): native-script L0 translation -> L1 / L2 with English. No LLM.

    en root ──translation──► hi/Deva (L0) ──code_mixing L1──► hi/Deva+en ──transliteration──► hi/Latn+en
                                          └─code_mixing L2──► hi/Deva+en ──transliteration──► hi/Latn+en

Method `mt_lexical_swap` (deterministic):

1. Align. Each English content word of the source (not a stopword, at least
   `min_word_chars` letters) is matched to at most one word of the
   translation, one to one, by two signals:
   - phonetic: the romanised target word looks like the English word
     (difflib ratio >= `phonetic_min_similarity`, both at least 4 letters).
     This finds loanwords the MT
     already transliterated (फिशिंग / phishing, डेटा / data): the most natural
     words to switch back.
   - lexical: the first word (before any hyphen) of the word's own IndicTrans2 translation (the
     same MT model, word in isolation) equals the target word, or its stem
     (trailing vowel signs removed, >= 3 code points) is a prefix of it
     (हल्ले -> हल्ल्यांचा).
   Swap order: phonetic matches, then lexical ones, then phonetic matches whose
   romanisation already spells the English word (similarity >= 0.9: the swap
   would not change the romanised variant); within a tier, higher score, then
   source order.
2. Swap. The first k aligned words are replaced by the English word. A case
   marker written onto the word (`clitics` per language, e.g. Marathi च्या) is
   kept as its own token after it (बँकेच्या -> "bank च्या"). k is chosen so the measured
   code_mix_ratio (code_mix_metrics, same tagger as the band check) is inside
   the level's band and closest to `target_point` of it. The swap order is
   fixed, so L1's swapped words are a subset of L2's.
3. If no k reaches the band (too few aligned words, very short prompts), the
   closest attempt is still returned. The engine's `code_mix_band` hook then
   FAILs the variant: it is kept for audit and never relabelled to the level it
   happens to reach.

Known limits: alignment by isolated-word translation misses words the MT
renders differently in context, and a prefix match can pick a wrong inflected
word; there is no POS tagger, so verbs and adjectives are swapped as readily
as nouns; inflection is dropped with the native word (attacks for हमलों).
Native-speaker review checks naturalness.
"""

from __future__ import annotations

import difflib
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from backend.config import ProviderConfig, Settings
from generator import code_mix_metrics as cmm
from generator.schemas import VariantRecord
from generator.text_utils import script_profile
from generator.transformation_engine import (
    ProviderError,
    ProviderInfo,
    ProviderOutput,
    ProviderUnavailableError,
    ResolvedRequest,
    TargetCondition,
    Transformation,
    TransformationError,
    build_provider,
    check_params,
    language_config,
)
from generator.translation import TranslationProvider
from generator.transliteration import Transliterator

ADAPTER_VERSION = "1.0"
PHONETIC_MIN_CHARS = 4      # shorter words look alike by chance (से "se" ~ "use")
# A loanword whose romanisation already spells the English word (फिशिंग -> "phishing")
# changes the native text when swapped but not the romanised one, so it is swapped last.
LATN_INVISIBLE_SIMILARITY = 0.9


@dataclass(frozen=True)
class LexicalSwapOptions:
    target_point: float = 0.5
    min_word_chars: int = 3
    phonetic_min_similarity: float = 0.70
    min_stem_chars: int = 2
    stopwords_en: tuple[str, ...] = ()
    clitics: Mapping[str, tuple[str, ...]] | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "LexicalSwapOptions":
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise ProviderUnavailableError(f"mt_lexical_swap: unknown option(s) {sorted(unknown)}")
        o = dict(options)
        o["stopwords_en"] = tuple(str(w).lower() for w in o.get("stopwords_en", ()))
        o["clitics"] = {k: tuple(v) for k, v in (o.get("clitics") or {}).items()}
        opts = cls(**o)
        if not 0.0 <= opts.target_point <= 1.0:
            raise ProviderUnavailableError("mt_lexical_swap: target_point must be in [0, 1]")
        return opts


@dataclass(frozen=True)
class Alignment:
    src_index: int          # token position in the English source
    src_word: str           # English surface form (edge punctuation stripped)
    tgt_index: int          # token position in the target text
    tgt_word: str
    method: str             # "phonetic" | "lexical"
    score: float
    clitic: str | None      # case marker kept after the English word


# ------------------------------------------------------------------ helpers

_NUKTA = "़"


def _norm(word: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).replace(_NUKTA, "")).lower()


def _stem(word: str) -> str:
    """Strip trailing dependent vowel signs, virama, anusvara, candrabindu, visarga."""
    i = len(word)
    while i > 0 and unicodedata.category(word[i - 1]) in ("Mn", "Mc"):
        i -= 1
    return word[:i]


def _is_native(word: str, native_script: str, scripts) -> bool:
    prof = script_profile(word, scripts)
    return bool(prof) and set(prof) == {native_script}


def english_form(word: str, position: int) -> str:
    """Keep acronyms and inner capitals (SIT, iPhone) and mid-sentence names; lowercase a sentence-initial word."""
    if (len(word) > 1 and word.isupper()) or any(c.isupper() for c in word[1:]):
        return word
    return word.lower() if position == 0 else word


def content_words(source: str, opts: LexicalSwapOptions) -> list[tuple[int, str]]:
    out = []
    for i, tok in enumerate(cmm.tokens(source)):
        core = cmm.split_token(tok)[1]
        if len(core) >= opts.min_word_chars and core.replace("-", "").isalpha() and core.isascii() \
                and core.lower() not in opts.stopwords_en:
            out.append((i, core))
    return out


def align(source: str, target: str, *, language: str, native_script: str, scripts,
          word_translations: Mapping[str, str], romanize: Callable[[str], str] | None,
          opts: LexicalSwapOptions) -> list[Alignment]:
    """One-to-one alignment of English content words to native target words, in swap order."""
    clitics = sorted((opts.clitics or {}).get(language, ()), key=len, reverse=True)
    targets = []
    for j, tok in enumerate(cmm.tokens(target)):
        core = cmm.split_token(tok)[1]
        if not core or not _is_native(core, native_script, scripts):
            continue
        clitic = next((c for c in clitics if core.endswith(c) and len(core) > len(c) + 1), None)
        base = core[: -len(clitic)] if clitic else core
        rom = romanize(base).lower() if romanize else None
        targets.append((j, core, _norm(core), base, rom, clitic))

    cands = []
    for i, word in content_words(source, opts):
        w = word.lower()
        tr = (word_translations.get(word) or word_translations.get(w) or "").split()
        head = _norm(cmm.split_token(tr[0])[1].split("-")[0]) if tr else ""
        stem = _stem(head)
        for j, core, ncore, base, rom, clitic in targets:
            if rom is not None and min(len(rom), len(w)) >= PHONETIC_MIN_CHARS:
                sim = difflib.SequenceMatcher(None, rom, w).ratio()
                if sim >= opts.phonetic_min_similarity:
                    tier = 2 if sim >= LATN_INVISIBLE_SIMILARITY else 0
                    cands.append((tier, -sim, i, j, Alignment(i, word, j, core, "phonetic", round(sim, 3), clitic)))
                    continue
            if head and (ncore == head or (len(stem) >= max(3, opts.min_stem_chars) and ncore.startswith(stem))):
                score = 1.0 if ncore == head else round(len(stem) / len(ncore), 3)
                cands.append((1, -score, i, j, Alignment(i, word, j, core, "lexical", score, clitic)))
    used_i, used_j, out = set(), set(), []
    for *_, a in sorted(cands, key=lambda c: c[:4]):
        if a.src_index in used_i or a.tgt_index in used_j:
            continue
        used_i.add(a.src_index)
        used_j.add(a.tgt_index)
        out.append(a)
    return out


def apply_swaps(target: str, alignments: Iterable[Alignment]) -> str:
    toks = cmm.tokens(target)
    for a in alignments:
        lead, core, trail = cmm.split_token(toks[a.tgt_index])
        eng = english_form(a.src_word, a.src_index)
        toks[a.tgt_index] = f"{lead}{eng} {a.clitic}{trail}" if a.clitic else f"{lead}{eng}{trail}"
    return " ".join(toks)


def choose_k(target: str, alignments: list[Alignment], band: tuple[float, float], target_point: float,
             measure: Callable[[str], float | None]) -> tuple[int, str, float | None]:
    """Smallest-distance k: inside the band nearest the target point, else nearest the band."""
    lo, hi = band
    goal = lo + target_point * (hi - lo)
    best = None
    for k in range(0, len(alignments) + 1):
        text = apply_swaps(target, alignments[:k])
        r = measure(text)
        if r is None:
            continue
        inside = lo <= r < hi
        key = (0 if inside else 1, abs(r - goal) if inside else min(abs(r - lo), abs(r - hi)), k)
        if best is None or key < best[0]:
            best = (key, k, text, r)
    if best is None:
        return 0, target, None
    return best[1], best[2], best[3]


# ------------------------------------------------------------------ provider


class CodeMixer(ABC):
    @property
    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @abstractmethod
    def supports(self, language: str, secondary_language: str) -> bool: ...

    @abstractmethod
    def mix(self, source: str, target: str, *, language: str, script: str, secondary_language: str,
            level: str, band: tuple[float, float], seed: int) -> ProviderOutput:
        """Rewrite `target` (translation of `source`) to the code-mix level. Raise ProviderError on failure."""


class LexicalSwapCodeMixer(CodeMixer):
    def __init__(self, name: str, settings: Settings, targets: list[str], opts: LexicalSwapOptions,
                 translator: TranslationProvider, romanizer: Transliterator | None):
        self.name = name
        self.settings = settings
        self.targets = set(targets)
        self.opts = opts
        self.translator = translator
        self.romanizer = romanizer
        self._scripts = {k: v.ranges for k, v in settings.languages.scripts.items()}
        rinfo = romanizer.info if romanizer else None
        self._info = ProviderInfo(
            name=name,
            version=(f"{ADAPTER_VERSION}+align:{translator.info.name}-{translator.info.version}"
                     + (f"+phon:{rinfo.name}-{rinfo.version}" if rinfo else "")
                     + f"+tp{opts.target_point}"),
            model=translator.info.model,
            generation_method="rule",
        )

    @classmethod
    def from_config(cls, name: str, cfg: ProviderConfig, settings: Settings, translator: TranslationProvider,
                    romanizer: Transliterator | None) -> "LexicalSwapCodeMixer":
        return cls(name, settings, cfg.target_languages, LexicalSwapOptions.from_mapping(cfg.options),
                   translator, romanizer)

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, language, secondary_language):
        return language in self.targets and secondary_language == "en" and self.translator.supports("en", language)

    def prepare(self, sources: Iterable[str], language: str) -> None:
        """Batch-translate every content word once (the translator caches; per-item calls hit the cache)."""
        words = sorted({w for s in sources for _, w in content_words(s, self.opts)})
        if words:
            script = self.settings.languages.languages[language].native_script
            self.translator.translate_batch(words, source_language="en", target_language=language,
                                            target_script=script, seeds=[0] * len(words))

    def _romanize(self, language: str, script: str) -> Callable[[str], str] | None:
        if self.romanizer is None or not self.romanizer.supports(language, script, "Latn"):
            return None
        return lambda w: self.romanizer.transliterate(w, language=language, source_script=script,
                                                      target_script="Latn", seed=0).text

    def mix(self, source, target, *, language, script, secondary_language, level, band, seed):
        if not self.supports(language, secondary_language):
            raise ProviderError(f"{self.name} does not support {language}-{secondary_language}")
        words = [w for _, w in content_words(source, self.opts)]
        try:
            outs = self.translator.translate_batch(words, source_language="en", target_language=language,
                                                   target_script=script, seeds=[0] * len(words)) if words else []
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 - any adapter failure is a provider failure
            raise ProviderError(f"word translation failed: {type(e).__name__}: {e}") from e
        lexicon = {w: o.text for w, o in zip(words, outs, strict=True)}
        alignments = align(source, target, language=language, native_script=script, scripts=self._scripts,
                           word_translations=lexicon, romanize=self._romanize(language, script), opts=self.opts)
        partner_script = self.settings.languages.languages[secondary_language].native_script

        def ratio(text: str) -> float | None:
            return cmm.summarize(cmm.tag_native(text, script, partner_script, self._scripts), "script_tags").ratio

        k, text, planned = choose_k(target, alignments, band, self.opts.target_point, ratio)
        return ProviderOutput(text, {
            "method": "mt_lexical_swap",
            "level": level,
            "band": list(band),
            "target_ratio": round(band[0] + self.opts.target_point * (band[1] - band[0]), 4),
            "planned_ratio": planned,
            "n_source_content_words": len(words),
            "n_aligned": len(alignments),
            "n_swapped": k,
            "alignments": [asdict(a) for a in alignments],
            "swapped": [{"en": english_form(a.src_word, a.src_index), "native": a.tgt_word, "method": a.method,
                         "clitic_kept": a.clitic} for a in alignments[:k]],
            "word_translations": lexicon,
            "aligner_model": self.translator.info.model,
        })


def build_code_mixer(settings: Settings, translator: TranslationProvider,
                     romanizer: Transliterator | None, name: str | None = None) -> CodeMixer:
    factories = {
        ("code_mixing", "mt_lexical_swap"):
            lambda n, c: LexicalSwapCodeMixer.from_config(n, c, settings, translator, romanizer),
    }
    return build_provider(settings, "code_mixing", name, factories=factories)


# ------------------------------------------------------------ transformation


class CodeMixingTransformation(Transformation):
    """Parent: a native-script, monolingual translation. Source: the English root of its lineage."""

    transformation_type = "code_mixing"
    PARAMS = ("level", "secondary_language")

    def __init__(self, mixer: CodeMixer, variants: Mapping[str, VariantRecord]):
        self.mixer = mixer
        self.variants = variants        # the engine's variants, to find the English source

    @property
    def provider_info(self) -> ProviderInfo:
        return self.mixer.info

    def _source(self, parent: VariantRecord, partner: str) -> VariantRecord:
        root = self.variants.get(parent.lineage[0]) if parent.lineage else None
        if root is None or root.language != partner or root.is_transliterated:
            raise TransformationError(
                f"code_mixing: {parent.prompt_id} has no {partner!r} source root in its lineage")
        return root

    def resolve(self, parent: VariantRecord, params: Mapping[str, Any], settings: Settings) -> ResolvedRequest:
        check_params(params, self.PARAMS, self.transformation_type)
        lang = language_config(settings, parent.language, enabled=True)
        partner = params.get("secondary_language", lang.code_mix_partner)
        if partner is None or partner != lang.code_mix_partner:
            raise TransformationError(f"code_mixing: {parent.language!r} mixes only with {lang.code_mix_partner!r}")
        level = params.get("level")
        if level not in settings.generation.code_mixing.levels or level not in lang.code_mix_levels:
            raise TransformationError(f"code_mixing: level {level!r} is not configured for {parent.language!r}")
        if parent.is_transliterated or parent.secondary_language is not None or parent.script != lang.native_script:
            raise TransformationError(
                f"code_mixing: parent must be monolingual native-script text ({parent.prompt_id} is not)")
        if not self.mixer.supports(parent.language, partner):
            raise TransformationError(
                f"code_mixing: provider {self.mixer.info.name!r} does not support {parent.language}-{partner}")
        src = self._source(parent, partner)
        band = settings.languages.code_mix_levels[level]
        return ResolvedRequest(
            parameters={"language": parent.language, "secondary_language": partner, "level": level,
                        "band": [band.min_ratio, band.max_ratio], "source_prompt_id": src.prompt_id,
                        "source_content_hash": src.content_hash},
            target=TargetCondition(language=parent.language, script=lang.native_script,
                                   secondary_language=partner, is_transliterated=False,
                                   code_mix_level=level, mixing_method=self.mixer.info.name),
        )

    def run(self, parent: VariantRecord, request: ResolvedRequest, derived_seed: int) -> ProviderOutput:
        p = request.parameters
        src = self.variants[p["source_prompt_id"]]
        return self.mixer.mix(src.prompt, parent.prompt, language=p["language"], script=parent.script,
                              secondary_language=p["secondary_language"], level=p["level"],
                              band=tuple(p["band"]), seed=derived_seed)
