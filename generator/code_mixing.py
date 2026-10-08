r"""Code-mixing (Phase 4): native-script L0 translation -> L1 / L2 with English. No LLM.

    en root ──translation──► hi/Deva (L0) ──code_mixing L1──► hi/Deva+en ──transliteration──► hi/Latn+en
                                          └─code_mixing L2──► hi/Deva+en ──transliteration──► hi/Latn+en

Method `mt_lexical_swap` (deterministic):

1. English units (generator/english_pos.py). spaCy POS-tags the English seed;
   only NOUN, PROPN, ADJ, hyphenated compounds and (optionally) VERB are
   candidates; function words, `stopwords_en` and `never_swap_en`
   (regarding, including, ...) never are. Consecutive proper nouns / one named
   entity form one unit ("Lok Sabha"), swapped completely or not at all.
2. Align each unit one-to-one to native words of the translation:
   - phonetic: the romanised target word looks like the English word
     (difflib ratio >= `phonetic_min_similarity`, both >= 4 letters, same
     onset sound class: "mate" (માટે) is not "date"). Finds loanwords the MT
     already transliterated (फिशिंग / phishing, डेटा / data).
   - lexical: the first word (before any hyphen) of the unit's own IndicTrans2
     translation equals the target word, or its stem (trailing vowel signs
     removed, >= 3 code points) is a prefix of it (हल्ले -> हल्ल्यांचा).
   - multi-word units: one native compound word built from the parts' heads
     (लोक + सभा -> लोकसभा, लोकसभेचे) or n consecutive native words, one per part.
   Native postpositions / function words (`function_words` per language) are
   never replaced ("regarding बारे में" cannot happen).
3. Verbs: the native verb is not kept next to the English one ("celebrate
   मनाने"). The English lemma takes the do-verb construction, keeping the
   original inflection via `verbs.<lang>` in config:
   - next word already a do-verb form (ઉજવણી કરવા, आयोजित किया): replace the aligned word only;
   - next word a light verb stem + suffix (जश्न मनाने, સેવા આપી): replace both
     with "<lemma> <do-form for that suffix>" -> "celebrate करने", "serve કરી";
   - the aligned word itself stem + suffix (चुराने, चोरण्यासाठी): "<lemma> <do-form>"
     ("steal करने", "steal करण्यासाठी"); only suffixes of >= 2 code points;
   - otherwise the verb is not swapped.
4. Swap. Order: phonetic, lexical non-verbs, verbs, then loanwords whose
   romanisation already spells the English word (swapping those would not
   change the romanised variant); within a tier by score, then source order.
   Case markers written onto a swapped word (`clitics`) stay as their own token
   (बँकेच्या -> "bank च्या"). k is chosen so the measured code_mix_ratio is
   inside the level's band and closest to `target_point` of it; L1's swaps are
   a subset of L2's.
5. If no k reaches the band, the closest attempt is still returned and the
   engine's `code_mix_band` hook FAILs it (kept for audit, never relabelled).

Known limits: isolated-word MT misses words rendered differently in context;
a prefix match can pick a wrong inflection; inflection of swapped nouns is
dropped (attacks for हमलों); the verb tables cover common non-finite and
habitual forms only. Native-speaker review checks naturalness.
"""

from __future__ import annotations

import difflib
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from backend.config import ProviderConfig, Settings
from generator import code_mix_metrics as cmm
from generator.english_pos import EnglishAnalyzer, EnUnit, build_analyzer
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

ADAPTER_VERSION = "2.0"
PHONETIC_MIN_CHARS = 4      # shorter words look alike by chance (से "se" ~ "use")
# A loanword whose romanisation already spells the English word (फिशिंग -> "phishing")
# changes the native text when swapped but not the romanised one, so it is swapped last.
LATN_INVISIBLE_SIMILARITY = 0.9
SPAN_PART_PHONETIC_MIN_CHARS = 3   # parts of a name span ("Pro" Kabaddi League) are matched together
_ONSET_CLASSES = ("aeiou", "ckqs", "vwb", "fp", "jzg")


@dataclass(frozen=True)
class VerbRules:
    do_forms: frozenset[str] = frozenset()     # existing do-verb forms: keep, replace the word before
    light_stems: tuple[str, ...] = ()          # light verbs turned into the do-verb (मना, दे, આપ)
    suffixes: Mapping[str, str] = field(default_factory=dict)   # inflection suffix -> do-verb form


@dataclass(frozen=True)
class LexicalSwapOptions:
    target_point: float = 0.5
    min_word_chars: int = 3
    phonetic_min_similarity: float = 0.70
    min_stem_chars: int = 2
    stopwords_en: tuple[str, ...] = ()
    never_swap_en: tuple[str, ...] = ()
    english_tagger: str = "spacy"
    spacy_model: str = "en_core_web_sm"
    swap_pos: tuple[str, ...] = ("NOUN", "PROPN", "ADJ", "COMPOUND", "VERB")
    swap_verbs: bool = True
    min_simple_suffix_chars: int = 2
    clitics: Mapping[str, tuple[str, ...]] | None = None
    function_words: Mapping[str, frozenset[str]] | None = None
    verbs: Mapping[str, VerbRules] | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "LexicalSwapOptions":
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise ProviderUnavailableError(f"mt_lexical_swap: unknown option(s) {sorted(unknown)}")
        o = dict(options)
        for k in ("stopwords_en", "never_swap_en"):
            o[k] = tuple(str(w).lower() for w in o.get(k, ()))
        if "swap_pos" in o:
            o["swap_pos"] = tuple(o["swap_pos"])
        o["clitics"] = {k: tuple(v) for k, v in (o.get("clitics") or {}).items()}
        o["function_words"] = {k: frozenset(v) for k, v in (o.get("function_words") or {}).items()}
        o["verbs"] = {k: VerbRules(frozenset(v.get("do_forms", ())), tuple(v.get("light_stems", ())),
                                   dict(v.get("suffixes", {})))
                      for k, v in (o.get("verbs") or {}).items()}
        opts = cls(**o)
        if not 0.0 <= opts.target_point <= 1.0:
            raise ProviderUnavailableError("mt_lexical_swap: target_point must be in [0, 1]")
        return opts

    @property
    def unit_pos(self) -> set[str]:
        return {p for p in self.swap_pos if p != "VERB" or self.swap_verbs}


@dataclass(frozen=True)
class Alignment:
    src_indices: tuple[int, ...]   # English token positions of the unit
    src_text: str                  # English unit text (surface)
    pos: str
    tgt_indices: tuple[int, ...]   # consecutive native token positions replaced
    tgt_text: str
    method: str                    # "phonetic" | "lexical" | "compound" | "span"
    score: float
    clitic: str | None             # case marker kept after the English unit
    replacement: str               # what replaces the native tokens
    construction: str | None = None   # verbs: do_verb_kept | light_verb_to_do | simple_verb_to_do


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


def _same_onset(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a[0] == b[0] or any(a[0] in c and b[0] in c for c in _ONSET_CLASSES)


def _head(translation: str) -> str:
    """First word of a word's isolated translation, before any hyphen, normalised."""
    tr = translation.split()
    return _norm(cmm.split_token(tr[0])[1].split("-")[0]) if tr else ""


def _lexical_score(ncore: str, head: str, min_stem: int) -> float | None:
    if not head:
        return None
    if ncore == head:
        return 1.0
    stem = _stem(head)
    if len(stem) >= max(3, min_stem) and ncore.startswith(stem):
        return round(len(stem) / len(ncore), 3)
    return None


def english_form(word: str, position: int) -> str:
    """Keep acronyms and inner capitals (SIT, iPhone) and mid-sentence names; lowercase a sentence-initial word."""
    if (len(word) > 1 and word.isupper()) or any(c.isupper() for c in word[1:]):
        return word
    return word.lower() if position == 0 else word


def unit_english(unit: EnUnit) -> str:
    return " ".join(english_form(w, i) for w, i in zip(unit.words, unit.indices, strict=True))


def verb_construction(cores: list[str], j: int, lemma: str, rules: VerbRules | None,
                      min_simple: int) -> tuple[tuple[int, ...], str, str] | None:
    """(native positions replaced, replacement, construction) for an English verb aligned to word j."""
    if rules is None or not rules.suffixes:
        return None
    nxt = cores[j + 1] if j + 1 < len(cores) else None
    if nxt in rules.do_forms:
        return (j,), lemma, "do_verb_kept"
    sufs = sorted(rules.suffixes, key=len, reverse=True)
    if nxt is not None:
        for stem in rules.light_stems:
            if nxt.startswith(stem) and nxt[len(stem):] in rules.suffixes:
                return (j, j + 1), f"{lemma} {rules.suffixes[nxt[len(stem):]]}", "light_verb_to_do"
    for suf in sufs:
        if len(suf) >= min_simple and cores[j].endswith(suf) and len(cores[j]) - len(suf) >= 2:
            return (j,), f"{lemma} {rules.suffixes[suf]}", "simple_verb_to_do"
    return None


def lexicon_keys(units: Iterable[EnUnit]) -> list[str]:
    """Texts whose isolated translation the aligner needs: each unit and each part of a multi-word unit."""
    keys: dict[str, None] = {}
    for u in units:
        keys[u.text] = None
        if len(u.words) > 1:
            keys.update(dict.fromkeys(u.words))
    return list(keys)


# ---------------------------------------------------------------- alignment


@dataclass(frozen=True)
class _Target:
    j: int
    core: str
    ncore: str
    rom: str | None
    clitic: str | None


def _targets(target: str, language: str, native_script: str, scripts, romanize, opts) -> tuple[list[str], list[_Target]]:
    clitics = sorted((opts.clitics or {}).get(language, ()), key=len, reverse=True)
    blocked = (opts.function_words or {}).get(language, frozenset())
    cores = [cmm.split_token(t)[1] for t in cmm.tokens(target)]
    out = []
    for j, core in enumerate(cores):
        if not core or core in blocked or not _is_native(core, native_script, scripts):
            continue
        clitic = next((c for c in clitics if core.endswith(c) and len(core) > len(c) + 1), None)
        base = core[: -len(clitic)] if clitic else core
        out.append(_Target(j, core, _norm(core), romanize(base).lower() if romanize else None, clitic))
    return cores, out


def _match_word(word: str, head: str, t: _Target, opts: LexicalSwapOptions,
                min_chars: int = PHONETIC_MIN_CHARS) -> tuple[str, float] | None:
    """('phonetic' | 'lexical', score) for one English word (or joined span) against one native word."""
    w = word.lower().replace(" ", "")
    if t.rom is not None and min(len(t.rom), len(w)) >= min_chars and _same_onset(t.rom, w):
        sim = difflib.SequenceMatcher(None, t.rom, w).ratio()
        if sim >= opts.phonetic_min_similarity:
            return "phonetic", round(sim, 3)
    score = _lexical_score(t.ncore, head, opts.min_stem_chars)
    return ("lexical", score) if score is not None else None


def align(target: str, *, language: str, native_script: str, scripts, units: list[EnUnit],
          word_translations: Mapping[str, str], romanize: Callable[[str], str] | None,
          opts: LexicalSwapOptions) -> list[Alignment]:
    """One-to-one alignment of English units to native words, in swap order."""
    cores, targets = _targets(target, language, native_script, scripts, romanize, opts)
    by_j = {t.j: t for t in targets}
    rules = (opts.verbs or {}).get(language)
    cands: list[tuple[tuple, Alignment]] = []
    for u in units:
        eng = unit_english(u)
        if len(u.words) == 1:
            head = _head(word_translations.get(u.text, ""))
            for t in targets:
                m = _match_word(u.text, head, t, opts)
                if m is None:
                    continue
                method, score = m
                if u.pos == "VERB":
                    vc = verb_construction(cores, t.j, u.lemma, rules, opts.min_simple_suffix_chars)
                    if vc is None:
                        continue
                    idx, repl, how = vc
                    a = Alignment(u.indices, u.text, u.pos, idx, " ".join(cores[i] for i in idx), method,
                                  score, None, repl, how)
                    tier = 2
                else:
                    repl = f"{eng} {t.clitic}" if t.clitic else eng
                    a = Alignment(u.indices, u.text, u.pos, (t.j,), t.core, method, score, t.clitic, repl)
                    tier = 0 if method == "phonetic" else 1
                    if method == "phonetic" and score >= LATN_INVISIBLE_SIMILARITY:
                        tier = 3
                cands.append(((tier, -score, u.indices[0], t.j), a))
            continue
        # multi-word unit: one native compound word ...
        heads = ["".join(_head(word_translations.get(w, "")) for w in u.words),
                 _head(word_translations.get(u.text, "").replace(" ", ""))]
        for t in targets:
            best = max((m for h in heads if h for m in [_match_word(u.text, h, t, opts)] if m),
                       key=lambda m: m[1], default=None)
            if best is not None:
                repl = f"{eng} {t.clitic}" if t.clitic else eng
                cands.append(((0, -best[1], u.indices[0], t.j),
                              Alignment(u.indices, u.text, u.pos, (t.j,), t.core, "compound", best[1], t.clitic, repl)))
        # ... or one native word per part, consecutive
        n = len(u.words)
        for t in targets:
            run = [by_j.get(t.j + k) for k in range(n)]
            if any(r is None for r in run):
                continue
            ms = [_match_word(w, _head(word_translations.get(w, "")), r, opts, SPAN_PART_PHONETIC_MIN_CHARS)
                  for w, r in zip(u.words, run, strict=True)]
            if all(ms):
                last = run[-1]
                repl = f"{eng} {last.clitic}" if last.clitic else eng
                score = round(min(m[1] for m in ms), 3)
                cands.append(((0, -score, u.indices[0], t.j),
                              Alignment(u.indices, u.text, u.pos, tuple(r.j for r in run),
                                        " ".join(r.core for r in run), "span", score, last.clitic, repl)))
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    out = []
    for _, a in sorted(cands, key=lambda c: c[0]):
        if used_src & set(a.src_indices) or used_tgt & set(a.tgt_indices):
            continue
        used_src |= set(a.src_indices)
        used_tgt |= set(a.tgt_indices)
        out.append(a)
    return out


def apply_swaps(target: str, alignments: Iterable[Alignment]) -> str:
    toks = cmm.tokens(target)
    for a in sorted(alignments, key=lambda a: a.tgt_indices[0], reverse=True):
        first, last = a.tgt_indices[0], a.tgt_indices[-1]
        lead = cmm.split_token(toks[first])[0]
        trail = cmm.split_token(toks[last])[2]
        toks[first:last + 1] = [f"{lead}{a.replacement}{trail}"]
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
                 translator: TranslationProvider, romanizer: Transliterator | None,
                 analyzer: EnglishAnalyzer | None = None):
        self.name = name
        self.settings = settings
        self.targets = set(targets)
        self.opts = opts
        self.translator = translator
        self.romanizer = romanizer
        self.analyzer = analyzer or build_analyzer(opts.english_tagger, opts.spacy_model)
        self._scripts = {k: v.ranges for k, v in settings.languages.scripts.items()}
        rinfo = romanizer.info if romanizer else None
        self._info = ProviderInfo(
            name=name,
            version=(f"{ADAPTER_VERSION}+align:{translator.info.name}-{translator.info.version}"
                     + (f"+phon:{rinfo.name}-{rinfo.version}" if rinfo else "")
                     + f"+pos:{self.analyzer.name}-{self.analyzer.version}"
                     + f"+verbs{int(opts.swap_verbs)}+tp{opts.target_point}"),
            model=translator.info.model,
            generation_method="rule",
        )

    @classmethod
    def from_config(cls, name: str, cfg: ProviderConfig, settings: Settings, translator: TranslationProvider,
                    romanizer: Transliterator | None, analyzer: EnglishAnalyzer | None = None) -> "LexicalSwapCodeMixer":
        return cls(name, settings, cfg.target_languages, LexicalSwapOptions.from_mapping(cfg.options),
                   translator, romanizer, analyzer)

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, language, secondary_language):
        return language in self.targets and secondary_language == "en" and self.translator.supports("en", language)

    def units(self, source: str) -> list[EnUnit]:
        o = self.opts
        return self.analyzer.units(source, stopwords=set(o.stopwords_en), never_swap=set(o.never_swap_en),
                                   min_chars=o.min_word_chars, swap_pos=o.unit_pos)

    def prepare(self, sources: Iterable[str], language: str) -> None:
        """Batch-translate every unit once (the translator caches; per-item calls hit the cache)."""
        keys = sorted({k for s in sources for k in lexicon_keys(self.units(s))})
        if keys:
            script = self.settings.languages.languages[language].native_script
            self.translator.translate_batch(keys, source_language="en", target_language=language,
                                            target_script=script, seeds=[0] * len(keys))

    def _romanize(self, language: str, script: str) -> Callable[[str], str] | None:
        if self.romanizer is None or not self.romanizer.supports(language, script, "Latn"):
            return None
        return lambda w: self.romanizer.transliterate(w, language=language, source_script=script,
                                                      target_script="Latn", seed=0).text

    def mix(self, source, target, *, language, script, secondary_language, level, band, seed):
        if not self.supports(language, secondary_language):
            raise ProviderError(f"{self.name} does not support {language}-{secondary_language}")
        units = self.units(source)
        keys = lexicon_keys(units)
        try:
            outs = self.translator.translate_batch(keys, source_language="en", target_language=language,
                                                   target_script=script, seeds=[0] * len(keys)) if keys else []
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 - any adapter failure is a provider failure
            raise ProviderError(f"word translation failed: {type(e).__name__}: {e}") from e
        lexicon = {k: o.text for k, o in zip(keys, outs, strict=True)}
        alignments = align(target, language=language, native_script=script, scripts=self._scripts,
                           units=units, word_translations=lexicon, romanize=self._romanize(language, script),
                           opts=self.opts)
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
            "english_tagger": f"{self.analyzer.name}-{self.analyzer.version}",
            "units": [{"text": u.text, "pos": u.pos, "lemma": u.lemma, "indices": list(u.indices)} for u in units],
            "n_units": len(units),
            "n_aligned": len(alignments),
            "n_swapped": k,
            "alignments": [asdict(a) for a in alignments],
            "swapped": [{"en": a.replacement, "native": a.tgt_text, "pos": a.pos, "method": a.method,
                         "construction": a.construction, "clitic_kept": a.clitic} for a in alignments[:k]],
            "word_translations": lexicon,
            "aligner_model": self.translator.info.model,
        })


def build_code_mixer(settings: Settings, translator: TranslationProvider, romanizer: Transliterator | None,
                     name: str | None = None, analyzer: EnglishAnalyzer | None = None) -> CodeMixer:
    factories = {
        ("code_mixing", "mt_lexical_swap"):
            lambda n, c: LexicalSwapCodeMixer.from_config(n, c, settings, translator, romanizer, analyzer),
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
