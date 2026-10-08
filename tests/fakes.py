"""Deterministic test doubles for the transformation adapters.

They look outputs up in small hand-written tables (benign sentences only) and
record every call, so tests can check what the engine passed to a provider.
They report generation_method "mock", so their output can never be mistaken
for real MT / transliteration output.
"""

from __future__ import annotations

from generator.english_pos import EnglishAnalyzer, EnUnit
from generator.paraphrase import ParaphraseProvider
from generator.transformation_engine import ProviderError, ProviderInfo, ProviderOutput
from generator.translation import TranslationProvider
from generator.transliteration import Transliterator

EN = "Which river flows through the city of Varanasi?"

TRANSLATIONS = {
    ("hi", EN): "वाराणसी शहर से कौन सी नदी बहती है?",
    ("mr", EN): "वाराणसी शहरातून कोणती नदी वाहते?",
    ("gu", EN): "વારાણસી શહેરમાંથી કઈ નદી વહે છે?",
}

ROMANIZED = {
    TRANSLATIONS[("hi", EN)]: "Varanasi shahar se kaun si nadi behti hai?",
    TRANSLATIONS[("mr", EN)]: "Varanasi shaharatun konti nadi vahate?",
    TRANSLATIONS[("gu", EN)]: "Varanasi shaherma thi kai nadi vahe chhe?",
}

PARAPHRASES = {
    (EN, 0): "What river runs through Varanasi?",
    (EN, 1): "Through the city of Varanasi, which river flows?",
}


class FakeTranslator(TranslationProvider):
    def __init__(self, table=None, *, name="fake_mt", version="1.0", targets=("hi", "mr", "gu"),
                 fail=False, metadata=None):
        self.table = TRANSLATIONS if table is None else table
        self._info = ProviderInfo(name, version, f"{name}-model", "mock")
        self.targets = set(targets)
        self.fail = fail
        self.metadata = metadata
        self.calls: list[dict] = []

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, source_language, target_language):
        return source_language == "en" and target_language in self.targets

    def translate(self, text, *, source_language, target_language, target_script, seed):
        self.calls.append(dict(text=text, source_language=source_language,
                               target_language=target_language, target_script=target_script, seed=seed))
        if self.fail:
            raise ProviderError("simulated MT backend failure")
        out = self.table.get((target_language, text))
        if out is None:
            raise ProviderError(f"no fixture translation for {target_language}: {text!r}")
        meta = self.metadata if self.metadata is not None else {"beam_size": 1}
        return ProviderOutput(out, meta)


class FakeTransliterator(Transliterator):
    def __init__(self, table=None, *, name="fake_translit", version="1.0"):
        self.table = ROMANIZED if table is None else table
        self._info = ProviderInfo(name, version, "fake-scheme", "mock")
        self.calls: list[dict] = []

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, language, source_script, target_script):
        return target_script == "Latn" and source_script in ("Deva", "Gujr")

    def transliterate(self, text, *, language, source_script, target_script, seed):
        self.calls.append(dict(text=text, language=language, source_script=source_script,
                               target_script=target_script, seed=seed))
        if text not in self.table:
            raise ProviderError(f"no fixture romanisation for {text!r}")
        return ProviderOutput(self.table[text], {"scheme": "fake"})


class FakeParaphraser(ParaphraseProvider):
    def __init__(self, table=None, *, name="fake_para", version="1.0"):
        self.table = PARAPHRASES if table is None else table
        self._info = ProviderInfo(name, version, "fake-para-model", "mock")
        self.calls: list[dict] = []

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def paraphrase(self, text, *, language, script, variant_index, seed):
        self.calls.append(dict(text=text, language=language, script=script,
                               variant_index=variant_index, seed=seed))
        if (text, variant_index) not in self.table:
            raise ProviderError("no fixture paraphrase")
        return ProviderOutput(self.table[(text, variant_index)])


class FakeAnalyzer(EnglishAnalyzer):
    """English units from a hand-written table: source -> [(indices, words, pos, lemma)]."""

    name, version = "fake_pos", "1"

    def __init__(self, table):
        self.table = {src: [EnUnit(tuple(i), tuple(w), pos, lemma) for i, w, pos, lemma in units]
                      for src, units in table.items()}

    def units(self, source, *, stopwords, never_swap, min_chars, swap_pos):
        return [u for u in self.table.get(source, []) if u.pos in swap_pos]
