"""Colloquial romanisation (native script -> Latn) for hi / mr / gu.

Chosen over strict schemes (ISO 15919, ITRANS) after comparing them on pilot
outputs (docs/phase2b_3_notes.md): the benchmark needs romanised text that
looks like what users type ("kaun si nadi"), not scholarly transliteration
("kauna sī nadī"). AI4Bharat IndicXlit was the preferred neural option but
cannot be installed here (it depends on fairseq, which has no Windows /
Python 3.13 build).

Method (deterministic, rule-based, via Aksharamukha):

    Gujarati text ─► Devanagari (1:1 script mapping)        [gu only]
    word-final anusvara / candrabindu ─► न्                  [final_nasal_as_n]
    Devanagari ─► Aksharamukha "RomanColloquial", pre-option RemoveSchwaHindi

Known limits: one spelling per word (real users vary: "hai" / "he", "kya" /
"kyaa"); Hindi schwa-deletion rules are applied to Marathi and Gujarati too,
which is mostly right but not always; long vowels are not marked, so a few
words become ambiguous. Native-speaker review checks naturalness.
"""

from __future__ import annotations

import importlib.metadata
import re

from backend.config import ProviderConfig
from generator.transformation_engine import ProviderError, ProviderInfo, ProviderOutput, ProviderUnavailableError
from generator.transliteration import Transliterator

ADAPTER_VERSION = "1.0"
SCHEME = "RomanColloquial"
_AKSHARAMUKHA_SCRIPT = {"Deva": "Devanagari", "Gujr": "Gujarati"}
# anusvara (U+0902) or candrabindu (U+0901) closing a Devanagari word
_FINAL_NASAL = re.compile(r"[ँं](?![ऀ-ॣॱ-ॿ])")
_OPTIONS = ("schwa_deletion", "final_nasal_as_n")


class ColloquialRomanizer(Transliterator):
    def __init__(self, name: str, targets: list[str], *, schwa_deletion: bool = True,
                 final_nasal_as_n: bool = True):
        try:
            from aksharamukha import transliterate
        except ImportError as e:
            raise ProviderUnavailableError(f"colloquial romanisation needs aksharamukha: {e}") from e
        self._ak = transliterate
        self.targets = set(targets)
        self.schwa_deletion = schwa_deletion
        self.final_nasal_as_n = final_nasal_as_n
        self.ak_version = importlib.metadata.version("aksharamukha")
        self._info = ProviderInfo(
            name=name,
            version=(f"{ADAPTER_VERSION}+aksharamukha-{self.ak_version}"
                     f"+schwa{int(schwa_deletion)}+nasal{int(final_nasal_as_n)}"),
            model=f"aksharamukha/{SCHEME}",
            generation_method="rule",
        )

    @classmethod
    def from_config(cls, name: str, cfg: ProviderConfig) -> "ColloquialRomanizer":
        unknown = set(cfg.options) - set(_OPTIONS)
        if unknown:
            raise ProviderUnavailableError(f"{name}: unknown option(s) {sorted(unknown)}")
        return cls(name, cfg.target_languages, **cfg.options)

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, language: str, source_script: str, target_script: str) -> bool:
        return language in self.targets and source_script in _AKSHARAMUKHA_SCRIPT and target_script == "Latn"

    def transliterate(self, text, *, language, source_script, target_script, seed):
        if not self.supports(language, source_script, target_script):
            raise ProviderError(f"{self._info.name} does not support {language} {source_script}->{target_script}")
        try:
            deva = text
            if source_script != "Deva":
                deva = self._ak.process(_AKSHARAMUKHA_SCRIPT[source_script], "Devanagari", text)
            if self.final_nasal_as_n:
                deva = _FINAL_NASAL.sub("न्", deva)
            pre = ["RemoveSchwaHindi"] if self.schwa_deletion else []
            out = self._ak.process("Devanagari", SCHEME, deva, pre_options=pre)
        except Exception as e:  # noqa: BLE001 - library failure is a provider failure
            raise ProviderError(f"aksharamukha failed: {type(e).__name__}: {e}") from e
        return ProviderOutput(out, {
            "scheme": SCHEME,
            "pivot_script": "Deva" if source_script != "Deva" else None,
            "schwa_deletion": self.schwa_deletion,
            "final_nasal_as_n": self.final_nasal_as_n,
            "aksharamukha_version": self.ak_version,
        })
