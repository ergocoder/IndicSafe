"""Concrete adapters, registered with the provider registry on import.

Import this module before `build_translation_provider` / `build_transliterator`
when the configured (real) providers should be used. Tests that use fakes do
not need it.
"""

from __future__ import annotations

from generator.indictrans2 import IndicTrans2Translator
from generator.romanization import ColloquialRomanizer
from generator.transformation_engine import register_provider

register_provider("translation", "indictrans2", IndicTrans2Translator.from_config)
register_provider("transliteration", "colloquial_roman", ColloquialRomanizer.from_config)
