"""Phase 2b adapters: IndicTrans2 (fake backend, no model needed), colloquial romaniser, registry."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import generator.providers  # noqa: F401  (registers the real adapters)
from backend.config import ProviderConfig
from generator.indictrans2 import IndicTrans2Options, IndicTrans2Translator
from generator.romanization import ColloquialRomanizer
from generator.transformation_engine import (
    _PROVIDER_FACTORIES,
    ProviderError,
    ProviderUnavailableError,
    TransformationEngine,
    build_provider,
)
from generator.translation import TranslationTransformation
from generator.transliteration import TransliterationTransformation, build_transliterator
from tests.fakes import EN, TRANSLATIONS
from tests.test_transformation_engine import make_seed

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
FLORES_TO_ISO = {"hin_Deva": "hi", "mar_Deva": "mr", "guj_Gujr": "gu"}


class FakeBackend:
    """Stands in for the HF model: maps preprocessed 'eng_Latn <tgt> <text>' to a fixed output."""

    revision = "0123456789abcdef"
    device = "cpu"
    dtype = "float32"
    versions = {"torch": "fake"}

    def __init__(self, oom_above: int | None = None, fail: bool = False, table=None):
        self.oom_above = oom_above
        self.fail = fail
        self.table = table or {}
        self.batches: list[list[str]] = []

    def generate(self, batch):
        self.batches.append(list(batch))
        if self.fail:
            raise RuntimeError("boom")
        if self.oom_above is not None and len(batch) > self.oom_above:
            raise MemoryError("CUDA out of memory (fake)")
        out = []
        for line in batch:
            _, tgt, text = line.split(" ", 2)
            # IndicTrans2 emits Devanagari for every Indic target; the processor maps it back.
            hyp = self.table.get(text, TRANSLATIONS.get((FLORES_TO_ISO[tgt], EN), "अनुवाद"))
            if tgt == "guj_Gujr" and text not in self.table:
                hyp = "वाराणसी शहेरमांथी कई नदी वहे छे ?"
            out.append((hyp, 12))
        return out


def make_mt(backend=None, **opts) -> IndicTrans2Translator:
    cfg = ProviderConfig(type="local_mt", enabled=True, target_languages=["hi", "mr", "gu"],
                         model="ai4bharat/indictrans2-en-indic-dist-200M", options=opts)
    return IndicTrans2Translator.from_config("indictrans2", cfg, backend=backend or FakeBackend())


def _tr(mt, text, lang="hi"):
    script = {"hi": "Deva", "mr": "Deva", "gu": "Gujr"}[lang]
    return mt.translate(text, source_language="en", target_language=lang, target_script=script, seed=1)


# ---------------------------------------------------------------- IndicTrans2


def test_indictrans2_info_and_metadata_record_model_version_and_params():
    mt = make_mt(batch_size=2, num_beams=4, max_new_tokens=64)
    assert mt.info.name == "indictrans2" and mt.info.generation_method == "mt"
    assert mt.info.model == "ai4bharat/indictrans2-en-indic-dist-200M@0123456789abcdef"
    assert mt.info.version == "1.0+float32+beam4+max64"
    out = _tr(mt, EN)
    assert out.text == TRANSLATIONS[("hi", EN)]
    md = out.metadata
    assert md["model_name"] == "ai4bharat/indictrans2-en-indic-dist-200M"
    assert md["model_revision"] == "0123456789abcdef"
    assert (md["src_lang"], md["tgt_lang"]) == ("eng_Latn", "hin_Deva")
    assert (md["num_beams"], md["max_new_tokens"], md["batch_size"]) == (4, 64, 2)
    assert md["device"] == "cpu" and md["dtype"] == "float32" and md["hit_max_new_tokens"] is False
    assert "IndicProcessor" in md["preprocessor"]


def test_indictrans2_postprocessing_maps_script_and_restores_placeholders():
    backend = FakeBackend(table={"Write to < ID1 > today": "आज <ID1> पर लिखें ।"})
    mt = make_mt(backend)
    assert _tr(mt, "Write to test@example.com today").text == "आज test@example.com पर लिखें।"
    # the model writes Devanagari for Gujarati too; IndicProcessor converts it back
    assert _tr(mt, EN, "gu").text.startswith("વારાણસી")


def test_indictrans2_batches_and_caches():
    backend = FakeBackend()
    mt = make_mt(backend, batch_size=2)
    texts = [f"sentence {i}" for i in range(5)] + ["sentence 0"]
    outs = mt.translate_batch(texts, source_language="en", target_language="mr",
                              target_script="Deva", seeds=[0] * 6)
    assert [len(b) for b in backend.batches] == [2, 2, 1]        # duplicates translated once
    assert len(outs) == 6 and outs[0] == outs[5]
    _tr(mt, "sentence 3", "mr")
    assert len(backend.batches) == 3                               # served from cache


def test_indictrans2_halves_batch_on_oom_then_gives_up_at_one():
    backend = FakeBackend(oom_above=1)
    mt = make_mt(backend, batch_size=4)
    outs = mt.translate_batch(["a b", "c d", "e f"], source_language="en", target_language="hi",
                              target_script="Deva", seeds=[0, 0, 0])
    assert len(outs) == 3 and [len(b) for b in backend.batches] == [3, 2, 1, 1, 1]
    with pytest.raises(ProviderError, match="out of memory at batch size 1"):
        _tr(make_mt(FakeBackend(oom_above=0), batch_size=2), EN)


def test_indictrans2_backend_failure_is_provider_error_and_recorded(settings):
    mt = make_mt(FakeBackend(fail=True))
    with pytest.raises(ProviderError, match="generation failed"):
        _tr(mt, EN)
    engine = TransformationEngine(settings, run_id="T", clock=lambda: FIXED)
    root = engine.root(make_seed()).variant
    res = engine.apply(root, TranslationTransformation(mt), {"target_language": "hi"})
    assert res.transformation.status == "ERROR" and "generation failed" in res.transformation.error
    # placeholder queue was drained: a working translator after a failure is unaffected
    mt.backend.fail = False
    assert _tr(mt, EN).text == TRANSLATIONS[("hi", EN)]


def test_indictrans2_through_engine_records_provider_details(settings):
    engine = TransformationEngine(settings, run_id="T", clock=lambda: FIXED)
    root = engine.root(make_seed()).variant
    res = engine.apply(root, TranslationTransformation(make_mt()), {"target_language": "mr"})
    t = res.transformation
    assert res.ok and res.variant.language == "mr" and res.variant.script == "Deva"
    assert t.provider == "indictrans2" and t.provider_version == "1.0+float32+beam5+max256"
    assert t.generator_model.endswith("@0123456789abcdef") and t.generation_method == "mt"
    assert t.provider_metadata["tgt_lang"] == "mar_Deva"


@pytest.mark.parametrize("opts,match", [
    ({"beam": 3}, "unknown option"),
    ({"device": "tpu"}, "device must be"),
    ({"batch_size": 0}, "batch_size"),
])
def test_indictrans2_option_validation(opts, match):
    with pytest.raises(ProviderUnavailableError, match=match):
        IndicTrans2Options.from_mapping(opts)


def test_indictrans2_supports_only_configured_en_to_indic():
    mt = make_mt()
    assert mt.supports("en", "gu") and not mt.supports("hi", "en") and not mt.supports("en", "or")


# ---------------------------------------------------------------- romaniser


@pytest.mark.parametrize("lang,script,text,expected", [
    ("hi", "Deva", "वाराणसी शहर से कौन सी नदी बहती है? मैं हूँ।",
     "varansi shahar se kaun si nadi bahti hai? main hun."),
    ("mr", "Deva", "वाराणसी शहरातून कोणती नदी वाहते?", "varansi shaharatun konti nadi vahte?"),
    ("gu", "Gujr", "વારાણસી શહેરમાંથી કઈ નદી વહે છે?", "varansi shahermanthi kai nadi vahe chhe?"),
])
def test_colloquial_romanizer_outputs(lang, script, text, expected):
    r = ColloquialRomanizer("colloquial_roman", ["hi", "mr", "gu"])
    out = r.transliterate(text, language=lang, source_script=script, target_script="Latn", seed=0)
    assert out.text == expected
    assert out.metadata["pivot_script"] == (None if script == "Deva" else "Deva")


def test_colloquial_romanizer_options_change_output_and_version():
    plain = ColloquialRomanizer("x", ["hi"], schwa_deletion=False, final_nasal_as_n=False)
    out = plain.transliterate("कौन हूँ", language="hi", source_script="Deva", target_script="Latn", seed=0)
    assert out.text == "kauna hum"
    assert plain.info.version.endswith("+schwa0+nasal0") and plain.info.generation_method == "rule"


def test_romanizer_from_config_and_engine(settings):
    tl = build_transliterator(settings)                    # real config default: colloquial_roman
    assert isinstance(tl, ColloquialRomanizer)
    assert not tl.supports("hi", "Latn", "Deva") and not tl.supports("en", "Deva", "Latn")
    engine = TransformationEngine(settings, run_id="T", clock=lambda: FIXED)
    root = engine.root(make_seed()).variant
    hi = engine.apply(root, TranslationTransformation(make_mt()), {"target_language": "hi"}).variant
    res = engine.apply(hi, TransliterationTransformation(tl))
    assert res.ok and res.variant.script == "Latn" and res.variant.is_transliterated
    assert res.transformation.parameters == {"language": "hi", "source_script": "Deva", "target_script": "Latn"}
    with pytest.raises(ProviderUnavailableError, match="unknown option"):
        ColloquialRomanizer.from_config("x", ProviderConfig(type="rule_based", enabled=True,
                                                            options={"scheme": "iso"}))


def test_real_adapters_are_registered(settings):
    assert _PROVIDER_FACTORIES[("translation", "indictrans2")] == IndicTrans2Translator.from_config
    # without the registry entry the configured default is refused, not faked
    with pytest.raises(ProviderUnavailableError, match="no adapter"):
        build_provider(settings, "translation", factories={})
