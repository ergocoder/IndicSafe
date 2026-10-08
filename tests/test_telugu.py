"""Telugu (te): 4th target language, translation + romanisation + QC only (no code-mixing). No model needed."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from backend.config import ProviderConfig
from generator.code_mixing import build_code_mixer
from generator.indictrans2 import FLORES, IndicTrans2Translator
from generator.language_qc import check_variant
from generator.pilot_translation import run_pilot_translation, write_outputs
from generator.qc_pipeline import run_qc_on_dir
from generator.romanization import ColloquialRomanizer, telugu_colloquial
from generator.text_utils import dominant_script
from generator.transformation_engine import TransformationEngine
from generator.translation import TranslationTransformation
from tests.fakes import EN, TRANSLATIONS, FakeTranslator, FakeTransliterator
from tests.test_code_mix import ANALYZER, ROMAN, WORDS
from tests.test_language_qc import FakeLID
from tests.test_qc_pipeline import FakeEncoder
from tests.test_transformation_engine import make_seed

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
TE = "వారణాసి నగరం గుండా ఏ నది ప్రవహిస్తుంది?"
TE_LATN = "varanasi nagaram gunda e nadi pravahistundi?"


class TeluguLID(FakeLID):
    """FakeLID that also recognises Telugu script."""

    def scores(self, text):
        if any("ఀ" <= c <= "౿" for c in text):
            return {"te": 1.0, "hi": 0.0, "mr": 0.0, "gu": 0.0, "en": 0.0}
        return {**super().scores(text), "te": 0.0}


def scripts(settings):
    return {k: v.ranges for k, v in settings.languages.scripts.items()}


# ----------------------------------------------------------------- config


def test_telugu_config(settings):
    te = settings.languages.languages["te"]
    assert (te.native_script, te.romanized_script, te.enabled, te.code_mix_levels) == ("Telu", "Latn", True, ["L0"])
    assert "suffix" in te.notes and settings.languages.scripts["Telu"].ranges == [(0x0C00, 0x0C7F)]
    gen = settings.generation
    assert "te" in gen.translation.providers["indictrans2"].target_languages
    assert "te" in gen.transliteration.providers["colloquial_roman"].target_languages
    assert "te" not in gen.code_mixing.providers["mt_lexical_swap"].target_languages
    assert "te" in gen.qc.language.candidates and FLORES["te"] == "tel_Telu"


def test_script_check_separates_telugu_from_hi_mr_gu(settings):
    s = scripts(settings)
    assert dominant_script(TE, s) == ("Telu", 1.0)
    for text, script in [(TRANSLATIONS[("hi", EN)], "Deva"), (TRANSLATIONS[("mr", EN)], "Deva"),
                         (TRANSLATIONS[("gu", EN)], "Gujr")]:
        assert dominant_script(text, s)[0] == script != "Telu"


# ------------------------------------------------------------- translation


class DevanagariBackend:
    """IndicTrans2 writes Devanagari for every Indic target; IndicProcessor maps it to the target script."""

    revision, device, dtype, versions = "r", "cpu", "float32", {}

    def __init__(self):
        self.batches = []

    def generate(self, batch):
        self.batches.append(list(batch))
        return [("वारणासि नगरं गुंडा ए नदि प्रवहिस्तुंदि ?", 9) for _ in batch]


def test_indictrans2_translates_to_telugu_script():
    cfg = ProviderConfig(type="local_mt", enabled=True, target_languages=["hi", "mr", "gu", "te"],
                         model="ai4bharat/indictrans2-en-indic-dist-200M", options={"batch_size": 2})
    backend = DevanagariBackend()
    mt = IndicTrans2Translator.from_config("indictrans2", cfg, backend=backend)
    assert mt.supports("en", "te")
    outs = mt.translate_batch([EN, "Who wrote it?", EN], source_language="en", target_language="te",
                              target_script="Telu", seeds=[0, 0, 0])
    assert outs[0].text == TE and outs[0].metadata["tgt_lang"] == "tel_Telu"
    assert backend.batches[0][0].startswith("eng_Latn tel_Telu ") and sum(map(len, backend.batches)) == 2  # cached


# ------------------------------------------------------------- romanisation


@pytest.mark.parametrize("text,expected", [
    ("మీరు ఎలా ఉన్నారు?", "miru ela unnaru?"),
    ("నేను పుస్తకం చదువుతున్నాను", "nenu pustakam chaduvutunnanu"),
    ("కృష్ణుడు జ్ఞానం గురించి చెప్పాడు", "krishnudu gnanam gurinchi cheppadu"),
    ("భారతదేశంలో ఏ నది పొడవైనది?", "bharatadeshamlo e nadi podavainadi?"),
    ("ఛత్రపతి శివాజీ ఏ సంవత్సరంలో జన్మించాడు?", "chhatrapati shivaji e samvatsaramlo janminchadu?"),
])
def test_telugu_colloquial_romanisation(text, expected):
    r = ColloquialRomanizer("colloquial_roman", ["te"])
    out = r.transliterate(text, language="te", source_script="Telu", target_script="Latn", seed=0)
    assert out.text == expected and out.text.isascii()
    assert out.metadata["scheme"] == "ISO+telugu_colloquial" and out.metadata["schwa_deletion"] is False
    assert out.metadata["pivot_script"] is None


def test_telugu_rules_on_iso_and_latin_pass_through():
    # long vowels collapse, retroflexes lose their dots, anusvara is n before stops and m elsewhere
    assert telugu_colloquial("nēnu bāṁk lōpala kūrcunnānu") == "nenu bank lopala kurchunnanu"
    assert telugu_colloquial("saṁsāraṁ") == "samsaram"
    r = ColloquialRomanizer("colloquial_roman", ["te"])
    out = r.transliterate("Lok Sabha యొక్క సంఖ్య ఎంత?", language="te", source_script="Telu",
                          target_script="Latn", seed=0).text
    assert out == "Lok Sabha yokka sankhya enta?"


def test_hindi_romanisation_unchanged_by_telugu_path():
    r = ColloquialRomanizer("colloquial_roman", ["hi", "te"])
    out = r.transliterate("वाराणसी शहर से कौन सी नदी बहती है?", language="hi", source_script="Deva",
                          target_script="Latn", seed=0)
    assert out.text == "varansi shahar se kaun si nadi bahti hai?" and out.metadata["schwa_deletion"] is True


# --------------------------------------------------------------------- QC


def test_language_qc_for_telugu(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    root = engine.root(make_seed()).variant
    te = engine.apply(root, TranslationTransformation(FakeTranslator({("te", EN): TE}, targets=("te",))),
                      {"target_language": "te"}).variant
    assert te.validation_status == "PASS" and te.script == "Telu"
    q = check_variant(settings, te, TeluguLID())
    assert (q.language_qc_status, q.lid_language, q.script_status) == ("PASS", "te", "PASS")
    wrong = te.model_copy(update={"prompt": TRANSLATIONS[("hi", EN)]})       # Hindi text labelled te
    assert check_variant(settings, wrong, TeluguLID()).script_reason == "script_mismatch"


def test_real_lingua_identifies_telugu(settings):
    from generator.language_qc import build_language_identifier
    lid = build_language_identifier(settings)
    scores = lid.scores(TE)
    assert max(scores, key=scores.get) == "te"


def test_pipeline_and_qc_with_telugu_as_fourth_language(settings, project):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    mt = FakeTranslator({**TRANSLATIONS, **WORDS, ("te", EN): TE}, targets=("hi", "mr", "gu", "te"))
    tl = FakeTransliterator({**ROMAN, TE: TE_LATN})
    res = run_pilot_translation(settings, [make_seed()], mt, tl, TeluguLID(), ["hi", "te"], engine=engine,
                                code_mixer=build_code_mixer(settings, mt, None, analyzer=ANALYZER))
    sid = "S-D10K-020000000401"
    assert res.native[(sid, "te")].prompt == TE and res.latin[(sid, "te")].prompt == TE_LATN
    assert {k[1] for k in res.code_mixed} == {"hi"}                     # te: no code-mix attempts
    paths = write_outputs(res, settings, project / "out")
    with paths["review_te"].open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["native_text"] == TE and rows[0]["latin_text"] == TE_LATN
    assert rows[0]["native_qc"].startswith("PASS script=Telu")
    out = run_qc_on_dir(settings, paths["manifest"].parent, FakeEncoder())
    s = json.loads(out["summary"].read_text(encoding="utf-8"))
    assert s["code_mix_scope"] == {"code_mixed": ["hi"], "not_code_mixed": ["te"]}
    assert all(set(c.get("by_language", {})) == {"hi"} for c in s["code_mix_coverage"].values())
    assert s["qc_status_by_kind"]["te/Telu/L0"] == {"PASS": 1} and s["qc_status_by_kind"]["te/Latn/L0"] == {"PASS": 1}
    assert "review_codemix_te" not in out and "review_codemix_hi" in out
    rec = next(json.loads(x) for x in out["report"].read_text(encoding="utf-8").splitlines()
               if json.loads(x)["language"] == "te" and json.loads(x)["script"] == "Latn")
    assert rec["code_mix_ratio"] == 0.0 and rec["checks"]["code_mix"]["status"] == "PASS"
