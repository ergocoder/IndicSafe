"""Phase 4 code-mixing: measurement, alignment, swap selection, engine integration. No model needed."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from generator import code_mix_metrics as cmm
from generator.code_mixing import (
    CodeMixingTransformation,
    LexicalSwapCodeMixer,
    LexicalSwapOptions,
    align,
    apply_swaps,
    build_code_mixer,
    choose_k,
    english_form,
)
from generator.pilot_translation import load_pilot_seeds, run_pilot_translation
from generator.transformation_engine import TransformationEngine, TransformationError
from generator.translation import TranslationTransformation
from generator.transliteration import TransliterationTransformation
from tests.fakes import EN, TRANSLATIONS, FakeTranslator, FakeTransliterator
from tests.test_language_qc import FakeLID
from tests.test_transformation_engine import make_seed

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]
HI = TRANSLATIONS[("hi", EN)]                       # वाराणसी शहर से कौन सी नदी बहती है?
WORDS = {("hi", "river"): "नदी", ("hi", "flows"): "बहती", ("hi", "city"): "शहर", ("hi", "Varanasi"): "वाराणसी"}
HI_L1 = "वाराणसी शहर से कौन सी river बहती है?"
HI_L2 = "वाराणसी शहर से कौन सी river flows है?"
ROMAN = {HI: "Varanasi shahar se kaun si nadi behti hai?",
         HI_L1: "Varanasi shahar se kaun si river behti hai?",
         HI_L2: "Varanasi shahar se kaun si river flows hai?"}


def scripts(settings):
    return {k: v.ranges for k, v in settings.languages.scripts.items()}


def opts(settings, **over):
    o = LexicalSwapOptions.from_mapping(settings.generation.code_mixing.providers["mt_lexical_swap"].options)
    return LexicalSwapOptions(**{**o.__dict__, **over})


# ------------------------------------------------------------ measurement


def test_native_tags_ratio_and_cmi(settings):
    tags = cmm.tag_native("हम phishing हमलों का use, 2024 में!", "Deva", "Latn", scripts(settings))
    assert [t for _, t in tags] == ["primary", "secondary", "primary", "primary", "secondary", "other", "primary"]
    m = cmm.summarize(tags, "script_tags")
    assert (m.n_primary, m.n_secondary, m.n_other) == (4, 2, 1)
    assert m.ratio == round(2 / 6, 4) and m.cmi == round(100 * (1 - 4 / 6), 2)
    assert m.secondary_tokens == ("phishing", "use")
    assert cmm.summarize(cmm.tag_native("123 ?", "Deva", "Latn", scripts(settings)), "x").ratio is None


def test_romanised_tags_come_from_native_parent_and_break_on_misalignment(settings):
    s = scripts(settings)
    m = cmm.measure(ROMAN[HI_L1], native_script="Deva", partner_script="Latn", scripts=s,
                    is_transliterated=True, parent_text=HI_L1)
    assert m.method == "aligned_to_native_parent" and m.secondary_tokens == ("river",) and m.ratio == 0.125
    # all-Latin text: letters alone would call every word English
    assert cmm.measure("Varanasi shahar se kaun si river behti", native_script="Deva", partner_script="Latn",
                       scripts=s, is_transliterated=True, parent_text=HI_L1) is None      # token count differs
    assert cmm.measure(ROMAN[HI_L1].replace("river", "rivr"), native_script="Deva", partner_script="Latn",
                       scripts=s, is_transliterated=True, parent_text=HI_L1) is None      # English word altered


@pytest.mark.parametrize("ratio,status,reason", [
    (0.10, "PASS", None), (0.05, "PASS", None), (0.20, "WARN", "code_mix_near_band_edge"),
    (0.23, "WARN", "code_mix_near_band_edge"), (0.24, "FAIL", "code_mix_out_of_band"),
    (0.0, "FAIL", "code_mix_out_of_band"), (None, "FAIL", "code_mix_unmeasurable"),
])
def test_band_check(settings, ratio, status, reason):
    st, rs, details = cmm.band_check(ratio, "L1", cmm.level_bands(settings), 0.03)
    assert (st, rs) == (status, reason) and details["band"] == [0.05, 0.20]


def test_level_for_ratio_reports_without_relabelling(settings):
    b = cmm.level_bands(settings)
    assert [cmm.level_for_ratio(r, b) for r in (0.0, 0.05, 0.2, 0.34, 0.5, 0.6)] == ["L0", "L1", "L2", "L2", "L3", None]


# -------------------------------------------------------------- alignment


def test_lexical_alignment_and_nested_swaps(settings):
    lex = {w: t for (_, w), t in WORDS.items()}
    al = align(EN, HI, language="hi", native_script="Deva", scripts=scripts(settings),
               word_translations=lex, romanize=None, opts=opts(settings))
    assert [(a.src_word, a.tgt_word, a.method) for a in al] == [
        ("river", "नदी", "lexical"), ("flows", "बहती", "lexical"), ("city", "शहर", "lexical"),
        ("Varanasi", "वाराणसी", "lexical")]
    assert apply_swaps(HI, al[:1]) == HI_L1 and apply_swaps(HI, al[:2]) == HI_L2


def test_prefix_match_keeps_marathi_case_marker_as_own_token(settings):
    al = align("Money from the bank", "बँकेच्या खात्यातून पैसे", language="mr", native_script="Deva",
               scripts=scripts(settings), word_translations={"bank": "बँक", "Money": "पैसा"},
               romanize=None, opts=opts(settings))
    bank = next(a for a in al if a.src_word == "bank")
    assert bank.tgt_word == "बँकेच्या" and bank.clitic == "च्या"
    assert apply_swaps("बँकेच्या खात्यातून पैसे", [bank]) == "bank च्या खात्यातून पैसे"


def test_phonetic_loanword_match_and_short_word_guard(settings):
    rom = {"डेटा": "deta", "से": "se", "चोरी": "chori"}
    al = align("Steal data from use", "से डेटा चोरी", language="hi", native_script="Deva",
               scripts=scripts(settings), word_translations={}, romanize=lambda w: rom[w], opts=opts(settings))
    assert [(a.src_word, a.tgt_word, a.method) for a in al] == [("data", "डेटा", "phonetic")]  # not use~से


def test_latin_identical_loanwords_are_swapped_last(settings):
    rom = {"फिशिंग": "phishing", "डेटा": "deta"}
    al = align("phishing data", "फिशिंग डेटा", language="hi", native_script="Deva", scripts=scripts(settings),
               word_translations={}, romanize=lambda w: rom[w], opts=opts(settings))
    assert [a.src_word for a in al] == ["data", "phishing"]


def test_choose_k_targets_band_and_reports_best_miss(settings):
    s = scripts(settings)
    lex = {w: t for (_, w), t in WORDS.items()}
    al = align(EN, HI, language="hi", native_script="Deva", scripts=s, word_translations=lex,
               romanize=None, opts=opts(settings))

    def ratio(t):
        return cmm.summarize(cmm.tag_native(t, "Deva", "Latn", s), "x").ratio
    assert choose_k(HI, al, (0.05, 0.20), 0.5, ratio)[:2] == (1, HI_L1)
    assert choose_k(HI, al, (0.20, 0.35), 0.5, ratio)[:2] == (2, HI_L2)
    k, text, r = choose_k(HI, al[:1], (0.20, 0.35), 0.5, ratio)      # cannot reach L2
    assert (k, r) == (1, 0.125)


def test_english_form():
    assert english_form("Phishing", 0) == "phishing"
    assert [english_form(w, 3) for w in ("SIT", "iPhone", "Varanasi", "data")] == ["SIT", "iPhone", "Varanasi", "data"]


# ----------------------------------------------------------------- engine


@pytest.fixture
def mixer(settings):
    return build_code_mixer(settings, FakeTranslator({**TRANSLATIONS, **WORDS}), None)


def _native(engine):
    root = engine.root(make_seed()).variant
    return root, engine.apply(root, TranslationTransformation(FakeTranslator()), {"target_language": "hi"}).variant


def test_code_mixed_variants_keep_lineage_label_and_measured_ratio(settings, mixer):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    root, hi = _native(engine)
    cm = CodeMixingTransformation(mixer, engine.variants)
    res = engine.apply(hi, cm, {"level": "L1"})
    v, t = res.variant, res.transformation
    assert res.ok and v.prompt == HI_L1
    assert (v.seed_id, v.parent_prompt_id, v.lineage) == (root.seed_id, hi.prompt_id, [root.prompt_id, hi.prompt_id])
    assert v.intended_label == root.intended_label and v.category == root.category
    assert (v.secondary_language, v.code_mix_level, v.code_mix_ratio, v.cmi) == ("en", "L1", 0.125, 12.5)
    assert v.mixing_method == "mt_lexical_swap" and v.generation_method == "rule"
    assert t.parameters["source_prompt_id"] == root.prompt_id and t.parameters["band"] == [0.05, 0.2]
    assert t.provider_metadata["swapped"] == [{"en": "river", "native": "नदी", "method": "lexical", "clitic_kept": None}]
    band = next(r for r in t.validation_results if r.hook == "code_mix_band")
    script = next(r for r in t.validation_results if r.hook == "expected_script")
    assert band.status == "PASS" and script.details["counts_partner_script"]
    # the romanised child keeps the level and is measured through its parent
    lat = engine.apply(v, TransliterationTransformation(FakeTransliterator(ROMAN))).variant
    assert (lat.script, lat.code_mix_level, lat.code_mix_ratio, lat.secondary_language) == ("Latn", "L1", 0.125, "en")
    # deterministic ids: same request -> same variant
    assert engine.apply(hi, cm, {"level": "L1"}).variant.prompt_id == v.prompt_id


def test_band_miss_is_failed_not_relabelled(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    _, hi = _native(engine)
    unrelated = {k: "कुछ" for k in WORDS}               # no word aligns
    mixer = build_code_mixer(settings, FakeTranslator({**TRANSLATIONS, **unrelated}), None)
    v = engine.apply(hi, CodeMixingTransformation(mixer, engine.variants), {"level": "L2"}).variant
    assert v.validation_status == "FAIL" and v.code_mix_level == "L2" and v.code_mix_ratio == 0.0
    assert "code_mix_band:code_mix_out_of_band" in v.validation_failures
    with pytest.raises(TransformationError, match="failed validation"):
        engine.apply(v, TransliterationTransformation(FakeTransliterator(ROMAN)))


def test_code_mixing_rejects_bad_requests(settings, mixer):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    root, hi = _native(engine)
    cm = CodeMixingTransformation(mixer, engine.variants)
    for parent, params, match in [
        (hi, {"level": "L3"}, "not configured"), (hi, {"level": "L1", "secondary_language": "hi"}, "mixes only"),
        (root, {"level": "L1"}, "disabled|mixes only|not configured"), (hi, {"level": "L1", "x": 1}, "unknown"),
    ]:
        with pytest.raises(TransformationError, match=match):
            engine.apply(parent, cm, params)


def test_mixer_config_and_version(settings):
    m = build_code_mixer(settings, FakeTranslator(), FakeTransliterator())
    assert isinstance(m, LexicalSwapCodeMixer) and m.info.name == "mt_lexical_swap"
    assert "align:fake_mt-1.0" in m.info.version and "phon:fake_translit-1.0" in m.info.version
    assert m.supports("gu", "en") and not m.supports("gu", "hi")


def test_pilot_run_with_code_mixing_fills_all_slots(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    mt = FakeTranslator({**TRANSLATIONS, **WORDS})
    res = run_pilot_translation(settings, [make_seed()], mt, FakeTransliterator({**ROMAN}), FakeLID(), ["hi"],
                                engine=engine, code_mixer=build_code_mixer(settings, mt, None))
    sid = "S-D10K-020000000401"
    got = {k[2:]: (v.prompt if v else None) for k, v in res.code_mixed.items()}
    assert got == {("L1", "Deva"): HI_L1, ("L1", "Latn"): ROMAN[HI_L1],
                   ("L2", "Deva"): HI_L2, ("L2", "Latn"): ROMAN[HI_L2]}
    assert all(k[0] == sid for k in res.code_mixed)
    assert Counter(v.transformation_type for v in engine.variants.values()) == Counter(
        {"identity": 1, "translation": 1, "transliteration": 3, "code_mixing": 2})


# --------------------------------------------------------------- v0.2 input


def test_load_pilot_v02_uses_human_final_label_as_intended_label():
    seeds, version = load_pilot_seeds(REPO / "data/pilot/pilot_seeds_v0.2-pilot-seeds.jsonl")
    assert version == "v0.2-pilot-seeds" and len(seeds) == 30
    assert Counter(s.intended_label for s in seeds) == {"UNSAFE": 16, "SAFE": 12, "AMBIGUOUS": 2}
    assert {s.label_status for s in seeds} == {"human_agreed", "human_adjudicated"}
    assert all(s.final_label is None and "Human final label of pilot v0.2" in s.intended_label_basis for s in seeds)
    v01, _ = load_pilot_seeds(REPO / "data/pilot/pilot_seeds_v0.1-pilot-seeds.jsonl")
    assert {s.label_status for s in v01} == {"provisional"}


# ------------------------------------------------- regressions from the pilot run


def test_romanizer_passes_latin_words_through_unchanged(settings):
    from generator.romanization import ColloquialRomanizer
    r = ColloquialRomanizer("colloquial_roman", ["hi", "gu"])
    out = r.transliterate("Munich ने royal wedding का celebrate मनाने के लिए Oktoberfest कब?", language="hi",
                          source_script="Deva", target_script="Latn", seed=0).text
    assert out.startswith("Munich ne royal wedding ka celebrate ") and "Oktoberfest" in out
    native = "Munich ने royal wedding का celebrate मनाने के लिए Oktoberfest कब?"
    m = cmm.measure(out, native_script="Deva", partner_script="Latn", scripts=scripts(settings),
                    is_transliterated=True, parent_text=native)
    assert m is not None and m.secondary_tokens == ("Munich", "royal", "wedding", "celebrate", "Oktoberfest")


def test_code_mixed_text_may_be_dominated_by_latin_letters(settings):
    from generator.transformation_engine import script_matches
    text = "Sabha की sanctioned संख्या कितनी है?"
    assert script_matches(settings, text, "Latn", "Deva", "en")
    assert not script_matches(settings, text, "Latn", "Deva", None)            # monolingual: still a mismatch
    assert not script_matches(settings, "only english words", "Latn", "Deva", "en")   # no native letters


def test_latin_dominated_code_mix_is_recorded_in_target_script_and_romanisable(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    _, hi = _native(engine)
    words = {("hi", "river"): "नदी", ("hi", "Varanasi"): "वाराणसी", ("hi", "city"): "शहर", ("hi", "flows"): "बहती"}
    mixer = build_code_mixer(settings, FakeTranslator({**TRANSLATIONS, **words}), None)
    text = "Varanasiiiiiiiiiii शहर riverrrrrrrrrrrrrr नदी flowsssssssss है?"
    from generator.text_utils import dominant_script
    assert dominant_script(text, scripts(settings))[0] == "Latn"

    class Fixed(type(mixer)):                   # provider returning Latin-dominated code-mix
        def mix(self, *a, **k):
            from generator.transformation_engine import ProviderOutput
            return ProviderOutput(text)
    fixed = Fixed("mt_lexical_swap", settings, ["hi"], mixer.opts, mixer.translator, None)
    w = engine.apply(hi, CodeMixingTransformation(fixed, engine.variants), {"level": "L2"}).variant
    assert w.script == "Deva" and "expected_script:script_mismatch" not in w.validation_failures
