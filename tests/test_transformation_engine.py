"""Transformation engine: lineage, determinism, validation hooks, adapters, config."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.config import ConfigError, load_settings
from generator.paraphrase import ParaphraseTransformation, build_paraphrase_provider
from generator.schemas import HookResult, SeedRecord, VariantRecord
from generator.transformation_engine import (
    LineageError,
    ProviderOutput,
    ProviderUnavailableError,
    TransformationEngine,
    TransformationError,
    build_provider,
    load_transformations_jsonl,
    load_variants_jsonl,
    trace_lineage,
    verify_lineage,
)
from generator.translation import TranslationTransformation, build_translation_provider
from generator.transliteration import TransliterationTransformation, build_transliterator
from generator.text_utils import content_hash
from tests.fakes import EN, ROMANIZED, TRANSLATIONS, FakeParaphraser, FakeTranslator, FakeTransliterator
from tests.test_config import _edit

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def make_seed(prompt: str = EN, **over) -> SeedRecord:
    fields = dict(
        seed_id="S-D10K-020000000401", seed_version=1, prompt=prompt, original_text=prompt,
        content_hash=content_hash(prompt), language="en", script="Latn", script_confidence=1.0,
        source_type="existing_dataset", source_dataset="dataset_10k", source_role="benign_control",
        source_file="dataset_10k.zip", source_member="dataset_10k.jsonl", source_file_sha256="0" * 64,
        source_reference="020000000401", source_line=5, source_category="indian_questions",
        category="benign_everyday", category_status="source_mapped", intended_label="SAFE",
        intended_label_basis="test fixture", seed_status="VALID", taxonomy_version="1.0",
        generator_version="0.1.0", import_run_id="IMPORT_TEST", imported_at="2026-10-08T00:00:00Z",
    )
    fields.update(over)
    return SeedRecord(**fields)


@pytest.fixture
def engine(settings):
    return TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)


@pytest.fixture
def root(engine):
    return engine.root(make_seed()).variant


def _chain(engine, seed=None):
    """seed -> en root -> hi Deva -> hi Latn, plus mr, gu and an en paraphrase."""
    mt, tl, pp = FakeTranslator(), FakeTransliterator(), FakeParaphraser()
    r = engine.root(seed or make_seed()).variant
    out = {"root": r}
    for lang in ("hi", "mr", "gu"):
        native = engine.apply(r, TranslationTransformation(mt), {"target_language": lang}).variant
        out[lang] = native
        out[f"{lang}_latn"] = engine.apply(native, TransliterationTransformation(tl)).variant
    out["para"] = engine.apply(r, ParaphraseTransformation(pp)).variant
    return out


# ------------------------------------------------------------------ root


def test_root_variant_copies_seed_and_starts_lineage(engine):
    seed = make_seed()
    res = engine.root(seed)
    v, t = res.variant, res.transformation
    assert res.ok and v.validation_status == "PASS"
    assert v.parent_prompt_id is None and v.lineage == []
    assert v.prompt == seed.prompt and v.content_hash == seed.content_hash
    assert v.prompt_id.startswith("P-D10K-020000000401-")
    assert (v.seed_id, v.seed_version, v.source_dataset, v.source_reference) == (
        seed.seed_id, 1, "dataset_10k", "020000000401")
    assert (v.category, v.intended_label, v.final_label) == ("benign_everyday", "SAFE", None)
    assert v.label_consistency_status == "UNCHECKED"
    assert (v.language, v.script, v.is_transliterated) == ("en", "Latn", False)
    assert t.transformation_type == "identity" and t.generation_method == "copy"
    assert t.output_prompt_id == v.prompt_id and t.transformation_id == v.transformation_id
    assert t.validation_hooks == ["non_empty", "text_integrity", "expected_script"]


@pytest.mark.parametrize("status,extra", [
    ("REJECTED", {"rejection_reasons": ["too_short"]}),
    ("DUPLICATE", {"duplicate_of": "S-NHQA-1"}),
])
def test_root_refuses_invalid_seeds(engine, status, extra):
    with pytest.raises(TransformationError, match="only VALID"):
        engine.root(make_seed(seed_status=status, **extra))


# --------------------------------------------------------------- lineage


def test_full_chain_lineage(engine):
    c = _chain(engine)
    r, hi, hil = c["root"], c["hi"], c["hi_latn"]
    assert hi.parent_prompt_id == r.prompt_id and hi.lineage == [r.prompt_id]
    assert hil.parent_prompt_id == hi.prompt_id and hil.lineage == [r.prompt_id, hi.prompt_id]
    for v in c.values():
        assert v.seed_id == r.seed_id and v.seed_version == 1
        assert (v.source_dataset, v.source_reference, v.intended_label) == ("dataset_10k", "020000000401", "SAFE")
        assert v.generation_run_id == "TRANSFORM_TEST"
    assert (hi.language, hi.script, hi.is_transliterated) == ("hi", "Deva", False)
    assert (hil.language, hil.script, hil.is_transliterated) == ("hi", "Latn", True)
    assert (c["gu"].script, c["gu_latn"].script) == ("Gujr", "Latn")
    assert (c["mr"].language, c["mr"].script) == ("mr", "Deva")
    assert c["para"].transformation_type == "paraphrase" and c["para"].language == "en"
    assert all(v.validation_status == "PASS" for v in c.values()), [v.validation_failures for v in c.values()]
    assert len(engine.variants) == 8 and verify_lineage(engine.variants, engine.transformations) == []

    steps = trace_lineage(hil.prompt_id, engine.variants, engine.transformations)
    assert [s["transformation_type"] for s in steps] == ["identity", "translation", "transliteration"]
    assert steps[1]["parameters"] == {"source_language": "en", "target_language": "hi", "target_script": "Deva"}
    assert steps[2]["parameters"] == {"language": "hi", "source_script": "Deva", "target_script": "Latn"}


def test_parent_is_never_modified(engine, root):
    before = root.model_dump()
    engine.apply(root, TranslationTransformation(FakeTranslator()), {"target_language": "hi"})
    assert root.model_dump() == before
    with pytest.raises(ValidationError):
        root.prompt = "changed"


def test_verify_lineage_detects_breaks(engine):
    c = _chain(engine)
    variants = dict(engine.variants)
    del variants[c["hi"].prompt_id]
    assert any("parent" in p and "missing" in p for p in verify_lineage(variants, engine.transformations))
    with pytest.raises(LineageError):
        trace_lineage(c["hi_latn"].prompt_id, variants, engine.transformations)
    transformations = dict(engine.transformations)
    del transformations[c["mr"].transformation_id]
    assert any(c["mr"].prompt_id in p for p in verify_lineage(engine.variants, transformations))


# ----------------------------------------------------------- determinism


def _dump(engine):
    return (
        [v.model_dump(mode="json") for v in engine.variants.values()],
        [t.model_dump(mode="json") for t in engine.transformations.values()],
    )


def test_same_inputs_give_identical_records(settings):
    runs = []
    for _ in range(2):
        e = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
        _chain(e)
        runs.append(_dump(e))
    assert runs[0] == runs[1]


def test_ids_independent_of_run_and_clock(settings):
    a = TransformationEngine(settings, run_id="TRANSFORM_A", clock=lambda: FIXED)
    b = TransformationEngine(settings, run_id="TRANSFORM_B",
                             clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    ca, cb = _chain(a), _chain(b)
    assert {k: v.prompt_id for k, v in ca.items()} == {k: v.prompt_id for k, v in cb.items()}
    assert list(a.transformations) == list(b.transformations)
    assert [t.derived_seed for t in a.transformations.values()] == [t.derived_seed for t in b.transformations.values()]
    assert ca["hi"].generation_run_id == "TRANSFORM_A" and cb["hi"].generation_run_id == "TRANSFORM_B"


def test_ids_change_with_request(engine, root):
    hi = engine.apply(root, TranslationTransformation(FakeTranslator()), {"target_language": "hi"})
    v2 = engine.apply(root, TranslationTransformation(FakeTranslator(version="2.0")), {"target_language": "hi"})
    other = engine.apply(root, TranslationTransformation(FakeTranslator(name="other_mt")), {"target_language": "hi"})
    ids = {hi.variant.prompt_id, v2.variant.prompt_id, other.variant.prompt_id}
    assert len(ids) == 3


def test_ids_depend_on_random_seed_only_via_derived_seed(project, root):
    e1 = TransformationEngine(load_settings(project_root=project), run_id="TRANSFORM_X", clock=lambda: FIXED)
    _edit(project, "generation.yaml", lambda d: d.__setitem__("random_seed", 7))
    e2 = TransformationEngine(load_settings(project_root=project), run_id="TRANSFORM_X", clock=lambda: FIXED)
    t1 = e1.apply(root, TranslationTransformation(FakeTranslator()), {"target_language": "hi"}).transformation
    t2 = e2.apply(root, TranslationTransformation(FakeTranslator()), {"target_language": "hi"}).transformation
    assert t1.transformation_id == t2.transformation_id
    assert t1.derived_seed != t2.derived_seed


def test_child_id_depends_on_actual_parent_text(settings):
    """A non-deterministic parent step keeps the parent id but must not alias the children."""
    alt = TRANSLATIONS[("hi", EN)].replace("कौन सी", "कौनसी")
    kids = []
    for hi_text in (TRANSLATIONS[("hi", EN)], alt):
        e = TransformationEngine(settings, run_id="TRANSFORM_T", clock=lambda: FIXED)
        r = e.root(make_seed()).variant
        hi = e.apply(r, TranslationTransformation(FakeTranslator({("hi", EN): hi_text})),
                     {"target_language": "hi"}).variant
        tl = FakeTransliterator({hi_text: "Varanasi shahar se kaun si nadi behti hai?"})
        kids.append((hi.prompt_id, e.apply(hi, TransliterationTransformation(tl)).variant.prompt_id))
    assert kids[0][0] == kids[1][0] and kids[0][1] != kids[1][1]


def test_derived_seed_and_params_reach_provider(engine, root):
    mt = FakeTranslator()
    t = engine.apply(root, TranslationTransformation(mt), {"target_language": "mr"}).transformation
    assert mt.calls == [dict(text=EN, source_language="en", target_language="mr",
                             target_script="Deva", seed=t.derived_seed)]
    assert t.provider == "fake_mt" and t.provider_version == "1.0" and t.generator_model == "fake_mt-model"
    assert t.provider_metadata == {"beam_size": 1} and t.raw_output == TRANSLATIONS[("mr", EN)]


def test_repeated_request_is_not_re_executed(engine, root):
    mt = FakeTranslator()
    a = engine.apply(root, TranslationTransformation(mt), {"target_language": "hi"})
    b = engine.apply(root, TranslationTransformation(mt), {"target_language": "hi", "target_script": "Deva"})
    assert a == b and len(mt.calls) == 1


def test_paraphrase_variant_index_gives_distinct_children(engine, root):
    pp = FakeParaphraser()
    a = engine.apply(root, ParaphraseTransformation(pp), {"variant_index": 0}).variant
    b = engine.apply(root, ParaphraseTransformation(pp), {"variant_index": 1}).variant
    assert a.prompt_id != b.prompt_id and a.prompt != b.prompt


# ------------------------------------------------------- validation hooks


def test_wrong_script_output_fails_but_is_kept(engine, root):
    mt = FakeTranslator({("hi", EN): "Which river flows through Varanasi city?"})
    res = engine.apply(root, TranslationTransformation(mt), {"target_language": "hi"})
    assert res.transformation.status == "VALIDATION_FAILED" and not res.ok
    assert res.variant.validation_status == "FAIL"
    assert res.variant.validation_failures == ["expected_script:script_mismatch"]
    hook = next(h for h in res.transformation.validation_results if h.hook == "expected_script")
    assert hook.details["expected"] == "Deva" and hook.details["measured"] == "Latn"
    assert res.variant.prompt_id in engine.variants          # kept for audit

    tl = FakeTransliterator({"Which river flows through Varanasi city?": "x"})
    with pytest.raises(TransformationError, match="failed validation"):
        engine.apply(res.variant, TransliterationTransformation(tl))


def test_low_script_share_fails(engine, root):
    mixed = "वाराणसी शहर से कौन सी नदी बहती है river?"   # 26 Deva / 31 letters = 0.84
    res = engine.apply(root, TranslationTransformation(FakeTranslator({("hi", EN): mixed})),
                       {"target_language": "hi"})
    assert res.variant.validation_failures == ["expected_script:low_script_share"]


def test_unchanged_output_is_condition_not_realised(engine, root):
    pp = FakeParaphraser({(EN, 0): EN + "  "})   # only whitespace differs
    res = engine.apply(root, ParaphraseTransformation(pp))
    assert res.variant.validation_failures == ["differs_from_parent:condition_not_realised"]


def test_length_ratio_warns(engine, root):
    pp = FakeParaphraser({(EN, 0): "Varanasi?"})   # 9 / 48 chars
    res = engine.apply(root, ParaphraseTransformation(pp))
    assert res.ok and res.variant.validation_status == "WARN"
    warn = next(h for h in res.transformation.validation_results if h.status == "WARN")
    assert warn.reason == "length_ratio_out_of_range" and warn.details["ratio"] < 0.30


@pytest.mark.parametrize("bad,reason", [
    ("वाराणसी � नदी कौन सी है?", "replacement_char"),
    ("वाराणसी\x07 नदी कौन सी है?", "control_chars"),
    ("", "empty_output"),
])
def test_text_integrity_and_empty(engine, root, bad, reason):
    res = engine.apply(root, TranslationTransformation(FakeTranslator({("hi", EN): bad})),
                       {"target_language": "hi"})
    assert res.variant.validation_status == "FAIL"
    assert any(f.endswith(reason) for f in res.variant.validation_failures)


def test_allow_failed_parent_is_explicit(engine, root):
    damaged = TRANSLATIONS[("hi", EN)].replace("नदी", "न�दी")
    v = engine.apply(root, TranslationTransformation(FakeTranslator({("hi", EN): damaged})),
                     {"target_language": "hi"}).variant
    assert v.validation_failures == ["text_integrity:replacement_char"]
    tl = FakeTransliterator({damaged: "Varanasi shahar se kaun si n?di behti hai?"})
    with pytest.raises(TransformationError, match="failed validation"):
        engine.apply(v, TransliterationTransformation(tl))
    child = engine.apply(v, TransliterationTransformation(tl), allow_failed_parent=True).variant
    assert child.parent_prompt_id == v.prompt_id and child.validation_status == "PASS"


def test_custom_hook_from_config(project):
    _edit(project, "generation.yaml",
          lambda d: d["transformation_engine"]["validation_hooks"].__setitem__("translation", ["no_question_mark"]))
    s = load_settings(project_root=project)

    def no_qm(ctx):
        ok = "?" not in ctx.text
        return HookResult(hook="no_question_mark", status="PASS" if ok else "WARN",
                          reason=None if ok else "has_question_mark")

    e = TransformationEngine(s, run_id="TRANSFORM_T", clock=lambda: FIXED, extra_hooks={"no_question_mark": no_qm})
    r = e.root(make_seed()).variant
    res = e.apply(r, TranslationTransformation(FakeTranslator()), {"target_language": "hi"})
    assert res.transformation.validation_hooks == ["no_question_mark"]
    assert res.variant.validation_status == "WARN"


def test_unknown_hook_in_config_fails_at_engine_start(project):
    _edit(project, "generation.yaml",
          lambda d: d["transformation_engine"]["validation_hooks"]["default"].append("no_such_hook"))
    with pytest.raises(ConfigError, match="no_such_hook"):
        TransformationEngine(load_settings(project_root=project))


# -------------------------------------------------------- provider errors


def test_provider_error_is_recorded_without_variant(engine, root):
    res = engine.apply(root, TranslationTransformation(FakeTranslator(fail=True)), {"target_language": "gu"})
    t = res.transformation
    assert res.variant is None and t.status == "ERROR" and t.output_prompt_id is None
    assert "simulated MT backend failure" in t.error
    assert t.transformation_id in engine.transformations
    assert verify_lineage(engine.variants, engine.transformations) == []


def test_non_json_metadata_is_provider_error(engine, root):
    mt = FakeTranslator(metadata={"obj": object()})
    res = engine.apply(root, TranslationTransformation(mt), {"target_language": "hi"})
    assert res.transformation.status == "ERROR" and "JSON" in res.transformation.error


def test_wrong_return_type_is_provider_error(engine, root):
    class Bad(FakeTranslator):
        def translate(self, text, **kw):
            return "plain string"

    res = engine.apply(root, TranslationTransformation(Bad()), {"target_language": "hi"})
    assert res.transformation.status == "ERROR" and "ProviderOutput" in res.transformation.error


# --------------------------------------------------- request validation


@pytest.mark.parametrize("params,match", [
    ({}, "target_language is required"),
    ({"target_language": "hi", "temperature": 0.7}, "unknown parameter"),
    ({"target_language": "en"}, "already 'en'"),
    ({"target_language": "xx"}, "not in languages.yaml"),
    ({"target_language": "hi", "target_script": "Latn"}, "native script"),
])
def test_translation_request_validation(engine, root, params, match):
    with pytest.raises(TransformationError, match=match):
        engine.apply(root, TranslationTransformation(FakeTranslator()), params)
    assert len(engine.transformations) == 1     # only the root; nothing recorded for bad requests


def test_translation_unsupported_pair_and_disabled_language(project):
    _edit(project, "languages.yaml", lambda d: d["languages"]["gu"].__setitem__("enabled", False))
    e = TransformationEngine(load_settings(project_root=project), run_id="TRANSFORM_T", clock=lambda: FIXED)
    r = e.root(make_seed()).variant
    with pytest.raises(TransformationError, match="disabled"):
        e.apply(r, TranslationTransformation(FakeTranslator()), {"target_language": "gu"})
    with pytest.raises(TransformationError, match="does not support"):
        e.apply(r, TranslationTransformation(FakeTranslator(targets=("hi",))), {"target_language": "mr"})


def test_translation_refuses_transliterated_parent(engine):
    c = _chain(engine)
    mt = FakeTranslator(targets=("hi", "mr", "gu"))
    with pytest.raises(TransformationError, match="monolingual, native-script"):
        engine.apply(c["hi_latn"], TranslationTransformation(mt), {"target_language": "mr"})


def test_transliteration_request_validation(engine, root):
    tl = FakeTransliterator()
    with pytest.raises(TransformationError, match="no romanized_script"):
        engine.apply(root, TransliterationTransformation(tl))       # en has no romanized form
    with pytest.raises(TransformationError, match="unknown script"):
        engine.apply(root, TransliterationTransformation(tl), {"target_script": "Zzzz"})
    with pytest.raises(TransformationError, match="does not support"):
        engine.apply(root, TransliterationTransformation(tl), {"target_script": "Deva"})


def test_paraphrase_limited_to_configured_languages(engine):
    c = _chain(engine)
    with pytest.raises(TransformationError, match="paraphrase.languages"):
        engine.apply(c["hi"], ParaphraseTransformation(FakeParaphraser()))
    with pytest.raises(TransformationError, match="variant_index"):
        engine.apply(c["root"], ParaphraseTransformation(FakeParaphraser()), {"variant_index": -1})


def test_disabled_transformation_type_refused(project):
    _edit(project, "generation.yaml", lambda d: (
        d["transformations"]["enabled"].remove("paraphrase"),
        d["transformations"]["disabled"].append("paraphrase"),
    ))
    e = TransformationEngine(load_settings(project_root=project), run_id="TRANSFORM_T", clock=lambda: FIXED)
    r = e.root(make_seed()).variant
    with pytest.raises(TransformationError, match="not enabled"):
        e.apply(r, ParaphraseTransformation(FakeParaphraser()))


def test_identity_only_via_root(engine, root):
    class Identity(ParaphraseTransformation):
        transformation_type = "identity"

    with pytest.raises(TransformationError, match="root"):
        engine.apply(root, Identity(FakeParaphraser()))


# --------------------------------------------------- provider registry


def test_default_providers_from_real_config(settings):
    # A configured default with no registered adapter is refused, not faked.
    with pytest.raises(ProviderUnavailableError, match="no adapter"):
        build_provider(settings, "translation", factories={})
    with pytest.raises(ProviderUnavailableError, match="no adapter"):
        build_provider(settings, "transliteration", factories={})
    with pytest.raises(ProviderUnavailableError, match="disabled"):
        build_translation_provider(settings, "llm")
    with pytest.raises(ProviderUnavailableError, match="no default paraphrase"):
        build_paraphrase_provider(settings)
    with pytest.raises(ProviderUnavailableError, match="not configured"):
        build_translation_provider(settings, "nope")


def test_registered_factory_receives_config(settings):
    seen = {}

    def factory(name, cfg):
        seen.update(name=name, targets=cfg.target_languages)
        return FakeTranslator(name=name, targets=cfg.target_languages)

    p = build_provider(settings, "translation", factories={("translation", "indictrans2"): factory})
    assert seen == {"name": "indictrans2", "targets": ["hi", "mr", "gu"]}
    assert p.info.name == "indictrans2"


# ----------------------------------------------------------------- config


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["transformations"]["enabled"].append("teleport"), "unknown transformation type"),
    (lambda d: d["transformations"]["enabled"].remove("identity"), "'identity' must be enabled"),
    (lambda d: d["transformations"]["disabled"].append("translation"), "both enabled and disabled"),
    (lambda d: d["translation"].__setitem__("default_provider", "ghost"), "not in providers"),
    (lambda d: d["translation"]["providers"]["indictrans2"]["target_languages"].append("xx"), "unknown language"),
    (lambda d: d["paraphrase"]["languages"].append("xx"), "paraphrase language"),
    (lambda d: d["transformation_engine"]["length_ratio"].__setitem__("min", 5.0), "min must be"),
    (lambda d: d["transformation_engine"]["validation_hooks"].pop("default"), "'default'"),
    (lambda d: d["translation"]["providers"]["llm"].__setitem__("type", "magic"), "type"),
])
def test_transformation_config_validation(project, mutate, match):
    _edit(project, "generation.yaml", mutate)
    with pytest.raises(ConfigError, match=match):
        load_settings(project_root=project)


# ----------------------------------------------------------------- schema


def test_variant_schema_guards(root):
    data = root.model_dump()
    with pytest.raises(ValidationError, match="final_label"):
        VariantRecord(**{**data, "final_label": "SAFE"})
    with pytest.raises(ValidationError, match="identity root"):
        VariantRecord(**{**data, "parent_prompt_id": "P-X-1-ab"})
    with pytest.raises(ValidationError, match="validation_failures"):
        VariantRecord(**{**data, "validation_status": "FAIL"})


# ----------------------------------------------------------------- export


def test_export_round_trip(engine, settings):
    c = _chain(engine)
    engine.apply(c["root"], TranslationTransformation(FakeTranslator(fail=True)), {"target_language": "hi",
                                                                                     "target_script": "Deva"})
    paths = engine.export(settings.project_root / "data" / "processed" / "transformations")
    assert paths["variants"].parent.name == "TRANSFORM_TEST"

    variants = load_variants_jsonl(paths["variants"])
    transformations = load_transformations_jsonl(paths["transformations"])
    assert variants == engine.variants
    assert transformations == engine.transformations
    steps = trace_lineage(c["gu_latn"].prompt_id, variants, transformations)
    assert [s["language"] + "/" + s["script"] for s in steps] == ["en/Latn", "gu/Gujr", "gu/Latn"]

    m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert m["run_id"] == "TRANSFORM_TEST" and m["run_type"] == "transformation"
    assert m["input_seeds"] == {"S-D10K-020000000401": {"seed_version": 1,
                                                        "content_hash": c["root"].content_hash}}
    assert m["counts"]["variants_by_validation_status"] == {"PASS": 8}
    assert m["counts"]["transformations_by_type_status"]["translation:SUCCEEDED"] == 3
    assert set(m["outputs"]) == {"variants.jsonl", "transformations.jsonl"}
    assert m["random_seed"] == settings.generation.random_seed and "generation.yaml" in m["config_sha256"]


def test_export_refuses_raw_dir(engine, settings):
    _chain(engine)
    with pytest.raises(ConfigError, match="read-only"):
        engine.export(settings.raw_dir)


def test_export_refuses_broken_lineage(engine, settings):
    c = _chain(engine)
    del engine.variants[c["hi"].prompt_id]
    with pytest.raises(LineageError):
        engine.export(settings.project_root / "out")


def test_real_pilot_seed_root(settings):
    """A real pilot seed (from the committed pilot file) becomes a valid root."""
    from generator.seed_manager import load_seeds_jsonl
    from tests.conftest import REPO

    seeds = load_seeds_jsonl(REPO / "data" / "pilot" / "pilot_seeds_v0.1-pilot-seeds.jsonl")
    e = TransformationEngine(settings, run_id="TRANSFORM_T", clock=lambda: FIXED)
    roots = [e.root(s).variant for s in seeds]
    assert len({r.prompt_id for r in roots}) == len(seeds)
    assert all(r.validation_status == "PASS" for r in roots)
    assert all(r.prompt == s.prompt and r.seed_id == s.seed_id for r, s in zip(roots, seeds))
