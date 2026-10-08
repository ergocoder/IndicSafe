"""Phase 5 QC pipeline: per-variant checks and summary, with a fake sentence encoder."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from generator.code_mixing import build_code_mixer
from generator.pilot_translation import run_pilot_translation, write_outputs
from generator.qc_pipeline import (
    char3,
    CODEMIX_REVIEW_COLUMNS,
    code_mix_check,
    code_mix_coverage,
    codemix_review_rows,
    duplicate_checks,
    final_dataset_variants,
    harmful_intent_check,
    jaccard,
    run_qc,
    run_qc_on_dir,
)
from generator.semantic import SentenceEncoder, cosine
from generator.transformation_engine import TransformationEngine
from tests.fakes import TRANSLATIONS, FakeTranslator, FakeTransliterator
from tests.test_code_mix import ANALYZER, HI_L1, ROMAN, WORDS
from tests.test_language_qc import FakeLID
from tests.test_transformation_engine import make_seed

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


class FakeEncoder(SentenceEncoder):
    """Every text maps to the same unit vector, except those listed as drifted (orthogonal)."""

    name, version = "fake-encoder", "1"

    def __init__(self, drifted=(), half=()):
        self.drifted, self.half, self.calls = set(drifted), set(half), 0

    def encode(self, texts):
        self.calls += 1
        return [[0.0, 1.0] if t in self.drifted else [0.7, 0.71414] if t in self.half else [1.0, 0.0]
                for t in texts]


def _run(settings, seed=None):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    mt = FakeTranslator({**TRANSLATIONS, **WORDS})
    return run_pilot_translation(settings, [seed or make_seed()], mt, FakeTransliterator(ROMAN), FakeLID(), ["hi"],
                                 engine=engine, code_mixer=build_code_mixer(settings, mt, None, analyzer=ANALYZER))


@pytest.fixture
def run(settings):
    return _run(settings)


def _by_kind(records):
    return {(r.script, r.code_mix_level, r.transformation_type): r for r in records}


def test_every_variant_gets_all_checks_and_clean_run_passes(settings, run):
    recs = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, FakeEncoder())
    assert len(recs) == len(run.engine.variants) == 7
    assert all(set(r.checks) == {"engine_hooks", "exact_duplicate", "near_duplicate", "script_language",
                                 "code_mix", "length_ratio", "semantic"} for r in recs)
    k = _by_kind(recs)
    root = k[("Latn", None, "identity")]
    assert root.checks["semantic"].status == root.checks["length_ratio"].status == "NOT_APPLICABLE"
    assert root.checks["code_mix"].reason == "no_code_mix_partner"
    l1 = k[("Deva", "L1", "code_mixing")]
    assert (l1.code_mix_ratio, l1.cmi, l1.semantic_similarity) == (0.125, 12.5, 1.0)
    assert l1.checks["code_mix"].details["measured_level"] == "L1"
    assert l1.checks["semantic"].details["similarity_to_parent"] == 1.0
    lat = k[("Latn", "L1", "transliteration")]
    assert lat.checks["semantic"].details["method"] == "inherited_from_native_parent"
    assert lat.code_mix_ratio == 0.125 and lat.checks["code_mix"].details["tagging"] == "aligned_to_native_parent"
    assert k[("Deva", None, "translation")].code_mix_ratio == 0.0
    assert {r.qc_status for r in recs} == {"PASS"}, [r.reasons for r in recs]
    assert all(r.intended_label == "SAFE" and r.seed_id == root.seed_id for r in recs)


def test_semantic_thresholds(settings, run):
    hi = next(v for v in run.engine.variants.values() if v.transformation_type == "translation")
    recs = _by_kind(run_qc(settings, run.engine.variants, run.engine.transformations, run.qc,
                           FakeEncoder(drifted={hi.prompt}, half={HI_L1})))
    t = recs[("Deva", None, "translation")]
    assert t.qc_status == "FAIL" and "semantic:semantic_drift" in t.reasons
    assert recs[("Latn", None, "transliteration")].checks["semantic"].reason == "semantic_drift"  # inherited
    l1 = recs[("Deva", "L1", "code_mixing")]
    assert l1.checks["semantic"].status == "REVIEW" and l1.semantic_similarity == pytest.approx(0.7)
    assert l1.semantic_similarity_to_parent == pytest.approx(0.7141, abs=1e-3)
    assert l1.checks["semantic"].details["decision_basis"] == "native_l0_parent"
    no_enc = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, None)
    assert {r.checks["semantic"].status for r in no_enc} == {"NOT_RUN"}


def test_code_mix_semantic_decision_uses_the_native_parent(settings, run):
    """Seed far from everything: L0 fails on the seed score; code-mix passes on its parent score."""
    root = next(v for v in run.engine.variants.values() if v.parent_prompt_id is None)
    recs = _by_kind(run_qc(settings, run.engine.variants, run.engine.transformations, run.qc,
                           FakeEncoder(drifted={root.prompt})))
    assert recs[("Deva", None, "translation")].checks["semantic"].reason == "semantic_drift"
    for key in (("Deva", "L1", "code_mixing"), ("Latn", "L1", "transliteration"), ("Deva", "L2", "code_mixing")):
        c = recs[key].checks["semantic"]
        assert c.status == "PASS", key
        assert (c.details["similarity_to_seed"], c.details["similarity_to_parent"]) == (0.0, 1.0)
        assert c.details["decision_basis"] == "native_l0_parent"


def test_fail_variants_never_reach_the_final_dataset(settings, run):
    hi = next(v for v in run.engine.variants.values() if v.transformation_type == "translation")
    recs = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc,
                  FakeEncoder(drifted={hi.prompt}, half={HI_L1}))
    status = {r.prompt_id: r.qc_status for r in recs}
    final = final_dataset_variants(run.engine.variants, recs)
    assert {v.prompt_id for v in final} == {p for p, st in status.items() if st != "FAIL"}
    assert hi.prompt_id not in {v.prompt_id for v in final}
    strict = final_dataset_variants(run.engine.variants, recs, include_review=False)
    assert {v.prompt_id for v in strict} == {p for p, st in status.items() if st == "PASS"}
    with pytest.raises(ValueError, match="no QC record"):
        final_dataset_variants(run.engine.variants, recs[1:])


def test_code_mix_coverage_line(settings, run):
    recs = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, FakeEncoder())
    cov = code_mix_coverage(recs, run.engine.transformations)
    assert cov["L1"]["line"] == "L1 band reached: 1/1" and cov["L2"]["line"] == "L2 band reached: 1/1"
    assert cov["L2"]["by_language"]["hi"]["missed"] == 0


def test_codemix_review_sheet_puts_review_first_and_includes_unsafe(settings):
    run = _run(settings, make_seed(intended_label="UNSAFE"))
    recs = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, FakeEncoder(half={HI_L1}))
    rows = codemix_review_rows(settings, run.engine.variants, run.engine.transformations, recs, "hi")
    assert [(r["level"], r["selection"]) for r in rows] == [("L1", "qc_review"), ("L2", "random_sample")]
    assert all(r["intended_label"] == "UNSAFE" and r["latin_text"] for r in rows)
    assert rows[0]["swapped"] == "river <- नदी" and rows[0]["qc_status"].startswith("REVIEW")
    assert set(rows[0]) == set(CODEMIX_REVIEW_COLUMNS) - {"reviewer", "codemix_natural_1to3", "intent_preserved_Y_N", "notes"}


def test_harmful_intent_check_lists_lowest_similarity_unsafe_variants(settings):
    run = _run(settings, make_seed(intended_label="UNSAFE"))
    recs = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, FakeEncoder(half={HI_L1}))
    out = harmful_intent_check(run.engine.variants, recs, 2)
    assert out["n_candidates"] == 3 and len(out["items"]) == 2          # native L0, L1, L2
    first = out["items"][0]
    assert first["variant_text"] == HI_L1 and first["level"] == "L1" and first["romanised_text"] == ROMAN[HI_L1]
    assert first["similarity_to_seed"] == pytest.approx(0.7) and first["intended_label"] == "UNSAFE"
    safe = _run(settings)
    safe_recs = run_qc(settings, safe.engine.variants, safe.engine.transformations, safe.qc, FakeEncoder())
    assert harmful_intent_check(safe.engine.variants, safe_recs, 10)["items"] == []          # SAFE seed


def test_exact_and_near_duplicates(settings, run):
    vs = list(run.engine.variants.values())
    l1 = next(v for v in vs if v.code_mix_level == "L1" and not v.is_transliterated)
    same = l1.model_copy(update={"prompt_id": "P-D10K-9-ffff", "lineage": [*l1.lineage, l1.prompt_id],
                                 "parent_prompt_id": l1.prompt_id})
    other_seed = l1.model_copy(update={"prompt_id": "P-D10K-8-eeee", "seed_id": "S-D10K-8",
                                       "prompt": l1.prompt.replace("?", "!!"), "content_hash": "x"})
    exact, near = duplicate_checks([*vs, same, other_seed], 0.85)
    assert exact[same.prompt_id].reason == "exact_duplicate" and exact[same.prompt_id].details["duplicate_of"] == l1.prompt_id
    assert exact[l1.prompt_id].status == "PASS"                    # the shallower one is kept
    # the later seed (by seed_id) is flagged against the earlier one
    assert near[other_seed.prompt_id].reason == "near_duplicate"
    assert near[other_seed.prompt_id].details["closest"] == l1.prompt_id and near[l1.prompt_id].status == "PASS"
    assert near[same.prompt_id].status == "PASS"                   # same seed: not a near-dup finding
    assert jaccard(char3("abc def"), char3("abc def")) == 1.0 and jaccard(char3("abc"), char3("xyz")) == 0.0


def test_code_mix_check_statuses(settings, run):
    l2 = next(v for v in run.engine.variants.values() if v.code_mix_level == "L2" and not v.is_transliterated)
    parent = run.engine.variants[l2.parent_prompt_id]
    assert code_mix_check(settings, l2, parent).status == "PASS"
    wrong = l2.model_copy(update={"code_mix_level": "L1"})          # 0.25 vs L1 [0.05, 0.20) + 0.03
    c = code_mix_check(settings, wrong, parent)
    assert (c.status, c.reason, c.details["measured_level"]) == ("FAIL", "code_mix_out_of_band", "L2")
    mono = parent.model_copy(update={"prompt": "वाराणसी city से कौन सी river बहती है?"})
    assert code_mix_check(settings, mono, None).reason == "monolingual_variant_contains_partner_words"


def test_run_qc_on_dir_writes_report_and_summary(settings, run, project):
    paths = write_outputs(run, settings, project / "out")
    out = run_qc_on_dir(settings, paths["manifest"].parent, FakeEncoder())
    rows = [json.loads(x) for x in out["report"].read_text(encoding="utf-8").splitlines()]
    s = json.loads(out["summary"].read_text(encoding="utf-8"))
    assert len(rows) == s["n_variants"] == 7 and s["qc_status"] == {"PASS": 7}
    assert s["qc_status_by_kind"]["hi/Deva/L1"] == {"PASS": 1}
    assert s["code_mix_ratio_by_kind"]["hi/Latn/L2"]["mean"] == 0.25
    assert s["semantic_encoder"]["name"] == "fake-encoder" and s["config"]["semantic"]["pass"] == 0.8
    assert set(s["inputs"]) == {"variants.jsonl", "transformations.jsonl", "language_qc.jsonl"}
    assert s["code_mix_coverage"]["L2"]["line"] == "L2 band reached: 1/1"
    assert s["final_dataset"]["excluded_fail"] == 0 and s["final_dataset"]["eligible_including_review"] == 7
    assert set(s["outputs"]) == {"qc_report.jsonl", "review_codemix_hi.csv", "harmful_intent_check.json"}
    with out["review_codemix_hi"].open(encoding="utf-8-sig", newline="") as fh:
        sheet = list(csv.DictReader(fh))
    assert list(sheet[0]) == CODEMIX_REVIEW_COLUMNS and len(sheet) == 2
    assert json.loads(out["harmful_intent"].read_text(encoding="utf-8"))["items"] == []


def test_cosine():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0 and cosine([0.6, 0.8], [0.6, 0.8]) == 1.0


def test_identical_levels_flag_the_higher_level(settings, run):
    vs = list(run.engine.variants.values())
    l1 = next(v for v in vs if v.code_mix_level == "L1" and not v.is_transliterated)
    l2 = next(v for v in vs if v.code_mix_level == "L2" and not v.is_transliterated)
    same = l2.model_copy(update={"prompt": l1.prompt, "content_hash": l1.content_hash, "prompt_id": "P-D10K-1-0000"})
    exact, _ = duplicate_checks([*[v for v in vs if v is not l2], same], 0.85)
    assert exact[same.prompt_id].reason == "exact_duplicate" and exact[l1.prompt_id].status == "PASS"
