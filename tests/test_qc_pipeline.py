"""Phase 5 QC pipeline: per-variant checks and summary, with a fake sentence encoder."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from generator.code_mixing import build_code_mixer
from generator.pilot_translation import run_pilot_translation, write_outputs
from generator.qc_pipeline import (
    char3,
    code_mix_check,
    duplicate_checks,
    jaccard,
    run_qc,
    run_qc_on_dir,
)
from generator.semantic import SentenceEncoder, cosine
from generator.transformation_engine import TransformationEngine
from tests.fakes import TRANSLATIONS, FakeTranslator, FakeTransliterator
from tests.test_code_mix import HI_L1, ROMAN, WORDS
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


@pytest.fixture
def run(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    mt = FakeTranslator({**TRANSLATIONS, **WORDS})
    return run_pilot_translation(settings, [make_seed()], mt, FakeTransliterator(ROMAN), FakeLID(), ["hi"],
                                 engine=engine, code_mixer=build_code_mixer(settings, mt, None))


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
    no_enc = run_qc(settings, run.engine.variants, run.engine.transformations, run.qc, None)
    assert {r.checks["semantic"].status for r in no_enc} == {"NOT_RUN"}


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


def test_cosine():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0 and cosine([0.6, 0.8], [0.6, 0.8]) == 1.0


def test_identical_levels_flag_the_higher_level(settings, run):
    vs = list(run.engine.variants.values())
    l1 = next(v for v in vs if v.code_mix_level == "L1" and not v.is_transliterated)
    l2 = next(v for v in vs if v.code_mix_level == "L2" and not v.is_transliterated)
    same = l2.model_copy(update={"prompt": l1.prompt, "content_hash": l1.content_hash, "prompt_id": "P-D10K-1-0000"})
    exact, _ = duplicate_checks([*[v for v in vs if v is not l2], same], 0.85)
    assert exact[same.prompt_id].reason == "exact_duplicate" and exact[l1.prompt_id].status == "PASS"
