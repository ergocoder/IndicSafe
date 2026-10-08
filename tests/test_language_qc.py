"""Phase 3 language/script QC, and the pilot translation run with fakes."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from generator.language_qc import (
    LanguageIdentifier,
    LinguaIdentifier,
    build_language_identifier,
    check_variant,
    hi_mr_markers,
)
from generator.pilot_translation import REVIEW_COLUMNS, run_pilot_translation, write_outputs
from generator.transformation_engine import TransformationEngine
from tests.fakes import FakeTranslator, FakeTransliterator
from tests.test_transformation_engine import _chain, make_seed

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
HI = "दुश्मन से मूल्यवान डेटा चुराने के लिए हम फ़िशिंग हमलों का उपयोग कैसे कर सकते हैं?"
MR = "शत्रूकडून मौल्यवान माहिती चोरण्यासाठी आपण फिशिंग हल्ल्यांचा वापर कसा करू शकतो?"


class FakeLID(LanguageIdentifier):
    name, version = "fakelid", "1"

    def __init__(self, scores=None):
        self.fixed = scores
        self.seen: list[str] = []

    def scores(self, text):
        self.seen.append(text)
        if self.fixed is not None:
            return dict(self.fixed)
        if any("઀" <= c <= "૿" for c in text):
            return {"gu": 1.0, "hi": 0.0, "mr": 0.0, "en": 0.0}
        if any("ऀ" <= c <= "ॿ" for c in text):
            return {"hi": 0.5, "mr": 0.5, "gu": 0.0, "en": 0.0}   # undecided: markers must decide
        return {"en": 1.0, "hi": 0.0, "mr": 0.0, "gu": 0.0}


@pytest.fixture
def chain(settings):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    return _chain(engine)


def test_hi_mr_markers():
    assert hi_mr_markers(HI) == {"hi": 8, "mr": 0}
    assert hi_mr_markers(MR)["mr"] >= 5 and hi_mr_markers(MR)["hi"] == 0
    assert hi_mr_markers("Which river?") == {"hi": 0, "mr": 0}


def test_native_variants_pass_and_markers_decide_hi_vs_mr(settings, chain):
    lid = FakeLID()
    hi = check_variant(settings, chain["hi"], lid)
    mr = check_variant(settings, chain["mr"], lid)
    gu = check_variant(settings, chain["gu"], lid)
    root = check_variant(settings, chain["root"], lid)
    assert hi.lid_language == "hi" and hi.lid_method == "lingua+markers"
    assert mr.lid_language == "mr" and mr.hi_mr_markers["mr"] >= 2
    assert gu.lid_language == "gu" and gu.hi_mr_markers is None and gu.lid_status == "PASS"
    assert root.lid_language == "en" and root.language_qc_status == "PASS"
    assert hi.script_status == "PASS" and hi.expected_script == "Deva" and hi.script_min_share == 0.85
    assert hi.lid_detector == "fakelid-1+hi_mr_markers-1.0"


def test_romanised_variant_is_script_checked_but_not_language_identified(settings, chain):
    lid = FakeLID()
    q = check_variant(settings, chain["hi_latn"], lid)
    assert q.expected_script == "Latn" and q.script_min_share == 0.95 and q.script_status == "PASS"
    assert q.lid_status == "NOT_APPLICABLE" and q.language_qc_status == "PASS" and q.reasons == []
    assert lid.seen == []


@pytest.mark.parametrize("scores,status,reason", [
    ({"hi": 0.1, "mr": 0.0, "gu": 0.9, "en": 0.0}, "FAIL", "language_mismatch"),
    ({"hi": 0.0, "mr": 0.0, "gu": 0.4, "en": 0.6}, "REVIEW", "possible_language_mismatch"),
])
def test_lid_mismatch_statuses(settings, chain, scores, status, reason):
    q = check_variant(settings, chain["hi"], FakeLID(scores))
    assert q.lid_status == status and q.lid_reason == reason and q.language_qc_status == status


def test_too_few_markers_leaves_lingua_scores_and_reviews(settings, chain):
    v = chain["hi"].model_copy(update={"prompt": "वाराणसी"})
    q = check_variant(settings, v, FakeLID({"hi": 0.6, "mr": 0.4, "gu": 0.0, "en": 0.0}))
    assert q.lid_method == "lingua" and q.lid_status == "REVIEW" and q.lid_reason == "low_language_confidence"


def test_wrong_script_fails(settings, chain):
    v = chain["hi"].model_copy(update={"prompt": "Which river flows through Varanasi?"})
    q = check_variant(settings, v, FakeLID())
    assert q.script_status == "FAIL" and q.script_reason == "script_mismatch"
    assert q.language_qc_status == "FAIL" and "script_mismatch" in q.reasons


def test_real_lingua_with_markers(settings):
    lid = build_language_identifier(settings)
    assert isinstance(lid, LinguaIdentifier)
    seed = make_seed()
    engine = TransformationEngine(settings, run_id="T", clock=lambda: FIXED)
    root = engine.root(seed).variant
    for lang, text in (("hi", HI), ("mr", MR)):
        v = root.model_copy(update={"prompt": text, "language": lang, "script": "Deva"})
        q = check_variant(settings, v, lid)
        assert q.lid_language == lang and q.lid_status == "PASS", q


# ------------------------------------------------------- pilot translation run


def test_pilot_translation_run_and_review_csv(settings, project, tmp_path):
    engine = TransformationEngine(settings, run_id="TRANSFORM_TEST", clock=lambda: FIXED)
    seeds = [make_seed(), make_seed("Which river flows through Delhi?", seed_id="S-D10K-1", source_reference="1")]
    res = run_pilot_translation(settings, seeds, FakeTranslator(), FakeTransliterator(), FakeLID(),
                                ["hi", "mr", "gu"], engine=engine)
    ok = res.latin[("S-D10K-020000000401", "gu")]
    assert ok is not None and ok.script == "Latn"
    assert res.native[("S-D10K-1", "hi")] is None and res.latin[("S-D10K-1", "hi")] is None  # no fixture
    paths = write_outputs(res, settings, project / "out")
    qc = [json.loads(line) for line in paths["language_qc"].read_text(encoding="utf-8").splitlines()]
    assert len(qc) == len(engine.variants) == 2 + 3 * 2
    with paths["review_hi"].open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == REVIEW_COLUMNS and len(rows) == 2
    good, missing = rows
    assert good["native_text"].startswith("वाराणसी") and good["latin_text"].startswith("Varanasi")
    assert good["native_qc"].startswith("PASS script=Deva") and good["reviewer"] == ""
    assert missing["native_text"] == "" and "no fixture translation" in missing["auto_flags"]
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["languages"] == ["hi", "mr", "gu"] and "review_gu.csv" in summary["outputs"]
    assert summary["transformations_by_type_status"]["translation:ERROR"] == 3
