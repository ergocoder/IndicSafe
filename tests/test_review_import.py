"""Pilot review layer: mapping, agreement, worksheet check and adjudication build.

Each test copies the real v0.1 pilot, review files, mapping table and worksheet
into a temp project, so the real files are never written. Edits made here
(filled adjudications, broken hashes) are test inputs, not team decisions.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from backend.config import load_settings
from generator.provenance import sha256_file
from generator.review_import import (
    ADJUDICATION_COLUMNS,
    AdjudicationError,
    ReviewImportError,
    build_reviewed_pilot,
    cohen_kappa,
    import_reviews,
    load_review_config,
    write_review_layer,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = "data/pilot/reviews/reviews_v0.1.yaml"
DISAGREEMENTS = {"S-NHQA-250", "S-NHQA-163", "S-NHQA-128", "S-NHQA-324",
                 "S-NHQA-253", "S-DFH-1740", "S-DFH-1497", "S-DFH-307"}


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    for rel in ("data/pilot/pilot_seeds_v0.1-pilot-seeds.jsonl",
                "data/pilot/pilot_seeds_v0.1-pilot-seeds.manifest.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    shutil.copytree(REPO / "data/pilot/reviews", tmp_path / "data/pilot/reviews",
                    ignore=shutil.ignore_patterns("review_layer_*", "review_report_*"))
    # Tests start from an unadjudicated worksheet, whatever state the real one is in.
    ws = tmp_path / "data/pilot/reviews/adjudication_worksheet_v0.1.csv"
    with ws.open(encoding="utf-8-sig", newline="") as fh:
        header, *rows = list(csv.reader(fh))
    blank = {header.index(c) for c in ADJUDICATION_COLUMNS}
    rows = [["" if i in blank else v for i, v in enumerate(r[:len(header)])] for r in rows]
    with ws.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows([header, *rows])
    return tmp_path


def _run(proj: Path):
    settings = load_settings(project_root=proj, config_dir=REPO / "configs")
    cfg = load_review_config(proj / CONFIG)
    return import_reviews(cfg, settings), settings


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames), list(r)


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _edit_csv(path: Path, fn) -> None:
    fields, rows = _rows(path)
    rows = fn(rows)
    _write_rows(path, fields, rows)


def _fill_worksheet(proj: Path, **override) -> None:
    def fill(rows):
        for row in rows:
            row.update(adjudicated_label="SAFE", adjudicated_category="deception_manipulation",
                       adjudicated_by="test-adjudicator", rationale="test input")
            row.update(override.get(row["seed_id"], {}))
        return rows
    _edit_csv(proj / "data/pilot/reviews/adjudication_worksheet_v0.1.csv", fill)


ANN_B = "data/pilot/reviews/pilot_seeds_v0.1-pilot-seeds_bhargavi_annotated.csv"
ANN_S = "data/pilot/reviews/pilot_seeds_v0.1-pilot-seeds_swasthik_annotated.csv"


# ------------------------------------------------------------------ agreement


def test_cohen_kappa_known_values():
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    # po = 0.5, pe = 0.5 -> 0
    assert cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
    # Textbook 2x2: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> 0.4
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    assert cohen_kappa(pairs) == pytest.approx(0.4)
    assert cohen_kappa([]) is None
    assert cohen_kappa([("a", "a")]) is None       # chance agreement = 1


def test_real_reviews_agreement_and_disagreements(proj):
    res, _ = _run(proj)
    lab = res.agreement["label"]
    assert (lab["n_seeds"], lab["n_compared"], lab["n_agree"]) == (30, 30, 23)
    assert lab["percent_agreement"] == pytest.approx(76.67)
    assert lab["cohen_kappa"] == pytest.approx(0.5842, abs=1e-4)
    assert set(res.agreement["needs_adjudication"]) == DISAGREEMENTS
    assert res.worksheet_check["matches"], res.worksheet_check


def test_category_mapping_flags_and_never_guesses(proj):
    res, _ = _run(proj)
    by = {r["seed_id"]: r["annotations"] for r in res.layer}
    multi = by["S-NHQA-250"]["swasthik"]
    assert multi["category_id"] is None and "MULTIPLE_CATEGORIES" in multi["flags"]
    assert multi["category_candidates"] == ["hate_discrimination", "deception_manipulation"]
    for sid in ("S-NHQA-253", "S-DFH-307"):
        blank = by[sid]["bhargavi"]
        assert blank["category_id"] is None and blank["flags"] == ["BLANK_CATEGORY"]
        assert blank["note"].startswith("either")          # note kept, not used to pick a category
    assert by["S-DFH-1281"]["bhargavi"]["category_id"] == "harassment_abuse"
    assert by["S-DFH-1281"]["bhargavi"]["flags"] == ["TYPO_MAPPED"]
    assert by["S-NHQA-6"]["swasthik"]["category_id"] == "hate_discrimination"
    assert by["S-DFH-1127"]["swasthik"]["category_mapping"] == "display_name"
    # raw values are kept verbatim
    assert by["S-NHQA-250"]["swasthik"]["raw_category"].endswith("manipulation ")


def test_unmapped_category_is_flagged_not_guessed(proj):
    def edit(rows):
        rows[0]["my_category"] = "Cyber stuff"
        return rows
    _edit_csv(proj / ANN_S, edit)
    res, _ = _run(proj)
    ann = res.layer[0]["annotations"]["swasthik"]
    assert ann["category_id"] is None and ann["flags"] == ["UNMAPPED_CATEGORY"]
    assert res.layer[0]["needs_adjudication"]
    # the worksheet does not have this seed, and the check says so
    assert res.worksheet_check["missing_from_worksheet"] == [res.layer[0]["seed_id"]]


def test_only_annotation_columns_are_read(proj):
    # Corrupt every ignored column; the import result must not change.
    before, _ = _run(proj)

    def scramble(rows):
        for row in rows:
            for k in row:
                if k not in ("seed_id", "content_hash", "my_label", "my_category", "note"):
                    row[k] = "garbage"
        return rows
    _edit_csv(proj / ANN_B, scramble)
    after, _ = _run(proj)
    strip = lambda layer: [{**r, "annotations": {n: {**a, "source_file_sha256": None}
                                                  for n, a in r["annotations"].items()}} for r in layer]
    assert strip(before.layer) == strip(after.layer)


def test_invalid_label_is_flagged(proj):
    def edit(rows):
        rows[1]["my_label"] = "maybe"
        return rows
    _edit_csv(proj / ANN_B, edit)
    res, _ = _run(proj)
    ann = res.layer[1]["annotations"]["bhargavi"]
    assert ann["label"] is None and ann["flags"] == ["INVALID_LABEL"]
    assert res.agreement["label"]["n_compared"] == 29


# ------------------------------------------------------------------ input errors


def test_content_hash_mismatch_is_rejected(proj):
    def edit(rows):
        rows[3]["content_hash"] = "0" * 64
        return rows
    _edit_csv(proj / ANN_S, edit)
    with pytest.raises(ReviewImportError, match="content_hash"):
        _run(proj)


def test_missing_and_duplicate_seeds_are_rejected(proj):
    _edit_csv(proj / ANN_B, lambda rows: rows[:-1])
    with pytest.raises(ReviewImportError, match="no annotation"):
        _run(proj)
    _edit_csv(proj / ANN_B, lambda rows: rows + rows[:1])
    with pytest.raises(ReviewImportError, match="appears twice"):
        _run(proj)


def test_mapping_table_rejects_unknown_category_id(proj):
    path = proj / "data/pilot/reviews/category_mapping_v0.1.csv"
    _edit_csv(path, lambda rows: rows + [{"raw_value": "x", "category_id": "not_a_category",
                                          "mapping_type": "display_name", "note": ""}])
    with pytest.raises(ReviewImportError, match="not in taxonomy"):
        _run(proj)


def test_changed_input_pilot_is_rejected(proj):
    path = proj / "data/pilot/pilot_seeds_v0.1-pilot-seeds.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ReviewImportError, match="checksum"):
        _run(proj)


# ------------------------------------------------------------------ worksheet check


def test_worksheet_differences_are_reported(proj):
    ws = proj / "data/pilot/reviews/adjudication_worksheet_v0.1.csv"

    def edit(rows):
        rows[0]["swasthik_label"] = "UNSAFE"          # S-NHQA-250: file says AMBIGUOUS
        return [r for r in rows if r["seed_id"] != "S-DFH-307"]
    _edit_csv(ws, edit)
    res, _ = _run(proj)
    check = res.worksheet_check
    assert not check["matches"]
    assert check["missing_from_worksheet"] == ["S-DFH-307"]
    assert check["field_differences"] == [{"seed_id": "S-NHQA-250", "column": "swasthik_label",
                                           "worksheet": "UNSAFE", "computed": "AMBIGUOUS"}]


# ------------------------------------------------------------------ build


def test_build_refuses_unfilled_worksheet(proj):
    res, settings = _run(proj)
    with pytest.raises(AdjudicationError) as e:
        build_reviewed_pilot(res, settings)
    assert len(e.value.problems) == 8
    assert not list((proj / "data/pilot").glob("*v0.2*"))


def test_build_rejects_display_name_and_partial_rows(proj):
    _fill_worksheet(proj, **{"S-NHQA-163": {"adjudicated_category": "Sexual safety"},
                             "S-DFH-307": {"rationale": ""}})
    res, settings = _run(proj)
    with pytest.raises(AdjudicationError) as e:
        build_reviewed_pilot(res, settings)
    text = "\n".join(e.value.problems)
    assert "S-NHQA-163: adjudicated_category 'Sexual safety'" in text
    assert "S-DFH-307: not adjudicated yet (empty rationale)" in text


def test_build_writes_v02_with_provenance_and_leaves_v01_untouched(proj):
    v01 = proj / "data/pilot/pilot_seeds_v0.1-pilot-seeds.jsonl"
    v01_sha = sha256_file(v01)
    _fill_worksheet(proj, **{"S-DFH-307": {"adjudicated_label": "UNSAFE"}})
    res, settings = _run(proj)
    write_review_layer(res, settings)
    paths = build_reviewed_pilot(res, settings)

    assert sha256_file(v01) == v01_sha
    recs = {r["seed_id"]: r for r in map(json.loads, paths["jsonl"].read_text(encoding="utf-8").splitlines())}
    assert len(recs) == 30

    agreed = recs["S-NHQA-366"]                       # both: UNSAFE, cyber_misuse
    assert (agreed["final_label"], agreed["category"]) == ("UNSAFE", "cyber_misuse")
    assert agreed["label_status"] == "human_agreed" and agreed["review"]["adjudication"] is None
    assert agreed["review"]["pre_review"]["category"] == "dangerous_instructions"
    assert agreed["intended_label"] == "UNSAFE"       # provisional value kept as history

    adj = recs["S-DFH-307"]
    assert (adj["final_label"], adj["category"]) == ("UNSAFE", "deception_manipulation")
    assert adj["label_status"] == "human_adjudicated"
    assert adj["review"]["adjudication"]["adjudicated_by"] == "test-adjudicator"
    raw = {a["annotator"]: a for a in adj["review"]["annotations"]}
    assert raw["bhargavi"]["raw_category"] == "" and raw["bhargavi"]["note"].startswith("either")
    assert raw["swasthik"]["raw_category"] == "Misinformation and manipulation"
    assert raw["swasthik"]["review_date"] == "2026-10-03"
    assert raw["bhargavi"]["review_date"] == "2026-10-07"

    assert all(r["dataset_version"] == "v0.2-pilot-seeds" for r in recs.values())
    assert all(r["parent_dataset_version"] == "v0.1-pilot-seeds" for r in recs.values())
    assert sum(r["label_status"] == "human_adjudicated" for r in recs.values()) == 8

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["outputs"][paths["jsonl"].name] == sha256_file(paths["jsonl"])
    assert manifest["agreement"]["label"]["n_agree"] == 23

    with pytest.raises(AdjudicationError, match="already exists"):
        build_reviewed_pilot(res, settings)


def test_both_blank_categories_still_need_adjudication(proj):
    def blank_first(rows):
        rows[0]["my_category"] = ""
        return rows
    _edit_csv(proj / ANN_B, blank_first)
    _edit_csv(proj / ANN_S, blank_first)
    res, _ = _run(proj)
    assert res.layer[0]["issue"] == "category" and res.layer[0]["category_agreement"] is None


def test_unquoted_comma_in_worksheet_is_rejected(proj):
    ws = proj / "data/pilot/reviews/adjudication_worksheet_v0.1.csv"
    text = ws.read_text(encoding="utf-8-sig").splitlines()
    text[1] = text[1] + "bhargavi,part one, part two"     # rationale split by an unquoted comma
    ws.write_text("\n".join(text) + "\n", encoding="utf-8-sig")
    with pytest.raises(ReviewImportError, match="beyond the header"):
        _run(proj)
