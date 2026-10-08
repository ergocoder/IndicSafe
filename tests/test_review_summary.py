"""Review-sheet summary (scripts/summarize_reviews.py) on small fake filled sheets."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

from generator.pilot_translation import REVIEW_COLUMNS
from generator.qc_pipeline import CODEMIX_REVIEW_COLUMNS
from generator.review_summary import match_sheets, render_markdown, summarize_reviews, write_summary

REPO = Path(__file__).resolve().parents[1]
KNOWN = ["P-A-1", "P-A-1L", "P-A-2", "P-A-2L", "P-A-3", "P-C-1", "P-C-1L", "P-C-2", "P-C-2L", "P-C-3", "P-T-1",
         "P-T-1L"]


def _csv(path: Path, columns: list[str], rows: list[dict], encoding: str = "utf-8-sig") -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", restval="")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding=encoding)


def tr(rid, native, latin, **cells):
    return {"review_id": rid, "seed_id": rid.split(":")[0], "native_prompt_id": native,
            "latin_prompt_id": latin, "native_text": f"text {rid}", **cells}


def cm(rid, native, latin, label, **cells):
    return {"review_id": rid, "seed_id": rid.split(":")[0], "native_prompt_id": native, "latin_prompt_id": latin,
            "intended_label": label, "native_text": f"cm {rid}", **cells}


@pytest.fixture
def run(tmp_path, project):
    run_dir = project / "data" / "pilot" / "translations" / "TRANSFORM_TEST"
    run_dir.mkdir(parents=True)
    (run_dir / "variants.jsonl").write_text("".join(json.dumps({"prompt_id": p}) + "\n" for p in KNOWN),
                                            encoding="utf-8")
    sheets = tmp_path / "filled"
    sheets.mkdir()
    _csv(sheets / "review_hi_alice.csv", REVIEW_COLUMNS, [
        tr("S-1:hi", "P-A-1", "P-A-1L", translation_adequacy_1to5="4", translation_fluency_1to5="5",
           intent_preserved_Y_N="Y", romanisation_natural_1to5="4"),
        tr("S-2:hi", "P-A-2", "P-A-2L", translation_adequacy_1to5="2", intent_preserved_Y_N="n",
           notes="meaning lost"),
        tr("S-3:hi", "P-A-3", ""),                                     # not rated yet
        {},                                                            # blank row (Excel leftover)
    ])
    _csv(sheets / "review_hi_bob.csv", REVIEW_COLUMNS, [
        tr("S-1:hi", "P-A-1", "P-ZZ-9", reviewer="bob", translation_adequacy_1to5="6",
           translation_fluency_1to5="x", intent_preserved_Y_N="maybe", romanisation_natural_1to5="3"),
    ])
    _csv(sheets / "review_codemix_hi.csv", CODEMIX_REVIEW_COLUMNS, [
        cm("S-1:hi:L1", "P-C-1", "P-C-1L", "UNSAFE", reviewer="carol", codemix_natural_1to3="1",
           intent_preserved_Y_N="N", notes="nobody talks like this"),
        cm("S-2:hi:L1", "P-C-2", "P-C-2L", "SAFE", reviewer="carol", codemix_natural_1to3="3",
           intent_preserved_Y_N="Y"),
        cm("S-3:hi:L2", "P-C-3", "", "UNSAFE", reviewer="carol", codemix_natural_1to3="2", intent_preserved_Y_N="Y"),
    ])
    (sheets / "review_mr.csv").write_bytes("review_id,notes\nS-1:mr,caf\xe9\n".encode("cp1252"))   # not UTF-8
    (sheets / "review_gu.csv").write_text("review_id;native_prompt_id\nS-1:gu;P-A-1\n", encoding="utf-8")
    _csv(sheets / "review_te.csv", REVIEW_COLUMNS, [
        tr("S-1:te", "P-T-1", "P-T-1L", reviewer="dev", translation_adequacy_1to5="5",
           translation_fluency_1to5="4", intent_preserved_Y_N="yes", romanisation_natural_1to5="2")])
    _csv(sheets / "review_codemix_te.csv", CODEMIX_REVIEW_COLUMNS, [])
    return run_dir, sheets


def test_match_sheets_by_prefix(run):
    _, sheets = run
    m = match_sheets(sheets, ["hi", "mr", "gu", "te"])
    assert [p.name for p in m[("hi", "translation")]] == ["review_hi_alice.csv", "review_hi_bob.csv"]
    assert [p.name for p in m[("hi", "codemix")]] == ["review_codemix_hi.csv"]   # not a review_hi* sheet
    assert ("gu", "codemix") not in m


def test_translation_stats_per_reviewer_and_partial_fills(settings, run):
    run_dir, sheets = run
    s = summarize_reviews(settings, run_dir, sheets)
    t = s["languages"]["hi"]["translation"]
    assert t["status"] == "ok"
    alice, bob, allr = t["by_reviewer"]["alice"], t["by_reviewer"]["bob"], t["overall"]
    assert (alice["n_rated"], alice["n_total"]) == (2, 3)                  # blank row skipped, S-3 unrated
    assert alice["translation_adequacy_1to5"] == {"mean": 3.0, "n": 2}
    assert alice["translation_fluency_1to5"] == {"mean": 5.0, "n": 1}
    assert alice["intent_preserved"] == {"pct_Y": 50.0, "n": 2}
    assert bob["translation_adequacy_1to5"]["n"] == 0 and bob["romanisation_natural_1to5"]["mean"] == 3.0
    assert (allr["n_rated"], allr["n_total"]) == (3, 4) and allr["romanisation_natural_1to5"] == {"mean": 3.5, "n": 2}
    w = "\n".join(s["warnings"])
    assert "translation_adequacy_1to5='6' outside 1-5" in w and "'x' is not a number" in w and "'MAYBE' is not Y/N" in w


def test_codemix_stats_with_unsafe_split(settings, run):
    run_dir, sheets = run
    c = summarize_reviews(settings, run_dir, sheets)["languages"]["hi"]["codemix"]
    o, u = c["overall"], c["unsafe_only"]["overall"]
    assert o["codemix_natural_1to3"] == {"mean": 2.0, "n": 3} and o["pct_rated_1_unnatural"] == pytest.approx(33.3)
    assert o["intent_preserved"]["pct_Y"] == pytest.approx(66.7)
    assert (u["n_total"], u["codemix_natural_1to3"]["mean"], u["pct_rated_1_unnatural"]) == (2, 1.5, 50.0)
    assert u["intent_preserved"]["pct_Y"] == 50.0 and set(c["by_reviewer"]) == {"carol"}


def test_lowest_rated_rows_carry_notes(settings, run):
    run_dir, sheets = run
    low = summarize_reviews(settings, run_dir, sheets)["languages"]["hi"]["lowest_rated"]
    assert low[0]["notes"] == "nobody talks like this" and low[0]["score_0to1"] == 0.0 and low[0]["label"] == "UNSAFE"
    assert low[1]["notes"] == "meaning lost" and low[1]["reviewer"] == "alice"
    assert len(low) == 6 and all(x["score_0to1"] <= y["score_0to1"] for x, y in zip(low, low[1:]))


def test_missing_bad_and_inapplicable_sheets_do_not_crash(settings, run):
    run_dir, sheets = run
    s = summarize_reviews(settings, run_dir, sheets)
    langs = s["languages"]
    assert langs["mr"]["translation"] == {"status": "unreadable", "files": ["review_mr.csv"]}
    assert langs["gu"]["translation"] == {"status": "unreadable", "files": ["review_gu.csv"]}
    assert langs["mr"]["codemix"] == {"status": "no_sheet"} and langs["gu"]["codemix"] == {"status": "no_sheet"}
    assert next(f for f in s["files"] if f["file"] == "review_mr.csv")["readable"] is False
    assert langs["te"]["codemix"] == {"status": "not_applicable"}
    assert langs["te"]["translation"]["overall"]["intent_preserved"] == {"pct_Y": 100.0, "n": 1}
    w = "\n".join(s["warnings"])
    assert "review_mr.csv: not UTF-8" in w and "review_gu.csv: missing column" in w
    assert "te: code-mix sheet(s) ['review_codemix_te.csv'] found" in w


def test_prompt_ids_checked_against_variants(settings, run):
    run_dir, sheets = run
    s = summarize_reviews(settings, run_dir, sheets)
    assert s["prompt_id_mismatches"] == 1 and s["variants_checked"]
    assert any("latin_prompt_id P-ZZ-9 is not in the run's variants.jsonl" in w for w in s["warnings"])
    (run_dir / "variants.jsonl").unlink()
    s2 = summarize_reviews(settings, run_dir, sheets)
    assert not s2["variants_checked"] and s2["prompt_id_mismatches"] == 0


def test_empty_sheets_folder(settings, run, tmp_path):
    run_dir, _ = run
    s = summarize_reviews(settings, run_dir, tmp_path / "nothing_here")
    assert all(e["translation"] == {"status": "no_sheet"} for e in s["languages"].values())
    assert "not found" in s["warnings"][0]
    assert "| hi | translation | no_sheet |" in render_markdown(s)


def test_outputs_and_cli(settings, run, monkeypatch, capsys):
    run_dir, sheets = run
    paths = write_summary(summarize_reviews(settings, run_dir, sheets), run_dir)
    md = paths["markdown"].read_text(encoding="utf-8")
    assert md.startswith("# Review summary — TRANSFORM_TEST") and "nobody talks like this" in md and "## Warnings" in md
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["languages"]["hi"]["translation"]["status"] == "ok"

    spec = importlib.util.spec_from_file_location("summarize_reviews", REPO / "scripts" / "summarize_reviews.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_settings", lambda: settings)
    assert mod.main(["--run", str(run_dir.relative_to(settings.project_root)), "--sheets", str(sheets)]) == 0
    out = capsys.readouterr().out
    assert "lang  sheet" in out and "prompt-id mismatches: 1" in out
    assert "alice" in out and "bob" in out          # two hi translation reviewers: one row each
    assert "carol" not in out                       # sole code-mix reviewer: only the ALL row
    assert mod.main(["--run", "data/nope", "--sheets", str(sheets)]) == 1
