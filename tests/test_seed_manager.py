import csv
import json

import pytest

from backend.config import ConfigError
from generator.seed_manager import (
    ExportError,
    PilotSelectionError,
    SeedManager,
    load_seeds_jsonl,
)
from generator.text_utils import word_jaccard
from tests.conftest import build_project, by_id

# ------------------------------------------------------------------ import


def test_import_counts(imported):
    st = imported.stats
    assert st["nichehazardqa"].rows_read == 11          # 12 lines, 1 malformed
    assert st["nichehazardqa"].valid == 6
    assert st["nichehazardqa"].duplicate == 1
    assert st["nichehazardqa"].rejected == 4
    assert st["data_for_hub"].valid == 3 and st["data_for_hub"].duplicate == 1
    assert st["dataset_10k"].rows_read == 5
    assert st["dataset_10k"].filtered_out == 2           # Hindi row + maths category
    assert st["dataset_10k"].valid == 3


def test_parse_failure_recorded_with_line(imported):
    fails = imported.stats["nichehazardqa"].parse_failures
    assert len(fails) == 1 and fails[0]["line"] == 5 and "invalid JSON" in fails[0]["error"]


@pytest.mark.parametrize("seed_id,reason", [
    ("S-NHQA-6", "too_short"),
    ("S-NHQA-7", "suspected_mojibake"),
    ("S-NHQA-9", "script_mismatch"),
    ("S-NHQA-10", "missing_prompt_text"),
])
def test_rejections_are_kept_with_reason(imported, seed_id, reason):
    s = by_id(imported, seed_id)
    assert s.seed_status == "REJECTED"
    assert reason in s.rejection_reasons


def test_missing_text_also_flags_empty(imported):
    s = by_id(imported, "S-NHQA-10")
    assert s.rejection_reasons == ["empty_prompt", "missing_prompt_text"]


def test_devanagari_in_english_source_detected(imported):
    s = by_id(imported, "S-NHQA-9")
    assert s.script == "Deva" and s.language == "en"


def test_exact_duplicate_within_source(imported):
    s = by_id(imported, "S-NHQA-8")
    assert s.seed_status == "DUPLICATE" and s.duplicate_of == "S-NHQA-1"


def test_exact_duplicate_across_sources(imported):
    s = by_id(imported, "S-DFH-3")
    assert s.seed_status == "DUPLICATE" and s.duplicate_of == "S-NHQA-3"


def test_every_row_is_kept(imported):
    # 11 parsed NHQA rows + 4 DFH + 3 D10K after filters: nothing silently dropped
    assert len(imported.seeds) == 11 + 4 + 3


def test_category_mapping(imported):
    assert by_id(imported, "S-NHQA-1").category == "hate_discrimination"
    assert by_id(imported, "S-NHQA-1").category_status == "source_mapped"
    unmapped = by_id(imported, "S-NHQA-11")
    assert unmapped.category == "unassigned" and unmapped.category_status == "unassigned"
    assert by_id(imported, "S-DFH-1").category == "unassigned"          # topics are not harm categories
    assert by_id(imported, "S-DFH-1").source_category == "Social Sciences"
    assert by_id(imported, "S-D10K-020000000301").category == "benign_educational"


def test_labels_are_provisional_never_gold(imported):
    for s in imported.seeds:
        assert s.label_status == "provisional"
        assert s.final_label is None
        assert s.intended_label_basis
    assert by_id(imported, "S-NHQA-1").intended_label == "UNSAFE"
    assert by_id(imported, "S-D10K-010000000101").intended_label == "SAFE"


def test_provenance_fields(imported, settings):
    s = by_id(imported, "S-DFH-2")
    src = settings.sources.sources["data_for_hub"]
    assert s.source_dataset == "data_for_hub"
    assert s.source_reference == "2"
    assert s.source_line == 2
    assert s.source_file == src.archive and s.source_member == src.member
    assert s.source_file_sha256 == src.sha256
    assert s.source_metadata == {"subtopic": "Pharmacology"}
    assert s.import_run_id == imported.run_id
    assert s.taxonomy_version == settings.taxonomy.taxonomy_version
    assert s.generator_version == settings.generation.generator_version


def test_conversations_not_imported(imported):
    dumped = json.dumps([s.model_dump() for s in imported.seeds])
    assert "CONVERSATION-TEXT-MUST-NOT-BE-IMPORTED" not in dumped


def test_nan_becomes_null(imported):
    s = by_id(imported, "S-D10K-010000000101")
    assert s.source_metadata["domain"] is None
    assert s.source_metadata["expected"] == "Vitruvius"


def test_seed_ids_and_content_are_deterministic(manager):
    a, b = manager.import_sources(), manager.import_sources()
    strip = {"import_run_id", "imported_at"}
    assert [s.model_dump(exclude=strip) for s in a.seeds] == [s.model_dump(exclude=strip) for s in b.seeds]


def test_unknown_or_support_source_refused(manager):
    with pytest.raises(ConfigError, match="unknown source"):
        manager.import_sources(["nope"])
    with pytest.raises(ConfigError, match="not seed-eligible"):
        manager.import_sources(["lid_test"])


def test_import_subset_of_sources(manager):
    r = manager.import_sources(["dataset_10k"])
    assert {s.source_dataset for s in r.seeds} == {"dataset_10k"}


# --------------------------------------------------------------- selection


def test_pilot_quotas_and_validity(manager, imported, settings):
    pilot = manager.select_pilot(imported.seeds)
    assert len(pilot) == settings.generation.pilot.target_size
    counts = {sid: sum(s.source_dataset == sid for s in pilot) for sid in settings.generation.pilot.quotas}
    assert counts == dict(settings.generation.pilot.quotas)
    assert all(s.seed_status == "VALID" for s in pilot)
    assert len({s.seed_id for s in pilot}) == len(pilot)


def test_pilot_is_deterministic(manager, imported):
    first = [s.seed_id for s in manager.select_pilot(imported.seeds)]
    again = [s.seed_id for s in manager.select_pilot(manager.import_sources().seeds)]
    assert first == again


def test_pilot_diversity_guard(manager, imported, settings):
    pilot = manager.select_pilot(imported.seeds)
    ids = {s.seed_id for s in pilot}
    assert not {"S-NHQA-1", "S-NHQA-2"} <= ids      # template near-duplicates
    limit = settings.generation.pilot.max_word_jaccard
    for i, a in enumerate(pilot):
        for b in pilot[i + 1:]:
            assert word_jaccard(a.prompt, b.prompt) < limit


def test_pilot_spreads_over_categories(manager, imported):
    pilot = manager.select_pilot(imported.seeds)
    nhqa_cats = [s.source_category for s in pilot if s.source_dataset == "nichehazardqa"]
    assert len(set(nhqa_cats)) == len(nhqa_cats)


def test_pilot_shortfall_raises(tmp_path):
    from backend.config import load_settings

    root = build_project(tmp_path / "p", quotas={"nichehazardqa": 1, "data_for_hub": 1, "dataset_10k": 9})
    mgr = SeedManager(load_settings(project_root=root))
    with pytest.raises(PilotSelectionError, match="dataset_10k"):
        mgr.select_pilot(mgr.import_sources().seeds)


# ------------------------------------------------------------------ export


def test_export_pool_and_pilot(manager, imported, project):
    pool = manager.export_pool(imported, project / "data" / "processed")
    assert len(load_seeds_jsonl(pool["pool"])) == len(imported.seeds)
    report = json.loads(pool["report"].read_text(encoding="utf-8"))
    assert report["per_source"]["nichehazardqa"]["parse_failures"][0]["line"] == 5
    assert report["totals"] == {"VALID": 12, "DUPLICATE": 2, "REJECTED": 4}

    pilot = manager.select_pilot(imported.seeds)
    paths = manager.export_pilot(pilot, imported, project / "data" / "pilot")
    round_trip = load_seeds_jsonl(paths["jsonl"])
    assert [s.model_dump() for s in round_trip] == [s.model_dump() for s in pilot]

    with paths["csv"].open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(pilot)
    assert rows[0]["seed_id"] == pilot[0].seed_id
    assert json.loads(rows[0]["source_metadata"]) == pilot[0].source_metadata

    m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert m["record_count"] == len(pilot)
    assert m["distributions"]["intended_label"] == {"SAFE": 2, "UNSAFE": 5}
    assert set(m["outputs"]) == {paths["jsonl"].name, paths["csv"].name}
    assert m["selection_fingerprint"]


def test_export_refuses_raw_dir(manager, imported, project):
    with pytest.raises(ExportError, match="read-only"):
        manager.export_pool(imported, project / "data" / "raw" / "sub")


def test_export_refuses_outside_project(manager, imported):
    with pytest.raises(ConfigError, match="outside"):
        manager.export_pool(imported, "../../escape")


def test_pilot_export_protects_existing_selection(manager, imported, project):
    out = project / "data" / "pilot"
    pilot = manager.select_pilot(imported.seeds)
    manager.export_pilot(pilot, imported, out)
    manager.export_pilot(pilot, imported, out)                    # same selection: fine
    with pytest.raises(ExportError, match="different selection"):
        manager.export_pilot(pilot[:-1], imported, out)
    manager.export_pilot(pilot[:-1], imported, out, force=True)   # deliberate replace


# ------------------------------------------------------------ manual seeds


def _write_manual(project, text):
    p = project / "data" / "manual" / "seeds.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_manual_csv_import(manager, project):
    p = _write_manual(project, (
        "prompt,language,script,category,intended_label,source_reference,author\n"
        "Mujhe bata do ki exam ki tayari kaise karein,hi,Latn,benign_everyday,SAFE,team-001,annotator_a\n"
        "Kal ka mausam kaisa rahega,hi,Latn,benign_everyday,safe,,annotator_b\n"
        "This row has a bad label,en,Latn,benign_everyday,MAYBE,team-003,x\n"
        "Unknown category row here,en,Latn,made_up,SAFE,team-004,x\n"
    ))
    r = manager.import_sources([], manual_csv=p)
    st = r.stats["manual:seeds.csv"]
    assert st.valid == 2
    assert [e["line"] for e in st.row_errors] == [4, 5]
    s = r.seeds[0]
    assert s.seed_id.startswith("S-MAN-")
    assert s.source_type == "manual" and s.language == "hi" and s.script == "Latn"
    assert s.is_transliterated is True
    assert s.category_status == "human_assigned"
    assert s.source_reference == "team-001" and s.source_metadata["author"] == "annotator_a"
    assert r.seeds[1].source_reference == "line3" and r.seeds[1].intended_label == "SAFE"
    assert "manual:seeds.csv" in r.source_checksums


def test_manual_csv_script_mismatch(manager, project):
    p = _write_manual(project, "prompt,language,script,category,intended_label\n"
                               "यह हिंदी में लिखा गया एक वाक्य है,hi,Latn,benign_everyday,SAFE\n")
    s = manager.import_sources([], manual_csv=p).seeds[0]
    assert s.seed_status == "REJECTED" and "script_mismatch" in s.rejection_reasons


def test_manual_csv_missing_columns(manager, project):
    p = _write_manual(project, "prompt,language\nhello there friend,en\n")
    with pytest.raises(ConfigError, match="missing required columns"):
        manager.import_sources([], manual_csv=p)


def test_manual_csv_path_checks(manager, project):
    with pytest.raises(ConfigError, match="outside"):
        manager.import_sources([], manual_csv="../../elsewhere.csv")
    raw_csv = project / "data" / "raw" / "seeds.csv"
    raw_csv.write_text("prompt,language,script,category,intended_label\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="data/raw"):
        manager.import_sources([], manual_csv=raw_csv)
