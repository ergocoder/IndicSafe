import re
import zipfile

import pytest
from pydantic import ValidationError

from generator.provenance import SourceIntegrityError, build_run_manifest, sha256_file, verify_source
from generator.schemas import SeedRecord
from tests.conftest import by_id


def _raw_hashes(project):
    return {p.name: sha256_file(p) for p in sorted((project / "data" / "raw").iterdir())}


def test_raw_files_unchanged_by_import_and_export(manager, project):
    before = _raw_hashes(project)
    r = manager.import_sources()
    manager.export_pool(r, project / "data" / "processed")
    manager.export_pilot(manager.select_pilot(r.seeds), r, project / "data" / "pilot")
    assert _raw_hashes(project) == before


def test_checksum_mismatch_stops_import(manager, settings, project):
    src = settings.sources.sources["nichehazardqa"]
    with zipfile.ZipFile(project / "data" / "raw" / src.archive, "a") as zf:
        zf.writestr("extra.txt", "tampered")
    with pytest.raises(SourceIntegrityError, match="checksum mismatch"):
        manager.import_sources()


def test_missing_raw_file(settings, project):
    (project / "data" / "raw" / settings.sources.sources["dataset_10k"].archive).unlink()
    with pytest.raises(SourceIntegrityError, match="not found"):
        verify_source(settings, "dataset_10k")


def test_wrong_member_name(settings, project):
    from generator.source_readers import SourceFormatError, iter_records

    src = settings.sources.sources["nichehazardqa"].model_copy(update={"member": "other.json"})
    with pytest.raises(SourceFormatError, match="expected member"):
        list(iter_records(project / "data" / "raw" / src.archive, src))


def test_run_manifest_contents(imported, settings):
    m = build_run_manifest(run_id=imported.run_id, run_type="t", started_at=imported.started_at,
                           settings=settings, source_checksums=imported.source_checksums)
    assert re.fullmatch(r"IMPORT_\d{8}T\d{6}Z_[0-9a-f]{8}", imported.run_id)
    assert m["random_seed"] == settings.generation.random_seed
    assert m["config_sha256"] == settings.config_hashes
    assert m["source_files"]["nichehazardqa"]["sha256"] == settings.sources.sources["nichehazardqa"].sha256
    assert m["environment"]["python"]
    assert "pydantic" in m["environment"]["packages"]
    assert set(m["git"]) == {"commit", "dirty"}   # both None outside a git repo


def test_git_state_outside_repo(tmp_path):
    from generator.provenance import git_state

    assert git_state(tmp_path) == {"commit": None, "dirty": None}


# ----------------------------------------------------------- schema rules


def _valid(imported):
    return by_id(imported, "S-NHQA-1").model_dump()


def test_schema_rejects_final_label_at_import(imported):
    d = _valid(imported) | {"final_label": "UNSAFE"}
    with pytest.raises(ValidationError, match="human annotation"):
        SeedRecord.model_validate(d)


@pytest.mark.parametrize("update,msg", [
    ({"rejection_reasons": ["x"]}, "VALID seed"),
    ({"seed_status": "REJECTED"}, "at least one"),
    ({"seed_status": "DUPLICATE"}, "duplicate_of"),
    ({"duplicate_of": "S-NHQA-2"}, "duplicate_of"),
    ({"intended_label": "MAYBE"}, "intended_label"),
    ({"seed_id": "not-an-id"}, "seed_id"),
    ({"unexpected": 1}, "unexpected"),
])
def test_schema_consistency_rules(imported, update, msg):
    with pytest.raises(ValidationError, match=msg):
        SeedRecord.model_validate(_valid(imported) | update)
