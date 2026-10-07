from pathlib import Path

import pytest
import yaml

from backend.config import ConfigError, load_settings, resolve_inside


def _edit(project: Path, fname: str, fn) -> None:
    p = project / "configs" / fname
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(data)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_real_repo_configs_load():
    s = load_settings()
    assert set(s.languages.enabled_languages()) == {"en", "hi", "mr", "gu"}
    assert s.taxonomy.labels == ["SAFE", "UNSAFE", "AMBIGUOUS"]
    assert s.taxonomy.status == "frozen" and s.taxonomy.taxonomy_version == "1.0"
    assert set(s.config_hashes) == {"sources.yaml", "languages.yaml", "taxonomy.yaml", "generation.yaml"}


def test_real_registry_roles():
    s = load_settings()
    seedable = {k for k, v in s.sources.sources.items() if v.seed_eligible}
    assert seedable == {"nichehazardqa", "data_for_hub", "dataset_10k"}
    # support files may never supply seeds or labels
    for k, v in s.sources.sources.items():
        if v.role.endswith("_support"):
            assert not v.seed_eligible and v.default_intended_label is None, k


def test_language_script_config(settings):
    langs = settings.languages.languages
    assert langs["hi"].native_script == "Deva" and langs["hi"].romanized_script == "Latn"
    assert langs["mr"].native_script == "Deva"
    assert langs["hi"].code_mix_partner == "en"
    assert settings.languages.code_mix_levels["L1"].min_ratio == settings.languages.code_mix_levels["L0"].max_ratio


def test_missing_config_file(project):
    (project / "configs" / "taxonomy.yaml").unlink()
    with pytest.raises(ConfigError, match="missing config file"):
        load_settings(project_root=project)


def test_invalid_yaml(project):
    (project / "configs" / "languages.yaml").write_text("languages: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_settings(project_root=project)


def test_pilot_quotas_must_sum_to_target(project):
    _edit(project, "generation.yaml", lambda d: d["pilot"].__setitem__("target_size", 999))
    with pytest.raises(ConfigError, match="quotas sum"):
        load_settings(project_root=project)


def test_unknown_script_rejected(project):
    _edit(project, "languages.yaml", lambda d: d["languages"]["hi"].__setitem__("native_script", "Xxxx"))
    with pytest.raises(ConfigError, match="unknown script"):
        load_settings(project_root=project)


def test_code_mix_bands_must_be_contiguous(project):
    _edit(project, "languages.yaml", lambda d: d["code_mix_levels"]["L2"].__setitem__("min_ratio", 0.25))
    with pytest.raises(ConfigError, match="contiguous"):
        load_settings(project_root=project)


def test_mapping_to_unknown_category_rejected(project):
    _edit(project, "taxonomy.yaml",
          lambda d: d["source_category_mappings"]["nichehazardqa"].__setitem__("X", "not_a_category"))
    with pytest.raises(ConfigError, match="unknown category"):
        load_settings(project_root=project)


def test_duplicate_category_rejected(project):
    _edit(project, "taxonomy.yaml", lambda d: d["categories"].append(dict(d["categories"][0])))
    with pytest.raises(ConfigError, match="duplicate category_id"):
        load_settings(project_root=project)


def test_archive_path_traversal_rejected(project):
    _edit(project, "sources.yaml",
          lambda d: d["sources"]["nichehazardqa"].__setitem__("archive", "../../evil.zip"))
    with pytest.raises(ConfigError, match="plain file name"):
        load_settings(project_root=project)


def test_seed_source_missing_text_field_rejected(project):
    _edit(project, "sources.yaml", lambda d: d["sources"]["nichehazardqa"].pop("text_field"))
    with pytest.raises(ConfigError, match="text_field"):
        load_settings(project_root=project)


def test_support_source_cannot_be_seed_eligible(project):
    def f(d):
        s = d["sources"]["lid_test"]
        s.update(seed_eligible=True, id_prefix="LID", reference_field="a", text_field="b",
                 source_language="hi", default_intended_label="SAFE", intended_label_basis="x")
    _edit(project, "sources.yaml", f)
    with pytest.raises(ConfigError, match="cannot be seed_eligible"):
        load_settings(project_root=project)


def test_raw_dir_outside_project_rejected(project):
    _edit(project, "sources.yaml", lambda d: d.__setitem__("raw_dir", "../elsewhere"))
    s = load_settings(project_root=project)
    with pytest.raises(ConfigError, match="outside"):
        _ = s.raw_dir


def test_resolve_inside(tmp_path):
    assert resolve_inside(tmp_path, "a/b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(ConfigError):
        resolve_inside(tmp_path, "../x")
