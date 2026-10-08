"""Load and validate the YAML configuration files.

Nothing about languages, sources, categories or thresholds is hard-coded in
Python: it all comes from configs/*.yaml and is validated here, so a broken
config fails at load time with a clear message instead of halfway through an
import.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"

Label = Literal["SAFE", "UNSAFE", "AMBIGUOUS"]
SourceRole = Literal[
    "safety_seed",
    "benign_control",
    "language_validation_support",
    "translation_support",
    "linguistic_support",
]


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- sources


class SourceConfig(_Strict):
    display_name: str
    archive: str
    member: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["jsonl", "json_array", "csv"]
    role: SourceRole
    seed_eligible: bool
    license: str
    id_prefix: str | None = None
    reference_field: str | None = None
    text_field: str | None = None
    source_category_field: str | None = None
    metadata_fields: list[str] = []
    ignored_fields: list[str] = []
    filters: dict[str, list[str]] = {}
    source_language: str | None = None
    default_intended_label: Label | None = None
    intended_label_basis: str | None = None

    @field_validator("archive", "member")
    @classmethod
    def _plain_filename(cls, v: str) -> str:
        # Registry entries name files, never paths: blocks "../" traversal.
        if not v or Path(v).name != v or v in {".", ".."}:
            raise ValueError(f"must be a plain file name, got {v!r}")
        return v

    @model_validator(mode="after")
    def _seed_fields_present(self) -> "SourceConfig":
        if self.seed_eligible:
            required = {
                "id_prefix": self.id_prefix,
                "reference_field": self.reference_field,
                "text_field": self.text_field,
                "source_language": self.source_language,
                "default_intended_label": self.default_intended_label,
                "intended_label_basis": self.intended_label_basis,
            }
            missing = [k for k, v in required.items() if not v]
            if missing:
                raise ValueError(f"seed-eligible source is missing: {', '.join(missing)}")
            if self.role not in ("safety_seed", "benign_control"):
                raise ValueError(f"role {self.role!r} cannot be seed_eligible")
        return self


class SourcesConfig(_Strict):
    sources_version: str
    raw_dir: str
    sources: dict[str, SourceConfig]


# -------------------------------------------------------------- languages


class LanguageConfig(_Strict):
    name: str
    role: Literal["reference", "target"]
    native_script: str
    romanized_script: str | None = None
    enabled: bool
    code_mix_partner: str | None = None
    code_mix_levels: list[str] = []
    notes: str | None = None


class ScriptConfig(_Strict):
    name: str
    ranges: list[tuple[int, int]]

    @field_validator("ranges")
    @classmethod
    def _ordered(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        for lo, hi in v:
            if lo > hi:
                raise ValueError(f"range start {lo:#x} > end {hi:#x}")
        return v


class CodeMixLevel(_Strict):
    name: str
    min_ratio: float = Field(ge=0.0, le=1.0)
    max_ratio: float = Field(ge=0.0, le=1.0)


class LanguagesConfig(_Strict):
    languages_version: str
    languages: dict[str, LanguageConfig]
    scripts: dict[str, ScriptConfig]
    code_mix_levels: dict[str, CodeMixLevel]

    @model_validator(mode="after")
    def _cross_refs(self) -> "LanguagesConfig":
        for code, lang in self.languages.items():
            for s in (lang.native_script, lang.romanized_script):
                if s and s not in self.scripts:
                    raise ValueError(f"language {code!r} uses unknown script {s!r}")
            if lang.code_mix_partner and lang.code_mix_partner not in self.languages:
                raise ValueError(f"language {code!r} has unknown code_mix_partner")
            for lvl in lang.code_mix_levels:
                if lvl not in self.code_mix_levels:
                    raise ValueError(f"language {code!r} uses unknown code-mix level {lvl!r}")
        levels = sorted(self.code_mix_levels.values(), key=lambda l: l.min_ratio)
        for a, b in zip(levels, levels[1:]):
            if abs(a.max_ratio - b.min_ratio) > 1e-9:
                raise ValueError("code-mix level bands must be contiguous and non-overlapping")
        return self

    def enabled_languages(self) -> list[str]:
        return [c for c, l in self.languages.items() if l.enabled]


# --------------------------------------------------------------- taxonomy


class Category(_Strict):
    category_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category_name: str
    definition: str
    inclusion_rules: list[str]
    exclusion_rules: list[str]
    examples: list[str] = []


class TaxonomyConfig(_Strict):
    taxonomy_version: str
    status: Literal["draft", "frozen"]
    labels: list[Label]
    unassigned_category: str
    categories: list[Category]
    source_category_mappings: dict[str, dict[str, str]] = {}

    @model_validator(mode="after")
    def _consistent(self) -> "TaxonomyConfig":
        ids = [c.category_id for c in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate category_id in taxonomy")
        if self.unassigned_category in ids:
            raise ValueError("unassigned_category must not be a real category")
        for src, mapping in self.source_category_mappings.items():
            for raw, cat in mapping.items():
                if cat not in ids:
                    raise ValueError(f"mapping {src}:{raw!r} -> unknown category {cat!r}")
        return self

    def category_ids(self) -> set[str]:
        return {c.category_id for c in self.categories}


# ------------------------------------------------------------- generation


class SeedValidationConfig(_Strict):
    min_chars: int = Field(ge=1)
    max_chars: int = Field(ge=1)
    min_script_confidence: float = Field(ge=0.0, le=1.0)
    reject_replacement_char: bool
    mojibake_markers: list[str]


class PilotConfig(_Strict):
    dataset_version: str
    target_size: int = Field(ge=1)
    quotas: dict[str, int]
    max_word_jaccard: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _quotas_sum(self) -> "PilotConfig":
        if sum(self.quotas.values()) != self.target_size:
            raise ValueError(
                f"pilot quotas sum to {sum(self.quotas.values())}, target_size is {self.target_size}"
            )
        return self


# Transformation types the engine knows about. Which of them may run is
# configuration (transformations.enabled); whether one is implemented is
# decided by the engine's registry.
TRANSFORMATION_TYPES = (
    "identity",
    "translation",
    "transliteration",
    "paraphrase",
    "code_mixing",
    "noisy_spelling",
    "regional_variation",
)
ProviderType = Literal["local_mt", "llm_api", "rule_based", "reference"]


class TransformationsConfig(_Strict):
    enabled: list[str]
    disabled: list[str] = []

    @model_validator(mode="after")
    def _known(self) -> "TransformationsConfig":
        for t in [*self.enabled, *self.disabled]:
            if t not in TRANSFORMATION_TYPES:
                raise ValueError(f"unknown transformation type {t!r}")
        both = set(self.enabled) & set(self.disabled)
        if both:
            raise ValueError(f"transformation(s) both enabled and disabled: {sorted(both)}")
        if "identity" not in self.enabled:
            raise ValueError("'identity' must be enabled: it creates the root variant of every chain")
        return self


class LengthRatioConfig(_Strict):
    min: float = Field(gt=0.0)
    max: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "LengthRatioConfig":
        if self.min >= self.max:
            raise ValueError("length_ratio.min must be < length_ratio.max")
        return self


class TransformationEngineConfig(_Strict):
    engine_version: str
    id_hash_chars: int = Field(ge=8, le=64)
    validation_hooks: dict[str, list[str]]
    length_ratio: LengthRatioConfig

    @model_validator(mode="after")
    def _hooks(self) -> "TransformationEngineConfig":
        if "default" not in self.validation_hooks:
            raise ValueError("validation_hooks needs a 'default' entry")
        for key in self.validation_hooks:
            if key != "default" and key not in TRANSFORMATION_TYPES:
                raise ValueError(f"validation_hooks for unknown transformation type {key!r}")
        return self

    def hooks_for(self, transformation_type: str) -> list[str]:
        return self.validation_hooks.get(transformation_type, self.validation_hooks["default"])


class ProviderConfig(_Strict):
    type: ProviderType
    enabled: bool
    target_languages: list[str] = []
    model: str | None = None
    options: dict[str, Any] = {}


class AdapterConfig(_Strict):
    default_provider: str | None
    providers: dict[str, ProviderConfig] = {}
    selection: str | None = None

    @model_validator(mode="after")
    def _default_exists(self) -> "AdapterConfig":
        if self.default_provider is not None and self.default_provider not in self.providers:
            raise ValueError(f"default_provider {self.default_provider!r} is not in providers")
        return self


class ParaphraseConfig(AdapterConfig):
    languages: list[str] = []


class CodeMixingConfig(AdapterConfig):
    levels: list[str] = Field(min_length=1)   # levels generated per language, e.g. [L1, L2]


class ScriptQCConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    native_min_share: float = Field(ge=0.0, le=1.0)
    romanized_min_share: float = Field(ge=0.0, le=1.0)


class LanguageQCConfig(_Strict):
    detector: Literal["lingua_markers"]
    candidates: list[str] = Field(min_length=2)
    min_marker_tokens: int = Field(ge=1)
    pass_confidence: float = Field(ge=0.0, le=1.0)
    review_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "LanguageQCConfig":
        if self.review_confidence > self.pass_confidence:
            raise ValueError("qc.language.review_confidence must be <= pass_confidence")
        return self


class CodeMixQCConfig(_Strict):
    tolerance: float = Field(ge=0.0, le=0.5)


class DuplicateQCConfig(_Strict):
    near_dup_char3_jaccard: float = Field(gt=0.0, le=1.0)
    near_dup_embedding_cosine: float = Field(gt=0.0, le=1.0)


class SemanticQCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    model: str
    revision: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: int = Field(default=16, ge=1)
    cache_dir: str | None = None
    pass_: float = Field(alias="pass", ge=0.0, le=1.0)
    review: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "SemanticQCConfig":
        if self.review > self.pass_:
            raise ValueError("qc.semantic.review must be <= pass")
        return self


class HumanReviewQCConfig(_Strict):
    pilot_review_fraction: float = Field(ge=0.0, le=1.0)
    pass_sample_fraction: float = Field(ge=0.0, le=1.0)
    codemix_sheet_rows: int = Field(ge=1)
    codemix_sheet_min_unsafe: int = Field(ge=0)
    harmful_intent_top_n: int = Field(ge=1)


class QCConfig(BaseModel):
    # extra="allow": only the parts used so far are typed; the remaining QC
    # design blocks are validated when their phase is implemented.
    model_config = ConfigDict(extra="allow", frozen=True)

    script: ScriptQCConfig
    language: LanguageQCConfig
    code_mix: CodeMixQCConfig
    duplicate: DuplicateQCConfig
    semantic: SemanticQCConfig
    human_review: HumanReviewQCConfig


class GenerationConfig(BaseModel):
    # extra="allow": design-only blocks may be added to the file before their
    # phase is implemented.
    model_config = ConfigDict(extra="allow", frozen=True)

    generator_version: str
    random_seed: int
    seed_validation: SeedValidationConfig
    pilot: PilotConfig
    transformations: TransformationsConfig
    transformation_engine: TransformationEngineConfig
    translation: AdapterConfig
    transliteration: AdapterConfig
    paraphrase: ParaphraseConfig
    code_mixing: CodeMixingConfig
    qc: QCConfig


# --------------------------------------------------------------- settings


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    project_root: Path
    config_dir: Path
    sources: SourcesConfig
    languages: LanguagesConfig
    taxonomy: TaxonomyConfig
    generation: GenerationConfig
    config_hashes: dict[str, str]

    @property
    def raw_dir(self) -> Path:
        return resolve_inside(self.project_root, self.sources.raw_dir)

    @model_validator(mode="after")
    def _cross_file(self) -> "Settings":
        for sid, src in self.sources.sources.items():
            if src.seed_eligible and src.source_language not in self.languages.languages:
                raise ValueError(f"source {sid!r} language {src.source_language!r} not in languages.yaml")
        for sid in self.generation.pilot.quotas:
            src = self.sources.sources.get(sid)
            if src is None or not src.seed_eligible:
                raise ValueError(f"pilot quota for {sid!r}, which is not a seed-eligible source")
        for sid in self.taxonomy.source_category_mappings:
            if sid not in self.sources.sources:
                raise ValueError(f"taxonomy mapping for unknown source {sid!r}")
        gen = self.generation
        langs = self.languages.languages
        for kind in ("translation", "transliteration", "paraphrase", "code_mixing"):
            for pname, p in getattr(gen, kind).providers.items():
                for lang in p.target_languages:
                    if lang not in langs:
                        raise ValueError(f"{kind} provider {pname!r} targets unknown language {lang!r}")
        for lang in gen.paraphrase.languages:
            if lang not in langs:
                raise ValueError(f"paraphrase language {lang!r} not in languages.yaml")
        for pname, p in gen.code_mixing.providers.items():
            for lang in p.target_languages:
                missing = [lv for lv in gen.code_mixing.levels if lv not in langs[lang].code_mix_levels]
                if missing:
                    raise ValueError(f"code_mixing levels {missing} not allowed for {lang!r} in languages.yaml")
        for lang in gen.qc.language.candidates:
            if lang not in langs:
                raise ValueError(f"qc.language candidate {lang!r} not in languages.yaml")
        return self


def resolve_inside(root: Path, relative: str | Path) -> Path:
    """Resolve `relative` under `root`, refusing anything that escapes it."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ConfigError(f"path {relative!s} resolves outside {root}")
    return candidate


_FILES = {
    "sources": ("sources.yaml", SourcesConfig),
    "languages": ("languages.yaml", LanguagesConfig),
    "taxonomy": ("taxonomy.yaml", TaxonomyConfig),
    "generation": ("generation.yaml", GenerationConfig),
}


def _read_yaml(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    data = path.read_bytes()
    try:
        parsed = yaml.safe_load(data)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path.name}: {e}") from e
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path.name} must contain a mapping at top level")
    return parsed, hashlib.sha256(data).hexdigest()


def load_settings(project_root: Path | None = None, config_dir: Path | None = None) -> Settings:
    """Load and validate all config files. Raises ConfigError on any problem."""
    root = (project_root or PROJECT_ROOT).resolve()
    cdir = (config_dir or root / "configs").resolve()
    parsed: dict[str, BaseModel] = {}
    hashes: dict[str, str] = {}
    for key, (fname, model) in _FILES.items():
        data, digest = _read_yaml(cdir / fname)
        try:
            parsed[key] = model.model_validate(data)
        except ValueError as e:
            raise ConfigError(f"{fname}: {e}") from e
        hashes[fname] = digest
    try:
        return Settings(project_root=root, config_dir=cdir, config_hashes=hashes, **parsed)
    except ValueError as e:
        raise ConfigError(f"cross-file config check failed: {e}") from e
