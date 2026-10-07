"""Seed record schema.

A seed is a controlled starting prompt. Every later variant (translation,
romanisation, code-mix) will point back to its seed_id, so this record carries
the full provenance of where the prompt came from.

Label fields follow the project rule: `intended_label` is a provisional,
source-derived expectation; `final_label` is only ever set by human
annotation and is always None at import time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["SAFE", "UNSAFE", "AMBIGUOUS"]
SeedStatus = Literal["VALID", "REJECTED", "DUPLICATE"]
CategoryStatus = Literal["source_mapped", "unassigned", "human_assigned"]
SourceType = Literal["existing_dataset", "manual"]

SEED_SCHEMA_VERSION = "1.0"


class SeedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # identity
    seed_id: str = Field(pattern=r"^S-[A-Z0-9]+-[A-Za-z0-9_.-]+$")
    seed_version: int = Field(ge=1)
    schema_version: str = SEED_SCHEMA_VERSION

    # text
    prompt: str                     # normalised (NFC, whitespace-collapsed)
    original_text: str              # exactly as read from the source
    content_hash: str               # sha256 of dedup_key(prompt)

    # language / script (validated from the text, not assumed from the file)
    language: str
    script: str | None
    script_confidence: float = Field(ge=0.0, le=1.0)
    is_transliterated: bool = False

    # provenance
    source_type: SourceType
    source_dataset: str
    source_role: str
    source_file: str | None         # archive name for existing datasets
    source_member: str | None       # file inside the archive
    source_file_sha256: str | None
    source_reference: str           # stable id within the source (Index, id, unique_id, ...)
    source_line: int | None         # 1-based line / array position in the source file
    source_category: str | None     # category as named by the source
    source_metadata: dict[str, Any] = {}

    # taxonomy / labels
    category: str
    category_status: CategoryStatus
    intended_label: Label
    intended_label_basis: str
    label_status: Literal["provisional"] = "provisional"
    final_label: Label | None = None

    # validation
    seed_status: SeedStatus
    rejection_reasons: list[str] = []
    duplicate_of: str | None = None

    # versions / run
    taxonomy_version: str
    generator_version: str
    import_run_id: str
    imported_at: str

    @model_validator(mode="after")
    def _status_consistent(self) -> "SeedRecord":
        if self.final_label is not None:
            raise ValueError("final_label can only be set by human annotation, never at import")
        if self.seed_status == "VALID" and self.rejection_reasons:
            raise ValueError("VALID seed cannot carry rejection reasons")
        if self.seed_status == "REJECTED" and not self.rejection_reasons:
            raise ValueError("REJECTED seed must record at least one rejection reason")
        if (self.seed_status == "DUPLICATE") != (self.duplicate_of is not None):
            raise ValueError("duplicate_of must be set exactly when seed_status is DUPLICATE")
        return self


# Column order for flat (CSV) exports.
EXPORT_COLUMNS: list[str] = list(SeedRecord.model_fields)


# ------------------------------------------------------------------ variants
#
# A variant is any prompt in a seed's tree. The root is the identity copy of
# the seed (parent_prompt_id = None); every other variant names the variant it
# was derived from, and the transformation record that produced it. Both
# records are frozen: a parent is never modified by deriving a child.

VARIANT_SCHEMA_VERSION = "1.0"

TransformationStatus = Literal["SUCCEEDED", "VALIDATION_FAILED", "ERROR"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]
GenerationMethod = Literal["copy", "rule", "mt", "llm", "human", "reference", "mock"]
LabelConsistency = Literal["UNCHECKED", "CONSISTENT", "INCONSISTENT", "UNCERTAIN"]
QCStatus = Literal["PENDING", "PASS", "REVIEW", "FAIL"]


class HookResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hook: str
    status: ValidationStatus
    reason: str | None = None          # machine-readable code, e.g. "script_mismatch"
    details: dict[str, Any] = {}


class VariantRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # identity / lineage
    prompt_id: str = Field(pattern=r"^P-[A-Z0-9]+-[A-Za-z0-9_.-]+-[0-9a-f]+$")
    seed_id: str
    seed_version: int = Field(ge=1)
    parent_prompt_id: str | None    # None only for the identity root
    transformation_id: str
    lineage: list[str]              # ancestor prompt_ids, root first; [] for the root
    schema_version: str = VARIANT_SCHEMA_VERSION

    # text
    prompt: str
    content_hash: str

    # language condition (target condition of the transformation; script is
    # measured from the text, language is checked in Phase 3)
    language: str
    secondary_language: str | None = None
    script: str | None
    is_transliterated: bool
    code_mix_level: str | None = None   # set by the code-mixing engine (Phase 4)
    code_mix_ratio: float | None = None
    cmi: float | None = None
    mixing_method: str | None = None

    # labels: inherited from the seed, provisional until consistency check + annotation
    category: str
    intended_label: Label
    label_consistency_status: LabelConsistency = "UNCHECKED"
    final_label: Label | None = None

    # generation
    transformation_type: str
    generation_method: GenerationMethod
    generator_model: str | None
    generation_run_id: str

    # provenance (copied from the seed)
    source_type: SourceType
    source_dataset: str
    source_reference: str

    # validation (engine hooks) / QC (Phase 5)
    script_confidence: float = Field(ge=0.0, le=1.0)
    language_confidence: float | None = None
    validation_status: ValidationStatus
    validation_failures: list[str] = []
    qc_status: QCStatus = "PENDING"

    # versions
    taxonomy_version: str
    generator_version: str
    created_at: str

    @model_validator(mode="after")
    def _consistent(self) -> "VariantRecord":
        if self.final_label is not None:
            raise ValueError("final_label can only be set by human annotation, never at generation")
        root = self.parent_prompt_id is None
        if root != (self.transformation_type == "identity"):
            raise ValueError("parent_prompt_id must be None exactly for the identity root")
        if root and self.lineage:
            raise ValueError("the identity root has no lineage")
        if not root and (not self.lineage or self.lineage[-1] != self.parent_prompt_id):
            raise ValueError("lineage must end with parent_prompt_id")
        if (self.validation_status == "FAIL") != bool(self.validation_failures):
            raise ValueError("validation_failures must be non-empty exactly when validation_status is FAIL")
        return self


class TransformationRecord(BaseModel):
    """One requested transformation and what came of it (spec §11)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transformation_id: str = Field(pattern=r"^T-[0-9a-f]+$")
    transformation_type: str
    seed_id: str
    parent_prompt_id: str | None
    output_prompt_id: str | None        # None when status is ERROR
    parameters: dict[str, Any]          # fully resolved (defaults filled in)
    provider: str
    provider_version: str
    generator_model: str | None
    generation_method: GenerationMethod
    request_fingerprint: str            # sha256 of the canonical request; ids derive from it
    derived_seed: int                   # per-request random seed handed to the provider
    provider_metadata: dict[str, Any] = {}
    raw_output: str | None = None       # provider text before normalisation
    output_content_hash: str | None = None
    validation_hooks: list[str]
    validation_results: list[HookResult] = []
    status: TransformationStatus
    error: str | None = None
    engine_version: str
    generation_run_id: str
    timestamp: str
    schema_version: str = VARIANT_SCHEMA_VERSION

    @model_validator(mode="after")
    def _consistent(self) -> "TransformationRecord":
        if (self.status == "ERROR") != (self.output_prompt_id is None):
            raise ValueError("output_prompt_id must be None exactly when status is ERROR")
        if (self.status == "ERROR") != (self.error is not None):
            raise ValueError("error must be set exactly when status is ERROR")
        return self
