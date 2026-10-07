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
