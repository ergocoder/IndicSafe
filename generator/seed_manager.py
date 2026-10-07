r"""Seed Manager: import, validate, de-duplicate, select and export seed prompts.

Flow:
    registered raw sources (read-only zips)   manual seed CSV (optional)
                     \                        /
                      import + validate each row
                                 |
                      exact-duplicate marking
                                 |
                   seed pool (every row kept, with status)
                                 |
                 deterministic stratified pilot selection
                                 |
                   pilot export (JSONL + CSV + manifest)

Nothing is deleted: rejected and duplicate rows stay in the pool with their
reasons. No label is invented: intended_label comes from the source registry
and is marked provisional; final_label stays empty until human annotation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backend.config import ConfigError, Settings, SourceConfig, resolve_inside
from generator.provenance import (
    build_run_manifest,
    iso,
    new_run_id,
    sha256_file,
    source_path,
    utc_now,
    verify_source,
)
from generator.schemas import EXPORT_COLUMNS, SeedRecord
from generator.source_readers import ParseFailure, iter_records
from generator.text_utils import (
    content_hash,
    dominant_script,
    has_control_chars,
    normalize_text,
    word_jaccard,
    word_set,
)

MANUAL_REQUIRED_COLUMNS = ("prompt", "language", "script", "category", "intended_label")


class PilotSelectionError(RuntimeError):
    """The pool cannot satisfy the configured pilot quotas."""


class ExportError(RuntimeError):
    """An export would write somewhere forbidden or silently change a release."""


@dataclass
class SourceStats:
    rows_read: int = 0
    filtered_out: int = 0
    valid: int = 0
    rejected: int = 0
    duplicate: int = 0
    rejection_reasons: Counter = field(default_factory=Counter)
    parse_failures: list[dict] = field(default_factory=list)
    row_errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "filtered_out": self.filtered_out,
            "valid": self.valid,
            "rejected": self.rejected,
            "duplicate": self.duplicate,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "parse_failures": self.parse_failures,
            "row_errors": self.row_errors,
        }


@dataclass
class ImportResult:
    run_id: str
    started_at: datetime
    seeds: list[SeedRecord]
    stats: dict[str, SourceStats]
    source_checksums: dict[str, str]

    def valid_seeds(self) -> list[SeedRecord]:
        return [s for s in self.seeds if s.seed_status == "VALID"]


class SeedManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._scripts = {k: v.ranges for k, v in settings.languages.scripts.items()}
        self._valid_categories = settings.taxonomy.category_ids()

    # ------------------------------------------------------------ import

    def import_sources(
        self,
        source_ids: Iterable[str] | None = None,
        manual_csv: Path | None = None,
    ) -> ImportResult:
        registry = self.settings.sources.sources
        ids = list(source_ids) if source_ids is not None else [
            sid for sid, s in registry.items() if s.seed_eligible
        ]
        for sid in ids:
            if sid not in registry:
                raise ConfigError(f"unknown source {sid!r}")
            if not registry[sid].seed_eligible:
                raise ConfigError(f"source {sid!r} (role {registry[sid].role}) is not seed-eligible")

        started = utc_now()
        run_id = new_run_id("import", started, self.settings)
        seeds: list[SeedRecord] = []
        stats: dict[str, SourceStats] = {}
        checksums: dict[str, str] = {}

        for sid in ids:
            checksums[sid] = verify_source(self.settings, sid)
            stats[sid] = SourceStats()
            seeds.extend(self._import_one(sid, registry[sid], run_id, started, stats[sid]))

        if manual_csv is not None:
            key = f"manual:{Path(manual_csv).name}"
            stats[key] = SourceStats()
            seeds.extend(self._import_manual(manual_csv, run_id, started, stats[key], checksums))

        seeds = self._mark_duplicates(seeds, stats)
        return ImportResult(run_id, started, seeds, stats, checksums)

    def _import_one(
        self, sid: str, src: SourceConfig, run_id: str, started: datetime, st: SourceStats
    ) -> list[SeedRecord]:
        out: list[SeedRecord] = []
        mapping = self.settings.taxonomy.source_category_mappings.get(sid, {})
        for item in iter_records(source_path(self.settings, src), src):
            if isinstance(item, ParseFailure):
                st.parse_failures.append({"line": item.line, "error": item.error})
                continue
            st.rows_read += 1
            row = item.data
            if any(str(row.get(f)) not in allowed for f, allowed in src.filters.items()):
                st.filtered_out += 1
                continue

            reasons: list[str] = []
            ref = row.get(src.reference_field)
            if ref is None or str(ref).strip() == "":
                reasons.append("missing_source_reference")
                ref = f"line{item.line}"
            ref = str(ref).strip()

            original = row.get(src.text_field)
            if not isinstance(original, str):
                reasons.append("missing_prompt_text")
                original = "" if original is None else str(original)

            source_cat = row.get(src.source_category_field) if src.source_category_field else None
            source_cat = None if source_cat is None else str(source_cat)
            category = mapping.get(source_cat) if source_cat is not None else None

            out.append(self._build(
                seed_id=f"S-{src.id_prefix}-{ref}",
                original=original,
                language=src.source_language,
                expected_script=self.settings.languages.languages[src.source_language].native_script,
                source_type="existing_dataset",
                source_dataset=sid,
                source_role=src.role,
                source_file=src.archive,
                source_member=src.member,
                source_file_sha256=src.sha256,
                source_reference=ref,
                source_line=item.line,
                source_category=source_cat,
                source_metadata={f: row.get(f) for f in src.metadata_fields},
                category=category or self.settings.taxonomy.unassigned_category,
                category_status="source_mapped" if category else "unassigned",
                intended_label=src.default_intended_label,
                intended_label_basis=src.intended_label_basis,
                pre_reasons=reasons,
                run_id=run_id,
                started=started,
            ))
        return out

    def _import_manual(
        self, csv_path: Path, run_id: str, started: datetime, st: SourceStats,
        checksums: dict[str, str],
    ) -> list[SeedRecord]:
        """Team-authored seeds (e.g. natively written Hinglish prompts).

        Rows with schema errors (unknown language/category/label) are listed in
        the report as row_errors; they cannot become records because their
        fields are not valid values.
        """
        path = resolve_inside(self.settings.project_root, csv_path)
        if not path.is_file():
            raise ConfigError(f"manual seed file not found: {path}")
        raw_dir = self.settings.raw_dir
        if path == raw_dir or raw_dir in path.parents:
            raise ConfigError("manual seeds must not live in data/raw/")
        digest = sha256_file(path)
        checksums[f"manual:{path.name}"] = digest
        languages = self.settings.languages.languages
        labels = set(self.settings.taxonomy.labels)
        allowed_cats = self._valid_categories | {self.settings.taxonomy.unassigned_category}

        out: list[SeedRecord] = []
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in MANUAL_REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(f"{path.name}: missing required columns {missing}")
            for n, row in enumerate(reader, start=2):  # line 1 is the header
                st.rows_read += 1
                errors = []
                lang = (row.get("language") or "").strip()
                script = (row.get("script") or "").strip()
                cat = (row.get("category") or "").strip()
                label = (row.get("intended_label") or "").strip().upper()
                if lang not in languages:
                    errors.append(f"unknown language {lang!r}")
                if script not in self._scripts:
                    errors.append(f"unknown script {script!r}")
                if cat not in allowed_cats:
                    errors.append(f"unknown category {cat!r}")
                if label not in labels:
                    errors.append(f"invalid intended_label {label!r}")
                if errors:
                    st.row_errors.append({"line": n, "errors": errors})
                    continue
                original = row.get("prompt") or ""
                ref = (row.get("source_reference") or "").strip() or f"line{n}"
                is_translit = (
                    languages[lang].romanized_script == script and languages[lang].native_script != script
                )
                out.append(self._build(
                    seed_id=f"S-MAN-{content_hash(original)[:12]}",
                    original=original,
                    language=lang,
                    expected_script=script,
                    is_transliterated=is_translit,
                    source_type="manual",
                    source_dataset=f"manual:{path.name}",
                    source_role="manual_seed",
                    source_file=path.name,
                    source_member=None,
                    source_file_sha256=digest,
                    source_reference=ref,
                    source_line=n,
                    source_category=None,
                    source_metadata={"author": (row.get("author") or "").strip() or None,
                                     "notes": (row.get("notes") or "").strip() or None},
                    category=cat,
                    category_status="unassigned" if cat == self.settings.taxonomy.unassigned_category
                    else "human_assigned",
                    intended_label=label,
                    intended_label_basis="Provisional: written by the seed author; not a gold label.",
                    pre_reasons=[],
                    run_id=run_id,
                    started=started,
                ))
        return out

    def _build(self, *, seed_id, original, language, expected_script, pre_reasons,
               run_id, started, is_transliterated=False, **fields) -> SeedRecord:
        cfg = self.settings.generation.seed_validation
        prompt = normalize_text(original)
        reasons = list(pre_reasons)

        if not prompt:
            reasons.append("empty_prompt")
        elif len(prompt) < cfg.min_chars:
            reasons.append("too_short")
        elif len(prompt) > cfg.max_chars:
            reasons.append("too_long")
        if has_control_chars(original):
            reasons.append("control_characters")
        if cfg.reject_replacement_char and "�" in original:
            reasons.append("replacement_character")
        if any(m in original for m in cfg.mojibake_markers):
            reasons.append("suspected_mojibake")

        script, conf = dominant_script(prompt, self._scripts)
        if prompt and (script != expected_script or conf < cfg.min_script_confidence):
            reasons.append("script_mismatch")

        reasons = sorted(set(reasons))
        return SeedRecord(
            seed_id=seed_id,
            seed_version=1,
            prompt=prompt,
            original_text=original,
            content_hash=content_hash(prompt),
            language=language,
            script=script,
            script_confidence=conf,
            is_transliterated=is_transliterated,
            seed_status="REJECTED" if reasons else "VALID",
            rejection_reasons=reasons,
            taxonomy_version=self.settings.taxonomy.taxonomy_version,
            generator_version=self.settings.generation.generator_version,
            import_run_id=run_id,
            imported_at=iso(started),
            **fields,
        )

    @staticmethod
    def _mark_duplicates(seeds: list[SeedRecord], stats: dict[str, SourceStats]) -> list[SeedRecord]:
        """Mark exact duplicates (same dedup key) and seed-id collisions.

        The first occurrence (config source order, then file order) stays
        VALID; later ones become DUPLICATE and point at it. Nothing is removed.
        """
        first_by_hash: dict[str, str] = {}
        seen_ids: set[str] = set()
        out: list[SeedRecord] = []
        for s in seeds:
            st = stats[s.source_dataset]
            if s.seed_id in seen_ids:
                reasons = sorted(set(s.rejection_reasons) | {"duplicate_seed_id"})
                s = s.model_copy(update={"seed_status": "REJECTED", "rejection_reasons": reasons,
                                         "duplicate_of": None})
            seen_ids.add(s.seed_id)
            if s.seed_status == "VALID":
                if s.content_hash in first_by_hash:
                    s = s.model_copy(update={"seed_status": "DUPLICATE",
                                             "duplicate_of": first_by_hash[s.content_hash]})
                else:
                    first_by_hash[s.content_hash] = s.seed_id
            if s.seed_status == "VALID":
                st.valid += 1
            elif s.seed_status == "DUPLICATE":
                st.duplicate += 1
            else:
                st.rejected += 1
                st.rejection_reasons.update(s.rejection_reasons)
            out.append(SeedRecord.model_validate(s.model_dump()))  # re-run validators
        return out

    # --------------------------------------------------------- selection

    def select_pilot(self, seeds: list[SeedRecord]) -> list[SeedRecord]:
        """Deterministic, stratified pilot selection.

        Order inside each source is sha256(random_seed:seed_id) — stable across
        Python versions and independent of file order. Within a source, picks
        rotate over source categories so each is represented; a candidate too
        similar to an already selected seed (word Jaccard) is skipped.
        """
        pcfg = self.settings.generation.pilot
        rs = self.settings.generation.random_seed

        def rank(s: SeedRecord) -> str:
            return hashlib.sha256(f"{rs}:{s.seed_id}".encode()).hexdigest()

        selected: list[SeedRecord] = []
        selected_words: list[frozenset[str]] = []

        for sid, quota in pcfg.quotas.items():
            pool = [s for s in seeds if s.source_dataset == sid and s.seed_status == "VALID"]
            by_cat: dict[str, list[SeedRecord]] = defaultdict(list)
            for s in sorted(pool, key=rank):
                by_cat[s.source_category or ""].append(s)
            queues = [by_cat[c] for c in sorted(by_cat)]
            taken = 0
            while taken < quota and any(queues):
                for q in queues:
                    if taken >= quota:
                        break
                    while q:
                        cand = q.pop(0)
                        words = word_set(cand.prompt)
                        if all(word_jaccard(words, w) < pcfg.max_word_jaccard for w in selected_words):
                            selected.append(cand)
                            selected_words.append(words)
                            taken += 1
                            break
            if taken < quota:
                raise PilotSelectionError(
                    f"source {sid!r}: only {taken} of {quota} pilot seeds available "
                    f"(valid pool {len(pool)}, max_word_jaccard {pcfg.max_word_jaccard})"
                )
        return selected

    # ------------------------------------------------------------ export

    def _check_output_dir(self, out_dir: Path) -> Path:
        out = resolve_inside(self.settings.project_root, out_dir)
        raw = self.settings.raw_dir
        if out == raw or raw in out.parents:
            raise ExportError("refusing to write into data/raw/: raw data is read-only")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def export_pool(self, result: ImportResult, out_dir: Path) -> dict[str, Path]:
        out = self._check_output_dir(out_dir)
        pool_path = out / "seed_pool.jsonl"
        _write_jsonl(pool_path, result.seeds)
        report = build_run_manifest(
            run_id=result.run_id,
            run_type="seed_import",
            started_at=result.started_at,
            settings=self.settings,
            source_checksums=result.source_checksums,
            extra={
                "totals": dict(Counter(s.seed_status for s in result.seeds)),
                "per_source": {k: v.as_dict() for k, v in result.stats.items()},
                "outputs": {pool_path.name: sha256_file(pool_path)},
            },
        )
        report_path = out / "import_report.json"
        _write_json(report_path, report)
        return {"pool": pool_path, "report": report_path}

    def export_pilot(
        self, pilot: list[SeedRecord], result: ImportResult, out_dir: Path, force: bool = False
    ) -> dict[str, Path]:
        out = self._check_output_dir(out_dir)
        version = self.settings.generation.pilot.dataset_version
        base = f"pilot_seeds_{version}"
        jsonl_path, csv_path = out / f"{base}.jsonl", out / f"{base}.csv"
        manifest_path = out / f"{base}.manifest.json"

        fingerprint = selection_fingerprint(pilot)
        if manifest_path.exists() and not force:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("selection_fingerprint") != fingerprint:
                raise ExportError(
                    f"{manifest_path.name} already holds a different selection for {version}. "
                    "Bump pilot.dataset_version, or pass force=True to replace it deliberately."
                )

        _write_jsonl(jsonl_path, pilot)
        _write_csv(csv_path, pilot)

        def dist(attr: str) -> dict:
            return dict(sorted(Counter(str(getattr(s, attr)) for s in pilot).items()))

        manifest = build_run_manifest(
            run_id=result.run_id,
            run_type="pilot_seed_export",
            started_at=result.started_at,
            settings=self.settings,
            source_checksums=result.source_checksums,
            extra={
                "dataset_version": version,
                "record_count": len(pilot),
                "selection": {
                    "method": "per-source quota; round-robin over source categories; "
                              "order = sha256(random_seed:seed_id); word-Jaccard diversity guard",
                    "quotas": self.settings.generation.pilot.quotas,
                    "max_word_jaccard": self.settings.generation.pilot.max_word_jaccard,
                },
                "selection_fingerprint": fingerprint,
                "distributions": {
                    "source_dataset": dist("source_dataset"),
                    "source_category": dist("source_category"),
                    "category": dist("category"),
                    "category_status": dist("category_status"),
                    "intended_label": dist("intended_label"),
                    "language": dist("language"),
                    "script": dist("script"),
                },
                "label_note": "intended_label is provisional and source-derived; "
                              "final_label is empty until human annotation.",
                "outputs": {p.name: sha256_file(p) for p in (jsonl_path, csv_path)},
            },
        )
        _write_json(manifest_path, manifest)
        return {"jsonl": jsonl_path, "csv": csv_path, "manifest": manifest_path}


# ------------------------------------------------------------------ helpers


def selection_fingerprint(seeds: list[SeedRecord]) -> str:
    """Identifies a selection by its seeds and their text, not by run timestamps."""
    body = "\n".join(sorted(f"{s.seed_id}:{s.seed_version}:{s.content_hash}" for s in seeds))
    return hashlib.sha256(body.encode()).hexdigest()


def _atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_jsonl(path: Path, seeds: list[SeedRecord]) -> None:
    _atomic_write(path, "".join(
        json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n" for s in seeds
    ))


def _write_json(path: Path, obj: dict) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, seeds: list[SeedRecord]) -> None:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for s in seeds:
        row = s.model_dump(mode="json")
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
        writer.writerow(row)
    # utf-8-sig so Excel on Windows shows Devanagari correctly.
    _atomic_write(path, buf.getvalue(), encoding="utf-8-sig")


def load_seeds_jsonl(path: Path) -> list[SeedRecord]:
    with path.open(encoding="utf-8") as fh:
        return [SeedRecord.model_validate_json(line) for line in fh if line.strip()]
