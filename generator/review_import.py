"""Pilot review layer: double annotations, agreement, and adjudication import.

Two teammates independently labelled every pilot seed in their own copy of the
pilot CSV. `import_reviews` reads only the annotation columns of those copies
(my_label, my_category, note), matches each row to the pilot by seed_id and
checks its content_hash, maps free-text categories to taxonomy category_ids
through an explicit mapping table, and computes label agreement. Cells that are
blank, hold several categories or are not in the table are flagged and left
unresolved; nothing is guessed.

Adjudication happens outside the code, in the team's worksheet.
`build_reviewed_pilot` writes the next pilot version from it: agreed seeds take
the shared label and category, the others take the adjudicated values. It never
chooses a label itself, refuses to run while any worksheet row is incomplete,
and only reads the input pilot version.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from backend.config import Settings, resolve_inside
from generator.provenance import build_run_manifest, git_state, iso, new_run_id, sha256_file, utc_now
from generator.schemas import SeedRecord

KEY_COLUMNS = ("seed_id", "content_hash")
ANNOTATION_COLUMNS = ("my_label", "my_category", "note")
MAPPING_COLUMNS = ("raw_value", "category_id", "mapping_type", "note")
MAPPING_TYPES = ("display_name", "typo", "multiple")
ADJUDICATION_COLUMNS = ("adjudicated_label", "adjudicated_category", "adjudicated_by", "rationale")
WORKSHEET_BLANK = "(blank)"


class ReviewImportError(RuntimeError):
    """Review inputs are inconsistent with the pilot (hash mismatch, missing seed, bad table)."""


class AdjudicationError(RuntimeError):
    """The worksheet cannot be applied: it is incomplete or does not match the review layer."""

    def __init__(self, message: str, problems: list[str]):
        super().__init__(message)
        self.problems = problems


# ------------------------------------------------------------------ config


class AnnotatorFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    file: str
    review_date: str
    review_date_source: str


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_layer_version: str
    input_pilot: str
    input_pilot_manifest: str
    category_mapping: str
    worksheet: str
    reviews_dir: str
    output_dataset_version: str
    output_dir: str
    annotators: list[AnnotatorFile] = Field(min_length=2, max_length=2)


def load_review_config(path: Path) -> ReviewConfig:
    return ReviewConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ helpers


def norm_key(text: str) -> str:
    """Mapping-table key: whitespace collapsed, case-folded. Nothing else."""
    return " ".join(text.split()).casefold()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV, refusing rows with more cells than the header.

    An unquoted comma in a free-text cell (e.g. a rationale) splits it, and the
    overflow would otherwise be dropped silently.
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows, start=2):
        if None in row:
            raise ReviewImportError(
                f"{path.name} line {i}: {len(row[None])} cell(s) beyond the header "
                f"({row[None]!r}); a text cell probably contains an unquoted comma")
    return rows


def _require_columns(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    present = set(rows[0]) if rows else set()
    missing = [c for c in columns if c not in present]
    if missing:
        raise ReviewImportError(f"{path.name}: missing column(s) {missing}")


def _cell(row: dict, col: str) -> str:
    return (row.get(col) or "").strip()


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters; None when undefined (no pairs, or chance agreement = 1)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def _agreement(pairs: list[tuple[str, str]], n_seeds: int) -> dict:
    agree = sum(a == b for a, b in pairs)
    kappa = cohen_kappa(pairs)
    return {
        "n_seeds": n_seeds,
        "n_compared": len(pairs),
        "n_agree": agree,
        "percent_agreement": round(100 * agree / len(pairs), 2) if pairs else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "pairs": dict(sorted(Counter(f"{a}|{b}" for a, b in pairs).items())),
    }


# ------------------------------------------------------------------ category mapping


@dataclass(frozen=True)
class CategoryMapping:
    table: dict[str, tuple[str, list[str], str]]   # norm key -> (raw_value, ids, mapping_type)
    category_ids: frozenset[str]

    def resolve(self, raw: str) -> dict:
        """Map one annotator cell. Returns category_id (or None), candidates, method, flags."""
        value = raw.strip()
        if not value:
            return {"category_id": None, "candidates": [], "method": "blank", "flags": ["BLANK_CATEGORY"]}
        if value in self.category_ids:
            return {"category_id": value, "candidates": [value], "method": "exact_id", "flags": []}
        hit = self.table.get(norm_key(value))
        if hit is None:
            return {"category_id": None, "candidates": [], "method": "unmapped", "flags": ["UNMAPPED_CATEGORY"]}
        _, ids, kind = hit
        if kind == "multiple":
            return {"category_id": None, "candidates": ids, "method": kind, "flags": ["MULTIPLE_CATEGORIES"]}
        flags = ["TYPO_MAPPED"] if kind == "typo" else []
        return {"category_id": ids[0], "candidates": ids, "method": kind, "flags": flags}


def load_category_mapping(path: Path, settings: Settings) -> CategoryMapping:
    ids = frozenset(c.category_id for c in settings.taxonomy.categories)
    rows = _read_csv(path)
    _require_columns(path, rows, MAPPING_COLUMNS)
    table: dict[str, tuple[str, list[str], str]] = {}
    for i, row in enumerate(rows, start=2):
        raw, kind = _cell(row, "raw_value"), _cell(row, "mapping_type")
        targets = [t.strip() for t in _cell(row, "category_id").split("+") if t.strip()]
        where = f"{path.name} line {i}"
        if not raw:
            raise ReviewImportError(f"{where}: empty raw_value")
        if kind not in MAPPING_TYPES:
            raise ReviewImportError(f"{where}: mapping_type {kind!r} not in {MAPPING_TYPES}")
        unknown = [t for t in targets if t not in ids]
        if unknown or not targets:
            raise ReviewImportError(f"{where}: category_id {unknown or '(empty)'} not in taxonomy "
                                    f"v{settings.taxonomy.taxonomy_version}")
        if (kind == "multiple") != (len(targets) > 1):
            raise ReviewImportError(f"{where}: 'multiple' needs 2+ category_ids, other types exactly one")
        key = norm_key(raw)
        if key in table:
            raise ReviewImportError(f"{where}: {raw!r} duplicates an earlier row")
        table[key] = (raw, targets, kind)
    return CategoryMapping(table=table, category_ids=ids)


# ------------------------------------------------------------------ import


@dataclass
class ReviewImport:
    config: ReviewConfig
    pilot: dict[str, dict]                      # seed_id -> v0.1 record (pilot order)
    layer: list[dict]                           # one record per seed
    agreement: dict
    category_values: list[dict]                 # each distinct raw category cell and its resolution
    worksheet_check: dict
    worksheet_rows: dict[str, dict]
    inputs: dict[str, str] = field(default_factory=dict)   # relative path -> sha256
    input_dataset_version: str | None = None


def _load_pilot(cfg: ReviewConfig, root: Path) -> tuple[dict[str, dict], dict[str, str], str | None]:
    path = resolve_inside(root, cfg.input_pilot)
    manifest_path = resolve_inside(root, cfg.input_pilot_manifest)
    digest = sha256_file(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("outputs", {}).get(path.name)
    if recorded != digest:
        raise ReviewImportError(f"{path.name} does not match the checksum in {manifest_path.name}; "
                                "the input pilot changed since it was exported")
    pilot: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = SeedRecord(**json.loads(line)).model_dump(mode="json")
            pilot[rec["seed_id"]] = rec
    inputs = {cfg.input_pilot: digest, cfg.input_pilot_manifest: sha256_file(manifest_path)}
    return pilot, inputs, manifest.get("dataset_version")


def _load_annotator(ann: AnnotatorFile, root: Path, pilot: dict[str, dict],
                    labels: list[str], mapping: CategoryMapping) -> tuple[dict[str, dict], str]:
    path = resolve_inside(root, ann.file)
    rows = _read_csv(path)
    _require_columns(path, rows, KEY_COLUMNS + ANNOTATION_COLUMNS)
    digest = sha256_file(path)
    out: dict[str, dict] = {}
    for i, row in enumerate(rows, start=2):
        sid = _cell(row, "seed_id")
        if sid not in pilot:
            raise ReviewImportError(f"{path.name} line {i}: seed_id {sid!r} is not in the pilot")
        if sid in out:
            raise ReviewImportError(f"{path.name} line {i}: seed_id {sid} appears twice")
        if _cell(row, "content_hash") != pilot[sid]["content_hash"]:
            raise ReviewImportError(f"{path.name} line {i}: content_hash for {sid} differs from the "
                                    "pilot; the prompt was edited or the row belongs to another version")
        raw_label, raw_cat, note = (row.get(c) or "" for c in ANNOTATION_COLUMNS)
        label = raw_label.strip().upper()
        flags = [] if label in labels else (["BLANK_LABEL"] if not label else ["INVALID_LABEL"])
        cat = mapping.resolve(raw_cat)
        out[sid] = {
            "annotator": ann.name,
            "review_date": ann.review_date,
            "review_date_source": ann.review_date_source,
            "source_file": ann.file,
            "source_file_sha256": digest,
            "raw_label": raw_label,
            "raw_category": raw_cat,
            "note": note.strip(),
            "label": label if label in labels else None,
            "category_id": cat["category_id"],
            "category_candidates": cat["candidates"],
            "category_mapping": cat["method"],
            "flags": flags + cat["flags"],
        }
    missing = [s for s in pilot if s not in out]
    if missing:
        raise ReviewImportError(f"{path.name}: no annotation for {len(missing)} pilot seed(s): {missing}")
    return out, digest


def _issue(a: dict, b: dict) -> str | None:
    parts = []
    if a["label"] is None or b["label"] is None or a["label"] != b["label"]:
        parts.append("label")
    if a["category_id"] is None or b["category_id"] is None or a["category_id"] != b["category_id"]:
        parts.append("category")
    return "+".join(parts) or None


def import_reviews(cfg: ReviewConfig, settings: Settings) -> ReviewImport:
    root = settings.project_root
    labels = list(settings.taxonomy.labels)
    pilot, inputs, input_version = _load_pilot(cfg, root)
    mapping_path = resolve_inside(root, cfg.category_mapping)
    mapping = load_category_mapping(mapping_path, settings)
    inputs[cfg.category_mapping] = sha256_file(mapping_path)

    per_annotator: list[dict[str, dict]] = []
    for ann in cfg.annotators:
        rows, digest = _load_annotator(ann, root, pilot, labels, mapping)
        per_annotator.append(rows)
        inputs[ann.file] = digest
    a_name, b_name = (ann.name for ann in cfg.annotators)
    a_rows, b_rows = per_annotator

    layer = []
    for sid, rec in pilot.items():
        a, b = a_rows[sid], b_rows[sid]
        issue = _issue(a, b)
        layer.append({
            "seed_id": sid,
            "content_hash": rec["content_hash"],
            "prompt": rec["prompt"],
            "pre_review": {k: rec[k] for k in ("category", "category_status", "intended_label")},
            "annotations": {a_name: a, b_name: b},
            "label_agreement": None if None in (a["label"], b["label"]) else a["label"] == b["label"],
            "category_agreement": (None if None in (a["category_id"], b["category_id"])
                                   else a["category_id"] == b["category_id"]),
            "issue": issue,
            "needs_adjudication": issue is not None,
            "review_layer_version": cfg.review_layer_version,
            "taxonomy_version": settings.taxonomy.taxonomy_version,
        })

    label_pairs = [(r["annotations"][a_name]["label"], r["annotations"][b_name]["label"])
                   for r in layer if r["label_agreement"] is not None]
    cat_pairs = [(r["annotations"][a_name]["category_id"], r["annotations"][b_name]["category_id"])
                 for r in layer if r["category_agreement"] is not None]
    agreement = {
        "annotators": [a_name, b_name],
        "label": _agreement(label_pairs, len(layer)),
        "category": {**_agreement(cat_pairs, len(layer)),
                     "note": "only seeds where both category cells resolved to one category_id"},
        "needs_adjudication": [r["seed_id"] for r in layer if r["needs_adjudication"]],
    }

    seen: dict[tuple[str, str], dict] = {}
    for r in layer:
        for name, ann in r["annotations"].items():
            key = (ann["raw_category"], name)
            entry = seen.setdefault(key, {
                "raw_value": ann["raw_category"], "annotator": name, "count": 0,
                "category_id": ann["category_id"], "candidates": ann["category_candidates"],
                "method": ann["category_mapping"], "seed_ids": [],
            })
            entry["count"] += 1
            entry["seed_ids"].append(r["seed_id"])
    category_values = sorted(seen.values(), key=lambda e: (e["method"], e["raw_value"], e["annotator"]))

    ws_path = resolve_inside(root, cfg.worksheet)
    ws_rows = _load_worksheet(ws_path)
    inputs[cfg.worksheet] = sha256_file(ws_path)
    check = compare_worksheet(layer, ws_rows, [a_name, b_name])

    return ReviewImport(config=cfg, pilot=pilot, layer=layer, agreement=agreement,
                        category_values=category_values, worksheet_check=check,
                        worksheet_rows=ws_rows, inputs=inputs, input_dataset_version=input_version)


# ------------------------------------------------------------------ worksheet


def _load_worksheet(path: Path) -> dict[str, dict]:
    rows = _read_csv(path)
    _require_columns(path, rows, ("seed_id", "issue") + ADJUDICATION_COLUMNS)
    out: dict[str, dict] = {}
    for i, row in enumerate(rows, start=2):
        sid = _cell(row, "seed_id")
        if sid in out:
            raise ReviewImportError(f"{path.name} line {i}: seed_id {sid} appears twice")
        out[sid] = row
    return out


def _ws_categories(cell: str) -> set[str]:
    cell = cell.strip()
    if not cell or cell == WORKSHEET_BLANK:
        return set()
    return {c.strip() for c in cell.split("+") if c.strip()}


def _ann_categories(ann: dict) -> set[str]:
    if ann["category_id"]:
        return {ann["category_id"]}
    if ann["category_candidates"]:
        return set(ann["category_candidates"])
    return {ann["raw_category"].strip()} if ann["raw_category"].strip() else set()


def compare_worksheet(layer: list[dict], ws: dict[str, dict], annotators: list[str]) -> dict:
    """Compare the computed disagreement list with the worksheet's rows and copied values."""
    computed = {r["seed_id"]: r for r in layer if r["needs_adjudication"]}
    diffs: list[dict] = []

    def diff(sid: str, column: str, worksheet_value, computed_value) -> None:
        diffs.append({"seed_id": sid, "column": column,
                      "worksheet": worksheet_value, "computed": computed_value})

    for sid in sorted(set(computed) & set(ws)):
        r, row = computed[sid], ws[sid]
        if "issue" in row and _cell(row, "issue") != r["issue"]:
            diff(sid, "issue", _cell(row, "issue"), r["issue"])
        if "prompt" in row and _cell(row, "prompt") != r["prompt"]:
            diff(sid, "prompt", _cell(row, "prompt"), r["prompt"])
        if "source_label" in row and _cell(row, "source_label") != r["pre_review"]["intended_label"]:
            diff(sid, "source_label", _cell(row, "source_label"), r["pre_review"]["intended_label"])
        for name in annotators:
            ann = r["annotations"][name]
            col = f"{name}_label"
            if col in row and _cell(row, col).upper() != ann["raw_label"].strip().upper():
                diff(sid, col, _cell(row, col), ann["raw_label"].strip())
            col = f"{name}_category"
            if col in row and _ws_categories(row[col] or "") != _ann_categories(ann):
                diff(sid, col, _cell(row, col), sorted(_ann_categories(ann)))
            col = f"{name}_note"
            if col in row and " ".join(_cell(row, col).split()) != " ".join(ann["note"].split()):
                diff(sid, col, _cell(row, col), ann["note"])

    missing = sorted(set(computed) - set(ws))
    extra = sorted(set(ws) - set(computed))
    return {
        "matches": not (missing or extra or diffs),
        "computed_disagreements": len(computed),
        "worksheet_rows": len(ws),
        "missing_from_worksheet": missing,
        "extra_in_worksheet": extra,
        "field_differences": diffs,
    }


# ------------------------------------------------------------------ outputs


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    tmp.replace(path)


def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def write_review_layer(result: ReviewImport, settings: Settings) -> dict[str, Path]:
    """Write the review layer (per-seed JSONL) and the agreement/worksheet report."""
    cfg = result.config
    out = resolve_inside(settings.project_root, cfg.reviews_dir)
    out.mkdir(parents=True, exist_ok=True)
    layer_path = out / f"review_layer_{cfg.review_layer_version}.jsonl"
    report_path = out / f"review_report_{cfg.review_layer_version}.json"
    _write(layer_path, _jsonl(result.layer))
    flagged = [
        {"seed_id": r["seed_id"], "annotator": name, "flags": ann["flags"],
         "raw_label": ann["raw_label"], "raw_category": ann["raw_category"], "note": ann["note"]}
        for r in result.layer for name, ann in r["annotations"].items() if ann["flags"]
    ]
    report = {
        "review_layer_version": cfg.review_layer_version,
        "taxonomy_version": settings.taxonomy.taxonomy_version,
        "generated_at": iso(utc_now()),
        "columns_read": list(KEY_COLUMNS + ANNOTATION_COLUMNS),
        "inputs": result.inputs,
        "agreement": result.agreement,
        "flagged_cells": flagged,
        "category_values": result.category_values,
        "worksheet_check": result.worksheet_check,
        "outputs": {layer_path.name: sha256_file(layer_path)},
    }
    _write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"layer": layer_path, "report": report_path}


# ------------------------------------------------------------------ adjudication -> next pilot


def _check_adjudication(result: ReviewImport, settings: Settings) -> list[str]:
    labels = set(settings.taxonomy.labels)
    ids = {c.category_id for c in settings.taxonomy.categories}
    problems = []
    check = result.worksheet_check
    if check["missing_from_worksheet"]:
        problems.append(f"disagreements missing from the worksheet: {check['missing_from_worksheet']}")
    if check["extra_in_worksheet"]:
        problems.append(f"worksheet rows that are not disagreements: {check['extra_in_worksheet']}")
    for d in check["field_differences"]:
        problems.append(f"{d['seed_id']}: worksheet {d['column']}={d['worksheet']!r}, "
                        f"review files give {d['computed']!r}")
    for sid, row in result.worksheet_rows.items():
        label, cat = _cell(row, "adjudicated_label"), _cell(row, "adjudicated_category")
        missing = [c for c in ADJUDICATION_COLUMNS if not _cell(row, c)]
        if missing:
            problems.append(f"{sid}: not adjudicated yet (empty {', '.join(missing)})")
            continue
        if label not in labels:
            problems.append(f"{sid}: adjudicated_label {label!r} is not one of {sorted(labels)}")
        if cat not in ids:
            problems.append(f"{sid}: adjudicated_category {cat!r} is not a taxonomy "
                            f"v{settings.taxonomy.taxonomy_version} category_id")
    return problems


def build_reviewed_pilot(result: ReviewImport, settings: Settings, *, force: bool = False) -> dict[str, Path]:
    """Write the next pilot version with human final labels and review provenance."""
    problems = _check_adjudication(result, settings)
    if problems:
        raise AdjudicationError(f"cannot build {result.config.output_dataset_version}: "
                                f"{len(problems)} problem(s)", problems)
    cfg = result.config
    started = utc_now()
    # Git state before any output is written: the outputs are tracked files, so
    # checking afterwards would report the build's own output as uncommitted.
    git_before = git_state(settings.project_root)
    ws_sha = result.inputs[cfg.worksheet]
    records = []
    for r in result.layer:
        rec = dict(result.pilot[r["seed_id"]])
        anns = list(r["annotations"].values())
        if r["needs_adjudication"]:
            row = result.worksheet_rows[r["seed_id"]]
            label, cat = _cell(row, "adjudicated_label"), _cell(row, "adjudicated_category")
            resolution, label_status = "adjudication", "human_adjudicated"
            adjudication = {"label": label, "category": cat,
                            "adjudicated_by": _cell(row, "adjudicated_by"),
                            "rationale": _cell(row, "rationale"),
                            "worksheet": cfg.worksheet, "worksheet_sha256": ws_sha}
        else:
            label, cat = anns[0]["label"], anns[0]["category_id"]
            resolution, label_status, adjudication = "agreement", "human_agreed", None
        rec.update({
            "category": cat,
            "category_status": "human_assigned",
            "label_status": label_status,
            "final_label": label,
            "dataset_version": cfg.output_dataset_version,
            "parent_dataset_version": result.input_dataset_version,
            "review": {
                "review_layer_version": cfg.review_layer_version,
                "taxonomy_version": settings.taxonomy.taxonomy_version,
                "resolution": resolution,
                "issue": r["issue"],
                "pre_review": r["pre_review"],
                "annotations": anns,
                "adjudication": adjudication,
            },
        })
        records.append(rec)

    out = resolve_inside(settings.project_root, cfg.output_dir)
    base = f"pilot_seeds_{cfg.output_dataset_version}"
    paths = {"jsonl": out / f"{base}.jsonl", "csv": out / f"{base}.csv",
             "manifest": out / f"{base}.manifest.json"}
    if resolve_inside(settings.project_root, cfg.input_pilot) in paths.values():
        raise AdjudicationError("output would overwrite the input pilot", [str(paths["jsonl"])])
    if paths["manifest"].exists() and not force:
        raise AdjudicationError(f"{paths['manifest'].name} already exists; pass force to replace it",
                                [str(paths["manifest"])])

    _write(paths["jsonl"], _jsonl(records))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    for rec in records:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in rec.items()})
    _write(paths["csv"], "﻿" + buf.getvalue())   # utf-8-sig, as for v0.1

    manifest = build_run_manifest(
        run_id=new_run_id("REVIEW", started, settings),
        run_type="pilot_review_build",
        started_at=started,
        settings=settings,
        source_checksums={},
        extra={
            "dataset_version": cfg.output_dataset_version,
            "parent_dataset_version": result.input_dataset_version,
            "input_pilot": cfg.input_pilot,
            "record_count": len(records),
            "inputs": result.inputs,
            "agreement": result.agreement,
            "distributions": {
                k: dict(sorted(Counter(str(rec[k]) for rec in records).items()))
                for k in ("final_label", "category", "label_status")
            },
            "label_note": "final_label comes from the two annotators where they agree on label and "
                          "category, otherwise from the team's adjudication worksheet.",
            "outputs": {p.name: sha256_file(p) for p in (paths["jsonl"], paths["csv"])},
        },
    )
    manifest["git"] = git_before
    _write(paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return paths
