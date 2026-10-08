"""Summarise filled native-speaker review sheets for one pilot run.

Inputs: the run folder (variants.jsonl, to check prompt ids) and a folder of
filled sheets, matched by filename prefix:

    review_<lang>*.csv           translation + romanisation sheet (pilot_translation.REVIEW_COLUMNS)
    review_codemix_<lang>*.csv   code-mix sheet (qc_pipeline.CODEMIX_REVIEW_COLUMNS)

Several files per language are allowed (one per reviewer, e.g.
review_hi_bhargavi.csv). The reviewer is the row's `reviewer` cell, else the
filename suffix after the prefix, else "unknown". Languages come from
languages.yaml (enabled targets); a code-mix sheet is expected only for
languages in code_mixing target_languages (not te).

Rows: a row with no review_id / seed_id and no prompt ids is blank and skipped.
A row counts as rated when at least one rating cell (scores or intent) is
filled. Bad cells (non-numeric, out of range, intent not Y/N) are ignored with
a warning, never a crash. Files that are not UTF-8 (Excel's plain "CSV" saves
cp1252 and destroys Indic script) or lack the expected columns are skipped
with a warning.

Outputs in the run folder: review_summary.json and review_summary.md.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import Settings
from generator.provenance import iso, sha256_file, utc_now
from generator.seed_manager import _atomic_write, _write_json

SUMMARY_VERSION = "1.0"
TRANSLATION = "translation"
CODEMIX = "codemix"
SCALES = {  # column -> (min, max)
    "translation_adequacy_1to5": (1, 5),
    "translation_fluency_1to5": (1, 5),
    "romanisation_natural_1to5": (1, 5),
    "codemix_natural_1to3": (1, 3),
}
INTENT = "intent_preserved_Y_N"
REQUIRED = {
    TRANSLATION: {"review_id", "native_prompt_id", "latin_prompt_id", "translation_adequacy_1to5",
                  "translation_fluency_1to5", INTENT, "romanisation_natural_1to5"},
    CODEMIX: {"review_id", "native_prompt_id", "latin_prompt_id", "intended_label", "codemix_natural_1to3", INTENT},
}
RATING_COLS = {
    TRANSLATION: ["translation_adequacy_1to5", "translation_fluency_1to5", "romanisation_natural_1to5"],
    CODEMIX: ["codemix_natural_1to3"],
}
LOWEST_N = 10
SHORT = {"translation_adequacy_1to5": "adeq", "translation_fluency_1to5": "flu",
         "romanisation_natural_1to5": "rom", "codemix_natural_1to3": "cm"}


@dataclass
class Row:
    file: str
    kind: str
    language: str
    reviewer: str
    line: int
    cells: dict[str, str]
    ratings: dict[str, float] = field(default_factory=dict)
    intent: bool | None = None

    @property
    def rated(self) -> bool:
        return bool(self.ratings) or self.intent is not None

    @property
    def label(self) -> str:
        return (self.cells.get("intended_label") or self.cells.get("provisional_label") or "").strip().upper()

    def score(self) -> float:
        """0 (worst) .. 1 (best): mean of ratings scaled to 0-1, with intent N = 0 and Y = 1."""
        parts = [(v - SCALES[k][0]) / (SCALES[k][1] - SCALES[k][0]) for k, v in self.ratings.items()]
        if self.intent is not None:
            parts.append(1.0 if self.intent else 0.0)
        return round(statistics.fmean(parts), 4) if parts else 1.0


def _cell(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def match_sheets(folder: Path, languages: Iterable[str]) -> dict[tuple[str, str], list[Path]]:
    """(language, kind) -> sheet files, by filename prefix."""
    out: dict[tuple[str, str], list[Path]] = defaultdict(list)
    files = sorted(p for p in folder.glob("*.csv") if p.is_file())
    for lang in languages:
        for p in files:
            name = p.name.lower()
            if name.startswith(f"review_codemix_{lang}"):
                out[(lang, CODEMIX)].append(p)
            elif name.startswith(f"review_{lang}"):
                out[(lang, TRANSLATION)].append(p)
    return out


def _reviewer_from_name(path: Path, lang: str, kind: str) -> str | None:
    prefix = f"review_codemix_{lang}" if kind == CODEMIX else f"review_{lang}"
    rest = path.stem[len(prefix):].strip("_- ")
    return rest or None


def read_sheet(path: Path, lang: str, kind: str, warnings: list[str]) -> list[Row] | None:
    """Rows of one sheet; None (with a warning) when the file cannot be used at all."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        warnings.append(f"{path.name}: not UTF-8 (Excel 'CSV' instead of 'CSV UTF-8'?); file skipped")
        return None
    reader = csv.DictReader(text.splitlines())
    header = set(reader.fieldnames or [])
    missing = REQUIRED[kind] - header
    if missing:
        warnings.append(f"{path.name}: missing column(s) {sorted(missing)} (wrong delimiter or sheet?); file skipped")
        return None
    default_reviewer = _reviewer_from_name(path, lang, kind)
    rows = []
    for i, raw in enumerate(reader, start=2):
        cells = {k: (v or "") for k, v in raw.items() if k is not None}
        if not any(_cell(cells, k) for k in ("review_id", "seed_id", "native_prompt_id", "latin_prompt_id")):
            continue                                  # blank row
        r = Row(path.name, kind, lang, _cell(cells, "reviewer") or default_reviewer or "unknown", i, cells)
        for col in RATING_COLS[kind]:
            v = _cell(cells, col)
            if not v:
                continue
            lo, hi = SCALES[col]
            try:
                x = float(v.replace(",", "."))
            except ValueError:
                warnings.append(f"{path.name}:{i}: {col}={v!r} is not a number; ignored")
                continue
            if not lo <= x <= hi:
                warnings.append(f"{path.name}:{i}: {col}={v!r} outside {lo}-{hi}; ignored")
                continue
            r.ratings[col] = x
        iv = _cell(cells, INTENT).upper()
        if iv in ("Y", "YES"):
            r.intent = True
        elif iv in ("N", "NO"):
            r.intent = False
        elif iv:
            warnings.append(f"{path.name}:{i}: {INTENT}={iv!r} is not Y/N; ignored")
        rows.append(r)
    return rows


def _mean(rows: list[Row], col: str) -> dict:
    xs = [r.ratings[col] for r in rows if col in r.ratings]
    return {"mean": round(statistics.fmean(xs), 3) if xs else None, "n": len(xs)}


def _intent(rows: list[Row]) -> dict:
    ans = [r.intent for r in rows if r.intent is not None]
    return {"pct_Y": round(100 * sum(ans) / len(ans), 1) if ans else None, "n": len(ans)}


def translation_stats(rows: list[Row]) -> dict:
    return {
        "n_rated": sum(r.rated for r in rows), "n_total": len(rows),
        "translation_adequacy_1to5": _mean(rows, "translation_adequacy_1to5"),
        "translation_fluency_1to5": _mean(rows, "translation_fluency_1to5"),
        "intent_preserved": _intent(rows),
        "romanisation_natural_1to5": _mean(rows, "romanisation_natural_1to5"),
    }


def codemix_stats(rows: list[Row]) -> dict:
    nat = [r.ratings["codemix_natural_1to3"] for r in rows if "codemix_natural_1to3" in r.ratings]
    return {
        "n_rated": sum(r.rated for r in rows), "n_total": len(rows),
        "codemix_natural_1to3": _mean(rows, "codemix_natural_1to3"),
        "pct_rated_1_unnatural": round(100 * sum(x == 1 for x in nat) / len(nat), 1) if nat else None,
        "intent_preserved": _intent(rows),
    }


def _by_reviewer(rows: list[Row], fn) -> dict:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.reviewer].append(r)
    return {"overall": fn(rows), "by_reviewer": {k: fn(v) for k, v in sorted(groups.items())}}


def _lowest(rows: list[Row], n: int) -> list[dict]:
    rated = sorted((r for r in rows if r.rated), key=lambda r: (r.score(), r.file, r.line))
    keep = ("review_id", "level", "native_prompt_id", "native_text", "latin_prompt_id", "latin_text", "notes")
    return [{"score_0to1": r.score(), "sheet": r.kind, "file": r.file, "line": r.line, "reviewer": r.reviewer,
             "label": r.label or None, "ratings": r.ratings, "intent_preserved": r.intent,
             **{k: _cell(r.cells, k) for k in keep if k in r.cells}} for r in rated[:n]]


def check_prompt_ids(rows: Iterable[Row], known: set[str], warnings: list[str]) -> int:
    bad = 0
    for r in rows:
        for col in ("native_prompt_id", "latin_prompt_id"):
            pid = _cell(r.cells, col)
            if pid and pid not in known:
                bad += 1
                warnings.append(f"{r.file}:{r.line}: {col} {pid} is not in the run's variants.jsonl")
    return bad


def summarize_reviews(settings: Settings, run_dir: Path, sheets_dir: Path) -> dict:
    warnings: list[str] = []
    variants_path = run_dir / "variants.jsonl"
    known: set[str] = set()
    if variants_path.exists():
        with variants_path.open(encoding="utf-8") as fh:
            known = {json.loads(line)["prompt_id"] for line in fh if line.strip()}
    else:
        warnings.append(f"{variants_path} not found: prompt ids not checked")
    languages = [c for c, lang in settings.languages.languages.items() if lang.enabled and lang.role == "target"]
    cm = settings.generation.code_mixing
    mixer = cm.providers.get(cm.default_provider) if cm.default_provider else None
    mixed = set(mixer.target_languages) if mixer and mixer.enabled else set()
    sheets = match_sheets(sheets_dir, languages) if sheets_dir.is_dir() else {}
    if not sheets_dir.is_dir():
        warnings.append(f"sheets folder {sheets_dir} not found")
    files, per_lang = [], {}
    n_bad_ids = 0
    for lang in languages:
        entry: dict = {}
        all_rows: list[Row] = []
        for kind in (TRANSLATION, CODEMIX):
            paths = sheets.get((lang, kind), [])
            if kind == CODEMIX and lang not in mixed:
                if paths:
                    warnings.append(f"{lang}: code-mix sheet(s) {[p.name for p in paths]} found, but {lang} "
                                    "is not code-mixed; ignored")
                entry[kind] = {"status": "not_applicable"}
                continue
            if not paths:
                warnings.append(f"{lang}: no {kind} sheet (review_{'codemix_' if kind == CODEMIX else ''}{lang}*.csv)")
                entry[kind] = {"status": "no_sheet"}
                continue
            rows, readable = [], 0
            for p in paths:
                got = read_sheet(p, lang, kind, warnings)
                files.append({"file": p.name, "sha256": sha256_file(p), "language": lang, "sheet": kind,
                              "readable": got is not None, "rows": len(got or []),
                              "rated": sum(r.rated for r in got or [])})
                if got is not None:
                    readable += 1
                    rows += got
            if not readable:
                entry[kind] = {"status": "unreadable", "files": [p.name for p in paths]}
                continue
            n_bad_ids += check_prompt_ids(rows, known, warnings) if known else 0
            all_rows += rows
            if kind == TRANSLATION:
                entry[kind] = {"status": "ok", **_by_reviewer(rows, translation_stats)}
            else:
                unsafe = [r for r in rows if r.label == "UNSAFE"]
                entry[kind] = {"status": "ok", **_by_reviewer(rows, codemix_stats),
                               "unsafe_only": _by_reviewer(unsafe, codemix_stats)}
        entry["lowest_rated"] = _lowest(all_rows, LOWEST_N)
        per_lang[lang] = entry
    return {
        "summary_version": SUMMARY_VERSION,
        "generated_at": iso(utc_now()),
        "run_dir": run_dir.name,
        "sheets_dir": str(sheets_dir),
        "variants_checked": bool(known),
        "prompt_id_mismatches": n_bad_ids,
        "files": files,
        "languages": per_lang,
        "warnings": warnings,
    }


# ------------------------------------------------------------------ rendering


def _f(stat: dict | None, pct: bool = False) -> str:
    if not stat:
        return "-"
    v = stat.get("pct_Y") if pct else stat.get("mean")
    return "-" if v is None else (f"{v:.0f}%" if pct else f"{v:.2f}")


def _several(by_reviewer: dict) -> list:
    """Per-reviewer rows only when there is more than one reviewer (else they repeat ALL)."""
    return list(by_reviewer.items()) if len(by_reviewer) > 1 else []


def table_rows(summary: dict) -> list[list[str]]:
    """lang, sheet, reviewer, rated, adequacy, fluency, intent Y, roman, cm natural, %1, (UNSAFE) cm nat, %1, intent Y."""
    out = []
    for lang, e in summary["languages"].items():
        t = e[TRANSLATION]
        if t["status"] != "ok":
            out.append([lang, "translation", t["status"], *["-"] * 10])
        else:
            for who, s in [("ALL", t["overall"]), *_several(t["by_reviewer"])]:
                out.append([lang, "translation", who, f"{s['n_rated']}/{s['n_total']}",
                            _f(s["translation_adequacy_1to5"]), _f(s["translation_fluency_1to5"]),
                            _f(s["intent_preserved"], True), _f(s["romanisation_natural_1to5"]), *["-"] * 5])
        c = e[CODEMIX]
        if c["status"] != "ok":
            out.append([lang, "codemix", c["status"], *["-"] * 10])
            continue
        for who, s in [("ALL", c["overall"]), *_several(c["by_reviewer"])]:
            u = c["unsafe_only"]["overall"] if who == "ALL" else c["unsafe_only"]["by_reviewer"].get(who)
            p1 = s["pct_rated_1_unnatural"]
            u1 = u["pct_rated_1_unnatural"] if u else None
            out.append([lang, "codemix", who, f"{s['n_rated']}/{s['n_total']}", "-", "-",
                        _f(s["intent_preserved"], True), "-", _f(s["codemix_natural_1to3"]),
                        "-" if p1 is None else f"{p1:.0f}%",
                        _f(u["codemix_natural_1to3"]) if u else "-", "-" if u1 is None else f"{u1:.0f}%",
                        _f(u["intent_preserved"], True) if u else "-"])
    return out


HEADER = ["lang", "sheet", "reviewer", "rated", "adequacy", "fluency", "intent Y", "roman", "cm natural",
          "cm %1", "UNSAFE cm nat", "UNSAFE %1", "UNSAFE intent Y"]


def render_table(rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(HEADER, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(HEADER), line(["-" * w for w in widths]), *map(line, rows)])


def render_markdown(summary: dict) -> str:
    rows = table_rows(summary)
    md = [f"# Review summary — {summary['run_dir']}", "",
          f"Generated {summary['generated_at']} from `{summary['sheets_dir']}`. "
          f"Prompt-id mismatches: {summary['prompt_id_mismatches']}"
          + ("" if summary["variants_checked"] else " (variants.jsonl not found: not checked)") + ".", "",
          "Means are over rated cells only; `intent Y` is % of answered rows. `cm %1` = % of code-mix rows "
          "rated 1 (unnatural). UNSAFE columns repeat the code-mix figures for UNSAFE seeds only.", "",
          "| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    for lang, e in summary["languages"].items():
        md += ["", f"## {lang}: lowest-rated rows", ""]
        if not e["lowest_rated"]:
            md.append("No rated rows.")
            continue
        md += ["| score | sheet | reviewer | review_id | label | ratings | intent | text | notes |",
               "|---|---|---|---|---|---|---|---|---|"]
        for x in e["lowest_rated"]:
            ratings = ", ".join(f"{SHORT[k]}={v:g}" for k, v in x["ratings"].items())
            intent = {True: "Y", False: "N", None: ""}[x["intent_preserved"]]
            text = (x.get("native_text") or "").replace("|", "/")
            notes = (x.get("notes") or "").replace("|", "/").replace("\n", " ")
            md.append(f"| {x['score_0to1']:.2f} | {x['sheet']} | {x['reviewer']} | {x.get('review_id', '')} | "
                      f"{x['label'] or ''} | {ratings} | {intent} | {text} | {notes} |")
    if summary["warnings"]:
        md += ["", "## Warnings", ""] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(md) + "\n"


def write_summary(summary: dict, run_dir: Path) -> dict[str, Path]:
    jp, mp = run_dir / "review_summary.json", run_dir / "review_summary.md"
    _write_json(jp, summary)
    _atomic_write(mp, render_markdown(summary))
    return {"json": jp, "markdown": mp}
