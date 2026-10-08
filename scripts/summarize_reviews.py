"""Summarise filled native-speaker review sheets for one pilot run.

Usage (from the project root):
    python scripts/summarize_reviews.py --run data/pilot/translations/<run_id> --sheets <folder>

Reads review_<lang>*.csv and review_codemix_<lang>*.csv from --sheets (several
reviewers per language allowed), checks every prompt id against the run's
variants.jsonl, and writes review_summary.json + review_summary.md into the
run folder. Missing sheets and partly filled rows are reported, not fatal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import ConfigError, load_settings, resolve_inside  # noqa: E402
from generator.review_summary import render_table, summarize_reviews, table_rows, write_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder (variants.jsonl; outputs go here)")
    ap.add_argument("--sheets", type=Path, required=True, help="folder with the filled review CSVs")
    args = ap.parse_args(argv)
    try:
        settings = load_settings()
        run_dir = resolve_inside(settings.project_root, args.run)
        if not run_dir.is_dir():
            raise ConfigError(f"run folder {run_dir} not found")
        sheets = args.sheets if args.sheets.is_absolute() else settings.project_root / args.sheets
        summary = summarize_reviews(settings, run_dir, sheets.resolve())
        paths = write_summary(summary, run_dir)
    except (ConfigError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(render_table(table_rows(summary)))
    print(f"\nfiles read: {len(summary['files'])}   prompt-id mismatches: {summary['prompt_id_mismatches']}"
          f"   warnings: {len(summary['warnings'])}")
    for w in summary["warnings"][:15]:
        print(f"  ! {w}")
    if len(summary["warnings"]) > 15:
        print(f"  ... {len(summary['warnings']) - 15} more in review_summary.md")
    for k, p in paths.items():
        print(f"  {k:8} {p.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
