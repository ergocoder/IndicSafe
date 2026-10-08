"""Import the pilot double-annotation reviews; optionally build the adjudicated pilot.

Usage (from the project root):
    python scripts/import_reviews.py                  # review layer + agreement + worksheet check
    python scripts/import_reviews.py --build          # ... then write the next pilot version
    python scripts/import_reviews.py --build --force  # replace an existing next pilot version

Reads the annotator CSVs, category mapping table and adjudication worksheet named
in the review config (default data/pilot/reviews/reviews_v0.1.yaml). Writes:
    <reviews_dir>/review_layer_<v>.jsonl, <reviews_dir>/review_report_<v>.json
    with --build: <output_dir>/pilot_seeds_<output version>.{jsonl,csv,manifest.json}
--build refuses (exit 2) until every worksheet row has adjudicated_label,
adjudicated_category, adjudicated_by and rationale. The input pilot is never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import ConfigError, load_settings  # noqa: E402
from generator.review_import import (  # noqa: E402
    AdjudicationError,
    ReviewImportError,
    build_reviewed_pilot,
    import_reviews,
    load_review_config,
    write_review_layer,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("data/pilot/reviews/reviews_v0.1.yaml"))
    ap.add_argument("--build", action="store_true", help="write the adjudicated next pilot version")
    ap.add_argument("--force", action="store_true", help="with --build: replace an existing output")
    args = ap.parse_args(argv)

    try:
        settings = load_settings()
        cfg = load_review_config(settings.project_root / args.config)
        result = import_reviews(cfg, settings)
    except (ConfigError, ReviewImportError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Build before rewriting the review report: the report is a tracked file, and
    # rewriting it first would mark the build's git state as dirty.
    built, build_error = None, None
    if args.build:
        try:
            built = build_reviewed_pilot(result, settings, force=args.force)
        except AdjudicationError as e:
            build_error = e
    paths = write_review_layer(result, settings)

    lab, cat = result.agreement["label"], result.agreement["category"]
    print(f"seeds: {lab['n_seeds']}   annotators: {', '.join(result.agreement['annotators'])}")
    print(f"label agreement:    {lab['n_agree']}/{lab['n_compared']} = {lab['percent_agreement']}%   "
          f"Cohen's kappa = {lab['cohen_kappa']}")
    print(f"category agreement: {cat['n_agree']}/{cat['n_compared']} = {cat['percent_agreement']}%   "
          f"(both cells resolved)   kappa = {cat['cohen_kappa']}")
    for r in result.layer:
        for name, ann in r["annotations"].items():
            if ann["flags"]:
                print(f"  flag {r['seed_id']:22} {name:10} {','.join(ann['flags'])}  "
                      f"raw_category={ann['raw_category']!r}")
    print(f"needs adjudication: {len(result.agreement['needs_adjudication'])} seed(s)")
    check = result.worksheet_check
    if check["matches"]:
        print(f"worksheet: matches ({check['worksheet_rows']} rows)")
    else:
        print("worksheet: DIFFERS from the computed disagreement list")
        for sid in check["missing_from_worksheet"]:
            print(f"  missing from worksheet: {sid}")
        for sid in check["extra_in_worksheet"]:
            print(f"  extra in worksheet:     {sid}")
        for d in check["field_differences"]:
            print(f"  {d['seed_id']} {d['column']}: worksheet={d['worksheet']!r} computed={d['computed']!r}")
    print(f"layer:  {paths['layer']}\nreport: {paths['report']}")

    if build_error is not None:
        print(f"NOT BUILT: {build_error}", file=sys.stderr)
        for p in build_error.problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    if built is not None:
        print(f"pilot:  {built['jsonl']}\n        manifest: {built['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
