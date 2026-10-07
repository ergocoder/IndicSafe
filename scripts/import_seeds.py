"""Import seeds from the registered raw sources and export the pilot seed set.

Usage (from the project root):
    python scripts/import_seeds.py                 # import + pool + pilot
    python scripts/import_seeds.py --no-pilot      # import + pool only
    python scripts/import_seeds.py --manual-csv data/manual/seeds.csv
    python scripts/import_seeds.py --force         # replace an existing pilot selection

Reads data/raw/*.zip read-only. Writes:
    data/processed/seed_pool.jsonl, data/processed/import_report.json
    data/pilot/pilot_seeds_<version>.{jsonl,csv,manifest.json}
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import ConfigError, load_settings  # noqa: E402
from generator.provenance import SourceIntegrityError  # noqa: E402
from generator.seed_manager import ExportError, PilotSelectionError, SeedManager  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*", help="source ids to import (default: all seed-eligible)")
    ap.add_argument("--manual-csv", type=Path, help="optional team-authored seed CSV")
    ap.add_argument("--pool-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--pilot-dir", type=Path, default=Path("data/pilot"))
    ap.add_argument("--no-pilot", action="store_true")
    ap.add_argument("--force", action="store_true", help="replace an existing, different pilot selection")
    args = ap.parse_args(argv)

    try:
        settings = load_settings()
        mgr = SeedManager(settings)
        result = mgr.import_sources(args.sources, manual_csv=args.manual_csv)
        pool = mgr.export_pool(result, args.pool_dir)
        print(f"run_id: {result.run_id}")
        for sid, st in result.stats.items():
            print(f"  {sid:16} read={st.rows_read:5} filtered={st.filtered_out:5} valid={st.valid:5} "
                  f"duplicate={st.duplicate:3} rejected={st.rejected:3} "
                  f"parse_failures={len(st.parse_failures)} row_errors={len(st.row_errors)}")
            if st.rejection_reasons:
                print(f"  {'':16} rejection reasons: {dict(st.rejection_reasons)}")
        print(f"pool:   {pool['pool']}\nreport: {pool['report']}")

        if not args.no_pilot:
            pilot = mgr.select_pilot(result.seeds)
            paths = mgr.export_pilot(pilot, result, args.pilot_dir, force=args.force)
            print(f"pilot:  {len(pilot)} seeds -> {paths['jsonl']}")
            print(f"        by source: {dict(Counter(s.source_dataset for s in pilot))}")
            print(f"        by intended_label: {dict(Counter(s.intended_label for s in pilot))}")
            print(f"        manifest: {paths['manifest']}")
    except (ConfigError, SourceIntegrityError, PilotSelectionError, ExportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
