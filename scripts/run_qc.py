"""Phase 5 QC report for one transformation run folder.

Usage (from the project root):
    python scripts/run_qc.py --run data/pilot/translations/<run_id>
    python scripts/run_qc.py --run <dir> --no-semantic      # skip the sentence encoder

Writes <run>/qc_report.jsonl (one record per variant: per-check status, reasons),
<run>/qc_summary.json (counts, code-mix coverage), review_codemix_<lang>.csv
(native-speaker sheets) and harmful_intent_check.json. Thresholds and the encoder come from
configs/generation.yaml → qc.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import ConfigError, load_settings, resolve_inside  # noqa: E402
from generator.qc_pipeline import run_qc_on_dir  # noqa: E402
from generator.semantic import EncoderUnavailableError, build_encoder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder with variants.jsonl etc.")
    ap.add_argument("--no-semantic", action="store_true")
    args = ap.parse_args(argv)
    try:
        settings = load_settings()
        run_dir = resolve_inside(settings.project_root, args.run)
        encoder = None
        if not args.no_semantic:
            t0 = time.perf_counter()
            encoder = build_encoder(settings)
            print(f"semantic encoder: {encoder.name} {encoder.version} (loaded in {time.perf_counter() - t0:.1f}s)")
        t0 = time.perf_counter()
        paths = run_qc_on_dir(settings, run_dir, encoder)
        print(f"QC done in {time.perf_counter() - t0:.1f}s")
    except (ConfigError, EncoderUnavailableError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    import json
    s = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print("qc_status:", s["qc_status"])
    for kind, c in s["qc_status_by_kind"].items():
        print(f"  {kind:16} {c}")
    print("top reasons:", dict(list(s["reasons"].items())[:10]))
    for level, c in s["code_mix_coverage"].items():
        print(f"  {c['line']}  (near band edge {c['near_band_edge']}, missed {c['missed']})")
    print("final dataset:", s["final_dataset"])
    for k, p in paths.items():
        print(f"  {k:8} {p.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
