"""Translate + romanise the pilot seeds and export native-speaker review sheets.

Usage (from the project root):
    python scripts/run_pilot_translation.py                      # v0.1 pilot, hi mr gu
    python scripts/run_pilot_translation.py --seeds data/pilot/pilot_seeds_v0.2-pilot-seeds.jsonl
    python scripts/run_pilot_translation.py --limit 3 --languages hi   # quick check

Providers come from configs/generation.yaml (translation: indictrans2,
transliteration: colloquial_roman, qc.language). Writes
<out>/<run_id>/{variants,transformations,language_qc}.jsonl, manifest.json,
review_<lang>.csv and pilot_translation_summary.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generator.providers  # noqa: E402,F401  (registers the real adapters)
from backend.config import ConfigError, load_settings  # noqa: E402
from generator.language_qc import build_language_identifier  # noqa: E402
from generator.pilot_translation import run_pilot_translation, write_outputs  # noqa: E402
from generator.seed_manager import load_seeds_jsonl  # noqa: E402
from generator.transformation_engine import ProviderUnavailableError  # noqa: E402
from generator.translation import build_translation_provider  # noqa: E402
from generator.transliteration import build_transliterator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=Path, default=Path("data/pilot/pilot_seeds_v0.1-pilot-seeds.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/pilot/translations"))
    ap.add_argument("--languages", nargs="+", default=["hi", "mr", "gu"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N seeds")
    args = ap.parse_args(argv)

    try:
        settings = load_settings()
        seeds_path = settings.project_root / args.seeds
        seeds = load_seeds_jsonl(seeds_path)[: args.limit]
        t0 = time.perf_counter()
        mt = build_translation_provider(settings)
        print(f"translation: {mt.info.name} {mt.info.version} {mt.info.model}")
        print(f"  device={mt.backend.device} dtype={mt.backend.dtype} batch_size={mt.opts.batch_size} "
              f"preprocessor={mt.preprocessor}  (loaded in {time.perf_counter() - t0:.1f}s)")
        tl = build_transliterator(settings)
        lid = build_language_identifier(settings)
        print(f"transliteration: {tl.info.name} {tl.info.version}   language id: {lid.name}-{lid.version}")
        t0 = time.perf_counter()
        result = run_pilot_translation(settings, seeds, mt, tl, lid, args.languages)
        print(f"{len(seeds)} seeds x {len(args.languages)} languages in {time.perf_counter() - t0:.1f}s")
        paths = write_outputs(result, settings, args.out, seeds_path=seeds_path)
    except (ConfigError, ProviderUnavailableError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    from collections import Counter
    print("transformations:", dict(Counter(f"{t.transformation_type}:{t.status}"
                                         for t in result.engine.transformations.values())))
    print("language QC:", dict(sorted(Counter(f"{q.expected_language}/{q.expected_script}:{q.language_qc_status}"
                                              for q in result.qc.values()).items())))
    for k, p in paths.items():
        print(f"  {k:16} {p.relative_to(settings.project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
