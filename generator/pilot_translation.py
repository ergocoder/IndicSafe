"""Pilot translation run: seeds → en root → hi/mr/gu native → Latn, code-mixed L1/L2
(native + Latn), plus language QC and native-speaker review sheets.

    for each language:  translate (batched, cached)  ─► engine.apply per root
    for each native child that passed validation:     ─► transliteration
                                                      ─► code_mixing per level (if a mixer is given)
    for each code-mixed child that passed validation: ─► transliteration
    every variant                                     ─► language_qc.check_variant

Outputs in <out_dir>/<run_id>/ (engine export plus):
    language_qc.jsonl         one LanguageQCRecord per variant
    review_<lang>.csv         one row per seed: English, native, Latin, auto-QC,
                              blank reviewer columns (utf-8-sig for Excel)
    pilot_translation_summary.json   inputs, providers, counts, output checksums

Input: a reviewed pilot (v0.2: human final labels) via `load_pilot_seeds`;
every variant inherits the seed's human label as `intended_label`. Variant
ids depend only on prompt text, so unchanged seeds keep their ids.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import Settings
from generator.code_mixing import CodeMixer, CodeMixingTransformation
from generator.language_qc import LanguageIdentifier, LanguageQCRecord, check_variant
from generator.provenance import sha256_file
from generator.schemas import SeedRecord, VariantRecord
from generator.seed_manager import _atomic_write, _write_json
from generator.transformation_engine import TransformationEngine
from generator.translation import TranslationProvider, TranslationTransformation
from generator.transliteration import Transliterator, TransliterationTransformation

def load_pilot_seeds(path: Path) -> tuple[list[SeedRecord], str | None]:
    """Seeds plus dataset_version. Reviewed records (final_label set) become generation seeds whose
    intended_label is the human final label; the review block stays in the pilot file."""
    seeds, version = [], None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            version = d.pop("dataset_version", version)
            d.pop("parent_dataset_version", None)
            d.pop("review", None)
            final = d.get("final_label")
            if final is not None:
                d["intended_label_basis"] = (
                    f"Human final label of pilot {version} ({d['label_status']}); "
                    f"provisional source-derived label was {d['intended_label']}.")
                d["intended_label"], d["final_label"] = final, None
            seeds.append(SeedRecord.model_validate(d))
    return seeds, version


REVIEW_COLUMNS = [
    "review_id", "seed_id", "language", "provisional_label", "category", "source_prompt_en",
    "native_prompt_id", "native_text", "native_qc", "latin_prompt_id", "latin_text", "latin_qc",
    "auto_flags",
    # filled by the reviewer
    "reviewer", "translation_adequacy_1to5", "translation_fluency_1to5", "intent_preserved_Y_N",
    "romanisation_natural_1to5", "corrected_native", "corrected_latin", "notes",
]


@dataclass
class PilotTranslationResult:
    engine: TransformationEngine
    roots: dict[str, VariantRecord]                       # seed_id -> root
    native: dict[tuple[str, str], VariantRecord | None]   # (seed_id, lang) -> native child
    latin: dict[tuple[str, str], VariantRecord | None]
    qc: dict[str, LanguageQCRecord]                       # prompt_id -> record
    languages: list[str]
    lid_detector: str
    # (seed_id, lang, level, script) -> code-mixed variant (None: not produced)
    code_mixed: dict[tuple[str, str, str, str], VariantRecord | None] = field(default_factory=dict)


def run_pilot_translation(
    settings: Settings,
    seeds: Iterable[SeedRecord],
    translator: TranslationProvider,
    transliterator: Transliterator,
    lid: LanguageIdentifier,
    languages: list[str],
    *,
    engine: TransformationEngine | None = None,
    code_mixer: CodeMixer | None = None,
) -> PilotTranslationResult:
    engine = engine or TransformationEngine(settings)
    roots = {s.seed_id: engine.root(s).variant for s in seeds}
    mt, tl = TranslationTransformation(translator), TransliterationTransformation(transliterator)
    cm = CodeMixingTransformation(code_mixer, engine.variants) if code_mixer else None
    levels = settings.generation.code_mixing.levels
    native: dict = {}
    latin: dict = {}
    code_mixed: dict = {}
    for lang in languages:
        script = settings.languages.languages[lang].native_script
        # one batched pass; the engine's per-item calls are then served from the provider cache
        try:
            translator.translate_batch([r.prompt for r in roots.values()], source_language="en",
                                       target_language=lang, target_script=script,
                                       seeds=[0] * len(roots))
        except Exception:  # noqa: BLE001 - per-item calls below record the failure properly
            pass
        for sid, root in roots.items():
            res = engine.apply(root, mt, {"target_language": lang})
            native[(sid, lang)] = res.variant
            latin[(sid, lang)] = None
            if res.variant is not None and res.variant.validation_status != "FAIL":
                latin[(sid, lang)] = engine.apply(res.variant, tl).variant
        if cm is None:
            continue
        script = settings.languages.languages[lang].native_script
        code_mixer.prepare([r.prompt for r in roots.values()], lang)
        for sid in roots:
            parent = native[(sid, lang)]
            for level in levels:
                mixed = None
                if parent is not None and parent.validation_status != "FAIL":
                    mixed = engine.apply(parent, cm, {"level": level}).variant
                code_mixed[(sid, lang, level, script)] = mixed
                code_mixed[(sid, lang, level, "Latn")] = (
                    engine.apply(mixed, tl).variant
                    if mixed is not None and mixed.validation_status != "FAIL" else None)
    qc = {pid: check_variant(settings, v, lid) for pid, v in engine.variants.items()}
    return PilotTranslationResult(engine, roots, native, latin, qc, list(languages),
                                  f"{lid.name}-{lid.version}", code_mixed)


def _qc_cell(v: VariantRecord | None, qc: dict[str, LanguageQCRecord]) -> str:
    if v is None:
        return "MISSING"
    q = qc[v.prompt_id]
    lid = f" lid={q.lid_language}@{q.lid_confidence}" if q.lid_language else ""
    return f"{q.language_qc_status} script={q.measured_script}@{q.script_share}{lid}"


def review_rows(result: PilotTranslationResult, lang: str) -> list[dict]:
    rows = []
    trans = result.engine.transformations
    for sid, root in result.roots.items():
        n, l = result.native[(sid, lang)], result.latin[(sid, lang)]
        flags = []
        for v in (n, l):
            if v is not None:
                flags += v.validation_failures
                flags += [f"{v.script or '?'}:{r}" for r in result.qc[v.prompt_id].reasons]
                t = trans[v.transformation_id]
                flags += [f"{r.hook}:{r.reason}" for r in t.validation_results if r.status == "WARN"]
                if t.provider_metadata.get("hit_max_new_tokens"):
                    flags.append("possibly_truncated")
        if n is None:
            errs = [t.error for t in trans.values() if t.seed_id == sid and t.transformation_type == "translation"
                    and t.parameters.get("target_language") == lang and t.error]
            flags += errs or ["translation_missing"]
        rows.append({
            "review_id": f"{sid}:{lang}", "seed_id": sid, "language": lang,
            "provisional_label": root.intended_label, "category": root.category,
            "source_prompt_en": root.prompt,
            "native_prompt_id": n.prompt_id if n else "", "native_text": n.prompt if n else "",
            "native_qc": _qc_cell(n, result.qc),
            "latin_prompt_id": l.prompt_id if l else "", "latin_text": l.prompt if l else "",
            "latin_qc": _qc_cell(l, result.qc),
            "auto_flags": "; ".join(dict.fromkeys(flags)),
        })
    return rows


def write_outputs(result: PilotTranslationResult, settings: Settings, out_dir: Path,
                  *, seeds_path: Path | None = None, dataset_version: str | None = None) -> dict[str, Path]:
    paths = result.engine.export(out_dir)
    run_dir = paths["manifest"].parent
    qc_path = run_dir / "language_qc.jsonl"
    _atomic_write(qc_path, "".join(json.dumps(q.model_dump(mode="json"), ensure_ascii=False) + "\n"
                                   for q in result.qc.values()))
    paths["language_qc"] = qc_path
    for lang in result.languages:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(review_rows(result, lang))
        p = run_dir / f"review_{lang}.csv"
        _atomic_write(p, buf.getvalue(), encoding="utf-8-sig")  # Excel shows Devanagari/Gujarati
        paths[f"review_{lang}"] = p

    trans = result.engine.transformations.values()
    summary = {
        "run_id": result.engine.run_id,
        "input_seeds": {"path": str(seeds_path) if seeds_path else None,
                        "sha256": sha256_file(seeds_path) if seeds_path else None,
                        "dataset_version": dataset_version,
                        "n": len(result.roots),
                        "intended_label_counts": dict(sorted(Counter(
                            r.intended_label for r in result.roots.values()).items()))},
        "languages": result.languages,
        "providers": sorted({f"{t.transformation_type}: {t.provider} {t.provider_version} {t.generator_model}"
                             for t in trans}),
        "language_qc": {
            "detector": result.lid_detector,
            "config": settings.generation.qc.language.model_dump(),
            "script_thresholds": settings.generation.qc.script.model_dump(),
            "status_by_variant_kind": dict(sorted(Counter(
                f"{q.expected_language}/{q.expected_script}:{q.language_qc_status}" for q in result.qc.values()
            ).items())),
        },
        "transformations_by_type_status": dict(sorted(Counter(
            f"{t.transformation_type}:{t.status}" for t in trans).items())),
        "code_mix": {
            "levels": settings.generation.code_mixing.levels,
            "bands": {k: [v.min_ratio, v.max_ratio] for k, v in settings.languages.code_mix_levels.items()},
            "tolerance": settings.generation.qc.code_mix.tolerance,
            "slots_by_lang_level_script_status": dict(sorted(Counter(
                f"{lang}/{level}/{script}:{v.validation_status if v else 'NOT_PRODUCED'}"
                for (_, lang, level, script), v in result.code_mixed.items()).items())),
        },
        "outputs": {p.name: sha256_file(p) for k, p in paths.items() if k != "manifest"},
    }
    sp = run_dir / "pilot_translation_summary.json"
    _write_json(sp, summary)
    paths["summary"] = sp
    return paths
