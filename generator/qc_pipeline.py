"""QC pipeline (Phase 5): one QC record per variant of a transformation run, plus a summary.

Inputs (a run folder from scripts/run_pilot_translation.py):
variants.jsonl, transformations.jsonl, language_qc.jsonl.
Outputs, in the same folder: qc_report.jsonl and qc_summary.json.

Checks per variant. Each one is PASS / REVIEW / FAIL / NOT_APPLICABLE / NOT_RUN,
with a reason code and details. Thresholds come from generation.yaml → qc.

    engine_hooks     engine validation FAILs other than script / length / band
                     (empty output, text integrity, condition_not_realised)
    exact_duplicate  same content_hash as another variant -> FAIL; the one with
                     the shorter lineage, then the lower level, keeps PASS. Catches
                     a code-mix level that changed nothing (L2 == L1, or a
                     romanised L1 == romanised L0).
    near_duplicate   char-3-gram Jaccard >= qc.duplicate.near_dup_char3_jaccard
                     with a variant of ANOTHER seed in the same language, script
                     and level -> FAIL. Variants of one seed are meant to be
                     similar, so they are not compared with each other.
    script_language  language_qc record (script share + LID), unchanged
    code_mix         code_mix_ratio and CMI measured on every target-language
                     variant (code_mix_metrics). With a level: band check, where
                     a ratio outside the band by <= tolerance is REVIEW and beyond
                     it is FAIL. Without a level (monolingual L0): a ratio above
                     the L0 band is REVIEW (the MT kept English words).
    length_ratio     chars(variant) / chars(parent) outside
                     transformation_engine.length_ratio -> REVIEW
    semantic         cosine(English seed, variant) with the sentence encoder
                     (generator/semantic.py). >= pass is PASS, >= review is
                     REVIEW, otherwise FAIL semantic_drift. Code-mixed variants
                     are also scored against their native L0 parent and that
                     score decides (English words shared with the seed inflate
                     the seed score); both are in the details. Romanised
                     variants inherit their native parent's scores.

qc_status = FAIL if any check FAILs, else REVIEW if any REVIEWs, else PASS.
The QC never edits a variant and never relabels a level. FAIL variants never
enter a final dataset: `final_dataset_variants` is the one filter any export
must use.

Also written by `run_qc_on_dir`:
    review_codemix_<lang>.csv   native-speaker sheet for L1/L2 code-mix
                                (QC REVIEW items first, then a fixed-seed sample)
    harmful_intent_check.json   lowest-similarity UNSAFE / AMBIGUOUS variants
"""

from __future__ import annotations

import csv
import io
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.config import Settings
from generator import code_mix_metrics as cmm
from generator.language_qc import LanguageQCRecord
from generator.provenance import iso, sha256_file, utc_now
from generator.schemas import TransformationRecord, VariantRecord
from generator.seed_manager import _atomic_write, _write_json
from generator.semantic import SentenceEncoder, cosine
from generator.text_utils import dedup_key
from generator.transformation_engine import (
    load_transformations_jsonl,
    load_variants_jsonl,
    measure_code_mix,
)

QC_VERSION = "1.1"
Status = Literal["PASS", "REVIEW", "FAIL", "NOT_APPLICABLE", "NOT_RUN"]
_OWN_CHECK_HOOKS = {"expected_script", "length_ratio", "code_mix_band"}  # reported by their own check


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    reason: str | None = None
    details: dict = {}


class QCRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    seed_id: str
    parent_prompt_id: str | None
    lineage: list[str]
    transformation_type: str
    language: str
    secondary_language: str | None
    script: str | None
    is_transliterated: bool
    code_mix_level: str | None
    intended_label: str
    qc_version: str = QC_VERSION
    checks: dict[str, Check]
    code_mix_ratio: float | None = None
    cmi: float | None = None
    semantic_similarity: float | None = None             # vs the English seed
    semantic_similarity_to_parent: float | None = None   # code-mix: vs the native L0 parent (decides)
    qc_status: Literal["PASS", "REVIEW", "FAIL"]
    reasons: list[str] = []


def char3(text: str) -> frozenset[str]:
    t = f" {dedup_key(text)} "
    return frozenset(t[i:i + 3] for i in range(len(t) - 2))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _variant_kind(v: VariantRecord) -> tuple[str, str | None, str | None]:
    return v.language, v.script, v.code_mix_level


# --------------------------------------------------------------- checks


def engine_hooks_check(t: TransformationRecord) -> Check:
    fails = [f"{r.hook}:{r.reason}" for r in t.validation_results
             if r.status == "FAIL" and r.hook not in _OWN_CHECK_HOOKS]
    if t.status == "ERROR":
        return Check(status="FAIL", reason="transformation_error", details={"error": t.error})
    if fails:
        return Check(status="FAIL", reason=fails[0].split(":", 1)[1], details={"failures": fails})
    return Check(status="PASS")


def duplicate_checks(variants: list[VariantRecord], threshold: float) -> tuple[dict[str, Check], dict[str, Check]]:
    exact: dict[str, Check] = {}
    first_by_hash: dict[str, str] = {}
    for v in sorted(variants, key=lambda v: (len(v.lineage), v.code_mix_level or "", v.prompt_id)):   # shallower, then lower level kept
        first = first_by_hash.setdefault(v.content_hash, v.prompt_id)
        exact[v.prompt_id] = (Check(status="PASS") if first == v.prompt_id else
                              Check(status="FAIL", reason="exact_duplicate", details={"duplicate_of": first}))
    near: dict[str, Check] = {}
    groups: dict[tuple, list[VariantRecord]] = defaultdict(list)
    for v in variants:
        groups[_variant_kind(v)].append(v)
    for group in groups.values():
        group.sort(key=lambda v: (v.seed_id, v.prompt_id))
        grams = {v.prompt_id: char3(v.prompt) for v in group}
        for i, v in enumerate(group):
            best, best_sim = None, 0.0
            for u in group[:i]:
                if u.seed_id == v.seed_id:
                    continue
                sim = jaccard(grams[v.prompt_id], grams[u.prompt_id])
                if sim > best_sim:
                    best, best_sim = u.prompt_id, sim
            details = {"max_char3_jaccard_other_seeds": round(best_sim, 4), "closest": best, "threshold": threshold}
            near[v.prompt_id] = (Check(status="FAIL", reason="near_duplicate", details=details)
                                 if best is not None and best_sim >= threshold else Check(status="PASS", details=details))
    return exact, near


def script_language_check(q: LanguageQCRecord | None) -> Check:
    if q is None:
        return Check(status="NOT_RUN", reason="no_language_qc_record")
    status = {"PASS": "PASS", "REVIEW": "REVIEW", "FAIL": "FAIL"}[q.language_qc_status]
    return Check(status=status, reason=q.reasons[0] if q.reasons else None,
                 details={"script": q.measured_script, "script_share": q.script_share,
                          "lid_language": q.lid_language, "lid_confidence": q.lid_confidence,
                          "lid_status": q.lid_status, "reasons": q.reasons})


def code_mix_check(settings: Settings, v: VariantRecord, parent: VariantRecord | None) -> Check:
    lang = settings.languages.languages[v.language]
    if lang.code_mix_partner is None:
        return Check(status="NOT_APPLICABLE", reason="no_code_mix_partner")
    m = measure_code_mix(settings, v.prompt, v.language, v.is_transliterated, parent.prompt if parent else None)
    bands = cmm.level_bands(settings)
    tol = settings.generation.qc.code_mix.tolerance
    if v.code_mix_level is not None:
        status, reason, details = cmm.band_check(m.ratio if m else None, v.code_mix_level, bands, tol)
        status = "REVIEW" if status == "WARN" else status
    else:
        details = {"level": None, "band": list(bands["L0"]) if "L0" in bands else None}
        if m is None:
            status, reason = "REVIEW", "code_mix_unmeasurable"
        elif "L0" in bands and m.ratio is not None and m.ratio >= bands["L0"][1]:
            status, reason = "REVIEW", "monolingual_variant_contains_partner_words"
        else:
            status, reason = "PASS", None
    if m is not None:
        details = {**details, **m.as_dict(),
                   "measured_level": cmm.level_for_ratio(m.ratio, bands) if m.ratio is not None else None}
    return Check(status=status, reason=reason, details=details)


def length_ratio_check(settings: Settings, v: VariantRecord, parent: VariantRecord | None) -> Check:
    if parent is None:
        return Check(status="NOT_APPLICABLE", reason="root")
    b = settings.generation.transformation_engine.length_ratio
    r = round(len(v.prompt) / max(len(parent.prompt), 1), 4)
    details = {"ratio": r, "min": b.min, "max": b.max, "parent_prompt_id": parent.prompt_id}
    if not b.min <= r <= b.max:
        return Check(status="REVIEW", reason="length_ratio_out_of_range", details=details)
    return Check(status="PASS", details=details)


def semantic_scores(variants: Mapping[str, VariantRecord], encoder: SentenceEncoder) -> dict[str, dict]:
    """Score native-script, non-root variants against their English root (one encoder pass)."""
    native = [v for v in variants.values() if v.parent_prompt_id is not None and not v.is_transliterated]
    roots = {v.lineage[0] for v in native}
    texts = sorted({variants[r].prompt for r in roots} | {v.prompt for v in native})
    emb = dict(zip(texts, encoder.encode(texts), strict=True))
    out = {}
    for v in native:
        root = variants[v.lineage[0]]
        parent = variants[v.parent_prompt_id]
        s = {"similarity_to_seed": cosine(emb[root.prompt], emb[v.prompt]), "method": "encoded",
             "seed_prompt_id": root.prompt_id}
        if parent.parent_prompt_id is not None:   # code-mixed: also against the L0 translation
            s["similarity_to_parent"] = cosine(emb[parent.prompt], emb[v.prompt])
        out[v.prompt_id] = s
    return out


def semantic_check(settings: Settings, v: VariantRecord, scores: dict[str, dict] | None,
                   variants: Mapping[str, VariantRecord]) -> Check:
    if scores is None:
        return Check(status="NOT_RUN", reason="semantic_check_disabled")
    if v.parent_prompt_id is None:
        return Check(status="NOT_APPLICABLE", reason="root")
    s = scores.get(v.prompt_id)
    if s is None and v.is_transliterated:
        ps = scores.get(v.parent_prompt_id)
        if ps is not None:
            s = {**ps, "method": "inherited_from_native_parent", "inherited_from": v.parent_prompt_id}
    if s is None:
        return Check(status="NOT_APPLICABLE", reason="no_native_parent_score")
    cfg = settings.generation.qc.semantic
    use_parent = v.code_mix_level is not None and s.get("similarity_to_parent") is not None
    sim = s["similarity_to_parent"] if use_parent else s["similarity_to_seed"]
    details = {**s, "decision_basis": "native_l0_parent" if use_parent else "english_seed",
               "pass": cfg.pass_, "review": cfg.review}
    if sim >= cfg.pass_:
        return Check(status="PASS", details=details)
    if sim >= cfg.review:
        return Check(status="REVIEW", reason="semantic_similarity_low", details=details)
    return Check(status="FAIL", reason="semantic_drift", details=details)


# ---------------------------------------------------------------- pipeline


def run_qc(settings: Settings, variants: Mapping[str, VariantRecord],
           transformations: Mapping[str, TransformationRecord],
           language_qc: Mapping[str, LanguageQCRecord], encoder: SentenceEncoder | None) -> list[QCRecord]:
    vs = sorted(variants.values(), key=lambda v: v.prompt_id)
    exact, near = duplicate_checks(vs, settings.generation.qc.duplicate.near_dup_char3_jaccard)
    scores = semantic_scores(variants, encoder) if encoder is not None else None
    out = []
    for v in vs:
        parent = variants.get(v.parent_prompt_id) if v.parent_prompt_id else None
        checks = {
            "engine_hooks": engine_hooks_check(transformations[v.transformation_id]),
            "exact_duplicate": exact[v.prompt_id],
            "near_duplicate": near[v.prompt_id],
            "script_language": script_language_check(language_qc.get(v.prompt_id)),
            "code_mix": code_mix_check(settings, v, parent),
            "length_ratio": length_ratio_check(settings, v, parent),
            "semantic": semantic_check(settings, v, scores, variants),
        }
        statuses = [c.status for c in checks.values()]
        overall = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
        reasons = [f"{name}:{c.reason}" for name, c in checks.items() if c.status in ("FAIL", "REVIEW")]
        cm = checks["code_mix"].details
        out.append(QCRecord(
            prompt_id=v.prompt_id, seed_id=v.seed_id, parent_prompt_id=v.parent_prompt_id, lineage=v.lineage,
            transformation_type=v.transformation_type, language=v.language,
            secondary_language=v.secondary_language, script=v.script, is_transliterated=v.is_transliterated,
            code_mix_level=v.code_mix_level, intended_label=v.intended_label, checks=checks,
            code_mix_ratio=cm.get("code_mix_ratio"), cmi=cm.get("cmi"),
            semantic_similarity=checks["semantic"].details.get("similarity_to_seed"),
            semantic_similarity_to_parent=checks["semantic"].details.get("similarity_to_parent"),
            qc_status=overall, reasons=reasons,
        ))
    return out


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), 4), "median": round(statistics.median(xs), 4),
            "mean": round(statistics.fmean(xs), 4), "max": round(max(xs), 4)}


def _kind(r: QCRecord) -> str:
    return f"{r.language}/{r.script}/{r.code_mix_level or 'L0'}"


def code_mix_coverage(records: list[QCRecord], transformations: Mapping[str, TransformationRecord]) -> dict:
    """Per level (and language): code-mixing attempts vs native variants inside their band."""
    by_out = {r.prompt_id: r for r in records}
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for t in transformations.values():
        if t.transformation_type != "code_mixing":
            continue
        level, lang = t.parameters.get("level"), t.parameters.get("language")
        r = by_out.get(t.output_prompt_id or "")
        status = r.checks["code_mix"].status if r else "ERROR"
        for key in ((level, "all"), (level, lang)):
            c = cells[key]
            c["attempted"] += 1
            c["band_reached"] += status == "PASS"
            c["near_band_edge"] += status == "REVIEW"
            c["missed"] += status not in ("PASS", "REVIEW")
    out: dict = {}
    for (level, lang), c in sorted(cells.items()):
        entry = {**c, "line": f"{level} band reached: {c['band_reached']}/{c['attempted']}"}
        if lang == "all":
            out.setdefault(level, {}).update(entry)
        else:
            out.setdefault(level, {}).setdefault("by_language", {})[lang] = entry
    return out


def final_dataset_variants(variants: Mapping[str, VariantRecord], qc_records: Iterable[QCRecord], *,
                           include_review: bool = True) -> list[VariantRecord]:
    """The only way variants may enter a final dataset: QC FAIL is always excluded,
    REVIEW only when include_review is False. Every variant must have a QC record."""
    qc = {r.prompt_id: r for r in qc_records}
    missing = sorted(set(variants) - set(qc))
    if missing:
        raise ValueError(f"{len(missing)} variant(s) have no QC record, e.g. {missing[:3]}")
    allowed = {"PASS", "REVIEW"} if include_review else {"PASS"}
    return [v for pid, v in sorted(variants.items()) if qc[pid].qc_status in allowed]


def summarize(records: list[QCRecord], settings: Settings, encoder: SentenceEncoder | None) -> dict:
    by_kind: dict[str, Counter] = defaultdict(Counter)
    ratios: dict[str, list[float]] = defaultdict(list)
    cmis: dict[str, list[float]] = defaultdict(list)
    sims: dict[str, list[float]] = defaultdict(list)
    psims: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_kind[_kind(r)][r.qc_status] += 1
        if r.code_mix_ratio is not None:
            ratios[_kind(r)].append(r.code_mix_ratio)
            cmis[_kind(r)].append(r.cmi)
        if r.semantic_similarity is not None and r.checks["semantic"].details.get("method") == "encoded":
            sims[_kind(r)].append(r.semantic_similarity)
            if r.semantic_similarity_to_parent is not None:
                psims[_kind(r)].append(r.semantic_similarity_to_parent)
    qc = settings.generation.qc
    return {
        "qc_version": QC_VERSION,
        "n_variants": len(records),
        "qc_status": dict(sorted(Counter(r.qc_status for r in records).items())),
        "qc_status_by_kind": {k: dict(sorted(c.items())) for k, c in sorted(by_kind.items())},
        "check_status": {name: dict(sorted(Counter(r.checks[name].status for r in records).items()))
                         for name in records[0].checks} if records else {},
        "reasons": dict(Counter(x for r in records for x in r.reasons).most_common()),
        "code_mix_ratio_by_kind": {k: _stats(v) for k, v in sorted(ratios.items())},
        "cmi_by_kind": {k: _stats(v) for k, v in sorted(cmis.items())},
        "semantic_similarity_by_kind": {k: _stats(v) for k, v in sorted(sims.items())},
        "semantic_similarity_to_parent_by_kind": {k: _stats(v) for k, v in sorted(psims.items())},
        "intended_label_counts": dict(sorted(Counter(r.intended_label for r in records).items())),
        "final_dataset": {
            "eligible_including_review": sum(r.qc_status != "FAIL" for r in records),
            "eligible_pass_only": sum(r.qc_status == "PASS" for r in records),
            "excluded_fail": sum(r.qc_status == "FAIL" for r in records),
            "rule": "QC FAIL variants are never exported (qc_pipeline.final_dataset_variants)",
        },
        "config": {
            "duplicate": qc.duplicate.model_dump(), "code_mix": qc.code_mix.model_dump(),
            "bands": {k: list(b) for k, b in cmm.level_bands(settings).items()},
            "semantic": qc.semantic.model_dump(by_alias=True),
            "length_ratio": settings.generation.transformation_engine.length_ratio.model_dump(),
            "script": qc.script.model_dump(), "language": qc.language.model_dump(),
        },
        "semantic_encoder": ({"name": encoder.name, "version": encoder.version,
                              **getattr(encoder, "versions", {})} if encoder else None),
    }


# ------------------------------------------------------ review sheets / reports

CODEMIX_REVIEW_COLUMNS = [
    "review_id", "selection", "seed_id", "language", "level", "intended_label", "category",
    "source_prompt_en", "l0_native_text", "native_prompt_id", "native_text", "latin_prompt_id", "latin_text",
    "code_mix_ratio", "cmi", "swapped", "qc_status", "qc_reasons",
    # filled by the reviewer
    "reviewer", "codemix_natural_1to3", "intent_preserved_Y_N", "notes",
]


def codemix_review_rows(settings: Settings, variants: Mapping[str, VariantRecord],
                        transformations: Mapping[str, TransformationRecord],
                        records: list[QCRecord], language: str) -> list[dict]:
    """One row per native L1/L2 variant (with its romanised child); FAIL variants are left out.

    Order: every item with a QC REVIEW (native or romanised) first, then a random sample
    (fixed seed: generation.random_seed + language) up to codemix_sheet_rows, taking UNSAFE
    rows first until codemix_sheet_min_unsafe of them are on the sheet."""
    cfg = settings.generation.qc.human_review
    qc = {r.prompt_id: r for r in records}
    latin_of = {v.parent_prompt_id: v for v in variants.values()
                if v.is_transliterated and v.code_mix_level is not None}
    rows = []
    for v in sorted(variants.values(), key=lambda v: (v.seed_id, v.code_mix_level or "")):
        if v.language != language or v.transformation_type != "code_mixing" or qc[v.prompt_id].qc_status == "FAIL":
            continue
        lat = latin_of.get(v.prompt_id)
        if lat is not None and qc[lat.prompt_id].qc_status == "FAIL":
            lat = None
        statuses = [qc[v.prompt_id].qc_status] + ([qc[lat.prompt_id].qc_status] if lat else [])
        meta = transformations[v.transformation_id].provider_metadata
        rows.append({
            "review_id": f"{v.seed_id}:{language}:{v.code_mix_level}", "seed_id": v.seed_id,
            "language": language, "level": v.code_mix_level, "intended_label": v.intended_label,
            "category": v.category, "source_prompt_en": variants[v.lineage[0]].prompt,
            "l0_native_text": variants[v.parent_prompt_id].prompt,
            "native_prompt_id": v.prompt_id, "native_text": v.prompt,
            "latin_prompt_id": lat.prompt_id if lat else "", "latin_text": lat.prompt if lat else "",
            "code_mix_ratio": qc[v.prompt_id].code_mix_ratio, "cmi": qc[v.prompt_id].cmi,
            "swapped": "; ".join(f"{s['en']} <- {s['native']}" for s in meta.get("swapped", [])),
            "qc_status": "/".join(statuses),
            "qc_reasons": "; ".join(dict.fromkeys(qc[v.prompt_id].reasons + (qc[lat.prompt_id].reasons if lat else []))),
            "_review": "REVIEW" in statuses,
        })
    picked = [r for r in rows if r["_review"]]
    for r in picked:
        r["selection"] = "qc_review"
    rest = [r for r in rows if not r["_review"]]
    rng = random.Random(f"{settings.generation.random_seed}:{language}")
    rng.shuffle(rest)
    need = max(0, cfg.codemix_sheet_rows - len(picked))
    n_unsafe = sum(r["intended_label"] == "UNSAFE" for r in picked)
    unsafe = [r for r in rest if r["intended_label"] == "UNSAFE"][: max(0, min(need, cfg.codemix_sheet_min_unsafe - n_unsafe))]
    others = [r for r in rest if r not in unsafe][: need - len(unsafe)]
    for r in unsafe + others:
        r["selection"] = "random_sample"
    out = picked + sorted(unsafe + others, key=lambda r: r["review_id"])
    for r in out:
        del r["_review"]
    return out


def harmful_intent_check(variants: Mapping[str, VariantRecord], records: list[QCRecord], top_n: int) -> dict:
    """The top_n native-script variants of UNSAFE / AMBIGUOUS seeds with the lowest similarity to the
    English seed: read them to see whether translation or code-mixing softened the request."""
    scored = [r for r in records
              if r.intended_label in ("UNSAFE", "AMBIGUOUS") and r.semantic_similarity is not None
              and r.checks["semantic"].details.get("method") == "encoded"]
    scored.sort(key=lambda r: (r.semantic_similarity, r.prompt_id))
    latin_of = {v.parent_prompt_id: v.prompt for v in variants.values() if v.is_transliterated}
    return {
        "purpose": "Manual check: is the harmful (or ambiguous) request still intact in these variants?",
        "selection": f"{top_n} lowest similarity_to_seed among native-script variants of UNSAFE/AMBIGUOUS seeds",
        "n_candidates": len(scored),
        "items": [{
            "prompt_id": r.prompt_id, "seed_id": r.seed_id, "intended_label": r.intended_label,
            "language": r.language, "script": r.script, "level": r.code_mix_level or "L0",
            "similarity_to_seed": r.semantic_similarity, "similarity_to_parent": r.semantic_similarity_to_parent,
            "qc_status": r.qc_status, "seed_text": variants[r.lineage[0]].prompt,
            "variant_text": variants[r.prompt_id].prompt, "romanised_text": latin_of.get(r.prompt_id),
        } for r in scored[:top_n]],
    }


def load_language_qc(path: Path) -> dict[str, LanguageQCRecord]:
    with path.open(encoding="utf-8") as fh:
        recs = [LanguageQCRecord.model_validate_json(line) for line in fh if line.strip()]
    return {r.prompt_id: r for r in recs}


def run_qc_on_dir(settings: Settings, run_dir: Path, encoder: SentenceEncoder | None) -> dict[str, Path]:
    started = utc_now()
    inputs = {n: run_dir / n for n in ("variants.jsonl", "transformations.jsonl", "language_qc.jsonl")}
    variants = load_variants_jsonl(inputs["variants.jsonl"])
    transformations = load_transformations_jsonl(inputs["transformations.jsonl"])
    records = run_qc(settings, variants, transformations, load_language_qc(inputs["language_qc.jsonl"]), encoder)
    paths: dict[str, Path] = {}
    report = run_dir / "qc_report.jsonl"
    _atomic_write(report, "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n"
                                  for r in records))
    paths["report"] = report
    for lang in sorted({v.language for v in variants.values() if v.code_mix_level is not None}):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CODEMIX_REVIEW_COLUMNS, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(codemix_review_rows(settings, variants, transformations, records, lang))
        p = run_dir / f"review_codemix_{lang}.csv"
        _atomic_write(p, buf.getvalue(), encoding="utf-8-sig")
        paths[f"review_codemix_{lang}"] = p
    harmful = run_dir / "harmful_intent_check.json"
    _write_json(harmful, harmful_intent_check(variants, records, settings.generation.qc.human_review.harmful_intent_top_n))
    paths["harmful_intent"] = harmful
    summary = {
        "run_dir": run_dir.name,
        "started_at": iso(started),
        "finished_at": iso(utc_now()),
        "inputs": {n: sha256_file(p) for n, p in inputs.items()},
        "code_mix_coverage": code_mix_coverage(records, transformations),
        **summarize(records, settings, encoder),
        "config_hashes": settings.config_hashes,
        "outputs": {p.name: sha256_file(p) for p in paths.values()},
    }
    sp = run_dir / "qc_summary.json"
    _write_json(sp, summary)
    paths["summary"] = sp
    return paths
