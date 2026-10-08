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
                     REVIEW, otherwise FAIL semantic_drift. Romanised variants
                     inherit their native parent's score.

qc_status = FAIL if any check FAILs, else REVIEW if any REVIEWs, else PASS.
The QC never edits a variant and never relabels a level.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
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

QC_VERSION = "1.0"
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
    semantic_similarity: float | None = None
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
    sim = s["similarity_to_seed"]
    details = {**s, "pass": cfg.pass_, "review": cfg.review}
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


def summarize(records: list[QCRecord], settings: Settings, encoder: SentenceEncoder | None) -> dict:
    by_kind: dict[str, Counter] = defaultdict(Counter)
    ratios: dict[str, list[float]] = defaultdict(list)
    cmis: dict[str, list[float]] = defaultdict(list)
    sims: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_kind[_kind(r)][r.qc_status] += 1
        if r.code_mix_ratio is not None:
            ratios[_kind(r)].append(r.code_mix_ratio)
            cmis[_kind(r)].append(r.cmi)
        if r.semantic_similarity is not None and r.checks["semantic"].details.get("method") == "encoded":
            sims[_kind(r)].append(r.semantic_similarity)
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
        "intended_label_counts": dict(sorted(Counter(r.intended_label for r in records).items())),
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


def load_language_qc(path: Path) -> dict[str, LanguageQCRecord]:
    with path.open(encoding="utf-8") as fh:
        recs = [LanguageQCRecord.model_validate_json(line) for line in fh if line.strip()]
    return {r.prompt_id: r for r in recs}


def run_qc_on_dir(settings: Settings, run_dir: Path, encoder: SentenceEncoder | None) -> dict[str, Path]:
    started = utc_now()
    inputs = {n: run_dir / n for n in ("variants.jsonl", "transformations.jsonl", "language_qc.jsonl")}
    variants = load_variants_jsonl(inputs["variants.jsonl"])
    records = run_qc(settings, variants, load_transformations_jsonl(inputs["transformations.jsonl"]),
                     load_language_qc(inputs["language_qc.jsonl"]), encoder)
    report = run_dir / "qc_report.jsonl"
    _atomic_write(report, "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n"
                                  for r in records))
    summary = {
        "run_dir": run_dir.name,
        "started_at": iso(started),
        "finished_at": iso(utc_now()),
        "inputs": {n: sha256_file(p) for n, p in inputs.items()},
        **summarize(records, settings, encoder),
        "config_hashes": settings.config_hashes,
        "outputs": {report.name: sha256_file(report)},
    }
    sp = run_dir / "qc_summary.json"
    _write_json(sp, summary)
    return {"report": report, "summary": sp}
