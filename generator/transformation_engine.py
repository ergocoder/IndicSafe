r"""Transformation engine: derive prompt variants from seeds, with full lineage.

    SeedRecord ──identity──► root variant (en/Latn) ──translation──► hi/Deva ──transliteration──► hi/Latn
                                                    ├─translation──► mr/Deva ── ...
                                                    └─paraphrase───► en/Latn

Every step goes through `TransformationEngine.apply` (or `root` for the first):

    resolve request (validate parameters, fill defaults, fix target condition)
      → deterministic ids from the canonical request
      → provider call (translation / transliteration / paraphrase adapter)
      → normalise text, measure script
      → child VariantRecord + validation hooks
      → TransformationRecord (always written, also on provider error)

Determinism: ids, derived random seeds and every recorded field except the
wall-clock `created_at` / `timestamp` and the run id are a pure function of
(parent, transformation type, resolved parameters, provider name/version/model,
config). Re-running the same request reproduces the same ids; if the output
text differs, the provider was non-deterministic and `output_content_hash`
shows it.

Records are frozen and the parent is never modified. A child that fails a
validation hook is kept (status VALIDATION_FAILED) for audit, but cannot be
used as a parent unless the caller explicitly allows it.

Not here (later phases): language identification, code-mixing, semantic and
label-consistency checks, batch generation jobs, LLM adapters.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from backend.config import (
    TRANSFORMATION_TYPES,
    ConfigError,
    LanguageConfig,
    ProviderConfig,
    Settings,
    resolve_inside,
)
from generator.provenance import build_run_manifest, iso, new_run_id, sha256_file, utc_now
from generator.schemas import (
    GenerationMethod,
    HookResult,
    SeedRecord,
    TransformationRecord,
    VariantRecord,
)
from generator.seed_manager import _atomic_write, _write_json
from generator.text_utils import content_hash, dominant_script, has_control_chars, normalize_text


class TransformationError(ValueError):
    """The request itself is invalid (bad parameters, disabled type, bad parent).

    Raised to the caller; nothing is recorded, because nothing was attempted.
    """


class ProviderError(RuntimeError):
    """A provider failed while producing output. Recorded as an ERROR transformation."""


class ProviderUnavailableError(ProviderError):
    """A configured provider cannot be built (disabled, missing, not implemented)."""


class LineageError(RuntimeError):
    """A variant's ancestry cannot be reconstructed from the records."""


# ------------------------------------------------------------ provider side


@dataclass(frozen=True)
class ProviderInfo:
    name: str                   # registry / config name, e.g. "indictrans2"
    version: str                # adapter or model release; part of every id
    model: str | None           # concrete model / scheme identifier
    generation_method: GenerationMethod


@dataclass(frozen=True)
class ProviderOutput:
    text: str
    # Must be JSON-serialisable and deterministic (no latencies, no wall clock):
    # it is stored on the transformation record.
    metadata: dict[str, Any] = field(default_factory=dict)


ProviderFactory = Callable[[str, ProviderConfig], Any]
_PROVIDER_FACTORIES: dict[tuple[str, str], ProviderFactory] = {}
PROVIDER_KINDS = ("translation", "transliteration", "paraphrase")


def register_provider(kind: str, name: str, factory: ProviderFactory) -> None:
    """Make a provider adapter buildable from config (`<kind>.providers.<name>`)."""
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"unknown provider kind {kind!r}")
    _PROVIDER_FACTORIES[(kind, name)] = factory


def build_provider(
    settings: Settings,
    kind: str,
    name: str | None = None,
    factories: Mapping[tuple[str, str], ProviderFactory] | None = None,
):
    """Build the configured provider `name` (default: the kind's default_provider)."""
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"unknown provider kind {kind!r}")
    adapter = getattr(settings.generation, kind)
    name = name or adapter.default_provider
    if name is None:
        raise ProviderUnavailableError(f"no default {kind} provider is configured")
    cfg = adapter.providers.get(name)
    if cfg is None:
        raise ProviderUnavailableError(f"{kind} provider {name!r} is not configured")
    if not cfg.enabled:
        raise ProviderUnavailableError(f"{kind} provider {name!r} is disabled in generation.yaml")
    factory = (factories if factories is not None else _PROVIDER_FACTORIES).get((kind, name))
    if factory is None:
        raise ProviderUnavailableError(
            f"{kind} provider {name!r} is configured but no adapter is implemented/registered for it"
        )
    return factory(name, cfg)


# -------------------------------------------------------- transformation side


@dataclass(frozen=True)
class TargetCondition:
    language: str
    script: str
    secondary_language: str | None = None
    is_transliterated: bool = False


@dataclass(frozen=True)
class ResolvedRequest:
    parameters: dict[str, Any]   # complete, canonical; recorded and hashed
    target: TargetCondition


class Transformation(ABC):
    """One kind of parent → child rewrite, bound to one provider instance."""

    transformation_type: ClassVar[str]

    @property
    @abstractmethod
    def provider_info(self) -> ProviderInfo: ...

    @abstractmethod
    def resolve(self, parent: VariantRecord, params: Mapping[str, Any], settings: Settings) -> ResolvedRequest:
        """Validate params against the parent and config; raise TransformationError if invalid."""

    @abstractmethod
    def run(self, parent: VariantRecord, request: ResolvedRequest, derived_seed: int) -> ProviderOutput:
        """Produce the child text. Provider failures must be raised as ProviderError."""


def check_params(params: Mapping[str, Any], allowed: Iterable[str], ttype: str) -> None:
    unknown = set(params) - set(allowed)
    if unknown:
        raise TransformationError(f"{ttype}: unknown parameter(s) {sorted(unknown)}")


def language_config(settings: Settings, code: str, *, enabled: bool = False) -> LanguageConfig:
    lang = settings.languages.languages.get(code)
    if lang is None:
        raise TransformationError(f"language {code!r} is not in languages.yaml")
    if enabled and not lang.enabled:
        raise TransformationError(f"language {code!r} is disabled in languages.yaml")
    return lang


# ---------------------------------------------------------- validation hooks


@dataclass(frozen=True)
class HookContext:
    settings: Settings
    transformation_type: str
    text: str                        # normalised child text
    raw_text: str                    # provider output before normalisation
    measured_script: str | None
    script_share: float
    target: TargetCondition
    parent_text: str                 # parent variant text (seed text for the root)
    parent_content_hash: str


HookFn = Callable[[HookContext], HookResult]
_HOOKS: dict[str, HookFn] = {}


def validation_hook(name: str) -> Callable[[HookFn], HookFn]:
    """Register a hook usable from transformation_engine.validation_hooks in config."""
    def deco(fn: HookFn) -> HookFn:
        _HOOKS[name] = fn
        return fn
    return deco


def registered_hooks() -> dict[str, HookFn]:
    return dict(_HOOKS)


@validation_hook("non_empty")
def _non_empty(ctx: HookContext) -> HookResult:
    if not ctx.text:
        return HookResult(hook="non_empty", status="FAIL", reason="empty_output")
    return HookResult(hook="non_empty", status="PASS")


@validation_hook("text_integrity")
def _text_integrity(ctx: HookContext) -> HookResult:
    sv = ctx.settings.generation.seed_validation
    problems = []
    if has_control_chars(ctx.raw_text):
        problems.append("control_chars")
    if sv.reject_replacement_char and "�" in ctx.text:
        problems.append("replacement_char")
    if any(m in ctx.text and m not in ctx.parent_text for m in sv.mojibake_markers):
        problems.append("mojibake")
    if problems:
        return HookResult(hook="text_integrity", status="FAIL", reason=problems[0],
                          details={"problems": problems})
    return HookResult(hook="text_integrity", status="PASS")


@validation_hook("expected_script")
def _expected_script(ctx: HookContext) -> HookResult:
    lang = ctx.settings.languages.languages[ctx.target.language]
    qc = ctx.settings.generation.qc.script
    native = ctx.target.script == lang.native_script
    min_share = qc.native_min_share if native else qc.romanized_min_share
    details = {"expected": ctx.target.script, "measured": ctx.measured_script,
               "share": ctx.script_share, "min_share": min_share}
    if ctx.measured_script != ctx.target.script:
        return HookResult(hook="expected_script", status="FAIL", reason="script_mismatch", details=details)
    if ctx.script_share < min_share:
        return HookResult(hook="expected_script", status="FAIL", reason="low_script_share", details=details)
    return HookResult(hook="expected_script", status="PASS", details=details)


@validation_hook("differs_from_parent")
def _differs_from_parent(ctx: HookContext) -> HookResult:
    if ctx.transformation_type != "identity" and content_hash(ctx.text) == ctx.parent_content_hash:
        return HookResult(hook="differs_from_parent", status="FAIL", reason="condition_not_realised")
    return HookResult(hook="differs_from_parent", status="PASS")


@validation_hook("length_ratio")
def _length_ratio(ctx: HookContext) -> HookResult:
    bounds = ctx.settings.generation.transformation_engine.length_ratio
    ratio = round(len(ctx.text) / max(len(ctx.parent_text), 1), 4)
    details = {"ratio": ratio, "min": bounds.min, "max": bounds.max}
    if not bounds.min <= ratio <= bounds.max:
        return HookResult(hook="length_ratio", status="WARN", reason="length_ratio_out_of_range",
                          details=details)
    return HookResult(hook="length_ratio", status="PASS", details=details)


# --------------------------------------------------------------------- engine


@dataclass(frozen=True)
class TransformationResult:
    transformation: TransformationRecord
    variant: VariantRecord | None

    @property
    def ok(self) -> bool:
        return self.transformation.status == "SUCCEEDED"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_IDENTITY_PROVIDER = "copy"


class TransformationEngine:
    """Applies transformations for one generation run and keeps its records."""

    def __init__(
        self,
        settings: Settings,
        *,
        run_id: str | None = None,
        clock: Callable[[], datetime] = utc_now,
        extra_hooks: Mapping[str, HookFn] | None = None,
    ):
        self.settings = settings
        self.cfg = settings.generation.transformation_engine
        self._clock = clock
        self.started_at = clock()
        self.run_id = run_id or new_run_id("TRANSFORM", self.started_at, settings)
        self._hooks = {**_HOOKS, **(extra_hooks or {})}
        for ttype, names in self.cfg.validation_hooks.items():
            missing = [n for n in names if n not in self._hooks]
            if missing:
                raise ConfigError(f"validation_hooks.{ttype}: unknown hook(s) {missing}")
        self._scripts = {k: v.ranges for k, v in settings.languages.scripts.items()}
        self.variants: dict[str, VariantRecord] = {}
        self.transformations: dict[str, TransformationRecord] = {}
        self._seeds: dict[str, tuple[int, str]] = {}   # seed_id -> (version, content_hash)

    # ------------------------------------------------------------ public

    def root(self, seed: SeedRecord) -> TransformationResult:
        """Identity transformation: the seed's root variant, parent of every chain."""
        self._check_enabled("identity")
        if seed.seed_status != "VALID":
            raise TransformationError(f"{seed.seed_id} is {seed.seed_status}; only VALID seeds are transformed")
        if seed.script is None:
            raise TransformationError(f"{seed.seed_id} has no measured script")
        target = TargetCondition(seed.language, seed.script, None, seed.is_transliterated)
        info = ProviderInfo(_IDENTITY_PROVIDER, self.cfg.engine_version, None, "copy")
        request = {
            "type": "identity",
            "seed_id": seed.seed_id,
            "seed_version": seed.seed_version,
            "seed_content_hash": seed.content_hash,
        }
        lineage_base = {
            "seed_id": seed.seed_id,
            "seed_version": seed.seed_version,
            "category": seed.category,
            "intended_label": seed.intended_label,
            "source_type": seed.source_type,
            "source_dataset": seed.source_dataset,
            "source_reference": seed.source_reference,
        }
        self._seeds[seed.seed_id] = (seed.seed_version, seed.content_hash)
        return self._produce(
            ttype="identity",
            parent=None,
            parent_text=seed.prompt,
            parent_hash=seed.content_hash,
            inherited=lineage_base,
            info=info,
            parameters={},
            target=target,
            request=request,
            run=lambda _seed: ProviderOutput(seed.prompt),
        )

    def apply(
        self,
        parent: VariantRecord,
        transformation: Transformation,
        params: Mapping[str, Any] | None = None,
        *,
        allow_failed_parent: bool = False,
    ) -> TransformationResult:
        """Derive one child of `parent`. The parent record is not modified."""
        ttype = transformation.transformation_type
        if ttype == "identity":
            raise TransformationError("use root(seed) for the identity transformation")
        self._check_enabled(ttype)
        if parent.validation_status == "FAIL" and not allow_failed_parent:
            raise TransformationError(
                f"{parent.prompt_id} failed validation ({parent.validation_failures}); "
                "pass allow_failed_parent=True to derive from it anyway"
            )
        resolved = transformation.resolve(parent, dict(params or {}), self.settings)
        info = transformation.provider_info
        request = {
            "type": ttype,
            "parent_prompt_id": parent.prompt_id,
            "parent_content_hash": parent.content_hash,
            "parameters": resolved.parameters,
            "provider": info.name,
            "provider_version": info.version,
            "model": info.model,
        }
        inherited = {
            k: getattr(parent, k)
            for k in ("seed_id", "seed_version", "category", "intended_label",
                      "source_type", "source_dataset", "source_reference")
        }
        return self._produce(
            ttype=ttype,
            parent=parent,
            parent_text=parent.prompt,
            parent_hash=parent.content_hash,
            inherited=inherited,
            info=info,
            parameters=resolved.parameters,
            target=resolved.target,
            request=request,
            run=lambda derived: transformation.run(parent, resolved, derived),
        )

    # ---------------------------------------------------------- internals

    def _check_enabled(self, ttype: str) -> None:
        if ttype not in TRANSFORMATION_TYPES:
            raise TransformationError(f"unknown transformation type {ttype!r}")
        if ttype not in self.settings.generation.transformations.enabled:
            raise TransformationError(f"transformation {ttype!r} is not enabled in generation.yaml")

    def _produce(self, *, ttype, parent, parent_text, parent_hash, inherited, info,
                 parameters, target, request, run) -> TransformationResult:
        canonical_json(parameters)  # fail early on non-JSON parameters
        fingerprint = _sha256(canonical_json(request))
        n = self.cfg.id_hash_chars
        tid = f"T-{fingerprint[:n]}"
        if tid in self.transformations:  # same request twice in one run: reuse, don't re-call
            rec = self.transformations[tid]
            return TransformationResult(rec, self.variants.get(rec.output_prompt_id or ""))
        prompt_id = f"P-{inherited['seed_id'].removeprefix('S-')}-{fingerprint[:n]}"
        derived_seed = int(_sha256(f"{self.settings.generation.random_seed}:{fingerprint}")[:8], 16)
        hooks = self.cfg.hooks_for(ttype)
        base = dict(
            transformation_id=tid,
            transformation_type=ttype,
            seed_id=inherited["seed_id"],
            parent_prompt_id=parent.prompt_id if parent else None,
            parameters=parameters,
            provider=info.name,
            provider_version=info.version,
            generator_model=info.model,
            generation_method=info.generation_method,
            request_fingerprint=fingerprint,
            derived_seed=derived_seed,
            validation_hooks=hooks,
            engine_version=self.cfg.engine_version,
            generation_run_id=self.run_id,
        )

        try:
            out = run(derived_seed)
            if not isinstance(out, ProviderOutput) or not isinstance(out.text, str):
                raise ProviderError(f"provider {info.name!r} returned {type(out).__name__}, not ProviderOutput")
            try:
                canonical_json(out.metadata)
            except (TypeError, ValueError) as e:
                raise ProviderError(f"provider {info.name!r} metadata is not JSON-serialisable: {e}") from e
        except ProviderError as e:
            rec = TransformationRecord(**base, output_prompt_id=None, status="ERROR",
                                       error=f"{type(e).__name__}: {e}", timestamp=self._now())
            self.transformations[tid] = rec
            return TransformationResult(rec, None)

        text = normalize_text(out.text)
        script, share = dominant_script(text, self._scripts)
        ctx = HookContext(self.settings, ttype, text, out.text, script, share, target,
                          parent_text, parent_hash)
        results = [self._hooks[h](ctx) for h in hooks]
        failures = [f"{r.hook}:{r.reason}" for r in results if r.status == "FAIL"]
        vstatus = "FAIL" if failures else ("WARN" if any(r.status == "WARN" for r in results) else "PASS")
        now = self._now()

        variant = VariantRecord(
            prompt_id=prompt_id,
            parent_prompt_id=parent.prompt_id if parent else None,
            transformation_id=tid,
            lineage=[*parent.lineage, parent.prompt_id] if parent else [],
            prompt=text,
            content_hash=content_hash(text),
            language=target.language,
            secondary_language=target.secondary_language,
            script=script,
            is_transliterated=target.is_transliterated,
            transformation_type=ttype,
            generation_method=info.generation_method,
            generator_model=info.model,
            generation_run_id=self.run_id,
            script_confidence=share,
            validation_status=vstatus,
            validation_failures=failures,
            taxonomy_version=self.settings.taxonomy.taxonomy_version,
            generator_version=self.settings.generation.generator_version,
            created_at=now,
            **inherited,
        )
        rec = TransformationRecord(
            **base,
            output_prompt_id=prompt_id,
            provider_metadata=out.metadata,
            raw_output=out.text,
            output_content_hash=variant.content_hash,
            validation_results=results,
            status="VALIDATION_FAILED" if failures else "SUCCEEDED",
            timestamp=now,
        )
        self.variants[prompt_id] = variant
        self.transformations[tid] = rec
        return TransformationResult(rec, variant)

    def _now(self) -> str:
        return iso(self._clock())

    # ------------------------------------------------------------- export

    def export(self, out_dir: Path) -> dict[str, Path]:
        """Write variants.jsonl, transformations.jsonl and manifest.json to out_dir/<run_id>/."""
        problems = verify_lineage(self.variants, self.transformations)
        if problems:
            raise LineageError("refusing to export broken lineage: " + "; ".join(problems[:5]))
        out = resolve_inside(self.settings.project_root, out_dir)
        raw = self.settings.raw_dir
        if out == raw or raw in out.parents:
            raise ConfigError("refusing to write into data/raw/: raw data is read-only")
        run_dir = out / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        variants_path = run_dir / "variants.jsonl"
        trans_path = run_dir / "transformations.jsonl"
        _atomic_write(variants_path, _jsonl(self.variants.values()))
        _atomic_write(trans_path, _jsonl(self.transformations.values()))

        gen = self.settings.generation
        manifest = build_run_manifest(
            run_id=self.run_id,
            run_type="transformation",
            started_at=self.started_at,
            settings=self.settings,
            source_checksums={},
            extra={
                "engine_version": self.cfg.engine_version,
                "input_seeds": {sid: {"seed_version": v, "content_hash": h}
                                for sid, (v, h) in sorted(self._seeds.items())},
                "transformation_config": {
                    "enabled": gen.transformations.enabled,
                    "validation_hooks": self.cfg.validation_hooks,
                    "length_ratio": self.cfg.length_ratio.model_dump(),
                    "id_hash_chars": self.cfg.id_hash_chars,
                },
                "providers_used": sorted(
                    {(t.transformation_type, t.provider, t.provider_version, t.generator_model or "")
                     for t in self.transformations.values()}
                ),
                "counts": {
                    "transformations_by_type_status": dict(sorted(Counter(
                        f"{t.transformation_type}:{t.status}" for t in self.transformations.values()
                    ).items())),
                    "variants_by_validation_status": dict(sorted(Counter(
                        v.validation_status for v in self.variants.values()
                    ).items())),
                },
                "label_note": "intended_label is inherited from the seed and provisional; "
                              "label_consistency_status is UNCHECKED until the QC phase.",
                "outputs": {p.name: sha256_file(p) for p in (variants_path, trans_path)},
            },
        )
        manifest_path = run_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        return {"variants": variants_path, "transformations": trans_path, "manifest": manifest_path}


# -------------------------------------------------------------------- lineage


def verify_lineage(
    variants: Mapping[str, VariantRecord],
    transformations: Mapping[str, TransformationRecord],
) -> list[str]:
    """Every variant must chain back to an identity root of the same seed."""
    problems = []
    for pid, v in variants.items():
        t = transformations.get(v.transformation_id)
        if t is None or t.output_prompt_id != pid:
            problems.append(f"{pid}: transformation {v.transformation_id} missing or points elsewhere")
            continue
        if t.parent_prompt_id != v.parent_prompt_id or t.seed_id != v.seed_id:
            problems.append(f"{pid}: transformation record disagrees on parent or seed")
        if v.parent_prompt_id is None:
            continue
        parent = variants.get(v.parent_prompt_id)
        if parent is None:
            problems.append(f"{pid}: parent {v.parent_prompt_id} missing")
        elif parent.seed_id != v.seed_id or v.lineage != [*parent.lineage, parent.prompt_id]:
            problems.append(f"{pid}: seed_id or lineage inconsistent with parent {parent.prompt_id}")
    return problems


def trace_lineage(
    prompt_id: str,
    variants: Mapping[str, VariantRecord],
    transformations: Mapping[str, TransformationRecord],
) -> list[dict[str, Any]]:
    """The chain of steps root → prompt_id, each with its transformation details."""
    steps = []
    current: str | None = prompt_id
    while current is not None:
        v = variants.get(current)
        if v is None:
            raise LineageError(f"variant {current} not found")
        t = transformations.get(v.transformation_id)
        if t is None:
            raise LineageError(f"transformation {v.transformation_id} of {current} not found")
        steps.append({
            "prompt_id": v.prompt_id,
            "transformation_id": t.transformation_id,
            "transformation_type": t.transformation_type,
            "parameters": t.parameters,
            "provider": t.provider,
            "language": v.language,
            "script": v.script,
        })
        current = v.parent_prompt_id
        if len(steps) > len(variants):
            raise LineageError(f"cycle in lineage of {prompt_id}")
    steps.reverse()
    return steps


# ----------------------------------------------------------------------- I/O


def _jsonl(records: Iterable) -> str:
    return "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in records)


def load_variants_jsonl(path: Path) -> dict[str, VariantRecord]:
    with path.open(encoding="utf-8") as fh:
        recs = [VariantRecord.model_validate_json(line) for line in fh if line.strip()]
    return {r.prompt_id: r for r in recs}


def load_transformations_jsonl(path: Path) -> dict[str, TransformationRecord]:
    with path.open(encoding="utf-8") as fh:
        recs = [TransformationRecord.model_validate_json(line) for line in fh if line.strip()]
    return {r.transformation_id: r for r in recs}
