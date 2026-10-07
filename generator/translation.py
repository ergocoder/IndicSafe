"""Translation adapter interface and translation transformation.

The pipeline only ever talks to `TranslationProvider`. Which concrete provider
runs is configuration (`generation.yaml` → `translation`): the default is a
local open-source MT model (IndicTrans2 is the first candidate, fixed only
after the pilot evaluation); an LLM adapter is optional and off. No provider
is hard-coded here. A concrete adapter registers itself with
`register_provider("translation", <name>, factory)` and is then built from
config by `build_translation_provider`.

Translation produces native-script, monolingual text. Romanisation is a
separate transliteration step, so the lineage records it explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from backend.config import Settings
from generator.schemas import VariantRecord
from generator.transformation_engine import (
    ProviderInfo,
    ProviderOutput,
    ResolvedRequest,
    TargetCondition,
    Transformation,
    TransformationError,
    build_provider,
    check_params,
    language_config,
)


class TranslationProvider(ABC):
    """Adapter contract for any MT system or LLM used for translation."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @abstractmethod
    def supports(self, source_language: str, target_language: str) -> bool: ...

    @abstractmethod
    def translate(
        self, text: str, *, source_language: str, target_language: str, target_script: str, seed: int
    ) -> ProviderOutput:
        """Translate one text. Raise ProviderError on failure."""

    def translate_batch(
        self, texts: list[str], *, source_language: str, target_language: str, target_script: str,
        seeds: list[int],
    ) -> list[ProviderOutput]:
        """Batch form for adapters that benefit from batching (local MT). Default: one by one."""
        return [
            self.translate(t, source_language=source_language, target_language=target_language,
                           target_script=target_script, seed=s)
            for t, s in zip(texts, seeds, strict=True)
        ]


def build_translation_provider(settings: Settings, name: str | None = None) -> TranslationProvider:
    return build_provider(settings, "translation", name)


class TranslationTransformation(Transformation):
    transformation_type = "translation"
    PARAMS = ("target_language", "target_script")

    def __init__(self, provider: TranslationProvider):
        self.provider = provider

    @property
    def provider_info(self) -> ProviderInfo:
        return self.provider.info

    def resolve(self, parent: VariantRecord, params: Mapping[str, Any], settings: Settings) -> ResolvedRequest:
        check_params(params, self.PARAMS, self.transformation_type)
        tgt = params.get("target_language")
        if not tgt:
            raise TransformationError("translation: target_language is required")
        lang = language_config(settings, tgt, enabled=True)
        if tgt == parent.language:
            raise TransformationError(f"translation: parent is already {tgt!r}")
        if parent.is_transliterated or parent.secondary_language is not None:
            raise TransformationError(
                "translation: parent must be monolingual, native-script text "
                f"({parent.prompt_id} is transliterated or code-mixed)"
            )
        script = params.get("target_script", lang.native_script)
        if script != lang.native_script:
            raise TransformationError(
                f"translation: target_script must be {tgt!r}'s native script {lang.native_script!r}; "
                "romanise with a separate transliteration step"
            )
        if not self.provider.supports(parent.language, tgt):
            raise TransformationError(
                f"translation: provider {self.provider.info.name!r} does not support {parent.language}->{tgt}"
            )
        return ResolvedRequest(
            parameters={"source_language": parent.language, "target_language": tgt, "target_script": script},
            target=TargetCondition(language=tgt, script=script),
        )

    def run(self, parent: VariantRecord, request: ResolvedRequest, derived_seed: int) -> ProviderOutput:
        p = request.parameters
        return self.provider.translate(
            parent.prompt, source_language=p["source_language"], target_language=p["target_language"],
            target_script=p["target_script"], seed=derived_seed,
        )
