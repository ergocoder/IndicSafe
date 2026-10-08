"""Transliteration / script-conversion adapter interface and transformation.

Rewrites a variant into another script without changing its language:
typically native → romanised (hi Deva → hi Latn), but any pair of configured
scripts is allowed (script conversion). The method — a deterministic
rule-based scheme or an informal LLM romanisation — is chosen in the
language/script phase; this module only fixes the contract, so swapping the
method changes the provider, not the pipeline.

`is_transliterated` is True whenever the target script is not the language's
native script (languages.yaml).
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


class Transliterator(ABC):
    """Adapter contract for a romanisation / script-conversion method."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @abstractmethod
    def supports(self, language: str, source_script: str, target_script: str) -> bool: ...

    @abstractmethod
    def transliterate(
        self, text: str, *, language: str, source_script: str, target_script: str, seed: int
    ) -> ProviderOutput:
        """Convert one text. Raise ProviderError on failure."""


def build_transliterator(settings: Settings, name: str | None = None) -> Transliterator:
    return build_provider(settings, "transliteration", name)


class TransliterationTransformation(Transformation):
    transformation_type = "transliteration"
    PARAMS = ("target_script",)

    def __init__(self, provider: Transliterator):
        self.provider = provider

    @property
    def provider_info(self) -> ProviderInfo:
        return self.provider.info

    def resolve(self, parent: VariantRecord, params: Mapping[str, Any], settings: Settings) -> ResolvedRequest:
        check_params(params, self.PARAMS, self.transformation_type)
        lang = language_config(settings, parent.language)
        target = params.get("target_script", lang.romanized_script)
        if target is None:
            raise TransformationError(
                f"transliteration: {parent.language!r} has no romanized_script; pass target_script"
            )
        if target not in settings.languages.scripts:
            raise TransformationError(f"transliteration: unknown script {target!r}")
        if parent.script is None or parent.script not in settings.languages.scripts:
            raise TransformationError(f"transliteration: parent {parent.prompt_id} has no known script")
        if target == parent.script:
            raise TransformationError(f"transliteration: parent is already in {target!r}")
        if not self.provider.supports(parent.language, parent.script, target):
            raise TransformationError(
                f"transliteration: provider {self.provider.info.name!r} does not support "
                f"{parent.language} {parent.script}->{target}"
            )
        return ResolvedRequest(
            parameters={"language": parent.language, "source_script": parent.script, "target_script": target},
            target=TargetCondition(
                language=parent.language,
                script=target,
                secondary_language=parent.secondary_language,
                is_transliterated=target != lang.native_script,
                code_mix_level=parent.code_mix_level,     # romanised code-mix keeps the level ...
                mixing_method=parent.mixing_method,       # ... and is band-checked again
            ),
        )

    def run(self, parent: VariantRecord, request: ResolvedRequest, derived_seed: int) -> ProviderOutput:
        p = request.parameters
        return self.provider.transliterate(
            parent.prompt, language=p["language"], source_script=p["source_script"],
            target_script=p["target_script"], seed=derived_seed,
        )
