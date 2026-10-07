"""Paraphrase adapter interface and transformation.

Same language, same script, different wording. Optional for the MVP and
limited to the languages in `generation.yaml` → `paraphrase.languages`
(English only for now). No provider is enabled by default; an LLM adapter
plugs in through `register_provider("paraphrase", <name>, factory)`.

`variant_index` lets one parent have several paraphrases with distinct,
reproducible ids.
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
)


class ParaphraseProvider(ABC):
    """Adapter contract for a paraphrasing model."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo: ...

    @abstractmethod
    def paraphrase(
        self, text: str, *, language: str, script: str, variant_index: int, seed: int
    ) -> ProviderOutput:
        """Reword one text, keeping its meaning and safety intent. Raise ProviderError on failure."""


def build_paraphrase_provider(settings: Settings, name: str | None = None) -> ParaphraseProvider:
    return build_provider(settings, "paraphrase", name)


class ParaphraseTransformation(Transformation):
    transformation_type = "paraphrase"
    PARAMS = ("variant_index",)

    def __init__(self, provider: ParaphraseProvider):
        self.provider = provider

    @property
    def provider_info(self) -> ProviderInfo:
        return self.provider.info

    def resolve(self, parent: VariantRecord, params: Mapping[str, Any], settings: Settings) -> ResolvedRequest:
        check_params(params, self.PARAMS, self.transformation_type)
        allowed = settings.generation.paraphrase.languages
        if parent.language not in allowed:
            raise TransformationError(
                f"paraphrase: language {parent.language!r} not in paraphrase.languages {allowed}"
            )
        if parent.script is None:
            raise TransformationError(f"paraphrase: parent {parent.prompt_id} has no known script")
        index = params.get("variant_index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TransformationError("paraphrase: variant_index must be a non-negative integer")
        return ResolvedRequest(
            parameters={"language": parent.language, "script": parent.script, "variant_index": index},
            target=TargetCondition(
                language=parent.language,
                script=parent.script,
                secondary_language=parent.secondary_language,
                is_transliterated=parent.is_transliterated,
            ),
        )

    def run(self, parent: VariantRecord, request: ResolvedRequest, derived_seed: int) -> ProviderOutput:
        p = request.parameters
        return self.provider.paraphrase(
            parent.prompt, language=p["language"], script=p["script"],
            variant_index=p["variant_index"], seed=derived_seed,
        )
