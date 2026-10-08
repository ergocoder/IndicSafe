"""Real IndicTrans2 on the GPU. Skipped unless torch sees CUDA and the model is already cached.

Never downloads: run scripts/run_pilot_translation.py once to fetch the model.
Run only this test with:  python -m pytest -m integration
"""

from __future__ import annotations

import pytest

import generator.providers  # noqa: F401
from backend.config import load_settings
from generator.language_qc import build_language_identifier, check_variant
from generator.transformation_engine import TransformationEngine
from generator.translation import TranslationTransformation, build_translation_provider
from generator.transliteration import TransliterationTransformation, build_transliterator
from tests.test_transformation_engine import make_seed

pytestmark = pytest.mark.integration


def _skip_reason(model: str, revision: str | None, cache_dir: str | None) -> str | None:
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU visible to torch"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return "huggingface_hub is not installed"
    if not isinstance(try_to_load_from_cache(model, "config.json", cache_dir=cache_dir, revision=revision), str):
        return f"{model} is not in the local Hugging Face cache"
    return None


def test_indictrans2_gpu_translates_and_romanises_hi_mr_gu():
    settings = load_settings()
    cfg = settings.generation.translation.providers["indictrans2"]
    reason = _skip_reason(cfg.model, cfg.options.get("revision"), cfg.options.get("cache_dir"))
    if reason:
        pytest.skip(reason)

    mt = build_translation_provider(settings)
    assert mt.backend.device == "cuda" and mt.backend.dtype == "float16"
    tl, lid = build_transliterator(settings), build_language_identifier(settings)
    engine = TransformationEngine(settings, run_id="TRANSFORM_INTEGRATION")
    root = engine.root(make_seed()).variant      # "Which river flows through the city of Varanasi?"
    for lang, script in (("hi", "Deva"), ("mr", "Deva"), ("gu", "Gujr")):
        res = engine.apply(root, TranslationTransformation(mt), {"target_language": lang})
        assert res.ok, res.transformation
        v, t = res.variant, res.transformation
        assert v.script == script and v.script_confidence >= 0.85
        assert t.provider_metadata["model_revision"] and t.provider_metadata["device"] == "cuda"
        assert check_variant(settings, v, lid).lid_language == lang
        latn = engine.apply(v, TransliterationTransformation(tl))
        assert latn.ok and latn.variant.script == "Latn"
