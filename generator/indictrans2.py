"""IndicTrans2 translation adapter (local MT, en -> hi / mr / gu).

Model: `ai4bharat/indictrans2-en-indic-dist-200M` on Hugging Face (gated: the
account must accept the model terms and be logged in with `hf auth login`).
Pre/post-processing is AI4Bharat's IndicTransToolkit `IndicProcessor`: the
compiled PyPI package when it is installed, otherwise the vendored pure-Python
port in `generator/vendor/` (the PyPI sdist needs an MSVC compiler on
Windows). Which one ran is recorded on every transformation record.

    text ─► IndicProcessor.preprocess_batch ─► tokenizer ─► model.generate (beam)
         ─► batch_decode ─► IndicProcessor.postprocess_batch ─► ProviderOutput

Device: `device: auto` uses CUDA when torch sees a GPU, else CPU. fp16 only on
CUDA. On CUDA out-of-memory the batch is halved and retried; at batch size 1
the request fails as a ProviderError (recorded, never silently moved to CPU,
because that would change dtype and therefore the provider version).

KV cache: off by default. The model's remote code (pinned commit) indexes
`past_key_values` as legacy tuples, which transformers 4.57 no longer passes,
so `use_cache=True` fails on the first decoding step. The cache only changes
speed, not output, so it is recorded in metadata but not in the version.

Reproducibility: the provider version encodes dtype and decoding parameters,
and the model id carries the resolved Hugging Face commit sha, so both are part
of every transformation id. Beam search is deterministic; the engine's derived
seed is not used.

torch / transformers are imported lazily, so importing this module (and the
unit tests, which use a fake backend) does not need them.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from backend.config import ProviderConfig
from generator.transformation_engine import ProviderError, ProviderInfo, ProviderOutput, ProviderUnavailableError
from generator.translation import TranslationProvider

ADAPTER_VERSION = "1.0"

# ISO 639-1 (languages.yaml) -> FLORES-200 codes used by IndicTrans2.
FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr"}


@dataclass(frozen=True)
class IndicTrans2Options:
    revision: str | None = None
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 4
    num_beams: int = 5
    max_new_tokens: int = 256
    use_cache: bool = False
    cache_dir: str | None = None

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> "IndicTrans2Options":
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise ProviderUnavailableError(f"indictrans2: unknown option(s) {sorted(unknown)}")
        opts = cls(**options)
        if opts.device not in ("auto", "cuda", "cpu"):
            raise ProviderUnavailableError(f"indictrans2: device must be auto|cuda|cpu, not {opts.device!r}")
        for name in ("batch_size", "num_beams", "max_new_tokens"):
            if not isinstance(getattr(opts, name), int) or getattr(opts, name) < 1:
                raise ProviderUnavailableError(f"indictrans2: {name} must be a positive integer")
        return opts


class Backend(Protocol):
    """The model side: tokenised generation on preprocessed sentences."""

    revision: str          # resolved model commit sha
    device: str            # "cuda" | "cpu"
    dtype: str             # "float16" | "float32"
    versions: dict[str, str]

    def generate(self, batch: list[str]) -> list[tuple[str, int]]:
        """Decoded hypotheses with their generated token counts. May raise MemoryError on OOM."""


def load_indic_processor():
    """(IndicProcessor class, description) — compiled toolkit if installed, else the vendored port."""
    try:
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore[import-not-found]
        version = importlib.metadata.version("indictranstoolkit")
        return IndicProcessor, f"IndicTransToolkit {version} (compiled)"
    except ImportError:
        from generator.vendor.indictranstoolkit_processor import IndicProcessor
        return IndicProcessor, "IndicTransToolkit 1.1.1 IndicProcessor (vendored pure-Python port)"


class HFBackend:
    """Loads the IndicTrans2 checkpoint with transformers and runs beam search."""

    def __init__(self, model_name: str, opts: IndicTrans2Options):
        try:
            import torch
            import transformers
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ProviderUnavailableError(f"indictrans2 needs torch and transformers: {e}") from e

        cuda = torch.cuda.is_available()
        if opts.device == "cuda" and not cuda:
            raise ProviderUnavailableError("indictrans2: device=cuda but torch.cuda.is_available() is False")
        self.device = "cuda" if opts.device in ("auto", "cuda") and cuda else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" and opts.fp16 else torch.float32
        self.dtype = str(torch_dtype).removeprefix("torch.")
        load = dict(trust_remote_code=True, revision=opts.revision, cache_dir=opts.cache_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch_dtype, **load)
        except OSError as e:
            raise ProviderUnavailableError(
                f"cannot load {model_name}: {e}. The model is gated: accept its terms on huggingface.co "
                "and run `hf auth login`."
            ) from e
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or opts.revision or "unknown"
        self.opts = opts
        self._torch = torch
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        if self.device == "cuda":
            self.versions["cuda"] = str(torch.version.cuda)
            self.versions["gpu"] = torch.cuda.get_device_name(0)

    def generate(self, batch: list[str]) -> list[tuple[str, int]]:
        torch = self._torch
        try:
            inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                                    return_attention_mask=True).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, num_beams=self.opts.num_beams, max_new_tokens=self.opts.max_new_tokens,
                    do_sample=False, num_return_sequences=1, use_cache=self.opts.use_cache, min_length=0,
                )
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise MemoryError(str(e)) from e
        pad = self.tokenizer.pad_token_id
        lengths = [int((row != pad).sum()) - 1 for row in out]  # minus the decoder start token
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return list(zip(texts, lengths, strict=True))


class IndicTrans2Translator(TranslationProvider):
    def __init__(self, name: str, model_name: str, targets: list[str], opts: IndicTrans2Options,
                 backend: Backend):
        self.name = name
        self.model_name = model_name
        self.targets = set(targets)
        self.opts = opts
        self.backend = backend
        processor_cls, self.preprocessor = load_indic_processor()
        self.processor = processor_cls(inference=True)
        self._cache: dict[tuple[str, str], ProviderOutput] = {}
        self._info = ProviderInfo(
            name=name,
            version=f"{ADAPTER_VERSION}+{backend.dtype}+beam{opts.num_beams}+max{opts.max_new_tokens}",
            model=f"{model_name}@{backend.revision}",
            generation_method="mt",
        )

    @classmethod
    def from_config(cls, name: str, cfg: ProviderConfig, backend: Backend | None = None) -> "IndicTrans2Translator":
        if not cfg.model:
            raise ProviderUnavailableError(f"translation provider {name!r}: `model` is not set")
        opts = IndicTrans2Options.from_mapping(cfg.options)
        return cls(name, cfg.model, cfg.target_languages, opts, backend or HFBackend(cfg.model, opts))

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language == "en" and target_language in self.targets and target_language in FLORES

    def translate(self, text, *, source_language, target_language, target_script, seed):
        return self.translate_batch([text], source_language=source_language, target_language=target_language,
                                    target_script=target_script, seeds=[seed])[0]

    def translate_batch(self, texts, *, source_language, target_language, target_script, seeds):
        """Translate in chunks of `batch_size`; results are cached per (text, target) for this instance.

        The pilot script calls this once per language so the engine's per-item
        `translate` calls are served from the cache (same output, batched GPU use).
        """
        if not self.supports(source_language, target_language):
            raise ProviderError(f"indictrans2 does not support {source_language}->{target_language}")
        todo = list(dict.fromkeys(t for t in texts if (t, target_language) not in self._cache))
        size = self.opts.batch_size
        i = 0
        while i < len(todo):
            chunk = todo[i:i + size]
            try:
                outs = self._run(chunk, FLORES[source_language], FLORES[target_language])
            except MemoryError as e:
                if size == 1:
                    raise ProviderError(f"CUDA out of memory at batch size 1: {e}") from e
                size = max(1, size // 2)
                continue
            for text, out in zip(chunk, outs, strict=True):
                self._cache[(text, target_language)] = out
            i += len(chunk)
        return [self._cache[(t, target_language)] for t in texts]

    def _run(self, chunk: list[str], src: str, tgt: str) -> list[ProviderOutput]:
        batch = self.processor.preprocess_batch(chunk, src_lang=src, tgt_lang=tgt)
        try:
            generated = self.backend.generate(batch)
        except MemoryError:
            self.processor.postprocess_batch([""] * len(chunk), lang=tgt)  # drain placeholder queue
            raise
        except Exception as e:  # noqa: BLE001 - any backend failure is a provider failure
            self.processor.postprocess_batch([""] * len(chunk), lang=tgt)
            raise ProviderError(f"IndicTrans2 generation failed: {type(e).__name__}: {e}") from e
        texts = self.processor.postprocess_batch([t for t, _ in generated], lang=tgt)
        return [
            ProviderOutput(text, {
                "model_name": self.model_name,
                "model_revision": self.backend.revision,
                "src_lang": src,
                "tgt_lang": tgt,
                "device": self.backend.device,
                "dtype": self.backend.dtype,
                "num_beams": self.opts.num_beams,
                "max_new_tokens": self.opts.max_new_tokens,
                "batch_size": self.opts.batch_size,
                "use_cache": self.opts.use_cache,
                "generated_tokens": n,
                "hit_max_new_tokens": n >= self.opts.max_new_tokens,
                "preprocessor": self.preprocessor,
                **{f"{k}_version": v for k, v in sorted(self.backend.versions.items())},
            })
            for text, (_, n) in zip(texts, generated, strict=True)
        ]
