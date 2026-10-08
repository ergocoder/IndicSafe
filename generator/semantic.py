"""Sentence encoder for the semantic-preservation check (Phase 5).

Default: LaBSE (`setu4993/LaBSE`, the transformers port of Google's
Language-agnostic BERT Sentence Embedding). It was trained to score
translation pairs across 109 languages, including Hindi, Marathi and
Gujarati. It is ungated (Apache-2.0) and has 471M parameters: about 1 GB in
fp16, which fits a 4 GB GTX 1650. The embedding is the pooler output,
L2-normalised, as on the model card. Model, revision, device and cache_dir
come from generation.yaml → qc.semantic; the cache resolves to HF_HUB_CACHE
when cache_dir is null.

torch / transformers are imported lazily; tests use a fake encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.config import SemanticQCConfig, Settings


class EncoderUnavailableError(RuntimeError):
    pass


class SentenceEncoder(ABC):
    name: str
    version: str

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Unit-length embeddings, one per text."""


def cosine(a: list[float], b: list[float]) -> float:
    return round(sum(x * y for x, y in zip(a, b, strict=True)), 4)


class LaBSEEncoder(SentenceEncoder):
    def __init__(self, cfg: SemanticQCConfig):
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EncoderUnavailableError(f"semantic QC needs torch and transformers: {e}") from e
        cuda = torch.cuda.is_available()
        if cfg.device == "cuda" and not cuda:
            raise EncoderUnavailableError("qc.semantic.device=cuda but CUDA is not available")
        self.device = "cuda" if cfg.device in ("auto", "cuda") and cuda else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        load = dict(revision=cfg.revision, cache_dir=cfg.cache_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, **load)
            self.model = AutoModel.from_pretrained(cfg.model, dtype=dtype, **load).to(self.device).eval()
        except OSError as e:
            raise EncoderUnavailableError(f"cannot load {cfg.model}: {e}") from e
        self._torch = torch
        self.batch_size = cfg.batch_size
        self.revision = getattr(self.model.config, "_commit_hash", None) or cfg.revision or "unknown"
        self.name = cfg.model
        self.version = f"{self.revision}+{str(dtype).removeprefix('torch.')}+{self.device}"
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def encode(self, texts):
        torch = self._torch
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                emb = self.model(**batch).pooler_output.float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.extend(emb.cpu().tolist())
        return out


def build_encoder(settings: Settings) -> SentenceEncoder:
    return LaBSEEncoder(settings.generation.qc.semantic)
