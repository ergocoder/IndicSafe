"""Read-only readers for the registered raw sources.

Records are streamed straight out of the zip archives in data/raw/; nothing is
extracted or written there. Malformed rows are reported (with their line
number), never silently skipped.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import SourceConfig


@dataclass(frozen=True)
class RawRecord:
    line: int                   # 1-based line (jsonl) or array position (json_array)
    data: dict[str, Any]


@dataclass(frozen=True)
class ParseFailure:
    line: int
    error: str


class SourceFormatError(RuntimeError):
    """The archive does not contain what the registry says it contains."""


def _clean(value: Any) -> Any:
    # dataset_10k uses bare NaN tokens; NaN is not valid JSON on export.
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _open_member(archive_path: Path, src: SourceConfig) -> bytes:
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if src.member not in names:
            raise SourceFormatError(
                f"{archive_path.name}: expected member {src.member!r}, found {names}"
            )
        return zf.read(src.member)


def iter_records(archive_path: Path, src: SourceConfig) -> Iterator[RawRecord | ParseFailure]:
    raw = _open_member(archive_path, src)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise SourceFormatError(f"{src.member}: not valid UTF-8 ({e})") from e

    if src.format == "jsonl":
        for n, line in enumerate(io.StringIO(text), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                yield ParseFailure(n, f"invalid JSON: {e.msg}")
                continue
            if not isinstance(obj, dict):
                yield ParseFailure(n, f"expected a JSON object, got {type(obj).__name__}")
                continue
            yield RawRecord(n, {k: _clean(v) for k, v in obj.items()})

    elif src.format == "json_array":
        try:
            arr = json.loads(text)
        except json.JSONDecodeError as e:
            raise SourceFormatError(f"{src.member}: invalid JSON array ({e.msg})") from e
        if not isinstance(arr, list):
            raise SourceFormatError(f"{src.member}: expected a JSON array at top level")
        for n, obj in enumerate(arr, start=1):
            if not isinstance(obj, dict):
                yield ParseFailure(n, f"expected a JSON object, got {type(obj).__name__}")
                continue
            yield RawRecord(n, {k: _clean(v) for k, v in obj.items()})

    else:
        raise SourceFormatError(f"format {src.format!r} is not a seed format")
