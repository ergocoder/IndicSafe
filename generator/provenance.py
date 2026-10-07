"""Provenance: raw-file integrity and run manifests.

Every import writes a manifest recording what went in (raw-file checksums,
config-file checksums), how (random seed, versions, environment, git commit if
any) and when. With the manifest plus the unchanged raw files, the output can
be rebuilt and compared.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from backend.config import Settings, SourceConfig, resolve_inside

TRACKED_PACKAGES = ("pydantic", "PyYAML", "pytest")


class SourceIntegrityError(RuntimeError):
    """A registered raw file is missing or its checksum changed."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def source_path(settings: Settings, src: SourceConfig) -> Path:
    return resolve_inside(settings.raw_dir, src.archive)


def verify_source(settings: Settings, source_id: str) -> str:
    """Check the raw archive exists and matches its registered checksum."""
    src = settings.sources.sources[source_id]
    path = source_path(settings, src)
    if not path.is_file():
        raise SourceIntegrityError(f"{source_id}: raw file not found: {path}")
    digest = sha256_file(path)
    if digest != src.sha256:
        raise SourceIntegrityError(
            f"{source_id}: checksum mismatch for {src.archive} "
            f"(registered {src.sha256[:12]}…, found {digest[:12]}…). "
            "The raw file changed; restore the original or re-register it deliberately."
        )
    return digest


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def new_run_id(kind: str, ts: datetime, settings: Settings) -> str:
    """e.g. IMPORT_20260930T171500Z_3f2a9c1b — timestamp + config fingerprint."""
    fingerprint = hashlib.sha256(
        "".join(f"{k}={v}" for k, v in sorted(settings.config_hashes.items())).encode()
    ).hexdigest()[:8]
    return f"{kind.upper()}_{ts.strftime('%Y%m%dT%H%M%SZ')}_{fingerprint}"


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(root: Path) -> dict:
    """HEAD commit plus whether tracked code/config differs from it.

    A run from uncommitted code is recorded as dirty: HEAD alone would claim a
    code version that did not actually produce the output.
    """
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    untracked_code = _git(root, "ls-files", "--others", "--exclude-standard",
                          "--", "backend", "generator", "scripts", "configs")
    return {"commit": commit, "dirty": bool(status) or bool(untracked_code)}


def environment_info() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _source_entry(settings: Settings, sid: str, digest: str) -> dict:
    src = settings.sources.sources.get(sid)
    if src is None:  # e.g. a manual seed CSV, keyed "manual:<file name>"
        return {"sha256": digest}
    return {"archive": src.archive, "member": src.member, "sha256": digest}


def build_run_manifest(
    *,
    run_id: str,
    run_type: str,
    started_at: datetime,
    settings: Settings,
    source_checksums: dict[str, str],
    extra: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "run_type": run_type,
        "started_at": iso(started_at),
        "finished_at": iso(utc_now()),
        "generator_version": settings.generation.generator_version,
        "taxonomy_version": settings.taxonomy.taxonomy_version,
        "taxonomy_status": settings.taxonomy.status,
        "languages_version": settings.languages.languages_version,
        "sources_version": settings.sources.sources_version,
        "random_seed": settings.generation.random_seed,
        "config_sha256": dict(settings.config_hashes),
        "source_files": {
            sid: _source_entry(settings, sid, digest) for sid, digest in source_checksums.items()
        },
        "git": git_state(settings.project_root),
        "environment": environment_info(),
        **(extra or {}),
    }
