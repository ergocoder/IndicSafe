"""Checks against the real files in data/raw/ (skipped when they are absent,
e.g. on a fresh clone without the data)."""

import pytest

from backend.config import load_settings
from generator.provenance import verify_source

SETTINGS = load_settings()
PRESENT = [sid for sid, s in SETTINGS.sources.sources.items() if (SETTINGS.raw_dir / s.archive).is_file()]


@pytest.mark.skipif(not PRESENT, reason="raw data not present")
@pytest.mark.parametrize("sid", PRESENT)
def test_registered_checksum_matches(sid):
    assert verify_source(SETTINGS, sid) == SETTINGS.sources.sources[sid].sha256
