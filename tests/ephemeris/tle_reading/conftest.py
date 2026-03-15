"""Fixtures for test_tle_reading tests."""

import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Generator

import pytest

import rust_ephem

# Test TLE data
TLE_2LINE = """1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995
2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"""

TLE_3LINE = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""

TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test time range
BEGIN = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 60


@pytest.fixture
def legacy_ephem() -> rust_ephem.TLEEphemeris:
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)


@pytest.fixture
def tle_2line_file() -> Generator[str, Any, None]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
        f.write(TLE_2LINE)
        f.flush()
        filepath = f.name
    yield filepath
    os.unlink(filepath)


@pytest.fixture
def tle_3line_file() -> Generator[str, Any, None]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
        f.write(TLE_3LINE)
        f.flush()
        filepath = f.name
    yield filepath
    os.unlink(filepath)
