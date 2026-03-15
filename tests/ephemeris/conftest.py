"""Pytest configuration for ephemeris tests."""

import os
from typing import Any

import pytest


@pytest.fixture
def sample_oem_file(tmp_path: Any) -> str:
    """Create a minimal OEM file for testing."""
    oem_path = tmp_path / "test_satellite.oem"
    oem_content = """CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00.000
ORIGINATOR = TEST

META_START
OBJECT_NAME = TEST_SAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = J2000
TIME_SYSTEM = UTC
START_TIME = 2024-01-01T00:00:00.000
STOP_TIME = 2024-01-01T03:00:00.000
META_STOP

DATA_START
2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T01:00:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
2024-01-01T02:00:00.000 6995.0 9000.0 0.0 -0.7791 7.4427 0.0
2024-01-01T03:00:00.000 6980.0 13500.0 0.0 -1.1677 7.3714 0.0
DATA_STOP
"""
    oem_path.write_text(oem_content)
    return str(oem_path)


@pytest.fixture
def spk_path() -> str:
    path = "test_data/de440s.bsp"
    if not os.path.exists(path):
        pytest.skip(f"SPICE kernel not found at {path}")
    return path
