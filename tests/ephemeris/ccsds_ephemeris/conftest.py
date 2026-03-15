"""Fixtures for ccsds_ephemeris tests."""

from typing import Any

import pytest


def create_sample_oem(path: str) -> None:
    """Create a simple OEM file for testing"""
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
STOP_TIME = 2024-01-01T01:00:00.000
META_STOP

DATA_START
2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T00:10:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
2024-01-01T00:20:00.000 6995.0 9000.0 0.0 -0.7791 7.4427 0.0
2024-01-01T00:30:00.000 6980.0 13500.0 0.0 -1.1677 7.3714 0.0
2024-01-01T00:40:00.000 6955.0 18000.0 0.0 -1.5550 7.2716 0.0
2024-01-01T00:50:00.000 6920.0 22500.0 0.0 -1.9407 7.1434 0.0
2024-01-01T01:00:00.000 6875.0 27000.0 0.0 -2.3243 6.9870 0.0
DATA_STOP
"""
    with open(path, "w") as f:
        f.write(oem_content)


@pytest.fixture
def sample_oem_path(tmp_path: Any) -> str:
    """Create a temporary OEM file for testing"""
    oem_path = tmp_path / "test_satellite.oem"
    create_sample_oem(str(oem_path))
    return str(oem_path)
