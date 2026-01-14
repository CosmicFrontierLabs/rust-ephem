"""Pytest configuration for ephemeris tests."""

import os
from datetime import datetime, timezone

import pytest

from rust_ephem import GroundEphemeris, TLEEphemeris

# Test data
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


@pytest.fixture
def sample_oem_file(tmp_path):
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
def spk_path():
    path = "test_data/de440s.bsp"
    if not os.path.exists(path):
        pytest.skip(f"SPICE kernel not found at {path}")
    return path


@pytest.fixture
def ground_ephemeris():
    return GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def tle_ephemeris():
    return TLEEphemeris(
        tle1=VALID_TLE1,
        tle2=VALID_TLE2,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )
