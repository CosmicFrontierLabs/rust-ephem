"""Fixtures for test_ephemeris_parameters tests."""

from datetime import datetime, timezone

import pytest

from rust_ephem import GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris

VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120


@pytest.fixture
def tle_ephem_default() -> TLEEphemeris:
    """TLEEphemeris with default parameters."""
    return TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)


@pytest.fixture
def tle_ephem_polar() -> TLEEphemeris:
    """TLEEphemeris with polar_motion=True."""
    return TLEEphemeris(
        VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE, polar_motion=True
    )


@pytest.fixture
def spice_ephem_default(spk_path: str) -> SPICEEphemeris:
    """SPICEEphemeris with default parameters."""
    return SPICEEphemeris(
        spk_path=spk_path,
        naif_id=301,
        center_id=399,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def spice_ephem_polar(spk_path: str) -> SPICEEphemeris:
    """SPICEEphemeris with polar_motion=True."""
    return SPICEEphemeris(
        spk_path=spk_path,
        naif_id=301,
        center_id=399,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
        polar_motion=True,
    )


@pytest.fixture
def oem_ephem_default(sample_oem_file: str) -> OEMEphemeris:
    """OEMEphemeris with default parameters."""
    return OEMEphemeris(
        sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
    )


@pytest.fixture
def oem_ephem_polar(sample_oem_file: str) -> OEMEphemeris:
    """OEMEphemeris with polar_motion=True."""
    return OEMEphemeris(
        sample_oem_file,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
        polar_motion=True,
    )


@pytest.fixture
def ground_ephem_default() -> GroundEphemeris:
    """GroundEphemeris with default parameters."""
    return GroundEphemeris(
        latitude=35.5,
        longitude=-120.7,
        height=250.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def ground_ephem_polar() -> GroundEphemeris:
    """GroundEphemeris with polar_motion=True."""
    return GroundEphemeris(
        latitude=35.5,
        longitude=-120.7,
        height=250.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
        polar_motion=True,
    )


@pytest.fixture
def ground_ephem_negative() -> GroundEphemeris:
    """GroundEphemeris with negative latitude."""
    return GroundEphemeris(
        latitude=-33.9,
        longitude=18.4,
        height=10.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def ground_ephem_zero() -> GroundEphemeris:
    """GroundEphemeris with zero coordinates."""
    return GroundEphemeris(
        latitude=0.0,
        longitude=0.0,
        height=0.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )
