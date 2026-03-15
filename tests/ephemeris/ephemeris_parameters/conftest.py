"""Fixtures for test_ephemeris_parameters tests."""

import os

# Import constants from main conftest
import sys

import pytest

from rust_ephem import GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from conftest import BEGIN_TIME, END_TIME, STEP_SIZE, VALID_TLE1, VALID_TLE2


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
