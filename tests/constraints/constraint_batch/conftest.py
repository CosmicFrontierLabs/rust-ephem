"""Fixtures for test_constraint_batch tests."""

import datetime

import pytest

from rust_ephem import GroundEphemeris, SunConstraint
from rust_ephem.constraints import EarthLimbConstraint


@pytest.fixture
def ground_ephem_2h() -> GroundEphemeris:
    """Ground ephemeris for 2 hours with 1-hour steps."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
    return GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 3 time points
    )


@pytest.fixture
def ground_ephem_5h() -> GroundEphemeris:
    """Ground ephemeris for 5 hours with 1-hour steps."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 5, 0, 0, tzinfo=datetime.timezone.utc)
    return GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 6 time points
    )


@pytest.fixture
def ground_ephem_earth_limb() -> GroundEphemeris:
    """Ground ephemeris for Earth limb testing."""
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 18, 0, 0, tzinfo=datetime.timezone.utc)
    return GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 7 time points
    )


@pytest.fixture
def ground_ephem_earth_limb_2h() -> GroundEphemeris:
    """Ground ephemeris for Earth limb testing (2 hours)."""
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 14, 0, 0, tzinfo=datetime.timezone.utc)
    return GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour -> 3 time points
    )


@pytest.fixture
def ground_ephem_earth_limb_large() -> GroundEphemeris:
    """Ground ephemeris for large-scale Earth limb testing."""
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
    return GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        600,  # step_size: 10 minutes -> 7 time points
    )


@pytest.fixture
def sun_constraint_45() -> SunConstraint:
    """Sun proximity constraint with 45 degree exclusion."""
    return SunConstraint(min_angle=45.0)


@pytest.fixture
def earth_limb_constraint_20() -> EarthLimbConstraint:
    """Earth limb constraint with 20 degree margin."""
    return EarthLimbConstraint(min_angle=20.0)


@pytest.fixture
def earth_limb_constraint_20_70() -> EarthLimbConstraint:
    """Earth limb constraint with min 20 and max 70 degree angles."""
    return EarthLimbConstraint(min_angle=20.0, max_angle=70.0)
