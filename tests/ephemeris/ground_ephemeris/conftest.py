"""Fixtures for ground_ephemeris tests."""

import datetime

import pytest

from rust_ephem import GroundEphemeris


@pytest.fixture
def kitt_peak_obs() -> GroundEphemeris:
    """Create GroundEphemeris for Kitt Peak Observatory."""
    latitude = 31.9583
    longitude = -111.6
    height = 2096.0
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(latitude, longitude, height, begin, end, step_size)


@pytest.fixture
def equator_obs() -> GroundEphemeris:
    """Create GroundEphemeris at equator on prime meridian."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(0.0, 0.0, 0.0, begin, end, step_size)


@pytest.fixture
def mauna_kea_obs() -> GroundEphemeris:
    """Create GroundEphemeris for Mauna Kea."""
    latitude = 19.8207
    longitude = -155.468
    height = 4205.0
    begin = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(latitude, longitude, height, begin, end, step_size)


@pytest.fixture
def multi_time_obs() -> GroundEphemeris:
    """Create GroundEphemeris with multiple time steps."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 6, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(35.0, -120.0, 500.0, begin, end, step_size)
