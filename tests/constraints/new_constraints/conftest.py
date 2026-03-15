"""Fixtures for test_new_constraints tests."""

import numpy as np
import pytest

import rust_ephem


@pytest.fixture
def ground_ephemeris() -> rust_ephem.GroundEphemeris:
    """Create a ground ephemeris for testing."""
    from datetime import datetime, timezone

    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=begin,
        end=end,
        step_size=120,  # 2 minutes
    )


@pytest.fixture
def tle_ephemeris() -> rust_ephem.TLEEphemeris:
    """Create a TLE ephemeris for testing."""
    from datetime import datetime, timezone

    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 23, 2, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, 300)


@pytest.fixture
def saa_polygon() -> list[tuple[float, float]]:
    """Simple rectangular SAA polygon for testing."""
    return [
        (-90.0, -50.0),  # Southwest
        (-40.0, -50.0),  # Southeast
        (-40.0, 0.0),  # Northeast
        (-90.0, 0.0),  # Northwest
    ]


def moon_ra_dec_deg(ephem: rust_ephem.Ephemeris, index: int) -> tuple[float, float]:
    """Get moon RA/Dec in degrees for a given ephemeris index."""
    moon_pos = ephem.moon_pv.position[index]
    obs_pos = ephem.gcrs_pv.position[index]
    rel = moon_pos - obs_pos
    dist = np.linalg.norm(rel)
    ra = np.degrees(np.arctan2(rel[1], rel[0]))
    if ra < 0.0:
        ra += 360.0
    dec = np.degrees(np.arcsin(rel[2] / dist))
    return ra, dec


def moon_altitude_deg(ephem: rust_ephem.Ephemeris, index: int) -> float:
    """Get moon altitude in degrees for a given ephemeris index."""
    ra, dec = moon_ra_dec_deg(ephem, index)
    altaz = ephem.radec_to_altaz(ra, dec, time_indices=[index])
    return float(altaz[0][0])
