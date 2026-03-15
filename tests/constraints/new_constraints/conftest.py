"""Fixtures for test_new_constraints tests."""

import numpy as np
import pytest

import rust_ephem


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
