"""Fixtures for moon_phase_constraint tests."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import rust_ephem


def moon_ra_dec_deg(ephem: "rust_ephem.Ephemeris", index: int) -> tuple[float, float]:
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


def moon_altitude_deg(ephem: "rust_ephem.Ephemeris", index: int) -> float:
    """Get moon altitude in degrees for a given ephemeris index."""
    ra, dec = moon_ra_dec_deg(ephem, index)
    altaz = ephem.radec_to_altaz(ra, dec, time_indices=[index])
    return float(altaz[0][0])
