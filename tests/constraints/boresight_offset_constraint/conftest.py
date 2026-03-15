"""Fixtures for test_boresight_offset_constraint tests."""

from datetime import datetime, timezone

import numpy as np
import pytest

import rust_ephem


@pytest.fixture
def tle_ephem() -> rust_ephem.TLEEphemeris:
    """Create a TLE ephemeris for testing."""
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 23, 3, 0, 0, tzinfo=timezone.utc)
    step_s = 600
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_s)


def rotate_radec_reference(
    ra_deg: float,
    dec_deg: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[float, float]:
    """Rotate RA/Dec coordinates by roll, pitch, yaw angles."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # Unit vector from RA/Dec
    v = np.array(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ],
        dtype=float,
    )

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    # Intrinsic Z-Y-X (yaw, pitch, roll) => R = Rz @ Ry @ Rx
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    v_rot = rz @ ry @ rx @ v

    dec_rot_deg = np.rad2deg(np.arcsin(np.clip(v_rot[2], -1.0, 1.0)))
    ra_rot_deg = np.rad2deg(np.arctan2(v_rot[1], v_rot[0]))
    if ra_rot_deg < 0.0:
        ra_rot_deg += 360.0

    return float(ra_rot_deg), float(dec_rot_deg)
