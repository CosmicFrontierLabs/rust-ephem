from datetime import datetime, timezone

import numpy as np

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import BoresightOffsetConstraint, SunConstraint


def _make_tle_ephem() -> rust_ephem.TLEEphemeris:
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 23, 3, 0, 0, tzinfo=timezone.utc)
    step_s = 600
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_s)


def test_boresight_offset_pydantic_json_roundtrip() -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=2.5)
    assert isinstance(config, BoresightOffsetConstraint)

    rust_constraint = Constraint.from_json(config.model_dump_json())
    js = rust_constraint.to_json()

    assert '"type":"boresight_offset"' in js.replace(" ", "")
    assert '"yaw_deg":2.5' in js.replace(" ", "")


def test_boresight_offset_yaw_matches_manual_ra_shift() -> None:
    ephem = _make_tle_ephem()

    base = Constraint.sun_proximity(45.0)
    wrapped = Constraint.boresight_offset(base, yaw_deg=5.0)

    target_ras = [5.0, 45.0, 135.0, 225.0, 315.0]
    target_decs = [-20.0, -5.0, 0.0, 10.0, 25.0]

    wrapped_result = wrapped.in_constraint_batch(ephem, target_ras, target_decs)

    shifted_ras = [((ra + 5.0) % 360.0) for ra in target_ras]
    manual_result = base.in_constraint_batch(ephem, shifted_ras, target_decs)

    assert np.array_equal(wrapped_result, manual_result)


def test_shared_axis_combination_constructs() -> None:
    primary = Constraint.sun_proximity(45.0)
    secondary = Constraint.moon_proximity(15.0)

    combined = Constraint.and_(
        primary,
        Constraint.boresight_offset(secondary, pitch_deg=1.5, yaw_deg=-0.7),
    )

    js = combined.to_json().replace(" ", "")
    assert '"type":"and"' in js
    assert '"type":"boresight_offset"' in js


def _rotate_radec_reference(
    ra_deg: float,
    dec_deg: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[float, float]:
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


def test_boresight_offset_roll_pitch_yaw_matches_reference_rotation() -> None:
    ephem = _make_tle_ephem()

    base = Constraint.sun_proximity(45.0)
    roll_deg = 8.0
    pitch_deg = -12.0
    yaw_deg = 17.0
    wrapped = Constraint.boresight_offset(
        base,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )

    target_ras = [15.0, 73.0, 149.0, 251.0, 332.0]
    target_decs = [-40.0, -12.0, 5.0, 22.0, 58.0]

    wrapped_result = wrapped.in_constraint_batch(ephem, target_ras, target_decs)

    rotated = [
        _rotate_radec_reference(ra, dec, roll_deg, pitch_deg, yaw_deg)
        for ra, dec in zip(target_ras, target_decs)
    ]
    ref_ras = [p[0] for p in rotated]
    ref_decs = [p[1] for p in rotated]
    manual_result = base.in_constraint_batch(ephem, ref_ras, ref_decs)

    assert np.array_equal(wrapped_result, manual_result)
