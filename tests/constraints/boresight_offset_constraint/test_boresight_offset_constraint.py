import numpy as np

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import BoresightOffsetConstraint, SunConstraint

from .conftest import rotate_radec_reference


def test_boresight_offset_pydantic_json_roundtrip() -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=2.5)
    assert isinstance(config, BoresightOffsetConstraint)

    rust_constraint = Constraint.from_json(config.model_dump_json())
    js = rust_constraint.to_json()

    assert '"type":"boresight_offset"' in js.replace(" ", "")
    assert '"yaw_deg":2.5' in js.replace(" ", "")


def test_boresight_offset_yaw_matches_manual_ra_shift(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem

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


def test_boresight_offset_roll_pitch_yaw_matches_reference_rotation(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem

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
        rotate_radec_reference(ra, dec, roll_deg, pitch_deg, yaw_deg)
        for ra, dec in zip(target_ras, target_decs)
    ]
    ref_ras = [p[0] for p in rotated]
    ref_decs = [p[1] for p in rotated]
    manual_result = base.in_constraint_batch(ephem, ref_ras, ref_decs)

    assert np.array_equal(wrapped_result, manual_result)
