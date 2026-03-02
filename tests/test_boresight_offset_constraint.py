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
