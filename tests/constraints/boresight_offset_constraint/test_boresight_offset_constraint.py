import numpy as np
import pytest

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import (
    BoresightOffsetConstraint,
    RollReference,
    SunConstraint,
)


def test_boresight_offset_pydantic_json_roundtrip() -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=2.5,
        roll_clockwise=False,
    )
    assert isinstance(config, BoresightOffsetConstraint)

    rust_constraint = Constraint.from_json(config.model_dump_json())
    js = rust_constraint.to_json()

    assert '"type":"boresight_offset"' in js.replace(" ", "")
    assert '"yaw_deg":2.5' in js.replace(" ", "")
    assert '"roll_clockwise":false' in js.replace(" ", "")


def test_boresight_offset_definition_accepts_fixed_roll_pitch_yaw() -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(
        roll_deg=1.5,
        roll_clockwise=True,
        roll_reference=RollReference.NORTH,
        pitch_deg=0.8,
        yaw_deg=-0.4,
    )
    assert config.roll_deg == 1.5
    assert config.roll_clockwise is True
    assert config.roll_reference == "north"


def test_boresight_offset_roll_reference_serializes() -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(
        roll_deg=5.0,
        roll_clockwise=False,
        roll_reference=RollReference.NORTH,
        pitch_deg=0.5,
        yaw_deg=1.0,
    )

    js = config.model_dump_json().replace(" ", "")
    assert '"roll_reference":"north"' in js


def test_boresight_offset_roll_optional_when_no_offset(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem

    base = Constraint.sun_proximity(45.0)
    wrapped = Constraint.boresight_offset(base, pitch_deg=0.0, yaw_deg=0.0)

    target_ras = [5.0, 45.0, 135.0, 225.0, 315.0]
    target_decs = [-20.0, -5.0, 0.0, 10.0, 25.0]

    wrapped_result = wrapped.in_constraint_batch(ephem, target_ras, target_decs)
    manual_result = base.in_constraint_batch(ephem, target_ras, target_decs)

    assert np.array_equal(wrapped_result, manual_result)


def test_shared_axis_combination_constructs() -> None:
    primary = Constraint.sun_proximity(45.0)
    secondary = Constraint.moon_proximity(15.0)

    combined = Constraint.and_(
        primary,
        Constraint.boresight_offset(
            secondary,
            roll_deg=0.0,
            pitch_deg=1.5,
            yaw_deg=-0.7,
        ),
    )

    js = combined.to_json().replace(" ", "")
    assert '"type":"and"' in js
    assert '"type":"boresight_offset"' in js


def test_eval_accepts_default_spacecraft_roll_for_offset_sensitive_boresight(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    config = SunConstraint(min_angle=45.0).boresight_offset(
        pitch_deg=1.5,
        yaw_deg=-0.7,
        roll_reference=RollReference.NORTH,
    )
    default_result = config.evaluate(tle_ephem, target_ra=10.0, target_dec=5.0)
    assert isinstance(default_result.all_satisfied, bool)

    result = config.evaluate(
        tle_ephem,
        target_ra=10.0,
        target_dec=5.0,
        target_roll=12.0,
    )
    assert isinstance(result.all_satisfied, bool)


def test_boresight_offset_roll_direction_is_configurable(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem

    base = Constraint.sun_proximity(45.0)

    wrapped_ccw = Constraint.boresight_offset(
        base,
        roll_deg=12.0,
        roll_clockwise=False,
        pitch_deg=2.5,
        yaw_deg=-1.5,
    )
    wrapped_cw = Constraint.boresight_offset(
        base,
        roll_deg=12.0,
        roll_clockwise=True,
        pitch_deg=2.5,
        yaw_deg=-1.5,
    )

    target_ras = [15.0, 73.0, 149.0, 251.0, 332.0]
    target_decs = [-40.0, -12.0, 5.0, 22.0, 58.0]

    ccw_result = wrapped_ccw.in_constraint_batch(ephem, target_ras, target_decs)
    cw_result = wrapped_cw.in_constraint_batch(ephem, target_ras, target_decs)

    assert ccw_result.shape == cw_result.shape
    assert '"roll_clockwise":false' in wrapped_ccw.to_json().replace(" ", "")
    assert '"roll_clockwise":true' in wrapped_cw.to_json().replace(" ", "")


def test_low_level_constraint_api_accepts_target_roll(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem
    target_ras = [15.0, 73.0, 149.0, 251.0, 332.0]
    target_decs = [-40.0, -12.0, 5.0, 22.0, 58.0]

    # Base boresight config with clockwise command convention.
    base = Constraint.boresight_offset(
        Constraint.sun_proximity(45.0),
        roll_deg=5.0,
        roll_clockwise=True,
        roll_reference="north",
        pitch_deg=2.5,
        yaw_deg=-1.5,
    )

    via_target_roll = base.in_constraint_batch(
        ephem,
        target_ras,
        target_decs,
        target_roll=12.0,
    )

    # With clockwise=True, target_roll is added with opposite sign.
    expected = Constraint.boresight_offset(
        Constraint.sun_proximity(45.0),
        roll_deg=5.0 - 12.0,
        roll_clockwise=True,
        roll_reference="north",
        pitch_deg=2.5,
        yaw_deg=-1.5,
    ).in_constraint_batch(ephem, target_ras, target_decs)

    assert np.array_equal(via_target_roll, expected)


def test_free_roll_for_gte_fixed_roll_for(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """FoR with no spacecraft roll specified sweeps all rolls, giving >= any single-roll FoR.

    When target_roll is not specified (None), instantaneous_field_of_regard sweeps all
    possible spacecraft roll angles for boresight-offset constraints with non-zero pitch/yaw.
    The result is always >= the FoR at any single fixed spacecraft roll.
    """
    ephem = tle_ephem

    c = SunConstraint(min_angle=45.0).boresight_offset(pitch_deg=30.0, yaw_deg=0.0)

    # target_roll=None (default) → sweep all spacecraft rolls
    for_sweep = c.instantaneous_field_of_regard(ephem, index=0, n_points=500)
    # target_roll=0.0 → evaluate at a single specific spacecraft roll
    for_fixed = c.instantaneous_field_of_regard(
        ephem, index=0, n_points=500, target_roll=0.0
    )

    assert for_sweep >= for_fixed, (
        f"Sweep FoR ({for_sweep:.4f} sr) should be >= fixed-roll FoR ({for_fixed:.4f} sr)"
    )


def test_instantaneous_for_zero_n_roll_samples_raises(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """n_roll_samples=0 must raise ValueError (no roll angles to sweep)."""
    ephem = tle_ephem

    c = Constraint.sun_proximity(45.0)

    with pytest.raises(ValueError, match="n_roll_samples must be greater than 0"):
        c.instantaneous_field_of_regard(ephem, index=0, n_points=100, n_roll_samples=0)
