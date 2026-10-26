"""Per-target None entries keep coordinated free-roll semantics and row order."""

import numpy as np
import pytest

from rust_ephem import (
    BodyConstraint,
    BrightStarConstraint,
    GroundEphemeris,
    SolarRollConstraint,
    SunConstraint,
)
from rust_ephem.constraints import RustConstraintMixin


def mixed_constraints() -> list[RustConstraintMixin]:
    polygon = [(-12.0, 8.0), (12.0, 8.0), (12.0, 20.0), (-12.0, 20.0)]
    solar = SolarRollConstraint(tolerance_deg=19, roll_deg=37)
    bright = BrightStarConstraint(
        stars=[(20.0, 5.0), (110.0, -15.0)], fov_polygon=polygon, roll_deg=21
    )
    body = BodyConstraint(body="Moon", fov_polygon=polygon, roll_deg=31)
    offset = SunConstraint(min_angle=54).boresight_offset(
        roll_deg=17, pitch_deg=12, yaw_deg=8
    )
    return [
        SolarRollConstraint(tolerance_deg=19),
        solar,
        bright,
        body,
        offset,
        solar | bright | body | offset,
    ]


@pytest.mark.parametrize("constraint", mixed_constraints())
@pytest.mark.parametrize("all_free", [False, True])
def test_mixed_rolls_match_individual_evaluations(
    ground_ephem_5h: GroundEphemeris, constraint: RustConstraintMixin, all_free: bool
) -> None:
    ras, decs = (
        [0.0, 30.0, 90.0, 149.0, 270.0, 300.0],
        [30.0, -40.0, 10.0, -20.0, 0.0, 69.0],
    )
    rolls: list[float | None] = (
        [None] * 6 if all_free else [None, 45.0, None, 0.0, -23.0, None]
    )
    indices = [5, 1, 5]
    mask = constraint.in_constraint_batch(
        ground_ephem_5h,
        ras,
        decs,
        indices=indices,
        target_rolls=rolls,
        n_roll_samples=12,
    )
    results = constraint.evaluate_batch(
        ground_ephem_5h,
        ras,
        decs,
        indices=indices,
        target_rolls=rolls,
        n_roll_samples=12,
    )
    assert mask.shape == (6, 3)
    for i, roll in enumerate(rolls):
        expected = constraint.evaluate(
            ground_ephem_5h,
            ras[i],
            decs[i],
            indices=indices,
            target_roll=roll,
            n_roll_samples=12,
        )
        actual = results[i]
        expected_mask = constraint.in_constraint_batch(
            ground_ephem_5h,
            [ras[i]],
            [decs[i]],
            indices=indices,
            target_rolls=None if roll is None else [roll],
            n_roll_samples=12,
        )[0]
        np.testing.assert_array_equal(mask[i], expected_mask)
        assert actual.constraint_array == expected.constraint_array
        np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
        assert actual.model_dump() == expected.model_dump()
        assert actual.cause_value_keys == expected.cause_value_keys
        assert actual.constraint_values.keys() == expected.constraint_values.keys()
        for key, values in actual.constraint_values.items():
            np.testing.assert_allclose(values, expected.constraint_values[key])
