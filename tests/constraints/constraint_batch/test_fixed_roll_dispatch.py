"""Explicit rolls reuse the tree while matching the established JSON injection."""

from typing import Any, cast

import numpy as np
import pytest

from rust_ephem import (
    BodyConstraint,
    BrightStarConstraint,
    Constraint,
    GroundEphemeris,
    SolarRollConstraint,
    SunConstraint,
)
from rust_ephem.constraints import ConstraintResult, RollReference, RustConstraintMixin


def fixed_roll_constraints() -> dict[str, RustConstraintMixin]:
    polygon = [(-12.0, 8.0), (12.0, 8.0), (12.0, 20.0), (-12.0, 20.0)]
    rng = np.random.default_rng(314)
    stars = rng.uniform([0, -80], [360, 80], (100, 2)).tolist()
    bright = BrightStarConstraint(stars=stars, fov_polygon=polygon, roll_deg=31)
    body = BodyConstraint(body="Moon", fov_polygon=polygon, roll_deg=-17)
    solar = SolarRollConstraint(tolerance_deg=19, roll_deg=47)
    sun = SunConstraint(min_angle=54, max_angle=126)
    nested = bright.at_least(2, solar, body) | ~(bright ^ sun)
    return {
        "circle": BrightStarConstraint(stars=stars, fov_radius=8),
        "polygon": bright,
        "solar": solar,
        "body": body,
        "and": bright & solar & body,
        "nested": nested,
        "north_offset": nested.boresight_offset(
            roll_deg=17,
            pitch_deg=8,
            yaw_deg=3,
            roll_clockwise=True,
            roll_reference=RollReference.NORTH,
        ),
        "sun_offset": (bright | solar | body).boresight_offset(
            roll_deg=-23, pitch_deg=11, yaw_deg=4, roll_reference=RollReference.SUN
        ),
        "multiple_offsets": (
            (bright | solar).boresight_offset(
                roll_deg=17, pitch_deg=8, roll_clockwise=True
            )
            | body.boresight_offset(roll_deg=38, yaw_deg=3)
        ),
        "nested_offsets": body.boresight_offset(
            roll_deg=12, pitch_deg=5
        ).boresight_offset(roll_deg=-8, yaw_deg=6, roll_clockwise=True),
    }


def assert_results_match(actual: Any, expected: Any) -> None:
    np.testing.assert_array_equal(actual.constraint_array, expected.constraint_array)
    assert actual.constraint_name == expected.constraint_name
    assert actual.all_satisfied == expected.all_satisfied
    assert actual.cause_value_keys == expected.cause_value_keys
    assert actual.constraint_values.keys() == expected.constraint_values.keys()
    for key in expected.constraint_values:
        np.testing.assert_allclose(
            actual.constraint_values[key],
            expected.constraint_values[key],
            rtol=1e-12,
            atol=1e-12,
        )
    for field in ["violations", "visibility"]:
        actual_items = getattr(actual, field)
        expected_items = getattr(expected, field)
        assert len(actual_items) == len(expected_items)
        keys = (
            ["start_time", "end_time", "max_severity", "description"]
            if field == "violations"
            else ["start_time", "end_time", "start_cause", "end_cause"]
        )
        for a, b in zip(actual_items, expected_items):
            for key in keys:
                assert getattr(a, key) == getattr(b, key)


@pytest.mark.parametrize("name", list(fixed_roll_constraints()))
def test_explicit_rolls_match_json_injection(
    ground_ephem_5h: GroundEphemeris, name: str
) -> None:
    constraint = fixed_roll_constraints()[name]
    original_json = constraint.model_dump_json()
    backend = Constraint.from_json(original_json)
    ras = [0.0, 360.0, 33.0, 149.0, 270.0, 20.0]
    decs = [90.0, -90.0, 34.0, -20.0, 0.0, 69.0]
    rolls = [17.0, -23.0, 377.0, 17.0, 0.0, -0.0]
    indices = [5, 1, 5]
    free_before = backend.in_constraint_batch(
        ground_ephem_5h, ras, decs, indices=indices, n_roll_samples=12
    )
    raw_mask = backend.in_constraint_batch(
        ground_ephem_5h, ras, decs, indices=indices, target_rolls=rolls
    )
    py_mask = constraint.in_constraint_batch(
        ground_ephem_5h, ras, decs, indices=indices, target_rolls=rolls
    )
    raw_results = backend.evaluate_batch(
        ground_ephem_5h, ras, decs, indices=indices, target_rolls=rolls
    )
    py_results = constraint.evaluate_batch(
        ground_ephem_5h, ras, decs, indices=indices, target_rolls=rolls
    )
    for i, roll in enumerate(rolls):
        # This independent implementation of roll injection remains in use by
        # the explicit-roll FoR API; it constructs the fully configured tree.
        reference = cast(Any, constraint)._to_rust_constraint(target_roll=roll)
        expected = reference.evaluate(ground_ephem_5h, ras[i], decs[i], indices=indices)
        expected_mask = reference.in_constraint_batch(
            ground_ephem_5h, [ras[i]], [decs[i]], indices=indices
        )[0]
        np.testing.assert_array_equal(raw_mask[i], expected_mask)
        np.testing.assert_array_equal(py_mask[i], expected_mask)
        assert_results_match(raw_results[i], expected)
        assert_results_match(py_results[i], expected)
        single = backend.evaluate(
            ground_ephem_5h, ras[i], decs[i], indices=indices, target_roll=roll
        )
        assert_results_match(single, expected)
        py_single: ConstraintResult = constraint.evaluate(
            ground_ephem_5h, ras[i], decs[i], indices=indices, target_roll=roll
        )
        assert_results_match(py_single, expected)
    assert constraint.model_dump_json() == original_json
    np.testing.assert_array_equal(
        free_before,
        backend.in_constraint_batch(
            ground_ephem_5h, ras, decs, indices=indices, n_roll_samples=12
        ),
    )


@pytest.mark.parametrize("roll", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rolls_are_rejected(
    ground_ephem_5h: GroundEphemeris, roll: float
) -> None:
    constraint = fixed_roll_constraints()["polygon"]
    backends: list[Constraint | RustConstraintMixin] = [
        constraint,
        Constraint.from_json(constraint.model_dump_json()),
    ]
    for backend in backends:
        with pytest.raises(ValueError, match="finite"):
            backend.in_constraint_batch(
                ground_ephem_5h, [0.0], [0.0], target_rolls=[roll]
            )
        with pytest.raises(ValueError, match="finite"):
            backend.evaluate_batch(ground_ephem_5h, [0.0], [0.0], target_rolls=[roll])
        with pytest.raises(ValueError, match="finite"):
            backend.evaluate(ground_ephem_5h, 0.0, 0.0, target_roll=roll)


def test_empty_fixed_roll_batch_shape(ground_ephem_5h: GroundEphemeris) -> None:
    constraint = fixed_roll_constraints()["polygon"]
    backends: list[Constraint | RustConstraintMixin] = [
        constraint,
        Constraint.from_json(constraint.model_dump_json()),
    ]
    for backend in backends:
        result = backend.in_constraint_batch(
            ground_ephem_5h, [], [], indices=[5, 1], target_rolls=[]
        )
        assert result.shape == (0, 2)
        assert backend.evaluate_batch(ground_ephem_5h, [], [], target_rolls=[]) == []


def test_single_target_multiple_times_with_fixed_roll(
    ground_ephem_5h: GroundEphemeris,
) -> None:
    constraint = fixed_roll_constraints()["north_offset"]
    backend = Constraint.from_json(constraint.model_dump_json())
    times = [ground_ephem_5h.timestamp[i] for i in [1, 5]]
    expected = (
        cast(Any, constraint)
        ._to_rust_constraint(target_roll=31)
        .in_constraint_batch(ground_ephem_5h, [33.0], [34.0], times=times)[0]
        .tolist()
    )
    evaluators: list[Constraint | RustConstraintMixin] = [constraint, backend]
    for evaluator in evaluators:
        assert (
            evaluator.in_constraint(times, ground_ephem_5h, 33.0, 34.0, target_roll=31)
            == expected
        )


def test_fixed_roll_apis_reuse_one_backend(
    ground_ephem_5h: GroundEphemeris, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Constraint.from_json
    creations = 0

    def count_creations(config: str) -> Constraint:
        nonlocal creations
        creations += 1
        return factory(config)

    monkeypatch.setattr(Constraint, "from_json", count_creations)
    constraint = fixed_roll_constraints()["polygon"]
    ras, decs, rolls = [10.0, 20.0, 30.0], [5.0, -10.0, 45.0], [0.0, 12.0, 37.0]
    constraint.in_constraint_batch(ground_ephem_5h, ras, decs, target_rolls=rolls)
    constraint.evaluate_batch(ground_ephem_5h, ras, decs, target_rolls=rolls)
    constraint.evaluate(ground_ephem_5h, 10.0, 5.0, target_roll=24.0)
    constraint.in_constraint(
        ground_ephem_5h.timestamp[0], ground_ephem_5h, 10.0, 5.0, target_roll=-13.0
    )
    assert creations == 1
