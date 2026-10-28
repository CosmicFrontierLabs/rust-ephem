"""Compare optimized free-roll evaluation with exhaustive fixed-roll evaluation."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import (
    BodyConstraint,
    BrightStarConstraint,
    GroundEphemeris,
    SolarRollConstraint,
    SunConstraint,
)
from rust_ephem.constraints import RustConstraintMixin


@pytest.fixture(scope="module")
def sweep_ephemeris(ensure_planetary_data: None) -> GroundEphemeris:
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return GroundEphemeris(35, -120, 0, begin, begin + timedelta(hours=6), 3600)


def constraints() -> dict[str, RustConstraintMixin]:
    rng = np.random.default_rng(123)
    stars = np.column_stack(
        [rng.uniform(0, 360, 160), np.degrees(np.arcsin(rng.uniform(-1, 1, 160)))]
    ).tolist()
    polygon = [(-12.0, 8.0), (12.0, 8.0), (12.0, 20.0), (-12.0, 20.0)]
    bright = BrightStarConstraint(stars=stars, fov_polygon=polygon)
    solar = SolarRollConstraint(tolerance_deg=19, panel_normal=(0, 0.6, 0.8))
    body = BodyConstraint(body="Moon", fov_polygon=polygon)
    sun = SunConstraint(min_angle=54, max_angle=126)
    return {
        "singleton_or": bright | sun,
        "singleton_and": bright & sun,
        "coupled_or": bright | solar | body | sun,
        "coupled_and": (bright | solar) & body & sun,
        "nested": ((bright ^ solar) | body) & ~sun,
        "threshold": bright.at_least(2, solar, body) | sun,
        "negated": ~(bright | solar | body),
        "body": body,
        "fallback": sun.boresight_offset(pitch_deg=11, yaw_deg=4) | solar | body,
    }


def exhaustive(
    constraint: RustConstraintMixin,
    ephemeris: GroundEphemeris,
    ras: list[float],
    decs: list[float],
    indices: list[int] | None,
    n_roll_samples: int,
) -> npt.NDArray[np.bool_]:
    result: npt.NDArray[np.bool_] | None = None
    for step in range(n_roll_samples):
        mask = constraint.in_constraint_batch(
            ephemeris,
            ras,
            decs,
            indices=indices,
            target_rolls=[step * (360 / n_roll_samples)] * len(ras),
        )
        result = mask if result is None else result & mask
    assert result is not None
    return result


@pytest.mark.parametrize("name", list(constraints()))
@pytest.mark.parametrize("n_roll_samples", [1, 12, 37])
def test_free_roll_matches_exhaustive_fixed_rolls(
    sweep_ephemeris: GroundEphemeris, name: str, n_roll_samples: int
) -> None:
    rng = np.random.default_rng(456)
    ras = [0, 360, 90, 270, *rng.uniform(0, 360, 44).tolist()]
    decs = [90, -90, 89.99999, -89.99999, *rng.uniform(-80, 80, 44).tolist()]
    constraint = constraints()[name]
    # Include nonzero, reordered, and duplicate indices. Preparation uses local
    # output columns, which must not be confused with ephemeris indices.
    indices = [5, 1, 5]
    actual = constraint.in_constraint_batch(
        sweep_ephemeris, ras, decs, indices=indices, n_roll_samples=n_roll_samples
    )
    expected = exhaustive(
        constraint, sweep_ephemeris, ras, decs, indices, n_roll_samples
    )
    np.testing.assert_array_equal(actual, expected)
    # Reusing the constraint with a different time selection must not reuse stale
    # prepared geometry from the preceding call.
    other = constraint.in_constraint_batch(
        sweep_ephemeris, ras, decs, indices=[0], n_roll_samples=n_roll_samples
    )
    np.testing.assert_array_equal(
        other, exhaustive(constraint, sweep_ephemeris, ras, decs, [0], n_roll_samples)
    )


@pytest.mark.parametrize("operator", ["or", "and"])
@pytest.mark.parametrize("empty", ["targets", "times"])
def test_empty_prepared_sweep(
    sweep_ephemeris: GroundEphemeris, operator: str, empty: str
) -> None:
    bright = BrightStarConstraint(stars=[], fov_polygon=[(-1, 1), (1, 1), (0, 2)])
    solar = SolarRollConstraint(tolerance_deg=10)
    constraint = bright | solar if operator == "or" else bright & solar
    ras: list[float]
    decs: list[float]
    ras, decs = ([], []) if empty == "targets" else ([0.0], [0.0])
    indices = [0, 3] if empty == "targets" else []
    result = constraint.in_constraint_batch(sweep_ephemeris, ras, decs, indices=indices)
    assert result.shape == (len(ras), len(indices))


@pytest.mark.parametrize("min_angle,max_angle", [(0, 180), (180, None)])
def test_independent_sibling_resolves_cells(
    sweep_ephemeris: GroundEphemeris, min_angle: float, max_angle: float | None
) -> None:
    bright = BrightStarConstraint(stars=[(0, 0)], fov_radius=5)
    solar = SolarRollConstraint(tolerance_deg=0)
    sun = SunConstraint(min_angle=min_angle, max_angle=max_angle)
    for constraint in [(solar | bright) | sun, (solar | bright) & sun]:
        result = constraint.in_constraint_batch(
            sweep_ephemeris,
            [30, 90, 180],
            [10, -40, 50],
            indices=[1, 5],
            n_roll_samples=12,
        )
        np.testing.assert_array_equal(
            result,
            exhaustive(
                constraint, sweep_ephemeris, [30, 90, 180], [10, -40, 50], [1, 5], 12
            ),
        )


@pytest.mark.parametrize("name", ["coupled_or", "negated", "body", "fallback"])
def test_field_of_regard_matches_exhaustive_sweep(
    sweep_ephemeris: GroundEphemeris, name: str
) -> None:
    n_points = 256
    golden_angle = np.pi * (3 - np.sqrt(5))
    # Same Fibonacci sphere used by the field-of-regard integration.
    z = 1 - 2 * (np.arange(n_points) + 0.5) / n_points
    ras = np.degrees(np.arange(n_points) * golden_angle % (2 * np.pi)).tolist()
    decs = np.degrees(np.arcsin(z)).tolist()
    constraint = constraints()[name]
    expected = exhaustive(constraint, sweep_ephemeris, ras, decs, [5], 37)
    actual = constraint.instantaneous_field_of_regard(
        sweep_ephemeris, index=5, n_points=n_points, n_roll_samples=37
    )
    assert actual == pytest.approx(4 * np.pi * np.mean(~expected))


def test_prepared_geometry_across_chunk_boundary(ensure_planetary_data: None) -> None:
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ephemeris = GroundEphemeris(35, -120, 0, begin, begin + timedelta(minutes=256), 60)
    rng = np.random.default_rng(789)
    ras = rng.uniform(0, 360, 257).tolist()
    decs = rng.uniform(-85, 85, 257).tolist()
    constraint = constraints()["coupled_or"]
    # 257 x 257 crosses the 65,536-cell temporary geometry limit, partway
    # through a target's time series rather than at a row boundary.
    result = constraint.in_constraint_batch(ephemeris, ras, decs, n_roll_samples=7)
    np.testing.assert_array_equal(
        result, exhaustive(constraint, ephemeris, ras, decs, None, 7)
    )


def test_coupled_constraints_cannot_choose_different_rolls(
    sweep_ephemeris: GroundEphemeris,
) -> None:
    first = SolarRollConstraint(tolerance_deg=20, panel_normal=(0, 1, 0))
    opposite = SolarRollConstraint(tolerance_deg=20, panel_normal=(0, -1, 0))
    ras, decs = [30.0, 90.0, 180.0], [10.0, -40.0, 50.0]
    for constraint in [first, opposite]:
        mask = constraint.in_constraint_batch(
            sweep_ephemeris, ras, decs, indices=[1, 5], n_roll_samples=12
        )
        assert not mask.any()  # Each constraint has a clear roll on its own.
    combined = (first | opposite).in_constraint_batch(
        sweep_ephemeris, ras, decs, indices=[1, 5], n_roll_samples=12
    )
    assert combined.all()  # Their acceptable roll ranges never intersect.
