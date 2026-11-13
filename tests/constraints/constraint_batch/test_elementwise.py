"""Elementwise evaluation must equal the diagonal of the Cartesian batch APIs."""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from rust_ephem import (
    AirmassConstraint,
    AltAzConstraint,
    BodyConstraint,
    BrightStarConstraint,
    Constraint,
    DaytimeConstraint,
    EarthLimbConstraint,
    EclipseConstraint,
    ElementwiseResult,
    GroundEphemeris,
    MoonConstraint,
    MoonPhaseConstraint,
    OrbitPoleConstraint,
    OrbitRamConstraint,
    SAAConstraint,
    SolarRollConstraint,
    SunConstraint,
    TLEEphemeris,
)
from rust_ephem.constraints import RollReference, RustConstraintMixin

N_ROLL = 24


def spacecraft_constraints() -> dict[str, RustConstraintMixin]:
    sun = SunConstraint(min_angle=45, max_angle=150)
    moon = MoonConstraint(min_angle=20)
    earth = EarthLimbConstraint(min_angle=15, include_refraction=True)
    eclipse = EclipseConstraint(umbra_only=False)
    polygon = [(-12.0, 8.0), (12.0, 8.0), (12.0, 20.0), (-12.0, 20.0)]
    return {
        "sun": sun,
        "moon": moon,
        "earth_limb": earth,
        "eclipse": eclipse,
        "saa": SAAConstraint(polygon=[(-80.0, -40.0), (0.0, -40.0), (0.0, 0.0)]),
        "orbit_ram": OrbitRamConstraint(min_angle=10, max_angle=120),
        "orbit_pole": OrbitPoleConstraint(min_angle=30, earth_limb_pole=True),
        "moon_phase": MoonPhaseConstraint(max_illumination=0.4, min_distance=20),
        "body": BodyConstraint(body="Jupiter barycenter", min_angle=30),
        "bright_star": BrightStarConstraint(
            stars=[(0.0, 0.0), (50.0, 30.0)], fov_radius=20
        ),
        "and": sun & moon & earth,
        "or": sun | moon | earth,
        "not": ~sun,
        "xor": sun ^ moon,
        "at_least": sun.at_least(2, moon, earth, eclipse),
        "nested": ~(sun ^ moon) | (earth & ~eclipse) | SunConstraint(min_angle=60),
        "boresight_fixed": (sun | moon).boresight_offset(
            roll_deg=17, pitch_deg=11, yaw_deg=4, roll_reference=RollReference.NORTH
        ),
        # Roll-dependent trees: free roll is swept, as in in_constraint_batch.
        "boresight_sun_ref": (sun | earth).boresight_offset(
            pitch_deg=25, yaw_deg=5, roll_reference=RollReference.SUN
        ),
        "solar_roll": SolarRollConstraint(tolerance_deg=20),
        "polygon_or": BrightStarConstraint(
            stars=[(20.0, 5.0), (110.0, -15.0)], fov_polygon=polygon, roll_deg=21
        )
        | sun,
    }


def ground_constraints() -> dict[str, RustConstraintMixin]:
    return {
        "airmass": AirmassConstraint(max_airmass=2.0, min_airmass=1.05),
        "alt_az": AltAzConstraint(min_altitude=20, max_altitude=70),
        "daytime": DaytimeConstraint(),
        "combo": AirmassConstraint(max_airmass=2.0)
        | DaytimeConstraint() ^ AltAzConstraint(min_altitude=30),
    }


def random_elements(
    timestamps: Any, n: int, seed: int
) -> tuple[list[float], list[float], np.ndarray]:
    rng = np.random.default_rng(seed)
    ras = rng.uniform(0, 360, n).tolist()
    decs = np.degrees(np.arcsin(rng.uniform(-1, 1, n))).tolist()
    idx = rng.integers(0, len(timestamps), n)
    return ras, decs, idx


def assert_matches_grid(
    constraint: Any,
    ephem: Any,
    ras: list[float],
    decs: list[float],
    idx: np.ndarray,
    target_rolls: Any = None,
) -> None:
    times = [ephem.timestamp[i] for i in idx]
    kwargs: dict[str, Any] = {"n_roll_samples": N_ROLL}
    if target_rolls is not None:
        kwargs["target_rolls"] = target_rolls
    grid = constraint.in_constraint_batch(ephem, ras, decs, **kwargs)
    expected = grid[np.arange(len(ras)), idx]
    mask = constraint.in_constraint_elementwise(ephem, ras, decs, times, **kwargs)
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, expected)

    result = constraint.evaluate_elementwise(ephem, ras, decs, times, **kwargs)
    np.testing.assert_array_equal(result.constraint_array, expected)
    np.testing.assert_array_equal(result.visibility_flags, ~expected)
    rows = constraint.evaluate_batch(ephem, ras, decs, **kwargs)
    assert result.constraint_values.keys() == rows[0].constraint_values.keys()
    for key, values in result.constraint_values.items():
        assert values.shape == (len(ras),)
        reference = [row.constraint_values[key][t] for row, t in zip(rows, idx)]
        np.testing.assert_allclose(values, reference, rtol=1e-9, atol=1e-9)
    assert {k: sorted(v) for k, v in result.cause_value_keys.items()} == {
        k: sorted(v) for k, v in rows[0].cause_value_keys.items()
    }
    assert set(result.component_violated) == set(result.cause_value_keys)
    assert list(result.timestamp) == times
    np.testing.assert_array_equal(result.ras, ras)
    np.testing.assert_array_equal(result.decs, decs)


@pytest.mark.parametrize("name", list(spacecraft_constraints()))
@pytest.mark.parametrize("backend_kind", ["python", "rust"])
def test_spacecraft_matches_grid_diagonal(
    tle_ephemeris: TLEEphemeris, name: str, backend_kind: str
) -> None:
    constraint: Any = spacecraft_constraints()[name]
    if backend_kind == "rust":
        # The definition-only backend: boresight roll fixed at its configured value.
        constraint = Constraint.from_json(constraint.model_dump_json())
    ras, decs, idx = random_elements(tle_ephemeris.timestamp, 40, seed=7)
    assert_matches_grid(constraint, tle_ephemeris, ras, decs, idx)


@pytest.mark.parametrize("name", list(ground_constraints()))
def test_ground_matches_grid_diagonal(
    ground_ephemeris: GroundEphemeris, name: str
) -> None:
    ras, decs, idx = random_elements(ground_ephemeris.timestamp, 40, seed=11)
    assert_matches_grid(ground_constraints()[name], ground_ephemeris, ras, decs, idx)


@pytest.mark.parametrize(
    "name", ["boresight_sun_ref", "solar_roll", "polygon_or", "sun"]
)
def test_mixed_target_rolls_match_grid(tle_ephemeris: TLEEphemeris, name: str) -> None:
    constraint = spacecraft_constraints()[name]
    ras, decs, idx = random_elements(tle_ephemeris.timestamp, 12, seed=3)
    rolls = [None, 45.0, None, 0.0, -23.0, None, 90.0, None, 10.0, 10.0, None, 200.0]
    assert_matches_grid(constraint, tle_ephemeris, ras, decs, idx, rolls)
    fixed = [r if r is not None else 5.0 for r in rolls]
    assert_matches_grid(constraint, tle_ephemeris, ras, decs, idx, fixed)


@pytest.mark.parametrize("name", ["sun", "or", "nested", "boresight_sun_ref", "body"])
def test_off_grid_matches_single_time_evaluation(
    tle_ephemeris: TLEEphemeris, name: str
) -> None:
    constraint = spacecraft_constraints()[name]
    ts = tle_ephemeris.timestamp
    times = [
        ts[3] + timedelta(seconds=37),
        ts[10] + timedelta(seconds=1.5),
        ts[3] + timedelta(seconds=37),  # duplicate
        ts[0],  # exact sample mixed with off-grid ones
        ts[-2] + timedelta(seconds=119),
    ]
    ras = [10.0, 120.0, 250.0, 300.0, 45.0]
    decs = [5.0, -30.0, 60.0, -75.0, 20.0]
    mask = constraint.in_constraint_elementwise(
        tle_ephemeris, ras, decs, times, n_roll_samples=N_ROLL
    )
    result = constraint.evaluate_elementwise(
        tle_ephemeris, ras, decs, times, n_roll_samples=N_ROLL
    )
    np.testing.assert_array_equal(result.constraint_array, mask)
    for i, (t, ra, dec) in enumerate(zip(times, ras, decs)):
        assert mask[i] == constraint.in_constraint(
            t, tle_ephemeris, ra, dec, n_roll_samples=N_ROLL
        )
        single = constraint.evaluate(
            tle_ephemeris, ra, dec, times=[t], n_roll_samples=N_ROLL
        )
        assert single.constraint_values.keys() == result.constraint_values.keys()
        for key, values in result.constraint_values.items():
            np.testing.assert_allclose(
                values[i], single.constraint_values[key][0], rtol=1e-9, atol=1e-9
            )


def test_unsorted_duplicate_times_and_datetime64(tle_ephemeris: TLEEphemeris) -> None:
    constraint = spacecraft_constraints()["or"]
    ts = tle_ephemeris.timestamp
    order = [7, 2, 7, 0, 30, 2, 59, 7]
    times = [ts[i] + timedelta(seconds=13) for i in order]
    ras = np.linspace(0, 350, len(order))
    decs = np.linspace(-70, 70, len(order))
    result = constraint.evaluate_elementwise(tle_ephemeris, ras, decs, times)
    # Reference: each element on its own.
    for i, t in enumerate(times):
        single = constraint.evaluate_elementwise(
            tle_ephemeris, ras[i : i + 1], decs[i : i + 1], [t]
        )
        assert result.constraint_array[i] == single.constraint_array[0]
        for key, values in result.constraint_values.items():
            assert values[i] == single.constraint_values[key][0]
    as_datetime64 = np.array(
        [t.replace(tzinfo=None) for t in times], dtype="datetime64[us]"
    )
    again = constraint.in_constraint_elementwise(
        tle_ephemeris, ras, decs, as_datetime64
    )
    np.testing.assert_array_equal(again, result.constraint_array)


@pytest.mark.parametrize("backend_kind", ["python", "rust"])
def test_times_none_pairs_with_ephemeris_rows(
    tle_ephemeris: TLEEphemeris, backend_kind: str
) -> None:
    constraint: Any = spacecraft_constraints()["nested"]
    if backend_kind == "rust":
        constraint = Constraint.from_json(constraint.model_dump_json())
    n = len(tle_ephemeris.timestamp)
    ras = np.linspace(0, 359, n).tolist()
    decs = np.linspace(-80, 80, n).tolist()
    expected = np.diag(constraint.in_constraint_batch(tle_ephemeris, ras, decs))
    np.testing.assert_array_equal(
        constraint.in_constraint_elementwise(tle_ephemeris, ras, decs), expected
    )
    result = constraint.evaluate_elementwise(tle_ephemeris, ras, decs)
    np.testing.assert_array_equal(result.constraint_array, expected)
    assert list(result.timestamp) == list(tle_ephemeris.timestamp)
    assert len(result) == n


def test_length_and_range_errors(tle_ephemeris: TLEEphemeris) -> None:
    constraint = SunConstraint(min_angle=45)
    ts = tle_ephemeris.timestamp
    for method in (
        constraint.in_constraint_elementwise,
        constraint.evaluate_elementwise,
    ):
        with pytest.raises(ValueError, match="same length"):
            method(tle_ephemeris, [0.0], [0.0, 1.0], [ts[0]])
        with pytest.raises(ValueError, match="times must have the same length"):
            method(tle_ephemeris, [0.0, 1.0], [0.0, 1.0], [ts[0]])
        with pytest.raises(ValueError, match="one entry per ephemeris timestamp"):
            method(tle_ephemeris, [0.0], [0.0])
        with pytest.raises(ValueError, match="outside ephemeris range"):
            method(tle_ephemeris, [0.0], [0.0], [ts[-1] + timedelta(seconds=1)])
        with pytest.raises(ValueError, match="outside ephemeris range"):
            method(tle_ephemeris, [0.0], [0.0], [ts[0] - timedelta(seconds=1)])
        with pytest.raises(ValueError, match="target_rolls"):
            method(tle_ephemeris, [0.0], [0.0], [ts[0]], target_rolls=[1.0, 2.0])
        with pytest.raises(ValueError, match="n_roll_samples"):
            method(tle_ephemeris, [0.0], [0.0], [ts[0]], n_roll_samples=0)


@pytest.mark.parametrize("name", ["sun", "nested", "boresight_sun_ref", "eclipse"])
def test_empty_input(tle_ephemeris: TLEEphemeris, name: str) -> None:
    constraint = spacecraft_constraints()[name]
    mask = constraint.in_constraint_elementwise(tle_ephemeris, [], [], [])
    assert mask.shape == (0,) and mask.dtype == np.bool_
    empty = constraint.evaluate_elementwise(tle_ephemeris, [], [], [])
    assert isinstance(empty, ElementwiseResult)
    assert empty.constraint_array.shape == (0,)
    assert empty.timestamp.shape == (0,)
    populated = constraint.evaluate_elementwise(
        tle_ephemeris, [1.0], [2.0], [tle_ephemeris.timestamp[0]]
    )
    assert empty.constraint_values.keys() == populated.constraint_values.keys()
    assert all(v.shape == (0,) for v in empty.constraint_values.values())


def test_timezone_naive_and_aware_times_agree(tle_ephemeris: TLEEphemeris) -> None:
    constraint = SunConstraint(min_angle=45)
    aware = [tle_ephemeris.timestamp[4] + timedelta(seconds=30)]
    naive = [datetime(*aware[0].timetuple()[:6])]
    a = constraint.evaluate_elementwise(tle_ephemeris, [10.0], [20.0], aware)
    b = constraint.evaluate_elementwise(tle_ephemeris, [10.0], [20.0], naive)
    assert (
        a.constraint_values["sun_angle_deg"][0]
        == b.constraint_values["sun_angle_deg"][0]
    )
