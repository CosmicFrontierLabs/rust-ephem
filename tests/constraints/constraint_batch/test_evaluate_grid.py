"""``evaluate_grid``: the target x time grid as arrays, against the other
evaluation methods on the same cells."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import rust_ephem as re

BEGIN = datetime(2026, 3, 20, tzinfo=timezone.utc)
STEP = 600
N = 60


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> Any:
    """A distant observer moving at a few km/s, so apparent angles differ."""
    seconds = np.arange(N + 1) * float(STEP)
    start, velocity = (
        np.array([180000.0, -150000.0, 90000.0]),
        np.array([1.4, 2.2, -1.5]),
    )
    positions = start + seconds[:, None] * velocity
    times = [BEGIN + timedelta(seconds=float(s)) for s in seconds]
    return re.FileEphemeris.from_arrays(
        times, positions, np.tile(velocity, (N + 1, 1)), step_size=STEP
    )


def targets(
    count: int, seed: int = 4
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return (
        np.asarray(rng.uniform(0.0, 360.0, count), dtype=np.float64),
        np.asarray(
            np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count))), dtype=np.float64
        ),
    )


def trees() -> dict[str, Any]:
    sun = re.SunConstraint(min_angle=60, max_angle=150)
    limb = re.EarthLimbConstraint(min_angle=20)
    moon = re.MoonLimbConstraint(min_angle=40)
    square = [(-25.0, -25.0), (25.0, -25.0), (25.0, 25.0), (-25.0, 25.0)]
    stars = [
        (float(ra), float(dec)) for ra in range(0, 360, 30) for dec in (-40, 0, 40)
    ]
    return {
        # Kernels that correct targets themselves: one grid call when apparent.
        "rules": sun | limb | moon,
        # A rule that does not: corrected per column or per target.
        "body": re.BodyConstraint(body="Jupiter barycenter", min_angle=70) | sun,
        "not_and": ~sun & limb,
        # Roll-dependent trees.
        "boresight": (sun | moon).boresight_offset(pitch_deg=20, yaw_deg=10),
        "solar_roll": re.SolarRollConstraint(tolerance_deg=40) | sun,
        "polygon": re.BrightStarConstraint(stars=stars, fov_polygon=square) | sun,
    }


TREES = trees()
# (n_targets, n_times): wider than long, and longer than wide.
SHAPES = {"wide": (14, 3), "long": (4, 11)}


def rolls_for(kind: str, count: int) -> list[float | None] | None:
    if kind == "free":
        return None
    if kind == "fixed":
        return [35.0] * count
    return [None if i % 3 == 0 else 20.0 * (i % 4) for i in range(count)]


def assert_same(a: Any, b: Any, what: str) -> None:
    np.testing.assert_allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0,
        atol=1e-9,
        equal_nan=True,
        err_msg=what,
    )


@pytest.mark.parametrize("apparent", [False, True], ids=["geometric", "apparent"])
@pytest.mark.parametrize("roll_kind", ["free", "fixed", "mixed"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("tree", list(TREES))
def test_grid_matches_the_other_methods(
    ephem: Any, tree: str, shape: str, roll_kind: str, apparent: bool
) -> None:
    constraint = TREES[tree]
    n_targets, n_times = SHAPES[shape]
    ras, decs = targets(n_targets)
    times = [ephem.timestamp[i] for i in np.linspace(0, N, n_times).astype(int)]
    rolls = rolls_for(roll_kind, n_targets)
    kwargs: dict[str, Any] = {
        "times": times,
        "n_roll_samples": 24,
        "apparent": apparent,
    }

    grid = constraint.evaluate_grid(ephem, ras, decs, target_rolls=rolls, **kwargs)
    assert grid.shape == (n_targets, n_times)
    mask = np.asarray(grid.constraint_array)
    assert mask.shape == (n_targets, n_times) and mask.dtype == np.bool_
    np.testing.assert_array_equal(grid.visibility_flags, ~mask)

    # The mask alone.
    np.testing.assert_array_equal(
        mask,
        constraint.in_constraint_batch(
            ephem, list(ras), list(decs), target_rolls=rolls, **kwargs
        ),
    )

    # One result per target.
    batch = constraint.evaluate_batch(
        ephem, list(ras), list(decs), target_rolls=rolls, **kwargs
    )
    values, causes = grid.constraint_values, grid.component_violated
    for i, result in enumerate(batch):
        np.testing.assert_array_equal(mask[i], result.constraint_array)
        assert set(result.constraint_values) == set(values)
        for key, row in result.constraint_values.items():
            assert_same(values[key][i], row, f"{key} target {i}")
        assert result.cause_value_keys == grid.cause_value_keys

    # The same cells, paired.
    flat_rolls = (
        None
        if rolls is None
        else list(np.repeat(np.array(rolls, dtype=object), n_times))
    )
    paired = constraint.evaluate_elementwise(
        ephem,
        np.repeat(ras, n_times),
        np.repeat(decs, n_times),
        times=times * n_targets,
        target_rolls=flat_rolls,
        n_roll_samples=24,
        apparent=apparent,
    )
    np.testing.assert_array_equal(mask.ravel(), paired.constraint_array)
    assert set(paired.constraint_values) == set(values)
    for key, flat in paired.constraint_values.items():
        assert_same(values[key].ravel(), flat, key)
    assert set(paired.component_violated) == set(causes)
    for key, flat in paired.component_violated.items():
        np.testing.assert_array_equal(causes[key].ravel(), flat, err_msg=key)


@pytest.mark.parametrize("apparent", [False, True], ids=["geometric", "apparent"])
def test_indices_and_off_grid_times(ephem: Any, apparent: bool) -> None:
    constraint = TREES["rules"]
    ras, decs = targets(6)
    rows = [0, 7, 30, N]
    by_index = constraint.evaluate_grid(
        ephem, ras, decs, indices=rows, apparent=apparent
    )
    by_time = constraint.evaluate_grid(
        ephem, ras, decs, times=[ephem.timestamp[i] for i in rows], apparent=apparent
    )
    np.testing.assert_array_equal(by_index.constraint_array, by_time.constraint_array)
    for key, array in by_index.constraint_values.items():
        assert_same(array, by_time.constraint_values[key], key)
    assert list(by_index.timestamp) == [ephem.timestamp[i] for i in rows]

    # No times: the whole ephemeris.
    everything = constraint.evaluate_grid(ephem, ras, decs, apparent=apparent)
    assert everything.shape == (6, N + 1)
    np.testing.assert_array_equal(
        everything.constraint_array[:, rows], by_index.constraint_array
    )

    # Between samples, as a datetime64 array.
    middle = np.array(
        [np.datetime64(ephem.timestamp[i].replace(tzinfo=None)) for i in (3, 20)]
    ) + np.timedelta64(STEP // 2, "s")
    off_grid = constraint.evaluate_grid(
        ephem, ras, decs, times=middle, apparent=apparent
    )
    paired = constraint.evaluate_elementwise(
        ephem,
        np.repeat(ras, 2),
        np.repeat(decs, 2),
        times=np.tile(middle, 6),
        apparent=apparent,
    )
    for key, flat in paired.constraint_values.items():
        assert_same(off_grid.constraint_values[key].ravel(), flat, key)


def test_apparent_differs_from_geometric_by_arcseconds(ephem: Any) -> None:
    constraint = TREES["rules"]
    ras, decs = targets(40)
    geometric = constraint.evaluate_grid(ephem, ras, decs, indices=[0, N])
    apparent = constraint.evaluate_grid(ephem, ras, decs, indices=[0, N], apparent=True)
    difference = (
        np.abs(
            apparent.constraint_values["sun.sun_angle_deg"]
            - geometric.constraint_values["sun.sun_angle_deg"]
        ).max()
        * 3600
    )
    assert 5 < difference < 45


class TestGridResult:
    @pytest.fixture
    def grid(self, ephem: Any) -> Any:
        ras, decs = targets(5)
        return TREES["rules"].evaluate_grid(
            ephem, ras, decs, indices=list(range(0, N, 4))
        )

    def test_axes(self, grid: Any) -> None:
        ras, decs = targets(5)
        np.testing.assert_array_equal(grid.ras, ras)
        np.testing.assert_array_equal(grid.decs, decs)
        assert len(grid.timestamp) == grid.shape[1] == 15
        assert grid.constraint_name.startswith("OR(")
        assert set(grid.cause_value_keys) == {"sun", "earth_limb", "moon"}
        assert set(grid.component_violated) == {"sun", "earth_limb", "moon"}
        assert "n_targets=5, n_times=15" in repr(grid)

    def test_arrays_are_built_once_and_read_only(self, grid: Any) -> None:
        assert grid.constraint_array is grid.constraint_array
        values = grid.constraint_values
        again = grid.constraint_values
        assert values is not again  # a fresh dict ...
        for key, array in values.items():
            assert again[key] is array  # ... over the same arrays
            assert array.dtype == np.float64 and array.shape == grid.shape
            with pytest.raises(ValueError, match="read-only"):
                array[0, 0] = 0.0
        values.clear()
        assert grid.constraint_values
        with pytest.raises(ValueError, match="read-only"):
            grid.constraint_array[0, 0] = True

    def test_target_gives_one_result_on_demand(self, grid: Any, ephem: Any) -> None:
        ras, decs = targets(5)
        batch = TREES["rules"].evaluate_batch(
            ephem, list(ras), list(decs), indices=list(range(0, N, 4))
        )
        for i in (0, 3, -1):
            one, expected = grid.target(i), batch[i]
            assert list(one.constraint_array) == list(expected.constraint_array)
            assert one.all_satisfied == expected.all_satisfied
            assert [(v.start_time, v.end_time) for v in one.violations] == [
                (v.start_time, v.end_time) for v in expected.violations
            ]
            assert [
                (w.start_time, w.end_time, w.start_cause) for w in one.visibility
            ] == [
                (w.start_time, w.end_time, w.start_cause) for w in expected.visibility
            ]
        with pytest.raises(IndexError):
            grid.target(5)

    def test_empty_grids(self, ephem: Any) -> None:
        for apparent in (False, True):
            none = TREES["rules"].evaluate_grid(
                ephem, [], [], indices=[0, 1], apparent=apparent
            )
            assert none.shape == (0, 2)
            assert none.constraint_values["sun.sun_angle_deg"].shape == (0, 2)

    def test_argument_checks(self, ephem: Any) -> None:
        constraint = TREES["rules"]
        with pytest.raises(ValueError, match="same length"):
            constraint.evaluate_grid(ephem, [1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="target_rolls must have the same length"):
            constraint.evaluate_grid(ephem, [1.0, 2.0], [1.0, 2.0], target_rolls=[1.0])
        with pytest.raises(ValueError, match="n_roll_samples"):
            constraint.evaluate_grid(ephem, [1.0], [1.0], n_roll_samples=0)
        with pytest.raises(ValueError, match="1-D sequence of numbers"):
            constraint._get_cached_rust_constraint().evaluate_grid(
                ephem, [[1.0, 2.0]], [[1.0, 2.0]]
            )


class TestCheaperResults:
    def test_constraint_values_are_cached_read_only_arrays(self, ephem: Any) -> None:
        result = TREES["rules"].evaluate(ephem, 200.0, 30.0)
        values = result.constraint_values
        array = values["sun.sun_angle_deg"]
        assert isinstance(array, np.ndarray) and array.dtype == np.float64
        assert array.shape == (N + 1,)
        assert result.constraint_values["sun.sun_angle_deg"] is array
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0
        values.pop("sun.sun_angle_deg")
        assert "sun.sun_angle_deg" in result.constraint_values

    def test_elementwise_takes_arrays_lists_and_datetime64(self, ephem: Any) -> None:
        constraint = TREES["rules"]
        ras, decs = targets(8)
        times = [ephem.timestamp[i] for i in range(8)]
        as_datetime64 = np.array([np.datetime64(t.replace(tzinfo=None)) for t in times])
        expected = constraint.evaluate_elementwise(
            ephem, list(ras), list(decs), times=times
        )
        for kwargs in (
            {"target_ras": ras, "target_decs": decs, "times": as_datetime64},
            {
                "target_ras": tuple(ras),
                "target_decs": tuple(decs),
                "times": tuple(times),
            },
            {
                # Non-contiguous views are read correctly too.
                "target_ras": np.repeat(ras, 2)[::2],
                "target_decs": np.asarray(decs)[np.arange(8)],
                "times": as_datetime64.astype("datetime64[ns]"),
            },
        ):
            result = constraint.evaluate_elementwise(ephem, **kwargs)
            assert_same(
                result.constraint_values["sun.sun_angle_deg"],
                expected.constraint_values["sun.sun_angle_deg"],
                "sun angle",
            )
            np.testing.assert_array_equal(
                constraint.in_constraint_elementwise(ephem, **kwargs),
                expected.constraint_array,
            )
