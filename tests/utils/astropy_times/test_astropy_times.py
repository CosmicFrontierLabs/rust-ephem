"""An astropy ``Time`` wherever times are taken: it must give what the same
instants give as ``datetime64``."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from rust_ephem import EarthLimbConstraint, MoonConstraint, SunConstraint, TLEEphemeris
from rust_ephem.constraints import ConstraintConfig

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = 60
N_ROWS = 121

Times = npt.NDArray[np.datetime64]


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> TLEEphemeris:
    end = BEGIN + timedelta(seconds=STEP * (N_ROWS - 1))
    return TLEEphemeris(TLE1, TLE2, BEGIN, end, STEP)


def instants(on_grid: bool) -> Times:
    start = np.datetime64(BEGIN.replace(tzinfo=None), "us")
    if on_grid:
        micros = np.array([0, 7, 7, 120, 33]) * STEP * 1_000_000
    else:
        micros = np.array([1, 97_123_457, 3_600_000_000, 12_345_678, 7_199_999_999])
    times: Times = start + micros.astype("timedelta64[us]")
    return times


def constraint() -> ConstraintConfig:
    return (
        SunConstraint(min_angle=45)
        | EarthLimbConstraint(min_angle=15)
        | MoonConstraint(min_angle=20)
    )


def targets() -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(8)
    ras = rng.uniform(0.0, 360.0, 5)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 5)))
    return ras.tolist(), decs.tolist()


@pytest.mark.parametrize("on_grid", [True, False])
@pytest.mark.parametrize("scale", ["utc", "tt", "tai", "tdb"])
def test_ephemeris_states(ephem: TLEEphemeris, on_grid: bool, scale: str) -> None:
    """The instants are the same whatever scale the ``Time`` is held in."""
    times = instants(on_grid)
    when = getattr(Time(times.astype("datetime64[ns]"), scale="utc"), scale)
    for name in ("gcrs_pv_at", "itrs_pv_at", "sun_pv_at", "moon_pv_at"):
        expected = getattr(ephem, name)(times)
        result = getattr(ephem, name)(when)
        np.testing.assert_array_equal(result.position, expected.position)
        np.testing.assert_array_equal(result.velocity, expected.velocity)
    for apparent in (False, True):
        expected = ephem.get_body_pv("moon", times=times, apparent=apparent)
        result = ephem.get_body_pv("moon", times=when, apparent=apparent)
        np.testing.assert_array_equal(result.position, expected.position)
    ras, decs = targets()
    np.testing.assert_array_equal(
        ephem.apparent_target_at(ras, decs, when),
        ephem.apparent_target_at(ras, decs, times),
    )


@pytest.mark.parametrize("on_grid", [True, False])
@pytest.mark.parametrize("apparent", [False, True])
def test_constraint_calls(ephem: TLEEphemeris, on_grid: bool, apparent: bool) -> None:
    rule = constraint()
    times = instants(on_grid)
    when = Time(times.astype("datetime64[ns]"), scale="utc")
    ras, decs = targets()
    mask = rule.in_constraint_batch(ephem, ras, decs, times=times, apparent=apparent)
    np.testing.assert_array_equal(
        rule.in_constraint_batch(ephem, ras, decs, times=when, apparent=apparent), mask
    )
    np.testing.assert_array_equal(
        rule.in_constraint_elementwise(ephem, ras, decs, times=when, apparent=apparent),
        np.diagonal(mask),
    )
    paired = rule.evaluate_elementwise(ephem, ras, decs, times=when, apparent=apparent)
    expected = rule.evaluate_elementwise(
        ephem, ras, decs, times=times, apparent=apparent
    )
    for key, values in expected.constraint_values.items():
        np.testing.assert_array_equal(paired.constraint_values[key], values)
    grid = rule.evaluate_grid(ephem, ras, decs, times=when, apparent=apparent)
    np.testing.assert_array_equal(grid.constraint_array, mask)
    np.testing.assert_array_equal(
        rule.violated_count(ephem, ras, decs, times=when, apparent=apparent),
        mask.sum(axis=1),
    )
    batch = rule.evaluate_batch(ephem, ras, decs, times=when, apparent=apparent)
    same = rule.evaluate_batch(ephem, ras, decs, times=times, apparent=apparent)
    for result, wanted in zip(batch, same, strict=True):
        np.testing.assert_array_equal(result.constraint_array, wanted.constraint_array)
        np.testing.assert_array_equal(result.timestamps, wanted.timestamps)
    one = rule.evaluate(ephem, ras[0], decs[0], times=when, apparent=apparent)
    np.testing.assert_array_equal(one.constraint_array, same[0].constraint_array)


def test_a_scalar_time_is_one_instant(ephem: TLEEphemeris) -> None:
    rule = constraint()
    times = instants(on_grid=False)
    when = Time(times.astype("datetime64[ns]"), scale="utc")
    ras, decs = targets()
    mask = rule.in_constraint_batch(ephem, ras, decs, times=times)
    assert rule.in_constraint(when[2], ephem, ras[1], decs[1]) is bool(mask[1, 2])
    assert rule.in_constraint(when, ephem, ras[1], decs[1]) == mask[1].tolist()
    np.testing.assert_array_equal(
        ephem.gcrs_pv_at(when[2]).position, ephem.gcrs_pv_at(times[2:3]).position
    )
    assert rule.in_constraint_batch(ephem, ras, decs, times=when[2]).shape == (5, 1)


def test_times_built_from_floats_land_on_the_grid(ephem: TLEEphemeris) -> None:
    """A ``Time`` holds two doubles; the nanoseconds it reports for a time
    built from an MJD or by adding seconds are a rounding error away from the
    intended instant. Rounded to the microsecond they are the grid rows."""
    rows = np.array([0, 1, 59, 60, 120])
    expected = ephem.gcrs_pv_at(
        [BEGIN + timedelta(seconds=int(row) * STEP) for row in rows]
    )
    mjd = Time(60310.0 + rows * STEP / 86400.0, format="mjd", scale="utc")
    np.testing.assert_array_equal(ephem.gcrs_pv_at(mjd).position, expected.position)
    unix = Time(1704067200.0 + rows * float(STEP), format="unix")
    np.testing.assert_array_equal(ephem.gcrs_pv_at(unix).position, expected.position)


def test_times_outside_the_ephemeris_are_rejected(ephem: TLEEphemeris) -> None:
    late = Time("2024-01-02T00:00:00", scale="utc")
    with pytest.raises(ValueError, match="outside ephemeris range"):
        ephem.gcrs_pv_at(late)
    with pytest.raises(ValueError, match="outside ephemeris range"):
        constraint().in_constraint_batch(ephem, [0.0], [0.0], times=late)
