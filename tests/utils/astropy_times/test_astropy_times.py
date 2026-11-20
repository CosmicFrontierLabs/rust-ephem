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


def _as_datetime64_does(when: Time) -> Times:
    """The instants of ``when`` as ``Time.datetime64`` reports them, rounded
    to the microsecond: the path the conversion used before."""
    nanos = np.asarray(when.utc.datetime64).astype("datetime64[ns]").astype("int64")
    micros: Times = np.floor_divide(nanos + 500, 1000).astype("datetime64[us]")
    return micros


def _long_ephemeris(begin: datetime, days: int) -> TLEEphemeris:
    return TLEEphemeris(TLE1, TLE2, begin, begin + timedelta(days=days), 600)


@pytest.mark.parametrize("built_from", ["mjd", "unix", "tt"])
def test_a_time_from_floats_gives_what_its_datetime64_does(
    ensure_planetary_data: None, built_from: str
) -> None:
    """The instants are computed from the ``Time``'s two doubles. They are
    the microseconds its ``datetime64`` gives, for every one of many times
    off the grid."""
    ephem = _long_ephemeris(datetime(2029, 1, 1, tzinfo=timezone.utc), 3)
    offsets = np.random.default_rng(11).uniform(0.0, 2.99, 20_000)
    if built_from == "unix":
        when = Time(1861920000.0 + offsets * 86400.0, format="unix")
    else:
        scale = "tt" if built_from == "tt" else "utc"
        when = Time(62137.01 + offsets, format="mjd", scale=scale)
    expected = ephem.gcrs_pv_at(_as_datetime64_does(when))
    # A new Time of the same instants: nothing is kept on it yet.
    result = ephem.gcrs_pv_at(when.copy())
    np.testing.assert_array_equal(result.position, expected.position)
    np.testing.assert_array_equal(result.velocity, expected.velocity)


def test_a_time_on_a_leap_second_day_gives_what_its_datetime64_does(
    ensure_planetary_data: None,
) -> None:
    """On 2016-12-31 the day is 86,401 s long and a ``Time``'s day fraction
    covers all of it."""
    ephem = _long_ephemeris(datetime(2016, 12, 30, tzinfo=timezone.utc), 4)
    # Up to half a second before the leap second, and the day after.
    fractions = np.random.default_rng(12).uniform(0.0, 86399.5 / 86401.0, 5_000)
    when = Time(
        np.concatenate([57753.0 + fractions, 57754.0 + fractions]), format="mjd"
    )
    expected = ephem.gcrs_pv_at(_as_datetime64_does(when))
    result = ephem.gcrs_pv_at(when.copy())
    np.testing.assert_array_equal(result.position, expected.position)


def test_a_time_before_1972_gives_what_its_datetime64_does(
    ensure_planetary_data: None,
) -> None:
    """Before 1972 UTC days had lengths of their own. Such a ``Time`` is
    still read from its ``datetime64``."""
    ephem = _long_ephemeris(datetime(1965, 6, 1, tzinfo=timezone.utc), 2)
    when = Time(
        38912.0 + np.random.default_rng(14).uniform(0.01, 1.99, 500), format="mjd"
    )
    expected = ephem.gcrs_pv_at(_as_datetime64_does(when))
    np.testing.assert_array_equal(
        ephem.gcrs_pv_at(when.copy()).position, expected.position
    )


def test_a_time_within_a_leap_second_is_refused(ensure_planetary_data: None) -> None:
    ephem = _long_ephemeris(datetime(2016, 12, 30, tzinfo=timezone.utc), 4)
    with pytest.raises(ValueError, match="leap second"):
        ephem.gcrs_pv_at(Time("2016-12-31T23:59:60.5", scale="utc"))


def test_a_time_keeps_its_shape(ephem: TLEEphemeris) -> None:
    times = instants(on_grid=False)
    when = Time(times.astype("datetime64[ns]"), scale="utc")
    grid = Time(
        np.stack([when.jd1, when.jd1]), np.stack([when.jd2, when.jd2]), format="jd"
    )
    assert grid.shape == (2, 5)
    both = ephem.gcrs_pv_at(grid)
    once = ephem.gcrs_pv_at(times)
    np.testing.assert_array_equal(both.position[:5], once.position)
    np.testing.assert_array_equal(both.position[5:], once.position)


def test_a_new_time_costs_little_more_than_datetime64(
    ensure_planetary_data: None,
) -> None:
    """``Time.datetime64`` is built from one string per element and then kept
    on the object, so only a ``Time`` that has not been used before shows its
    cost. Each timed call here is given a ``Time`` built for it."""
    import time

    ephem = _long_ephemeris(datetime(2029, 1, 1, tzinfo=timezone.utc), 3)
    rng = np.random.default_rng(13)

    def new_time() -> Time:
        return Time(62137.01 + rng.uniform(0.0, 2.99, 10_000), format="mjd")

    def seconds(times: object) -> float:
        began = time.perf_counter()
        ephem.gcrs_pv_at(times)
        return time.perf_counter() - began

    as_time = min(seconds(new_time()) for _ in range(5))
    as_datetime64 = min(seconds(_as_datetime64_does(new_time())) for _ in range(5))
    # Through the strings a new Time took about nine times the datetime64 call.
    assert as_time < 3.0 * as_datetime64 + 0.002
