"""Tests for the ``*_pv_at`` methods: ephemeris states at arbitrary times."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

import rust_ephem
from rust_ephem.constraints import SunConstraint

VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
STEP = 60
OFF_GRID = BEGIN + timedelta(seconds=30 * 60 + 30)  # midpoint between grid rows

PV_NAMES = ("gcrs", "itrs", "sun", "moon")


@pytest.fixture
def tle(ensure_planetary_data: Any) -> Any:
    return rust_ephem.TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN, END, STEP)


@pytest.fixture
def ground(ensure_planetary_data: Any) -> Any:
    return rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN,
        end=END,
        step_size=STEP,
    )


@pytest.fixture(params=["tle", "ground"])
def ephem(request: Any) -> Any:
    return request.getfixturevalue(request.param)


def grid_pv(eph: Any, name: str) -> Any:
    return getattr(eph, f"{name}_pv")


def pv_at(eph: Any, name: str, times: Any) -> Any:
    return getattr(eph, f"{name}_pv_at")(times)


@pytest.mark.parametrize(
    "cls",
    [
        rust_ephem.TLEEphemeris,
        rust_ephem.SPICEEphemeris,
        rust_ephem.GroundEphemeris,
        rust_ephem.OEMEphemeris,
        rust_ephem.FileEphemeris,
        rust_ephem.ParquetEphemeris,
    ],
)
def test_every_ephemeris_class_has_pv_at(cls: Any) -> None:
    for name in PV_NAMES:
        assert callable(getattr(cls, f"{name}_pv_at"))


@pytest.mark.parametrize("name", PV_NAMES)
def test_grid_times_reproduce_rows_exactly(ephem: Any, name: str) -> None:
    grid = grid_pv(ephem, name)
    stamps = list(ephem.timestamp.astype(datetime))
    stamps = [t.replace(tzinfo=timezone.utc) for t in stamps]
    got = pv_at(ephem, name, stamps)
    np.testing.assert_array_equal(got.position, grid.position)
    np.testing.assert_array_equal(got.velocity, grid.velocity)


@pytest.mark.parametrize("name", PV_NAMES)
def test_single_datetime_returns_one_row(ephem: Any, name: str) -> None:
    got = pv_at(ephem, name, BEGIN + timedelta(minutes=5))
    assert got.position.shape == (1, 3)
    assert got.velocity.shape == (1, 3)
    np.testing.assert_array_equal(got.position[0], grid_pv(ephem, name).position[5])


def test_midpoint_matches_finer_grid(tle: Any) -> None:
    fine = rust_ephem.TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN, END, 30)
    fine_idx = 61  # BEGIN + 30.5 min on the 30 s grid
    for name in PV_NAMES:
        got = pv_at(tle, name, OFF_GRID)
        truth = grid_pv(fine, name)
        dpos = np.linalg.norm(got.position[0] - truth.position[fine_idx])
        dvel = np.linalg.norm(got.velocity[0] - truth.velocity[fine_idx])
        assert dpos < 0.1, f"{name}: position error {dpos} km"
        assert dvel < 1e-3, f"{name}: velocity error {dvel} km/s"
    # Sanity: interpolated spacecraft state is not just a neighbouring row.
    grid = tle.gcrs_pv.position
    got = tle.gcrs_pv_at(OFF_GRID).position[0]
    assert np.linalg.norm(got - grid[30]) > 100.0
    assert np.linalg.norm(got - grid[31]) > 100.0


@pytest.mark.parametrize(
    "bad",
    [BEGIN - timedelta(seconds=1), END + timedelta(seconds=1)],
)
@pytest.mark.parametrize("name", PV_NAMES)
def test_out_of_range_raises(ephem: Any, name: str, bad: datetime) -> None:
    with pytest.raises(ValueError, match="outside ephemeris range"):
        pv_at(ephem, name, [BEGIN, bad])


def test_order_and_duplicates_preserved(ephem: Any) -> None:
    t_late = BEGIN + timedelta(minutes=40, seconds=10)
    times = [t_late, OFF_GRID, BEGIN, OFF_GRID, t_late]
    for name in PV_NAMES:
        got = pv_at(ephem, name, times)
        assert got.position.shape == (5, 3)
        for i, t in enumerate(times):
            single = pv_at(ephem, name, t)
            np.testing.assert_array_equal(got.position[i], single.position[0])
            np.testing.assert_array_equal(got.velocity[i], single.velocity[0])
        np.testing.assert_array_equal(got.position[2], grid_pv(ephem, name).position[0])


def test_accepts_generator_and_naive_utc(tle: Any) -> None:
    naive = OFF_GRID.replace(tzinfo=None)
    got = tle.gcrs_pv_at(t for t in [naive])
    np.testing.assert_array_equal(got.position, tle.gcrs_pv_at([OFF_GRID]).position)


def test_sun_constraint_agrees_with_pv_at(ephem: Any) -> None:
    ra, dec = 10.0, 20.0
    result = SunConstraint(min_angle=45.0).evaluate(
        ephemeris=ephem, target_ra=ra, target_dec=dec, times=[OFF_GRID]
    )
    angle = result.constraint_values["sun_angle_deg"][0]

    sun = ephem.sun_pv_at(OFF_GRID).position[0]
    obs = ephem.gcrs_pv_at(OFF_GRID).position[0]
    rel = sun - obs
    rel /= np.linalg.norm(rel)
    r, d = np.radians(ra), np.radians(dec)
    target = np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    expected = np.degrees(np.arccos(np.clip(rel @ target, -1.0, 1.0)))
    assert angle == pytest.approx(expected, abs=1e-9)
