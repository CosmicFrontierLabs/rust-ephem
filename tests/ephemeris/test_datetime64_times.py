"""NumPy ``datetime64`` is accepted wherever a time or sequence of times is."""

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

import rust_ephem
from rust_ephem.constraints import EarthLimbConstraint, SunConstraint

VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
STEP = 60

# On-grid and off-grid, unsorted, with a repeat.
OFFSETS_S = [90.5, 0.0, 1800.0, 90.5, 3599.25]
AS_DATETIME = [BEGIN + timedelta(seconds=s) for s in OFFSETS_S]
AS_DATETIME64 = np.array(
    [t.replace(tzinfo=None).isoformat() for t in AS_DATETIME], dtype="datetime64[us]"
)


@pytest.fixture
def ephem(ensure_planetary_data: Any) -> Any:
    return rust_ephem.TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN, END, STEP)


@pytest.fixture
def constraint() -> Any:
    return SunConstraint(min_angle=45) | EarthLimbConstraint(min_angle=20)


@pytest.mark.parametrize("name", ["gcrs", "itrs", "sun", "moon"])
@pytest.mark.parametrize("unit", ["us", "ns", "ms"])
def test_pv_at_accepts_datetime64_arrays(ephem: Any, name: str, unit: str) -> None:
    pv_at = getattr(ephem, f"{name}_pv_at")
    expected = pv_at(AS_DATETIME)
    times = AS_DATETIME64.astype(f"datetime64[{unit}]")
    if unit == "ms":
        # Millisecond resolution cannot hold the quarter-second offsets exactly.
        expected = pv_at([t.replace(microsecond=250000) for t in AS_DATETIME[-1:]])
        times = times[-1:]
    result = pv_at(times)
    np.testing.assert_array_equal(result.position, expected.position)
    np.testing.assert_array_equal(result.velocity, expected.velocity)


def test_pv_at_accepts_a_datetime64_scalar_and_a_list_of_them(ephem: Any) -> None:
    expected = ephem.gcrs_pv_at(AS_DATETIME)
    scalar = ephem.gcrs_pv_at(AS_DATETIME64[0])
    np.testing.assert_array_equal(scalar.position, expected.position[:1])
    listed = ephem.gcrs_pv_at(list(AS_DATETIME64))
    np.testing.assert_array_equal(listed.position, expected.position)


def test_pv_at_rejects_nat(ephem: Any) -> None:
    times = AS_DATETIME64.copy()
    times[2] = np.datetime64("NaT")
    with pytest.raises(ValueError, match=r"times\[2\] is NaT"):
        ephem.gcrs_pv_at(times)


@pytest.mark.parametrize("position", [0, 1, 4])
def test_nat_in_a_list_of_scalars_is_reported_at_its_own_index(
    ephem: Any, position: int
) -> None:
    times = list(AS_DATETIME64)
    times[position] = np.datetime64("NaT")
    with pytest.raises(ValueError, match=rf"times\[{position}\] is NaT"):
        ephem.gcrs_pv_at(times)


def test_nat_among_datetimes_is_reported_at_its_own_index(ephem: Any) -> None:
    times: list[Any] = list(AS_DATETIME)
    times[3] = np.datetime64("NaT")
    with pytest.raises(ValueError, match=r"times\[3\] is NaT"):
        ephem.gcrs_pv_at(times)


def test_pv_at_still_rejects_non_times(ephem: Any) -> None:
    with pytest.raises(AttributeError):
        ephem.gcrs_pv_at(["2024-01-01T00:00:00"])


def test_in_constraint_batch_accepts_datetime64(ephem: Any, constraint: Any) -> None:
    ras, decs = [10.0, 120.0, 250.0], [5.0, -40.0, 60.0]
    expected = constraint.in_constraint_batch(ephem, ras, decs, times=AS_DATETIME)
    result = constraint.in_constraint_batch(ephem, ras, decs, times=AS_DATETIME64)
    np.testing.assert_array_equal(result, expected)
    one = constraint.in_constraint_batch(ephem, ras, decs, times=AS_DATETIME64[0])
    np.testing.assert_array_equal(one, expected[:, :1])


def test_evaluate_and_evaluate_batch_accept_datetime64(
    ephem: Any, constraint: Any
) -> None:
    # evaluate() reports windows, so use sorted times.
    times64 = np.unique(AS_DATETIME64)
    times = sorted(set(AS_DATETIME))
    expected = constraint.evaluate(ephem, 120.0, -40.0, times=times)
    result = constraint.evaluate(ephem, 120.0, -40.0, times=times64)
    assert list(result.constraint_array) == list(expected.constraint_array)
    assert list(result.timestamps) == list(expected.timestamps)
    (batch,) = constraint.evaluate_batch(ephem, [120.0], [-40.0], times=times64)
    assert list(batch.constraint_array) == list(expected.constraint_array)


def test_in_constraint_accepts_datetime64(ephem: Any, constraint: Any) -> None:
    expected = constraint.in_constraint(AS_DATETIME, ephem, 120.0, -40.0)
    assert constraint.in_constraint(AS_DATETIME64, ephem, 120.0, -40.0) == expected
    assert (
        constraint.in_constraint(AS_DATETIME64[0], ephem, 120.0, -40.0) == expected[0]
    )


def test_elementwise_accepts_datetime64_from_the_rust_api(
    ephem: Any, constraint: Any
) -> None:
    """The low-level Constraint takes datetime64 directly, not only the
    Python wrapper that used to convert it."""
    rust = constraint._get_cached_rust_constraint()
    ras, decs = [10.0] * len(OFFSETS_S), [5.0] * len(OFFSETS_S)
    expected = rust.in_constraint_elementwise(ephem, ras, decs, AS_DATETIME)
    result = rust.in_constraint_elementwise(ephem, ras, decs, AS_DATETIME64)
    np.testing.assert_array_equal(result, expected)
