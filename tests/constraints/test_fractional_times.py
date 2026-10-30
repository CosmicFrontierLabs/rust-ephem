"""Requested-time geometry must not be snapped to the ephemeris grid."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import rust_ephem as re
from rust_ephem.constraints import RustConstraintMixin

# datetime.UTC is only available on Python 3.11+, while CI supports 3.10.
BEGIN = datetime(2027, 12, 1, 19, tzinfo=timezone.utc)


@pytest.fixture
def orbit_file(tmp_path: Path) -> str:
    # Quadratic Cartesian motion has an exact Hermite solution, including velocity.
    path = tmp_path / "quadratic.txt"
    path.write_text(
        f"ScenarioEpoch {BEGIN.isoformat()}\nCoordinateSystem GCRS\n"
        "0 7000 0 0 0 7.5 0\n20 7004 150 0 0.4 7.5 0\n"
    )
    return str(path)


def ephemeris(
    path: str,
    begin: datetime = BEGIN,
    end: datetime = BEGIN + timedelta(seconds=20),
) -> re.FileEphemeris:
    return re.FileEphemeris(path, begin=begin, end=end, step_size=2)


@pytest.mark.parametrize("seconds", [0, 10, 10.25, 11.994764, 20])
def test_requested_geometry_matches_direct_sample(
    orbit_file: str, seconds: float
) -> None:
    time = BEGIN + timedelta(seconds=seconds)
    sampled = ephemeris(orbit_file)
    direct = ephemeris(orbit_file, time, time)
    constraint = re.EarthLimbConstraint(min_angle=5)
    actual = constraint.evaluate(sampled, 100, 0, times=time, target_roll=0)
    expected = constraint.evaluate(direct, 100, 0, indices=0, target_roll=0)
    assert actual.timestamps == [time]
    assert actual.constraint_array == expected.constraint_array
    for name, values in expected.constraint_values.items():
        np.testing.assert_allclose(actual.constraint_values[name], values, atol=1e-10)


@pytest.mark.parametrize("rolls", [None, [0, 27]])
def test_batch_preserves_requested_order_duplicates_and_exact_rows(
    orbit_file: str, rolls: list[float] | None
) -> None:
    sampled = ephemeris(orbit_file)
    times = [BEGIN + timedelta(seconds=s) for s in [11.994764, 0, 10.25, 11.994764, 20]]
    constraint = re.EarthLimbConstraint(min_angle=5).boresight_offset(pitch_deg=90)
    mask = constraint.in_constraint_batch(
        sampled, [10, 190], [0, 0], times=times, target_rolls=rolls, n_roll_samples=12
    )
    results = constraint.evaluate_batch(
        sampled, [10, 190], [0, 0], times=times, target_rolls=rolls, n_roll_samples=12
    )
    assert mask.shape == (2, len(times))
    for i, time in enumerate(times):
        direct = ephemeris(orbit_file, time, time)
        expected = constraint.in_constraint_batch(
            direct, [10, 190], [0, 0], indices=0, target_rolls=rolls, n_roll_samples=12
        )
        np.testing.assert_array_equal(mask[:, i], expected[:, 0])
    for row, result in enumerate(results):
        np.testing.assert_array_equal(result.timestamps, times)
        np.testing.assert_array_equal(result.constraint_array, mask[row])


def test_scalar_and_roll_range_use_fractional_time(orbit_file: str) -> None:
    time = BEGIN + timedelta(seconds=11.994764)
    sampled = ephemeris(orbit_file)
    direct = ephemeris(orbit_file, time, time)
    constraint = re.EarthLimbConstraint(min_angle=5).boresight_offset(pitch_deg=90)
    assert constraint.in_constraint(
        time=time, ephemeris=sampled, target_ra=10, target_dec=0
    ) == constraint.in_constraint(
        time=time, ephemeris=direct, target_ra=10, target_dec=0
    )
    assert constraint.roll_range(
        time=time, ephemeris=sampled, target_ra=10, target_dec=0, n_roll_samples=12
    ) == constraint.roll_range(
        time=time, ephemeris=direct, target_ra=10, target_dec=0, n_roll_samples=12
    )


@pytest.mark.parametrize("seconds", [-0.000001, 20.000001])
def test_no_extrapolation(orbit_file: str, seconds: float) -> None:
    with pytest.raises(ValueError, match="outside ephemeris range"):
        re.EarthLimbConstraint(min_angle=5).in_constraint(
            ephemeris=ephemeris(orbit_file),
            target_ra=100,
            target_dec=0,
            time=BEGIN + timedelta(seconds=seconds),
        )


def test_times_and_indices_remain_mutually_exclusive(orbit_file: str) -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        re.SunConstraint(min_angle=30).in_constraint_batch(
            ephemeris(orbit_file), [0], [0], times=BEGIN, indices=0
        )


def test_keepout_crossing_is_not_rounded_to_either_neighbor(orbit_file: str) -> None:
    sampled = ephemeris(orbit_file)
    times = [BEGIN + timedelta(seconds=s) for s in (10.1, 10.9)]
    # Independent analytic angle above the limb for r=(7000+.01*t², 7.5*t, 0).
    values = []
    for time in times:
        seconds = (time - BEGIN).total_seconds()
        r = np.array([7000.0 + 0.01 * seconds**2, 7.5 * seconds, 0.0])
        target = np.array([np.cos(np.deg2rad(100)), np.sin(np.deg2rad(100)), 0.0])
        center_angle = np.rad2deg(np.arccos(np.dot(-r, target) / np.linalg.norm(r)))
        values.append(
            center_angle - np.rad2deg(np.arcsin(6378.137 / np.linalg.norm(r)))
        )
    assert abs(values[0] - values[1]) > 0.01
    constraint = re.EarthLimbConstraint(min_angle=sum(values) / 2)
    result = constraint.in_constraint(
        ephemeris=sampled, target_ra=100, target_dec=0, time=times
    )
    assert isinstance(result, list)
    assert result == [value < sum(values) / 2 for value in values]
    assert result[0] != result[1]


@pytest.mark.parametrize(
    "constraint",
    [
        re.SunConstraint(min_angle=30),
        re.MoonConstraint(min_angle=30),
        re.EclipseConstraint(),
        re.AltAzConstraint(min_altitude=10),
    ],
)
@pytest.mark.parametrize("seconds", [10.25, 11.994764])
def test_other_geometry_at_requested_time(
    orbit_file: str, constraint: RustConstraintMixin, seconds: float
) -> None:
    time = BEGIN + timedelta(seconds=seconds)
    actual = constraint.evaluate(ephemeris(orbit_file), 100, 0, times=time)
    expected = constraint.evaluate(ephemeris(orbit_file, time, time), 100, 0)
    assert actual.constraint_array == expected.constraint_array
    for name, values in expected.constraint_values.items():
        np.testing.assert_allclose(
            actual.constraint_values[name], values, rtol=0, atol=1e-5
        )


def test_singleton_ephemeris_accepts_only_its_timestamp(orbit_file: str) -> None:
    sampled = ephemeris(orbit_file, BEGIN, BEGIN)
    constraint = re.SunConstraint(min_angle=30)
    assert isinstance(
        constraint.in_constraint(
            ephemeris=sampled, target_ra=0, target_dec=0, time=BEGIN
        ),
        bool,
    )
    with pytest.raises(ValueError, match="outside ephemeris range"):
        constraint.in_constraint(
            ephemeris=sampled,
            target_ra=0,
            target_dec=0,
            time=BEGIN + timedelta(microseconds=1),
        )
