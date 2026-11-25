"""Tests for FileEphemeris.from_arrays (in-memory state vectors)."""

import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import Ephemeris, FileEphemeris, SunConstraint

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
STEP = 60

Arrays = tuple[list[datetime], npt.NDArray[np.float64], npt.NDArray[np.float64]]


@pytest.fixture
def file_eph(seconds_file: str) -> FileEphemeris:
    return FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)


@pytest.fixture
def arrays(file_eph: FileEphemeris) -> Arrays:
    """The raw samples of the seconds_file fixture, as in-memory arrays."""
    return (
        list(file_eph.file_timestamp),
        np.asarray(file_eph.file_pv.position),
        np.asarray(file_eph.file_pv.velocity),
    )


class TestMatchesFileConstructor:
    def test_same_grid_and_states(
        self, file_eph: FileEphemeris, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r, v, BEGIN, END, STEP)
        assert list(eph.timestamp) == list(file_eph.timestamp)
        np.testing.assert_array_equal(eph.gcrs_pv.position, file_eph.gcrs_pv.position)
        np.testing.assert_array_equal(eph.gcrs_pv.velocity, file_eph.gcrs_pv.velocity)
        np.testing.assert_array_equal(eph.itrs_pv.position, file_eph.itrs_pv.position)
        np.testing.assert_array_equal(eph.sun_pv.position, file_eph.sun_pv.position)
        np.testing.assert_array_equal(eph.moon_pv.position, file_eph.moon_pv.position)

    def test_raw_data_round_trips(
        self, file_eph: FileEphemeris, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r, v, BEGIN, END, STEP)
        assert eph.file_timestamp == file_eph.file_timestamp
        np.testing.assert_array_equal(eph.file_pv.position, r)
        np.testing.assert_array_equal(eph.file_pv.velocity, v)

    def test_datetime64_times(self, file_eph: FileEphemeris, arrays: Arrays) -> None:
        t, r, v = arrays
        t64 = np.array([x.replace(tzinfo=None) for x in t], dtype="datetime64[ns]")
        eph = FileEphemeris.from_arrays(t64, r, v, BEGIN, END, STEP)
        np.testing.assert_array_equal(eph.gcrs_pv.position, file_eph.gcrs_pv.position)

    def test_naive_datetimes_are_utc(
        self, file_eph: FileEphemeris, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        naive = [x.replace(tzinfo=None) for x in t]
        eph = FileEphemeris.from_arrays(
            naive, r, v, BEGIN.replace(tzinfo=None), END.replace(tzinfo=None), STEP
        )
        np.testing.assert_array_equal(eph.gcrs_pv.position, file_eph.gcrs_pv.position)

    def test_lists_accepted(self, file_eph: FileEphemeris, arrays: Arrays) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r.tolist(), v.tolist(), BEGIN, END, STEP)
        np.testing.assert_array_equal(eph.gcrs_pv.position, file_eph.gcrs_pv.position)


class TestDefaultsAndMetadata:
    def test_default_begin_end(self, arrays: Arrays) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r, v, step_size=600)
        assert eph.begin == t[0]
        assert eph.end == t[-1]
        assert len(eph.timestamp) == len(t)
        np.testing.assert_allclose(eph.gcrs_pv.position, r, rtol=0, atol=1e-9)

    def test_metadata(self, arrays: Arrays) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r, v, BEGIN, END, STEP)
        assert eph.file_path is None
        assert eph.source_frame == "J2000"
        assert eph.source_position_unit == "km"
        assert eph.source_velocity_unit == "km/s"
        assert eph.polar_motion is False
        assert eph.step_size == STEP
        assert isinstance(eph, Ephemeris)

    def test_velocity_unit_follows_position_unit(
        self, file_eph: FileEphemeris, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(
            t, r * 1000.0, v * 1000.0, BEGIN, END, STEP, position_unit="m"
        )
        assert eph.source_velocity_unit == "m/s"
        np.testing.assert_allclose(eph.file_pv.velocity, v, rtol=1e-12)
        np.testing.assert_allclose(
            eph.gcrs_pv.position, file_eph.gcrs_pv.position, rtol=1e-12
        )
        assert FileEphemeris.from_arrays(t, r, v).source_velocity_unit == "km/s"

    def test_metre_units(self, file_eph: FileEphemeris, arrays: Arrays) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(
            t,
            r * 1000.0,
            v * 1000.0,
            BEGIN,
            END,
            STEP,
            position_unit="m",
            velocity_unit="m/s",
        )
        assert eph.source_position_unit == "m"
        np.testing.assert_allclose(eph.file_pv.position, r, rtol=1e-12)
        np.testing.assert_allclose(
            eph.gcrs_pv.position, file_eph.gcrs_pv.position, rtol=1e-12
        )
        np.testing.assert_allclose(
            eph.gcrs_pv.velocity, file_eph.gcrs_pv.velocity, rtol=1e-12
        )

    def test_earth_fixed_frame(self, ecef_file: str, arrays: Arrays) -> None:
        t, r, v = arrays
        ref = FileEphemeris(ecef_file, begin=BEGIN, end=END, step_size=STEP)
        eph = FileEphemeris.from_arrays(t, r, v, BEGIN, END, STEP, frame="ECEF")
        np.testing.assert_array_equal(eph.itrs_pv.position, ref.itrs_pv.position)
        np.testing.assert_array_equal(eph.gcrs_pv.position, ref.gcrs_pv.position)


class TestUnitMismatchWarning:
    """Positions and velocities a unit factor apart are reported."""

    def test_metres_with_velocities_left_in_km_s(self, arrays: Arrays) -> None:
        # The reviewer's case: only position_unit="m" is given, so the
        # velocities are read as m/s although they are in km/s.
        t, r, v = arrays
        with pytest.warns(UserWarning, match="1000 times too small") as caught:
            eph = FileEphemeris.from_arrays(
                t, r * 1000.0, v, BEGIN, END, STEP, position_unit="m"
            )
        assert eph.source_velocity_unit == "m/s"
        assert "velocity_unit='m/s'" in str(caught[0].message)

    def test_kilometres_with_velocities_in_m_s(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.warns(UserWarning, match="1000 times too large"):
            FileEphemeris.from_arrays(t, r, v * 1000.0, BEGIN, END, STEP)

    @pytest.mark.parametrize(
        ("position_unit", "velocity_unit", "factor"),
        [("m", "cm/s", 100), ("cm", "km/s", 100000)],
    )
    def test_other_unit_factors(
        self, arrays: Arrays, position_unit: str, velocity_unit: str, factor: int
    ) -> None:
        # Data in km and km/s, declared in two other units.
        t, r, v = arrays
        with pytest.warns(UserWarning, match=f"{factor} times too"):
            FileEphemeris.from_arrays(
                t,
                r,
                v,
                BEGIN,
                END,
                STEP,
                position_unit=position_unit,
                velocity_unit=velocity_unit,
            )

    def test_file_constructor_warns_too(self, meters_file: str) -> None:
        # Metres read as km: the positions are 1000 times too large.
        with pytest.warns(UserWarning, match="1000 times too small"):
            FileEphemeris(
                meters_file,
                begin=BEGIN,
                end=END,
                step_size=STEP,
                position_unit="km",
                velocity_unit="m/s",
            )

    @pytest.mark.parametrize("scale", [1.0, 1000.0])
    def test_consistent_units_do_not_warn(self, arrays: Arrays, scale: float) -> None:
        t, r, v = arrays
        unit = "km" if scale == 1.0 else "m"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FileEphemeris.from_arrays(
                t, r * scale, v * scale, BEGIN, END, STEP, position_unit=unit
            )

    def test_zero_velocities_do_not_warn(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FileEphemeris.from_arrays(t, r, np.zeros_like(v), BEGIN, END, STEP)

    def test_a_warning_made_an_error_does_not_stop_construction(
        self, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eph = FileEphemeris.from_arrays(t, r, v * 1000.0, BEGIN, END, STEP)
        assert eph.source_velocity_unit == "km/s"


class TestValidation:
    def test_positions_wrong_shape(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match=r"positions must have shape \(N, 3\)"):
            FileEphemeris.from_arrays(t, r[:, :2], v)

    def test_velocities_wrong_shape(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match=r"velocities must have shape \(N, 3\)"):
            FileEphemeris.from_arrays(t, r, v.ravel())

    def test_length_mismatch(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match="Length mismatch"):
            FileEphemeris.from_arrays(t[:-1], r, v)
        with pytest.raises(ValueError, match="Length mismatch"):
            FileEphemeris.from_arrays(t, r, v[:-1])

    def test_non_monotonic_times(self, arrays: Arrays) -> None:
        t, r, v = arrays
        swapped = [t[0], t[2], t[1], *t[3:]]
        with pytest.raises(ValueError, match="strictly increasing"):
            FileEphemeris.from_arrays(swapped, r, v)

    def test_duplicate_times(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match="strictly increasing"):
            FileEphemeris.from_arrays([t[0], *t[:-1]], r, v)

    def test_too_few_samples(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match="At least 2"):
            FileEphemeris.from_arrays(t[:1], r[:1], v[:1])

    def test_non_finite(self, arrays: Arrays) -> None:
        t, r, v = arrays
        r = r.copy()
        r[3, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            FileEphemeris.from_arrays(t, r, v)

    def test_range_outside_data(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match="exceeds input data range"):
            FileEphemeris.from_arrays(t, r, v, end=t[-1] + timedelta(seconds=1))
        with pytest.raises(ValueError, match="exceeds input data range"):
            FileEphemeris.from_arrays(t, r, v, begin=t[0] - timedelta(seconds=1))

    def test_bad_times_type(self, arrays: Arrays) -> None:
        _, r, v = arrays
        with pytest.raises(TypeError):
            FileEphemeris.from_arrays(list(range(len(r))), r, v)  # type: ignore[arg-type]

    def test_unknown_unit_and_frame(self, arrays: Arrays) -> None:
        t, r, v = arrays
        with pytest.raises(ValueError, match="Unknown position unit"):
            FileEphemeris.from_arrays(t, r, v, position_unit="furlongs")
        with pytest.raises(ValueError, match="Unsupported coordinate frame"):
            FileEphemeris.from_arrays(t, r, v, frame="BADFRAME")


class TestConstraint:
    def test_sun_constraint_matches_file(
        self, file_eph: FileEphemeris, arrays: Arrays
    ) -> None:
        t, r, v = arrays
        eph = FileEphemeris.from_arrays(t, r, v, BEGIN, END, STEP)
        c = SunConstraint(min_angle=45)
        res = c.evaluate(eph, 10.0, 20.0)
        ref = c.evaluate(file_eph, 10.0, 20.0)
        assert len(res.timestamps) == len(eph.timestamp)
        assert list(res.constraint_array) == list(ref.constraint_array)


def circular_orbit(
    radius_km: float, step: float, count: int
) -> tuple[list[datetime], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rate = np.sqrt(398600.4418 / radius_km**3)
    phase = rate * np.arange(count) * step
    position = radius_km * np.stack(
        [np.cos(phase), 0.6 * np.sin(phase), 0.8 * np.sin(phase)], axis=1
    )
    velocity = (
        radius_km
        * rate
        * np.stack([-np.sin(phase), 0.6 * np.cos(phase), 0.8 * np.cos(phase)], axis=1)
    )
    times = [BEGIN + timedelta(seconds=float(step * i)) for i in range(count)]
    return times, position, velocity


def not_a_knot_slopes(
    seconds: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Slopes at the samples of the not-a-knot cubic spline through one
    coordinate, from its defining conditions as a dense linear system in the
    coefficients of every piece."""
    pieces = len(seconds) - 1
    h = np.diff(seconds)
    system = np.zeros((4 * pieces, 4 * pieces))
    right = np.zeros(4 * pieces)
    row = 0
    for k in range(pieces):  # each piece passes through its two samples
        for x, value in ((0.0, values[k]), (h[k], values[k + 1])):
            system[row, 4 * k : 4 * k + 4] = [1.0, x, x**2, x**3]
            right[row] = value
            row += 1
    for k in range(pieces - 1):  # first and second derivatives are continuous
        x = h[k]
        system[row, 4 * k : 4 * k + 4] = [0.0, 1.0, 2 * x, 3 * x**2]
        system[row, 4 * k + 5] = -1.0
        row += 1
        system[row, 4 * k : 4 * k + 4] = [0.0, 0.0, 2.0, 6 * x]
        system[row, 4 * k + 6] = -2.0
        row += 1
    for k in (0, pieces - 2):  # the third is too, across the end knots
        system[row, 4 * k + 3] = 1.0
        system[row, 4 * k + 7] = -1.0
        row += 1
    coefficients = np.linalg.solve(system, right).reshape(pieces, 4)
    last = coefficients[-1]
    end = last[1] + 2 * last[2] * h[-1] + 3 * last[3] * h[-1] ** 2
    slopes: npt.NDArray[np.float64] = np.append(coefficients[:, 1], end)
    return slopes


class TestDerivedVelocities:
    def test_are_the_spline_derivative(self) -> None:
        """Unevenly spaced samples, against the spline's defining conditions."""
        rng = np.random.default_rng(2)
        seconds = np.cumsum(rng.uniform(20.0, 90.0, 40)).round(3)
        seconds -= seconds[0]
        phase = 1.1e-3 * seconds
        position = 7000.0 * np.stack(
            [np.cos(phase), 0.6 * np.sin(phase), 0.8 * np.sin(phase)], axis=1
        )
        times = [BEGIN + timedelta(seconds=float(s)) for s in seconds]
        eph = FileEphemeris.from_arrays(times, position, step_size=60)
        expected = np.stack(
            [not_a_knot_slopes(seconds, position[:, axis]) for axis in range(3)],
            axis=1,
        )
        np.testing.assert_allclose(eph.file_pv.velocity, expected, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(eph.file_pv.position, position)

    @pytest.mark.parametrize(
        "radius_km,step,inside,ends",
        [
            (6900.0, 10, 1e-8, 3e-6),
            (6900.0, 60, 2e-6, 5e-4),
            (42164.0, 60, 1e-9, 1e-7),
        ],
    )
    def test_are_close_to_the_true_velocity(
        self, radius_km: float, step: int, inside: float, ends: float
    ) -> None:
        times, position, velocity = circular_orbit(radius_km, step, 400)
        eph = FileEphemeris.from_arrays(times, position, step_size=step)
        error = np.linalg.norm(eph.file_pv.velocity - velocity, axis=1)
        assert error[5:-5].max() < inside
        assert error.max() < ends
        with_velocity = FileEphemeris.from_arrays(
            times, position, velocity, step_size=step
        )
        np.testing.assert_allclose(
            eph.gcrs_pv.position, with_velocity.gcrs_pv.position, rtol=0, atol=1e-9
        )

    def test_use_the_position_unit_per_second(self) -> None:
        times, position, _ = circular_orbit(42164.0, 60, 30)
        km = FileEphemeris.from_arrays(times, position, step_size=60)
        metres = FileEphemeris.from_arrays(
            times, position * 1000.0, step_size=60, position_unit="m"
        )
        assert metres.source_velocity_unit == "m/s"
        np.testing.assert_allclose(
            metres.file_pv.velocity, km.file_pv.velocity, rtol=1e-12, atol=1e-11
        )

    def test_two_and_three_samples(self) -> None:
        times, position, _ = circular_orbit(42164.0, 60, 3)
        line = FileEphemeris.from_arrays(times[:2], position[:2], step_size=60)
        chord = (position[1] - position[0]) / 60.0
        np.testing.assert_allclose(
            line.file_pv.velocity, [chord, chord], rtol=1e-12, atol=0
        )
        parabola = FileEphemeris.from_arrays(times, position, step_size=60)
        np.testing.assert_allclose(
            parabola.file_pv.velocity[1],
            (position[2] - position[0]) / 120.0,
            rtol=1e-12,
            atol=0,
        )

    def test_invalid_requests(self, arrays: Arrays) -> None:
        t, r, _ = arrays
        with pytest.raises(ValueError, match="velocity_unit was given without"):
            FileEphemeris.from_arrays(t, r, velocity_unit="km/s")
        with pytest.raises(ValueError, match="Length mismatch"):
            FileEphemeris.from_arrays(t[:-1], r)
        with pytest.raises(ValueError, match="At least 2 samples"):
            FileEphemeris.from_arrays(t[:1], r[:1])
