"""Tests for FileEphemeris.from_arrays (in-memory state vectors)."""

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
