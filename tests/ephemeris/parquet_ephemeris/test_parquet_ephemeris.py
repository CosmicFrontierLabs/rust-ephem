"""Tests for ParquetEphemeris."""

import math
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb
import numpy as np
import pytest

from rust_ephem import Ephemeris, ParquetEphemeris

_V_NOMINAL = math.sqrt(398600.4418 / 7000.0)  # ≈ 7.546 km/s

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
EPOCH = BEGIN  # first row of the fixtures, which span one hour
END = datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
STEP = 60  # → 31 output points


class TestBasicConstruction:
    """Tests for ParquetEphemeris initialization and basic properties."""

    def test_basic_initialization(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph is not None

    def test_duckdb_connection_uses_worker_limit(
        self, gcrs_parquet: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAYON_NUM_THREADS", "2")
        eph = ParquetEphemeris(
            gcrs_parquet,
            BEGIN,
            END,
            STEP,
            where_clause="current_setting('threads') = 2",
        )
        assert len(eph.timestamp) == 31

    def test_is_ephemeris_subclass(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert isinstance(eph, Ephemeris)

    def test_timestamp_count(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert len(eph.timestamp) == 31  # 30 min / 60 s + 1

    def test_step_size_property(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.step_size == STEP

    def test_source_property(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.source == gcrs_parquet
        assert eph.file_path == gcrs_parquet  # alias for symmetry with FileEphemeris

    def test_default_units_and_frame(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.source_position_unit == "km"
        assert eph.source_velocity_unit == "km/s"
        assert eph.source_frame == "GCRS"

    def test_polar_motion_defaults_false(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.polar_motion is False

    def test_polar_motion_flag_propagates(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP, polar_motion=True)
        assert eph.polar_motion is True


class TestGcrsItrsOutput:
    """Tests for GCRS and ITRS position/velocity output."""

    def test_gcrs_pv_shape(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.gcrs_pv.position.shape == (31, 3)
        assert eph.gcrs_pv.velocity.shape == (31, 3)

    def test_gcrs_position_in_leo_range(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
        assert np.all(radii > 6000)
        assert np.all(radii < 8000)

    def test_gcrs_orbital_speed_plausible(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        speeds = np.linalg.norm(eph.gcrs_pv.velocity, axis=1)
        assert np.all(speeds > _V_NOMINAL * 0.8)
        assert np.all(speeds < _V_NOMINAL * 1.2)

    def test_itrs_pv_shape(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert eph.itrs_pv.position.shape == (31, 3)
        assert eph.itrs_pv.velocity.shape == (31, 3)

    def test_gcrs_and_itrs_differ(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert not np.allclose(eph.gcrs_pv.position[0], eph.itrs_pv.position[0])


class TestRawDataPassthrough:
    """Tests for raw file data access (file_pv, file_timestamp)."""

    def test_file_pv_row_count(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        # 13 rows in the parquet, all within the 1-hour data window. The Hermite
        # filter pulls everything within ±1h margin.
        assert eph.file_pv.position.shape[0] == 13

    def test_file_timestamps_are_utc(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        ts = eph.file_timestamp
        assert all(t.tzinfo is not None for t in ts)
        assert ts[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TestColumnOverrides:
    """Tests for custom column name mapping."""

    def test_renamed_columns(self, renamed_parquet: str) -> None:
        eph = ParquetEphemeris(
            renamed_parquet,
            BEGIN,
            END,
            STEP,
            time_col="epoch_utc",
            pos_cols=("rx", "ry", "rz"),
            vel_cols=("rdotx", "rdoty", "rdotz"),
        )
        radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
        assert np.all(radii > 6000)
        assert np.all(radii < 8000)

    def test_unsafe_column_name_rejected(self, gcrs_parquet: str) -> None:
        with pytest.raises(ValueError, match="not a valid identifier"):
            ParquetEphemeris(
                gcrs_parquet,
                BEGIN,
                END,
                STEP,
                time_col='time"; DROP TABLE x; --',
            )


class TestUnitConversion:
    """Tests for coordinate unit conversion."""

    def test_meters_unit_override(self, meters_parquet: str) -> None:
        eph = ParquetEphemeris(
            meters_parquet,
            BEGIN,
            END,
            STEP,
            position_unit="m",
        )
        radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
        # Should be in km after conversion, ~7000 km.
        assert np.all(radii > 6000)
        assert np.all(radii < 8000)
        assert eph.source_position_unit == "m"
        assert eph.source_velocity_unit == "m/s"

    def test_explicit_velocity_unit_preserved(self, meters_parquet: str) -> None:
        eph = ParquetEphemeris(
            meters_parquet,
            BEGIN,
            END,
            STEP,
            position_unit="m",
            velocity_unit="m/s",
        )
        assert eph.source_position_unit == "m"
        assert eph.source_velocity_unit == "m/s"

    def test_meters_file_pv_converted_to_km(self, meters_parquet: str) -> None:
        """file_pv should expose values in km / km/s after unit conversion."""
        eph = ParquetEphemeris(
            meters_parquet,
            BEGIN,
            END,
            STEP,
            position_unit="m",
            velocity_unit="m/s",
        )
        # At t=0 the circular orbit starts at (R, 0, 0) with velocity (0, V, 0).
        assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-4)
        assert eph.file_pv.velocity[0, 1] == pytest.approx(_V_NOMINAL, rel=1e-4)


class TestErrors:
    """Tests for error handling and validation."""

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        with pytest.raises(duckdb.Error, match="does_not_exist"):
            ParquetEphemeris(
                str(tmp_path / "does_not_exist.parquet"),
                BEGIN,
                END,
                STEP,
            )

    def test_range_outside_data_raises(self, gcrs_parquet: str) -> None:
        too_late_begin = datetime(2025, 1, 1, tzinfo=timezone.utc)
        too_late_end = datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            ParquetEphemeris(gcrs_parquet, too_late_begin, too_late_end, STEP)

    def test_unsupported_frame_raises(self, gcrs_parquet: str) -> None:
        with pytest.raises(ValueError, match="Unsupported coordinate frame"):
            ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP, frame="GALACTIC")

    def test_unknown_position_unit_raises(self, gcrs_parquet: str) -> None:
        with pytest.raises(ValueError, match="Unknown position unit"):
            ParquetEphemeris(
                gcrs_parquet,
                BEGIN,
                END,
                STEP,
                position_unit="furlongs",
            )

    def test_unknown_velocity_unit_raises(self, gcrs_parquet: str) -> None:
        with pytest.raises(ValueError, match="Unknown velocity unit"):
            ParquetEphemeris(
                gcrs_parquet,
                BEGIN,
                END,
                STEP,
                velocity_unit="furlongs/fortnight",
            )


class TestConstraintIntegration:
    """Tests for constraint evaluation with ParquetEphemeris."""

    def test_constraint_evaluation_accepts_parquet_ephemeris(
        self, gcrs_parquet: str
    ) -> None:
        """ParquetEphemeris flows through the constraint dispatch in py_api_helpers.rs."""
        from rust_ephem import SunConstraint

        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        constraint = SunConstraint(min_angle=10.0)
        result = constraint.evaluate(eph, target_ra=83.63, target_dec=22.01)
        # 31 output points; one boolean per timestamp.
        assert len(result.constraint_array) == 31

    def test_in_constraint_batch_accepts_parquet_ephemeris(
        self, gcrs_parquet: str
    ) -> None:
        """Batch evaluation path also dispatches on ParquetEphemeris."""
        from rust_ephem import SunConstraint

        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        constraint = SunConstraint(min_angle=10.0)
        out = constraint.in_constraint_batch(
            eph,
            target_ras=[83.63, 0.0],
            target_decs=[22.01, 0.0],
        )
        assert out.shape == (2, 31)


class TestOptionalRange:
    """``begin`` / ``end`` default to the span of the data."""

    def test_defaults_to_the_data_range(self, gcrs_parquet: str) -> None:
        eph = ParquetEphemeris(gcrs_parquet, step_size=300)
        full = ParquetEphemeris(gcrs_parquet, EPOCH, EPOCH + timedelta(hours=1), 300)
        assert list(eph.timestamp) == list(full.timestamp)
        assert eph.timestamp[0] == EPOCH
        assert eph.timestamp[-1] == EPOCH + timedelta(hours=1)
        np.testing.assert_array_equal(eph.gcrs_pv.position, full.gcrs_pv.position)

    def test_one_bound_may_be_given(self, gcrs_parquet: str) -> None:
        late = ParquetEphemeris(gcrs_parquet, begin=BEGIN + timedelta(minutes=30))
        assert late.timestamp[0] == BEGIN + timedelta(minutes=30)
        assert late.timestamp[-1] == EPOCH + timedelta(hours=1)
        early = ParquetEphemeris(gcrs_parquet, end=END)
        assert early.timestamp[0] == EPOCH
        assert early.timestamp[-1] == END


class TestPyArrowFallback:
    """Without duckdb, local sources are read with pyarrow."""

    @pytest.fixture
    def no_duckdb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``None`` in sys.modules makes ``import duckdb`` raise ImportError.
        monkeypatch.setitem(sys.modules, "duckdb", None)

    def test_matches_duckdb(
        self, gcrs_parquet: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        monkeypatch.setitem(sys.modules, "duckdb", None)
        eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
        assert list(eph.timestamp) == list(expected.timestamp)
        assert list(eph.file_timestamp) == list(expected.file_timestamp)
        np.testing.assert_array_equal(eph.gcrs_pv.position, expected.gcrs_pv.position)
        np.testing.assert_array_equal(eph.gcrs_pv.velocity, expected.gcrs_pv.velocity)

    def test_whole_file_without_a_range(
        self, gcrs_parquet: str, no_duckdb: None
    ) -> None:
        eph = ParquetEphemeris(gcrs_parquet, step_size=300)
        assert len(eph.timestamp) == 13
        assert len(eph.file_timestamp) == 13

    def test_renamed_columns_and_units(
        self, renamed_parquet: str, meters_parquet: str, no_duckdb: None
    ) -> None:
        renamed = ParquetEphemeris(
            renamed_parquet,
            BEGIN,
            END,
            STEP,
            time_col="epoch_utc",
            pos_cols=("rx", "ry", "rz"),
            vel_cols=("rdotx", "rdoty", "rdotz"),
        )
        assert np.linalg.norm(renamed.gcrs_pv.position[0]) == pytest.approx(7000.0)
        metres = ParquetEphemeris(meters_parquet, BEGIN, END, STEP, position_unit="m")
        assert np.linalg.norm(metres.gcrs_pv.position[0]) == pytest.approx(7000.0)

    def test_glob_and_unsorted_files(self, tmp_path: Any, no_duckdb: None) -> None:
        """Rows split across files in the wrong order come back time-sorted."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            (EPOCH + timedelta(seconds=300 * i), 7000.0 + i, 0.0, 0.0, 0.0, 7.5, 0.0)
            for i in range(13)
        ]
        names = ["time", "x", "y", "z", "vx", "vy", "vz"]
        for name, part in (("a", rows[7:]), ("b", rows[:7])):
            columns = {n: [r[i] for r in part] for i, n in enumerate(names)}
            pq.write_table(pa.table(columns), tmp_path / f"part_{name}.parquet")
        eph = ParquetEphemeris(str(tmp_path / "part_*.parquet"), step_size=300)
        assert list(eph.file_timestamp) == [r[0] for r in rows]
        np.testing.assert_allclose(eph.file_pv.position[:, 0], [r[1] for r in rows])

    def test_missing_glob_raises(self, tmp_path: Any, no_duckdb: None) -> None:
        with pytest.raises(FileNotFoundError, match="No files match"):
            ParquetEphemeris(str(tmp_path / "nothing_*.parquet"))

    def test_duckdb_only_features_say_so(
        self, gcrs_parquet: str, no_duckdb: None
    ) -> None:
        with pytest.raises(ImportError, match="where_clause need duckdb"):
            ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP, where_clause="x > 0")
        with pytest.raises(ImportError, match="remote sources need duckdb"):
            ParquetEphemeris("s3://bucket/orbit.parquet", BEGIN, END, STEP)
