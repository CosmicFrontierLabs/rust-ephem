"""Tests for ParquetEphemeris."""

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from rust_ephem import Ephemeris, ParquetEphemeris

_V_NOMINAL = math.sqrt(398600.4418 / 7000.0)  # ≈ 7.546 km/s

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
STEP = 60  # → 31 output points


# ── Basic construction ────────────────────────────────────────────────────────


def test_basic_initialization(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph is not None


def test_is_ephemeris_subclass(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert isinstance(eph, Ephemeris)


def test_timestamp_count(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert len(eph.timestamp) == 31  # 30 min / 60 s + 1


def test_step_size_property(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.step_size == STEP


def test_source_property(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.source == gcrs_parquet
    assert eph.file_path == gcrs_parquet  # alias for symmetry with FileEphemeris


def test_default_units_and_frame(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.source_unit == "km"
    assert eph.source_frame == "GCRS"


def test_polar_motion_defaults_false(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.polar_motion is False


def test_polar_motion_flag_propagates(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP, polar_motion=True)
    assert eph.polar_motion is True


# ── GCRS / ITRS output ────────────────────────────────────────────────────────


def test_gcrs_pv_shape(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.gcrs_pv.position.shape == (31, 3)
    assert eph.gcrs_pv.velocity.shape == (31, 3)


def test_gcrs_position_in_leo_range(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
    assert np.all(radii > 6000)
    assert np.all(radii < 8000)


def test_gcrs_orbital_speed_plausible(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    speeds = np.linalg.norm(eph.gcrs_pv.velocity, axis=1)
    assert np.all(speeds > _V_NOMINAL * 0.8)
    assert np.all(speeds < _V_NOMINAL * 1.2)


def test_itrs_pv_shape(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert eph.itrs_pv.position.shape == (31, 3)
    assert eph.itrs_pv.velocity.shape == (31, 3)


def test_gcrs_and_itrs_differ(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    assert not np.allclose(eph.gcrs_pv.position[0], eph.itrs_pv.position[0])


# ── Raw data passthrough ──────────────────────────────────────────────────────


def test_file_pv_row_count(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    # 13 rows in the parquet, all within the 1-hour data window. The Hermite
    # filter pulls everything within ±1h margin.
    assert eph.file_pv.position.shape[0] == 13


def test_file_timestamps_are_utc(gcrs_parquet: str) -> None:
    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    ts = eph.file_timestamp
    assert all(t.tzinfo is not None for t in ts)
    assert ts[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ── Column overrides ──────────────────────────────────────────────────────────


def test_renamed_columns(renamed_parquet: str) -> None:
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


def test_unsafe_column_name_rejected(gcrs_parquet: str) -> None:
    with pytest.raises(ValueError, match="not a valid identifier"):
        ParquetEphemeris(
            gcrs_parquet,
            BEGIN,
            END,
            STEP,
            time_col='time"; DROP TABLE x; --',
        )


# ── Unit conversion ───────────────────────────────────────────────────────────


def test_meters_unit_override(meters_parquet: str) -> None:
    eph = ParquetEphemeris(
        meters_parquet,
        BEGIN,
        END,
        STEP,
        unit="m",
    )
    radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
    # Should be in km after conversion, ~7000 km.
    assert np.all(radii > 6000)
    assert np.all(radii < 8000)


# ── Errors ────────────────────────────────────────────────────────────────────


def test_missing_file_raises(tmp_path: Any) -> None:
    with pytest.raises(Exception):
        ParquetEphemeris(
            str(tmp_path / "does_not_exist.parquet"),
            BEGIN,
            END,
            STEP,
        )


def test_range_outside_data_raises(gcrs_parquet: str) -> None:
    too_late_begin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    too_late_end = datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        ParquetEphemeris(gcrs_parquet, too_late_begin, too_late_end, STEP)


def test_unsupported_frame_raises(gcrs_parquet: str) -> None:
    with pytest.raises(ValueError, match="Unsupported coordinate frame"):
        ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP, frame="GALACTIC")


# ── Constraint integration ────────────────────────────────────────────────────


def test_constraint_evaluation_accepts_parquet_ephemeris(gcrs_parquet: str) -> None:
    """ParquetEphemeris flows through the constraint dispatch in py_api_helpers.rs."""
    from rust_ephem import SunConstraint

    eph = ParquetEphemeris(gcrs_parquet, BEGIN, END, STEP)
    constraint = SunConstraint(min_angle=10.0)
    result = constraint.evaluate(eph, target_ra=83.63, target_dec=22.01)
    # 31 output points; one boolean per timestamp.
    assert len(result.constraint_array) == 31


def test_in_constraint_batch_accepts_parquet_ephemeris(gcrs_parquet: str) -> None:
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
