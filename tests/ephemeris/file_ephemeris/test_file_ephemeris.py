"""Tests for FileEphemeris."""

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from rust_ephem import Ephemeris, FileEphemeris
from rust_ephem._rust_ephem import PositionVelocityData

# Matches the circular orbit radius used in conftest.py
_V_NOMINAL = math.sqrt(398600.4418 / 7000.0)  # ≈ 7.546 km/s

# Standard query window — inside the fixture data range of 00:00–01:00.
BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
STEP = 60  # seconds → 31 output points (0, 60, …, 1800 s)


# ── Basic construction ────────────────────────────────────────────────────────


def test_basic_initialization(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph is not None


def test_timestamp_count(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.timestamp is not None
    assert len(eph.timestamp) == 31  # 30 min / 60 s + 1


def test_step_size_property(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.step_size == STEP


# ── Source metadata properties ────────────────────────────────────────────────


def test_source_frame_from_header(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.source_frame == "J2000"


def test_file_path_property(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.file_path == seconds_file


def test_source_units_default_to_km(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.source_position_unit == "km"
    assert eph.source_velocity_unit == "km/s"


def test_polar_motion_defaults_false(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.polar_motion is False


def test_polar_motion_flag_propagates(seconds_file: str) -> None:
    eph = FileEphemeris(
        seconds_file, begin=BEGIN, end=END, step_size=STEP, polar_motion=True
    )
    assert eph.polar_motion is True


# ── GCRS output ───────────────────────────────────────────────────────────────


def test_gcrs_pv_shape(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    gcrs_pv: PositionVelocityData = eph.gcrs_pv
    assert gcrs_pv.position.shape == (31, 3)
    assert gcrs_pv.velocity.shape == (31, 3)


def test_gcrs_position_in_leo_range(seconds_file: str) -> None:
    """Interpolated positions should remain in the expected LEO altitude band."""
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    radii = np.linalg.norm(eph.gcrs_pv.position, axis=1)
    assert np.all(radii > 6000)
    assert np.all(radii < 8000)


def test_gcrs_orbital_speed_plausible(seconds_file: str) -> None:
    """Orbital speed should stay close to the nominal circular-orbit value."""
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    speeds = np.linalg.norm(eph.gcrs_pv.velocity, axis=1)
    # Allow ±20 % of nominal — tight enough to catch a unit-conversion mistake
    assert np.all(speeds > _V_NOMINAL * 0.8)
    assert np.all(speeds < _V_NOMINAL * 1.2)


# ── ITRS output ───────────────────────────────────────────────────────────────


def test_itrs_pv_shape(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    itrs_pv: PositionVelocityData = eph.itrs_pv
    assert itrs_pv.position.shape == (31, 3)
    assert itrs_pv.velocity.shape == (31, 3)


def test_gcrs_and_itrs_positions_differ(seconds_file: str) -> None:
    """Earth rotation means GCRS and ITRS positions differ at t=0."""
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert not np.allclose(eph.gcrs_pv.position[0], eph.itrs_pv.position[0])


# ── Raw file data ─────────────────────────────────────────────────────────────


def test_file_pv_row_count(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.file_pv.position.shape == (7, 3)
    assert eph.file_pv.velocity.shape == (7, 3)


def test_file_pv_first_row_values(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-5)
    assert eph.file_pv.position[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert eph.file_pv.velocity[0, 1] == pytest.approx(_V_NOMINAL, rel=1e-4)


def test_file_timestamp_count(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert len(eph.file_timestamp) == 7


def test_file_timestamp_utc_aware(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    for ts in eph.file_timestamp:
        assert ts.tzinfo is not None


def test_file_timestamp_first_and_last(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    stamps = eph.file_timestamp
    assert stamps[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert stamps[-1] == datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)


def test_file_timestamp_count_differs_from_output(seconds_file: str) -> None:
    """Raw file rows (7) should differ from the resampled output grid (31)."""
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert len(eph.file_timestamp) == 7
    assert len(eph.timestamp) == 31


# ── Timestamp format variants ─────────────────────────────────────────────────


def test_natural_language_epoch_parses(natural_lang_epoch_file: str) -> None:
    eph = FileEphemeris(natural_lang_epoch_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.file_pv.position.shape[0] == 7
    assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-5)


def test_iso8601_explicit_format(iso8601_file: str) -> None:
    eph = FileEphemeris(
        iso8601_file, begin=BEGIN, end=END, step_size=STEP, time_format="iso8601"
    )
    assert len(eph.timestamp) == 31
    assert eph.file_pv.position.shape[0] == 7


def test_iso8601_auto_detected(iso8601_file: str) -> None:
    """Auto mode should parse absolute ISO 8601 row timestamps without an epoch."""
    eph = FileEphemeris(iso8601_file, begin=BEGIN, end=END, step_size=STEP)
    assert len(eph.timestamp) == 31


def test_days_time_format(days_file: str) -> None:
    eph = FileEphemeris(
        days_file, begin=BEGIN, end=END, step_size=STEP, time_format="days"
    )
    assert eph.file_pv.position.shape[0] == 7
    assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-5)


def test_seconds_time_format_explicit(seconds_file: str) -> None:
    eph = FileEphemeris(
        seconds_file, begin=BEGIN, end=END, step_size=STEP, time_format="seconds"
    )
    assert eph.file_pv.position.shape[0] == 7


def test_explicit_epoch_override(no_epoch_file: str) -> None:
    """Epoch supplied via parameter when file has no header epoch."""
    eph = FileEphemeris(
        no_epoch_file,
        begin=BEGIN,
        end=END,
        step_size=STEP,
        epoch=datetime(2024, 1, 1, 0, 0, 0),
        time_format="seconds",
    )
    assert eph.file_pv.position.shape[0] == 7
    assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-5)


# ── Unit conversion ───────────────────────────────────────────────────────────


def test_meters_source_unit_preserved(meters_file: str) -> None:
    eph = FileEphemeris(
        meters_file,
        begin=BEGIN,
        end=END,
        step_size=STEP,
        position_unit="m",
        velocity_unit="m/s",
    )
    assert eph.source_position_unit == "m"
    assert eph.source_velocity_unit == "m/s"


def test_meters_file_pv_converted_to_km(meters_file: str) -> None:
    """file_pv should expose values in km/km/s after unit conversion."""
    eph = FileEphemeris(
        meters_file,
        begin=BEGIN,
        end=END,
        step_size=STEP,
        position_unit="m",
        velocity_unit="m/s",
    )
    assert eph.file_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-4)
    assert eph.file_pv.velocity[0, 1] == pytest.approx(_V_NOMINAL, rel=1e-4)


def test_meters_gcrs_pv_shape(meters_file: str) -> None:
    eph = FileEphemeris(
        meters_file,
        begin=BEGIN,
        end=END,
        step_size=STEP,
        position_unit="m",
        velocity_unit="m/s",
    )
    assert eph.gcrs_pv.position.shape == (31, 3)


# ── Earth-fixed frame ─────────────────────────────────────────────────────────


def test_ecef_frame_from_header(ecef_file: str) -> None:
    eph = FileEphemeris(ecef_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.source_frame == "ECEF"
    assert eph.gcrs_pv.position.shape == (31, 3)
    assert eph.itrs_pv.position.shape == (31, 3)


def test_frame_override_parameter(seconds_file: str) -> None:
    """frame= parameter overrides whatever frame is stated in the file header."""
    eph = FileEphemeris(
        seconds_file, begin=BEGIN, end=END, step_size=STEP, frame="ECEF"
    )
    assert eph.source_frame == "ECEF"
    assert eph.gcrs_pv.position.shape == (31, 3)


# ── Comment-line tolerance ────────────────────────────────────────────────────


def test_comment_lines_skipped(tmp_path: Any) -> None:
    """Lines beginning with #, //, and ! should be ignored."""
    lines = [
        "# hash comment",
        "ScenarioEpoch  2024-01-01T00:00:00",
        "// double-slash comment",
        "CoordinateSystem  J2000",
        "! exclamation comment",
        "",
        "0.0    7000.0  0.0  0.0  0.0  7.5  0.0",
        "# mid-data comment",
        "600.0  6990.0  4450.0  0.0  -0.32  7.46  0.0",
        "1200.0 6960.0  8850.0  0.0  -0.64  7.40  0.0",
        "1800.0 6910.0  13200.0 0.0  -0.96  7.30  0.0",
        "2400.0 6840.0  17450.0 0.0  -1.28  7.17  0.0",
        "3000.0 6750.0  21550.0 0.0  -1.59  7.01  0.0",
        "3600.0 6640.0  25500.0 0.0  -1.90  6.82  0.0",
    ]
    p = tmp_path / "commented.txt"
    p.write_text("\n".join(lines))
    eph = FileEphemeris(str(p), begin=BEGIN, end=END, step_size=STEP)
    assert eph.file_pv.position.shape[0] == 7


# ── Error cases ───────────────────────────────────────────────────────────────


def test_time_range_before_file_data_raises(seconds_file: str) -> None:
    begin_late = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    end_late = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="exceeds file data range"):
        FileEphemeris(seconds_file, begin=begin_late, end=end_late, step_size=STEP)


def test_no_data_rows_raises(tmp_path: Any) -> None:
    p = tmp_path / "comments_only.txt"
    p.write_text("# comment only\n// another\n")
    with pytest.raises(ValueError):
        FileEphemeris(str(p), begin=BEGIN, end=END, step_size=STEP)


def test_unknown_frame_raises(tmp_path: Any) -> None:
    lines = [
        "ScenarioEpoch  2024-01-01T00:00:00",
        "CoordinateSystem  BADFRAME",
        "0.0    7000.0  0.0  0.0  0.0  7.5  0.0",
        "3600.0 6640.0  25500.0  0.0  -1.9  6.82  0.0",
    ]
    p = tmp_path / "bad_frame.txt"
    p.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="Unsupported coordinate frame"):
        FileEphemeris(str(p), begin=BEGIN, end=END, step_size=STEP)


def test_seconds_format_without_epoch_raises(no_epoch_file: str) -> None:
    with pytest.raises(ValueError, match="epoch"):
        FileEphemeris(
            no_epoch_file, begin=BEGIN, end=END, step_size=STEP, time_format="seconds"
        )


def test_nonexistent_file_raises() -> None:
    with pytest.raises((IOError, OSError)):
        FileEphemeris(
            "/nonexistent/path/orbit.txt", begin=BEGIN, end=END, step_size=STEP
        )


def test_unknown_position_unit_raises(seconds_file: str) -> None:
    with pytest.raises(ValueError, match="Unknown position unit"):
        FileEphemeris(
            seconds_file, begin=BEGIN, end=END, step_size=STEP, position_unit="furlongs"
        )


def test_unknown_velocity_unit_raises(seconds_file: str) -> None:
    with pytest.raises(ValueError, match="Unknown velocity unit"):
        FileEphemeris(
            seconds_file,
            begin=BEGIN,
            end=END,
            step_size=STEP,
            velocity_unit="furlongs/fortnight",
        )


# ── Ephemeris ABC membership ──────────────────────────────────────────────────


def test_isinstance_ephemeris(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert isinstance(eph, Ephemeris)


def test_issubclass_ephemeris() -> None:
    assert issubclass(FileEphemeris, Ephemeris)


# ── Sun / Moon ────────────────────────────────────────────────────────────────


def test_sun_and_moon_accessible(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    assert eph.sun is not None
    assert eph.moon is not None


def test_sun_pv_shape(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    sun_pv: PositionVelocityData = eph.sun_pv
    assert sun_pv.position.shape == (31, 3)
    assert sun_pv.velocity.shape == (31, 3)


# ── Geodetic properties ───────────────────────────────────────────────────────


def test_latitude_deg_shape(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    lat = eph.latitude_deg
    assert lat is not None
    assert len(lat) == 31


def test_latitude_deg_in_range(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    lat = eph.latitude_deg
    assert lat is not None
    assert np.all(np.array(lat) >= -90)
    assert np.all(np.array(lat) <= 90)


def test_height_km_plausible(seconds_file: str) -> None:
    """Altitude should be close to the circular orbit altitude throughout."""
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    h = eph.height_km
    assert h is not None
    h_arr = np.array(h)
    # Circular orbit at 7000 km → altitude ≈ 621.863 km; allow ±50 km for
    # interpolation error and WGS-84 ellipsoid correction.
    assert np.all(h_arr > 570)
    assert np.all(h_arr < 680)


# ── index() method ────────────────────────────────────────────────────────────


def test_index_exact_time(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    target = datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc)
    assert eph.index(target) == 15  # 15 min = 15 × 60 s steps


def test_index_between_steps(seconds_file: str) -> None:
    eph = FileEphemeris(seconds_file, begin=BEGIN, end=END, step_size=STEP)
    target = datetime(2024, 1, 1, 0, 15, 30, tzinfo=timezone.utc)
    assert eph.index(target) in (15, 16)
