"""Fixtures for file_ephemeris tests."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Circular LEO orbit at R = 7000 km.  All 7 state vectors are exact solutions
# to the two-body problem so position radius is constant and Hermite
# interpolation between them stays close to the true orbit.
_GM = 398600.4418  # km^3/s^2
_R = 7000.0  # km
_V = math.sqrt(_GM / _R)  # ≈ 7.546 km/s
_OMEGA = _V / _R  # rad/s


def _circular_row(t: float) -> tuple[float, ...]:
    theta = _OMEGA * t
    return (
        t,
        _R * math.cos(theta),
        _R * math.sin(theta),
        0.0,
        -_V * math.sin(theta),
        _V * math.cos(theta),
        0.0,
    )


# 7 rows at 10-minute (600 s) intervals, spanning 0–3600 s.
_ROWS: list[tuple[float, ...]] = [_circular_row(float(t)) for t in range(0, 3601, 600)]


def _data_lines() -> list[str]:
    return [
        f"{t:.1f}  {x:.4f}  {y:.4f}  {z:.4f}  {vx:.5f}  {vy:.5f}  {vz:.5f}"
        for t, x, y, z, vx, vy, vz in _ROWS
    ]


@pytest.fixture
def seconds_file(tmp_path: Any) -> str:
    """Offset-in-seconds file with ISO epoch header and J2000 frame."""
    lines = [
        "# Test ephemeris file",
        "ScenarioEpoch  2024-01-01T00:00:00",
        "CoordinateSystem  J2000",
        "EphemerisTimePosVel",
        "",
        *_data_lines(),
    ]
    p = tmp_path / "orbit_seconds.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def natural_lang_epoch_file(tmp_path: Any) -> str:
    """Offset-in-seconds file whose epoch uses natural-language format."""
    lines = [
        "ScenarioEpoch  01 Jan 2024 00:00:00.000000",
        "CoordinateSystem  J2000",
        "",
        *_data_lines(),
    ]
    p = tmp_path / "orbit_natural.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def iso8601_file(tmp_path: Any) -> str:
    """File with one absolute ISO 8601 timestamp per data row, no epoch header."""
    rows = []
    for t, x, y, z, vx, vy, vz in _ROWS:
        ts = EPOCH + timedelta(seconds=t)
        rows.append(
            f"{ts.strftime('%Y-%m-%dT%H:%M:%S.000000')}  "
            f"{x:.4f}  {y:.4f}  {z:.4f}  {vx:.5f}  {vy:.5f}  {vz:.5f}"
        )
    lines = [
        "# ISO 8601 timestamp ephemeris",
        "CoordinateSystem  J2000",
        "",
        *rows,
    ]
    p = tmp_path / "orbit_iso8601.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def days_file(tmp_path: Any) -> str:
    """Offset-in-days file with ISO epoch header."""
    rows = [
        f"{t / 86400.0:.10f}  {x:.4f}  {y:.4f}  {z:.4f}  {vx:.5f}  {vy:.5f}  {vz:.5f}"
        for t, x, y, z, vx, vy, vz in _ROWS
    ]
    lines = [
        "ScenarioEpoch  2024-01-01T00:00:00",
        "CoordinateSystem  J2000",
        "",
        *rows,
    ]
    p = tmp_path / "orbit_days.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def meters_file(tmp_path: Any) -> str:
    """Same orbit as seconds_file but positions in metres and velocities in m/s."""
    rows = [
        (
            f"{t:.1f}  {x * 1000:.1f}  {y * 1000:.1f}  {z * 1000:.1f}  "
            f"{vx * 1000:.3f}  {vy * 1000:.3f}  {vz * 1000:.3f}"
        )
        for t, x, y, z, vx, vy, vz in _ROWS
    ]
    lines = [
        "ScenarioEpoch  2024-01-01T00:00:00",
        "CoordinateSystem  J2000",
        "",
        *rows,
    ]
    p = tmp_path / "orbit_meters.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def ecef_file(tmp_path: Any) -> str:
    """File that declares an Earth-fixed (ECEF) coordinate frame."""
    lines = [
        "ScenarioEpoch  2024-01-01T00:00:00",
        "CoordinateSystem  ECEF",
        "",
        *_data_lines(),
    ]
    p = tmp_path / "orbit_ecef.txt"
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def no_epoch_file(tmp_path: Any) -> str:
    """Bare seconds-offset file with no epoch header; caller must supply epoch."""
    lines = [
        "# No epoch header in this file",
        "",
        *_data_lines(),
    ]
    p = tmp_path / "orbit_no_epoch.txt"
    p.write_text("\n".join(lines))
    return str(p)
