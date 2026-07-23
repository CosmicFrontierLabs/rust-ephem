"""Fixtures for parquet_ephemeris tests."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Same circular LEO orbit used by file_ephemeris tests so we can cross-check.
_GM = 398600.4418  # km^3/s^2
_R = 7000.0  # km
_V = math.sqrt(_GM / _R)  # ≈ 7.546 km/s
_OMEGA = _V / _R  # rad/s


def _circular_state(
    t_seconds: float,
) -> tuple[float, float, float, float, float, float]:
    theta = _OMEGA * t_seconds
    return (
        _R * math.cos(theta),
        _R * math.sin(theta),
        0.0,
        -_V * math.sin(theta),
        _V * math.cos(theta),
        0.0,
    )


# 13 rows at 5-minute (300 s) intervals spanning 0–3600 s.
_STEP_S = 300
_DURATION_S = 3600
_TIMES_S = list(range(0, _DURATION_S + 1, _STEP_S))


def _state_rows() -> list[tuple[datetime, float, float, float, float, float, float]]:
    rows = []
    for t in _TIMES_S:
        x, y, z, vx, vy, vz = _circular_state(float(t))
        rows.append((EPOCH + timedelta(seconds=t), x, y, z, vx, vy, vz))
    return rows


def _write_parquet(path: str, schema_overrides: dict[str, str] | None = None) -> None:
    """Write the circular-orbit fixture to ``path`` as a Parquet file.

    ``schema_overrides`` lets a fixture rename columns away from the defaults
    (``time``, ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``) so we can exercise
    the ``time_col`` / ``pos_cols`` / ``vel_cols`` overrides.
    """
    overrides = schema_overrides or {}
    cols = {
        "time": [r[0] for r in _state_rows()],
        "x": [r[1] for r in _state_rows()],
        "y": [r[2] for r in _state_rows()],
        "z": [r[3] for r in _state_rows()],
        "vx": [r[4] for r in _state_rows()],
        "vy": [r[5] for r in _state_rows()],
        "vz": [r[6] for r in _state_rows()],
    }
    renamed = {overrides.get(k, k): v for k, v in cols.items()}
    table = pa.table(renamed)
    pq.write_table(table, path)


@pytest.fixture
def gcrs_parquet(tmp_path: Any) -> str:
    """Default schema (`time`, `x/y/z`, `vx/vy/vz`) in km / km/s / GCRS."""
    p = tmp_path / "orbit_gcrs.parquet"
    _write_parquet(str(p))
    return str(p)


@pytest.fixture
def renamed_parquet(tmp_path: Any) -> str:
    """Same data but with non-default column names."""
    p = tmp_path / "orbit_renamed.parquet"
    _write_parquet(
        str(p),
        schema_overrides={
            "time": "epoch_utc",
            "x": "rx",
            "y": "ry",
            "z": "rz",
            "vx": "rdotx",
            "vy": "rdoty",
            "vz": "rdotz",
        },
    )
    return str(p)


@pytest.fixture
def meters_parquet(tmp_path: Any) -> str:
    """Same orbit but positions in metres and velocities in m/s."""
    p = tmp_path / "orbit_meters.parquet"
    rows = _state_rows()
    cols = {
        "time": [r[0] for r in rows],
        "x": [r[1] * 1000.0 for r in rows],
        "y": [r[2] * 1000.0 for r in rows],
        "z": [r[3] * 1000.0 for r in rows],
        "vx": [r[4] * 1000.0 for r in rows],
        "vy": [r[5] * 1000.0 for r in rows],
        "vz": [r[6] * 1000.0 for r in rows],
    }
    pq.write_table(pa.table(cols), str(p))
    return str(p)
