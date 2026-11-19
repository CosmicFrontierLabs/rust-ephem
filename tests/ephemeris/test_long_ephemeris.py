"""Ephemerides longer than the old 100,000-sample cap."""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

import rust_ephem

BEGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_a_year_at_two_minutes(ensure_planetary_data: Any) -> None:
    eph = rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN,
        end=END,
        step_size=120,
    )
    assert len(eph.timestamp) == 262_801
    assert eph.timestamp[-1] == END
    # Rows converted on different threads agree with a short ephemeris of the
    # same instants: splitting the work does not change the result.
    rows = [0, 1, 131_400, 262_799, 262_800]
    short = rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=eph.timestamp[131_400],
        end=eph.timestamp[131_400],
        step_size=120,
    )
    np.testing.assert_allclose(
        eph.gcrs_pv.position[131_400], short.gcrs_pv.position[0], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        eph.sun_pv.position[131_400], short.sun_pv.position[0], rtol=0, atol=1e-6
    )
    radius = np.linalg.norm(eph.gcrs_pv.position[rows], axis=1)
    np.testing.assert_allclose(radius, radius[0], rtol=1e-9)


def test_runaway_ranges_are_still_refused() -> None:
    with pytest.raises(ValueError, match=r"timestamps \(max: 5000000\)"):
        rust_ephem.GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN,
            end=END,
            step_size=1,
        )
