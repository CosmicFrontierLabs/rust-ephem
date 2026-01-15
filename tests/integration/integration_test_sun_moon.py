"""
Integration test to verify Sun and Moon position calculations.

This test verifies that:
1. Sun and Moon positions are calculated and accessible
2. obsgeoloc and obsgeovel properties work correctly
3. Values are within expected physical ranges
"""

import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from rust_ephem._rust_ephem import TLEEphemeris

try:
    import rust_ephem

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False

# Test TLE for NORAD ID 28485 (2004-047A)
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"


class TestSingleTimestampSunMoon:
    """Single-assert tests for sun/moon at one timestamp."""

    @pytest.fixture
    def ephem(self) -> TLEEphemeris:
        test_time = datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
        return rust_ephem.TLEEphemeris(TLE1, TLE2, test_time, test_time, 1)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_has_sun(self: Any, ephem: Any) -> None:
        assert hasattr(ephem, "sun_pv")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_has_moon(self: Any, ephem: Any) -> None:
        assert hasattr(ephem, "moon_pv")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_has_obsgeoloc(self: Any, ephem: Any) -> None:
        assert hasattr(ephem, "obsgeoloc")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_has_obsgeovel(self: Any, ephem: Any) -> None:
        assert hasattr(ephem, "obsgeovel")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_sun_distance_in_range(self: Any, ephem: Any) -> None:
        sun_distance: Any = np.linalg.norm(ephem.sun_pv.position[0])
        assert 147e6 < sun_distance < 152e6

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_moon_distance_in_range(self: Any, ephem: Any) -> None:
        moon_distance: Any = np.linalg.norm(ephem.moon_pv.position[0])
        assert 356000 < moon_distance < 407000

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_obsgeoloc_matches_gcrs_position(self: Any, ephem: Any) -> None:
        assert np.allclose(ephem.gcrs_pv.position[0], ephem.obsgeoloc[0])

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_obsgeovel_matches_gcrs_velocity(self: Any, ephem: Any) -> None:
        assert np.allclose(ephem.gcrs_pv.velocity[0], ephem.obsgeovel[0])


class TestMultipleTimestampsSunMoon:
    """Single-assert tests for multi-timestamp sun/moon behavior."""

    @pytest.fixture
    def ephem(self) -> TLEEphemeris:
        begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
        return rust_ephem.TLEEphemeris(TLE1, TLE2, begin, end, 3600)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_sun_position_array_length(self: Any, ephem: Any) -> None:
        assert ephem.sun_pv.position.shape[0] == len(ephem.timestamp)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_moon_position_array_length(self: Any, ephem: Any) -> None:
        assert ephem.moon_pv.position.shape[0] == len(ephem.timestamp)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_obsgeoloc_array_length(self: Any, ephem: Any) -> None:
        assert ephem.obsgeoloc.shape[0] == len(ephem.timestamp)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_obsgeovel_array_length(self: Any, ephem: Any) -> None:
        assert ephem.obsgeovel.shape[0] == len(ephem.timestamp)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_sun_position_changes_over_time(self: Any, ephem: Any) -> None:
        assert np.linalg.norm(ephem.sun_pv.position[-1] - ephem.sun_pv.position[0]) > 0

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    def test_moon_position_changes_over_time(self: Any, ephem: Any) -> None:
        assert (
            np.linalg.norm(ephem.moon_pv.position[-1] - ephem.moon_pv.position[0]) > 0
        )


def main() -> int | pytest.ExitCode:  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
