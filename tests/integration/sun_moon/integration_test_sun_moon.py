"""
Integration test to verify Sun and Moon position calculations.

This test verifies that:
1. Sun and Moon positions are calculated and accessible
2. obsgeoloc and obsgeovel properties work correctly
3. Values are within expected physical ranges
"""

import sys
from typing import Any

import numpy as np
import pytest


class TestSingleTimestampSunMoon:
    """Single-assert tests for sun/moon at one timestamp."""

    def test_has_sun(self: Any, single_timestamp_ephem: Any) -> None:
        assert hasattr(single_timestamp_ephem, "sun_pv")

    def test_has_moon(self: Any, single_timestamp_ephem: Any) -> None:
        assert hasattr(single_timestamp_ephem, "moon_pv")

    def test_has_obsgeoloc(self: Any, single_timestamp_ephem: Any) -> None:
        assert hasattr(single_timestamp_ephem, "obsgeoloc")

    def test_has_obsgeovel(self: Any, single_timestamp_ephem: Any) -> None:
        assert hasattr(single_timestamp_ephem, "obsgeovel")

    def test_sun_distance_in_range(self: Any, single_timestamp_ephem: Any) -> None:
        sun_distance: Any = np.linalg.norm(single_timestamp_ephem.sun_pv.position[0])
        assert 147e6 < sun_distance < 152e6

    def test_moon_distance_in_range(self: Any, single_timestamp_ephem: Any) -> None:
        moon_distance: Any = np.linalg.norm(single_timestamp_ephem.moon_pv.position[0])
        assert 356000 < moon_distance < 407000

    def test_obsgeoloc_matches_gcrs_position(
        self: Any, single_timestamp_ephem: Any
    ) -> None:
        assert np.allclose(
            single_timestamp_ephem.gcrs_pv.position[0],
            single_timestamp_ephem.obsgeoloc[0],
        )

    def test_obsgeovel_matches_gcrs_velocity(
        self: Any, single_timestamp_ephem: Any
    ) -> None:
        assert np.allclose(
            single_timestamp_ephem.gcrs_pv.velocity[0],
            single_timestamp_ephem.obsgeovel[0],
        )


class TestMultipleTimestampsSunMoon:
    """Single-assert tests for multi-timestamp sun/moon behavior."""

    def test_sun_position_array_length(self: Any, multi_timestamp_ephem: Any) -> None:
        assert multi_timestamp_ephem.sun_pv.position.shape[0] == len(
            multi_timestamp_ephem.timestamp
        )

    def test_moon_position_array_length(self: Any, multi_timestamp_ephem: Any) -> None:
        assert multi_timestamp_ephem.moon_pv.position.shape[0] == len(
            multi_timestamp_ephem.timestamp
        )

    def test_obsgeoloc_array_length(self: Any, multi_timestamp_ephem: Any) -> None:
        assert multi_timestamp_ephem.obsgeoloc.shape[0] == len(
            multi_timestamp_ephem.timestamp
        )

    def test_obsgeovel_array_length(self: Any, multi_timestamp_ephem: Any) -> None:
        assert multi_timestamp_ephem.obsgeovel.shape[0] == len(
            multi_timestamp_ephem.timestamp
        )

    def test_sun_position_changes_over_time(
        self: Any, multi_timestamp_ephem: Any
    ) -> None:
        assert (
            np.linalg.norm(
                multi_timestamp_ephem.sun_pv.position[-1]
                - multi_timestamp_ephem.sun_pv.position[0]
            )
            > 0
        )

    def test_moon_position_changes_over_time(
        self: Any, multi_timestamp_ephem: Any
    ) -> None:
        assert (
            np.linalg.norm(
                multi_timestamp_ephem.moon_pv.position[-1]
                - multi_timestamp_ephem.moon_pv.position[0]
            )
            > 0
        )


def main() -> int | pytest.ExitCode:  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
