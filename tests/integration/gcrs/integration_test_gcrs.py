#!/usr/bin/env python3
"""Integration tests for TEME -> GCRS transformation accuracy.

Refactored so that:
    - All tests reside in a test class
    - Each test function performs exactly ONE assertion
    - Per–timestamp accuracy is validated via parametrization

The original single test that looped over times with multiple asserts has been
split for clearer failure reporting and to satisfy the single-assert rule.
"""

import sys
from typing import Any

import numpy as np
import pytest

# Try to import required modules
try:
    import astropy.units as u  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


from .conftest import POSITION_TOLERANCE_KM


class TestGCRSTransformation:
    """Per‑timestamp TEME→GCRS accuracy tests (single assertion each)."""

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_position_error_within_tolerance(
        self: Any, tle_ephemeris: Any, gcrs_coordinates: Any
    ) -> None:
        """GCRS position error for a single timestamp is below tolerance."""
        rust_gcrs_pos = tle_ephemeris.gcrs_pv.position[0]

        astropy_gcrs_pos = np.array(
            [
                gcrs_coordinates.cartesian.x.to(u.km).value,
                gcrs_coordinates.cartesian.y.to(u.km).value,
                gcrs_coordinates.cartesian.z.to(u.km).value,
            ]
        )
        pos_error_km = np.linalg.norm(rust_gcrs_pos - astropy_gcrs_pos)
        assert pos_error_km < POSITION_TOLERANCE_KM


if __name__ == "__main__":  # pragma: no cover - optional direct execution
    if not ASTROPY_AVAILABLE:
        print("Missing dependencies: astropy")
        sys.exit(1)
    # Run via pytest to leverage parametrization
    raise SystemExit(pytest.main([__file__, "-v"]))
