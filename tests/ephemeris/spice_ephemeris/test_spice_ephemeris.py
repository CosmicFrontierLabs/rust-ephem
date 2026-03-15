"""
Integration test for SPICEEphemeris.

This test is skipped by default because it requires a SPICE kernel file.

To run this test:
1. Download a SPICE kernel file (e.g., de440s.bsp) from:
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp

2. Place it in the test_data directory:
    mkdir -p test_data
    # Download file to test_data/de440s.bsp

3. Run the test:
    python tests/test_spice_ephemeris.py

Requirements:
     pip install numpy
"""

import os
import sys

import pytest

from rust_ephem import SPICEEphemeris

# Check if test data is available
TEST_SPK_PATH = "test_data/de440s.bsp"
SKIP_TEST = not os.path.exists(TEST_SPK_PATH)


class TestSPICEEphemeris:
    pytestmark = [
        pytest.mark.skipif(
            SKIP_TEST, reason=f"SPICE kernel file not found at {TEST_SPK_PATH}"
        ),
    ]

    def test_timestamps_count(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris generates the expected number of timestamps"""
        timestamps = spice_ephemeris_1h.timestamp
        expected_count = 7  # 0, 10, 20, 30, 40, 50, 60 minutes
        assert len(timestamps) == expected_count, (
            f"Expected {expected_count} timestamps, got {len(timestamps)}"
        )

    def test_gcrs_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that GCRS data is available"""
        gcrs = spice_ephemeris_1h.gcrs_pv
        assert gcrs is not None, "GCRS data should be available"

    def test_position_shape_rows(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test position shape rows"""
        position = spice_ephemeris_1h.gcrs_pv.position
        expected_count = 7
        assert position.shape[0] == expected_count, (
            f"Position should have {expected_count} rows"
        )

    def test_position_shape_columns(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test position shape columns"""
        position = spice_ephemeris_1h.gcrs_pv.position
        assert position.shape[1] == 3, "Position should have 3 columns (x, y, z)"

    def test_velocity_shape_rows(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test velocity shape rows"""
        velocity = spice_ephemeris_1h.gcrs_pv.velocity
        expected_count = 7
        assert velocity.shape[0] == expected_count, (
            f"Velocity should have {expected_count} rows"
        )

    def test_velocity_shape_columns(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test velocity shape columns"""
        velocity = spice_ephemeris_1h.gcrs_pv.velocity
        assert velocity.shape[1] == 3, "Velocity should have 3 columns (vx, vy, vz)"

    def test_position_magnitude_min(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test position magnitude minimum"""
        import numpy as np

        position = spice_ephemeris_1h.gcrs_pv.position
        position_magnitude = np.linalg.norm(position, axis=1)
        assert np.all(position_magnitude > 300000), (
            "Moon distance should be > 300,000 km"
        )

    def test_position_magnitude_max(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test position magnitude maximum"""
        import numpy as np

        position = spice_ephemeris_1h.gcrs_pv.position
        position_magnitude = np.linalg.norm(position, axis=1)
        assert np.all(position_magnitude < 500000), (
            "Moon distance should be < 500,000 km"
        )

    def test_sun_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that Sun data is available"""
        sun = spice_ephemeris_1h.sun_pv
        assert sun is not None, "Sun data should be available"

    def test_moon_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that Moon data is available"""
        moon = spice_ephemeris_1h.moon_pv
        assert moon is not None, "Moon data should be available"

    def test_obsgeoloc_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that obsgeoloc is available"""
        obsgeoloc = spice_ephemeris_1h.obsgeoloc
        assert obsgeoloc is not None, "obsgeoloc should be available"

    def test_obsgeovel_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that obsgeovel is available"""
        obsgeovel = spice_ephemeris_1h.obsgeovel
        assert obsgeovel is not None, "obsgeovel should be available"

    def test_has_gcrs(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has gcrs attribute"""
        assert hasattr(spice_ephemeris_10m, "gcrs_pv"), (
            "SPICEEphemeris should have attribute: gcrs_pv"
        )

    def test_has_timestamp(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has timestamp attribute"""
        assert hasattr(spice_ephemeris_10m, "timestamp"), (
            "SPICEEphemeris should have attribute: timestamp"
        )

    def test_has_sun(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has sun attribute"""
        assert hasattr(spice_ephemeris_10m, "sun_pv"), (
            "SPICEEphemeris should have attribute: sun_pv"
        )

    def test_has_moon(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has moon attribute"""
        assert hasattr(spice_ephemeris_10m, "moon_pv"), (
            "SPICEEphemeris should have attribute: moon_pv"
        )

    def test_has_obsgeoloc(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has obsgeoloc attribute"""
        assert hasattr(spice_ephemeris_10m, "obsgeoloc"), (
            "SPICEEphemeris should have attribute: obsgeoloc"
        )

    def test_has_obsgeovel(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has obsgeovel attribute"""
        assert hasattr(spice_ephemeris_10m, "obsgeovel"), (
            "SPICEEphemeris should have attribute: obsgeovel"
        )

    def test_has_gcrs_skycoord(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has gcrs attribute"""
        assert hasattr(spice_ephemeris_10m, "gcrs"), (
            "SPICEEphemeris should have attribute: gcrs"
        )

    def test_has_earth_skycoord(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has earth attribute"""
        assert hasattr(spice_ephemeris_10m, "earth"), (
            "SPICEEphemeris should have attribute: earth"
        )

    def test_has_sun_skycoord(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has sun attribute"""
        assert hasattr(spice_ephemeris_10m, "sun"), (
            "SPICEEphemeris should have attribute: sun"
        )

    def test_has_moon_skycoord(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has moon attribute"""
        assert hasattr(spice_ephemeris_10m, "moon"), (
            "SPICEEphemeris should have attribute: moon"
        )

    def test_has_itrs(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has itrs attribute"""
        assert hasattr(spice_ephemeris_10m, "itrs_pv"), (
            "SPICEEphemeris should have attribute: itrs_pv"
        )

    def test_has_itrs_skycoord(self, spice_ephemeris_10m: SPICEEphemeris) -> None:
        """Test that SPICEEphemeris has itrs SkyCoord attribute"""
        assert hasattr(spice_ephemeris_10m, "itrs"), (
            "SPICEEphemeris should have attribute: itrs"
        )

    def test_itrs_available(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test that ITRS data is available"""
        itrs = spice_ephemeris_1h.itrs_pv
        assert itrs is not None, "ITRS data should be available"

    def test_itrs_shape(self, spice_ephemeris_1h: SPICEEphemeris) -> None:
        """Test ITRS position and velocity shapes"""
        position = spice_ephemeris_1h.itrs_pv.position
        velocity = spice_ephemeris_1h.itrs_pv.velocity
        expected_count = 7
        assert position.shape[0] == expected_count, (
            f"ITRS position should have {expected_count} rows"
        )
        assert position.shape[1] == 3, "ITRS position should have 3 columns (x, y, z)"
        assert velocity.shape[0] == expected_count, (
            f"ITRS velocity should have {expected_count} rows"
        )
        assert velocity.shape[1] == 3, (
            "ITRS velocity should have 3 columns (vx, vy, vz)"
        )


def main() -> int:  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
