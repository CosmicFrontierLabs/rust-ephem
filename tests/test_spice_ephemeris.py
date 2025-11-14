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
from datetime import datetime, timezone

import pytest

# Check if test data is available
TEST_SPK_PATH = "test_data/de440s.bsp"
SKIP_TEST = not os.path.exists(TEST_SPK_PATH)

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False


class TestSPICEEphemeris:
    pytestmark = [
        pytest.mark.skipif(
            SKIP_TEST, reason=f"SPICE kernel file not found at {TEST_SPK_PATH}"
        ),
        pytest.mark.skipif(
            not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
        ),
    ]

    def test_timestamps_count(self):
        """Test that SPICEEphemeris generates the expected number of timestamps"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600  # 10 minutes

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,  # Moon
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,  # Earth
        )

        timestamps = ephem.timestamp
        expected_count = 7  # 0, 10, 20, 30, 40, 50, 60 minutes
        assert len(timestamps) == expected_count, (
            f"Expected {expected_count} timestamps, got {len(timestamps)}"
        )

    def test_gcrs_available(self):
        """Test that GCRS data is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        gcrs = ephem.gcrs_pv
        assert gcrs is not None, "GCRS data should be available"

    def test_position_shape_rows(self):
        """Test position shape rows"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        position = ephem.gcrs_pv.position
        expected_count = 7
        assert position.shape[0] == expected_count, (
            f"Position should have {expected_count} rows"
        )

    def test_position_shape_columns(self):
        """Test position shape columns"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        position = ephem.gcrs_pv.position
        assert position.shape[1] == 3, "Position should have 3 columns (x, y, z)"

    def test_velocity_shape_rows(self):
        """Test velocity shape rows"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        velocity = ephem.gcrs_pv.velocity
        expected_count = 7
        assert velocity.shape[0] == expected_count, (
            f"Velocity should have {expected_count} rows"
        )

    def test_velocity_shape_columns(self):
        """Test velocity shape columns"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        velocity = ephem.gcrs_pv.velocity
        assert velocity.shape[1] == 3, "Velocity should have 3 columns (vx, vy, vz)"

    def test_position_magnitude_min(self):
        """Test position magnitude minimum"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        import numpy as np

        position = ephem.gcrs_pv.position
        position_magnitude = np.linalg.norm(position, axis=1)
        assert np.all(position_magnitude > 300000), (
            "Moon distance should be > 300,000 km"
        )

    def test_position_magnitude_max(self):
        """Test position magnitude maximum"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        import numpy as np

        position = ephem.gcrs_pv.position
        position_magnitude = np.linalg.norm(position, axis=1)
        assert np.all(position_magnitude < 500000), (
            "Moon distance should be < 500,000 km"
        )

    def test_sun_available(self):
        """Test that Sun data is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        sun = ephem.sun_pv
        assert sun is not None, "Sun data should be available"

    def test_moon_available(self):
        """Test that Moon data is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        moon = ephem.moon_pv
        assert moon is not None, "Moon data should be available"

    def test_obsgeoloc_available(self):
        """Test that obsgeoloc is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        obsgeoloc = ephem.obsgeoloc
        assert obsgeoloc is not None, "obsgeoloc should be available"

    def test_obsgeovel_available(self):
        """Test that obsgeovel is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        obsgeovel = ephem.obsgeovel
        assert obsgeovel is not None, "obsgeovel should be available"

    def test_has_gcrs(self):
        """Test that SPICEEphemeris has gcrs attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "gcrs_pv"), (
            "SPICEEphemeris should have attribute: gcrs_pv"
        )

    def test_has_timestamp(self):
        """Test that SPICEEphemeris has timestamp attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "timestamp"), (
            "SPICEEphemeris should have attribute: timestamp"
        )

    def test_has_sun(self):
        """Test that SPICEEphemeris has sun attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "sun_pv"), (
            "SPICEEphemeris should have attribute: sun_pv"
        )

    def test_has_moon(self):
        """Test that SPICEEphemeris has moon attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "moon_pv"), (
            "SPICEEphemeris should have attribute: moon_pv"
        )

    def test_has_obsgeoloc(self):
        """Test that SPICEEphemeris has obsgeoloc attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "obsgeoloc"), (
            "SPICEEphemeris should have attribute: obsgeoloc"
        )

    def test_has_obsgeovel(self):
        """Test that SPICEEphemeris has obsgeovel attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "obsgeovel"), (
            "SPICEEphemeris should have attribute: obsgeovel"
        )

    def test_has_gcrs_skycoord(self):
        """Test that SPICEEphemeris has gcrs attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "gcrs"), (
            "SPICEEphemeris should have attribute: gcrs"
        )

    def test_has_earth_skycoord(self):
        """Test that SPICEEphemeris has earth attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "earth"), (
            "SPICEEphemeris should have attribute: earth"
        )

    def test_has_sun_skycoord(self):
        """Test that SPICEEphemeris has sun attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "sun"), "SPICEEphemeris should have attribute: sun"

    def test_has_moon_skycoord(self):
        """Test that SPICEEphemeris has moon attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "moon"), (
            "SPICEEphemeris should have attribute: moon"
        )

    def test_has_itrs(self):
        """Test that SPICEEphemeris has itrs attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "itrs_pv"), (
            "SPICEEphemeris should have attribute: itrs_pv"
        )

    def test_has_itrs_skycoord(self):
        """Test that SPICEEphemeris has itrs SkyCoord attribute"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
        step_size = 600

        spice_ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
        )

        assert hasattr(spice_ephem, "itrs"), (
            "SPICEEphemeris should have attribute: itrs"
        )

    def test_itrs_available(self):
        """Test that ITRS data is available"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        itrs = ephem.itrs_pv
        assert itrs is not None, "ITRS data should be available"

    def test_itrs_shape(self):
        """Test ITRS position and velocity shapes"""
        begin = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 600

        ephem = rust_ephem.SPICEEphemeris(
            spk_path=TEST_SPK_PATH,
            naif_id=301,
            begin=begin,
            end=end,
            step_size=step_size,
            center_id=399,
        )

        position = ephem.itrs_pv.position
        velocity = ephem.itrs_pv.velocity
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


def main():  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
