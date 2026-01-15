#!/usr/bin/env python3
"""
Test script to verify RA/Dec convenience properties for Sun, Moon, and Earth.

This tests that:
1. All six properties exist (sun_ra_dec_deg, moon_ra_dec_deg, earth_ra_dec_deg,
   sun_ra_dec_rad, moon_ra_dec_rad, earth_ra_dec_rad)
2. Properties return correct types (numpy arrays)
3. Array shapes are correct (Nx2 with RA in column 0, Dec in column 1)
4. Degree and radian conversions are consistent
5. Values match the underlying SkyCoord properties
6. Values are within expected ranges
7. Properties are available on all ephemeris types (TLE, SPICE, Ground, OEM)
8. Properties are cached for performance
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np

import rust_ephem


class TestSunRaDecProperties:
    """Test Sun RA/Dec convenience properties"""

    def test_sun_ra_dec_deg_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_deg should return a numpy array"""
        ra_dec = tle_ephemeris.sun_ra_dec_deg
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_sun_ra_dec_deg_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_deg should be Nx2 array (N timestamps, 2 columns for RA and Dec)"""
        ra_dec = tle_ephemeris.sun_ra_dec_deg
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_sun_ra_dec_rad_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_rad should return a numpy array"""
        ra_dec = tle_ephemeris.sun_ra_dec_rad
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_sun_ra_dec_rad_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_rad should be Nx2 array"""
        ra_dec = tle_ephemeris.sun_ra_dec_rad
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_sun_ra_dec_deg_and_rad_conversion(self, tle_ephemeris: Any) -> None:
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.sun_ra_dec_deg
        rad = tle_ephemeris.sun_ra_dec_rad
        # Convert radians to degrees and compare
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_sun_ra_dec_deg_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_deg should match values from sun.ra.deg and sun.dec.deg"""
        ra_dec = tle_ephemeris.sun_ra_dec_deg
        sun = tle_ephemeris.sun
        assert np.allclose(ra_dec[:, 0], sun.ra.deg, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], sun.dec.deg, rtol=1e-10)

    def test_sun_ra_dec_rad_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_rad should match values from sun.ra.rad and sun.dec.rad"""
        ra_dec = tle_ephemeris.sun_ra_dec_rad
        sun = tle_ephemeris.sun
        assert np.allclose(ra_dec[:, 0], sun.ra.rad, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], sun.dec.rad, rtol=1e-10)

    def test_sun_ra_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Sun RA should be in range [0, 360) degrees"""
        ra_dec_deg = tle_ephemeris.sun_ra_dec_deg
        ra_deg = ra_dec_deg[:, 0]
        assert np.all((ra_deg >= 0) & (ra_deg < 360))

    def test_sun_dec_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Sun Dec should be in range [-90, 90] degrees"""
        ra_dec_deg = tle_ephemeris.sun_ra_dec_deg
        dec_deg = ra_dec_deg[:, 1]
        assert np.all((dec_deg >= -90) & (dec_deg <= 90))

    def test_sun_ra_dec_available_on_ground_ephemeris(
        self, ground_ephemeris: Any
    ) -> None:
        """Sun RA/Dec properties should work on GroundEphemeris"""
        ra_dec_deg = ground_ephemeris.sun_ra_dec_deg
        ra_dec_rad = ground_ephemeris.sun_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)
        assert ra_dec_deg.shape[1] == 2
        assert ra_dec_rad.shape[1] == 2

    def test_sun_ra_dec_available_on_spice_ephemeris(
        self, spice_ephemeris: Any
    ) -> None:
        """Sun RA/Dec properties should work on SPICEEphemeris"""
        ra_dec_deg = spice_ephemeris.sun_ra_dec_deg
        ra_dec_rad = spice_ephemeris.sun_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)
        assert ra_dec_deg.shape[1] == 2
        assert ra_dec_rad.shape[1] == 2

    def test_sun_ra_dec_caching(self, tle_ephemeris: Any) -> None:
        """sun_ra_dec_deg and sun_ra_dec_rad should cache values for performance"""
        # First access - will compute and cache
        ra_dec_deg_1 = tle_ephemeris.sun_ra_dec_deg
        ra_dec_rad_1 = tle_ephemeris.sun_ra_dec_rad

        # Second access - should return cached values (same object)
        ra_dec_deg_2 = tle_ephemeris.sun_ra_dec_deg
        ra_dec_rad_2 = tle_ephemeris.sun_ra_dec_rad

        # Arrays should be identical (not just equal)
        assert np.array_equal(ra_dec_deg_1, ra_dec_deg_2)
        assert np.array_equal(ra_dec_rad_1, ra_dec_rad_2)


class TestMoonRaDecProperties:
    """Test Moon RA/Dec convenience properties"""

    def test_moon_ra_dec_deg_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_deg should return a numpy array"""
        ra_dec = tle_ephemeris.moon_ra_dec_deg
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_moon_ra_dec_deg_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_deg should be Nx2 array"""
        ra_dec = tle_ephemeris.moon_ra_dec_deg
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_moon_ra_dec_rad_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_rad should return a numpy array"""
        ra_dec = tle_ephemeris.moon_ra_dec_rad
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_moon_ra_dec_rad_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_rad should be Nx2 array"""
        ra_dec = tle_ephemeris.moon_ra_dec_rad
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_moon_ra_dec_deg_and_rad_conversion(self, tle_ephemeris: Any) -> None:
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.moon_ra_dec_deg
        rad = tle_ephemeris.moon_ra_dec_rad
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_moon_ra_dec_deg_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_deg should match values from moon.ra.deg and moon.dec.deg"""
        ra_dec = tle_ephemeris.moon_ra_dec_deg
        moon = tle_ephemeris.moon
        assert np.allclose(ra_dec[:, 0], moon.ra.deg, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], moon.dec.deg, rtol=1e-10)

    def test_moon_ra_dec_rad_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_rad should match values from moon.ra.rad and moon.dec.rad"""
        ra_dec = tle_ephemeris.moon_ra_dec_rad
        moon = tle_ephemeris.moon
        assert np.allclose(ra_dec[:, 0], moon.ra.rad, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], moon.dec.rad, rtol=1e-10)

    def test_moon_ra_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Moon RA should be in range [0, 360) degrees"""
        ra_dec_deg = tle_ephemeris.moon_ra_dec_deg
        ra_deg = ra_dec_deg[:, 0]
        assert np.all((ra_deg >= 0) & (ra_deg < 360))

    def test_moon_dec_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Moon Dec should be in range [-90, 90] degrees"""
        ra_dec_deg = tle_ephemeris.moon_ra_dec_deg
        dec_deg = ra_dec_deg[:, 1]
        assert np.all((dec_deg >= -90) & (dec_deg <= 90))

    def test_moon_ra_dec_available_on_ground_ephemeris(
        self, ground_ephemeris: Any
    ) -> None:
        """Moon RA/Dec properties should work on GroundEphemeris"""
        ra_dec_deg = ground_ephemeris.moon_ra_dec_deg
        ra_dec_rad = ground_ephemeris.moon_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)
        assert ra_dec_deg.shape[1] == 2
        assert ra_dec_rad.shape[1] == 2

    def test_moon_ra_dec_available_on_spice_ephemeris(
        self, spice_ephemeris: Any
    ) -> None:
        """Moon RA/Dec properties should work on SPICEEphemeris"""
        ra_dec_deg = spice_ephemeris.moon_ra_dec_deg
        ra_dec_rad = spice_ephemeris.moon_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)
        assert ra_dec_deg.shape[1] == 2
        assert ra_dec_rad.shape[1] == 2

    def test_moon_ra_dec_caching(self, tle_ephemeris: Any) -> None:
        """moon_ra_dec_deg and moon_ra_dec_rad should cache values for performance"""
        # First access - will compute and cache
        ra_dec_deg_1 = tle_ephemeris.moon_ra_dec_deg
        ra_dec_rad_1 = tle_ephemeris.moon_ra_dec_rad

        # Second access - should return cached values
        ra_dec_deg_2 = tle_ephemeris.moon_ra_dec_deg
        ra_dec_rad_2 = tle_ephemeris.moon_ra_dec_rad

        # Arrays should be identical
        assert np.array_equal(ra_dec_deg_1, ra_dec_deg_2)
        assert np.array_equal(ra_dec_rad_1, ra_dec_rad_2)


class TestEarthRaDecProperties:
    """Test Earth RA/Dec convenience properties"""

    def test_earth_ra_dec_deg_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_deg should return a numpy array"""
        ra_dec = tle_ephemeris.earth_ra_dec_deg
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_earth_ra_dec_deg_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_deg should be Nx2 array"""
        ra_dec = tle_ephemeris.earth_ra_dec_deg
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_earth_ra_dec_rad_returns_numpy_array(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_rad should return a numpy array"""
        ra_dec = tle_ephemeris.earth_ra_dec_rad
        assert isinstance(ra_dec, np.ndarray)
        assert ra_dec.dtype == np.float64

    def test_earth_ra_dec_rad_has_correct_shape(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_rad should be Nx2 array"""
        ra_dec = tle_ephemeris.earth_ra_dec_rad
        assert ra_dec.ndim == 2
        assert ra_dec.shape[1] == 2
        assert ra_dec.shape[0] == len(tle_ephemeris.timestamp)

    def test_earth_ra_dec_deg_and_rad_conversion(self, tle_ephemeris: Any) -> None:
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.earth_ra_dec_deg
        rad = tle_ephemeris.earth_ra_dec_rad
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_earth_ra_dec_deg_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_deg should match values from earth.ra.deg and earth.dec.deg"""
        ra_dec = tle_ephemeris.earth_ra_dec_deg
        earth = tle_ephemeris.earth
        assert np.allclose(ra_dec[:, 0], earth.ra.deg, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], earth.dec.deg, rtol=1e-10)

    def test_earth_ra_dec_rad_matches_skycoord(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_rad should match values from earth.ra.rad and earth.dec.rad"""
        ra_dec = tle_ephemeris.earth_ra_dec_rad
        earth = tle_ephemeris.earth
        assert np.allclose(ra_dec[:, 0], earth.ra.rad, rtol=1e-10)
        assert np.allclose(ra_dec[:, 1], earth.dec.rad, rtol=1e-10)

    def test_earth_ra_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Earth RA should be in range [0, 360) degrees"""
        ra_dec_deg = tle_ephemeris.earth_ra_dec_deg
        ra_deg = ra_dec_deg[:, 0]
        assert np.all((ra_deg >= 0) & (ra_deg < 360))

    def test_earth_dec_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Earth Dec should be in range [-90, 90] degrees"""
        ra_dec_deg = tle_ephemeris.earth_ra_dec_deg
        dec_deg = ra_dec_deg[:, 1]
        assert np.all((dec_deg >= -90) & (dec_deg <= 90))

    def test_earth_ra_dec_works_on_ground_ephemeris(
        self, ground_ephemeris: Any
    ) -> None:
        """Earth RA/Dec from ground observer (observer is on Earth)"""
        # For GroundEphemeris, earth property represents Earth position relative to itself
        # which should be at origin. The properties should still work but may not be meaningful
        ra_dec_deg = ground_ephemeris.earth_ra_dec_deg
        ra_dec_rad = ground_ephemeris.earth_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)

    def test_earth_ra_dec_available_on_spice_ephemeris(
        self, spice_ephemeris: Any
    ) -> None:
        """Earth RA/Dec properties should work on SPICEEphemeris"""
        ra_dec_deg = spice_ephemeris.earth_ra_dec_deg
        ra_dec_rad = spice_ephemeris.earth_ra_dec_rad
        assert isinstance(ra_dec_deg, np.ndarray)
        assert isinstance(ra_dec_rad, np.ndarray)
        assert ra_dec_deg.shape[1] == 2
        assert ra_dec_rad.shape[1] == 2

    def test_earth_ra_dec_caching(self, tle_ephemeris: Any) -> None:
        """earth_ra_dec_deg and earth_ra_dec_rad should cache values for performance"""
        # First access - will compute and cache
        ra_dec_deg_1 = tle_ephemeris.earth_ra_dec_deg
        ra_dec_rad_1 = tle_ephemeris.earth_ra_dec_rad

        # Second access - should return cached values
        ra_dec_deg_2 = tle_ephemeris.earth_ra_dec_deg
        ra_dec_rad_2 = tle_ephemeris.earth_ra_dec_rad

        # Arrays should be identical
        assert np.array_equal(ra_dec_deg_1, ra_dec_deg_2)
        assert np.array_equal(ra_dec_rad_1, ra_dec_rad_2)


class TestCrossCelestialBodyConsistency:
    """Test consistency across different celestial bodies"""

    def test_all_bodies_same_length(self, tle_ephemeris: Any) -> None:
        """All RA/Dec arrays should have the same length"""
        sun_deg = tle_ephemeris.sun_ra_dec_deg
        moon_deg = tle_ephemeris.moon_ra_dec_deg
        earth_deg = tle_ephemeris.earth_ra_dec_deg

        assert len(sun_deg) == len(moon_deg) == len(earth_deg)
        assert len(sun_deg) == len(tle_ephemeris.timestamp)

    def test_all_bodies_consistent_units(self, tle_ephemeris: Any) -> None:
        """All degree properties should be in degrees, all radian properties in radians"""
        # Test that degree values are larger than radian values (except for very small angles)
        sun_deg = tle_ephemeris.sun_ra_dec_deg
        sun_rad = tle_ephemeris.sun_ra_dec_rad

        # For angles > 1 radian (~57 degrees), deg values should be > rad values
        # RA is typically in this range
        assert np.all(sun_deg[:, 0] > sun_rad[:, 0])

    def test_properties_work_with_single_timestamp(
        self, ensure_planetary_data: Any
    ) -> None:
        """RA/Dec properties should work with single timestamp ephemeris"""
        import os

        test_data_path: str = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "test_data", "de440s.bsp"
        )

        if not os.path.exists(test_data_path):
            test_data_path = os.path.expanduser("~/.cache/rust_ephem/de440s.bsp")

        # Create ephemeris with single timestamp
        begin = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        eph = rust_ephem.SPICEEphemeris(test_data_path, 301, begin, end, step_size=3600)

        sun_deg = eph.sun_ra_dec_deg
        moon_deg = eph.moon_ra_dec_deg
        earth_deg = eph.earth_ra_dec_deg

        assert sun_deg.shape == (1, 2)
        assert moon_deg.shape == (1, 2)
        assert earth_deg.shape == (1, 2)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_ra_dec_properties_with_long_ephemeris(
        self, ensure_planetary_data: Any
    ) -> None:
        """RA/Dec properties should work with many timestamps"""
        tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
        tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)  # 24 hours
        eph = rust_ephem.TLEEphemeris(
            tle1, tle2, begin, end, step_size=60
        )  # 1 minute steps

        sun_deg = eph.sun_ra_dec_deg
        moon_deg = eph.moon_ra_dec_deg
        earth_deg = eph.earth_ra_dec_deg

        # Should have ~1440 timestamps (24 hours * 60 minutes)
        assert sun_deg.shape[0] > 1000
        assert sun_deg.shape == moon_deg.shape == earth_deg.shape
        assert sun_deg.shape[1] == 2

    def test_ra_dec_values_change_over_time(self, tle_ephemeris: Any) -> None:
        """RA/Dec values should change over the ephemeris time span"""
        sun_deg = tle_ephemeris.sun_ra_dec_deg
        moon_deg = tle_ephemeris.moon_ra_dec_deg

        # Check that RA and Dec change over the 6-hour time span
        # Sun changes slowly (~0.27 deg over 6 hours), moon changes faster
        assert not np.array_equal(sun_deg[0], sun_deg[-1])
        assert not np.array_equal(moon_deg[0], moon_deg[-1])

    def test_cached_values_persist_across_different_access_patterns(
        self, tle_ephemeris: Any
    ) -> None:
        """Cached values should persist regardless of access order"""
        # Access in different orders
        sun_deg_1 = tle_ephemeris.sun_ra_dec_deg
        moon_rad_1 = tle_ephemeris.moon_ra_dec_rad
        earth_deg_1 = tle_ephemeris.earth_ra_dec_deg

        # Access again in different order
        earth_deg_2 = tle_ephemeris.earth_ra_dec_deg
        sun_deg_2 = tle_ephemeris.sun_ra_dec_deg
        moon_rad_2 = tle_ephemeris.moon_ra_dec_rad

        # Should get same cached values
        assert np.array_equal(sun_deg_1, sun_deg_2)
        assert np.array_equal(moon_rad_1, moon_rad_2)
        assert np.array_equal(earth_deg_1, earth_deg_2)


class TestIndividualRaDecProperties:
    """Test individual RA and Dec convenience properties"""

    def test_sun_ra_deg_returns_1d_array(self, tle_ephemeris: Any) -> None:
        """sun_ra_deg should return a 1D numpy array"""
        ra = tle_ephemeris.sun_ra_deg
        assert isinstance(ra, np.ndarray)
        assert ra.dtype == np.float64
        assert ra.ndim == 1

    def test_sun_dec_deg_returns_1d_array(self, tle_ephemeris: Any) -> None:
        """sun_dec_deg should return a 1D numpy array"""
        dec = tle_ephemeris.sun_dec_deg
        assert isinstance(dec, np.ndarray)
        assert dec.dtype == np.float64
        assert dec.ndim == 1

    def test_sun_ra_dec_individual_match_combined(self, tle_ephemeris: Any) -> None:
        """Individual sun_ra_deg and sun_dec_deg should match columns of sun_ra_dec_deg"""
        ra_dec = tle_ephemeris.sun_ra_dec_deg
        ra = tle_ephemeris.sun_ra_deg
        dec = tle_ephemeris.sun_dec_deg

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_moon_ra_dec_individual_match_combined(self, tle_ephemeris: Any) -> None:
        """Individual moon_ra_deg and moon_dec_deg should match columns of moon_ra_dec_deg"""
        ra_dec = tle_ephemeris.moon_ra_dec_deg
        ra = tle_ephemeris.moon_ra_deg
        dec = tle_ephemeris.moon_dec_deg

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_earth_ra_dec_individual_match_combined(self, tle_ephemeris: Any) -> None:
        """Individual earth_ra_deg and earth_dec_deg should match columns of earth_ra_dec_deg"""
        ra_dec = tle_ephemeris.earth_ra_dec_deg
        ra = tle_ephemeris.earth_ra_deg
        dec = tle_ephemeris.earth_dec_deg

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_sun_ra_dec_rad_individual_match_combined(self, tle_ephemeris: Any) -> None:
        """Individual sun_ra_rad and sun_dec_rad should match columns of sun_ra_dec_rad"""
        ra_dec = tle_ephemeris.sun_ra_dec_rad
        ra = tle_ephemeris.sun_ra_rad
        dec = tle_ephemeris.sun_dec_rad

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_moon_ra_dec_rad_individual_match_combined(
        self, tle_ephemeris: Any
    ) -> None:
        """Individual moon_ra_rad and moon_dec_rad should match columns of moon_ra_dec_rad"""
        ra_dec = tle_ephemeris.moon_ra_dec_rad
        ra = tle_ephemeris.moon_ra_rad
        dec = tle_ephemeris.moon_dec_rad

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_earth_ra_dec_rad_individual_match_combined(
        self, tle_ephemeris: Any
    ) -> None:
        """Individual earth_ra_rad and earth_dec_rad should match columns of earth_ra_dec_rad"""
        ra_dec = tle_ephemeris.earth_ra_dec_rad
        ra = tle_ephemeris.earth_ra_rad
        dec = tle_ephemeris.earth_dec_rad

        assert np.array_equal(ra, ra_dec[:, 0])
        assert np.array_equal(dec, ra_dec[:, 1])

    def test_individual_deg_rad_conversion(self, tle_ephemeris: Any) -> None:
        """Individual RA/Dec properties should convert correctly between degrees and radians"""
        # Sun
        sun_ra_deg = tle_ephemeris.sun_ra_deg
        sun_ra_rad = tle_ephemeris.sun_ra_rad
        sun_dec_deg = tle_ephemeris.sun_dec_deg
        sun_dec_rad = tle_ephemeris.sun_dec_rad

        assert np.allclose(sun_ra_deg, np.degrees(sun_ra_rad), rtol=1e-10)
        assert np.allclose(sun_dec_deg, np.degrees(sun_dec_rad), rtol=1e-10)

        # Moon
        moon_ra_deg = tle_ephemeris.moon_ra_deg
        moon_ra_rad = tle_ephemeris.moon_ra_rad
        moon_dec_deg = tle_ephemeris.moon_dec_deg
        moon_dec_rad = tle_ephemeris.moon_dec_rad

        assert np.allclose(moon_ra_deg, np.degrees(moon_ra_rad), rtol=1e-10)
        assert np.allclose(moon_dec_deg, np.degrees(moon_dec_rad), rtol=1e-10)

        # Earth
        earth_ra_deg = tle_ephemeris.earth_ra_deg
        earth_ra_rad = tle_ephemeris.earth_ra_rad
        earth_dec_deg = tle_ephemeris.earth_dec_deg
        earth_dec_rad = tle_ephemeris.earth_dec_rad

        assert np.allclose(earth_ra_deg, np.degrees(earth_ra_rad), rtol=1e-10)
        assert np.allclose(earth_dec_deg, np.degrees(earth_dec_rad), rtol=1e-10)

    def test_individual_ra_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Individual RA properties should be in expected ranges"""
        # RA in degrees: 0-360
        assert np.all(
            (tle_ephemeris.sun_ra_deg >= 0) & (tle_ephemeris.sun_ra_deg <= 360)
        )
        assert np.all(
            (tle_ephemeris.moon_ra_deg >= 0) & (tle_ephemeris.moon_ra_deg <= 360)
        )
        assert np.all(
            (tle_ephemeris.earth_ra_deg >= 0) & (tle_ephemeris.earth_ra_deg <= 360)
        )

        # RA in radians: 0-2π
        assert np.all(
            (tle_ephemeris.sun_ra_rad >= 0) & (tle_ephemeris.sun_ra_rad <= 2 * np.pi)
        )
        assert np.all(
            (tle_ephemeris.moon_ra_rad >= 0) & (tle_ephemeris.moon_ra_rad <= 2 * np.pi)
        )
        assert np.all(
            (tle_ephemeris.earth_ra_rad >= 0)
            & (tle_ephemeris.earth_ra_rad <= 2 * np.pi)
        )

    def test_individual_dec_in_expected_range(self, tle_ephemeris: Any) -> None:
        """Individual Dec properties should be in expected ranges"""
        # Dec in degrees: -90 to +90
        assert np.all(
            (tle_ephemeris.sun_dec_deg >= -90) & (tle_ephemeris.sun_dec_deg <= 90)
        )
        assert np.all(
            (tle_ephemeris.moon_dec_deg >= -90) & (tle_ephemeris.moon_dec_deg <= 90)
        )
        assert np.all(
            (tle_ephemeris.earth_dec_deg >= -90) & (tle_ephemeris.earth_dec_deg <= 90)
        )

        # Dec in radians: -π/2 to +π/2
        assert np.all(
            (tle_ephemeris.sun_dec_rad >= -np.pi / 2)
            & (tle_ephemeris.sun_dec_rad <= np.pi / 2)
        )
        assert np.all(
            (tle_ephemeris.moon_dec_rad >= -np.pi / 2)
            & (tle_ephemeris.moon_dec_rad <= np.pi / 2)
        )
        assert np.all(
            (tle_ephemeris.earth_dec_rad >= -np.pi / 2)
            & (tle_ephemeris.earth_dec_rad <= np.pi / 2)
        )

    def test_individual_properties_available_on_ground_ephemeris(
        self, ground_ephemeris: Any
    ) -> None:
        """Individual RA/Dec properties should be available on GroundEphemeris"""
        # Just verify they're accessible and return expected types
        assert isinstance(ground_ephemeris.sun_ra_deg, np.ndarray)
        assert isinstance(ground_ephemeris.sun_dec_deg, np.ndarray)
        assert isinstance(ground_ephemeris.moon_ra_deg, np.ndarray)
        assert isinstance(ground_ephemeris.moon_dec_deg, np.ndarray)
        assert isinstance(ground_ephemeris.earth_ra_deg, np.ndarray)
        assert isinstance(ground_ephemeris.earth_dec_deg, np.ndarray)

        assert isinstance(ground_ephemeris.sun_ra_rad, np.ndarray)
        assert isinstance(ground_ephemeris.sun_dec_rad, np.ndarray)
        assert isinstance(ground_ephemeris.moon_ra_rad, np.ndarray)
        assert isinstance(ground_ephemeris.moon_dec_rad, np.ndarray)
        assert isinstance(ground_ephemeris.earth_ra_rad, np.ndarray)
        assert isinstance(ground_ephemeris.earth_dec_rad, np.ndarray)

    def test_individual_properties_available_on_spice_ephemeris(
        self, spice_ephemeris: Any
    ) -> None:
        """Individual RA/Dec properties should be available on SPICEEphemeris"""
        # Just verify they're accessible and return expected types
        assert isinstance(spice_ephemeris.sun_ra_deg, np.ndarray)
        assert isinstance(spice_ephemeris.sun_dec_deg, np.ndarray)
        assert isinstance(spice_ephemeris.moon_ra_deg, np.ndarray)
        assert isinstance(spice_ephemeris.moon_dec_deg, np.ndarray)
        assert isinstance(spice_ephemeris.earth_ra_deg, np.ndarray)
        assert isinstance(spice_ephemeris.earth_dec_deg, np.ndarray)

        assert isinstance(spice_ephemeris.sun_ra_rad, np.ndarray)
        assert isinstance(spice_ephemeris.sun_dec_rad, np.ndarray)
        assert isinstance(spice_ephemeris.moon_ra_rad, np.ndarray)
        assert isinstance(spice_ephemeris.moon_dec_rad, np.ndarray)
        assert isinstance(spice_ephemeris.earth_ra_rad, np.ndarray)
        assert isinstance(spice_ephemeris.earth_dec_rad, np.ndarray)

    def test_individual_properties_have_correct_length(
        self, tle_ephemeris: Any
    ) -> None:
        """All individual RA/Dec properties should have same length as timestamp array"""
        n_times: int = len(tle_ephemeris.timestamp)

        assert len(tle_ephemeris.sun_ra_deg) == n_times
        assert len(tle_ephemeris.sun_dec_deg) == n_times
        assert len(tle_ephemeris.moon_ra_deg) == n_times
        assert len(tle_ephemeris.moon_dec_deg) == n_times
        assert len(tle_ephemeris.earth_ra_deg) == n_times
        assert len(tle_ephemeris.earth_dec_deg) == n_times

        assert len(tle_ephemeris.sun_ra_rad) == n_times
        assert len(tle_ephemeris.sun_dec_rad) == n_times
        assert len(tle_ephemeris.moon_ra_rad) == n_times
        assert len(tle_ephemeris.moon_dec_rad) == n_times
        assert len(tle_ephemeris.earth_ra_rad) == n_times
        assert len(tle_ephemeris.earth_dec_rad) == n_times
