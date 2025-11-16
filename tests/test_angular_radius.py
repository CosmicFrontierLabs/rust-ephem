#!/usr/bin/env python3
"""
Test script to verify angular radius properties for Sun, Moon, and Earth.

This tests that:
1. All three properties exist for each body (Quantity, degrees, radians)
2. Properties return correct types (Quantity vs numpy arrays)
3. Degree and radian conversions are consistent
4. Values are within expected ranges
5. Properties are available on all ephemeris types (TLE, SPICE, Ground)
6. Properties are cached for performance
"""

from datetime import datetime, timezone

import numpy as np
import pytest

import rust_ephem  # type: ignore[import-untyped]


# Test fixtures for different ephemeris types
@pytest.fixture(scope="module")
def ensure_planetary_data():
    """Ensure planetary ephemeris is loaded once for all tests"""
    # Use local test data file if available, otherwise download once
    import os

    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test_data", "de440s.bsp"
    )

    # If file exists locally, use it without downloading
    if os.path.exists(test_data_path):
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=False
        )
    else:
        # File doesn't exist, allow download (will happen once per machine)
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=True
        )


@pytest.fixture
def tle_ephemeris(ensure_planetary_data):
    """Create a TLEEphemeris instance for testing"""
    tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
    tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=600)


@pytest.fixture
def ground_ephemeris(ensure_planetary_data):
    """Create a GroundEphemeris instance for testing (Mauna Kea Observatory)"""
    latitude = 19.8207  # degrees
    longitude = -155.4681  # degrees
    height = 4207  # meters
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude, longitude, height, begin, end, step_size=600
    )


@pytest.fixture
def spice_ephemeris(ensure_planetary_data):
    """Create a SPICEEphemeris instance for testing"""
    import os

    # Use the same test data path as ensure_planetary_data
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test_data", "de440s.bsp"
    )

    # If file doesn't exist in test_data, try cache directory as fallback
    if not os.path.exists(test_data_path):
        test_data_path = os.path.expanduser("~/.cache/rust_ephem/de440s.bsp")

    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    # Use Moon (NAIF ID 301) as the observer
    return rust_ephem.SPICEEphemeris(test_data_path, 301, begin, end, step_size=600)


class TestSunAngularRadius:
    """Test Sun angular radius properties across all ephemeris types"""

    def test_sun_radius_quantity_returns_astropy_quantity(self, tle_ephemeris):
        """sun_radius should return an astropy Quantity"""
        sun_radius = tle_ephemeris.sun_radius
        # Check it has the Quantity interface
        assert hasattr(sun_radius, "unit")
        assert hasattr(sun_radius, "value")
        assert str(sun_radius.unit) == "deg"

    def test_sun_radius_deg_returns_numpy_array(self, tle_ephemeris):
        """sun_radius_deg should return a numpy array"""
        sun_radius_deg = tle_ephemeris.sun_radius_deg
        assert isinstance(sun_radius_deg, np.ndarray)
        assert sun_radius_deg.dtype == np.float64

    def test_sun_radius_rad_returns_numpy_array(self, tle_ephemeris):
        """sun_radius_rad should return a numpy array"""
        sun_radius_rad = tle_ephemeris.sun_radius_rad
        assert isinstance(sun_radius_rad, np.ndarray)
        assert sun_radius_rad.dtype == np.float64

    def test_sun_radius_deg_and_rad_conversion(self, tle_ephemeris):
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.sun_radius_deg
        rad = tle_ephemeris.sun_radius_rad
        # Convert radians to degrees and compare
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_sun_radius_quantity_matches_deg(self, tle_ephemeris):
        """Quantity value should match degrees array"""
        quantity = tle_ephemeris.sun_radius
        deg = tle_ephemeris.sun_radius_deg
        assert np.allclose(quantity.value, deg, rtol=1e-10)

    def test_sun_radius_in_expected_range(self, tle_ephemeris):
        """Sun angular radius should be around 0.25-0.28 degrees"""
        deg = tle_ephemeris.sun_radius_deg
        # Sun's angular diameter varies from 0.524-0.542 degrees, so radius ~0.262-0.271
        assert np.all((deg > 0.25) & (deg < 0.28))

    def test_sun_radius_available_on_ground_ephemeris(self, ground_ephemeris):
        """Sun radius properties should work on GroundEphemeris"""
        assert hasattr(ground_ephemeris.sun_radius, "unit")
        assert isinstance(ground_ephemeris.sun_radius_deg, np.ndarray)
        assert isinstance(ground_ephemeris.sun_radius_rad, np.ndarray)

    def test_sun_radius_available_on_spice_ephemeris(self, spice_ephemeris):
        """Sun radius properties should work on SPICEEphemeris"""
        assert hasattr(spice_ephemeris.sun_radius, "unit")
        assert isinstance(spice_ephemeris.sun_radius_deg, np.ndarray)
        assert isinstance(spice_ephemeris.sun_radius_rad, np.ndarray)


class TestMoonAngularRadius:
    """Test Moon angular radius properties across all ephemeris types"""

    def test_moon_radius_quantity_returns_astropy_quantity(self, tle_ephemeris):
        """moon_radius should return an astropy Quantity"""
        moon_radius = tle_ephemeris.moon_radius
        assert hasattr(moon_radius, "unit")
        assert hasattr(moon_radius, "value")
        assert str(moon_radius.unit) == "deg"

    def test_moon_radius_deg_returns_numpy_array(self, tle_ephemeris):
        """moon_radius_deg should return a numpy array"""
        moon_radius_deg = tle_ephemeris.moon_radius_deg
        assert isinstance(moon_radius_deg, np.ndarray)
        assert moon_radius_deg.dtype == np.float64

    def test_moon_radius_rad_returns_numpy_array(self, tle_ephemeris):
        """moon_radius_rad should return a numpy array"""
        moon_radius_rad = tle_ephemeris.moon_radius_rad
        assert isinstance(moon_radius_rad, np.ndarray)
        assert moon_radius_rad.dtype == np.float64

    def test_moon_radius_deg_and_rad_conversion(self, tle_ephemeris):
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.moon_radius_deg
        rad = tle_ephemeris.moon_radius_rad
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_moon_radius_quantity_matches_deg(self, tle_ephemeris):
        """Quantity value should match degrees array"""
        quantity = tle_ephemeris.moon_radius
        deg = tle_ephemeris.moon_radius_deg
        assert np.allclose(quantity.value, deg, rtol=1e-10)

    def test_moon_radius_in_expected_range(self, tle_ephemeris):
        """Moon angular radius should be around 0.24-0.27 degrees"""
        deg = tle_ephemeris.moon_radius_deg
        # Moon's angular diameter varies from 0.49-0.55 degrees, so radius ~0.245-0.275
        assert np.all((deg > 0.24) & (deg < 0.28))

    def test_moon_radius_available_on_ground_ephemeris(self, ground_ephemeris):
        """Moon radius properties should work on GroundEphemeris"""
        assert hasattr(ground_ephemeris.moon_radius, "unit")
        assert isinstance(ground_ephemeris.moon_radius_deg, np.ndarray)
        assert isinstance(ground_ephemeris.moon_radius_rad, np.ndarray)

    def test_moon_radius_available_on_spice_ephemeris(self, spice_ephemeris):
        """Moon radius properties should work on SPICEEphemeris"""
        assert hasattr(spice_ephemeris.moon_radius, "unit")
        assert isinstance(spice_ephemeris.moon_radius_deg, np.ndarray)
        assert isinstance(spice_ephemeris.moon_radius_rad, np.ndarray)


class TestEarthAngularRadius:
    """Test Earth angular radius properties across all ephemeris types"""

    def test_earth_radius_quantity_returns_astropy_quantity(self, tle_ephemeris):
        """earth_radius should return an astropy Quantity"""
        earth_radius = tle_ephemeris.earth_radius
        assert hasattr(earth_radius, "unit")
        assert hasattr(earth_radius, "value")
        assert str(earth_radius.unit) == "deg"

    def test_earth_radius_deg_returns_numpy_array(self, tle_ephemeris):
        """earth_radius_deg should return a numpy array"""
        earth_radius_deg = tle_ephemeris.earth_radius_deg
        assert isinstance(earth_radius_deg, np.ndarray)
        assert earth_radius_deg.dtype == np.float64

    def test_earth_radius_rad_returns_numpy_array(self, tle_ephemeris):
        """earth_radius_rad should return a numpy array"""
        earth_radius_rad = tle_ephemeris.earth_radius_rad
        assert isinstance(earth_radius_rad, np.ndarray)
        assert earth_radius_rad.dtype == np.float64

    def test_earth_radius_deg_and_rad_conversion(self, tle_ephemeris):
        """Degrees and radians should be consistent"""
        deg = tle_ephemeris.earth_radius_deg
        rad = tle_ephemeris.earth_radius_rad
        deg_from_rad = np.degrees(rad)
        assert np.allclose(deg, deg_from_rad, rtol=1e-10)

    def test_earth_radius_quantity_matches_deg(self, tle_ephemeris):
        """Quantity value should match degrees array"""
        quantity = tle_ephemeris.earth_radius
        deg = tle_ephemeris.earth_radius_deg
        assert np.allclose(quantity.value, deg, rtol=1e-10)

    def test_earth_radius_in_expected_range_for_leo(self, tle_ephemeris):
        """Earth angular radius for LEO should be 60-80 degrees"""
        deg = tle_ephemeris.earth_radius_deg
        # ISS is in LEO (~400-450 km altitude), Earth radius ~65-70 degrees
        assert np.all((deg > 60) & (deg < 80))

    def test_earth_radius_near_90_for_ground_observer(self, ground_ephemeris):
        """Earth radius should be ~90 degrees for ground-based observer (fills half the sky)"""
        deg = ground_ephemeris.earth_radius_deg
        # Ground observer is at Earth's surface, so distance to center ≈ Earth radius
        # Angular radius = arcsin(R_earth / R_earth) = arcsin(1) = 90 degrees
        # For Mauna Kea at 4205m altitude, value is ~88.65 degrees
        assert np.all((deg > 85.0) & (deg < 90.1))  # Near 90 degrees

    def test_earth_radius_small_from_moon(self, spice_ephemeris):
        """Earth angular radius from Moon should be small (~1 degree)"""
        deg = spice_ephemeris.earth_radius_deg
        # Moon is ~384,400 km from Earth, Earth radius ~6378 km
        # Angular radius = arcsin(6378/384400) ≈ 0.95 degrees
        assert np.all((deg > 0.9) & (deg < 1.1))

    def test_earth_radius_rad_near_pi_over_2_for_ground_observer(
        self, ground_ephemeris
    ):
        """Earth radius in radians should be ~π/2 for ground-based observer"""
        rad = ground_ephemeris.earth_radius_rad
        # π/2 radians = 90 degrees
        # For Mauna Kea at 4205m altitude, value is ~1.547 radians (88.65°)
        assert np.all(rad > 1.48) and np.all(rad < np.pi / 2 + 0.01)

    def test_earth_radius_quantity_near_90_for_ground_observer(self, ground_ephemeris):
        """Earth radius Quantity should be ~90 degrees for ground-based observer"""
        from astropy import units as u

        radius = ground_ephemeris.earth_radius
        # Should have astropy units
        assert hasattr(radius, "unit")
        # Convert to degrees and check value
        # For Mauna Kea at 4205m altitude, value is ~88.65 degrees
        deg_values = radius.to(u.deg).value
        assert np.all((deg_values > 85.0) & (deg_values < 90.1))


class TestAngularRadiusCaching:
    """Test that angular radius properties are cached for performance"""

    def test_sun_radius_deg_is_cached(self, tle_ephemeris):
        """Multiple accesses should return the same object (cached)"""
        first = tle_ephemeris.sun_radius_deg
        second = tle_ephemeris.sun_radius_deg
        # In Python, if they're the same cached object, they'll have the same id
        assert first is second

    def test_moon_radius_rad_is_cached(self, tle_ephemeris):
        """Multiple accesses should return the same object (cached)"""
        first = tle_ephemeris.moon_radius_rad
        second = tle_ephemeris.moon_radius_rad
        assert first is second

    def test_earth_radius_deg_is_cached(self, tle_ephemeris):
        """Multiple accesses should return the same object (cached)"""
        first = tle_ephemeris.earth_radius_deg
        second = tle_ephemeris.earth_radius_deg
        assert first is second

    def test_caching_improves_performance(self, tle_ephemeris):
        """Cached access should be much faster than initial computation"""
        import time

        # First access (computes and caches)
        _ = tle_ephemeris.sun_radius_deg

        # Time many cached accesses
        start = time.perf_counter()
        for _ in range(1000):
            _ = tle_ephemeris.sun_radius_deg
        elapsed = time.perf_counter() - start

        # 1000 cached accesses should take less than 10ms
        assert elapsed < 0.01


class TestAngularRadiusConsistency:
    """Test consistency between different ephemeris types"""

    def test_sun_radius_similar_for_tle_and_ground(
        self, tle_ephemeris, ground_ephemeris
    ):
        """Sun radius should be similar for LEO satellite and ground observer"""
        tle_sun = tle_ephemeris.sun_radius_deg[0]
        ground_sun = ground_ephemeris.sun_radius_deg[0]
        # Both are essentially at Earth distance from Sun, should be very close
        assert np.isclose(tle_sun, ground_sun, rtol=0.01)

    def test_moon_radius_similar_for_tle_and_ground(
        self, tle_ephemeris, ground_ephemeris
    ):
        """Moon radius should be similar for LEO satellite and ground observer"""
        tle_moon = tle_ephemeris.moon_radius_deg[0]
        ground_moon = ground_ephemeris.moon_radius_deg[0]
        # Both are at similar distance from Moon, should be close
        assert np.isclose(tle_moon, ground_moon, rtol=0.05)

    def test_all_properties_return_same_length_arrays(self, tle_ephemeris):
        """All angular radius arrays should have the same length"""
        sun_deg = tle_ephemeris.sun_radius_deg
        sun_rad = tle_ephemeris.sun_radius_rad
        moon_deg = tle_ephemeris.moon_radius_deg
        moon_rad = tle_ephemeris.moon_radius_rad
        earth_deg = tle_ephemeris.earth_radius_deg
        earth_rad = tle_ephemeris.earth_radius_rad

        lengths = [
            len(sun_deg),
            len(sun_rad),
            len(moon_deg),
            len(moon_rad),
            len(earth_deg),
            len(earth_rad),
        ]
        assert len(set(lengths)) == 1  # All lengths should be the same


class TestAngularRadiusEdgeCases:
    """Test edge cases and error handling"""

    def test_properties_available_for_single_timestamp(self):
        """Properties should work for ephemeris with single timestamp"""
        tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
        tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        eph = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=600)

        assert len(eph.sun_radius_deg) == 1
        assert len(eph.moon_radius_deg) == 1
        assert len(eph.earth_radius_deg) == 1

    def test_properties_available_for_many_timestamps(self):
        """Properties should work for ephemeris with many timestamps"""
        tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
        tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        eph = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=60)

        # 24 hours with 60 second steps = 1441 timestamps
        assert len(eph.sun_radius_deg) == 1441
        assert len(eph.moon_radius_deg) == 1441
        assert len(eph.earth_radius_deg) == 1441

    def test_radians_are_smaller_than_degrees(self, tle_ephemeris):
        """Radians should be numerically smaller than degrees for these angles"""
        sun_deg = tle_ephemeris.sun_radius_deg
        sun_rad = tle_ephemeris.sun_radius_rad
        # For angles < 57.3 degrees, radians < degrees
        assert np.all(sun_rad < sun_deg)
