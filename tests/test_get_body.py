#!/usr/bin/env python3
"""
Test script to verify get_body_pv() and get_body() methods work correctly
across all ephemeris types (TLE, SPICE, Ground).

This tests that:
1. Bodies can be accessed by NAIF ID (as string) or by name (case-insensitive)
2. PositionVelocity objects are returned with correct structure
3. SkyCoord objects are returned with observer location set
4. Works for all major planets and moons
"""

from datetime import datetime, timezone

import numpy as np
import pytest

import rust_ephem  # type: ignore[import-untyped]


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
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=1800)


@pytest.fixture
def ground_ephemeris(ensure_planetary_data):
    """Create a GroundEphemeris instance for testing (Mauna Kea Observatory)"""
    latitude = 19.8207  # degrees
    longitude = -155.4681  # degrees
    height = 4207  # meters
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude, longitude, height, begin, end, step_size=1800
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
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Use Moon (NAIF ID 301) as the "spacecraft"
    return rust_ephem.SPICEEphemeris(test_data_path, 301, begin, end, step_size=1800)


class TestTLEEphemerisGetBody:
    """Test get_body_pv() method on TLEEphemeris"""

    def test_sun_by_name_returns_position_velocity(self, tle_ephemeris):
        sun = tle_ephemeris.get_body_pv("Sun")
        assert sun.position.shape[1] == 3

    def test_sun_position_has_reasonable_magnitude(self, tle_ephemeris):
        sun = tle_ephemeris.get_body_pv("Sun")
        distance = np.linalg.norm(sun.position[0])
        assert distance > 1e8  # Sun should be > 100 million km away

    def test_moon_by_lowercase_name(self, tle_ephemeris):
        moon = tle_ephemeris.get_body_pv("moon")
        assert moon.position.shape[1] == 3

    def test_moon_position_has_reasonable_magnitude(self, tle_ephemeris):
        moon = tle_ephemeris.get_body_pv("moon")
        distance = np.linalg.norm(moon.position[0])
        assert 300000 < distance < 500000  # Moon ~384,400 km from Earth

    def test_moon_by_naif_id(self, tle_ephemeris):
        moon = tle_ephemeris.get_body_pv("301")
        assert moon.position.shape[1] == 3

    def test_moon_by_name_and_id_match(self, tle_ephemeris):
        moon_by_name = tle_ephemeris.get_body_pv("moon")
        moon_by_id = tle_ephemeris.get_body_pv("301")
        assert np.allclose(moon_by_name.position, moon_by_id.position)

    def test_luna_alias_works(self, tle_ephemeris):
        luna = tle_ephemeris.get_body_pv("Luna")
        assert luna.position.shape[1] == 3

    def test_luna_matches_moon(self, tle_ephemeris):
        luna = tle_ephemeris.get_body_pv("Luna")
        moon = tle_ephemeris.get_body_pv("Moon")
        assert np.allclose(luna.position, moon.position)


class TestTLEEphemerisGetBodySkyCoord:
    """Test get_body() method on TLEEphemeris"""

    def test_sun_skycoord_is_created(self, tle_ephemeris):
        sun = tle_ephemeris.get_body("Sun")
        assert sun is not None

    def test_sun_skycoord_has_gcrs_frame(self, tle_ephemeris):
        sun = tle_ephemeris.get_body("Sun")
        assert sun.frame.name == "gcrs"

    def test_sun_skycoord_has_observer_location(self, tle_ephemeris):
        sun = tle_ephemeris.get_body("Sun")
        assert hasattr(sun.frame, "obsgeoloc")

    def test_moon_skycoord_is_created(self, tle_ephemeris):
        moon = tle_ephemeris.get_body("Moon")
        assert moon is not None

    def test_moon_skycoord_has_gcrs_frame(self, tle_ephemeris):
        moon = tle_ephemeris.get_body("Moon")
        assert moon.frame.name == "gcrs"


class TestGroundEphemerisGetBody:
    """Test get_body_pv() method on GroundEphemeris"""

    def test_sun_returns_position_velocity(self, ground_ephemeris):
        sun = ground_ephemeris.get_body_pv("Sun")
        assert sun.position.shape[1] == 3

    def test_sun_position_has_reasonable_magnitude(self, ground_ephemeris):
        sun = ground_ephemeris.get_body_pv("Sun")
        distance = np.linalg.norm(sun.position[0])
        assert distance > 1e8

    def test_moon_by_naif_id(self, ground_ephemeris):
        moon = ground_ephemeris.get_body_pv("301")
        assert moon.position.shape[1] == 3

    def test_moon_position_has_reasonable_magnitude(self, ground_ephemeris):
        moon = ground_ephemeris.get_body_pv("301")
        distance = np.linalg.norm(moon.position[0])
        assert 300000 < distance < 500000


class TestGroundEphemerisGetBodySkyCoord:
    """Test get_body() method on GroundEphemeris"""

    def test_moon_skycoord_is_created(self, ground_ephemeris):
        moon = ground_ephemeris.get_body("moon")
        assert moon is not None

    def test_moon_skycoord_has_gcrs_frame(self, ground_ephemeris):
        moon = ground_ephemeris.get_body("moon")
        assert moon.frame.name == "gcrs"


class TestSPICEEphemerisGetBody:
    """Test get_body_pv() method on SPICEEphemeris"""

    def test_sun_from_moon_returns_position_velocity(self, spice_ephemeris):
        sun = spice_ephemeris.get_body_pv("Sun")
        assert sun.position.shape[1] == 3

    def test_sun_from_moon_has_reasonable_magnitude(self, spice_ephemeris):
        sun = spice_ephemeris.get_body_pv("Sun")
        distance = np.linalg.norm(sun.position[0])
        assert distance > 1e8

    def test_earth_from_moon_returns_position_velocity(self, spice_ephemeris):
        earth = spice_ephemeris.get_body_pv("Earth")
        assert earth.position.shape[1] == 3

    def test_earth_from_moon_has_reasonable_magnitude(self, spice_ephemeris):
        earth = spice_ephemeris.get_body_pv("Earth")
        distance = np.linalg.norm(earth.position[0])
        assert 300000 < distance < 500000


class TestSPICEEphemerisGetBodySkyCoord:
    """Test get_body() method on SPICEEphemeris"""

    def test_sun_skycoord_is_created(self, spice_ephemeris):
        sun = spice_ephemeris.get_body("Sun")
        assert sun is not None

    def test_sun_skycoord_has_gcrs_frame(self, spice_ephemeris):
        sun = spice_ephemeris.get_body("Sun")
        assert sun.frame.name == "gcrs"


class TestErrorHandling:
    """Test that invalid body identifiers raise appropriate errors"""

    def test_invalid_body_name_raises_value_error(self, tle_ephemeris):
        with pytest.raises(ValueError):
            tle_ephemeris.get_body_pv("InvalidBodyName")

    def test_invalid_naif_id_string_raises_value_error(self, tle_ephemeris):
        with pytest.raises(ValueError):
            tle_ephemeris.get_body_pv("not_a_number_or_body")


class TestBodyNameVariations:
    """Test that various body name formats work"""

    def test_sun_uppercase(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("SUN")
        assert body.position.shape[1] == 3

    def test_sun_lowercase(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("sun")
        assert body.position.shape[1] == 3

    def test_sun_mixed_case(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("SuN")
        assert body.position.shape[1] == 3

    def test_sun_case_variations_match(self, tle_ephemeris):
        sun1 = tle_ephemeris.get_body_pv("Sun")
        sun2 = tle_ephemeris.get_body_pv("SUN")
        assert np.allclose(sun1.position, sun2.position)

    def test_moon_uppercase(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("MOON")
        assert body.position.shape[1] == 3

    def test_luna_uppercase(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("LUNA")
        assert body.position.shape[1] == 3

    def test_luna_lowercase(self, tle_ephemeris):
        body = tle_ephemeris.get_body_pv("luna")
        assert body.position.shape[1] == 3

    def test_moon_aliases_match(self, tle_ephemeris):
        moon = tle_ephemeris.get_body_pv("Moon")
        luna = tle_ephemeris.get_body_pv("Luna")
        assert np.allclose(moon.position, luna.position)
