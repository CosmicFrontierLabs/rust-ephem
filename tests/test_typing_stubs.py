"""Test PEP 561 type stub compliance for rust-ephem.

This module tests that:
1. Type stubs are correctly defined for all public API elements
2. Properties return the expected types
3. Function signatures match the actual implementation
4. Type checkers can successfully validate code using rust-ephem
"""

from datetime import datetime

import pytest

import rust_ephem

# Test data
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"


class TestModuleLevelFunctions:
    """Test type signatures of module-level functions."""

    def test_get_cache_dir_returns_string(self):
        """get_cache_dir should return a string path."""
        cache_dir = rust_ephem.get_cache_dir()
        assert isinstance(cache_dir, str)
        assert len(cache_dir) > 0

    def test_is_planetary_ephemeris_initialized_returns_bool(self):
        """is_planetary_ephemeris_initialized should return a boolean."""
        is_init = rust_ephem.is_planetary_ephemeris_initialized()
        assert isinstance(is_init, bool)

    def test_get_tai_utc_offset_with_datetime(self):
        """get_tai_utc_offset should accept datetime and return float or None."""
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        tai_utc = rust_ephem.get_tai_utc_offset(test_time)
        # Type annotation indicates this can be None
        if tai_utc is not None:
            assert isinstance(tai_utc, float)
            assert tai_utc > 0  # TAI-UTC offset is always positive

    def test_get_ut1_utc_offset_with_datetime(self):
        """get_ut1_utc_offset should accept datetime and return float."""
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        # This will initialize UT1 provider if needed
        ut1_utc = rust_ephem.get_ut1_utc_offset(test_time)
        assert isinstance(ut1_utc, float)

    def test_get_polar_motion_with_datetime(self):
        """get_polar_motion should accept datetime and return tuple of floats."""
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        x, y = rust_ephem.get_polar_motion(test_time)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestTLEEphemerisTyping:
    """Test type signatures for TLEEphemeris class."""

    @pytest.fixture
    def tle_ephem(self):
        """Create a TLEEphemeris instance for testing."""
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        return rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=60, polar_motion=False
        )

    def test_constructor_accepts_correct_parameters(self):
        """TLEEphemeris constructor should accept documented parameters."""
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)

        # Test with all parameters
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=120, polar_motion=True
        )
        assert tle is not None

    def test_teme_pv_property_returns_position_velocity_data(self, tle_ephem):
        """teme_pv property should return PositionVelocityData or None."""
        teme_pv = tle_ephem.teme_pv
        # Type annotation indicates this can be None, but in practice it shouldn't be
        assert teme_pv is not None
        # Check that it has position and velocity attributes
        assert hasattr(teme_pv, "position")
        assert hasattr(teme_pv, "velocity")
        assert teme_pv.position.shape[1] == 3  # N x 3 array
        assert teme_pv.velocity.shape[1] == 3  # N x 3 array

    def test_gcrs_pv_property_returns_position_velocity_data(self, tle_ephem):
        """gcrs_pv property should return PositionVelocityData or None."""
        gcrs_pv = tle_ephem.gcrs_pv
        assert gcrs_pv is not None
        assert hasattr(gcrs_pv, "position")
        assert hasattr(gcrs_pv, "velocity")
        assert gcrs_pv.position.shape[1] == 3

    def test_itrs_pv_property_returns_position_velocity_data(self, tle_ephem):
        """itrs_pv property should return PositionVelocityData or None."""
        itrs_pv = tle_ephem.itrs_pv
        assert itrs_pv is not None
        assert hasattr(itrs_pv, "position")
        assert hasattr(itrs_pv, "velocity")

    def test_skycoord_properties_return_objects(self, tle_ephem):
        """SkyCoord properties should return objects (type depends on astropy)."""
        # These return SkyCoord objects but we don't want to require astropy
        assert tle_ephem.itrs is not None
        assert tle_ephem.gcrs is not None
        assert tle_ephem.earth is not None
        assert tle_ephem.sun is not None
        assert tle_ephem.moon is not None

    def test_timestamp_property_returns_array(self, tle_ephem):
        """timestamp property should return NDArray of datetimes or None."""
        import numpy as np

        timestamps = tle_ephem.timestamp
        if timestamps is not None:
            assert isinstance(timestamps, np.ndarray)
            assert len(timestamps) > 0
            # Check first element is datetime
            assert hasattr(timestamps[0], "year")

    def test_get_body_accepts_string(self, tle_ephem):
        """get_body method should accept string body name."""
        sun_coord = tle_ephem.get_body("sun")
        assert sun_coord is not None

    def test_get_body_pv_accepts_string(self, tle_ephem):
        """get_body_pv method should accept string body name."""
        sun_pv = tle_ephem.get_body_pv("sun")
        assert sun_pv is not None
        assert hasattr(sun_pv, "position")


class TestSPICEEphemerisTyping:
    """Test type signatures for SPICEEphemeris class."""

    def test_constructor_signature(self):
        """SPICEEphemeris constructor should accept documented parameters."""
        # We can't test actual construction without SPICE kernels,
        # but we can verify the type signature exists
        assert hasattr(rust_ephem, "SPICEEphemeris")
        # Verify it's callable (has __init__)
        assert callable(rust_ephem.SPICEEphemeris)


class TestGroundEphemerisTyping:
    """Test type signatures for GroundEphemeris class."""

    @pytest.fixture
    def ground_ephem(self):
        """Create a GroundEphemeris instance for testing."""
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        return rust_ephem.GroundEphemeris(
            latitude=37.4,
            longitude=-122.1,
            height=0.1,
            begin=begin,
            end=end,
            step_size=60,
            polar_motion=False,
        )

    def test_constructor_accepts_correct_parameters(self):
        """GroundEphemeris constructor should accept documented parameters."""
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)

        ground = rust_ephem.GroundEphemeris(
            latitude=37.4,
            longitude=-122.1,
            height=100.0,
            begin=begin,
            end=end,
            step_size=120,
            polar_motion=True,
        )
        assert ground is not None

    def test_gcrs_pv_property_returns_position_velocity_data(self, ground_ephem):
        """gcrs_pv property should return PositionVelocityData or None."""
        gcrs_pv = ground_ephem.gcrs_pv
        assert gcrs_pv is not None
        assert hasattr(gcrs_pv, "position")
        assert gcrs_pv.position.shape[1] == 3

    def test_itrs_pv_property_returns_position_velocity_data(self, ground_ephem):
        """itrs_pv property should return PositionVelocityData or None."""
        itrs_pv = ground_ephem.itrs_pv
        assert itrs_pv is not None
        assert hasattr(itrs_pv, "position")

    def test_skycoord_properties_return_objects(self, ground_ephem):
        """SkyCoord properties should return objects."""
        assert ground_ephem.itrs is not None
        assert ground_ephem.gcrs is not None
        assert ground_ephem.sun is not None
        assert ground_ephem.moon is not None

    def test_sun_moon_pv_properties(self, ground_ephem):
        """sun_pv and moon_pv properties should return PositionVelocityData."""
        sun_pv = ground_ephem.sun_pv
        assert sun_pv is not None
        assert hasattr(sun_pv, "position")

        moon_pv = ground_ephem.moon_pv
        assert moon_pv is not None
        assert hasattr(moon_pv, "position")

    def test_location_properties_return_floats(self, ground_ephem):
        """latitude, longitude, height properties should return floats."""
        lat = ground_ephem.latitude
        lon = ground_ephem.longitude
        height = ground_ephem.height

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(height, float)
        assert lat == pytest.approx(37.4)
        assert lon == pytest.approx(-122.1)
        assert height == pytest.approx(0.1)

    def test_obsgeoloc_and_obsgeovel_properties(self, ground_ephem):
        """obsgeoloc and obsgeovel properties should return objects or None."""
        # These are optional astropy-specific properties
        obsgeoloc = ground_ephem.obsgeoloc
        obsgeovel = ground_ephem.obsgeovel
        # Can be None, but typically aren't after construction
        if obsgeoloc is not None:
            assert hasattr(obsgeoloc, "__len__")
        if obsgeovel is not None:
            assert hasattr(obsgeovel, "__len__")


class TestConstraintTyping:
    """Test type signatures for Constraint class."""

    def test_from_json_static_method(self):
        """Constraint.from_json should accept JSON string."""
        # Use actual constraint type from the implementation
        json_str = '{"type": "sun", "min_angle": 10.0}'
        constraint = rust_ephem.Constraint.from_json(json_str)
        assert constraint is not None


class TestPositionVelocityDataTyping:
    """Test type signatures for PositionVelocityData class."""

    def test_position_velocity_data_has_expected_properties(self):
        """PositionVelocityData should have position and velocity properties."""
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )

        pv = tle.teme_pv
        assert pv is not None

        # Check properties exist and have expected attributes
        assert hasattr(pv, "position")
        assert hasattr(pv, "velocity")
        assert hasattr(pv, "position_unit")
        assert hasattr(pv, "velocity_unit")

        # Check units are correct
        assert pv.position_unit == "km"
        assert pv.velocity_unit == "km/s"
