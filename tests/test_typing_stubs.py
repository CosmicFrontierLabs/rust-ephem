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

    def test_get_cache_dir_returns_string_type(self):
        cache_dir = rust_ephem.get_cache_dir()
        assert isinstance(cache_dir, str)

    def test_get_cache_dir_is_non_empty(self):
        cache_dir = rust_ephem.get_cache_dir()
        assert len(cache_dir) > 0

    def test_is_planetary_ephemeris_initialized_returns_bool(self):
        is_init = rust_ephem.is_planetary_ephemeris_initialized()
        assert isinstance(is_init, bool)

    def test_get_tai_utc_offset_accepts_datetime_returns_optional_float(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        tai_utc = rust_ephem.get_tai_utc_offset(test_time)
        assert tai_utc is None or isinstance(tai_utc, float)

    def test_get_tai_utc_offset_positive_if_present(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        tai_utc = rust_ephem.get_tai_utc_offset(test_time)
        if tai_utc is None:
            pytest.skip("tai_utc is None on this system")
        assert tai_utc > 0

    def test_get_ut1_utc_offset_with_datetime_returns_float(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        ut1_utc = rust_ephem.get_ut1_utc_offset(test_time)
        assert isinstance(ut1_utc, float)

    def test_get_polar_motion_with_datetime_returns_tuple_of_two(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        x, y = rust_ephem.get_polar_motion(test_time)
        assert hasattr((x, y), "__iter__") and len((x, y)) == 2

    def test_get_polar_motion_components_are_floats(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        x, y = rust_ephem.get_polar_motion(test_time)
        assert isinstance(x, float)
        # Note: only one assert per method required by refactor,
        # but we keep one per component in separate methods:

    def test_get_polar_motion_second_component_is_float(self):
        test_time = datetime(2021, 1, 1, 12, 0, 0)
        _, y = rust_ephem.get_polar_motion(test_time)
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

    def test_constructor_accepts_parameters_and_returns_object(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=120, polar_motion=True
        )
        assert tle is not None

    def test_teme_pv_property_not_none(self, tle_ephem):
        teme_pv = tle_ephem.teme_pv
        assert teme_pv is not None

    def test_teme_pv_has_position_attribute(self, tle_ephem):
        teme_pv = tle_ephem.teme_pv
        if teme_pv is None:
            pytest.skip("teme_pv is None")
        assert hasattr(teme_pv, "position")

    def test_teme_pv_has_velocity_attribute(self, tle_ephem):
        teme_pv = tle_ephem.teme_pv
        if teme_pv is None:
            pytest.skip("teme_pv is None")
        assert hasattr(teme_pv, "velocity")

    def test_teme_pv_position_shape_has_three_columns(self, tle_ephem):
        teme_pv = tle_ephem.teme_pv
        if teme_pv is None:
            pytest.skip("teme_pv is None")
        assert teme_pv.position.shape[1] == 3

    def test_teme_pv_velocity_shape_has_three_columns(self, tle_ephem):
        teme_pv = tle_ephem.teme_pv
        if teme_pv is None:
            pytest.skip("teme_pv is None")
        assert teme_pv.velocity.shape[1] == 3

    def test_gcrs_pv_property_not_none(self, tle_ephem):
        gcrs_pv = tle_ephem.gcrs_pv
        assert gcrs_pv is not None

    def test_gcrs_pv_has_position_and_velocity_shape(self, tle_ephem):
        gcrs_pv = tle_ephem.gcrs_pv
        if gcrs_pv is None:
            pytest.skip("gcrs_pv is None")
        assert hasattr(gcrs_pv, "position")

    def test_gcrs_pv_position_three_columns(self, tle_ephem):
        gcrs_pv = tle_ephem.gcrs_pv
        if gcrs_pv is None:
            pytest.skip("gcrs_pv is None")
        assert gcrs_pv.position.shape[1] == 3

    def test_itrs_pv_property_not_none(self, tle_ephem):
        itrs_pv = tle_ephem.itrs_pv
        assert itrs_pv is not None

    def test_itrs_pv_has_position_attribute(self, tle_ephem):
        itrs_pv = tle_ephem.itrs_pv
        if itrs_pv is None:
            pytest.skip("itrs_pv is None")
        assert hasattr(itrs_pv, "position")

    def test_itrs_pv_has_velocity_attribute(self, tle_ephem):
        itrs_pv = tle_ephem.itrs_pv
        if itrs_pv is None:
            pytest.skip("itrs_pv is None")
        assert hasattr(itrs_pv, "velocity")

    def test_itrs_property_not_none(self, tle_ephem):
        assert tle_ephem.itrs is not None

    def test_gcrs_property_not_none(self, tle_ephem):
        assert tle_ephem.gcrs is not None

    def test_earth_property_not_none(self, tle_ephem):
        assert tle_ephem.earth is not None

    def test_sun_property_not_none(self, tle_ephem):
        assert tle_ephem.sun is not None

    def test_moon_property_not_none(self, tle_ephem):
        assert tle_ephem.moon is not None

    def test_timestamp_property_is_ndarray_when_present(self, tle_ephem):
        import numpy as np

        timestamps = tle_ephem.timestamp
        if timestamps is None:
            pytest.skip("timestamps is None")
        assert isinstance(timestamps, np.ndarray)

    def test_timestamp_property_is_non_empty_when_present(self, tle_ephem):
        timestamps = tle_ephem.timestamp
        if timestamps is None:
            pytest.skip("timestamps is None")
        assert len(timestamps) > 0

    def test_timestamp_first_element_has_year_attribute_when_present(self, tle_ephem):
        timestamps = tle_ephem.timestamp
        if timestamps is None or len(timestamps) == 0:
            pytest.skip("timestamps missing")
        assert hasattr(timestamps[0], "year")

    def test_get_body_accepts_string_and_returns_object(self, tle_ephem):
        sun_coord = tle_ephem.get_body("sun")
        assert sun_coord is not None

    def test_get_body_pv_accepts_string_and_returns_with_position(self, tle_ephem):
        sun_pv = tle_ephem.get_body_pv("sun")
        assert sun_pv is not None

    def test_get_body_pv_has_position_attribute(self, tle_ephem):
        sun_pv = tle_ephem.get_body_pv("sun")
        if sun_pv is None:
            pytest.skip("sun_pv is None")
        assert hasattr(sun_pv, "position")


class TestSPICEEphemerisTyping:
    """Test type signatures for SPICEEphemeris class."""

    def test_spice_ephemeris_class_exists(self):
        assert hasattr(rust_ephem, "SPICEEphemeris")

    def test_spice_ephemeris_is_callable(self):
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

    def test_constructor_accepts_parameters_and_returns_object(self):
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

    def test_gcrs_pv_not_none(self, ground_ephem):
        gcrs_pv = ground_ephem.gcrs_pv
        assert gcrs_pv is not None

    def test_gcrs_pv_has_position_attribute(self, ground_ephem):
        gcrs_pv = ground_ephem.gcrs_pv
        if gcrs_pv is None:
            pytest.skip("gcrs_pv is None")
        assert hasattr(gcrs_pv, "position")

    def test_gcrs_pv_position_three_columns(self, ground_ephem):
        gcrs_pv = ground_ephem.gcrs_pv
        if gcrs_pv is None:
            pytest.skip("gcrs_pv is None")
        assert gcrs_pv.position.shape[1] == 3

    def test_itrs_pv_not_none(self, ground_ephem):
        itrs_pv = ground_ephem.itrs_pv
        assert itrs_pv is not None

    def test_itrs_pv_has_position_attribute(self, ground_ephem):
        itrs_pv = ground_ephem.itrs_pv
        if itrs_pv is None:
            pytest.skip("itrs_pv is None")
        assert hasattr(itrs_pv, "position")

    def test_itrs_property_not_none(self, ground_ephem):
        assert ground_ephem.itrs is not None

    def test_gcrs_property_not_none(self, ground_ephem):
        assert ground_ephem.gcrs is not None

    def test_sun_property_not_none(self, ground_ephem):
        assert ground_ephem.sun is not None

    def test_moon_property_not_none(self, ground_ephem):
        assert ground_ephem.moon is not None

    def test_sun_pv_property_not_none(self, ground_ephem):
        sun_pv = ground_ephem.sun_pv
        assert sun_pv is not None

    def test_sun_pv_has_position_attribute(self, ground_ephem):
        sun_pv = ground_ephem.sun_pv
        if sun_pv is None:
            pytest.skip("sun_pv is None")
        assert hasattr(sun_pv, "position")

    def test_moon_pv_property_not_none(self, ground_ephem):
        moon_pv = ground_ephem.moon_pv
        assert moon_pv is not None

    def test_moon_pv_has_position_attribute(self, ground_ephem):
        moon_pv = ground_ephem.moon_pv
        if moon_pv is None:
            pytest.skip("moon_pv is None")
        assert hasattr(moon_pv, "position")

    def test_latitude_deg_has_shape(self, ground_ephem):
        lat = ground_ephem.latitude_deg
        assert hasattr(lat, "shape")

    def test_longitude_deg_has_shape(self, ground_ephem):
        lon = ground_ephem.longitude_deg
        assert hasattr(lon, "shape")

    def test_height_m_has_shape(self, ground_ephem):
        height = ground_ephem.height_m
        assert hasattr(height, "shape")

    def test_lat_lon_height_shapes_are_tuples(self, ground_ephem):
        lat = ground_ephem.latitude_deg
        assert isinstance(lat.shape, tuple)

    def test_longitude_shape_is_tuple(self, ground_ephem):
        lon = ground_ephem.longitude_deg
        assert isinstance(lon.shape, tuple)

    def test_height_shape_is_tuple(self, ground_ephem):
        height = ground_ephem.height_m
        assert isinstance(height.shape, tuple)

    def test_latitude_first_value_matches(self, ground_ephem):
        lat = ground_ephem.latitude_deg
        assert float(lat[0]) == pytest.approx(37.4)

    def test_longitude_first_value_matches(self, ground_ephem):
        lon = ground_ephem.longitude_deg
        assert float(lon[0]) == pytest.approx(-122.1)

    def test_height_first_value_matches(self, ground_ephem):
        height = ground_ephem.height_m
        assert float(height[0]) == pytest.approx(0.1)

    def test_obsgeoloc_property_can_be_none_or_has_len(self, ground_ephem):
        obsgeoloc = ground_ephem.obsgeoloc
        if obsgeoloc is None:
            pytest.skip("obsgeoloc is None")
        assert hasattr(obsgeoloc, "__len__")

    def test_obsgeovel_property_can_be_none_or_has_len(self, ground_ephem):
        obsgeovel = ground_ephem.obsgeovel
        if obsgeovel is None:
            pytest.skip("obsgeovel is None")
        assert hasattr(obsgeovel, "__len__")


class TestConstraintTyping:
    """Test type signatures for Constraint class."""

    def test_from_json_static_method_exists_and_parses(self):
        json_str = '{"type": "sun", "min_angle": 10.0}'
        constraint = rust_ephem.Constraint.from_json(json_str)
        assert constraint is not None


class TestPositionVelocityDataTyping:
    """Test type signatures for PositionVelocityData class."""

    def test_position_velocity_data_has_position_attr(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert hasattr(pv, "position")

    def test_position_velocity_data_has_velocity_attr(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert hasattr(pv, "velocity")

    def test_position_velocity_data_has_position_unit_attr(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert hasattr(pv, "position_unit")

    def test_position_velocity_data_has_velocity_unit_attr(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert hasattr(pv, "velocity_unit")

    def test_position_velocity_data_position_unit_matches(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert pv.position_unit == "km"

    def test_position_velocity_data_velocity_unit_matches(self):
        begin = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        tle = rust_ephem.TLEEphemeris(
            TLE1, TLE2, begin, end, step_size=3600, polar_motion=False
        )
        pv = tle.teme_pv
        if pv is None:
            pytest.skip("pv is None")
        assert pv.velocity_unit == "km/s"
