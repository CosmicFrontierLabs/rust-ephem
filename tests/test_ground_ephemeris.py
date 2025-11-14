"""Integration tests for GroundEphemeris class.

Tests the ground-based observatory ephemeris calculations, including:
- Construction with geodetic coordinates
- ITRS position calculation from lat/lon/alt
- GCRS transformation
- Sun/Moon position calculations
- SkyCoord creation
"""

import datetime

import numpy as np
import pytest

from rust_ephem import GroundEphemeris  # type: ignore[import-untyped]


# Fixtures for common test data
@pytest.fixture
def kitt_peak_obs():
    """Create GroundEphemeris for Kitt Peak Observatory."""
    latitude = 31.9583  # degrees N
    longitude = -111.6  # degrees W
    height = 2096.0  # meters
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600  # 1 hour
    return GroundEphemeris(latitude, longitude, height, begin, end, step_size)


@pytest.fixture
def equator_obs():
    """Create GroundEphemeris at equator on prime meridian."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(0.0, 0.0, 0.0, begin, end, step_size)


@pytest.fixture
def mauna_kea_obs():
    """Create GroundEphemeris for Mauna Kea."""
    latitude = 19.8207
    longitude = -155.468
    height = 4205.0
    begin = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600
    return GroundEphemeris(latitude, longitude, height, begin, end, step_size)


@pytest.fixture
def multi_time_obs():
    """Create GroundEphemeris with multiple time steps."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 6, 0, 0, tzinfo=datetime.timezone.utc)
    step_size = 3600  # 1 hour
    return GroundEphemeris(35.0, -120.0, 500.0, begin, end, step_size)


class TestGroundEphemerisBasicProperties:
    """Test basic properties of GroundEphemeris."""

    def test_preserves_latitude(self, kitt_peak_obs):
        """Test that latitude is preserved."""
        assert kitt_peak_obs.latitude == 31.9583

    def test_preserves_longitude(self, kitt_peak_obs):
        """Test that longitude is preserved."""
        assert kitt_peak_obs.longitude == -111.6

    def test_preserves_height(self, kitt_peak_obs):
        """Test that height is preserved."""
        assert kitt_peak_obs.height == 2096.0


class TestGroundEphemerisTimestamps:
    """Test timestamp generation."""

    def test_timestamp_not_none(self, kitt_peak_obs):
        """Test that timestamps are not None."""
        assert kitt_peak_obs.timestamp is not None

    def test_timestamp_count(self, kitt_peak_obs):
        """Test that timestamp count is correct."""
        assert len(kitt_peak_obs.timestamp) == 2

    def test_timestamp_begin(self, kitt_peak_obs):
        """Test that begin timestamp is correct."""
        begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = kitt_peak_obs.timestamp
        assert times[0].replace(tzinfo=None) == begin.replace(tzinfo=None)

    def test_timestamp_end(self, kitt_peak_obs):
        """Test that end timestamp is correct."""
        end = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        times = kitt_peak_obs.timestamp
        assert times[1].replace(tzinfo=None) == end.replace(tzinfo=None)


class TestGroundEphemerisITRS:
    """Test ITRS coordinate system calculations."""

    def test_itrs_not_none(self, equator_obs):
        """Test that ITRS object is not None."""
        assert equator_obs.itrs_pv is not None

    def test_position_shape(self, equator_obs):
        """Test ITRS position array shape."""
        pos = equator_obs.itrs_pv.position
        assert pos.shape == (1, 3)

    def test_position_x_at_equator(self, equator_obs):
        """Test ITRS X position at equator on prime meridian."""
        pos = equator_obs.itrs_pv.position
        assert np.abs(pos[0, 0] - 6378.137) < 0.01

    def test_position_y_at_equator(self, equator_obs):
        """Test ITRS Y position at equator on prime meridian."""
        pos = equator_obs.itrs_pv.position
        assert np.abs(pos[0, 1]) < 0.01

    def test_position_z_at_equator(self, equator_obs):
        """Test ITRS Z position at equator on prime meridian."""
        pos = equator_obs.itrs_pv.position
        assert np.abs(pos[0, 2]) < 0.01

    def test_velocity_shape(self, equator_obs):
        """Test ITRS velocity array shape."""
        vel = equator_obs.itrs_pv.velocity
        assert vel.shape == (1, 3)

    def test_velocity_x_at_equator(self, equator_obs):
        """Test ITRS X velocity at equator."""
        vel = equator_obs.itrs_pv.velocity
        assert np.abs(vel[0, 0]) < 0.001

    def test_velocity_y_at_equator(self, equator_obs):
        """Test ITRS Y velocity at equator (Earth rotation)."""
        vel = equator_obs.itrs_pv.velocity
        expected_vy = 7.292115e-5 * 6378.137  # ~0.465 km/s
        assert np.abs(vel[0, 1] - expected_vy) < 0.001

    def test_velocity_z_at_equator(self, equator_obs):
        """Test ITRS Z velocity at equator."""
        vel = equator_obs.itrs_pv.velocity
        assert np.abs(vel[0, 2]) < 0.001


class TestGroundEphemerisGCRS:
    """Test GCRS coordinate system calculations."""

    def test_gcrs_not_none(self, mauna_kea_obs):
        """Test that GCRS object is not None."""
        assert mauna_kea_obs.gcrs_pv is not None

    def test_position_magnitude(self, mauna_kea_obs):
        """Test GCRS position magnitude near Earth radius."""
        pos = mauna_kea_obs.gcrs_pv.position
        pos_mag = np.linalg.norm(pos[0])
        assert 6300 < pos_mag < 6500

    def test_velocity_magnitude(self, mauna_kea_obs):
        """Test GCRS velocity magnitude is small."""
        vel = mauna_kea_obs.gcrs_pv.velocity
        vel_mag = np.linalg.norm(vel[0])
        assert vel_mag < 1.0


class TestGroundEphemerisSkyCoords:
    """Test SkyCoord creation for various frames."""

    @pytest.fixture
    def sample_obs(self):
        """Create a sample GroundEphemeris for SkyCoord tests."""
        begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        return GroundEphemeris(40.0, -74.0, 100.0, begin, end, 3600)

    def test_itrs_skycoord_not_none(self, sample_obs):
        """Test that ITRS SkyCoord is created."""
        assert sample_obs.itrs is not None

    def test_gcrs_skycoord_not_none(self, sample_obs):
        """Test that GCRS SkyCoord is created."""
        assert sample_obs.gcrs is not None

    def test_earth_skycoord_not_none(self, sample_obs):
        """Test that Earth SkyCoord is created."""
        assert sample_obs.earth is not None

    def test_sun_skycoord_not_none(self, sample_obs):
        """Test that Sun SkyCoord is created."""
        assert sample_obs.sun is not None

    def test_moon_skycoord_not_none(self, sample_obs):
        """Test that Moon SkyCoord is created."""
        assert sample_obs.moon is not None


class TestGroundEphemerisSunMoon:
    """Test Sun and Moon position calculations."""

    def test_sun_not_none(self, equator_obs):
        """Test that Sun object is not None."""
        assert equator_obs.sun_pv is not None

    def test_sun_distance(self, equator_obs):
        """Test Sun distance is approximately 1 AU."""
        sun_pos = equator_obs.sun_pv.position
        sun_mag = np.linalg.norm(sun_pos[0])
        assert 145000000 < sun_mag < 152000000

    def test_moon_not_none(self, equator_obs):
        """Test that Moon object is not None."""
        assert equator_obs.moon_pv is not None

    def test_moon_distance(self, equator_obs):
        """Test Moon distance is approximately 384,400 km."""
        moon_pos = equator_obs.moon_pv.position
        moon_mag = np.linalg.norm(moon_pos[0])
        assert 356000 < moon_mag < 407000


class TestGroundEphemerisMultiTime:
    """Test multi-timestep ephemeris generation."""

    def test_timestamp_count(self, multi_time_obs):
        """Test that multiple time steps are generated correctly."""
        assert len(multi_time_obs.timestamp) == 7

    def test_itrs_position_shape(self, multi_time_obs):
        """Test ITRS position shape with multiple time steps."""
        assert multi_time_obs.itrs_pv.position.shape == (7, 3)

    def test_itrs_velocity_shape(self, multi_time_obs):
        """Test ITRS velocity shape with multiple time steps."""
        assert multi_time_obs.itrs_pv.velocity.shape == (7, 3)

    def test_gcrs_position_shape(self, multi_time_obs):
        """Test GCRS position shape with multiple time steps."""
        assert multi_time_obs.gcrs_pv.position.shape == (7, 3)

    def test_gcrs_velocity_shape(self, multi_time_obs):
        """Test GCRS velocity shape with multiple time steps."""
        assert multi_time_obs.gcrs_pv.velocity.shape == (7, 3)


class TestGroundEphemerisValidation:
    """Test input validation and error handling."""

    @pytest.fixture
    def valid_times(self):
        """Create valid begin and end times."""
        begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        end = datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        return begin, end

    def test_invalid_latitude_high(self, valid_times):
        """Test that latitude above 90 raises an error."""
        begin, end = valid_times
        with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
            GroundEphemeris(95.0, 0.0, 0.0, begin, end, 3600)

    def test_invalid_latitude_low(self, valid_times):
        """Test that latitude below -90 raises an error."""
        begin, end = valid_times
        with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
            GroundEphemeris(-95.0, 0.0, 0.0, begin, end, 3600)

    def test_invalid_longitude_high(self, valid_times):
        """Test that longitude above 180 raises an error."""
        begin, end = valid_times
        with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
            GroundEphemeris(0.0, 185.0, 0.0, begin, end, 3600)

    def test_invalid_longitude_low(self, valid_times):
        """Test that longitude below -180 raises an error."""
        begin, end = valid_times
        with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
            GroundEphemeris(0.0, -185.0, 0.0, begin, end, 3600)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
