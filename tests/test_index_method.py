"""
Tests for the index() method on all ephemeris types
"""

from datetime import datetime

import numpy as np
import pytest

from rust_ephem import GroundEphemeris, SPICEEphemeris, TLEEphemeris


@pytest.fixture
def tle_ephemeris():
    """Create a TLE ephemeris for testing"""
    tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
    tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return TLEEphemeris(tle1, tle2, begin, end, step_size=60)


@pytest.fixture
def ground_ephemeris():
    """Create a ground ephemeris for testing"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    # Austin, Texas coordinates
    return GroundEphemeris(30.2672, -97.7431, 150.0, begin, end, step_size=60)


class TestIndexMethod:
    """Test the index() method functionality"""

    def test_index_returns_int(self, tle_ephemeris):
        """index() should return an integer"""
        target_time = datetime(2024, 1, 1, 0, 30, 0)
        idx = tle_ephemeris.index(target_time)
        assert isinstance(idx, int)

    def test_index_in_valid_range(self, tle_ephemeris):
        """index() should return a value within the valid array range"""
        target_time = datetime(2024, 1, 1, 0, 30, 0)
        idx = tle_ephemeris.index(target_time)
        num_timestamps = len(tle_ephemeris.timestamp)
        assert 0 <= idx < num_timestamps

    def test_index_finds_exact_match(self, tle_ephemeris):
        """index() should return correct index for exact timestamp match"""
        timestamps = tle_ephemeris.timestamp
        # Pick the 10th timestamp
        target_time = timestamps[10]
        idx = tle_ephemeris.index(target_time)
        assert idx == 10

    def test_index_finds_closest_before(self, tle_ephemeris):
        """index() should find closest timestamp when target is between points"""
        # Time that's 10 seconds after the first timestamp (closer to first than second)
        target_time = datetime(2024, 1, 1, 0, 0, 10)
        idx = tle_ephemeris.index(target_time)
        assert idx == 0  # Should be closest to first timestamp

    def test_index_finds_closest_after(self, tle_ephemeris):
        """index() should find closest timestamp when target is between points"""
        # Time that's 50 seconds after the first timestamp (closer to second)
        target_time = datetime(2024, 1, 1, 0, 0, 50)
        idx = tle_ephemeris.index(target_time)
        assert idx == 1  # Should be closest to second timestamp at 1 minute

    def test_index_at_start(self, tle_ephemeris):
        """index() should handle time at the start of the range"""
        target_time = datetime(2024, 1, 1, 0, 0, 0)
        idx = tle_ephemeris.index(target_time)
        assert idx == 0

    def test_index_at_end(self, tle_ephemeris):
        """index() should handle time at the end of the range"""
        target_time = datetime(2024, 1, 1, 1, 0, 0)
        idx = tle_ephemeris.index(target_time)
        timestamps = tle_ephemeris.timestamp
        assert idx == len(timestamps) - 1

    def test_index_before_range(self, tle_ephemeris):
        """index() should return first index for time before range"""
        target_time = datetime(2023, 12, 31, 23, 0, 0)  # Before range
        idx = tle_ephemeris.index(target_time)
        assert idx == 0  # Should return closest (first) timestamp

    def test_index_after_range(self, tle_ephemeris):
        """index() should return last index for time after range"""
        target_time = datetime(2024, 1, 1, 2, 0, 0)  # After range
        idx = tle_ephemeris.index(target_time)
        timestamps = tle_ephemeris.timestamp
        assert idx == len(timestamps) - 1  # Should return closest (last) timestamp

    def test_index_can_access_position_data(self, tle_ephemeris):
        """Returned index should be usable to access position data"""
        target_time = datetime(2024, 1, 1, 0, 30, 0)
        idx = tle_ephemeris.index(target_time)

        # Should be able to use index to access data
        position = tle_ephemeris.gcrs_pv.position[idx]
        assert position.shape == (3,)  # Should be a 3D vector
        assert np.all(np.isfinite(position))  # Should be valid numbers

    def test_index_can_access_velocity_data(self, tle_ephemeris):
        """Returned index should be usable to access velocity data"""
        target_time = datetime(2024, 1, 1, 0, 15, 0)
        idx = tle_ephemeris.index(target_time)

        velocity = tle_ephemeris.gcrs_pv.velocity[idx]
        assert velocity.shape == (3,)
        assert np.all(np.isfinite(velocity))

    def test_index_works_with_ground_ephemeris(self, ground_ephemeris):
        """index() should work with GroundEphemeris"""
        target_time = datetime(2024, 1, 1, 0, 30, 0)
        idx = ground_ephemeris.index(target_time)

        assert isinstance(idx, int)
        assert 0 <= idx < len(ground_ephemeris.timestamp)

        # Should be able to access Sun position
        sun_pos = ground_ephemeris.sun_pv.position[idx]
        assert sun_pos.shape == (3,)

    def test_index_consistency_across_properties(self, tle_ephemeris):
        """Same index should work for all properties"""
        target_time = datetime(2024, 1, 1, 0, 20, 0)
        idx = tle_ephemeris.index(target_time)

        # Access various properties with the same index
        timestamp = tle_ephemeris.timestamp[idx]
        position = tle_ephemeris.gcrs_pv.position[idx]
        velocity = tle_ephemeris.gcrs_pv.velocity[idx]
        sun_radius = tle_ephemeris.sun_radius_deg[idx]

        # All should return valid data
        assert isinstance(timestamp, datetime)
        assert position.shape == (3,)
        assert velocity.shape == (3,)
        assert isinstance(sun_radius, (float, np.floating))


class TestIndexMethodSPICE:
    """Test index() method with SPICE ephemeris"""

    def test_index_works_with_spice(self):
        """index() should work with SPICEEphemeris"""
        # Use the test data SPK file
        spk_path = "test_data/de440s.bsp"
        begin = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        # Create ephemeris for the Moon (NAIF ID 301)
        eph = SPICEEphemeris(spk_path, 301, begin, end, step_size=60, center_id=399)

        target_time = datetime(2024, 1, 1, 0, 30, 0)
        idx = eph.index(target_time)

        assert isinstance(idx, int)
        assert 0 <= idx < len(eph.timestamp)

        # Should be able to access position data
        position = eph.gcrs_pv.position[idx]
        assert position.shape == (3,)
        assert np.all(np.isfinite(position))


class TestIndexMethodEdgeCases:
    """Test edge cases for index() method"""

    def test_index_with_single_timestamp(self):
        """index() should work with ephemeris containing only one timestamp"""
        tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
        tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
        begin = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 0, 0)  # Same as begin
        eph = TLEEphemeris(tle1, tle2, begin, end, step_size=60)

        target_time = datetime(2024, 1, 1, 0, 5, 0)
        idx = eph.index(target_time)
        assert idx == 0  # Only one timestamp, so index must be 0

    def test_index_with_many_timestamps(self):
        """index() should work efficiently with many timestamps"""
        tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
        tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
        begin = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)  # 24 hours
        eph = TLEEphemeris(tle1, tle2, begin, end, step_size=10)  # 10 second steps

        # Should have 8641 timestamps (24*60*6 + 1)
        assert len(eph.timestamp) == 8641

        # Test finding a timestamp in the middle
        target_time = datetime(2024, 1, 1, 12, 0, 0)
        idx = eph.index(target_time)

        # Verify it's approximately in the middle
        assert 4000 < idx < 5000

        # Verify the returned index is close to target time
        found_time = eph.timestamp[idx]
        # Convert to naive datetime for comparison (timestamp returns UTC datetimes)
        if found_time.tzinfo is not None:
            found_time = found_time.replace(tzinfo=None)
        time_diff = abs((found_time - target_time).total_seconds())
        assert time_diff <= 10  # Should be within one step_size

    def test_index_midpoint_rounding(self, tle_ephemeris):
        """Test behavior when target is exactly between two timestamps"""
        # Time exactly between first two timestamps (30 seconds after first)
        target_time = datetime(2024, 1, 1, 0, 0, 30)
        idx = tle_ephemeris.index(target_time)

        # Should return either 0 or 1 (both are equidistant)
        assert idx in [0, 1]
