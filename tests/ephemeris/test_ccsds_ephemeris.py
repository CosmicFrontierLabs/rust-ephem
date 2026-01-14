"""Tests for CCSDS OEM ephemeris"""

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from rust_ephem import OEMEphemeris

# Sample OEM data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")


def create_sample_oem(path):
    """Create a simple OEM file for testing"""
    oem_content = """CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00.000
ORIGINATOR = TEST

META_START
OBJECT_NAME = TEST_SAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = J2000
TIME_SYSTEM = UTC
START_TIME = 2024-01-01T00:00:00.000
STOP_TIME = 2024-01-01T01:00:00.000
META_STOP

DATA_START
2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T00:10:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
2024-01-01T00:20:00.000 6995.0 9000.0 0.0 -0.7791 7.4427 0.0
2024-01-01T00:30:00.000 6980.0 13500.0 0.0 -1.1677 7.3714 0.0
2024-01-01T00:40:00.000 6955.0 18000.0 0.0 -1.5550 7.2716 0.0
2024-01-01T00:50:00.000 6920.0 22500.0 0.0 -1.9407 7.1434 0.0
2024-01-01T01:00:00.000 6875.0 27000.0 0.0 -2.3243 6.9870 0.0
DATA_STOP
"""
    with open(path, "w") as f:
        f.write(oem_content)


@pytest.fixture
def sample_oem_path(tmp_path):
    """Create a temporary OEM file for testing"""
    oem_path = tmp_path / "test_satellite.oem"
    create_sample_oem(str(oem_path))
    return str(oem_path)


def test_ccsds_ephemeris_initialization(sample_oem_path):
    """Test basic initialization of OEMEphemeris"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=60)

    assert eph is not None
    assert eph.timestamp is not None
    assert len(eph.timestamp) == 31  # 30 minutes at 60-second steps + 1


def test_ccsds_ephemeris_gcrs_data(sample_oem_path):
    """Test GCRS position and velocity data"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    eph = OEMEphemeris(
        sample_oem_path,
        begin=begin,
        end=end,
        step_size=300,  # 5 minute steps for easier checking
    )

    # Check GCRS data exists
    gcrs_pv = eph.gcrs_pv
    assert gcrs_pv.position.shape[0] == 7  # 30 minutes / 5 minutes + 1
    assert gcrs_pv.position.shape[1] == 3
    assert gcrs_pv.velocity.shape[1] == 3

    # Check that position values are reasonable (in km, LEO orbit)
    positions = gcrs_pv.position
    assert np.all(positions[:, 0] > 6000)  # X position should be > 6000 km
    assert np.all(positions[:, 0] < 8000)  # and < 8000 km for LEO


def test_ccsds_ephemeris_interpolation(sample_oem_path):
    """Test that interpolation produces smooth results"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    # High-resolution ephemeris with 1-second steps
    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=1)

    # Check that interpolation produces continuous results
    velocities = eph.gcrs_pv.velocity

    # Velocities should be relatively smooth (no huge jumps)
    velocity_diffs = np.diff(velocities, axis=0)
    max_velocity_change = np.max(np.abs(velocity_diffs))
    assert max_velocity_change < 0.1  # Less than 0.1 km/s change per second


def test_ccsds_ephemeris_itrs_conversion(sample_oem_path):
    """Test ITRS coordinate conversion"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Check ITRS data exists
    itrs_pv = eph.itrs_pv
    assert itrs_pv.position.shape[0] == 7
    assert itrs_pv.position.shape[1] == 3

    # ITRS and GCRS positions should be different (due to Earth rotation)
    gcrs_pos = eph.gcrs_pv.position[0]
    itrs_pos = itrs_pv.position[0]
    assert not np.allclose(gcrs_pos, itrs_pos)


def test_ccsds_ephemeris_sun_moon(sample_oem_path):
    """Test Sun and Moon position calculations"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Check that Sun and Moon SkyCoord objects can be accessed
    sun = eph.sun
    moon = eph.moon
    assert sun is not None
    assert moon is not None


def test_ccsds_ephemeris_index_method(sample_oem_path):
    """Test the index() method for finding closest timestamps"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=60)

    # Test finding index for exact time
    target_time = datetime(2024, 1, 1, 0, 15, 0)
    idx = eph.index(target_time)
    assert idx == 15  # 15 minutes = 15 steps of 60 seconds

    # Test finding index for time between steps
    target_time = datetime(2024, 1, 1, 0, 15, 30)
    idx = eph.index(target_time)
    assert idx in [15, 16]  # Should be close to 15 or 16


def test_ccsds_ephemeris_time_range_validation(sample_oem_path):
    """Test that requesting times outside OEM range raises an error"""
    # OEM data only covers 2024-01-01 00:00 to 01:00
    begin = datetime(2024, 1, 1, 2, 0, 0)  # Outside range
    end = datetime(2024, 1, 1, 3, 0, 0)

    with pytest.raises(ValueError, match="exceeds OEM data range"):
        OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=60)


def test_ccsds_ephemeris_oem_pv_property(sample_oem_path):
    """Test accessing raw OEM data"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 30, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Access raw OEM data
    oem_pv = eph.oem_pv
    assert oem_pv.position.shape[0] == 7  # 7 data points in sample OEM
    assert oem_pv.position.shape[1] == 3

    # First position should match the first OEM entry
    assert oem_pv.position[0, 0] == pytest.approx(7000.0, rel=1e-6)
    assert oem_pv.position[0, 1] == pytest.approx(0.0, rel=1e-6)


def test_ccsds_ephemeris_angular_radius(sample_oem_path):
    """Test angular radius calculations for celestial bodies"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Check that angular radius properties exist
    sun_radius_deg = eph.sun_radius_deg
    moon_radius_deg = eph.moon_radius_deg
    earth_radius_deg = eph.earth_radius_deg

    assert sun_radius_deg.shape[0] == 3
    assert moon_radius_deg.shape[0] == 3
    assert earth_radius_deg.shape[0] == 3

    # Sun angular radius should be around 0.25 degrees from LEO
    assert np.all(sun_radius_deg > 0.2)
    assert np.all(sun_radius_deg < 0.3)

    # Earth angular radius should be large from LEO (around 60-70 degrees)
    assert np.all(earth_radius_deg > 50)
    assert np.all(earth_radius_deg < 80)


def test_ccsds_ephemeris_oem_timestamp(sample_oem_path):
    """Test that oem_timestamp property returns raw OEM timestamps"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)

    # Use step_size=300 (5 minutes) which will create 13 interpolated points
    # vs 7 raw OEM timestamps (every 10 minutes)
    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Get raw OEM timestamps
    oem_timestamps = eph.oem_timestamp

    # Should have 7 raw timestamps (from the sample OEM file, every 10 minutes)
    assert len(oem_timestamps) == 7

    # Check they are datetime objects
    assert all(isinstance(ts, datetime) for ts in oem_timestamps)

    # Check they are timezone-aware (UTC)
    assert all(ts.tzinfo is not None for ts in oem_timestamps)

    # Check first and last timestamps match expected values (with UTC timezone)
    assert oem_timestamps[0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert oem_timestamps[-1] == datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Should match the number of OEM position vectors
    assert len(oem_timestamps) == len(eph.oem_pv.position)

    # Interpolated timestamps should be different (13 vs 7)
    assert len(eph.timestamp) == 13  # 0, 5, 10, 15, ..., 60 minutes
    assert len(eph.timestamp) != len(oem_timestamps)


def test_ccsds_ephemeris_invalid_reference_frame(tmp_path):
    """Test that invalid reference frames are rejected"""
    # Create OEM with ITRF (Earth-fixed) frame instead of inertial
    oem_content = """CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00.000
ORIGINATOR = TEST

META_START
OBJECT_NAME = TEST_SAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = ITRF
TIME_SYSTEM = UTC
START_TIME = 2024-01-01T00:00:00.000
STOP_TIME = 2024-01-01T01:00:00.000
META_STOP

2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T00:10:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
"""

    oem_path = tmp_path / "invalid_frame.oem"
    with open(oem_path, "w") as f:
        f.write(oem_content)

    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    # Should raise ValueError for unsupported frame
    with pytest.raises(ValueError, match="Unsupported reference frame 'ITRF'"):
        OEMEphemeris(str(oem_path), begin=begin, end=end, step_size=60)


def test_ccsds_ephemeris_missing_reference_frame(tmp_path):
    """Test that missing reference frames are rejected"""
    # Create OEM without REF_FRAME field
    oem_content = """CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00.000
ORIGINATOR = TEST

META_START
OBJECT_NAME = TEST_SAT
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
TIME_SYSTEM = UTC
START_TIME = 2024-01-01T00:00:00.000
STOP_TIME = 2024-01-01T01:00:00.000
META_STOP

2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T00:10:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
"""

    oem_path = tmp_path / "missing_frame.oem"
    with open(oem_path, "w") as f:
        f.write(oem_content)

    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    # Should raise ValueError for missing frame
    with pytest.raises(ValueError, match="does not specify a REF_FRAME"):
        OEMEphemeris(str(oem_path), begin=begin, end=end, step_size=60)


def test_ccsds_ephemeris_sun_moon_pv(sample_oem_path):
    """Test that sun_pv and moon_pv properties are exposed"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Test sun_pv
    sun_pv = eph.sun_pv
    assert sun_pv is not None
    assert hasattr(sun_pv, "position")
    assert hasattr(sun_pv, "velocity")
    assert sun_pv.position.shape[0] == 3  # 0, 5, 10 minutes
    assert sun_pv.position.shape[1] == 3  # x, y, z
    assert sun_pv.velocity.shape == sun_pv.position.shape

    # Test moon_pv
    moon_pv = eph.moon_pv
    assert moon_pv is not None
    assert hasattr(moon_pv, "position")
    assert hasattr(moon_pv, "velocity")
    assert moon_pv.position.shape[0] == 3
    assert moon_pv.position.shape[1] == 3
    assert moon_pv.velocity.shape == moon_pv.position.shape


def test_ccsds_ephemeris_obsgeoloc_obsgeovel(sample_oem_path):
    """Test that obsgeoloc and obsgeovel properties are exposed"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 0, 10, 0)

    eph = OEMEphemeris(sample_oem_path, begin=begin, end=end, step_size=300)

    # Test obsgeoloc (should be same as gcrs_pv.position)
    obsgeoloc = eph.obsgeoloc
    assert obsgeoloc is not None
    assert obsgeoloc.shape[0] == 3  # 0, 5, 10 minutes
    assert obsgeoloc.shape[1] == 3  # x, y, z

    # Should match GCRS position
    gcrs_pv = eph.gcrs_pv
    assert np.allclose(obsgeoloc, gcrs_pv.position)

    # Test obsgeovel (should be same as gcrs_pv.velocity)
    obsgeovel = eph.obsgeovel
    assert obsgeovel is not None
    assert obsgeovel.shape[0] == 3
    assert obsgeovel.shape[1] == 3

    # Should match GCRS velocity
    assert np.allclose(obsgeovel, gcrs_pv.velocity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
