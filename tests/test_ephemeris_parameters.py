"""Tests for ephemeris parameter reflection.

Tests that all ephemeris classes properly reflect their constructor
parameters back as readable properties, enabling introspection of
ephemeris configuration.

Tests cover:
- TLEEphemeris: tle1, tle2, begin, end, step_size, polar_motion, tle_epoch
- SPICEEphemeris: spk_path, naif_id, center_id, begin, end, step_size, polar_motion
- OEMEphemeris: oem_path, begin, end, step_size, polar_motion
- GroundEphemeris: input_latitude, input_longitude, input_height, begin, end, step_size, polar_motion
"""

import os
from datetime import datetime, timezone

import pytest

from rust_ephem import GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris

# Test data
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


@pytest.fixture
def sample_oem_file(tmp_path):
    """Create a minimal OEM file for testing."""
    oem_path = tmp_path / "test_satellite.oem"
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
STOP_TIME = 2024-01-01T03:00:00.000
META_STOP

DATA_START
2024-01-01T00:00:00.000 7000.0 0.0 0.0 0.0 7.5 0.0
2024-01-01T01:00:00.000 7000.0 4500.0 0.0 -0.3897 7.4856 0.0
2024-01-01T02:00:00.000 6995.0 9000.0 0.0 -0.7791 7.4427 0.0
2024-01-01T03:00:00.000 6980.0 13500.0 0.0 -1.1677 7.3714 0.0
DATA_STOP
"""
    oem_path.write_text(oem_content)
    return str(oem_path)


class TestTLEEphemerisParameters:
    """Test parameter reflection for TLEEphemeris."""

    def test_tle1_parameter(self):
        """Test that tle1 property returns the TLE line 1."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle1 == VALID_TLE1

    def test_tle2_parameter(self):
        """Test that tle2 property returns the TLE line 2."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle2 == VALID_TLE2

    def test_begin_parameter(self):
        """Test that begin property returns the start time."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.begin == BEGIN_TIME
        assert ephem.begin.tzinfo is not None  # Should have timezone

    def test_end_parameter(self):
        """Test that end property returns the end time."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.end == END_TIME
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self):
        """Test that step_size property returns the time step."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self):
        """Test that polar_motion defaults to False."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self):
        """Test that polar_motion can be set to True."""
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE, polar_motion=True
        )
        assert ephem.polar_motion is True

    def test_tle_epoch_parameter(self):
        """Test that tle_epoch is extracted from TLE data."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle_epoch is not None
        # TLE epoch should be in 2008 based on the test TLE
        assert ephem.tle_epoch.year == 2008
        assert ephem.tle_epoch.tzinfo is not None

    def test_all_parameters_accessible(self):
        """Test that all parameters are accessible without errors."""
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE, polar_motion=True
        )
        # Access all properties to ensure no exceptions
        _ = ephem.tle1
        _ = ephem.tle2
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion
        _ = ephem.tle_epoch


class TestSPICEEphemerisParameters:
    """Test parameter reflection for SPICEEphemeris."""

    @pytest.fixture
    def spk_path(self):
        """Return path to SPICE kernel if available."""
        path = "test_data/de440s.bsp"
        if not os.path.exists(path):
            pytest.skip(f"SPICE kernel not found at {path}")
        return path

    def test_spk_path_parameter(self, spk_path):
        """Test that spk_path property returns the kernel path."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,  # Moon
            center_id=399,  # Earth
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.spk_path == spk_path

    def test_naif_id_parameter(self, spk_path):
        """Test that naif_id property returns the target body ID."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.naif_id == 301

    def test_center_id_parameter(self, spk_path):
        """Test that center_id property returns the center body ID."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.center_id == 399

    def test_begin_parameter(self, spk_path):
        """Test that begin property returns the start time."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin == BEGIN_TIME
        assert ephem.begin.tzinfo is not None

    def test_end_parameter(self, spk_path):
        """Test that end property returns the end time."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end == END_TIME
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self, spk_path):
        """Test that step_size property returns the time step."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self, spk_path):
        """Test that polar_motion defaults to False."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self, spk_path):
        """Test that polar_motion can be set to True."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        assert ephem.polar_motion is True

    def test_all_parameters_accessible(self, spk_path):
        """Test that all parameters are accessible without errors."""
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        # Access all properties to ensure no exceptions
        _ = ephem.spk_path
        _ = ephem.naif_id
        _ = ephem.center_id
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestOEMEphemerisParameters:
    """Test parameter reflection for OEMEphemeris."""

    def test_oem_path_parameter(self, sample_oem_file):
        """Test that oem_path property returns the OEM file path."""
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.oem_path == sample_oem_file

    def test_begin_parameter(self, sample_oem_file):
        """Test that begin property returns the start time."""
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.begin == BEGIN_TIME
        assert ephem.begin.tzinfo is not None

    def test_end_parameter(self, sample_oem_file):
        """Test that end property returns the end time."""
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.end == END_TIME
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self, sample_oem_file):
        """Test that step_size property returns the time step."""
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self, sample_oem_file):
        """Test that polar_motion defaults to False."""
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self, sample_oem_file):
        """Test that polar_motion can be set to True."""
        ephem = OEMEphemeris(
            sample_oem_file,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        assert ephem.polar_motion is True

    def test_all_parameters_accessible(self, sample_oem_file):
        """Test that all parameters are accessible without errors."""
        ephem = OEMEphemeris(
            sample_oem_file,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        # Access all properties to ensure no exceptions
        _ = ephem.oem_path
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestGroundEphemerisParameters:
    """Test parameter reflection for GroundEphemeris."""

    def test_input_latitude_parameter(self):
        """Test that input_latitude property returns the latitude."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == 35.5

    def test_input_longitude_parameter(self):
        """Test that input_longitude property returns the longitude."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_longitude == -120.7

    def test_input_height_parameter(self):
        """Test that input_height property returns the height."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_height == 250.0

    def test_begin_parameter(self):
        """Test that begin property returns the start time."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin == BEGIN_TIME
        assert ephem.begin.tzinfo is not None

    def test_end_parameter(self):
        """Test that end property returns the end time."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end == END_TIME
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self):
        """Test that step_size property returns the time step."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self):
        """Test that polar_motion defaults to False."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self):
        """Test that polar_motion can be set to True."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        assert ephem.polar_motion is True

    def test_negative_latitude(self):
        """Test parameter reflection with negative latitude (southern hemisphere)."""
        ephem = GroundEphemeris(
            latitude=-33.9,
            longitude=18.4,
            height=10.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == -33.9
        assert ephem.input_longitude == 18.4
        assert ephem.input_height == 10.0

    def test_zero_coordinates(self):
        """Test parameter reflection with zero coordinates (equator/prime meridian)."""
        ephem = GroundEphemeris(
            latitude=0.0,
            longitude=0.0,
            height=0.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == 0.0
        assert ephem.input_longitude == 0.0
        assert ephem.input_height == 0.0

    def test_all_parameters_accessible(self):
        """Test that all parameters are accessible without errors."""
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        # Access all properties to ensure no exceptions
        _ = ephem.input_latitude
        _ = ephem.input_longitude
        _ = ephem.input_height
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestCommonParameterBehavior:
    """Test common behavior across all ephemeris classes."""

    def test_step_size_calculated_from_timestamps(self):
        """Test that step_size is computed from actual timestamps."""
        # Test with TLE
        tle_ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, step_size=300
        )
        assert tle_ephem.step_size == 300

        # Test with Ground
        ground_ephem = GroundEphemeris(
            35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, step_size=600
        )
        assert ground_ephem.step_size == 600

    def test_begin_end_preserved_exactly(self):
        """Test that begin/end times match exactly what was provided."""
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        custom_end = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)

        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_end, step_size=60
        )

        assert ephem.begin == custom_begin
        assert ephem.end == custom_end
        assert ephem.begin.year == 2024
        assert ephem.begin.month == 6
        assert ephem.begin.day == 15
        assert ephem.begin.hour == 10
        assert ephem.begin.minute == 30
        assert ephem.begin.second == 45

    def test_polar_motion_flag_behavior(self):
        """Test that polar_motion flag behaves correctly for all classes."""
        # Test default (False)
        tle_false = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 60)
        assert tle_false.polar_motion is False

        # Test explicit True
        tle_true = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 60, polar_motion=True
        )
        assert tle_true.polar_motion is True

        # Test with Ground ephemeris
        ground_false = GroundEphemeris(0, 0, 0, BEGIN_TIME, END_TIME, 60)
        assert ground_false.polar_motion is False

        ground_true = GroundEphemeris(
            0, 0, 0, BEGIN_TIME, END_TIME, 60, polar_motion=True
        )
        assert ground_true.polar_motion is True

    def test_timezone_awareness(self):
        """Test that all datetime properties maintain timezone information."""
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 120)

        assert ephem.begin.tzinfo == timezone.utc
        assert ephem.end.tzinfo == timezone.utc
        assert ephem.tle_epoch.tzinfo is not None

    def test_parameter_types(self):
        """Test that parameters return expected types."""
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)

        # Numeric types
        assert isinstance(ephem.input_latitude, float)
        assert isinstance(ephem.input_longitude, float)
        assert isinstance(ephem.input_height, float)
        assert isinstance(ephem.step_size, int)

        # Boolean type
        assert isinstance(ephem.polar_motion, bool)

        # Datetime types
        assert isinstance(ephem.begin, datetime)
        assert isinstance(ephem.end, datetime)
