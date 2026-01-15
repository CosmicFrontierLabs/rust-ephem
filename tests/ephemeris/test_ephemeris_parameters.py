"""Tests for ephemeris parameter reflection.

Tests that all ephemeris classes properly reflect their constructor
parameters back as readable properties, enabling introspection of
ephemeris configuration.
"""

from datetime import datetime, timezone

from rust_ephem import GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris

# Test data
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


class TestTLEEphemerisParameters:
    """Test parameter reflection for TLEEphemeris."""

    def test_tle1_parameter(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle1 == VALID_TLE1

    def test_tle2_parameter(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle2 == VALID_TLE2

    def test_begin_parameter_value(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.begin.tzinfo is not None

    def test_end_parameter_value(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.end == END_TIME

    def test_end_parameter_tzinfo(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self) -> None:
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE, polar_motion=True
        )
        assert ephem.polar_motion is True

    def test_tle_epoch_not_none(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle_epoch is not None

    def test_tle_epoch_year_2008(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle_epoch.year == 2008

    def test_tle_epoch_tzinfo(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE)
        assert ephem.tle_epoch.tzinfo is not None

    def test_all_parameters_accessible(self) -> None:
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE, polar_motion=True
        )
        _ = ephem.tle1
        _ = ephem.tle2
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion
        _ = ephem.tle_epoch


class TestSPICEEphemerisParameters:
    """Test parameter reflection for SPICEEphemeris."""

    def test_spk_path_parameter(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.spk_path == spk_path

    def test_naif_id_parameter(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.naif_id == 301

    def test_center_id_parameter(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.center_id == 399

    def test_begin_parameter_value(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin.tzinfo is not None

    def test_end_parameter_value(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end == END_TIME

    def test_end_parameter_tzinfo(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self, spk_path: str) -> None:
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

    def test_all_parameters_accessible(self, spk_path: str) -> None:
        ephem = SPICEEphemeris(
            spk_path=spk_path,
            naif_id=301,
            center_id=399,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        _ = ephem.spk_path
        _ = ephem.naif_id
        _ = ephem.center_id
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestOEMEphemerisParameters:
    """Test parameter reflection for OEMEphemeris."""

    def test_oem_path_parameter(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.oem_path == sample_oem_file

    def test_begin_parameter_value(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.begin.tzinfo is not None

    def test_end_parameter_value(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.end == END_TIME

    def test_end_parameter_tzinfo(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        assert ephem.polar_motion is True

    def test_all_parameters_accessible(self, sample_oem_file: str) -> None:
        ephem = OEMEphemeris(
            sample_oem_file,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        _ = ephem.oem_path
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestGroundEphemerisParameters:
    """Test parameter reflection for GroundEphemeris."""

    def test_input_latitude_parameter(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == 35.5

    def test_input_longitude_parameter(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_longitude == -120.7

    def test_input_height_parameter(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_height == 250.0

    def test_begin_parameter_value(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.begin.tzinfo is not None

    def test_end_parameter_value(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end == END_TIME

    def test_end_parameter_tzinfo(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.end.tzinfo is not None

    def test_step_size_parameter(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.polar_motion is False

    def test_polar_motion_parameter_true(self) -> None:
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

    def test_negative_latitude_value(self) -> None:
        ephem = GroundEphemeris(
            latitude=-33.9,
            longitude=18.4,
            height=10.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == -33.9

    def test_negative_latitude_longitude(self) -> None:
        ephem = GroundEphemeris(
            latitude=-33.9,
            longitude=18.4,
            height=10.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_longitude == 18.4

    def test_negative_latitude_height(self) -> None:
        ephem = GroundEphemeris(
            latitude=-33.9,
            longitude=18.4,
            height=10.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_height == 10.0

    def test_zero_coords_latitude(self) -> None:
        ephem = GroundEphemeris(
            latitude=0.0,
            longitude=0.0,
            height=0.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_latitude == 0.0

    def test_zero_coords_longitude(self) -> None:
        ephem = GroundEphemeris(
            latitude=0.0,
            longitude=0.0,
            height=0.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_longitude == 0.0

    def test_zero_coords_height(self) -> None:
        ephem = GroundEphemeris(
            latitude=0.0,
            longitude=0.0,
            height=0.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert ephem.input_height == 0.0

    def test_all_parameters_accessible(self) -> None:
        ephem = GroundEphemeris(
            latitude=35.5,
            longitude=-120.7,
            height=250.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            polar_motion=True,
        )
        _ = ephem.input_latitude
        _ = ephem.input_longitude
        _ = ephem.input_height
        _ = ephem.begin
        _ = ephem.end
        _ = ephem.step_size
        _ = ephem.polar_motion


class TestCommonParameterBehavior:
    """Test common behavior across all ephemeris classes."""

    def test_step_size_tle(self) -> None:
        tle_ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, step_size=300
        )
        assert tle_ephem.step_size == 300

    def test_step_size_ground(self) -> None:
        ground_ephem = GroundEphemeris(
            35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, step_size=600
        )
        assert ground_ephem.step_size == 600

    def test_begin_preserved(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        custom_end = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)

        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_end, step_size=60
        )
        assert ephem.begin == custom_begin

    def test_end_preserved(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        custom_end = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)

        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_end, step_size=60
        )

        assert ephem.end == custom_end

    def test_begin_preserved_year(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.year == 2024

    def test_begin_preserved_month(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.month == 6

    def test_begin_preserved_day(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.day == 15

    def test_begin_preserved_hour(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.hour == 10

    def test_begin_preserved_minute(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.minute == 30

    def test_begin_preserved_second(self) -> None:
        custom_begin = datetime(2024, 6, 15, 10, 30, 45, tzinfo=timezone.utc)
        ephem = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, custom_begin, custom_begin, step_size=60
        )
        assert ephem.begin.second == 45

    def test_polar_motion_default_tle(self) -> None:
        tle_false = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 60)
        assert tle_false.polar_motion is False

    def test_polar_motion_true_tle(self) -> None:
        tle_true = TLEEphemeris(
            VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 60, polar_motion=True
        )
        assert tle_true.polar_motion is True

    def test_polar_motion_default_ground(self) -> None:
        ground_false = GroundEphemeris(0, 0, 0, BEGIN_TIME, END_TIME, 60)
        assert ground_false.polar_motion is False

    def test_polar_motion_true_ground(self) -> None:
        ground_true = GroundEphemeris(
            0, 0, 0, BEGIN_TIME, END_TIME, 60, polar_motion=True
        )
        assert ground_true.polar_motion is True

    def test_timezone_awareness_begin(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 120)
        assert ephem.begin.tzinfo == timezone.utc

    def test_timezone_awareness_end(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 120)
        assert ephem.end.tzinfo == timezone.utc

    def test_timezone_awareness_tle_epoch(self) -> None:
        ephem = TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, 120)
        assert ephem.tle_epoch.tzinfo is not None

    def test_parameter_type_latitude(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.input_latitude, float)

    def test_parameter_type_longitude(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.input_longitude, float)

    def test_parameter_type_height(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.input_height, float)

    def test_parameter_type_step_size(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.step_size, int)

    def test_parameter_type_polar_motion_bool(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.polar_motion, bool)

    def test_parameter_type_begin_datetime(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.begin, datetime)

    def test_parameter_type_end_datetime(self) -> None:
        ephem = GroundEphemeris(35.5, -120.7, 250.0, BEGIN_TIME, END_TIME, 120)
        assert isinstance(ephem.end, datetime)
