"""Tests for ephemeris parameter reflection.

Tests that all ephemeris classes properly reflect their constructor
parameters back as readable properties, enabling introspection of
ephemeris configuration.
"""

import os

# Import constants from main conftest
import sys
from datetime import datetime, timezone

from rust_ephem import GroundEphemeris, OEMEphemeris, SPICEEphemeris, TLEEphemeris

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from conftest import BEGIN_TIME, END_TIME, STEP_SIZE, VALID_TLE1, VALID_TLE2


class TestTLEEphemerisParameters:
    """Test parameter reflection for TLEEphemeris."""

    def test_tle1_parameter(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.tle1 == VALID_TLE1

    def test_tle2_parameter(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.tle2 == VALID_TLE2

    def test_begin_parameter_value(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.begin.tzinfo is not None

    def test_end_parameter_value(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.end == END_TIME

    def test_end_parameter_tzinfo(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.end.tzinfo is not None

    def test_step_size_parameter(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(
        self, tle_ephem_default: TLEEphemeris
    ) -> None:
        assert tle_ephem_default.polar_motion is False

    def test_polar_motion_parameter_true(self, tle_ephem_polar: TLEEphemeris) -> None:
        assert tle_ephem_polar.polar_motion is True

    def test_tle_epoch_not_none(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.tle_epoch is not None

    def test_tle_epoch_year_2008(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.tle_epoch.year == 2008

    def test_tle_epoch_tzinfo(self, tle_ephem_default: TLEEphemeris) -> None:
        assert tle_ephem_default.tle_epoch.tzinfo is not None


class TestSPICEEphemerisParameters:
    """Test parameter reflection for SPICEEphemeris."""

    def test_spk_path_parameter(
        self, spice_ephem_default: SPICEEphemeris, spk_path: str
    ) -> None:
        assert spice_ephem_default.spk_path == spk_path

    def test_naif_id_parameter(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.naif_id == 301

    def test_center_id_parameter(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.center_id == 399

    def test_begin_parameter_value(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.begin.tzinfo is not None

    def test_end_parameter_value(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.end == END_TIME

    def test_end_parameter_tzinfo(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.end.tzinfo is not None

    def test_step_size_parameter(self, spice_ephem_default: SPICEEphemeris) -> None:
        assert spice_ephem_default.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(
        self, spice_ephem_default: SPICEEphemeris
    ) -> None:
        assert spice_ephem_default.polar_motion is False

    def test_polar_motion_parameter_true(
        self, spice_ephem_polar: SPICEEphemeris
    ) -> None:
        assert spice_ephem_polar.polar_motion is True


class TestOEMEphemerisParameters:
    """Test parameter reflection for OEMEphemeris."""

    def test_oem_path_parameter(
        self, oem_ephem_default: OEMEphemeris, sample_oem_file: str
    ) -> None:
        assert oem_ephem_default.oem_path == sample_oem_file

    def test_begin_parameter_value(self, oem_ephem_default: OEMEphemeris) -> None:
        assert oem_ephem_default.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(self, oem_ephem_default: OEMEphemeris) -> None:
        assert oem_ephem_default.begin.tzinfo is not None

    def test_end_parameter_value(self, oem_ephem_default: OEMEphemeris) -> None:
        assert oem_ephem_default.end == END_TIME

    def test_end_parameter_tzinfo(self, oem_ephem_default: OEMEphemeris) -> None:
        assert oem_ephem_default.end.tzinfo is not None

    def test_step_size_parameter(self, oem_ephem_default: OEMEphemeris) -> None:
        assert oem_ephem_default.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(
        self, oem_ephem_default: OEMEphemeris
    ) -> None:
        assert oem_ephem_default.polar_motion is False

    def test_polar_motion_parameter_true(self, oem_ephem_polar: OEMEphemeris) -> None:
        assert oem_ephem_polar.polar_motion is True


class TestGroundEphemerisParameters:
    """Test parameter reflection for GroundEphemeris."""

    def test_input_latitude_parameter(
        self, ground_ephem_default: GroundEphemeris
    ) -> None:
        assert ground_ephem_default.input_latitude == 35.5

    def test_input_longitude_parameter(
        self, ground_ephem_default: GroundEphemeris
    ) -> None:
        assert ground_ephem_default.input_longitude == -120.7

    def test_input_height_parameter(
        self, ground_ephem_default: GroundEphemeris
    ) -> None:
        assert ground_ephem_default.input_height == 250.0

    def test_begin_parameter_value(self, ground_ephem_default: GroundEphemeris) -> None:
        assert ground_ephem_default.begin == BEGIN_TIME

    def test_begin_parameter_tzinfo(
        self, ground_ephem_default: GroundEphemeris
    ) -> None:
        assert ground_ephem_default.begin.tzinfo is not None

    def test_end_parameter_value(self, ground_ephem_default: GroundEphemeris) -> None:
        assert ground_ephem_default.end == END_TIME

    def test_end_parameter_tzinfo(self, ground_ephem_default: GroundEphemeris) -> None:
        assert ground_ephem_default.end.tzinfo is not None

    def test_step_size_parameter(self, ground_ephem_default: GroundEphemeris) -> None:
        assert ground_ephem_default.step_size == STEP_SIZE

    def test_polar_motion_parameter_default(
        self, ground_ephem_default: GroundEphemeris
    ) -> None:
        assert ground_ephem_default.polar_motion is False

    def test_polar_motion_parameter_true(
        self, ground_ephem_polar: GroundEphemeris
    ) -> None:
        assert ground_ephem_polar.polar_motion is True

    def test_negative_latitude_value(
        self, ground_ephem_negative: GroundEphemeris
    ) -> None:
        assert ground_ephem_negative.input_latitude == -33.9

    def test_negative_latitude_longitude(
        self, ground_ephem_negative: GroundEphemeris
    ) -> None:
        assert ground_ephem_negative.input_longitude == 18.4

    def test_negative_latitude_height(
        self, ground_ephem_negative: GroundEphemeris
    ) -> None:
        assert ground_ephem_negative.input_height == 10.0

    def test_zero_coords_latitude(self, ground_ephem_zero: GroundEphemeris) -> None:
        assert ground_ephem_zero.input_latitude == 0.0

    def test_zero_coords_longitude(self, ground_ephem_zero: GroundEphemeris) -> None:
        assert ground_ephem_zero.input_longitude == 0.0

    def test_zero_coords_height(self, ground_ephem_zero: GroundEphemeris) -> None:
        assert ground_ephem_zero.input_height == 0.0


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
