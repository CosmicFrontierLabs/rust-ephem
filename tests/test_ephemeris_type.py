from abc import ABC
from datetime import datetime, timezone

import pytest

from rust_ephem import (
    Ephemeris,
    GroundEphemeris,
    OEMEphemeris,
    SPICEEphemeris,
    TLEEphemeris,
)

# Test data (kept as module constants for backward compatibility)
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


class TestTLEEphemerisType:
    def test_tle_ephemeris_type(self):
        assert isinstance(
            TLEEphemeris(
                tle1=VALID_TLE1,
                tle2=VALID_TLE2,
                begin=BEGIN_TIME,
                end=END_TIME,
                step_size=STEP_SIZE,
            ),
            Ephemeris,
        )


class TestGroundEphemerisType:
    def test_ground_ephemeris_type(self):
        assert isinstance(
            GroundEphemeris(
                latitude=34.0,
                longitude=-118.0,
                height=100.0,
                begin=BEGIN_TIME,
                end=END_TIME,
                step_size=STEP_SIZE,
            ),
            Ephemeris,
        )


class TestOEMEphemerisType:
    def test_oem_ephemeris_type(self, sample_oem_file):
        ephem = OEMEphemeris(
            sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
        )
        assert isinstance(ephem, Ephemeris)


class TestSPICEEphemerisType:
    def test_spice_ephemeris_type(self, spk_path):
        ephem = SPICEEphemeris(
            spk_path,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
            naif_id=301,
            center_id=399,
        )
        assert isinstance(ephem, Ephemeris)


class TestEphemerisABCBehavior:
    """Test that Ephemeris behaves correctly as an Abstract Base Class."""

    def test_ephemeris_is_subclass_of_abc(self):
        """Test that Ephemeris is a subclass of ABC."""
        assert issubclass(Ephemeris, ABC)

    def test_ephemeris_is_type_instance(self):
        """Test that Ephemeris is an instance of type."""
        assert isinstance(Ephemeris, type)

    def test_cannot_instantiate_ephemeris_directly(self):
        """Test that Ephemeris cannot be instantiated directly."""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class Ephemeris"
        ):
            Ephemeris()

    def test_ephemeris_has_abstract_methods(self):
        """Test that Ephemeris defines the expected abstract methods."""
        # Check that all expected abstract methods are defined
        expected_abstract_methods = {
            "timestamp",
            "gcrs_pv",
            "itrs_pv",
            "itrs",
            "gcrs",
            "earth",
            "sun",
            "moon",
            "sun_pv",
            "moon_pv",
            "obsgeoloc",
            "obsgeovel",
            "latitude",
            "latitude_deg",
            "latitude_rad",
            "longitude",
            "longitude_deg",
            "longitude_rad",
            "height",
            "height_m",
            "height_km",
            "sun_radius",
            "sun_radius_deg",
            "moon_radius",
            "moon_radius_deg",
            "earth_radius",
            "earth_radius_deg",
            "sun_radius_rad",
            "moon_radius_rad",
            "earth_radius_rad",
            "index",
        }

        # Get all abstract methods from Ephemeris
        abstract_methods = Ephemeris.__abstractmethods__
        assert abstract_methods == expected_abstract_methods

    def test_tle_ephemeris_is_registered_subclass(self):
        """Test that TLEEphemeris is registered as a virtual subclass of Ephemeris."""
        assert issubclass(TLEEphemeris, Ephemeris)

    def test_spice_ephemeris_is_registered_subclass(self):
        """Test that SPICEEphemeris is registered as a virtual subclass of Ephemeris."""
        assert issubclass(SPICEEphemeris, Ephemeris)

    def test_oem_ephemeris_is_registered_subclass(self):
        """Test that OEMEphemeris is registered as a virtual subclass of Ephemeris."""
        assert issubclass(OEMEphemeris, Ephemeris)

    def test_ground_ephemeris_is_registered_subclass(self):
        """Test that GroundEphemeris is registered as a virtual subclass of Ephemeris."""
        assert issubclass(GroundEphemeris, Ephemeris)

    def test_ground_ephemeris_isinstance_check(self):
        """Test isinstance check works for GroundEphemeris."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert isinstance(ground_eph, Ephemeris)

    def test_tle_ephemeris_isinstance_check(self):
        """Test isinstance check works for TLEEphemeris."""
        tle_eph = TLEEphemeris(
            tle1=VALID_TLE1,
            tle2=VALID_TLE2,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert isinstance(tle_eph, Ephemeris)

    def test_ephemeris_is_subclass_of_abc_again(self):
        """Test that Ephemeris is a subclass of ABC (redundant check for completeness)."""
        assert issubclass(Ephemeris, ABC)

    def test_tle_ephemeris_is_subclass_of_abc(self):
        """Test that TLEEphemeris is a subclass of ABC."""
        assert issubclass(TLEEphemeris, ABC)

    def test_spice_ephemeris_is_subclass_of_abc(self):
        """Test that SPICEEphemeris is a subclass of ABC."""
        assert issubclass(SPICEEphemeris, ABC)

    def test_oem_ephemeris_is_subclass_of_abc(self):
        """Test that OEMEphemeris is a subclass of ABC."""
        assert issubclass(OEMEphemeris, ABC)

    def test_ground_ephemeris_is_subclass_of_abc(self):
        """Test that GroundEphemeris is a subclass of ABC."""
        assert issubclass(GroundEphemeris, ABC)

    def test_ephemeris_instance_has_timestamp_attribute(self):
        """Test that ephemeris instances have timestamp attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "timestamp")

    def test_ephemeris_instance_has_gcrs_pv_attribute(self):
        """Test that ephemeris instances have gcrs_pv attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "gcrs_pv")

    def test_ephemeris_instance_has_itrs_pv_attribute(self):
        """Test that ephemeris instances have itrs_pv attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "itrs_pv")

    def test_ephemeris_instance_has_itrs_attribute(self):
        """Test that ephemeris instances have itrs attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "itrs")

    def test_ephemeris_instance_has_gcrs_attribute(self):
        """Test that ephemeris instances have gcrs attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "gcrs")

    def test_ephemeris_instance_has_earth_attribute(self):
        """Test that ephemeris instances have earth attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "earth")

    def test_ephemeris_instance_has_sun_attribute(self):
        """Test that ephemeris instances have sun attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "sun")

    def test_ephemeris_instance_has_moon_attribute(self):
        """Test that ephemeris instances have moon attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "moon")

    def test_ephemeris_instance_has_sun_pv_attribute(self):
        """Test that ephemeris instances have sun_pv attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "sun_pv")

    def test_ephemeris_instance_has_moon_pv_attribute(self):
        """Test that ephemeris instances have moon_pv attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "moon_pv")

    def test_ephemeris_instance_has_obsgeoloc_attribute(self):
        """Test that ephemeris instances have obsgeoloc attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "obsgeoloc")

    def test_ephemeris_instance_has_obsgeovel_attribute(self):
        """Test that ephemeris instances have obsgeovel attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "obsgeovel")

    def test_ephemeris_instance_has_latitude_attribute(self):
        """Test that ephemeris instances have latitude attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "latitude")

    def test_ephemeris_instance_has_latitude_deg_attribute(self):
        """Test that ephemeris instances have latitude_deg attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "latitude_deg")

    def test_ephemeris_instance_has_latitude_rad_attribute(self):
        """Test that ephemeris instances have latitude_rad attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "latitude_rad")

    def test_ephemeris_instance_has_longitude_attribute(self):
        """Test that ephemeris instances have longitude attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "longitude")

    def test_ephemeris_instance_has_longitude_deg_attribute(self):
        """Test that ephemeris instances have longitude_deg attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "longitude_deg")

    def test_ephemeris_instance_has_longitude_rad_attribute(self):
        """Test that ephemeris instances have longitude_rad attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "longitude_rad")

    def test_ephemeris_instance_has_height_attribute(self):
        """Test that ephemeris instances have height attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "height")

    def test_ephemeris_instance_has_height_m_attribute(self):
        """Test that ephemeris instances have height_m attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "height_m")

    def test_ephemeris_instance_has_height_km_attribute(self):
        """Test that ephemeris instances have height_km attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "height_km")

    def test_ephemeris_instance_has_sun_radius_attribute(self):
        """Test that ephemeris instances have sun_radius attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "sun_radius")

    def test_ephemeris_instance_has_sun_radius_deg_attribute(self):
        """Test that ephemeris instances have sun_radius_deg attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "sun_radius_deg")

    def test_ephemeris_instance_has_moon_radius_attribute(self):
        """Test that ephemeris instances have moon_radius attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "moon_radius")

    def test_ephemeris_instance_has_moon_radius_deg_attribute(self):
        """Test that ephemeris instances have moon_radius_deg attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "moon_radius_deg")

    def test_ephemeris_instance_has_earth_radius_attribute(self):
        """Test that ephemeris instances have earth_radius attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "earth_radius")

    def test_ephemeris_instance_has_earth_radius_deg_attribute(self):
        """Test that ephemeris instances have earth_radius_deg attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "earth_radius_deg")

    def test_ephemeris_instance_has_sun_radius_rad_attribute(self):
        """Test that ephemeris instances have sun_radius_rad attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "sun_radius_rad")

    def test_ephemeris_instance_has_moon_radius_rad_attribute(self):
        """Test that ephemeris instances have moon_radius_rad attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "moon_radius_rad")

    def test_ephemeris_instance_has_earth_radius_rad_attribute(self):
        """Test that ephemeris instances have earth_radius_rad attribute."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "earth_radius_rad")

    def test_ephemeris_instance_has_index_method(self):
        """Test that ephemeris instances have index method."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert hasattr(ground_eph, "index")

    def test_ephemeris_instance_index_is_callable(self):
        """Test that ephemeris index attribute is callable."""
        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert callable(getattr(ground_eph, "index"))

    def test_ephemeris_type_annotation_accepts_ephemeris_instance(self):
        """Test that Ephemeris can be used in type annotations and accepts ephemeris instances."""

        def accepts_ephemeris(eph: Ephemeris) -> bool:
            return isinstance(eph, Ephemeris)

        ground_eph = GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )

        assert accepts_ephemeris(ground_eph)

    def test_ephemeris_type_hints_include_eph_parameter(self):
        """Test that type hints work for functions accepting Ephemeris."""
        from typing import get_type_hints

        def accepts_ephemeris(eph: Ephemeris) -> bool:
            return isinstance(eph, Ephemeris)

        hints = get_type_hints(accepts_ephemeris)
        assert "eph" in hints

    def test_tle_ephemeris_not_in_ephemeris_mro(self):
        """Test that TLEEphemeris does not inherit from Ephemeris in MRO."""
        tle_mro = [cls.__name__ for cls in TLEEphemeris.__mro__]
        assert "Ephemeris" not in tle_mro

    def test_ground_ephemeris_not_in_ephemeris_mro(self):
        """Test that GroundEphemeris does not inherit from Ephemeris in MRO."""
        ground_mro = [cls.__name__ for cls in GroundEphemeris.__mro__]
        assert "Ephemeris" not in ground_mro

    def test_tle_ephemeris_instance_passes_isinstance_check(self):
        """Test that TLEEphemeris instances pass isinstance check with Ephemeris."""
        tle_eph = TLEEphemeris(
            tle1=VALID_TLE1,
            tle2=VALID_TLE2,
            begin=BEGIN_TIME,
            end=END_TIME,
            step_size=STEP_SIZE,
        )
        assert isinstance(tle_eph, Ephemeris)

    def test_integer_not_ephemeris_instance(self):
        """Test that integers are not considered Ephemeris instances."""
        assert not isinstance(42, Ephemeris)

    def test_string_not_ephemeris_instance(self):
        """Test that strings are not considered Ephemeris instances."""
        assert not isinstance("string", Ephemeris)

    def test_list_not_ephemeris_instance(self):
        """Test that lists are not considered Ephemeris instances."""
        assert not isinstance([], Ephemeris)

    def test_dict_not_ephemeris_instance(self):
        """Test that dictionaries are not considered Ephemeris instances."""
        assert not isinstance({}, Ephemeris)

    def test_none_not_ephemeris_instance(self):
        """Test that None is not considered an Ephemeris instance."""
        assert not isinstance(None, Ephemeris)

    def test_unrelated_class_instance_not_ephemeris(self):
        """Test that instances of unrelated classes are not considered Ephemeris instances."""

        class UnrelatedClass:
            pass

        unrelated = UnrelatedClass()
        assert not isinstance(unrelated, Ephemeris)
