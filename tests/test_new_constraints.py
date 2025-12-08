"""
Test suite for new astronomical constraints.

Tests AirmassConstraint, DaytimeConstraint, MoonPhaseConstraint, and SAAConstraint.
"""

import numpy as np
import pytest
from pydantic import ValidationError

import rust_ephem
from rust_ephem.constraints import (
    AirmassConstraint,
    DaytimeConstraint,
    MoonPhaseConstraint,
    SAAConstraint,
)


class TestAirmassConstraint:
    """Test AirmassConstraint functionality."""

    def test_airmass_constraint_creation_max_only(self) -> None:
        """Test creating airmass constraint with max_airmass only."""
        constraint = AirmassConstraint(max_airmass=2.0)
        assert constraint.max_airmass == 2.0

    def test_airmass_constraint_creation_min_and_max(self) -> None:
        """Test creating airmass constraint with both min and max."""
        constraint = AirmassConstraint(max_airmass=3.0, min_airmass=1.2)
        assert constraint.min_airmass == 1.2

    def test_airmass_constraint_creation_max_only_min_none(self) -> None:
        """Test creating airmass constraint with max_airmass only has min_airmass None."""
        constraint = AirmassConstraint(max_airmass=2.0)
        assert constraint.min_airmass is None

    def test_airmass_constraint_creation_both_max(self) -> None:
        """Test creating airmass constraint with both has correct max."""
        constraint = AirmassConstraint(max_airmass=3.0, min_airmass=1.2)
        assert constraint.max_airmass == 3.0

    def test_airmass_constraint_validation_valid_max(self) -> None:
        """Test airmass constraint parameter validation with valid max."""
        AirmassConstraint(max_airmass=1.5)

    def test_airmass_constraint_validation_valid_both(self) -> None:
        """Test airmass constraint parameter validation with valid both."""
        AirmassConstraint(max_airmass=5.0, min_airmass=1.0)

    def test_airmass_constraint_validation_invalid_max_low(self) -> None:
        """Test airmass constraint parameter validation with invalid max (< 1.0)."""
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=0.5)

    def test_airmass_constraint_validation_invalid_min_low(self) -> None:
        """Test airmass constraint parameter validation with invalid min (< 1.0)."""
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=2.0, min_airmass=0.8)

    def test_airmass_constraint_validation_min_greater_than_max(self) -> None:
        """Test airmass constraint parameter validation with min > max."""
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=1.5, min_airmass=2.0)

    def test_airmass_constraint_evaluation_zenith_satisfied(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test airmass constraint evaluation with target at zenith."""
        constraint = AirmassConstraint(max_airmass=1.5)
        result = constraint.evaluate(ground_ephemeris, target_ra=0.0, target_dec=35.0)
        assert result.all_satisfied

    def test_airmass_constraint_evaluation_horizon_violated(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test airmass constraint evaluation with target at horizon."""
        constraint = AirmassConstraint(max_airmass=1.5)
        result = constraint.evaluate(ground_ephemeris, target_ra=0.0, target_dec=-27.0)
        assert not result.all_satisfied

    def test_airmass_constraint_batch_shape(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test batch airmass constraint evaluation shape."""
        constraint = AirmassConstraint(max_airmass=1.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert result.shape == (3, len(ground_ephemeris.timestamp))

    def test_airmass_constraint_batch_target0_satisfied(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test batch airmass constraint evaluation target 0 satisfied."""
        constraint = AirmassConstraint(max_airmass=1.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert np.all(result[0, :])

    # def test_airmass_constraint_batch_target1_satisfied(
    #     self, ground_ephemeris: rust_ephem.GroundEphemeris
    # ) -> None:
    #     """Test batch airmass constraint evaluation target 1 has some satisfied."""
    #     constraint = AirmassConstraint(max_airmass=1.5)
    #     target_ras = [0.0, 90.0, 180.0]
    #     target_decs = [35.0, 35.0, -27.0]
    #     result = constraint.in_constraint_batch(
    #         ground_ephemeris, target_ras, target_decs
    #     )
    #     assert np.any(result[1, :])

    def test_airmass_constraint_batch_target2_some_violations(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test batch airmass constraint evaluation target 2 has some violations."""
        constraint = AirmassConstraint(max_airmass=1.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert np.any(~result[2, :])


class TestDaytimeConstraint:
    """Test DaytimeConstraint functionality."""

    def test_daytime_constraint_creation_default(self) -> None:
        """Test creating daytime constraint with default."""
        constraint = DaytimeConstraint()
        assert constraint.twilight == "civil"

    def test_daytime_constraint_creation_nautical(self) -> None:
        """Test creating daytime constraint with nautical."""
        constraint = DaytimeConstraint(twilight="nautical")
        assert constraint.twilight == "nautical"

    def test_daytime_constraint_creation_astronomical(self) -> None:
        """Test creating daytime constraint with astronomical."""
        constraint = DaytimeConstraint(twilight="astronomical")
        assert constraint.twilight == "astronomical"

    def test_daytime_constraint_creation_none(self) -> None:
        """Test creating daytime constraint with none."""
        constraint = DaytimeConstraint(twilight="none")
        assert constraint.twilight == "none"

    def test_daytime_constraint_validation_civil(self) -> None:
        """Test daytime constraint parameter validation with civil."""
        DaytimeConstraint(twilight="civil")

    def test_daytime_constraint_validation_nautical(self) -> None:
        """Test daytime constraint parameter validation with nautical."""
        DaytimeConstraint(twilight="nautical")

    def test_daytime_constraint_validation_astronomical(self) -> None:
        """Test daytime constraint parameter validation with astronomical."""
        DaytimeConstraint(twilight="astronomical")

    def test_daytime_constraint_validation_none(self) -> None:
        """Test daytime constraint parameter validation with none."""
        DaytimeConstraint(twilight="none")

    def test_daytime_constraint_validation_invalid(self) -> None:
        """Test daytime constraint parameter validation with invalid."""
        with pytest.raises(ValueError):
            DaytimeConstraint(twilight="invalid")  # type: ignore[arg-type]

    def test_daytime_constraint_evaluation_type(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test daytime constraint evaluation returns bool."""
        constraint = DaytimeConstraint()
        daytime_result = constraint.evaluate(
            ground_ephemeris, target_ra=0.0, target_dec=0.0
        )
        assert isinstance(daytime_result.all_satisfied, bool)

    def test_daytime_constraint_batch_shape(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test batch daytime constraint evaluation shape."""
        constraint = DaytimeConstraint()
        target_ras = [0.0, 120.0, 240.0]
        target_decs = [0.0, 0.0, 0.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert result.shape == (3, len(ground_ephemeris.timestamp))

    def test_daytime_constraint_batch_dtype(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test batch daytime constraint evaluation dtype."""
        constraint = DaytimeConstraint()
        target_ras = [0.0, 120.0, 240.0]
        target_decs = [0.0, 0.0, 0.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert result.dtype == bool


class TestMoonPhaseConstraint:
    """Test MoonPhaseConstraint functionality."""

    def test_moon_phase_constraint_creation_basic_max(self) -> None:
        """Test creating moon phase constraint with max_illumination."""
        constraint = MoonPhaseConstraint(max_illumination=0.3)
        assert constraint.max_illumination == 0.3

    def test_moon_phase_constraint_creation_basic_min_none(self) -> None:
        """Test creating moon phase constraint with max_illumination has min None."""
        constraint = MoonPhaseConstraint(max_illumination=0.3)
        assert constraint.min_illumination is None

    def test_moon_phase_constraint_creation_full_max(self) -> None:
        """Test creating full moon phase constraint max."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.max_illumination == 0.8

    def test_moon_phase_constraint_creation_full_min(self) -> None:
        """Test creating full moon phase constraint min."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.min_illumination == 0.1

    def test_moon_phase_constraint_creation_full_min_distance(self) -> None:
        """Test creating full moon phase constraint min_distance."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.min_distance == 30.0

    def test_moon_phase_constraint_creation_full_max_distance(self) -> None:
        """Test creating full moon phase constraint max_distance."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.max_distance == 120.0

    def test_moon_phase_constraint_creation_full_enforce_below(self) -> None:
        """Test creating full moon phase constraint enforce_when_below_horizon."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.enforce_when_below_horizon is True

    def test_moon_phase_constraint_creation_full_moon_visibility(self) -> None:
        """Test creating full moon phase constraint moon_visibility."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.moon_visibility == "full"

    def test_moon_phase_constraint_validation_valid_max(self) -> None:
        """Test moon phase constraint parameter validation with valid max."""
        MoonPhaseConstraint(max_illumination=0.5)

    def test_moon_phase_constraint_validation_valid_both(self) -> None:
        """Test moon phase constraint parameter validation with valid both."""
        MoonPhaseConstraint(max_illumination=1.0, min_illumination=0.0)

    def test_moon_phase_constraint_validation_invalid_max_high(self) -> None:
        """Test moon phase constraint parameter validation with invalid max (>1.0)."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=1.5)

    def test_moon_phase_constraint_validation_invalid_min_low(self) -> None:
        """Test moon phase constraint parameter validation with invalid min (<0.0)."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, min_illumination=-0.1)

    def test_moon_phase_constraint_validation_min_greater_than_max(self) -> None:
        """Test moon phase constraint parameter validation with min > max."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.3, min_illumination=0.8)

    def test_moon_phase_constraint_validation_invalid_min_distance(self) -> None:
        """Test moon phase constraint parameter validation with invalid min_distance."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, min_distance=-10.0)

    def test_moon_phase_constraint_validation_max_distance_less_than_min(self) -> None:
        """Test moon phase constraint parameter validation with max_distance < min_distance."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(
                max_illumination=0.5, min_distance=50.0, max_distance=20.0
            )

    def test_moon_phase_constraint_validation_invalid_moon_visibility(self) -> None:
        """Test moon phase constraint parameter validation with invalid moon_visibility."""
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, moon_visibility="invalid")  # type: ignore[arg-type]

    def test_moon_phase_constraint_evaluation_type(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test moon phase constraint evaluation returns bool."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_moon_phase_constraint_batch_shape(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test batch moon phase constraint evaluation shape."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.shape == (3, len(tle_ephemeris.timestamp))

    def test_moon_phase_constraint_batch_dtype(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test batch moon phase constraint evaluation dtype."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.dtype == bool


class TestSAAConstraint:
    """Test SAAConstraint functionality."""

    @pytest.fixture
    def saa_polygon(self) -> list[tuple[float, float]]:
        """Simple rectangular SAA polygon for testing."""
        return [
            (-90.0, -50.0),  # Southwest
            (-40.0, -50.0),  # Southeast
            (-40.0, 0.0),  # Northeast
            (-90.0, 0.0),  # Northwest
        ]

    def test_saa_constraint_creation(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test creating SAA constraint."""
        constraint = SAAConstraint(polygon=saa_polygon)
        assert constraint.polygon == saa_polygon

    def test_saa_constraint_validation_valid_triangle(self) -> None:
        """Test SAA constraint parameter validation with valid triangle."""
        SAAConstraint(polygon=[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)])

    def test_saa_constraint_validation_invalid_few_vertices(self) -> None:
        """Test SAA constraint parameter validation with too few vertices."""
        with pytest.raises(ValueError):
            SAAConstraint(polygon=[(0.0, 0.0), (10.0, 0.0)])

    def test_saa_constraint_evaluation_type(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test SAA constraint evaluation returns bool."""
        constraint = SAAConstraint(polygon=saa_polygon)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_saa_constraint_batch_shape(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test batch SAA constraint evaluation shape."""
        constraint = SAAConstraint(polygon=saa_polygon)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.shape == (3, len(tle_ephemeris.timestamp))

    def test_saa_constraint_batch_dtype(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test batch SAA constraint evaluation dtype."""
        constraint = SAAConstraint(polygon=saa_polygon)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.dtype == bool

    def test_saa_point_in_polygon_logic_polygon_length(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test the point-in-polygon logic polygon length."""
        constraint = SAAConstraint(polygon=saa_polygon)
        assert len(constraint.polygon) == 4

    def test_saa_constraint_serialization_type(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test SAA constraint JSON serialization type."""
        constraint = SAAConstraint(polygon=saa_polygon)
        json_data = constraint.model_dump()
        assert json_data["type"] == "saa"

    def test_saa_constraint_serialization_polygon(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test SAA constraint JSON serialization polygon."""
        constraint = SAAConstraint(polygon=saa_polygon)
        json_data = constraint.model_dump()
        assert json_data["polygon"] == saa_polygon

    def test_saa_constraint_serialization_round_trip(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test SAA constraint JSON serialization round-trip."""
        constraint = SAAConstraint(polygon=saa_polygon)
        json_data = constraint.model_dump()
        constraint2 = SAAConstraint(**json_data)
        assert constraint2.polygon == constraint.polygon

    def test_saa_factory_method_serialization_type(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test the SAA factory method serialization type."""
        rust_constraint = rust_ephem.SAAConstraint(polygon=saa_polygon)
        json_str = rust_constraint.model_dump_json()
        assert '"type":"saa"' in json_str

    def test_saa_factory_method_serialization_polygon(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test the SAA factory method serialization polygon."""
        rust_constraint = rust_ephem.SAAConstraint(polygon=saa_polygon)
        json_str = rust_constraint.model_dump_json()
        assert '"polygon"' in json_str

    def test_saa_factory_method_round_trip(
        self, saa_polygon: list[tuple[float, float]]
    ) -> None:
        """Test the SAA factory method round-trip."""
        rust_constraint = rust_ephem.SAAConstraint(polygon=saa_polygon)
        json_str = rust_constraint.model_dump_json()
        rust_constraint2 = rust_ephem.SAAConstraint.model_validate_json(json_str)
        json_str2 = rust_constraint2.model_dump_json()
        assert json_str == json_str2


class TestConstraintIntegration:
    """Test integration of new constraints with existing functionality."""

    def test_combined_constraints_evaluation_type(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        ground_ephemeris: rust_ephem.GroundEphemeris,
    ) -> None:
        """Test combining new constraints evaluation returns bool."""
        from rust_ephem.constraints import AndConstraint, SunConstraint

        sun = SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        daytime = DaytimeConstraint()
        combined = AndConstraint(constraints=[sun, airmass, daytime])
        result = combined.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_constraint_not_operation_inverts(
        self, ground_ephemeris: rust_ephem.GroundEphemeris
    ) -> None:
        """Test NOT operation inverts the result."""

        airmass = AirmassConstraint(max_airmass=2.0)
        not_airmass = ~airmass
        result1 = airmass.evaluate(ground_ephemeris, target_ra=0.0, target_dec=0.0)
        result2 = not_airmass.evaluate(ground_ephemeris, target_ra=0.0, target_dec=0.0)

        assert np.array_equal(
            np.array(result1.constraint_array), ~np.array(result2.constraint_array)
        )

    def test_constraint_operator_overloads_and_type(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test operator overloads AND creates AndConstraint."""
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        combined = sun & airmass
        assert isinstance(combined, rust_ephem.constraints.AndConstraint)

    def test_constraint_operator_overloads_not_type(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test operator overloads NOT creates NotConstraint."""
        airmass = AirmassConstraint(max_airmass=2.0)
        not_airmass = ~airmass
        assert isinstance(not_airmass, rust_ephem.constraints.NotConstraint)

    def test_constraint_operator_overloads_combined_evaluation_type(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Test operator overloads combined evaluation returns bool."""
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        combined = sun & airmass
        result = combined.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)
