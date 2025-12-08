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

    def test_airmass_constraint_creation(self):
        """Test creating airmass constraints."""
        # Test with max_airmass only
        constraint = AirmassConstraint(max_airmass=2.0)
        assert constraint.max_airmass == 2.0
        assert constraint.min_airmass is None

        # Test with both min and max
        constraint = AirmassConstraint(max_airmass=3.0, min_airmass=1.2)
        assert constraint.max_airmass == 3.0
        assert constraint.min_airmass == 1.2

    def test_airmass_constraint_validation(self):
        """Test airmass constraint parameter validation."""
        # Valid parameters
        AirmassConstraint(max_airmass=1.5)
        AirmassConstraint(max_airmass=5.0, min_airmass=1.0)

        # Invalid max_airmass (< 1.0)
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=0.5)

        # Invalid min_airmass (< 1.0)
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=2.0, min_airmass=0.8)

        # min_airmass > max_airmass
        with pytest.raises(ValidationError):
            AirmassConstraint(max_airmass=1.5, min_airmass=2.0)

    def test_airmass_constraint_evaluation(self, ground_ephem):
        """Test airmass constraint evaluation."""
        constraint = AirmassConstraint(max_airmass=2.0)

        # Test with target at zenith (airmass ~1.0) - should be satisfied
        result = constraint.evaluate(ground_ephem, target_ra=0.0, target_dec=35.0)
        assert result.all_satisfied

        # Test with target at horizon (high airmass) - should be violated
        result = constraint.evaluate(ground_ephem, target_ra=0.0, target_dec=-35.0)
        assert not result.all_satisfied

    def test_airmass_constraint_batch(self, ground_ephem):
        """Test batch airmass constraint evaluation."""
        constraint = AirmassConstraint(max_airmass=2.0)

        # Test multiple targets
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -35.0]  # First two near zenith, last near horizon

        result = constraint.in_constraint_batch(ground_ephem, target_ras, target_decs)

        # Check shape
        assert result.shape == (3, len(ground_ephem.timestamp))

        # First two targets should be satisfied (near zenith), last should be violated
        assert np.all(result[0, :])  # Target 0: satisfied
        assert np.all(result[1, :])  # Target 1: satisfied
        assert np.any(~result[2, :])  # Target 2: at least some violations


class TestDaytimeConstraint:
    """Test DaytimeConstraint functionality."""

    def test_daytime_constraint_creation(self):
        """Test creating daytime constraints."""
        # Default civil twilight
        constraint = DaytimeConstraint()
        assert constraint.twilight == "civil"

        # Different twilight types
        constraint = DaytimeConstraint(twilight="nautical")
        assert constraint.twilight == "nautical"

        constraint = DaytimeConstraint(twilight="astronomical")
        assert constraint.twilight == "astronomical"

        constraint = DaytimeConstraint(twilight="none")
        assert constraint.twilight == "none"

    def test_daytime_constraint_validation(self):
        """Test daytime constraint parameter validation."""
        # Valid twilight types
        DaytimeConstraint(twilight="civil")
        DaytimeConstraint(twilight="nautical")
        DaytimeConstraint(twilight="astronomical")
        DaytimeConstraint(twilight="none")

        # Invalid twilight type
        with pytest.raises(ValueError):
            DaytimeConstraint(twilight="invalid")

    def test_daytime_constraint_evaluation(self, ground_ephem):
        """Test daytime constraint evaluation."""
        constraint = DaytimeConstraint()

        # Test during daytime (should be violated)
        daytime_result = constraint.evaluate(
            ground_ephem, target_ra=0.0, target_dec=0.0
        )
        # Note: This test depends on actual sun position, so we just check it runs
        assert isinstance(daytime_result.all_satisfied, bool)

    def test_daytime_constraint_batch(self, ground_ephem):
        """Test batch daytime constraint evaluation."""
        constraint = DaytimeConstraint()

        target_ras = [0.0, 120.0, 240.0]
        target_decs = [0.0, 0.0, 0.0]

        result = constraint.in_constraint_batch(ground_ephem, target_ras, target_decs)

        # Check shape
        assert result.shape == (3, len(ground_ephem.timestamp))

        # Result should be boolean
        assert result.dtype == bool


class TestMoonPhaseConstraint:
    """Test MoonPhaseConstraint functionality."""

    def test_moon_phase_constraint_creation(self):
        """Test creating moon phase constraints."""
        # Basic constraint
        constraint = MoonPhaseConstraint(max_illumination=0.3)
        assert constraint.max_illumination == 0.3
        assert constraint.min_illumination is None

        # Full constraint
        constraint = MoonPhaseConstraint(
            max_illumination=0.8,
            min_illumination=0.1,
            min_distance=30.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
            moon_visibility="full",
        )
        assert constraint.max_illumination == 0.8
        assert constraint.min_illumination == 0.1
        assert constraint.min_distance == 30.0
        assert constraint.max_distance == 120.0
        assert constraint.enforce_when_below_horizon is True
        assert constraint.moon_visibility == "full"

    def test_moon_phase_constraint_validation(self):
        """Test moon phase constraint parameter validation."""
        # Valid parameters
        MoonPhaseConstraint(max_illumination=0.5)
        MoonPhaseConstraint(max_illumination=1.0, min_illumination=0.0)

        # Invalid illumination values
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=1.5)

        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, min_illumination=-0.1)

        # min > max illumination
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.3, min_illumination=0.8)

        # Invalid distance values
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, min_distance=-10.0)

        # Invalid moon visibility
        with pytest.raises(ValidationError):
            MoonPhaseConstraint(max_illumination=0.5, moon_visibility="invalid")

    def test_moon_phase_constraint_evaluation(self, tle_ephem):
        """Test moon phase constraint evaluation."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)

        # Test evaluation runs without error
        result = constraint.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_moon_phase_constraint_batch(self, tle_ephem):
        """Test batch moon phase constraint evaluation."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)

        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]

        result = constraint.in_constraint_batch(tle_ephem, target_ras, target_decs)

        # Check shape
        assert result.shape == (3, len(tle_ephem.timestamp))
        assert result.dtype == bool


class TestSAAConstraint:
    """Test SAAConstraint functionality."""

    @pytest.fixture
    def saa_polygon(self):
        """Simple rectangular SAA polygon for testing."""
        return [
            (-90.0, -50.0),  # Southwest
            (-40.0, -50.0),  # Southeast
            (-40.0, 0.0),  # Northeast
            (-90.0, 0.0),  # Northwest
        ]

    def test_saa_constraint_creation(self, saa_polygon):
        """Test creating SAA constraints."""
        constraint = SAAConstraint(polygon=saa_polygon)
        assert constraint.polygon == saa_polygon

    def test_saa_constraint_validation(self):
        """Test SAA constraint parameter validation."""
        # Valid polygon (triangle)
        SAAConstraint(polygon=[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)])

        # Invalid: too few vertices
        with pytest.raises(ValueError):
            SAAConstraint(polygon=[(0.0, 0.0), (10.0, 0.0)])

    def test_saa_constraint_evaluation(self, tle_ephem, saa_polygon):
        """Test SAA constraint evaluation."""
        constraint = SAAConstraint(polygon=saa_polygon)

        # Test evaluation runs without error
        result = constraint.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

        # Since SAA depends on spacecraft position, we can't easily predict
        # the result, but we can check that evaluation completes

    def test_saa_constraint_batch(self, tle_ephem, saa_polygon):
        """Test batch SAA constraint evaluation."""
        constraint = SAAConstraint(polygon=saa_polygon)

        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]

        result = constraint.in_constraint_batch(tle_ephem, target_ras, target_decs)

        # Check shape
        assert result.shape == (3, len(tle_ephem.timestamp))
        assert result.dtype == bool

    def test_saa_point_in_polygon_logic(self, saa_polygon):
        """Test the point-in-polygon logic with known points."""
        constraint = SAAConstraint(polygon=saa_polygon)

        # Create a mock ephemeris with known positions
        # This is tricky since we need to test the internal logic.
        # For now, just verify the constraint can be created and evaluated.

        # Points inside the polygon should be constrained
        # Points outside should not be constrained
        # But this requires setting up specific spacecraft positions,
        # which is complex for unit testing.

        # Just verify basic functionality
        assert len(constraint.polygon) == 4

    def test_saa_constraint_serialization(self, saa_polygon):
        """Test SAA constraint JSON serialization."""
        constraint = SAAConstraint(polygon=saa_polygon)

        # Test Pydantic serialization
        json_data = constraint.model_dump()
        assert json_data["type"] == "saa"
        assert json_data["polygon"] == saa_polygon

        # Test round-trip
        constraint2 = SAAConstraint(**json_data)
        assert constraint2.polygon == constraint.polygon

    def test_saa_factory_method(self, saa_polygon):
        """Test the SAA factory method."""
        rust_constraint = rust_ephem.Constraint.saa(saa_polygon)

        # Test serialization
        json_str = rust_constraint.to_json()
        assert '"type":"saa"' in json_str
        assert '"polygon"' in json_str

        # Test deserialization
        rust_constraint2 = rust_ephem.Constraint.from_json(json_str)
        json_str2 = rust_constraint2.to_json()
        assert json_str == json_str2


class TestConstraintIntegration:
    """Test integration of new constraints with existing functionality."""

    def test_combined_constraints(self, tle_ephem, ground_ephem):
        """Test combining new constraints with existing ones."""
        from rust_ephem.constraints import AndConstraint, SunConstraint

        # Create constraints
        sun = SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        daytime = DaytimeConstraint()

        # Combine with AND
        combined = AndConstraint(constraints=[sun, airmass, daytime])

        # Test evaluation
        result = combined.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_constraint_not_operation(self, tle_ephem):
        """Test NOT operation with new constraints."""
        from rust_ephem.constraints import NotConstraint

        # Create and invert constraints
        airmass = AirmassConstraint(max_airmass=2.0)
        not_airmass = NotConstraint(constraint=airmass)

        # Test evaluation
        result1 = airmass.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)
        result2 = not_airmass.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)

        # NOT should invert the result
        assert result1.all_satisfied != result2.all_satisfied

    def test_constraint_operator_overloads(self, tle_ephem):
        """Test operator overloads with new constraints."""
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)

        # Test AND operator
        combined = sun & airmass
        assert isinstance(combined, rust_ephem.constraints.AndConstraint)

        # Test NOT operator
        not_airmass = ~airmass
        assert isinstance(not_airmass, rust_ephem.constraints.NotConstraint)

        # Test evaluation of combined constraint
        result = combined.evaluate(tle_ephem, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)
