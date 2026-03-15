"""
Test suite for AltAzConstraint.
"""

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from rust_ephem.constraints import AltAzConstraint

if TYPE_CHECKING:
    import rust_ephem


class TestAltAzConstraint:
    """Test AltAzConstraint functionality."""

    def test_alt_az_constraint_creation_basic(self) -> None:
        """Test creating alt-az constraint with basic parameters."""
        constraint = AltAzConstraint(min_altitude=10.0, max_altitude=80.0)
        assert constraint.min_altitude == 10.0
        assert constraint.max_altitude == 80.0

    def test_alt_az_constraint_creation_with_azimuth(self) -> None:
        """Test creating alt-az constraint with azimuth limits."""
        constraint = AltAzConstraint(
            min_altitude=20.0, max_altitude=70.0, min_azimuth=90.0, max_azimuth=270.0
        )
        assert constraint.min_azimuth == 90.0
        assert constraint.max_azimuth == 270.0

    def test_alt_az_constraint_creation_with_polygon(self) -> None:
        """Test creating alt-az constraint with polygon."""
        polygon = [(10.0, 0.0), (10.0, 90.0), (80.0, 90.0), (80.0, 0.0)]
        constraint = AltAzConstraint(polygon=polygon)
        assert constraint.polygon == polygon

    def test_alt_az_constraint_validation_valid_basic(self) -> None:
        """Test alt-az constraint parameter validation with valid basic."""
        AltAzConstraint(min_altitude=10.0, max_altitude=80.0)

    def test_alt_az_constraint_validation_invalid_altitude_low(self) -> None:
        """Test alt-az constraint parameter validation with invalid low altitude."""
        with pytest.raises(ValidationError):
            AltAzConstraint(min_altitude=-10.0)

    def test_alt_az_constraint_validation_invalid_altitude_high(self) -> None:
        """Test alt-az constraint parameter validation with invalid high altitude."""
        with pytest.raises(ValidationError):
            AltAzConstraint(max_altitude=100.0)

    def test_alt_az_constraint_validation_invalid_azimuth_low(self) -> None:
        """Test alt-az constraint parameter validation with invalid low azimuth."""
        with pytest.raises(ValidationError):
            AltAzConstraint(min_altitude=10.0, min_azimuth=-10.0)

    def test_alt_az_constraint_validation_invalid_azimuth_high(self) -> None:
        """Test alt-az constraint parameter validation with invalid high azimuth."""
        with pytest.raises(ValidationError):
            AltAzConstraint(min_altitude=10.0, max_azimuth=400.0)

    def test_alt_az_constraint_validation_invalid_polygon_few_points(self) -> None:
        """Test alt-az constraint with invalid polygon - should still create but may fail later."""
        # Pydantic doesn't validate polygon, so this should create the constraint
        constraint = AltAzConstraint(polygon=[(10.0, 0.0), (20.0, 0.0)])
        assert constraint.polygon == [(10.0, 0.0), (20.0, 0.0)]

    def test_alt_az_constraint_evaluation_type(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test alt-az constraint evaluation returns bool."""
        constraint = AltAzConstraint(min_altitude=10.0, max_altitude=80.0)
        result = constraint.evaluate(ground_ephemeris, target_ra=0.0, target_dec=35.0)
        assert isinstance(result.all_satisfied, bool)

    def test_alt_az_constraint_batch_shape(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test batch alt-az constraint evaluation shape."""
        constraint = AltAzConstraint(min_altitude=10.0, max_altitude=80.0)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert result.shape == (3, len(ground_ephemeris.timestamp))
