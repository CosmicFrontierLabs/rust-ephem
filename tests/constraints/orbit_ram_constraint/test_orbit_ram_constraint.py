"""Tests for OrbitRamConstraint functionality."""

import pytest
from pydantic import ValidationError

import rust_ephem
from rust_ephem import OrbitRamConstraint


class TestOrbitRamConstraint:
    """Test OrbitRamConstraint functionality."""

    def test_orbit_ram_constraint_creation_basic(self) -> None:
        """Test creating orbit RAM constraint with basic parameters."""
        constraint = OrbitRamConstraint(min_angle=30.0)
        assert constraint.min_angle == 30.0

    def test_orbit_ram_constraint_creation_with_max(self) -> None:
        """Test creating orbit RAM constraint with max angle."""
        constraint = OrbitRamConstraint(min_angle=15.0, max_angle=90.0)
        assert constraint.max_angle == 90.0

    def test_orbit_ram_constraint_validation_valid_basic(self) -> None:
        """Test orbit RAM constraint parameter validation with valid basic."""
        OrbitRamConstraint(min_angle=30.0)

    def test_orbit_ram_constraint_validation_invalid_angle_low(self) -> None:
        """Test orbit RAM constraint parameter validation with invalid low angle."""
        with pytest.raises(ValidationError):
            OrbitRamConstraint(min_angle=-10.0)

    def test_orbit_ram_constraint_validation_invalid_angle_high(self) -> None:
        """Test orbit RAM constraint parameter validation with invalid high angle."""
        with pytest.raises(ValidationError):
            OrbitRamConstraint(min_angle=30.0, max_angle=200.0)

    def test_orbit_ram_constraint_evaluation_type(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test orbit RAM constraint evaluation returns bool."""
        constraint = OrbitRamConstraint(min_angle=30.0)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_orbit_ram_constraint_batch_shape(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test batch orbit RAM constraint evaluation shape."""
        constraint = OrbitRamConstraint(min_angle=30.0)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.shape == (3, len(tle_ephemeris.timestamp))
