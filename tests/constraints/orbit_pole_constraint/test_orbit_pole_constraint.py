"""Tests for OrbitPoleConstraint functionality."""

import pytest
from pydantic import ValidationError

import rust_ephem
from rust_ephem import OrbitPoleConstraint


class TestOrbitPoleConstraint:
    """Test OrbitPoleConstraint functionality."""

    def test_orbit_pole_constraint_creation_basic(self) -> None:
        """Test creating orbit pole constraint with basic parameters."""
        constraint = OrbitPoleConstraint(min_angle=45.0)
        assert constraint.min_angle == 45.0

    def test_orbit_pole_constraint_creation_with_max(self) -> None:
        """Test creating orbit pole constraint with max angle."""
        constraint = OrbitPoleConstraint(min_angle=20.0, max_angle=120.0)
        assert constraint.max_angle == 120.0

    def test_orbit_pole_constraint_creation_with_earth_limb_pole(self) -> None:
        """Test creating orbit pole constraint with earth limb pole."""
        constraint = OrbitPoleConstraint(min_angle=30.0, earth_limb_pole=True)
        assert constraint.earth_limb_pole is True

    def test_orbit_pole_constraint_validation_valid_basic(self) -> None:
        """Test orbit pole constraint parameter validation with valid basic."""
        OrbitPoleConstraint(min_angle=45.0)

    def test_orbit_pole_constraint_validation_invalid_angle_low(self) -> None:
        """Test orbit pole constraint parameter validation with invalid low angle."""
        with pytest.raises(ValidationError):
            OrbitPoleConstraint(min_angle=-10.0)

    def test_orbit_pole_constraint_validation_invalid_angle_high(self) -> None:
        """Test orbit pole constraint parameter validation with invalid high angle."""
        with pytest.raises(ValidationError):
            OrbitPoleConstraint(min_angle=30.0, max_angle=200.0)

    def test_orbit_pole_constraint_evaluation_type(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test orbit pole constraint evaluation returns bool."""
        constraint = OrbitPoleConstraint(min_angle=45.0)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_orbit_pole_constraint_batch_shape(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test batch orbit pole constraint evaluation shape."""
        constraint = OrbitPoleConstraint(min_angle=45.0)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.shape == (3, len(tle_ephemeris.timestamp))
