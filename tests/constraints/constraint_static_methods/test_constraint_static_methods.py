#!/usr/bin/env python3
"""
Test suite for Constraint static methods that create constraints.
Tests body_proximity, airmass, alt_az, orbit_ram, and orbit_pole methods.
"""

import pytest

import rust_ephem


class TestConstraintStaticMethods:
    """Test Constraint static methods for creating constraints."""

    def test_body_proximity_creation(self) -> None:
        """Test creating a body proximity constraint."""
        constraint = rust_ephem.Constraint.body_proximity("Jupiter", 30.0)
        assert constraint is not None

    def test_body_proximity_with_max_angle(self) -> None:
        """Test creating a body proximity constraint with max angle."""
        constraint = rust_ephem.Constraint.body_proximity("Mars", 15.0, 90.0)
        assert constraint is not None

    def test_body_proximity_invalid_angles(self) -> None:
        """Test body proximity constraint with invalid angles."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.body_proximity("Sun", -10.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.body_proximity("Moon", 30.0, 200.0)

    def test_airmass_creation_max_only(self) -> None:
        """Test creating an airmass constraint with max only."""
        constraint = rust_ephem.Constraint.airmass(max_airmass=2.0)
        assert constraint is not None

    def test_airmass_creation_min_and_max(self) -> None:
        """Test creating an airmass constraint with min and max."""
        constraint = rust_ephem.Constraint.airmass(min_airmass=1.2, max_airmass=3.0)
        assert constraint is not None

    def test_airmass_invalid_values(self) -> None:
        """Test airmass constraint with invalid values."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.airmass(max_airmass=0.5)  # max_airmass must be > 0

        with pytest.raises(ValueError):
            rust_ephem.Constraint.airmass(
                min_airmass=0.8, max_airmass=2.0
            )  # min_airmass must be >= 1.0

    def test_alt_az_creation_basic(self) -> None:
        """Test creating a basic alt-az constraint."""
        constraint = rust_ephem.Constraint.alt_az(min_altitude=10.0, max_altitude=80.0)
        assert constraint is not None

    def test_alt_az_creation_with_azimuth(self) -> None:
        """Test creating an alt-az constraint with azimuth limits."""
        constraint = rust_ephem.Constraint.alt_az(
            min_altitude=20.0, max_altitude=70.0, min_azimuth=90.0, max_azimuth=270.0
        )
        assert constraint is not None

    def test_alt_az_creation_with_polygon(self) -> None:
        """Test creating an alt-az constraint with polygon."""
        polygon = [(10.0, 0.0), (10.0, 90.0), (80.0, 90.0), (80.0, 0.0)]
        constraint = rust_ephem.Constraint.alt_az(polygon=polygon)
        assert constraint is not None

    def test_alt_az_invalid_angles(self) -> None:
        """Test alt-az constraint with invalid angles."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.alt_az(min_altitude=-10.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.alt_az(max_altitude=100.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.alt_az(min_azimuth=-10.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.alt_az(max_azimuth=400.0)

    def test_alt_az_invalid_polygon(self) -> None:
        """Test alt-az constraint with invalid polygon."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.alt_az(
                polygon=[(10.0, 0.0), (20.0, 0.0)]
            )  # < 3 points

    def test_orbit_ram_creation(self) -> None:
        """Test creating an orbit RAM constraint."""
        constraint = rust_ephem.Constraint.orbit_ram(30.0)
        assert constraint is not None

    def test_orbit_ram_with_max_angle(self) -> None:
        """Test creating an orbit RAM constraint with max angle."""
        constraint = rust_ephem.Constraint.orbit_ram(15.0, 90.0)
        assert constraint is not None

    def test_orbit_ram_invalid_angles(self) -> None:
        """Test orbit RAM constraint with invalid angles."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.orbit_ram(-10.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.orbit_ram(30.0, 200.0)

    def test_orbit_pole_creation(self) -> None:
        """Test creating an orbit pole constraint."""
        constraint = rust_ephem.Constraint.orbit_pole(45.0)
        assert constraint is not None

    def test_orbit_pole_with_max_angle(self) -> None:
        """Test creating an orbit pole constraint with max angle."""
        constraint = rust_ephem.Constraint.orbit_pole(20.0, 120.0)
        assert constraint is not None

    def test_orbit_pole_with_earth_limb_pole(self) -> None:
        """Test creating an orbit pole constraint with earth limb pole."""
        constraint = rust_ephem.Constraint.orbit_pole(30.0, earth_limb_pole=True)
        assert constraint is not None

    def test_orbit_pole_invalid_angles(self) -> None:
        """Test orbit pole constraint with invalid angles."""
        with pytest.raises(ValueError):
            rust_ephem.Constraint.orbit_pole(-10.0)

        with pytest.raises(ValueError):
            rust_ephem.Constraint.orbit_pole(30.0, 200.0)
