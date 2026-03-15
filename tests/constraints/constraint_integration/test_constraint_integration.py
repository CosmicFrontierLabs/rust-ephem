"""Tests for constraint integration functionality."""

import numpy as np

import rust_ephem
from rust_ephem import AirmassConstraint


class TestConstraintIntegration:
    """Test integration of new constraints with existing functionality."""

    def test_combined_constraints_evaluation_type(
        self,
        tle_ephemeris: "rust_ephem.TLEEphemeris",
        ground_ephemeris: "rust_ephem.GroundEphemeris",
    ) -> None:
        """Test combining new constraints evaluation returns bool."""
        from rust_ephem.constraints import AndConstraint, SunConstraint

        sun = SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        daytime = rust_ephem.DaytimeConstraint()
        combined = AndConstraint(constraints=[sun, airmass, daytime])
        result = combined.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_constraint_not_operation_inverts(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
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
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test operator overloads AND creates AndConstraint."""
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        combined = sun & airmass
        assert isinstance(combined, rust_ephem.constraints.AndConstraint)

    def test_constraint_operator_overloads_not_type(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test operator overloads NOT creates NotConstraint."""
        airmass = AirmassConstraint(max_airmass=2.0)
        not_airmass = ~airmass
        assert isinstance(not_airmass, rust_ephem.constraints.NotConstraint)

    def test_constraint_operator_overloads_combined_evaluation_type(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test operator overloads combined evaluation returns bool."""
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        airmass = AirmassConstraint(max_airmass=2.0)
        combined = sun & airmass
        result = combined.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)
