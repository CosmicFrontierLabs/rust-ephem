"""
Test suite for AirmassConstraint.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from rust_ephem.constraints import AirmassConstraint

if TYPE_CHECKING:
    import rust_ephem


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
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test airmass constraint evaluation with target at zenith."""
        constraint = AirmassConstraint(max_airmass=1.5)
        result = constraint.evaluate(ground_ephemeris, target_ra=0.0, target_dec=35.0)
        assert result.all_satisfied

    def test_airmass_constraint_evaluation_horizon_violated(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test airmass constraint evaluation with target at horizon."""
        constraint = AirmassConstraint(max_airmass=1.5)
        result = constraint.evaluate(ground_ephemeris, target_ra=0.0, target_dec=-27.0)
        assert not result.all_satisfied

    def test_airmass_constraint_batch_shape(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
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
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test batch airmass constraint evaluation target 0 satisfied."""
        constraint = AirmassConstraint(max_airmass=1.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert np.all(~result[0, :])  # All False = all satisfied (not violated)

    # def test_airmass_constraint_batch_target1_satisfied(
    #     self, ground_ephemeris: "rust_ephem.GroundEphemeris"
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
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test batch airmass constraint evaluation target 2 has some violations."""
        constraint = AirmassConstraint(max_airmass=1.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [35.0, 35.0, -27.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert np.any(result[2, :])  # Some True = some violated

    def test_airmass_batch_matches_single(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test that batch constraint evaluation matches single-target evaluations."""
        constraint = AirmassConstraint(max_airmass=2.5, min_airmass=1.2)

        # Test with multiple targets
        target_ras = [0.0, 45.0, 90.0, 180.0, 270.0]
        target_decs = [0.0, 30.0, -30.0, 60.0, -60.0]

        # Get batch results
        batch_result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )

        # Verify shape
        assert batch_result.shape == (len(target_ras), len(ground_ephemeris.timestamp))

        # Compare each row with single evaluation
        for i in range(len(target_ras)):
            single_result = constraint.evaluate(
                ground_ephemeris, target_ras[i], target_decs[i]
            )

            # Single result's constraint_array should match batch row
            np.testing.assert_array_equal(
                batch_result[i, :],
                single_result.constraint_array,
                err_msg=f"Batch result row {i} doesn't match single evaluation for "
                f"target (RA={target_ras[i]}, Dec={target_decs[i]})",
            )
