"""
Test suite for DaytimeConstraint.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rust_ephem.constraints import DaytimeConstraint

if TYPE_CHECKING:
    import rust_ephem


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
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test daytime constraint evaluation returns bool."""
        constraint = DaytimeConstraint()
        daytime_result = constraint.evaluate(
            ground_ephemeris, target_ra=0.0, target_dec=0.0
        )
        assert isinstance(daytime_result.all_satisfied, bool)

    def test_daytime_constraint_batch_shape(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
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
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test batch daytime constraint evaluation dtype."""
        constraint = DaytimeConstraint()
        target_ras = [0.0, 120.0, 240.0]
        target_decs = [0.0, 0.0, 0.0]
        result = constraint.in_constraint_batch(
            ground_ephemeris, target_ras, target_decs
        )
        assert result.dtype == bool

    def test_daytime_batch_matches_single(
        self, ground_ephemeris: "rust_ephem.GroundEphemeris"
    ) -> None:
        """Test that batch constraint evaluation matches single-target evaluations."""
        constraint = DaytimeConstraint(twilight="civil")

        # Test with multiple targets
        target_ras = [0.0, 60.0, 120.0, 180.0, 300.0]
        target_decs = [0.0, 45.0, -45.0, 90.0, -30.0]

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
