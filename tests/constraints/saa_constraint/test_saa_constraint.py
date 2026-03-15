"""
Test suite for SAAConstraint.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

import rust_ephem
from rust_ephem.constraints import SAAConstraint

if TYPE_CHECKING:
    import rust_ephem


class TestSAAConstraint:
    """Test SAAConstraint functionality."""

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
        tle_ephemeris: "rust_ephem.TLEEphemeris",
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test SAA constraint evaluation returns bool."""
        constraint = SAAConstraint(polygon=saa_polygon)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_saa_constraint_batch_shape(
        self,
        tle_ephemeris: "rust_ephem.TLEEphemeris",
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
        tle_ephemeris: "rust_ephem.TLEEphemeris",
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test batch SAA constraint evaluation dtype."""
        constraint = SAAConstraint(polygon=saa_polygon)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.dtype == bool

    def test_saa_batch_matches_single(
        self,
        tle_ephemeris: "rust_ephem.TLEEphemeris",
        saa_polygon: list[tuple[float, float]],
    ) -> None:
        """Test that batch constraint evaluation matches single-target evaluations."""
        constraint = SAAConstraint(polygon=saa_polygon)

        # Test with multiple targets
        target_ras = [0.0, 90.0, 180.0, 270.0, 45.0]
        target_decs = [0.0, 30.0, -30.0, 60.0, -45.0]

        # Get batch results
        batch_result = constraint.in_constraint_batch(
            tle_ephemeris, target_ras, target_decs
        )

        # Verify shape
        assert batch_result.shape == (len(target_ras), len(tle_ephemeris.timestamp))

        # Compare each row with single evaluation
        for i in range(len(target_ras)):
            single_result = constraint.evaluate(
                tle_ephemeris, target_ras[i], target_decs[i]
            )

            # Single result's constraint_array should match batch row
            np.testing.assert_array_equal(
                batch_result[i, :],
                single_result.constraint_array,
                err_msg=f"Batch result row {i} doesn't match single evaluation for "
                f"target (RA={target_ras[i]}, Dec={target_decs[i]})",
            )

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
