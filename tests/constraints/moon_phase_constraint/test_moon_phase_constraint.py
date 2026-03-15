"""
Test suite for MoonPhaseConstraint.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

import rust_ephem
from rust_ephem.constraints import MoonPhaseConstraint

if TYPE_CHECKING:
    import rust_ephem


def moon_ra_dec_deg(ephem: "rust_ephem.Ephemeris", index: int) -> tuple[float, float]:
    """Get moon RA/Dec in degrees for a given ephemeris index."""
    moon_pos = ephem.moon_pv.position[index]
    obs_pos = ephem.gcrs_pv.position[index]
    rel = moon_pos - obs_pos
    dist = np.linalg.norm(rel)
    ra = np.degrees(np.arctan2(rel[1], rel[0]))
    if ra < 0.0:
        ra += 360.0
    dec = np.degrees(np.arcsin(rel[2] / dist))
    return ra, dec


def moon_altitude_deg(ephem: "rust_ephem.Ephemeris", index: int) -> float:
    """Get moon altitude in degrees for a given ephemeris index."""
    ra, dec = moon_ra_dec_deg(ephem, index)
    altaz = ephem.radec_to_altaz(ra, dec, time_indices=[index])
    return float(altaz[0][0])


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
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test moon phase constraint evaluation returns bool."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        result = constraint.evaluate(tle_ephemeris, target_ra=0.0, target_dec=0.0)
        assert isinstance(result.all_satisfied, bool)

    def test_moon_phase_constraint_batch_shape(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test batch moon phase constraint evaluation shape."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.shape == (3, len(tle_ephemeris.timestamp))

    def test_moon_phase_constraint_batch_dtype(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test batch moon phase constraint evaluation dtype."""
        constraint = MoonPhaseConstraint(max_illumination=0.5)
        target_ras = [0.0, 90.0, 180.0]
        target_decs = [0.0, 30.0, -30.0]
        result = constraint.in_constraint_batch(tle_ephemeris, target_ras, target_decs)
        assert result.dtype == bool

    def test_moon_phase_constraint_illumination_violation(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Illumination threshold should trigger a violation."""
        illumination = tle_ephemeris.moon_illumination(time_indices=[0])[0]
        if illumination <= 0.02:
            constraint = MoonPhaseConstraint(
                max_illumination=1.0,
                min_illumination=illumination + 0.05,
                enforce_when_below_horizon=True,
            )
        else:
            constraint = MoonPhaseConstraint(
                max_illumination=max(illumination - 0.02, 0.0),
                enforce_when_below_horizon=True,
            )
        result = constraint.evaluate(
            tle_ephemeris, target_ra=0.0, target_dec=0.0, indices=0
        )
        assert result.all_satisfied is False

    def test_moon_phase_constraint_min_distance_violation(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Min distance should trigger a violation when target is at the Moon."""
        ra, dec = moon_ra_dec_deg(tle_ephemeris, 0)
        constraint = MoonPhaseConstraint(
            max_illumination=1.0,
            min_distance=1.0,
            enforce_when_below_horizon=True,
        )
        result = constraint.evaluate(
            tle_ephemeris, target_ra=ra, target_dec=dec, indices=0
        )
        assert result.all_satisfied is False

    def test_moon_phase_constraint_max_distance_violation(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Max distance should trigger a violation for a far target."""
        ra, dec = moon_ra_dec_deg(tle_ephemeris, 0)
        far_ra = (ra + 180.0) % 360.0
        far_dec = -dec
        constraint = MoonPhaseConstraint(
            max_illumination=1.0,
            max_distance=90.0,
            enforce_when_below_horizon=True,
        )
        result = constraint.evaluate(
            tle_ephemeris, target_ra=far_ra, target_dec=far_dec, indices=0
        )
        assert result.all_satisfied is False

    def test_moon_phase_constraint_below_horizon_skip(self) -> None:
        """Below-horizon Moon should be skipped when enforcement is disabled."""
        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = begin + timedelta(hours=24)
        ground_ephemeris = rust_ephem.GroundEphemeris(
            latitude=34.0,
            longitude=-118.0,
            height=100.0,
            begin=begin,
            end=end,
            step_size=600,
        )
        below_index = None
        for idx in range(len(ground_ephemeris.timestamp)):
            if moon_altitude_deg(ground_ephemeris, idx) < 0.0:
                below_index = idx
                break
        assert below_index is not None, "Expected below-horizon Moon sample"

        illumination = ground_ephemeris.moon_illumination(time_indices=[below_index])[0]
        max_illum = max(illumination - 0.02, 0.0)

        skip_constraint = MoonPhaseConstraint(
            max_illumination=max_illum,
            enforce_when_below_horizon=False,
        )
        enforced_constraint = MoonPhaseConstraint(
            max_illumination=max_illum,
            enforce_when_below_horizon=True,
        )

        skipped = skip_constraint.evaluate(
            ground_ephemeris, target_ra=0.0, target_dec=0.0, indices=below_index
        )
        enforced = enforced_constraint.evaluate(
            ground_ephemeris, target_ra=0.0, target_dec=0.0, indices=below_index
        )

        assert skipped.all_satisfied is True
        assert enforced.all_satisfied is False

    def test_moon_phase_batch_matches_single(
        self, tle_ephemeris: "rust_ephem.TLEEphemeris"
    ) -> None:
        """Test that batch constraint evaluation matches single-target evaluations."""
        constraint = MoonPhaseConstraint(
            max_illumination=0.6,
            min_distance=10.0,
            max_distance=120.0,
            enforce_when_below_horizon=True,
        )

        # Test with multiple targets
        target_ras = [0.0, 45.0, 90.0, 180.0, 270.0]
        target_decs = [0.0, 30.0, -30.0, 60.0, -60.0]

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
