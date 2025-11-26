"""Test vectorized constraint evaluation with multiple RA/Dec positions."""

import datetime

import numpy as np
import pytest

from rust_ephem import (
    GroundEphemeris,
    SunConstraint,
)
from rust_ephem.constraints import EarthLimbConstraint


def test_sun_proximity_batch():
    """Test batch constraint evaluation with multiple targets."""
    # Create ephemeris for a ground station
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 3 time points
    )

    # Create sun proximity constraint (45 degree exclusion)
    constraint = SunConstraint(min_angle=45.0)

    # Test with 4 different target positions
    # RA in degrees, Dec in degrees
    target_ras = [0.0, 90.0, 180.0, 270.0]
    target_decs = [0.0, 30.0, -30.0, 60.0]

    # Evaluate for all targets at once
    result = constraint.in_constraint_batch(ephem, target_ras, target_decs)

    # Check result shape
    assert result.shape == (4, 3), f"Expected shape (4, 3), got {result.shape}"

    # Result should be boolean array
    assert result.dtype == bool, f"Expected bool dtype, got {result.dtype}"

    # Each row should correspond to one target
    for i in range(4):
        # Get single-target evaluation for comparison
        single_result = constraint.evaluate(
            ephem, target_ras[i], target_decs[i]
        ).constraint_array

        # Should match the corresponding row in batch result
        np.testing.assert_array_equal(
            result[i, :],
            single_result,
            err_msg=f"Batch result row {i} doesn't match single evaluation for target {i}",
        )


def test_batch_with_times_filter():
    """Test batch evaluation with time filtering."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 5, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 6 time points
    )

    constraint = SunConstraint(min_angle=45.0)

    # 3 targets
    target_ras = [0.0, 90.0, 180.0]
    target_decs = [0.0, 30.0, -30.0]

    # Only evaluate at specific times
    times = [
        datetime.datetime(2024, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2024, 1, 1, 3, 0, 0, tzinfo=datetime.timezone.utc),
    ]

    result = constraint.in_constraint_batch(ephem, target_ras, target_decs, times=times)

    # Should have 3 targets x 2 times
    assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"


def test_batch_with_indices_filter():
    """Test batch evaluation with index filtering."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 5, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 6 time points
    )

    constraint = SunConstraint(min_angle=45.0)

    # 2 targets
    target_ras = [45.0, 135.0]
    target_decs = [15.0, -15.0]

    # Only evaluate at specific indices
    indices = [0, 2, 4]

    result = constraint.in_constraint_batch(
        ephem, target_ras, target_decs, indices=indices
    )

    # Should have 2 targets x 3 times
    assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"


def test_batch_mismatched_array_lengths():
    """Test that mismatched RA/Dec array lengths raise an error."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,
    )

    constraint = SunConstraint(min_angle=45.0)

    # Mismatched lengths
    target_ras = [0.0, 90.0, 180.0]
    target_decs = [0.0, 30.0]  # Only 2 values

    with pytest.raises(
        ValueError, match="target_ras and target_decs must have the same length"
    ):
        constraint.in_constraint_batch(ephem, target_ras, target_decs)


def test_batch_empty_arrays():
    """Test batch evaluation with empty arrays."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,
    )

    constraint = SunConstraint(min_angle=45.0)

    # Empty arrays
    result = constraint.in_constraint_batch(ephem, [], [])

    # Should return empty array with correct shape (0, 3)
    assert result.shape == (0, 3), f"Expected shape (0, 3), got {result.shape}"


def test_batch_single_target():
    """Test batch evaluation with single target matches regular evaluation."""
    begin = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 1, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        35.0,  # latitude
        -120.0,  # longitude
        0.0,  # height
        begin,
        end,
        3600,
    )

    constraint = SunConstraint(min_angle=45.0)

    ra = 123.45
    dec = -23.67

    # Single target batch evaluation
    batch_result = constraint.in_constraint_batch(ephem, [ra], [dec])

    # Regular evaluation
    single_result = constraint.evaluate(ephem, ra, dec).constraint_array

    # Should match
    assert batch_result.shape == (1, 3)
    np.testing.assert_array_equal(batch_result[0, :], single_result)


def test_earth_limb_batch():
    """Test Earth limb constraint batch evaluation with optimized implementation."""
    # Create ephemeris for a ground station
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 18, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour steps -> 7 time points
    )

    # Create Earth limb constraint (20 degree margin)
    constraint = EarthLimbConstraint(min_angle=20.0)

    # Test with diverse target positions:
    # - Some near zenith (should pass)
    # - Some near horizon (should fail)
    # - Some at intermediate elevations
    target_ras = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    target_decs = [89.0, 60.0, 30.0, 0.0, -30.0, -60.0, 0.0, 45.0]

    # Evaluate for all targets at once
    result = constraint.in_constraint_batch(ephem, target_ras, target_decs)

    # Check result shape: 8 targets × 7 time points
    assert result.shape == (8, 7), f"Expected shape (8, 7), got {result.shape}"

    # Result should be boolean array
    assert result.dtype == bool, f"Expected bool dtype, got {result.dtype}"

    # Verify each row matches single-target evaluation
    for i in range(len(target_ras)):
        single_result = constraint.evaluate(
            ephem, target_ras[i], target_decs[i]
        ).constraint_array

        # Both batch and single results use same convention: True = violated
        np.testing.assert_array_equal(
            result[i, :],
            np.array(single_result),
            err_msg=f"Batch result row {i} doesn't match single evaluation for target {i} "
            f"(RA={target_ras[i]}, Dec={target_decs[i]})",
        )


def test_earth_limb_batch_with_max_angle():
    """Test Earth limb constraint batch evaluation with max_angle parameter."""
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 14, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        3600,  # step_size: 1 hour -> 3 time points
    )

    # Earth limb constraint with both min and max angles
    constraint = EarthLimbConstraint(min_angle=20.0, max_angle=70.0)

    target_ras = [0.0, 90.0, 180.0, 270.0]
    target_decs = [60.0, 30.0, -30.0, 0.0]

    result = constraint.in_constraint_batch(ephem, target_ras, target_decs)

    assert result.shape == (4, 3), f"Expected shape (4, 3), got {result.shape}"

    # Verify against single evaluations
    for i in range(len(target_ras)):
        single_result = constraint.evaluate(
            ephem, target_ras[i], target_decs[i]
        ).constraint_array

        np.testing.assert_array_equal(
            result[i, :],
            np.array(single_result),
            err_msg=f"Batch result with max_angle doesn't match for target {i}",
        )


def test_earth_limb_batch_large_scale():
    """Test Earth limb batch evaluation with many targets for performance validation."""
    begin = datetime.datetime(2024, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2024, 6, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
    ephem = GroundEphemeris(
        40.0,  # latitude
        -75.0,  # longitude
        100.0,  # height
        begin,
        end,
        600,  # step_size: 10 minutes -> 7 time points
    )

    constraint = EarthLimbConstraint(min_angle=20.0)

    # Generate a grid of RA/Dec positions
    n_ra = 36  # Every 10 degrees
    n_dec = 18  # Every 10 degrees from -90 to 90
    target_ras = []
    target_decs = []

    for ra in np.linspace(0, 350, n_ra):
        for dec in np.linspace(-90, 90, n_dec):
            target_ras.append(ra)
            target_decs.append(dec)

    # Should have 36 * 18 = 648 targets
    assert len(target_ras) == 648

    # Evaluate all targets at once
    result = constraint.in_constraint_batch(ephem, target_ras, target_decs)

    # Check shape: 648 targets × 7 time points
    assert result.shape == (648, 7), f"Expected shape (648, 7), got {result.shape}"

    # Spot check a few random targets
    np.random.seed(42)
    indices_to_check = np.random.choice(len(target_ras), size=5, replace=False)

    for idx in indices_to_check:
        single_result = constraint.evaluate(
            ephem, target_ras[idx], target_decs[idx]
        ).constraint_array

        np.testing.assert_array_equal(
            result[idx, :],
            np.array(single_result),
            err_msg=f"Large-scale batch result doesn't match for target {idx}",
        )
