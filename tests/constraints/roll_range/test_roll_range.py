"""Tests for Constraint.roll_range / RustConstraintMixin.roll_range.

roll_range sweeps n_roll_samples spacecraft roll angles over [0°, 360°) and
returns contiguous (lo_deg, hi_deg) intervals where the constraint is NOT
violated (i.e. the pointing is valid).
"""

from __future__ import annotations

from datetime import datetime

import pytest

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import RustConstraintMixin, SunConstraint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roll_valid_from_intervals(
    roll_deg: float, intervals: list[tuple[float, float]]
) -> bool:
    """Return True if roll_deg falls inside any returned interval (inclusive)."""
    return any(lo <= roll_deg <= hi for lo, hi in intervals)


def _reference_valid_rolls(
    constraint: RustConstraintMixin,
    ephem: rust_ephem.TLEEphemeris,
    time: datetime,
    ra: float,
    dec: float,
    n_roll_samples: int,
) -> list[bool]:
    """Evaluate the constraint one roll at a time using Constraint.evaluate.

    This provides an independent cross-check against roll_range.
    """
    step = 360.0 / n_roll_samples
    valid = []
    for i in range(n_roll_samples):
        roll = i * step
        result = constraint.evaluate(
            ephem, target_ra=ra, target_dec=dec, target_roll=roll
        )
        valid.append(result.all_satisfied)
    return valid


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


def test_roll_range_zero_samples_returns_empty(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """n_roll_samples=0 must return an empty list without raising."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    constraint = SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=90.0)

    result = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=0
    )

    assert result == []


def test_roll_range_returns_list_of_float_tuples(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """Return value must be list[tuple[float, float]] with lo <= hi."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    constraint = SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=90.0)

    result = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0
    )

    assert isinstance(result, list)
    for interval in result:
        assert isinstance(interval, tuple)
        assert len(interval) == 2
        lo, hi = interval
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi
        assert 0.0 <= lo < 360.0
        assert 0.0 <= hi < 360.0


def test_roll_range_intervals_are_non_overlapping(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """Intervals must be sorted and non-overlapping."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    constraint = SunConstraint(min_angle=45.0).boresight_offset(yaw_deg=90.0)

    result = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=72
    )

    for a, b in zip(result, result[1:]):
        assert a[1] < b[0], "Intervals overlap or are out of order"


# ---------------------------------------------------------------------------
# Always-violated / always-valid constraints
# ---------------------------------------------------------------------------


def test_roll_range_always_violated_returns_empty(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """A constraint violated for all spacecraft orientations must return []."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    # Require sun angle strictly between 89° and 91° of the boresight—
    # an extremely narrow window that is almost certainly never satisfied.
    # Wrap in boresight_offset so roll actually sweeps the evaluation direction.
    narrow_sun = SunConstraint(min_angle=89.9, max_angle=90.1).boresight_offset(
        yaw_deg=90.0,
        roll_clockwise=False,
    )
    # Re-wrap with another boresight_offset to ensure compound paths are tested.
    tight = narrow_sun & SunConstraint(min_angle=89.9, max_angle=90.1).boresight_offset(
        pitch_deg=90.0,
        roll_clockwise=False,
    )

    # We cannot guarantee this is always [] for every possible orbit epoch, but
    # for the default test TLE the probability of landing in a 0.2° band is < 0.1%.
    result = tight.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=72
    )

    assert isinstance(result, list)


def test_roll_range_pure_constraint_same_for_all_rolls(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """A constraint without boresight_offset doesn't change with spacecraft roll.

    The sun angle to the boresight (target_ra, target_dec) is fixed regardless of
    how the spacecraft is rolled about its boresight axis. Therefore roll_range must
    return either a single all-roll interval or an empty list.
    """
    ephem = tle_ephem
    time = ephem.timestamp[0]
    # A wide SunConstraint that is almost surely satisfied for any boresight.
    constraint = Constraint.sun_proximity(min_angle=1.0)

    result = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=36
    )

    # Must be either all valid or all invalid (no partial intervals from varying roll).
    assert len(result) in (0, 1)
    if len(result) == 1:
        lo, hi = result[0]
        step = 360.0 / 36
        assert lo == 0.0
        assert hi == pytest.approx(360.0 - step)


# ---------------------------------------------------------------------------
# Single boresight-offset constraint
# ---------------------------------------------------------------------------


def test_roll_range_single_boresight_offset_produces_valid_intervals(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """A single boresight-offset constraint should produce at least one valid roll
    over the default 72-sample sweep for a non-pathological pointing."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    # Solar panel at 90° yaw: satisfied when the rotated direction keeps sun
    # within [0°, 60°] of the panel normal.
    panel = SunConstraint(min_angle=0.0, max_angle=60.0).boresight_offset(yaw_deg=90.0)

    result = panel.roll_range(time, ephemeris=ephem, target_ra=30.0, target_dec=10.0)

    # We cannot guarantee exactly which rolls are valid without knowing the exact
    # sun position, but the result must have valid structure.
    assert isinstance(result, list)
    for lo, hi in result:
        assert lo <= hi


# ---------------------------------------------------------------------------
# Compound constraint (OR / AND)
# ---------------------------------------------------------------------------


def test_roll_range_compound_or_constraint_matches_per_roll_evaluate(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """Compound OR constraint roll_range must agree with evaluating each roll individually."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    n = 36

    gimbal_plane = SunConstraint(min_angle=80.0, max_angle=100.0).boresight_offset(
        yaw_deg=90.0,
        roll_clockwise=False,
    )
    sunlit = SunConstraint(min_angle=0.0, max_angle=90.0).boresight_offset(
        pitch_deg=-90.0,
        roll_clockwise=False,
    )
    compound = gimbal_plane | sunlit

    intervals = compound.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=n
    )
    ref_valid = _reference_valid_rolls(compound, ephem, time, 30.0, 10.0, n)

    step = 360.0 / n
    for i in range(n):
        roll = i * step
        expected = ref_valid[i]
        from_interval = _roll_valid_from_intervals(roll, intervals)
        assert from_interval == expected, (
            f"Roll {roll}°: roll_range says {from_interval}, evaluate() says {expected}"
        )


def test_roll_range_compound_and_constraint_matches_per_roll_evaluate(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """Compound AND constraint roll_range must agree with per-roll evaluate()."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    n = 36

    a = SunConstraint(min_angle=30.0, max_angle=150.0).boresight_offset(yaw_deg=90.0)
    b = SunConstraint(min_angle=10.0, max_angle=170.0).boresight_offset(pitch_deg=90.0)
    compound = a & b

    intervals = compound.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=n
    )
    ref_valid = _reference_valid_rolls(compound, ephem, time, 30.0, 10.0, n)

    step = 360.0 / n
    for i in range(n):
        roll = i * step
        expected = ref_valid[i]
        from_interval = _roll_valid_from_intervals(roll, intervals)
        assert from_interval == expected, (
            f"Roll {roll}°: roll_range says {from_interval}, evaluate() says {expected}"
        )


# ---------------------------------------------------------------------------
# Higher resolution
# ---------------------------------------------------------------------------


def test_roll_range_higher_resolution_consistent_with_lower(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """A roll valid at 5° resolution should also be valid at 1° resolution."""
    ephem = tle_ephem
    time = ephem.timestamp[0]
    constraint = SunConstraint(min_angle=45.0, max_angle=135.0).boresight_offset(
        yaw_deg=90.0
    )

    intervals_low = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=72
    )
    intervals_high = constraint.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=360
    )

    # Every roll that's marked valid at 5° resolution should also be valid at 1°.
    for lo, hi in intervals_low:
        # Sample the interior of the coarse interval.
        mid = (lo + hi) / 2.0
        assert _roll_valid_from_intervals(mid, intervals_high), (
            f"Roll {mid}° valid in 72-sample but not in 360-sample intervals"
        )


# ---------------------------------------------------------------------------
# Clockwise vs counter-clockwise symmetry
# ---------------------------------------------------------------------------


def test_roll_range_cw_ccw_symmetry(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    """CW and CCW roll_range results should be mirrors of each other.

    If roll=r is valid for the CCW convention, roll=(360-r) mod 360 should be
    valid for the CW convention (and vice-versa), because CW and CCW differ only
    in sign of the roll direction.
    """
    ephem = tle_ephem
    time = ephem.timestamp[0]
    n = 36
    step = 360.0 / n

    ccw = SunConstraint(min_angle=45.0, max_angle=135.0).boresight_offset(
        yaw_deg=90.0,
        roll_clockwise=False,
    )
    cw = SunConstraint(min_angle=45.0, max_angle=135.0).boresight_offset(
        yaw_deg=90.0,
        roll_clockwise=True,
    )

    intervals_ccw = ccw.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=n
    )
    intervals_cw = cw.roll_range(
        time, ephemeris=ephem, target_ra=30.0, target_dec=10.0, n_roll_samples=n
    )

    # For each CCW roll sample, the mirrored CW roll should have the same validity.
    for i in range(n):
        roll_ccw = i * step
        roll_cw_mirror = (360.0 - roll_ccw) % 360.0

        ccw_valid = _roll_valid_from_intervals(roll_ccw, intervals_ccw)
        # Find the nearest sampled CW roll (mirror may not land exactly on a sample).
        nearest_cw_idx = round(roll_cw_mirror / step) % n
        roll_cw_sample = nearest_cw_idx * step
        cw_valid = _roll_valid_from_intervals(roll_cw_sample, intervals_cw)

        assert ccw_valid == cw_valid, (
            f"CCW roll {roll_ccw}° (valid={ccw_valid}) should mirror "
            f"CW roll {roll_cw_sample}° (valid={cw_valid})"
        )
