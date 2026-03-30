"""Tests for Constraint.roll_range / RustConstraintMixin.roll_range.

roll_range sweeps n_roll_samples spacecraft roll angles over [0°, 360°) and
returns contiguous (lo_deg, hi_deg) intervals where the constraint is NOT
violated (i.e. the pointing is valid).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pytest

import rust_ephem
from rust_ephem.constraints import RustConstraintMixin, SunConstraint

# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


class TestRollRangeBasicContract:
    def test_roll_range_zero_samples_raises(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """n_roll_samples=0 must raise a ValueError."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        with pytest.raises(
            ValueError, match="n_roll_samples must be a positive integer"
        ):
            panel_constraint.roll_range(
                time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=0
            )

    def test_roll_range_returns_list(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Return value must be a list."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        assert isinstance(result, list)

    def test_roll_range_returns_list_of_tuples(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Return value must contain tuples."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for interval in result:
            assert isinstance(interval, tuple)

    def test_roll_range_returns_list_of_pairs(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Each interval must be a pair."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for interval in result:
            assert len(interval) == 2

    def test_roll_range_returns_list_of_floats(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Each interval element must be a float."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for interval in result:
            lo, hi = interval
            assert isinstance(lo, float)
            assert isinstance(hi, float)

    def test_roll_range_intervals_have_lo_le_hi(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Each interval must have lo <= hi."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for interval in result:
            lo, hi = interval
            assert lo <= hi

    def test_roll_range_intervals_in_valid_range(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Each interval must be in [0, 360) degrees."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for interval in result:
            lo, hi = interval
            assert 0.0 <= lo < 360.0
            assert 0.0 <= hi < 360.0

    def test_roll_range_intervals_are_non_overlapping(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Intervals must be sorted and non-overlapping."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=72
        )

        for a, b in zip(result, result[1:]):
            assert a[1] < b[0], "Intervals overlap or are out of order"


# ---------------------------------------------------------------------------
# Always-violated / always-valid constraints
# ---------------------------------------------------------------------------


class TestRollRangeEdgeCases:
    def test_roll_range_always_violated_returns_list(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        always_violated_constraint: RustConstraintMixin,
    ) -> None:
        """Always violated constraint should return a list."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = always_violated_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=72
        )

        assert isinstance(result, list)

    def test_roll_range_roll_invariant_returns_list(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        roll_invariant_constraint: rust_ephem.Constraint,
    ) -> None:
        """Roll-invariant constraint should return a list."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = roll_invariant_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=36
        )

        assert isinstance(result, list)

    def test_roll_range_roll_invariant_length(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        roll_invariant_constraint: rust_ephem.Constraint,
    ) -> None:
        """Roll-invariant constraint must return either empty list or single interval."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = roll_invariant_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=36
        )

        assert len(result) in (0, 1)

    def test_roll_range_roll_invariant_single_interval_bounds(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        roll_invariant_constraint: rust_ephem.Constraint,
    ) -> None:
        """Roll-invariant constraint single interval must cover full range."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = roll_invariant_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=36
        )

        if len(result) == 1:
            lo, hi = result[0]
            step = 360.0 / 36
            assert lo == 0.0
            assert hi == pytest.approx(360.0 - step)


# ---------------------------------------------------------------------------
# Single boresight-offset constraint
# ---------------------------------------------------------------------------


class TestRollRangeSingleBoresight:
    def test_roll_range_single_boresight_returns_list(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Single boresight constraint should return a list."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        assert isinstance(result, list)

    def test_roll_range_single_boresight_intervals_valid(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        panel_constraint: RustConstraintMixin,
    ) -> None:
        """Single boresight constraint intervals must have lo <= hi."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec

        result = panel_constraint.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec
        )

        for lo, hi in result:
            assert lo <= hi


# ---------------------------------------------------------------------------
# Compound constraint (OR / AND)
# ---------------------------------------------------------------------------


class TestRollRangeCompoundConstraints:
    @pytest.mark.parametrize(
        "constraint_name,constraint_factory",
        [
            (
                "OR",
                lambda: (
                    SunConstraint(min_angle=80.0, max_angle=100.0).boresight_offset(
                        yaw_deg=90.0, roll_clockwise=False
                    )
                    | SunConstraint(min_angle=0.0, max_angle=90.0).boresight_offset(
                        pitch_deg=-90.0, roll_clockwise=False
                    )
                ),
            ),
            (
                "AND",
                lambda: (
                    SunConstraint(min_angle=30.0, max_angle=150.0).boresight_offset(
                        yaw_deg=90.0
                    )
                    & SunConstraint(min_angle=10.0, max_angle=170.0).boresight_offset(
                        pitch_deg=90.0
                    )
                ),
            ),
        ],
    )
    def test_roll_range_compound_constraint_matches_per_roll_evaluate(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        reference_valid_rolls: Callable[
            [RustConstraintMixin, rust_ephem.TLEEphemeris, datetime, float, float, int],
            list[bool],
        ],
        roll_valid_from_intervals: Callable[[float, list[tuple[float, float]]], bool],
        constraint_name: str,
        constraint_factory: Callable[[], RustConstraintMixin],
    ) -> None:
        """Compound constraint roll_range must agree with evaluating each roll individually."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec
        n = 36
        compound = constraint_factory()

        intervals = compound.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=n
        )
        ref_valid = reference_valid_rolls(compound, ephem, time, ra, dec, n)

        # Check that roll_range intervals match the reference evaluation for all rolls
        step = 360.0 / n
        mismatches = []
        for i in range(n):
            roll = i * step
            expected = ref_valid[i]
            from_interval = roll_valid_from_intervals(roll, intervals)
            if from_interval != expected:
                mismatches.append(
                    f"Roll {roll}°: roll_range says {from_interval}, evaluate() says {expected}"
                )

        assert not mismatches, f"Found {len(mismatches)} mismatches: {mismatches}"


# ---------------------------------------------------------------------------
# Higher resolution
# ---------------------------------------------------------------------------


class TestRollRangeResolution:
    @pytest.mark.parametrize(
        "low_res,high_res",
        [
            (72, 360),  # 5° to 1° resolution
            (36, 180),  # 10° to 2° resolution
            (18, 90),  # 20° to 4° resolution
        ],
    )
    def test_roll_range_higher_resolution_consistent_with_lower(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        roll_valid_from_intervals: Callable[[float, list[tuple[float, float]]], bool],
        low_res: int,
        high_res: int,
    ) -> None:
        """A roll valid at lower resolution should also be valid at higher resolution."""
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec
        constraint = SunConstraint(min_angle=45.0, max_angle=135.0).boresight_offset(
            yaw_deg=90.0
        )

        intervals_low = constraint.roll_range(
            time,
            ephemeris=ephem,
            target_ra=ra,
            target_dec=dec,
            n_roll_samples=low_res,
        )
        intervals_high = constraint.roll_range(
            time,
            ephemeris=ephem,
            target_ra=ra,
            target_dec=dec,
            n_roll_samples=high_res,
        )

        # Check that all rolls valid at lower resolution are also valid at higher resolution
        violations = []
        for lo, hi in intervals_low:
            # Sample the interior of the coarse interval.
            mid = (lo + hi) / 2.0
            if not roll_valid_from_intervals(mid, intervals_high):
                violations.append(
                    f"Roll {mid}° valid in {low_res}-sample but not in {high_res}-sample intervals"
                )

        assert not violations, (
            f"Found {len(violations)} resolution consistency violations: {violations}"
        )


# ---------------------------------------------------------------------------
# Clockwise vs counter-clockwise symmetry
# ---------------------------------------------------------------------------


class TestRollRangeSymmetry:
    @pytest.mark.parametrize(
        "constraint_factory",
        [
            lambda: SunConstraint(min_angle=45.0, max_angle=135.0),
            lambda: SunConstraint(min_angle=30.0, max_angle=150.0),
            lambda: SunConstraint(min_angle=60.0, max_angle=120.0),
        ],
    )
    def test_roll_range_cw_ccw_symmetry(
        self,
        tle_ephem: rust_ephem.TLEEphemeris,
        sample_time: datetime,
        sample_target_ra: float,
        sample_target_dec: float,
        roll_valid_from_intervals: Callable[[float, list[tuple[float, float]]], bool],
        constraint_factory: Callable[[], SunConstraint],
    ) -> None:
        """CW and CCW roll_range results should be mirrors of each other.

        If roll=r is valid for the CCW convention, roll=(360-r) mod 360 should be
        valid for the CW convention (and vice-versa), because CW and CCW differ only
        in sign of the roll direction.
        """
        ephem = tle_ephem
        time = sample_time
        ra = sample_target_ra
        dec = sample_target_dec
        n = 36
        step = 360.0 / n

        base_constraint = constraint_factory()
        ccw = base_constraint.boresight_offset(yaw_deg=90.0, roll_clockwise=False)
        cw = base_constraint.boresight_offset(yaw_deg=90.0, roll_clockwise=True)

        intervals_ccw = ccw.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=n
        )
        intervals_cw = cw.roll_range(
            time, ephemeris=ephem, target_ra=ra, target_dec=dec, n_roll_samples=n
        )

        # Check that CW and CCW results are mirrors of each other
        step = 360.0 / n
        symmetry_violations = []
        for i in range(n):
            roll_ccw = i * step
            roll_cw_mirror = (360.0 - roll_ccw) % 360.0

            ccw_valid = roll_valid_from_intervals(roll_ccw, intervals_ccw)
            # Find the nearest sampled CW roll (mirror may not land exactly on a sample).
            nearest_cw_idx = round(roll_cw_mirror / step) % n
            roll_cw_sample = nearest_cw_idx * step
            cw_valid = roll_valid_from_intervals(roll_cw_sample, intervals_cw)

            if ccw_valid != cw_valid:
                symmetry_violations.append(
                    f"CCW roll {roll_ccw}° (valid={ccw_valid}) should mirror "
                    f"CW roll {roll_cw_sample}° (valid={cw_valid})"
                )

        assert not symmetry_violations, (
            f"Found {len(symmetry_violations)} symmetry violations: {symmetry_violations}"
        )
