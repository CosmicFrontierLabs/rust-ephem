"""Tests for SolarRollConstraint, including the panel_normal parameter.

The constraint is satisfied when the spacecraft roll is within tolerance_deg of
the solar-optimal roll — the roll that maximises illumination of the panel whose
body-frame normal is given by panel_normal.

Key relationship tested:  rotating panel_normal by α degrees in the body Y-Z
plane shifts the optimal roll by -α degrees.  The default panel_normal (0,1,0)
recovers the pre-existing +Y-panel behaviour.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import cast

import pytest
from pydantic import ValidationError

import rust_ephem
from rust_ephem.constraints import SolarRollConstraint

# ---------------------------------------------------------------------------
# Model creation and validation
# ---------------------------------------------------------------------------


class TestSolarRollConstraintModel:
    def test_creation_basic(self) -> None:
        c = SolarRollConstraint(tolerance_deg=10.0)
        assert c.tolerance_deg == 10.0

    def test_type_field(self) -> None:
        c = SolarRollConstraint(tolerance_deg=5.0)
        assert c.type == "solar_roll"

    def test_panel_normal_default(self) -> None:
        c = SolarRollConstraint(tolerance_deg=5.0)
        assert c.panel_normal == (0.0, 1.0, 0.0)

    def test_panel_normal_explicit(self) -> None:
        c = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, 0.0, 1.0))
        assert c.panel_normal == (0.0, 0.0, 1.0)

    def test_panel_normal_arbitrary_angle(self) -> None:
        alpha = math.radians(30.0)
        n = (0.0, math.cos(alpha), math.sin(alpha))
        c = SolarRollConstraint(tolerance_deg=5.0, panel_normal=n)
        assert c.panel_normal[0] == pytest.approx(0.0)
        assert c.panel_normal[1] == pytest.approx(math.cos(alpha))
        assert c.panel_normal[2] == pytest.approx(math.sin(alpha))

    def test_tolerance_deg_zero_valid(self) -> None:
        SolarRollConstraint(tolerance_deg=0.0)

    def test_tolerance_deg_180_valid(self) -> None:
        SolarRollConstraint(tolerance_deg=180.0)

    def test_tolerance_deg_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            SolarRollConstraint(tolerance_deg=-1.0)

    def test_tolerance_deg_above_180_raises(self) -> None:
        with pytest.raises(ValidationError):
            SolarRollConstraint(tolerance_deg=181.0)

    def test_json_round_trip_default_panel_normal(self) -> None:
        c = SolarRollConstraint(tolerance_deg=15.0)
        json_str = c.model_dump_json()
        c2 = SolarRollConstraint.model_validate_json(json_str)
        assert c2.tolerance_deg == c.tolerance_deg
        assert c2.panel_normal == c.panel_normal

    def test_json_round_trip_custom_panel_normal(self) -> None:
        alpha = math.radians(45.0)
        n = (0.0, math.cos(alpha), math.sin(alpha))
        c = SolarRollConstraint(tolerance_deg=10.0, panel_normal=n)
        json_str = c.model_dump_json()
        c2 = SolarRollConstraint.model_validate_json(json_str)
        assert c2.panel_normal[1] == pytest.approx(math.cos(alpha))
        assert c2.panel_normal[2] == pytest.approx(math.sin(alpha))


# ---------------------------------------------------------------------------
# Evaluation without a roll (target_roll=None → always satisfied)
# ---------------------------------------------------------------------------


class TestSolarRollConstraintNoRoll:
    def test_no_roll_always_satisfied(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """Without a target_roll the constraint reports all_satisfied."""
        c = SolarRollConstraint(tolerance_deg=10.0)
        result = c.evaluate(tle_ephemeris, target_ra=30.0, target_dec=10.0)
        assert result.all_satisfied

    def test_no_roll_always_satisfied_custom_panel_normal(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """panel_normal has no effect when roll is not evaluated."""
        c = SolarRollConstraint(tolerance_deg=10.0, panel_normal=(0.0, 0.0, 1.0))
        result = c.evaluate(tle_ephemeris, target_ra=30.0, target_dec=10.0)
        assert result.all_satisfied


# ---------------------------------------------------------------------------
# Wide tolerance — always satisfied for any roll
# ---------------------------------------------------------------------------


class TestSolarRollConstraintWideTolerance:
    def test_wide_tolerance_satisfied_at_zero_roll(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """tolerance_deg=180 covers the full circle; any roll is within tolerance."""
        c = SolarRollConstraint(tolerance_deg=180.0)
        result = c.evaluate(
            tle_ephemeris, target_ra=30.0, target_dec=10.0, target_roll=0.0
        )
        assert result.all_satisfied

    def test_wide_tolerance_satisfied_at_arbitrary_roll(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        for roll in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.0]:
            c = SolarRollConstraint(tolerance_deg=180.0)
            result = c.evaluate(
                tle_ephemeris, target_ra=30.0, target_dec=10.0, target_roll=roll
            )
            assert result.all_satisfied, f"Wide tolerance violated at roll={roll}°"

    def test_wide_tolerance_custom_panel_normal_satisfied(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        c = SolarRollConstraint(tolerance_deg=180.0, panel_normal=(0.0, 0.0, 1.0))
        result = c.evaluate(
            tle_ephemeris, target_ra=30.0, target_dec=10.0, target_roll=90.0
        )
        assert result.all_satisfied


# ---------------------------------------------------------------------------
# Default panel_normal matches explicit (0, 1, 0)
# ---------------------------------------------------------------------------


class TestSolarRollConstraintDefaultMatchesExplicit:
    def test_default_panel_normal_matches_explicit_y(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """panel_normal=(0,1,0) must produce identical results to the default."""
        c_default = SolarRollConstraint(tolerance_deg=20.0)
        c_explicit = SolarRollConstraint(
            tolerance_deg=20.0, panel_normal=(0.0, 1.0, 0.0)
        )

        for roll in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
            r_default = c_default.evaluate(
                tle_ephemeris, target_ra=30.0, target_dec=10.0, target_roll=roll
            )
            r_explicit = c_explicit.evaluate(
                tle_ephemeris, target_ra=30.0, target_dec=10.0, target_roll=roll
            )
            assert r_default.all_satisfied == r_explicit.all_satisfied, (
                f"Mismatch at roll={roll}°: default={r_default.all_satisfied}, "
                f"explicit={r_explicit.all_satisfied}"
            )


# ---------------------------------------------------------------------------
# Panel normal shift observable in roll_range
# ---------------------------------------------------------------------------


class TestSolarRollConstraintPanelNormalShift:
    """Rotating panel_normal by 90° in the body Y-Z plane should shift the
    optimal roll window by 90°.
    """

    @pytest.fixture
    def sample_time(self, tle_ephemeris: rust_ephem.TLEEphemeris) -> datetime:
        times = tle_ephemeris.timestamp
        return cast(datetime, times[len(times) // 2])

    def _window_center(self, intervals: list[tuple[float, float]]) -> float | None:
        """Return the midpoint of the widest valid interval, or None if empty."""
        if not intervals:
            return None
        lo, hi = max(intervals, key=lambda iv: iv[1] - iv[0])
        return (lo + hi) / 2.0

    def test_panel_normal_90deg_shifts_optimal_roll(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        sample_time: datetime,
    ) -> None:
        """Rotating panel from +Y to +Z shifts the optimal roll by ~90°."""
        c_y = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, 1.0, 0.0))
        c_z = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, 0.0, 1.0))

        n = 360
        intervals_y = c_y.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )
        intervals_z = c_z.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )

        center_y = self._window_center(intervals_y)
        center_z = self._window_center(intervals_z)

        assert center_y is not None, (
            "panel_normal=(0,1,0) produced no valid roll window"
        )
        assert center_z is not None, (
            "panel_normal=(0,0,1) produced no valid roll window"
        )

        # The +Z panel window should be ~90° away from the +Y panel window.
        diff = abs(((center_z - center_y + 180.0) % 360.0) - 180.0)
        assert diff == pytest.approx(90.0, abs=5.0), (
            f"Expected ~90° shift between +Y and +Z panels, "
            f"got {diff:.1f}° (center_y={center_y:.1f}°, center_z={center_z:.1f}°)"
        )

    def test_panel_normal_180deg_shifts_optimal_roll(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        sample_time: datetime,
    ) -> None:
        """Panel normal -Y (180° from +Y) shifts the optimal roll by ~180°."""
        c_y = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, 1.0, 0.0))
        c_neg_y = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, -1.0, 0.0))

        n = 360
        intervals_y = c_y.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )
        intervals_neg_y = c_neg_y.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )

        center_y = self._window_center(intervals_y)
        center_neg_y = self._window_center(intervals_neg_y)

        assert center_y is not None
        assert center_neg_y is not None

        diff = abs(((center_neg_y - center_y + 180.0) % 360.0) - 180.0)
        assert diff == pytest.approx(180.0, abs=5.0), (
            f"Expected ~180° shift between +Y and -Y panels, got {diff:.1f}°"
        )

    def test_panel_normal_arbitrary_angle_shift(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        sample_time: datetime,
    ) -> None:
        """A 30° panel rotation should shift the optimal roll by ~30°."""
        alpha = 30.0
        c_y = SolarRollConstraint(tolerance_deg=5.0, panel_normal=(0.0, 1.0, 0.0))
        c_tilted = SolarRollConstraint(
            tolerance_deg=5.0,
            panel_normal=(
                0.0,
                math.cos(math.radians(alpha)),
                math.sin(math.radians(alpha)),
            ),
        )

        n = 360
        intervals_y = c_y.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )
        intervals_tilted = c_tilted.roll_range(
            sample_time,
            ephemeris=tle_ephemeris,
            target_ra=30.0,
            target_dec=10.0,
            n_roll_samples=n,
        )

        center_y = self._window_center(intervals_y)
        center_tilted = self._window_center(intervals_tilted)

        assert center_y is not None
        assert center_tilted is not None

        # Tilted panel needs alpha° less roll than +Y panel.
        diff = abs(((center_y - center_tilted + 180.0) % 360.0) - 180.0)
        assert diff == pytest.approx(alpha, abs=5.0), (
            f"Expected ~{alpha}° shift for {alpha}° panel tilt, got {diff:.1f}°"
        )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestSolarRollConstraintBatch:
    def test_in_constraint_batch_shape(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """in_constraint_batch must return (n_targets, n_times) shape."""
        c = SolarRollConstraint(tolerance_deg=10.0)
        n_targets = 3
        ras = [0.0, 90.0, 180.0]
        decs = [0.0, 30.0, -30.0]
        result = c.in_constraint_batch(
            tle_ephemeris, ras, decs, target_rolls=[0.0] * n_targets
        )
        assert result.shape == (n_targets, len(tle_ephemeris.timestamp))

    def test_in_constraint_batch_wide_tolerance_no_violations(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        """tolerance_deg=180 → no time step is violated for any target."""
        import numpy as np

        c = SolarRollConstraint(tolerance_deg=180.0)
        n_targets = 4
        ras = [0.0, 90.0, 180.0, 270.0]
        decs = [0.0, 45.0, -45.0, 0.0]
        result = c.in_constraint_batch(
            tle_ephemeris, ras, decs, target_rolls=[45.0] * n_targets
        )
        # in_constraint_batch returns True where VIOLATED; wide tolerance → never violated
        assert not np.any(result), "Wide tolerance should never produce a violation"

    def test_in_constraint_batch_custom_panel_normal_shape(
        self, tle_ephemeris: rust_ephem.TLEEphemeris
    ) -> None:
        c = SolarRollConstraint(tolerance_deg=10.0, panel_normal=(0.0, 0.0, 1.0))
        n_targets = 2
        ras = [15.0, 45.0]
        decs = [5.0, -10.0]
        result = c.in_constraint_batch(
            tle_ephemeris, ras, decs, target_rolls=[0.0] * n_targets
        )
        assert result.shape == (n_targets, len(tle_ephemeris.timestamp))
