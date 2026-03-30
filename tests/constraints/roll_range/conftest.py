"""Fixtures for roll_range tests."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pytest

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import RustConstraintMixin, SunConstraint


@pytest.fixture
def roll_valid_from_intervals() -> Callable[[float, list[tuple[float, float]]], bool]:
    """Helper function to check if a roll angle falls within any interval."""

    def _roll_valid_from_intervals(
        roll_deg: float, intervals: list[tuple[float, float]]
    ) -> bool:
        """Return True if roll_deg falls inside any returned interval (inclusive)."""
        return any(lo <= roll_deg <= hi for lo, hi in intervals)

    return _roll_valid_from_intervals


@pytest.fixture
def reference_valid_rolls() -> Callable[
    [RustConstraintMixin, rust_ephem.TLEEphemeris, datetime, float, float, int],
    list[bool],
]:
    """Helper function to evaluate constraint validity for each roll sample."""

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

    return _reference_valid_rolls


@pytest.fixture
def sample_time(tle_ephem: rust_ephem.TLEEphemeris) -> datetime:
    """A sample timestamp from the middle of the ephemeris."""

    return tle_ephem.timestamp[len(tle_ephem.timestamp) // 2]  # type: ignore


@pytest.fixture
def sample_target_ra() -> float:
    """Sample target right ascension for testing."""
    return 30.0


@pytest.fixture
def sample_target_dec() -> float:
    """Sample target declination for testing."""
    return 10.0


@pytest.fixture
def basic_sun_constraint() -> SunConstraint:
    """Basic SunConstraint for testing."""
    return SunConstraint(min_angle=45.0)


@pytest.fixture
def panel_constraint(basic_sun_constraint: SunConstraint) -> RustConstraintMixin:
    """Solar panel constraint with 90° yaw offset."""
    return basic_sun_constraint.boresight_offset(yaw_deg=90.0)


@pytest.fixture
def compound_or_constraint() -> RustConstraintMixin:
    """Compound OR constraint combining gimbal plane and sunlit constraints."""
    gimbal_plane = SunConstraint(min_angle=80.0, max_angle=100.0).boresight_offset(
        yaw_deg=90.0, roll_clockwise=False
    )
    sunlit = SunConstraint(min_angle=0.0, max_angle=90.0).boresight_offset(
        pitch_deg=-90.0, roll_clockwise=False
    )
    return gimbal_plane | sunlit


@pytest.fixture
def compound_and_constraint() -> RustConstraintMixin:
    """Compound AND constraint combining two different angle ranges."""
    a = SunConstraint(min_angle=30.0, max_angle=150.0).boresight_offset(yaw_deg=90.0)
    b = SunConstraint(min_angle=10.0, max_angle=170.0).boresight_offset(pitch_deg=90.0)
    return a & b


@pytest.fixture
def always_violated_constraint() -> RustConstraintMixin:
    """Constraint that should almost always be violated."""
    narrow_sun = SunConstraint(min_angle=89.9, max_angle=90.1).boresight_offset(
        yaw_deg=90.0, roll_clockwise=False
    )
    tight = narrow_sun & SunConstraint(min_angle=89.9, max_angle=90.1).boresight_offset(
        pitch_deg=90.0, roll_clockwise=False
    )
    return tight


@pytest.fixture
def roll_invariant_constraint() -> rust_ephem.Constraint:
    """Constraint that doesn't depend on spacecraft roll."""
    return Constraint.sun_proximity(min_angle=1.0)
