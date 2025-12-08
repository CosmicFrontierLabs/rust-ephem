"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, TypeAdapter

from .ephemeris import Ephemeris

if TYPE_CHECKING:
    from rust_ephem import ConstraintResult


class RustConstraintMixin(BaseModel):
    """Base class for Rust constraint configurations"""

    def evaluate(
        self,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> ConstraintResult:
        """
        Evaluate the constraint using the Rust backend.

        This method lazily creates the corresponding Rust constraint
        object on first use.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate
            indices: Optional specific time index/indices to evaluate

        Returns:
            ConstraintResult containing violation windows
        """
        if not hasattr(self, "_rust_constraint"):
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint.evaluate(
            ephemeris,
            target_ra,
            target_dec,
            times,
            indices,
        )

    def in_constraint_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """
        Check if targets are in-constraint for multiple RA/Dec positions (vectorized).

        This method lazily creates the corresponding Rust constraint
        object on first use and evaluates it for multiple RA/Dec positions.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ras: List of target right ascensions in degrees (ICRS/J2000)
            target_decs: List of target declinations in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate
            indices: Optional specific time index/indices to evaluate

        Returns:
            2D numpy array of shape (n_targets, n_times) with boolean violation status
        """
        if not hasattr(self, "_rust_constraint"):
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint.in_constraint_batch(
            ephemeris,
            target_ras,
            target_decs,
            times,
            indices,
        )

    def in_constraint(
        self,
        time: datetime | list[datetime] | npt.NDArray[np.datetime64],
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
    ) -> bool | list[bool]:
        """
        Check if target violates the constraint at given time(s).

        This method lazily creates the corresponding Rust constraint
        object and delegates to its in_constraint method.

        Args:
            time: The time(s) to check (must exist in ephemeris). Can be a single datetime,
                  list of datetimes, or numpy array of datetimes.
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)

        Returns:
            True if constraint is violated at the given time(s). Returns a single bool
            for a single time, or a list of bools for multiple times.
        """
        if not hasattr(self, "_rust_constraint"):
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint.in_constraint(
            time,
            ephemeris,
            target_ra,
            target_dec,
        )

    def and_(self, other: ConstraintConfig) -> AndConstraint:
        """Combine this constraint with another using logical AND

        Args:
            other: Another constraint

        Returns:
            AndConstraint combining both constraints
        """
        return AndConstraint(constraints=[cast("ConstraintConfig", self), other])

    def or_(self, other: ConstraintConfig) -> OrConstraint:
        """Combine this constraint with another using logical OR

        Args:
            other: Another constraint

        Returns:
            OrConstraint combining both constraints
        """
        return OrConstraint(constraints=[cast("ConstraintConfig", self), other])

    def xor_(self, other: ConstraintConfig) -> XorConstraint:
        """Combine this constraint with another using logical XOR

        Args:
            other: Another constraint

        Returns:
            XorConstraint combining both constraints (violation when exactly one is violated)
        """
        return XorConstraint(constraints=[cast("ConstraintConfig", self), other])

    def not_(self) -> NotConstraint:
        """Negate this constraint using logical NOT

        Returns:
            NotConstraint negating this constraint
        """
        return NotConstraint(constraint=cast("ConstraintConfig", self))

    def __and__(self, other: ConstraintConfig) -> AndConstraint:
        """Combine constraints using & operator (logical AND)

        Args:
            other: Another constraint

        Returns:
            AndConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> combined = sun & moon
        """
        return self.and_(other)

    def __or__(self, other: ConstraintConfig) -> OrConstraint:
        """Combine constraints using | operator (logical OR)

        Args:
            other: Another constraint

        Returns:
            OrConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> combined = sun | moon
        """
        return self.or_(other)

    def __xor__(self, other: ConstraintConfig) -> XorConstraint:
        """Combine constraints using ^ operator (logical XOR)

        Args:
            other: Another constraint

        Returns:
            XorConstraint combining both constraints

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> moon = MoonConstraint(min_angle=30.0)
            >>> exclusive = sun ^ moon
        """
        return self.xor_(other)

    def __invert__(self) -> NotConstraint:
        """Negate constraint using ~ operator (logical NOT)

        Returns:
            NotConstraint negating this constraint

        Example:
            >>> sun = SunConstraint(min_angle=45.0)
            >>> not_sun = ~sun
        """
        return self.not_()


class SunConstraint(RustConstraintMixin):
    """Sun proximity constraint

    Ensures target maintains minimum angular separation from Sun.

    Attributes:
        type: Always "sun"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["sun"] = "sun"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Sun in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from Sun in degrees"
    )


class EarthLimbConstraint(RustConstraintMixin):
    """Earth limb avoidance constraint

    Ensures target maintains minimum angular separation from Earth's limb.
    For ground observers, optionally accounts for geometric horizon dip and atmospheric refraction.

    Attributes:
        type: Always "earth_limb"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
        include_refraction: Include atmospheric refraction correction (~0.57°) for ground observers (default: False)
        horizon_dip: Include geometric horizon dip correction for ground observers (default: False)
    """

    type: Literal["earth_limb"] = "earth_limb"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Earth's limb in degrees"
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from Earth's limb in degrees",
    )
    include_refraction: bool = Field(
        default=False,
        description="Include atmospheric refraction correction for ground observers",
    )
    horizon_dip: bool = Field(
        default=False,
        description="Include geometric horizon dip correction for ground observers",
    )


class BodyConstraint(RustConstraintMixin):
    """Solar system body proximity constraint

    Ensures target maintains minimum angular separation from specified body.

    Attributes:
        type: Always "body"
        body: Name of the solar system body (e.g., "Mars", "Jupiter")
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["body"] = "body"
    body: str = Field(..., description="Name of the solar system body")
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from body in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from body in degrees"
    )


class MoonConstraint(RustConstraintMixin):
    """Moon proximity constraint

    Ensures target maintains minimum angular separation from Moon.

    Attributes:
        type: Always "moon"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["moon"] = "moon"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from Moon in degrees"
    )
    max_angle: float | None = Field(
        default=None, ge=0.0, le=180.0, description="Maximum angle from Moon in degrees"
    )


class EclipseConstraint(RustConstraintMixin):
    """Eclipse constraint

    Checks if observer is in Earth's shadow (umbra and/or penumbra).

    Attributes:
        type: Always "eclipse"
        umbra_only: If True, only umbra counts. If False, includes penumbra.
    """

    type: Literal["eclipse"] = "eclipse"
    umbra_only: bool = Field(
        default=True, description="Count only umbra (True) or include penumbra (False)"
    )


class AndConstraint(RustConstraintMixin):
    """Logical AND constraint combinator

    Satisfied only if ALL sub-constraints are satisfied.

    Attributes:
        type: Always "and"
        constraints: List of constraints to combine with AND
    """

    type: Literal["and"] = "and"
    constraints: list[ConstraintConfig] = Field(
        ..., min_length=1, description="Constraints to AND together"
    )


class OrConstraint(RustConstraintMixin):
    """Logical OR constraint combinator

    Satisfied if ANY sub-constraint is satisfied.

    Attributes:
        type: Always "or"
        constraints: List of constraints to combine with OR
    """

    type: Literal["or"] = "or"
    constraints: list[ConstraintConfig] = Field(
        ..., min_length=1, description="Constraints to OR together"
    )


class XorConstraint(RustConstraintMixin):
    """Logical XOR constraint combinator

    Satisfied if EXACTLY ONE sub-constraint is satisfied.

    Attributes:
        type: Always "xor"
        constraints: List of constraints to combine with XOR (minimum 2)
    """

    type: Literal["xor"] = "xor"
    constraints: list[ConstraintConfig] = Field(
        ...,
        min_length=2,
        description="Constraints to XOR together (exactly one satisfied)",
    )


class NotConstraint(RustConstraintMixin):
    """Logical NOT constraint combinator

    Inverts a constraint - satisfied when inner constraint is violated.

    Attributes:
        type: Always "not"
        constraint: Constraint to negate
    """

    type: Literal["not"] = "not"
    constraint: ConstraintConfig = Field(..., description="Constraint to negate")


class DaytimeConstraint(RustConstraintMixin):
    """Daytime visibility constraint

    Prevents observations during daytime hours,
    with configurable twilight definitions.

    Attributes:
        type: Always "daytime"
        twilight: Twilight definition ("civil", "nautical", "astronomical", "none")
    """

    type: Literal["daytime"] = "daytime"
    twilight: Literal["civil", "nautical", "astronomical", "none"] = Field(
        default="civil", description="Twilight definition for daytime boundary"
    )


class AirmassConstraint(RustConstraintMixin):
    """Airmass constraint

    Limits observations based on atmospheric airmass (secant of zenith angle).
    Lower airmass values indicate better observing conditions.

    Attributes:
        type: Always "airmass"
        min_airmass: Minimum allowed airmass (≥1.0), optional
        max_airmass: Maximum allowed airmass (>0.0)
    """

    type: Literal["airmass"] = "airmass"
    min_airmass: float | None = Field(
        default=None, ge=1.0, description="Minimum allowed airmass"
    )
    max_airmass: float = Field(..., gt=0.0, description="Maximum allowed airmass")


class MoonPhaseConstraint(RustConstraintMixin):
    """Moon phase constraint

    Limits observations based on Moon illumination fraction and distance.

    Attributes:
        type: Always "moon_phase"
        min_illumination: Minimum allowed illumination fraction (0.0-1.0), optional
        max_illumination: Maximum allowed illumination fraction (0.0-1.0)
        min_distance: Minimum allowed Moon distance in degrees from target, optional
        max_distance: Maximum allowed Moon distance in degrees from target, optional
        enforce_when_below_horizon: Whether to enforce constraint when Moon is below horizon
        moon_visibility: Moon visibility requirement ("full" or "partial")
    """

    type: Literal["moon_phase"] = "moon_phase"
    min_illumination: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum allowed illumination fraction",
    )
    max_illumination: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum allowed illumination fraction"
    )
    min_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum allowed Moon distance in degrees from target",
    )
    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed Moon distance in degrees from target",
    )
    enforce_when_below_horizon: bool = Field(
        default=False,
        description="Whether to enforce constraint when Moon is below horizon",
    )
    moon_visibility: Literal["full", "partial"] = Field(
        default="full",
        description="Moon visibility requirement: 'full' (only when fully above horizon) or 'partial' (when any part visible)",
    )


class SAAConstraint(RustConstraintMixin):
    """South Atlantic Anomaly constraint

    Limits observations based on whether the spacecraft is within a defined
    geographic region (typically the South Atlantic Anomaly).

    Attributes:
        type: Always "saa"
        polygon: List of (longitude, latitude) pairs defining the region boundary
    """

    type: Literal["saa"] = "saa"
    polygon: list[tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="List of (longitude, latitude) pairs defining the region boundary in degrees",
    )


class AltAzConstraint(RustConstraintMixin):
    """Altitude/Azimuth constraint

    Limits observations based on target's altitude and azimuth angles
    from the observer's location.

    Attributes:
        type: Always "alt_az"
        min_altitude: Minimum allowed altitude in degrees (0-90)
        max_altitude: Maximum allowed altitude in degrees (0-90), optional
        min_azimuth: Minimum allowed azimuth in degrees (0-360), optional
        max_azimuth: Maximum allowed azimuth in degrees (0-360), optional
    """

    type: Literal["alt_az"] = "alt_az"
    min_altitude: float = Field(
        ..., ge=0.0, le=90.0, description="Minimum allowed altitude in degrees"
    )
    max_altitude: float | None = Field(
        default=None, ge=0.0, le=90.0, description="Maximum allowed altitude in degrees"
    )
    min_azimuth: float | None = Field(
        default=None, ge=0.0, le=360.0, description="Minimum allowed azimuth in degrees"
    )
    max_azimuth: float | None = Field(
        default=None, ge=0.0, le=360.0, description="Maximum allowed azimuth in degrees"
    )


# Union type for all constraints
ConstraintConfig = Union[
    SunConstraint,
    MoonConstraint,
    EclipseConstraint,
    EarthLimbConstraint,
    BodyConstraint,
    DaytimeConstraint,
    AirmassConstraint,
    MoonPhaseConstraint,
    SAAConstraint,
    AltAzConstraint,
    AndConstraint,
    OrConstraint,
    XorConstraint,
    NotConstraint,
]


# Update forward references after ConstraintConfig is defined
AndConstraint.model_rebuild()
OrConstraint.model_rebuild()
XorConstraint.model_rebuild()
NotConstraint.model_rebuild()


# Type adapter for ConstraintConfig union
CombinedConstraintConfig: TypeAdapter[ConstraintConfig] = TypeAdapter(ConstraintConfig)
