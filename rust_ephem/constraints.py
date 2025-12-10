"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

import rust_ephem

from .ephemeris import Ephemeris

if TYPE_CHECKING:
    import rust_ephem

if TYPE_CHECKING:
    pass


class ConstraintViolation(BaseModel):
    """A time window where a constraint was violated."""

    start_time: datetime = Field(..., description="Start time of violation window")
    end_time: datetime = Field(..., description="End time of violation window")
    max_severity: float = Field(
        ..., description="Maximum severity of violation in this window"
    )
    description: str = Field(
        ..., description="Human-readable description of the violation"
    )


class ConstraintResult(BaseModel):
    """Result of constraint evaluation containing all violations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    violations: list[ConstraintViolation] = Field(
        default_factory=list, description="List of violation windows"
    )
    all_satisfied: bool = Field(
        ..., description="Whether constraint was satisfied for entire time range"
    )
    constraint_name: str = Field(..., description="Name/description of the constraint")

    # Store reference to Rust result for lazy access to timestamps/constraint_array
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._rust_result_ref = data.get("_rust_result_ref", None)

    @property
    def timestamps(self) -> npt.NDArray[np.datetime64] | list[datetime]:
        """Evaluation timestamps (lazily accessed from Rust result)."""
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(
                npt.NDArray[np.datetime64] | list[datetime],
                self._rust_result_ref.timestamp,
            )
        return []

    @property
    def constraint_array(self) -> list[bool]:
        """
        Boolean array indicating constraint violations (lazily accessed from Rust result).

        Returns
        -------
        numpy.ndarray or list of bool
            Boolean array where True indicates the constraint is violated at that time,
            and False indicates the constraint is satisfied.
        """
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(list[bool], self._rust_result_ref.constraint_array)
        return []

    @property
    def visibility(self) -> list["rust_ephem.VisibilityWindow"]:
        """Visibility windows when the constraint is satisfied (target visible)."""
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(
                list["rust_ephem.VisibilityWindow"], self._rust_result_ref.visibility
            )
        return []

    def total_violation_duration(self) -> float:
        """Get the total duration of violations in seconds."""
        total_seconds = 0.0
        for violation in self.violations:
            total_seconds += (violation.end_time - violation.start_time).total_seconds()
        return total_seconds

    def in_constraint(self, time: datetime) -> bool:
        """Check if target is in-constraint at a given time.

        This method operates on timestamps from the evaluate() call.
        The given time must exist in the evaluated timestamps.

        Args:
            time: The datetime to check (must be in evaluated timestamps)

        Returns:
            True if the constraint is violated at this time (target is in-constraint),
            False if the constraint is satisfied (target is out-of-constraint).

        Raises:
            ValueError: If the time is not found in evaluated timestamps
        """
        if hasattr(self, "_rust_result_ref") and self._rust_result_ref is not None:
            return cast(bool, self._rust_result_ref.in_constraint(time))
        raise ValueError(
            "ConstraintResult has no evaluated timestamps (was not created from evaluate())"
        )

    def __repr__(self) -> str:
        return f"ConstraintResult(constraint='{self.constraint_name}', violations={len(self.violations)}, all_satisfied={self.all_satisfied})"


if TYPE_CHECKING:
    pass


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

        # Get the Rust result
        rust_result = self._rust_constraint.evaluate(
            ephemeris,
            target_ra,
            target_dec,
            times,
            indices,
        )

        # Convert to Pydantic model, parsing ISO 8601 timestamps to datetime objects
        return ConstraintResult(
            violations=[
                ConstraintViolation(
                    start_time=datetime.fromisoformat(
                        v.start_time.replace("Z", "+00:00")
                    ),
                    end_time=datetime.fromisoformat(v.end_time.replace("Z", "+00:00")),
                    max_severity=v.max_severity,
                    description=v.description,
                )
                for v in rust_result.violations
            ],
            all_satisfied=rust_result.all_satisfied,
            constraint_name=rust_result.constraint_name,
            _rust_result_ref=rust_result,
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
        """Check if target is in-constraint at given time(s).

        This method performs full constraint evaluation for the given times.
        Use this to check constraint status without pre-computing evaluate().

        **API Note:** This differs from ConstraintResult.in_constraint() which
        operates on pre-evaluated timestamps. Use this method when you need
        to check arbitrary times, and use ConstraintResult.in_constraint()
        only for times already evaluated via evaluate().

        Args:
            time: The time(s) to check (must exist in ephemeris). Can be a single datetime,
                  list of datetimes, or numpy array of datetimes.
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)

        Returns:
            True if constraint is violated at the given time(s) (in-constraint).
            False if constraint is satisfied (out-of-constraint).
            Returns a single bool for a single time, or a list of bools for multiple times.
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
    max_airmass: float = Field(..., ge=1.0, description="Maximum allowed airmass")

    @model_validator(mode="after")
    def validate_airmass_values(self) -> AirmassConstraint:
        if self.min_airmass is not None and self.max_airmass < self.min_airmass:
            raise ValueError("max_airmass must be >= min_airmass")
        return self


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

    @model_validator(mode="after")
    def validate_moon_phase_values(self) -> MoonPhaseConstraint:
        if (
            self.min_illumination is not None
            and self.max_illumination < self.min_illumination
        ):
            raise ValueError("max_illumination must be >= min_illumination")
        if (
            self.min_distance is not None
            and self.max_distance is not None
            and self.max_distance < self.min_distance
        ):
            raise ValueError("max_distance must be >= min_distance")
        return self


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
    from the observer's location. Can use simple min/max ranges or a
    custom polygon defining an allowed region.

    Attributes:
        type: Always "alt_az"
        min_altitude: Minimum allowed altitude in degrees (0-90), optional
        max_altitude: Maximum allowed altitude in degrees (0-90), optional
        min_azimuth: Minimum allowed azimuth in degrees (0-360), optional
        max_azimuth: Maximum allowed azimuth in degrees (0-360), optional
        polygon: List of (altitude, azimuth) pairs defining allowed region, optional
    """

    type: Literal["alt_az"] = "alt_az"
    min_altitude: float | None = Field(
        default=None, ge=0.0, le=90.0, description="Minimum allowed altitude in degrees"
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
    polygon: list[tuple[float, float]] | None = Field(
        default=None,
        description="List of (altitude, azimuth) pairs in degrees defining allowed region",
    )


class OrbitRamConstraint(RustConstraintMixin):
    """Orbit RAM direction constraint

    Ensures target maintains minimum angular separation from the spacecraft's
    velocity vector (RAM direction). Useful for avoiding pointing
    directions that may cause contamination.

    Attributes:
        type: Always "orbit_ram"
        min_angle: Minimum allowed angular separation from RAM direction in degrees (0-180)
        max_angle: Maximum allowed angular separation from RAM direction in degrees (0-180), optional
    """

    type: Literal["orbit_ram"] = "orbit_ram"
    min_angle: float = Field(
        ..., ge=0.0, le=180.0, description="Minimum angle from RAM direction in degrees"
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from RAM direction in degrees",
    )


class OrbitPoleConstraint(RustConstraintMixin):
    """Orbit pole direction constraint

    Ensures target maintains minimum angular separation from both the north and south
    orbital poles (directions perpendicular to the orbital plane). Useful for maintaining
    specific orientations relative to the spacecraft's orbit.

    Attributes:
        type: Always "orbit_pole"
        min_angle: Minimum allowed angular separation from both orbital poles in degrees (0-180)
        max_angle: Maximum allowed angular separation from both orbital poles in degrees (0-180), optional
        earth_limb_pole: If True, pole avoidance angle is earth_radius_deg + min_angle - 90.
                        Used for NASA's Neil Gehrels Swift Observatory where the pole is an emergent
                        property of Earth size plus Earth limb avoidance angle > 90°.
    """

    type: Literal["orbit_pole"] = "orbit_pole"
    min_angle: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description="Minimum angle from both orbital poles in degrees",
    )
    max_angle: float | None = Field(
        default=None,
        ge=0.0,
        le=180.0,
        description="Maximum angle from both orbital poles in degrees",
    )
    earth_limb_pole: bool = Field(
        default=False,
        description="If True, pole avoidance angle is earth_radius_deg + min_angle - 90",
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
    OrbitRamConstraint,
    OrbitPoleConstraint,
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


def _to_datetime_list(times: Any) -> list[datetime]:
    """Convert supported timestamp inputs to a list of Python datetimes."""
    if times is None:
        return []
    if isinstance(times, datetime):
        return [times]
    if isinstance(times, np.ndarray):
        return [_convert_single_datetime(t) for t in times.tolist()]
    return [_convert_single_datetime(t) for t in times]


def _convert_single_datetime(t: Any) -> datetime:
    if isinstance(t, datetime):
        return t
    if isinstance(t, np.datetime64):
        return datetime.fromisoformat(np.datetime_as_string(t))
    raise TypeError(f"Unsupported timestamp type: {type(t)}")


class VisibilityWindowResult(BaseModel):
    """Visibility window for a moving target."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float


class MovingVisibilityResult(BaseModel):
    """Result for moving target visibility evaluation."""

    timestamps: list[datetime]
    ras: list[float]  # Right ascension in degrees for each timestamp
    decs: list[float]  # Declination in degrees for each timestamp
    constraint_array: list[bool]
    visibility_flags: list[bool]
    visibility: list[VisibilityWindowResult]
    all_satisfied: bool
    constraint_name: str


def moving_body_visibility(
    constraint: ConstraintConfig,
    ephemeris: Ephemeris,
    ras: npt.ArrayLike | None = None,
    decs: npt.ArrayLike | None = None,
    timestamps: npt.ArrayLike | None = None,
    body: str | int | None = None,
    use_horizons: bool = False,
) -> MovingVisibilityResult:
    """Evaluate constraint visibility for a moving target.

    Supports either explicit RA/Dec arrays aligned with timestamps or automatic
    ephemeris-based positions via NAIF ID/body name using the default planetary
    ephemeris (de440s) available through the provided ephemeris instance.

    Args:
        constraint: Constraint configuration to evaluate.
        ephemeris: Ephemeris providing observer timeline and geometry.
        ras: Array-like of right ascension values in degrees (matches timestamps).
        decs: Array-like of declination values in degrees (matches timestamps).
        timestamps: Array-like of datetimes aligned with ras/decs. If None and
            `body` is provided, uses the ephemeris timestamps.
        body: NAIF ID (int/str) or body name to resolve using ephemeris.get_body().
        use_horizons: If True, query JPL Horizons when SPICE kernels don't have the body.

    Returns:
        MovingVisibilityResult with per-timestamp violation flags, visibility flags,
        and merged visibility windows.
    """

    # Resolve positions from body identifier if provided
    if body is not None:
        body_id = str(body)
        skycoord = ephemeris.get_body(body_id, use_horizons=use_horizons)
        ras_array = np.asarray(skycoord.ra.deg)
        decs_array = np.asarray(skycoord.dec.deg)
        ts_list = _to_datetime_list(ephemeris.timestamp)
    else:
        if ras is None or decs is None or timestamps is None:
            raise ValueError(
                "ras, decs, and timestamps must be provided when body is None"
            )
        ras_array = np.asarray(ras, dtype=float)
        decs_array = np.asarray(decs, dtype=float)
        ts_list = _to_datetime_list(timestamps)

    if ras_array.shape != decs_array.shape:
        raise ValueError("ras and decs must have the same shape")
    if len(ts_list) != ras_array.shape[0]:
        raise ValueError("timestamps length must match ras/decs length")

    # Compute violation flags per timestamp using single-time checks to align RA/Dec per time
    violations: list[bool] = []
    for ra, dec, t in zip(ras_array.tolist(), decs_array.tolist(), ts_list):
        violations.append(
            bool(
                constraint.in_constraint(
                    time=t, ephemeris=ephemeris, target_ra=ra, target_dec=dec
                )
            )
        )

    visibility_flags = [not v for v in violations]
    all_satisfied = not any(violations)

    windows = _build_visibility_windows(ts_list, visibility_flags)

    return MovingVisibilityResult(
        timestamps=ts_list,
        ras=ras_array.tolist(),
        decs=decs_array.tolist(),
        constraint_array=violations,
        visibility_flags=visibility_flags,
        visibility=windows,
        all_satisfied=all_satisfied,
        constraint_name=getattr(constraint, "__class__", type(constraint)).__name__,
    )


def _build_visibility_windows(
    timestamps: list[datetime], visibility: list[bool]
) -> list[VisibilityWindowResult]:
    """Merge consecutive visibility samples into windows."""
    windows: list[VisibilityWindowResult] = []
    if not timestamps or not visibility:
        return windows

    current_start: datetime | None = None
    for idx, is_visible in enumerate(visibility):
        if is_visible and current_start is None:
            current_start = timestamps[idx]
        if (not is_visible or idx == len(visibility) - 1) and current_start is not None:
            end_idx = idx if not is_visible else idx
            end_time = timestamps[end_idx]
            duration = (end_time - current_start).total_seconds()
            windows.append(
                VisibilityWindowResult(
                    start_time=current_start,
                    end_time=end_time,
                    duration_seconds=duration,
                )
            )
            current_start = None
    return windows
