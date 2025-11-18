"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Literal, Union, cast

from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    from rust_ephem import (
        ConstraintResult,
        GroundEphemeris,
        OEMEphemeris,
        SPICEEphemeris,
        TLEEphemeris,
    )


class RustConstraintMixin(BaseModel):
    """Base class for Rust constraint configurations"""

    def evaluate(
        self,
        ephemeris: Union[
            "TLEEphemeris", "SPICEEphemeris", "GroundEphemeris", "OEMEphemeris"
        ],
        target_ra: float,
        target_dec: float,
        times: Union[datetime, list[datetime], None] = None,
        indices: Union[int, list[int], None] = None,
    ) -> "ConstraintResult":
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
            ephemeris, target_ra, target_dec, times, indices
        )

    def in_constraint(
        self,
        time: datetime,
        ephemeris: Union[
            "TLEEphemeris", "SPICEEphemeris", "GroundEphemeris", "OEMEphemeris"
        ],
        target_ra: float,
        target_dec: float,
    ) -> bool:
        """
        Check if target violates the constraint at a given time.

        This method lazily creates the corresponding Rust constraint
        object and delegates to its in_constraint method.

        Args:
            time: The time to check (must exist in ephemeris)
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)

        Returns:
            True if constraint is violated at the given time
        """
        if not hasattr(self, "_rust_constraint"):
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint.in_constraint(
            time, ephemeris, target_ra, target_dec
        )

    def and_(self, other: "ConstraintConfig") -> "AndConstraint":
        """Combine this constraint with another using logical AND

        Args:
            other: Another constraint

        Returns:
            AndConstraint combining both constraints
        """
        return AndConstraint(constraints=[cast("ConstraintConfig", self), other])

    def or_(self, other: "ConstraintConfig") -> "OrConstraint":
        """Combine this constraint with another using logical OR

        Args:
            other: Another constraint

        Returns:
            OrConstraint combining both constraints
        """
        return OrConstraint(constraints=[cast("ConstraintConfig", self), other])

    def xor_(self, other: "ConstraintConfig") -> "XorConstraint":
        """Combine this constraint with another using logical XOR

        Args:
            other: Another constraint

        Returns:
            XorConstraint combining both constraints (violation when exactly one is violated)
        """
        return XorConstraint(constraints=[cast("ConstraintConfig", self), other])

    def not_(self) -> "NotConstraint":
        """Negate this constraint using logical NOT

        Returns:
            NotConstraint negating this constraint
        """
        return NotConstraint(constraint=cast("ConstraintConfig", self))

    def __and__(self, other: "ConstraintConfig") -> "AndConstraint":
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

    def __or__(self, other: "ConstraintConfig") -> "OrConstraint":
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

    def __xor__(self, other: "ConstraintConfig") -> "XorConstraint":
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

    def __invert__(self) -> "NotConstraint":
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
    constraints: List["ConstraintConfig"] = Field(
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
    constraints: List["ConstraintConfig"] = Field(
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
    constraints: List["ConstraintConfig"] = Field(
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
    constraint: "ConstraintConfig" = Field(..., description="Constraint to negate")


# Union type for all constraints
ConstraintConfig = Union[
    SunConstraint,
    MoonConstraint,
    EclipseConstraint,
    EarthLimbConstraint,
    BodyConstraint,
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
