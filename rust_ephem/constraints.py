"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from typing import List, Literal, Union, cast

from pydantic import BaseModel, Field, TypeAdapter


class RustConstraintMixin(BaseModel):
    """Base class for Rust constraint configurations"""

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the constraint using the Rust backend.

        This method lazily creates the corresponding Rust constraint
        object on first use.

        Args:
            *args: Positional arguments for evaluation
            **kwargs: Keyword arguments for evaluation

        Returns:
            Result of the Rust constraint evaluation
        """
        if not hasattr(self, "_rust_constraint"):
            from rust_ephem import Constraint

            self._rust_constraint = Constraint.from_json(self.model_dump_json())
        return self._rust_constraint.evaluate(*args, **kwargs)

    def and_(self, other: "ConstraintConfig") -> "AndConstraintConfig":
        """Combine this constraint with another using logical AND

        Args:
            other: Another constraint configuration

        Returns:
            AndConstraintConfig combining both constraints
        """
        return AndConstraintConfig(constraints=[cast("ConstraintConfig", self), other])

    def or_(self, other: "ConstraintConfig") -> "OrConstraintConfig":
        """Combine this constraint with another using logical OR

        Args:
            other: Another constraint configuration

        Returns:
            OrConstraintConfig combining both constraints
        """
        return OrConstraintConfig(constraints=[cast("ConstraintConfig", self), other])

    def not_(self) -> "NotConstraintConfig":
        """Negate this constraint using logical NOT

        Returns:
            NotConstraintConfig negating this constraint
        """
        return NotConstraintConfig(constraint=cast("ConstraintConfig", self))

    def __and__(self, other: "ConstraintConfig") -> "AndConstraintConfig":
        """Combine constraints using & operator (logical AND)

        Args:
            other: Another constraint configuration

        Returns:
            AndConstraintConfig combining both constraints

        Example:
            >>> sun = SunConstraintConfig(min_angle=45.0)
            >>> moon = MoonConstraintConfig(min_angle=30.0)
            >>> combined = sun & moon
        """
        return self.and_(other)

    def __or__(self, other: "ConstraintConfig") -> "OrConstraintConfig":
        """Combine constraints using | operator (logical OR)

        Args:
            other: Another constraint configuration

        Returns:
            OrConstraintConfig combining both constraints

        Example:
            >>> sun = SunConstraintConfig(min_angle=45.0)
            >>> moon = MoonConstraintConfig(min_angle=30.0)
            >>> combined = sun | moon
        """
        return self.or_(other)

    def __invert__(self) -> "NotConstraintConfig":
        """Negate constraint using ~ operator (logical NOT)

        Returns:
            NotConstraintConfig negating this constraint

        Example:
            >>> sun = SunConstraintConfig(min_angle=45.0)
            >>> not_sun = ~sun
        """
        return self.not_()


class SunConstraintConfig(RustConstraintMixin):
    """Configuration for Sun  constraint

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


class EarthLimbConstraintConfig(RustConstraintMixin):
    """Configuration for Earth limb constraint

    Ensures target maintains minimum angular separation from Earth's limb.

    Attributes:
        type: Always "earth_limb"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
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


class BodyConstraintConfig(RustConstraintMixin):
    """Configuration for generic solar system body proximity constraint

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


class MoonConstraintConfig(RustConstraintMixin):
    """Configuration for Moon  constraint

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


class EclipseConstraintConfig(RustConstraintMixin):
    """Configuration for eclipse constraint

    Checks if observer is in Earth's shadow (umbra and/or penumbra).

    Attributes:
        type: Always "eclipse"
        umbra_only: If True, only umbra counts. If False, includes penumbra.
    """

    type: Literal["eclipse"] = "eclipse"
    umbra_only: bool = Field(
        default=True, description="Count only umbra (True) or include penumbra (False)"
    )


class AndConstraintConfig(RustConstraintMixin):
    """Configuration for logical AND combinator

    Satisfied only if ALL sub-constraints are satisfied.

    Attributes:
        type: Always "and"
        constraints: List of constraint configurations to combine with AND
    """

    type: Literal["and"] = "and"
    constraints: List["ConstraintConfig"] = Field(
        ..., min_length=1, description="Constraints to AND together"
    )


class OrConstraintConfig(RustConstraintMixin):
    """Configuration for logical OR combinator

    Satisfied if ANY sub-constraint is satisfied.

    Attributes:
        type: Always "or"
        constraints: List of constraint configurations to combine with OR
    """

    type: Literal["or"] = "or"
    constraints: List["ConstraintConfig"] = Field(
        ..., min_length=1, description="Constraints to OR together"
    )


class NotConstraintConfig(RustConstraintMixin):
    """Configuration for logical NOT combinator

    Inverts a constraint - satisfied when inner constraint is violated.

    Attributes:
        type: Always "not"
        constraint: Constraint configuration to negate
    """

    type: Literal["not"] = "not"
    constraint: "ConstraintConfig" = Field(..., description="Constraint to negate")


# Union type for all constraint configs
ConstraintConfig = Union[
    SunConstraintConfig,
    MoonConstraintConfig,
    EclipseConstraintConfig,
    EarthLimbConstraintConfig,
    BodyConstraintConfig,
    AndConstraintConfig,
    OrConstraintConfig,
    NotConstraintConfig,
]


# Update forward references after ConstraintConfig is defined
AndConstraintConfig.model_rebuild()
OrConstraintConfig.model_rebuild()
NotConstraintConfig.model_rebuild()


# A single type adapter for ConstraintConfig
CombinedConstraintConfig: TypeAdapter[ConstraintConfig] = TypeAdapter(ConstraintConfig)
