"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, TypeAdapter

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
    ) -> ConstraintResult: ...
    def in_constraint_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> npt.NDArray[np.bool_]: ...
    def evaluate_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> npt.NDArray[np.bool_]: ...
    def in_constraint(
        self,
        time: datetime,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
    ) -> bool: ...
    def and_(self, other: ConstraintConfig) -> AndConstraint: ...
    def or_(self, other: ConstraintConfig) -> OrConstraint: ...
    def xor_(self, other: ConstraintConfig) -> XorConstraint: ...
    def not_(self) -> NotConstraint: ...
    def __and__(self, other: ConstraintConfig) -> AndConstraint: ...
    def __or__(self, other: ConstraintConfig) -> OrConstraint: ...
    def __xor__(self, other: ConstraintConfig) -> XorConstraint: ...
    def __invert__(self) -> NotConstraint: ...

class SunConstraint(RustConstraintMixin):
    """Sun proximity constraint

    Ensures target maintains minimum angular separation from Sun.

    Attributes:
        type: Always "sun"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["sun"] = "sun"
    min_angle: float
    max_angle: float | None = None

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
    min_angle: float
    max_angle: float | None = None
    include_refraction: bool = False
    horizon_dip: bool = False

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
    body: str
    min_angle: float
    max_angle: float | None = None

class MoonConstraint(RustConstraintMixin):
    """Moon proximity constraint

    Ensures target maintains minimum angular separation from Moon.

    Attributes:
        type: Always "moon"
        min_angle: Minimum allowed angular separation in degrees (0-180)
        max_angle: Maximum allowed angular separation in degrees (0-180), optional
    """

    type: Literal["moon"] = "moon"
    min_angle: float
    max_angle: float | None = None

class EclipseConstraint(RustConstraintMixin):
    """Eclipse constraint

    Checks if observer is in Earth's shadow (umbra and/or penumbra).

    Attributes:
        type: Always "eclipse"
        umbra_only: If True, only umbra counts. If False, includes penumbra.
    """

    type: Literal["eclipse"] = "eclipse"
    umbra_only: bool = True

class AndConstraint(RustConstraintMixin):
    """Logical AND constraint combinator

    Satisfied only if ALL sub-constraints are satisfied.

    Attributes:
        type: Always "and"
        constraints: List of constraints to combine with AND
    """

    type: Literal["and"] = "and"
    constraints: list[ConstraintConfig]

class OrConstraint(RustConstraintMixin):
    """Logical OR constraint combinator

    Satisfied if ANY sub-constraint is satisfied.

    Attributes:
        type: Always "or"
        constraints: List of constraints to combine with OR
    """

    type: Literal["or"] = "or"
    constraints: list[ConstraintConfig]

class XorConstraint(RustConstraintMixin):
    """Logical XOR constraint combinator

    Satisfied if EXACTLY ONE sub-constraint is satisfied.

    Attributes:
        type: Always "xor"
        constraints: List of constraints to combine with XOR (minimum 2)
    """

    type: Literal["xor"] = "xor"
    constraints: list[ConstraintConfig]

class NotConstraint(RustConstraintMixin):
    """Logical NOT constraint combinator

    Inverts a constraint - satisfied when inner constraint is violated.

    Attributes:
        type: Always "not"
        constraint: Constraint to negate
    """

    type: Literal["not"] = "not"
    constraint: ConstraintConfig

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

# Type adapter for ConstraintConfig union
CombinedConstraintConfig: TypeAdapter[ConstraintConfig] = TypeAdapter(ConstraintConfig)
