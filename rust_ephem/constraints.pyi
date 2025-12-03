"""Type stubs for constraint configuration models"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

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
        self, time: datetime, ephemeris: Ephemeris, target_ra: float, target_dec: float
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
    type: Literal["sun"]
    min_angle: float
    max_angle: float | None

class EarthLimbConstraint(RustConstraintMixin):
    type: Literal["earth_limb"]
    min_angle: float
    max_angle: float | None
    include_refraction: bool
    horizon_dip: bool

class BodyConstraint(RustConstraintMixin):
    type: Literal["body"]
    body: str
    min_angle: float
    max_angle: float | None

class MoonConstraint(RustConstraintMixin):
    type: Literal["moon"]
    min_angle: float
    max_angle: float | None

class EclipseConstraint(RustConstraintMixin):
    type: Literal["eclipse"]
    umbra_only: bool

class AndConstraint(RustConstraintMixin):
    type: Literal["and"]
    constraints: list[ConstraintConfig]

class OrConstraint(RustConstraintMixin):
    type: Literal["or"]
    constraints: list[ConstraintConfig]

class XorConstraint(RustConstraintMixin):
    type: Literal["xor"]
    constraints: list[ConstraintConfig]

class NotConstraint(RustConstraintMixin):
    type: Literal["not"]
    constraint: ConstraintConfig

ConstraintConfig = (
    SunConstraint
    | MoonConstraint
    | EclipseConstraint
    | EarthLimbConstraint
    | BodyConstraint
    | AndConstraint
    | OrConstraint
    | XorConstraint
    | NotConstraint
)
CombinedConstraintConfig: TypeAdapter[ConstraintConfig]
