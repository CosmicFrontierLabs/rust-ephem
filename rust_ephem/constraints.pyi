"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

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
    def in_constraint(
        self,
        time: datetime | list[datetime] | npt.NDArray[np.datetime64],
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
    ) -> bool | list[bool]: ...
    def and_(self, other: ConstraintConfig) -> AndConstraint: ...
    def or_(self, other: ConstraintConfig) -> OrConstraint: ...
    def xor_(self, other: ConstraintConfig) -> XorConstraint: ...
    def not_(self) -> NotConstraint: ...
    def __and__(self, other: ConstraintConfig) -> AndConstraint: ...
    def __or__(self, other: ConstraintConfig) -> OrConstraint: ...
    def __xor__(self, other: ConstraintConfig) -> XorConstraint: ...
    def __invert__(self) -> NotConstraint: ...

class SunConstraint(RustConstraintMixin):
    type: Literal["sun"] = "sun"
    min_angle: float
    max_angle: float | None = None

class EarthLimbConstraint(RustConstraintMixin):
    type: Literal["earth_limb"] = "earth_limb"
    min_angle: float
    max_angle: float | None = None
    include_refraction: bool = False
    horizon_dip: bool = False

class BodyConstraint(RustConstraintMixin):
    type: Literal["body"] = "body"
    body: str
    min_angle: float
    max_angle: float | None = None

class MoonConstraint(RustConstraintMixin):
    type: Literal["moon"] = "moon"
    min_angle: float
    max_angle: float | None = None

class EclipseConstraint(RustConstraintMixin):
    type: Literal["eclipse"] = "eclipse"
    umbra_only: bool = True

class DaytimeConstraint(RustConstraintMixin):
    type: Literal["daytime"] = "daytime"
    twilight: Literal["civil", "nautical", "astronomical", "none"] = "civil"

class AirmassConstraint(RustConstraintMixin):
    type: Literal["airmass"] = "airmass"
    min_airmass: float | None = None
    max_airmass: float

class MoonPhaseConstraint(RustConstraintMixin):
    type: Literal["moon_phase"] = "moon_phase"
    min_illumination: float | None = None
    max_illumination: float
    min_distance: float | None = None
    max_distance: float | None = None
    enforce_when_below_horizon: bool = False
    moon_visibility: Literal["full", "partial"] = "full"

class SAAConstraint(RustConstraintMixin):
    type: Literal["saa"] = "saa"
    polygon: list[tuple[float, float]]

class AltAzConstraint(RustConstraintMixin):
    type: Literal["alt_az"] = "alt_az"
    min_altitude: float
    max_altitude: float | None = None
    min_azimuth: float | None = None
    max_azimuth: float | None = None

class AndConstraint(RustConstraintMixin):
    type: Literal["and"] = "and"
    constraints: list[ConstraintConfig]

class OrConstraint(RustConstraintMixin):
    type: Literal["or"] = "or"
    constraints: list[ConstraintConfig]

class XorConstraint(RustConstraintMixin):
    type: Literal["xor"] = "xor"
    constraints: list[ConstraintConfig]

class NotConstraint(RustConstraintMixin):
    type: Literal["not"] = "not"
    constraint: ConstraintConfig

ConstraintConfig = (
    SunConstraint
    | MoonConstraint
    | EclipseConstraint
    | EarthLimbConstraint
    | BodyConstraint
    | DaytimeConstraint
    | AirmassConstraint
    | MoonPhaseConstraint
    | SAAConstraint
    | AltAzConstraint
    | AndConstraint
    | OrConstraint
    | XorConstraint
    | NotConstraint
)
CombinedConstraintConfig: TypeAdapter[ConstraintConfig]
