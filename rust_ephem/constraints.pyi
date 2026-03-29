"""
Pydantic models for constraint configuration

This module provides type-safe configuration models for constraints
using Pydantic. These models can be serialized to/from JSON and used
to configure the Rust constraint evaluators.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, TypeAdapter

from .ephemeris import Ephemeris

DEFAULT_N_POINTS: int
DEFAULT_N_ROLL_SAMPLES: int

if TYPE_CHECKING:
    from rust_ephem import VisibilityWindow

if TYPE_CHECKING:
    pass

class RollReference(str, Enum):
    SUN = "sun"
    NORTH = "north"

class ConstraintViolation(BaseModel):
    """A time window where a constraint was violated."""

    start_time: datetime
    end_time: datetime
    max_severity: float
    description: str

class ConstraintResult(BaseModel):
    """Result of constraint evaluation containing all violations."""

    violations: list[ConstraintViolation]
    all_satisfied: bool
    constraint_name: str

    @property
    def timestamps(self) -> npt.NDArray[np.datetime64] | list[datetime]: ...
    @property
    def constraint_array(self) -> list[bool]: ...
    @property
    def visibility(self) -> list[VisibilityWindow]: ...
    def total_violation_duration(self) -> float: ...
    def in_constraint(self, time: datetime) -> bool: ...

class VisibilityWindowResult(BaseModel):
    start_time: datetime
    end_time: datetime
    duration_seconds: float

class MovingVisibilityResult(BaseModel):
    timestamps: list[datetime]
    ras: list[float]
    decs: list[float]
    constraint_array: list[bool]
    visibility_flags: list[bool]
    visibility: list[VisibilityWindowResult]
    all_satisfied: bool
    constraint_name: str

class RustConstraintMixin(BaseModel):
    """Base class for Rust constraint configurations"""

    def evaluate(
        self,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
        target_roll: float | None = None,
    ) -> ConstraintResult: ...
    def in_constraint_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
        target_roll: float | None = None,
    ) -> npt.NDArray[np.bool_]: ...
    def in_constraint(
        self,
        time: datetime | list[datetime] | npt.NDArray[np.datetime64],
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        target_roll: float | None = None,
    ) -> bool | list[bool]: ...
    def instantaneous_field_of_regard(
        self,
        ephemeris: Ephemeris,
        time: datetime | None = None,
        index: int | None = None,
        n_points: int = DEFAULT_N_POINTS,
        n_roll_samples: int = DEFAULT_N_ROLL_SAMPLES,
        target_roll: float | None = None,
    ) -> float: ...
    def evaluate_moving_body(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float] | npt.ArrayLike | None = None,
        target_decs: list[float] | npt.ArrayLike | None = None,
        times: datetime | list[datetime] | None = None,
        body: str | int | None = None,
        use_horizons: bool = False,
        spice_kernel: str | None = None,
        target_roll: float | None = None,
    ) -> MovingVisibilityResult:
        """
        Evaluate constraint for a moving body (varying RA/Dec over time).

        Performance:
            The constraint evaluation itself is highly optimized (~0.3 µs per timestamp).
            When using `body=`, each call fetches positions from SPICE (~80ms for 10k timestamps).
            For best performance with multiple constraints, pre-fetch coordinates once::

                # FAST: Pre-fetch once
                skycoord = ephem.get_body("Jupiter")
                ras, decs = list(skycoord.ra.deg), list(skycoord.dec.deg)
                result1 = sun_c.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)
                result2 = moon_c.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)

                # SLOWER: Re-fetches each call
                result1 = sun_c.evaluate_moving_body(ephem, body="Jupiter")
                result2 = moon_c.evaluate_moving_body(ephem, body="Jupiter")
        """
        ...
    def and_(self, other: ConstraintConfig) -> AndConstraint: ...
    def or_(self, other: ConstraintConfig) -> OrConstraint: ...
    def xor_(self, other: ConstraintConfig) -> XorConstraint: ...
    def at_least(
        self, min_violated: int, *others: ConstraintConfig
    ) -> AtLeastConstraint: ...
    def not_(self) -> NotConstraint: ...
    def boresight_offset(
        self,
        roll_deg: float = 0.0,
        roll_clockwise: bool = False,
        roll_reference: RollReference = RollReference.NORTH,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> BoresightOffsetConstraint: ...
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
    min_altitude: float | None = None
    max_altitude: float | None = None
    min_azimuth: float | None = None
    max_azimuth: float | None = None
    polygon: list[tuple[float, float]] | None = None

class OrbitRamConstraint(RustConstraintMixin):
    type: Literal["orbit_ram"] = "orbit_ram"
    min_angle: float
    max_angle: float | None = None

class OrbitPoleConstraint(RustConstraintMixin):
    type: Literal["orbit_pole"] = "orbit_pole"
    min_angle: float
    max_angle: float | None = None
    earth_limb_pole: bool = False

class AndConstraint(RustConstraintMixin):
    type: Literal["and"] = "and"
    constraints: list[ConstraintConfig]

class OrConstraint(RustConstraintMixin):
    type: Literal["or"] = "or"
    constraints: list[ConstraintConfig]

class XorConstraint(RustConstraintMixin):
    type: Literal["xor"] = "xor"
    constraints: list[ConstraintConfig]

class AtLeastConstraint(RustConstraintMixin):
    type: Literal["at_least"] = "at_least"
    min_violated: int
    constraints: list[ConstraintConfig]

class NotConstraint(RustConstraintMixin):
    type: Literal["not"] = "not"
    constraint: ConstraintConfig

class BoresightOffsetConstraint(RustConstraintMixin):
    type: Literal["boresight_offset"] = "boresight_offset"
    constraint: ConstraintConfig
    roll_deg: float = 0.0
    roll_clockwise: bool = False
    roll_reference: RollReference = RollReference.NORTH
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

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
    | OrbitRamConstraint
    | OrbitPoleConstraint
    | AndConstraint
    | OrConstraint
    | XorConstraint
    | AtLeastConstraint
    | NotConstraint
    | BoresightOffsetConstraint
)
CombinedConstraintConfig: TypeAdapter[ConstraintConfig]
