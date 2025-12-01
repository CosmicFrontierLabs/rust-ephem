from typing import Union

from ._rust_ephem import (  # type: ignore[import-untyped]
    Constraint,
    ConstraintResult,
    ConstraintViolation,
    GroundEphemeris,
    OEMEphemeris,
    SPICEEphemeris,
    TLEEphemeris,
    VisibilityWindow,
    download_planetary_ephemeris,
    ensure_planetary_ephemeris,
    get_cache_dir,
    get_polar_motion,
    get_tai_utc_offset,
    get_ut1_utc_offset,
    init_eop_provider,
    init_planetary_ephemeris,
    init_ut1_provider,
    is_eop_available,
    is_planetary_ephemeris_initialized,
    is_ut1_available,
)
from .constraints import (
    AndConstraint,
    BodyConstraint,
    CombinedConstraintConfig,
    EarthLimbConstraint,
    EclipseConstraint,
    MoonConstraint,
    NotConstraint,
    OrConstraint,
    SunConstraint,
    XorConstraint,
)


# Create a type alias that supports isinstance checks
class _EphemerisMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(
            instance, (TLEEphemeris, SPICEEphemeris, OEMEphemeris, GroundEphemeris)
        )


class Ephemeris(metaclass=_EphemerisMeta):
    """Type alias for all ephemeris types that supports isinstance checks."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} cannot be instantiated directly. "
            "Use one of the concrete ephemeris classes: "
            "TLEEphemeris, SPICEEphemeris, OEMEphemeris, or GroundEphemeris"
        )


# Also create a Union type for type checking
EphemerisType = Union[TLEEphemeris, SPICEEphemeris, OEMEphemeris, GroundEphemeris]

__all__ = [
    "ConstraintConfig",
    "SunConstraint",
    "MoonConstraint",
    "EarthLimbConstraint",
    "EclipseConstraint",
    "BodyConstraint",
    "CombinedConstraintConfig",
    "AndConstraint",
    "OrConstraint",
    "XorConstraint",
    "NotConstraint",
    "TLEEphemeris",
    "SPICEEphemeris",
    "OEMEphemeris",
    "GroundEphemeris",
    "Ephemeris",
    "Constraint",
    "ConstraintResult",
    "ConstraintViolation",
    "VisibilityWindow",
    "init_planetary_ephemeris",
    "download_planetary_ephemeris",
    "ensure_planetary_ephemeris",
    "is_planetary_ephemeris_initialized",
    "get_tai_utc_offset",
    "get_ut1_utc_offset",
    "is_ut1_available",
    "init_ut1_provider",
    "get_polar_motion",
    "is_eop_available",
    "init_eop_provider",
    "get_cache_dir",
]
