"""Type stubs for rust_ephem package"""

from typing import Any

# Re-export from _rust_ephem
from rust_ephem._rust_ephem import (
    Constraint as Constraint,
)
from rust_ephem._rust_ephem import (
    ConstraintResult as ConstraintResult,
)
from rust_ephem._rust_ephem import (
    ConstraintViolation as ConstraintViolation,
)
from rust_ephem._rust_ephem import (
    GroundEphemeris as GroundEphemeris,
)
from rust_ephem._rust_ephem import (
    OEMEphemeris as OEMEphemeris,
)
from rust_ephem._rust_ephem import (
    SPICEEphemeris as SPICEEphemeris,
)
from rust_ephem._rust_ephem import (
    TLEEphemeris as TLEEphemeris,
)
from rust_ephem._rust_ephem import (
    VisibilityWindow as VisibilityWindow,
)
from rust_ephem._rust_ephem import (
    download_planetary_ephemeris as download_planetary_ephemeris,
)
from rust_ephem._rust_ephem import (
    ensure_planetary_ephemeris as ensure_planetary_ephemeris,
)
from rust_ephem._rust_ephem import (
    get_cache_dir as get_cache_dir,
)
from rust_ephem._rust_ephem import (
    get_polar_motion as get_polar_motion,
)
from rust_ephem._rust_ephem import (
    get_tai_utc_offset as get_tai_utc_offset,
)
from rust_ephem._rust_ephem import (
    get_ut1_utc_offset as get_ut1_utc_offset,
)
from rust_ephem._rust_ephem import (
    init_eop_provider as init_eop_provider,
)
from rust_ephem._rust_ephem import (
    init_planetary_ephemeris as init_planetary_ephemeris,
)
from rust_ephem._rust_ephem import (
    init_ut1_provider as init_ut1_provider,
)
from rust_ephem._rust_ephem import (
    is_eop_available as is_eop_available,
)
from rust_ephem._rust_ephem import (
    is_planetary_ephemeris_initialized as is_planetary_ephemeris_initialized,
)
from rust_ephem._rust_ephem import (
    is_ut1_available as is_ut1_available,
)

# Re-export from constraints
from rust_ephem.constraints import (
    AndConstraint as AndConstraint,
)
from rust_ephem.constraints import (
    BodyConstraint as BodyConstraint,
)
from rust_ephem.constraints import (
    CombinedConstraintConfig as CombinedConstraintConfig,
)
from rust_ephem.constraints import (
    EarthLimbConstraint as EarthLimbConstraint,
)
from rust_ephem.constraints import (
    EclipseConstraint as EclipseConstraint,
)
from rust_ephem.constraints import (
    MoonConstraint as MoonConstraint,
)
from rust_ephem.constraints import (
    NotConstraint as NotConstraint,
)
from rust_ephem.constraints import (
    OrConstraint as OrConstraint,
)
from rust_ephem.constraints import (
    SunConstraint as SunConstraint,
)
from rust_ephem.constraints import (
    XorConstraint as XorConstraint,
)

# Optional convenience functions (may not be available)
def and_constraint(*args: Any, **kwargs: Any) -> Any: ...
def constraint_to_rust(*args: Any, **kwargs: Any) -> Any: ...
def eclipse(*args: Any, **kwargs: Any) -> Any: ...
def moon_proximity(*args: Any, **kwargs: Any) -> Any: ...
def not_constraint(*args: Any, **kwargs: Any) -> Any: ...
def sun_proximity(*args: Any, **kwargs: Any) -> Any: ...

__all__ = [
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
