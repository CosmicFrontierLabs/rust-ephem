from ._rust_ephem import (  # type: ignore[import-untyped]
    Constraint,
    ConstraintResult,
    ConstraintViolation,
    GroundEphemeris,
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
    AndConstraintConfig,
    BodyConstraintConfig,
    CombinedConstraintConfig,
    EarthLimbConstraintConfig,
    EclipseConstraintConfig,
    MoonConstraintConfig,
    NotConstraintConfig,
    OrConstraintConfig,
    SunConstraintConfig,
)

# Re-export convenience builders from the pydantic helper module
try:
    from rust_ephem_constraints import (
        and_constraint,
        constraint_to_rust,
        moon_proximity,
        not_constraint,
        sun_proximity,
    )
    from rust_ephem_constraints import (
        eclipse as eclipse_config,
    )

    # provide the names expected by examples
    and_constraint = and_constraint
    constraint_to_rust = constraint_to_rust
    eclipse = eclipse_config
    moon_proximity = moon_proximity
    not_constraint = not_constraint
    sun_proximity = sun_proximity
except Exception:
    # optional; examples may run without these helpers
    pass

__all__ = [
    "ConstraintConfig",
    "SunConstraintConfig",
    "MoonConstraintConfig",
    "EarthLimbConstraintConfig",
    "EclipseConstraintConfig",
    "BodyConstraintConfig",
    "CombinedConstraintConfig",
    "AndConstraintConfig",
    "OrConstraintConfig",
    "NotConstraintConfig",
    "TLEEphemeris",
    "SPICEEphemeris",
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
