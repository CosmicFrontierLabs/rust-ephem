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

# Cache for ConstraintResult timestamp property
_timestamp_cache: dict = {}

# Save the original Rust timestamp descriptor BEFORE any modification
_original_timestamp_descriptor = ConstraintResult.timestamp


def _get_cached_timestamp(self):  # type: ignore[no-untyped-def]
    """Get timestamp with caching to avoid recomputing on every access."""
    obj_id = id(self)
    if obj_id not in _timestamp_cache:
        # Call the original Rust descriptor
        _timestamp_cache[obj_id] = _original_timestamp_descriptor.__get__(
            self, ConstraintResult
        )
    return _timestamp_cache[obj_id]


# Replace the timestamp property with cached version
ConstraintResult.timestamp = property(_get_cached_timestamp)  # type: ignore[misc, assignment]
