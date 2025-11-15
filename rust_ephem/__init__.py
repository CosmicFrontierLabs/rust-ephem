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

# Cache for timestamp and constraint_array properties
_timestamp_cache: dict = {}
_constraint_array_cache: dict = {}

# Save the original Rust descriptors BEFORE any modification
_original_constraint_result_timestamp = ConstraintResult.timestamp
_original_constraint_result_constraint_array = ConstraintResult.constraint_array
_original_tle_timestamp = TLEEphemeris.timestamp
_original_spice_timestamp = SPICEEphemeris.timestamp
_original_ground_timestamp = GroundEphemeris.timestamp


def _get_cached_timestamp(self):  # type: ignore[no-untyped-def]
    """Get timestamp with caching to avoid recomputing on every access."""
    obj_id = id(self)
    if obj_id not in _timestamp_cache:
        # Determine which original descriptor to use based on type
        if isinstance(self, ConstraintResult):
            descriptor = _original_constraint_result_timestamp
        elif isinstance(self, TLEEphemeris):
            descriptor = _original_tle_timestamp
        elif isinstance(self, SPICEEphemeris):
            descriptor = _original_spice_timestamp
        elif isinstance(self, GroundEphemeris):
            descriptor = _original_ground_timestamp
        else:
            raise TypeError(f"Unsupported type for timestamp caching: {type(self)}")

        # Call the original Rust descriptor
        _timestamp_cache[obj_id] = descriptor.__get__(self, type(self))
    return _timestamp_cache[obj_id]


def _get_cached_constraint_array(self):  # type: ignore[no-untyped-def]
    """Get constraint_array with caching to avoid recomputing on every access."""
    obj_id = id(self)
    if obj_id not in _constraint_array_cache:
        # Call the original Rust descriptor
        _constraint_array_cache[obj_id] = (
            _original_constraint_result_constraint_array.__get__(self, ConstraintResult)
        )
    return _constraint_array_cache[obj_id]


# Replace properties with cached versions
ConstraintResult.timestamp = property(_get_cached_timestamp)  # type: ignore[misc, assignment]
ConstraintResult.constraint_array = property(_get_cached_constraint_array)  # type: ignore[misc, assignment]
TLEEphemeris.timestamp = property(_get_cached_timestamp)  # type: ignore[misc, assignment]
SPICEEphemeris.timestamp = property(_get_cached_timestamp)  # type: ignore[misc, assignment]
GroundEphemeris.timestamp = property(_get_cached_timestamp)  # type: ignore[misc, assignment]
