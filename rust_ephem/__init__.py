# Create a type alias that supports isinstance checks
import abc
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


class Ephemeris(abc.ABC):
    """Abstract base class for all ephemeris types that supports isinstance checks."""

    # Abstract properties that all ephemeris types must have
    @property
    @abc.abstractmethod
    def timestamp(self):
        """Array of timestamps for the ephemeris."""
        ...

    @property
    @abc.abstractmethod
    def gcrs_pv(self):
        """Position and velocity data in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def itrs_pv(self):
        """Position and velocity data in ITRS (Earth-fixed) frame."""
        ...

    @property
    @abc.abstractmethod
    def itrs(self):
        """SkyCoord object in ITRS frame."""
        ...

    @property
    @abc.abstractmethod
    def gcrs(self):
        """SkyCoord object in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def earth(self):
        """SkyCoord object for Earth position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def sun(self):
        """SkyCoord object for Sun position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def moon(self):
        """SkyCoord object for Moon position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def sun_pv(self):
        """Sun position and velocity in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def moon_pv(self):
        """Moon position and velocity in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def obsgeoloc(self):
        """Observer geocentric location (GCRS position)."""
        ...

    @property
    @abc.abstractmethod
    def obsgeovel(self):
        """Observer geocentric velocity (GCRS velocity)."""
        ...

    @property
    @abc.abstractmethod
    def latitude(self):
        """Geodetic latitude as an astropy Quantity array (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def latitude_deg(self):
        """Geodetic latitude in degrees as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def latitude_rad(self):
        """Geodetic latitude in radians as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def longitude(self):
        """Geodetic longitude as an astropy Quantity array (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def longitude_deg(self):
        """Geodetic longitude in degrees as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def longitude_rad(self):
        """Geodetic longitude in radians as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def height(self):
        """Geodetic height above the WGS84 ellipsoid as an astropy Quantity array (meters)."""
        ...

    @property
    @abc.abstractmethod
    def height_m(self):
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in meters."""
        ...

    @property
    @abc.abstractmethod
    def height_km(self):
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in kilometers."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius(self):
        """Angular radius of the Sun with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius_deg(self):
        """Angular radius of the Sun as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius(self):
        """Angular radius of the Moon with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius_deg(self):
        """Angular radius of the Moon as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius(self):
        """Angular radius of the Earth with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius_deg(self):
        """Angular radius of the Earth as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius_rad(self):
        """Angular radius of the Sun as seen from the observer (in radians)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius_rad(self):
        """Angular radius of the Moon as seen from the observer (in radians)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius_rad(self):
        """Angular radius of the Earth as seen from the observer (in radians)."""
        ...

    @abc.abstractmethod
    def index(self, time):
        """Find the index of the closest timestamp to the given datetime."""
        ...


# Register all concrete ephemeris classes as virtual subclasses
Ephemeris.register(TLEEphemeris)
Ephemeris.register(SPICEEphemeris)
Ephemeris.register(OEMEphemeris)
Ephemeris.register(GroundEphemeris)


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
