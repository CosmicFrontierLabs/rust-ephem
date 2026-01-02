# Create a type alias that supports isinstance checks
import abc
from datetime import datetime

import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.units import Quantity  # type: ignore[import-untyped]

from ._rust_ephem import (
    GroundEphemeris,
    OEMEphemeris,
    PositionVelocityData,
    SPICEEphemeris,
    TLEEphemeris,
)


class Ephemeris(abc.ABC):
    """Abstract base class for all Ephemeris types that supports isinstance checks."""

    # Abstract properties that all ephemeris types must have
    @property
    @abc.abstractmethod
    def timestamp(self) -> npt.NDArray[np.datetime64]:
        """Array of timestamps for the ephemeris."""
        ...

    @property
    @abc.abstractmethod
    def gcrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def itrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in ITRS (Earth-fixed) frame."""
        ...

    @property
    @abc.abstractmethod
    def itrs(self) -> "SkyCoord":
        """SkyCoord object in ITRS frame."""
        ...

    @property
    @abc.abstractmethod
    def gcrs(self) -> "SkyCoord":
        """SkyCoord object in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def earth(self) -> "SkyCoord":
        """SkyCoord object for Earth position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def sun(self) -> "SkyCoord":
        """SkyCoord object for Sun position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def moon(self) -> "SkyCoord":
        """SkyCoord object for Moon position relative to observer."""
        ...

    @property
    @abc.abstractmethod
    def sun_pv(self) -> PositionVelocityData:
        """Sun position and velocity in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def moon_pv(self) -> PositionVelocityData:
        """Moon position and velocity in GCRS frame."""
        ...

    @property
    @abc.abstractmethod
    def obsgeoloc(self) -> npt.NDArray[np.float64]:
        """Observer geocentric location (GCRS position)."""
        ...

    @property
    @abc.abstractmethod
    def obsgeovel(self) -> npt.NDArray[np.float64]:
        """Observer geocentric velocity (GCRS velocity)."""
        ...

    @property
    @abc.abstractmethod
    def latitude(self) -> "Quantity":
        """Geodetic latitude as an astropy Quantity array (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def latitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in degrees as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def latitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in radians as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def longitude(self) -> "Quantity":
        """Geodetic longitude as an astropy Quantity array (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def longitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in degrees as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def longitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in radians as a raw NumPy array."""
        ...

    @property
    @abc.abstractmethod
    def height(self) -> "Quantity":
        """Geodetic height above the WGS84 ellipsoid as an astropy Quantity array (meters)."""
        ...

    @property
    @abc.abstractmethod
    def height_m(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in meters."""
        ...

    @property
    @abc.abstractmethod
    def height_km(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in kilometers."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius(self) -> "Quantity":
        """Angular radius of the Sun with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius_deg(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Sun as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius(self) -> "Quantity":
        """Angular radius of the Moon with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius_deg(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Moon as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius(self) -> "Quantity":
        """Angular radius of the Earth with astropy units (degrees)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius_deg(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Earth as seen from the observer (in degrees)."""
        ...

    @property
    @abc.abstractmethod
    def sun_radius_rad(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Sun as seen from the observer (in radians)."""
        ...

    @property
    @abc.abstractmethod
    def moon_radius_rad(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Moon as seen from the observer (in radians)."""
        ...

    @property
    @abc.abstractmethod
    def earth_radius_rad(self) -> npt.NDArray[np.float64]:
        """Angular radius of the Earth as seen from the observer (in radians)."""
        ...

    @abc.abstractmethod
    def index(self, time: datetime) -> int:
        """Find the index of the closest timestamp to the given datetime."""
        ...

    @property
    @abc.abstractmethod
    def begin(self) -> datetime:
        """Start time of the ephemeris."""
        ...

    @property
    @abc.abstractmethod
    def end(self) -> datetime:
        """End time of the ephemeris."""
        ...

    @property
    @abc.abstractmethod
    def step_size(self) -> int:
        """Time step size in seconds between ephemeris points."""
        ...

    @property
    @abc.abstractmethod
    def polar_motion(self) -> bool:
        """Whether polar motion corrections are applied."""
        ...

    @property
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Sun in degrees."""
        if not hasattr(self, "_sun_ra_deg") or not hasattr(self, "_sun_dec_deg"):
            self._sun_ra_deg = self.sun.ra.deg
            self._sun_dec_deg = self.sun.dec.deg
        return np.vstack((self._sun_ra_deg, self._sun_dec_deg)).T

    @property
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Moon in degrees."""
        if not hasattr(self, "_moon_ra_deg") or not hasattr(self, "_moon_dec_deg"):
            self._moon_ra_deg = self.moon.ra.deg
            self._moon_dec_deg = self.moon.dec.deg
        return np.vstack((self._moon_ra_deg, self._moon_dec_deg)).T

    @property
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Earth in degrees."""
        if not hasattr(self, "_earth_ra_deg") or not hasattr(self, "_earth_dec_deg"):
            self._earth_ra_deg = self.earth.ra.deg
            self._earth_dec_deg = self.earth.dec.deg
        return np.vstack((self._earth_ra_deg, self._earth_dec_deg)).T

    @property
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Sun in radians."""
        if not hasattr(self, "_sun_ra_rad") or not hasattr(self, "_sun_dec_rad"):
            self._sun_ra_rad = self.sun.ra.rad
            self._sun_dec_rad = self.sun.dec.rad
        return np.vstack((self._sun_ra_rad, self._sun_dec_rad)).T

    @property
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Moon in radians."""
        if not hasattr(self, "_moon_ra_rad") or not hasattr(self, "_moon_dec_rad"):
            self._moon_ra_rad = self.moon.ra.rad
            self._moon_dec_rad = self.moon.dec.rad
        return np.vstack((self._moon_ra_rad, self._moon_dec_rad)).T

    @property
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Earth in radians."""
        if not hasattr(self, "_earth_ra_rad") or not hasattr(self, "_earth_dec_rad"):
            self._earth_ra_rad = self.earth.ra.rad
            self._earth_dec_rad = self.earth.dec.rad
        return np.vstack((self._earth_ra_rad, self._earth_dec_rad)).T


# Register all concrete ephemeris classes as virtual subclasses
Ephemeris.register(TLEEphemeris)
Ephemeris.register(SPICEEphemeris)
Ephemeris.register(OEMEphemeris)
Ephemeris.register(GroundEphemeris)


# Also create a Union type for type checking
EphemerisType = TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris
