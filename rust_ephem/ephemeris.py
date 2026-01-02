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
    @abc.abstractmethod
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Sun in degrees.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Sun
        relative to the observer (spacecraft or ground station) in the GCRS frame.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.sun_ra_dec_deg
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Moon in degrees.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Moon
        relative to the observer (spacecraft or ground station) in the GCRS frame.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.moon_ra_dec_deg
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Earth in degrees.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Earth
        relative to the observer (spacecraft). For GroundEphemeris, this represents
        the ground station's own position.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.earth_ra_dec_deg
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Sun in radians.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Sun
        relative to the observer (spacecraft or ground station) in the GCRS frame.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.sun_ra_dec_rad
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Moon in radians.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Moon
        relative to the observer (spacecraft or ground station) in the GCRS frame.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.moon_ra_dec_rad
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension and Declination of the Earth in radians.

        Provides a convenient Nx2 NumPy array with celestial coordinates of the Earth
        relative to the observer (spacecraft). For GroundEphemeris, this represents
        the ground station's own position.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. Subsequent accesses return
            the same array without recomputation.

        Example:
            >>> ra_dec = ephem.earth_ra_dec_rad
            >>> ra = ra_dec[:, 0]   # All RA values
            >>> dec = ra_dec[:, 1]  # All Dec values
        """
        ...

    @property
    @abc.abstractmethod
    def sun_ra_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Sun in degrees.

        Convenience property that extracts just the RA column from sun_ra_dec_deg.

        Returns:
            1D NumPy array of Right Ascension values in degrees [0, 360)
        """
        ...

    @property
    @abc.abstractmethod
    def sun_dec_deg(self) -> npt.NDArray[np.float64]:
        """Declination of the Sun in degrees.

        Convenience property that extracts just the Dec column from sun_ra_dec_deg.

        Returns:
            1D NumPy array of Declination values in degrees [-90, 90]
        """
        ...

    @property
    @abc.abstractmethod
    def moon_ra_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Moon in degrees.

        Convenience property that extracts just the RA column from moon_ra_dec_deg.

        Returns:
            1D NumPy array of Right Ascension values in degrees [0, 360)
        """
        ...

    @property
    @abc.abstractmethod
    def moon_dec_deg(self) -> npt.NDArray[np.float64]:
        """Declination of the Moon in degrees.

        Convenience property that extracts just the Dec column from moon_ra_dec_deg.

        Returns:
            1D NumPy array of Declination values in degrees [-90, 90]
        """
        ...

    @property
    @abc.abstractmethod
    def earth_ra_deg(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Earth in degrees.

        Convenience property that extracts just the RA column from earth_ra_dec_deg.

        Returns:
            1D NumPy array of Right Ascension values in degrees [0, 360)
        """
        ...

    @property
    @abc.abstractmethod
    def earth_dec_deg(self) -> npt.NDArray[np.float64]:
        """Declination of the Earth in degrees.

        Convenience property that extracts just the Dec column from earth_ra_dec_deg.

        Returns:
            1D NumPy array of Declination values in degrees [-90, 90]
        """
        ...

    @property
    @abc.abstractmethod
    def sun_ra_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Sun in radians.

        Convenience property that extracts just the RA column from sun_ra_dec_rad.

        Returns:
            1D NumPy array of Right Ascension values in radians [0, 2π)
        """
        ...

    @property
    @abc.abstractmethod
    def sun_dec_rad(self) -> npt.NDArray[np.float64]:
        """Declination of the Sun in radians.

        Convenience property that extracts just the Dec column from sun_ra_dec_rad.

        Returns:
            1D NumPy array of Declination values in radians [-π/2, π/2]
        """
        ...

    @property
    @abc.abstractmethod
    def moon_ra_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Moon in radians.

        Convenience property that extracts just the RA column from moon_ra_dec_rad.

        Returns:
            1D NumPy array of Right Ascension values in radians [0, 2π)
        """
        ...

    @property
    @abc.abstractmethod
    def moon_dec_rad(self) -> npt.NDArray[np.float64]:
        """Declination of the Moon in radians.

        Convenience property that extracts just the Dec column from moon_ra_dec_rad.

        Returns:
            1D NumPy array of Declination values in radians [-π/2, π/2]
        """
        ...

    @property
    @abc.abstractmethod
    def earth_ra_rad(self) -> npt.NDArray[np.float64]:
        """Right Ascension of the Earth in radians.

        Convenience property that extracts just the RA column from earth_ra_dec_rad.

        Returns:
            1D NumPy array of Right Ascension values in radians [0, 2π)
        """
        ...

    @property
    @abc.abstractmethod
    def earth_dec_rad(self) -> npt.NDArray[np.float64]:
        """Declination of the Earth in radians.

        Convenience property that extracts just the Dec column from earth_ra_dec_rad.

        Returns:
            1D NumPy array of Declination values in radians [-π/2, π/2]
        """
        ...


# Register all concrete ephemeris classes as virtual subclasses
Ephemeris.register(TLEEphemeris)
Ephemeris.register(SPICEEphemeris)
Ephemeris.register(OEMEphemeris)
Ephemeris.register(GroundEphemeris)


# Also create a Union type for type checking
EphemerisType = TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris
