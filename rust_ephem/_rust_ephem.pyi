"""Type stubs for the Rust extension module _rust_ephem"""

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

class PositionVelocityData:
    """Position and velocity data container"""

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """Position array (N x 3) in kilometers"""
        ...

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        """Velocity array (N x 3) in km/s"""
        ...

    @property
    def position_unit(self) -> str:
        """Unit for position (always 'km')"""
        ...

    @property
    def velocity_unit(self) -> str:
        """Unit for velocity (always 'km/s')"""
        ...

class ConstraintViolation:
    """A single violation of a constraint within a time window"""

    start_time: str
    end_time: str
    max_severity: float
    description: str

    def __repr__(self) -> str: ...

class VisibilityWindow:
    """A time window when the target is not constrained (visible)"""

    start_time: datetime
    end_time: datetime

    def __repr__(self) -> str: ...
    @property
    def duration_seconds(self) -> float:
        """Duration of the visibility window in seconds"""
        ...

class ConstraintResult:
    """Result of constraint evaluation containing all violations"""

    violations: list[ConstraintViolation]
    all_satisfied: bool
    constraint_name: str

    def __repr__(self) -> str: ...
    def total_violation_duration(self) -> float:
        """Get the total duration of violations in seconds"""
        ...

    @property
    def constraint_array(self) -> npt.NDArray[np.bool_]:
        """
        Array of booleans for each timestamp where True means constraint satisfied.

        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.object_]:
        """
        Array of Python datetime objects for each evaluation time.

        Returns a NumPy array (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def visibility(self) -> list[VisibilityWindow]:
        """Array of visibility windows when target is not constrained"""
        ...

    def in_constraint(self, time: datetime) -> bool:
        """
        Check if the target is in-constraint at a given time.

        Args:
            time: A Python datetime object (naive datetimes are treated as UTC)

        Returns:
            True if constraint is satisfied at the given time

        Raises:
            ValueError: If time is not in the evaluated timestamps
        """
        ...

class Constraint:
    """Wrapper for constraint evaluation with ephemeris data"""

    @staticmethod
    def sun_proximity(min_angle: float, max_angle: float | None = None) -> Constraint:
        """
        Create a Sun proximity constraint.

        Args:
            min_angle: Minimum allowed angular separation from Sun in degrees (0-180)
            max_angle: Maximum allowed angular separation from Sun in degrees (optional)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range
        """
        ...

    @staticmethod
    def moon_proximity(min_angle: float, max_angle: float | None = None) -> Constraint:
        """
        Create a Moon proximity constraint.

        Args:
            min_angle: Minimum allowed angular separation from Moon in degrees (0-180)
            max_angle: Maximum allowed angular separation from Moon in degrees (optional)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range
        """
        ...

    @staticmethod
    def earth_limb(min_angle: float, max_angle: float | None = None) -> Constraint:
        """
        Create an Earth limb avoidance constraint.

        Args:
            min_angle: Additional margin beyond Earth's apparent angular radius (degrees)
            max_angle: Maximum allowed angular separation from Earth limb (degrees, optional)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range
        """
        ...

    @staticmethod
    def body_proximity(
        body: str, min_angle: float, max_angle: float | None = None
    ) -> Constraint:
        """
        Create a generic solar system body avoidance constraint.

        Args:
            body: Body identifier - NAIF ID or name (e.g., "Jupiter", "499", "Mars")
            min_angle: Minimum allowed angular separation in degrees (0-180)
            max_angle: Maximum allowed angular separation in degrees (optional)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range

        Note:
            Supported bodies depend on the ephemeris type and loaded kernels.
        """
        ...

    @staticmethod
    def eclipse(umbra_only: bool = True) -> Constraint:
        """
        Create an eclipse constraint.

        Args:
            umbra_only: If True, only umbra counts as eclipse. If False, penumbra also counts.

        Returns:
            A new Constraint instance
        """
        ...

    @staticmethod
    def and_(*constraints: Constraint) -> Constraint:
        """
        Combine constraints with logical AND.

        Args:
            *constraints: Variable number of Constraint objects

        Returns:
            A new Constraint that is satisfied only if all input constraints are satisfied

        Raises:
            ValueError: If no constraints provided
        """
        ...

    @staticmethod
    def or_(*constraints: Constraint) -> Constraint:
        """
        Combine constraints with logical OR.

        Args:
            *constraints: Variable number of Constraint objects

        Returns:
            A new Constraint that is satisfied if any input constraint is satisfied

        Raises:
            ValueError: If no constraints provided
        """
        ...

    @staticmethod
    def xor_(*constraints: Constraint) -> Constraint:
        """
        Combine constraints with logical XOR.

        Args:
            *constraints: Variable number of Constraint objects (minimum 2)

        Returns:
            A new Constraint that is violated when EXACTLY ONE input constraint is violated.

        Raises:
            ValueError: If fewer than two constraints are provided
        """
        ...

    @staticmethod
    def not_(constraint: Constraint) -> Constraint:
        """
        Negate a constraint with logical NOT.

        Args:
            constraint: Constraint to negate

        Returns:
            A new Constraint that is satisfied when the input is violated
        """
        ...

    @staticmethod
    def from_json(json_str: str) -> Constraint:
        """
        Create a constraint from a JSON string.

        Args:
            json_str: JSON representation of the constraint configuration

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If JSON is invalid or contains unknown constraint type

        Example JSON formats:
            {"type": "sun", "min_angle": 45.0}
            {"type": "moon", "min_angle": 10.0}
            {"type": "eclipse", "umbra_only": true}
            {"type": "and", "constraints": [ ... ]}
            {"type": "or", "constraints": [ ... ]}
            {"type": "xor", "constraints": [ ... ]}
            {"type": "not", "constraint": { ... }}
        """
        ...

    def evaluate(
        self,
        ephemeris: TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris,
        target_ra: float,
        target_dec: float,
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> ConstraintResult:
        """
        Evaluate constraint against ephemeris data.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, or GroundEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate. Can be a single datetime
                   or list of datetimes. If provided, only these times will be
                   evaluated (must exist in the ephemeris).
            indices: Optional specific time index/indices to evaluate. Can be a
                     single index or list of indices into the ephemeris timestamp array.

        Returns:
            ConstraintResult containing violation windows

        Raises:
            ValueError: If both times and indices are provided, or if times/indices
                       are not found in the ephemeris
            TypeError: If ephemeris type is not supported

        Note:
            Only one of `times` or `indices` should be provided. If neither is
            provided, all ephemeris times are evaluated.
        """
        ...

    def in_constraint(
        self,
        time: datetime,
        ephemeris: TLEEphemeris | SPICEEphemeris | OEMEphemeris | GroundEphemeris,
        target_ra: float,
        target_dec: float,
    ) -> bool:
        """
        Check if the target is in-constraint at a single time.

        This is optimized for single-time evaluation.

        Args:
            time: The time to check (must exist in ephemeris timestamps)
            ephemeris: One of TLEEphemeris, SPICEEphemeris, or GroundEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)

        Returns:
            True if constraint is satisfied at the given time, False otherwise

        Raises:
            ValueError: If time is not found in ephemeris timestamps
            TypeError: If ephemeris type is not supported
        """
        ...

    def to_json(self) -> str:
        """
        Get constraint configuration as JSON string.

        Returns:
            JSON string representation of the constraint
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Get constraint configuration as Python dictionary.

        Returns:
            Dictionary representation of the constraint
        """
        ...

    def __repr__(self) -> str: ...

class TLEEphemeris:
    """Ephemeris calculator using Two-Line Element (TLE) data"""

    def __init__(
        self,
        tle1: str,
        tle2: str,
        begin: datetime,
        end: datetime,
        step_size: int = 60,
        *,
        polar_motion: bool = False,
    ) -> None:
        """
        Initialize TLE ephemeris from TLE lines.

        Args:
            tle1: First or second line of TLE (line order doesn't matter)
            tle2: Second or first line of TLE (line order doesn't matter)
            begin: Start time (naive datetime treated as UTC)
            end: End time (naive datetime treated as UTC)
            step_size: Time step in seconds (default: 60)
            polar_motion: Whether to apply polar motion correction (default: False)
        """
        ...

    @property
    def teme_pv(self) -> PositionVelocityData:
        """Position and velocity data in TEME frame"""
        ...

    @property
    def itrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in ITRS (Earth-fixed) frame"""
        ...

    @property
    def itrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in ITRS frame"""
        ...

    @property
    def gcrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in GCRS frame"""
        ...

    @property
    def earth(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Earth position relative to satellite"""
        ...

    @property
    def sun(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Sun position relative to satellite"""
        ...

    @property
    def moon(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Moon position relative to satellite"""
        ...

    @property
    def gcrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in GCRS frame"""
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.object_]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def sun_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Sun with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Moon with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Earth with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    def get_body_pv(self, body: str) -> PositionVelocityData:
        """
        Get position and velocity of a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')

        Returns:
            Position and velocity data for the requested body
        """
        ...

    def get_body(self, body: str) -> Any:  # Returns astropy.coordinates.SkyCoord
        """
        Get SkyCoord for a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')

        Returns:
            astropy.coordinates.SkyCoord object
        """
        ...

    def index(self, time: datetime) -> int:
        """
        Find the index of the closest timestamp to the given datetime.

        Returns the index in the ephemeris timestamp array that is closest to the provided time.
        This can be used to index into any of the ephemeris arrays (positions, velocities, etc.)

        Args:
            time: Python datetime object to find the closest match for

        Returns:
            Index of the closest timestamp

        Raises:
            ValueError: If no timestamps are available in the ephemeris

        Example:
            >>> from datetime import datetime
            >>> eph = TLEEphemeris(...)
            >>> target_time = datetime(2024, 1, 15, 12, 0, 0)
            >>> idx = eph.index(target_time)
            >>> position = eph.gcrs_pv.position[idx]
        """
        ...

class SPICEEphemeris:
    """Ephemeris calculator using SPICE kernels"""

    def __init__(
        self,
        spk_path: str,
        naif_id: int,
        begin: datetime,
        end: datetime,
        step_size: int = 60,
        center_id: int = 399,
        *,
        polar_motion: bool = False,
    ) -> None:
        """
        Initialize SPICE ephemeris for a celestial body.

        Args:
            spk_path: Path to SPICE SPK kernel file
            naif_id: NAIF ID of the target body
            begin: Start time (naive datetime treated as UTC)
            end: End time (naive datetime treated as UTC)
            step_size: Time step in seconds (default: 60)
            center_id: NAIF ID of the observer/center (default: 399 = Earth)
            polar_motion: Whether to apply polar motion correction (default: False)
        """
        ...

    @property
    def gcrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in GCRS frame"""
        ...

    @property
    def itrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in ITRS (Earth-fixed) frame"""
        ...

    @property
    def itrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in ITRS frame"""
        ...

    @property
    def gcrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in GCRS frame"""
        ...

    @property
    def earth(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Earth position relative to body"""
        ...

    @property
    def sun(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Sun position relative to body"""
        ...

    @property
    def moon(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Moon position relative to body"""
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.object_]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def sun_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Sun with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the observer (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Moon with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the observer (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Earth with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the observer (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the observer (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the observer (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the observer (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    def index(self, time: datetime) -> int:
        """
        Find the index of the closest timestamp to the given datetime.

        Returns the index in the ephemeris timestamp array that is closest to the provided time.
        This can be used to index into any of the ephemeris arrays (positions, velocities, etc.)

        Args:
            time: Python datetime object to find the closest match for

        Returns:
            Index of the closest timestamp

        Raises:
            ValueError: If no timestamps are available in the ephemeris

        Example:
            >>> from datetime import datetime
            >>> eph = SPICEEphemeris(...)
            >>> target_time = datetime(2024, 1, 15, 12, 0, 0)
            >>> idx = eph.index(target_time)
            >>> position = eph.gcrs_pv.position[idx]
        """
        ...

class OEMEphemeris:
    """
    Ephemeris calculator using CCSDS Orbit Ephemeris Messages (OEM).

    The OEM file must specify a reference frame that is compatible with GCRS
    (Geocentric Celestial Reference System). Compatible frames include:
    - J2000 / EME2000 (Earth Mean Equator and Equinox of J2000.0)
    - GCRF (Geocentric Celestial Reference Frame)
    - ICRF (International Celestial Reference Frame)

    OEM files using Earth-fixed frames (e.g., ITRF) or other incompatible frames
    will be rejected with a ValueError.
    """

    def __init__(
        self,
        oem_path: str,
        begin: datetime,
        end: datetime,
        step_size: int = 60,
        *,
        polar_motion: bool = False,
    ) -> None:
        """
        Initialize CCSDS OEM ephemeris from an OEM file.

        Args:
            oem_path: Path to CCSDS OEM file
            begin: Start time (naive datetime treated as UTC)
            end: End time (naive datetime treated as UTC)
            step_size: Time step in seconds (default: 60)
            polar_motion: Whether to apply polar motion correction (default: False)

        Raises:
            ValueError: If OEM file cannot be parsed, time range exceeds available data,
                       reference frame is missing, or reference frame is incompatible with GCRS
        """
        ...

    @property
    def gcrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in GCRS frame (interpolated)"""
        ...

    @property
    def itrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in ITRS (Earth-fixed) frame"""
        ...

    @property
    def itrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in ITRS frame"""
        ...

    @property
    def gcrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in GCRS frame"""
        ...

    @property
    def earth(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Earth position relative to spacecraft"""
        ...

    @property
    def sun(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Sun position relative to spacecraft"""
        ...

    @property
    def moon(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Moon position relative to spacecraft"""
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.object_]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def oem_pv(self) -> PositionVelocityData:
        """
        Raw OEM position and velocity data without interpolation.

        Returns the original state vectors from the OEM file.
        """
        ...

    @property
    def oem_timestamp(self) -> list[datetime]:
        """
        Raw OEM timestamps without interpolation.

        Returns the original timestamps from the OEM file as UTC datetime objects.
        """
        ...

    @property
    def sun_pv(self) -> PositionVelocityData:
        """Sun position and velocity in GCRS frame"""
        ...

    @property
    def moon_pv(self) -> PositionVelocityData:
        """Moon position and velocity in GCRS frame"""
        ...

    @property
    def obsgeoloc(self) -> npt.NDArray[np.float64]:
        """
        Observer geocentric location (GCRS position).

        Returns position in km, compatible with astropy's GCRS frame obsgeoloc parameter.
        Shape: (N, 3) where N is the number of timestamps.
        """
        ...

    @property
    def obsgeovel(self) -> npt.NDArray[np.float64]:
        """
        Observer geocentric velocity (GCRS velocity).

        Returns velocity in km/s, compatible with astropy's GCRS frame obsgeovel parameter.
        Shape: (N, 3) where N is the number of timestamps.
        """
        ...

    @property
    def sun_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Sun with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Moon with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Earth with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the spacecraft (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def sun_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def moon_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    @property
    def earth_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the spacecraft (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.
        """
        ...

    def index(self, time: datetime) -> int:
        """
        Find the index of the closest timestamp to the given datetime.

        Returns the index in the ephemeris timestamp array that is closest to the provided time.
        This can be used to index into any of the ephemeris arrays (positions, velocities, etc.)

        Args:
            time: Python datetime object to find the closest match for

        Returns:
            Index of the closest timestamp

        Raises:
            ValueError: If no timestamps are available in the ephemeris

        Example:
            >>> from datetime import datetime
            >>> eph = OEMEphemeris("spacecraft.oem", ...)
            >>> target_time = datetime(2024, 1, 15, 12, 0, 0)
            >>> idx = eph.index(target_time)
            >>> position = eph.gcrs_pv.position[idx]
        """
        ...

class GroundEphemeris:
    """Ephemeris for a fixed ground location"""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        height: float,
        begin: datetime,
        end: datetime,
        step_size: int = 60,
        *,
        polar_motion: bool = False,
    ) -> None:
        """
        Initialize ground ephemeris for a fixed location.

        Args:
            latitude: Geodetic latitude in degrees (-90 to 90)
            longitude: Geodetic longitude in degrees (-180 to 180)
            height: Altitude in meters above WGS84 ellipsoid
            begin: Start time (naive datetime treated as UTC)
            end: End time (naive datetime treated as UTC)
            step_size: Time step in seconds (default: 60)
            polar_motion: Whether to apply polar motion correction (default: False)
        """
        ...

    @property
    def gcrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in GCRS frame"""
        ...

    @property
    def itrs_pv(self) -> PositionVelocityData:
        """Position and velocity data in ITRS (Earth-fixed) frame"""
        ...

    @property
    def itrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in ITRS frame for ground location"""
        ...

    @property
    def gcrs(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object in GCRS frame for ground location"""
        ...

    @property
    def earth(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Earth (same as ground location)"""
        ...

    @property
    def sun(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Sun position relative to ground location"""
        ...

    @property
    def moon(self) -> Any:  # Returns astropy.coordinates.SkyCoord
        """SkyCoord object for Moon position relative to ground location"""
        ...

    @property
    def sun_pv(self) -> PositionVelocityData:
        """Position and velocity data for Sun"""
        ...

    @property
    def moon_pv(self) -> PositionVelocityData:
        """Position and velocity data for Moon"""
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.object_]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def obsgeoloc(self) -> Any:  # Returns astropy quantity array
        """Observatory geocentric location for astropy"""
        ...

    @property
    def obsgeovel(self) -> Any:  # Returns astropy quantity array
        """Observatory geocentric velocity for astropy"""
        ...

    @property
    def latitude(self) -> float:
        """Geodetic latitude in degrees"""
        ...

    @property
    def longitude(self) -> float:
        """Geodetic longitude in degrees"""
        ...

    @property
    def height(self) -> float:
        """Altitude in meters above WGS84 ellipsoid"""
        ...

    @property
    def sun_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Sun with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.

        Returns:
            astropy Quantity array with units of degrees
        """
        ...

    @property
    def sun_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the ground station (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in degrees
        """
        ...

    @property
    def moon_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Moon with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.

        Returns:
            astropy Quantity array with units of degrees
        """
        ...

    @property
    def moon_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the ground station (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in degrees
        """
        ...

    @property
    def earth_radius(self) -> Any:  # Returns astropy.units.Quantity
        """
        Angular radius of the Earth with astropy units (degrees).

        Returns an astropy Quantity with units of degrees.
        This property is cached for performance.

        Returns:
            astropy Quantity array with units of degrees
        """
        ...

    @property
    def earth_radius_deg(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the ground station (in degrees).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in degrees
        """
        ...

    @property
    def sun_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Sun as seen from the ground station (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in radians
        """
        ...

    @property
    def moon_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Moon as seen from the ground station (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in radians
        """
        ...

    @property
    def earth_radius_rad(self) -> npt.NDArray[np.float64]:
        """
        Angular radius of the Earth as seen from the ground station (in radians).

        Returns a NumPy array of angular radii for each timestamp.
        Angular radius = arcsin(physical_radius / distance)
        This property is cached for performance.

        Returns:
            NumPy array of angular radii in radians
        """
        ...

    def index(self, time: datetime) -> int:
        """
        Find the index of the closest timestamp to the given datetime.

        Returns the index in the ephemeris timestamp array that is closest to the provided time.
        This can be used to index into any of the ephemeris arrays (positions, velocities, etc.)

        Args:
            time: Python datetime object to find the closest match for

        Returns:
            Index of the closest timestamp

        Raises:
            ValueError: If no timestamps are available in the ephemeris

        Example:
            >>> from datetime import datetime
            >>> eph = GroundEphemeris(...)
            >>> target_time = datetime(2024, 1, 15, 12, 0, 0)
            >>> idx = eph.index(target_time)
            >>> sun_position = eph.sun_pv.position[idx]
        """
        ...

def init_planetary_ephemeris(
    py_path: str,
) -> None:
    """
    Initialize SPICE planetary ephemeris kernels from file.

    Args:
        py_path: Path to the planetary ephemeris kernel file (SPK)

    Raises:
        RuntimeError: If initialization fails
    """
    ...

def download_planetary_ephemeris(
    url: str,
    dest: str,
) -> None:
    """
    Download planetary ephemeris kernel from URL to destination.

    Args:
        url: URL to download the kernel from
        dest: Destination file path

    Raises:
        RuntimeError: If download fails
    """
    ...

def ensure_planetary_ephemeris(
    py_path: str | None = None,
    download_if_missing: bool = True,
    spk_url: str | None = None,
    prefer_full: bool = False,
) -> None:
    """
    Ensure planetary ephemeris is available, downloading if necessary.

    Args:
        py_path: Optional explicit path to kernel file
        download_if_missing: If True, download if file not found
        spk_url: Optional custom URL for download
        prefer_full: If True, prefer full DE440 over slim DE440S

    Raises:
        FileNotFoundError: If file not found and download_if_missing=False
        RuntimeError: If download or initialization fails
    """
    ...

def is_planetary_ephemeris_initialized() -> bool:
    """
    Check if planetary ephemeris has been initialized.

    Returns:
        True if ephemeris is initialized and ready to use
    """
    ...

def get_tai_utc_offset(py_datetime: datetime) -> float | None:
    """
    Get TAI-UTC offset (leap seconds) at the given time.

    Args:
        py_datetime: UTC datetime (naive datetime treated as UTC)

    Returns:
        TAI-UTC offset in seconds, or None if not available
    """
    ...

def get_ut1_utc_offset(py_datetime: datetime) -> float:
    """
    Get UT1-UTC offset at the given time.

    Args:
        py_datetime: UTC datetime (naive datetime treated as UTC)

    Returns:
        UT1-UTC offset in seconds

    Raises:
        RuntimeError: If UT1 provider is not initialized
    """
    ...

def is_ut1_available() -> bool:
    """
    Check if UT1 data is available.

    Returns:
        True if UT1 provider is initialized
    """
    ...

def init_ut1_provider() -> bool:
    """
    Initialize UT1 provider with IERS data.

    Returns:
        True if initialization succeeded
    """
    ...

def get_polar_motion(py_datetime: datetime) -> tuple[float, float]:
    """
    Get polar motion (x, y) at the given time.

    Args:
        py_datetime: UTC datetime (naive datetime treated as UTC)

    Returns:
        Tuple of (x, y) polar motion in arcseconds

    Raises:
        RuntimeError: If EOP provider is not initialized
    """
    ...

def is_eop_available() -> bool:
    """
    Check if Earth Orientation Parameters (EOP) data is available.

    Returns:
        True if EOP provider is initialized
    """
    ...

def init_eop_provider() -> bool:
    """
    Initialize EOP provider with IERS data.

    Returns:
        True if initialization succeeded
    """
    ...

def get_cache_dir() -> str:
    """
    Get the cache directory used for storing ephemeris data.

    Returns:
        String path to the cache directory
    """
    ...
