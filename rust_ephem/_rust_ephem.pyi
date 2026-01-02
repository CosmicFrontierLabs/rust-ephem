"""Type stubs for the Rust extension module _rust_ephem"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .ephemeris import Ephemeris

@runtime_checkable
class TLELike(Protocol):
    """Protocol for objects that can be used as TLE data (e.g., TLERecord)."""

    @property
    def line1(self) -> str:
        """First line of TLE."""
        ...

    @property
    def line2(self) -> str:
        """Second line of TLE."""
        ...

    @property
    def epoch(self) -> datetime:
        """TLE epoch."""
        ...

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

class VisibilityWindow:
    """A time window when the target is not constrained (visible)"""

    start_time: datetime
    end_time: datetime

    def __repr__(self) -> str: ...
    @property
    def duration_seconds(self) -> float:
        """Duration of the visibility window in seconds"""
        ...

class ConstraintViolation:
    """A time window when a constraint was violated."""

    start_time: datetime
    end_time: datetime
    max_severity: float
    description: str

    def __repr__(self) -> str: ...
    @property
    def duration_seconds(self) -> float:
        """Duration of the violation window in seconds."""
        ...

class MovingBodyResult:
    """Result from evaluating a constraint against a moving body.

    Contains the constraint evaluation results along with the body's
    RA/Dec coordinates at each timestamp.
    """

    @property
    def violations(self) -> list["ConstraintViolation"]:
        """List of time windows when constraint was violated."""
        ...

    @property
    def all_satisfied(self) -> bool:
        """True if constraint was satisfied at all timestamps."""
        ...

    @property
    def constraint_name(self) -> str:
        """Name/description of the constraint."""
        ...

    @property
    def timestamp(self) -> list[datetime]:
        """List of timestamps that were evaluated."""
        ...

    @property
    def ras(self) -> list[float]:
        """Right ascensions in degrees at each timestamp."""
        ...

    @property
    def decs(self) -> list[float]:
        """Declinations in degrees at each timestamp."""
        ...

    @property
    def constraint_array(self) -> list[bool]:
        """Boolean array where True indicates constraint violation at that timestamp."""
        ...

    @property
    def visibility(self) -> list[VisibilityWindow]:
        """List of time windows when constraint was satisfied (target was visible)."""
        ...

    def in_constraint(self, time: datetime) -> bool:
        """Check if constraint was violated at a specific time.

        Args:
            time: The time to check (must exist in timestamps).

        Returns:
            True if constraint was violated at the given time.

        Raises:
            ValueError: If time is not found in timestamps.
        """
        ...

    def total_violation_duration(self) -> float:
        """Get total duration of all constraint violations in seconds.

        Returns:
            Sum of all violation window durations in seconds.
        """
        ...

    def __repr__(self) -> str: ...

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
    def daytime(twilight: str = "civil") -> Constraint:
        """
        Create a daytime visibility constraint.

        Args:
            twilight: Twilight definition ("civil", "nautical", "astronomical", "none")

        Returns:
            A new Constraint instance
        """
        ...

    @staticmethod
    def airmass(
        min_airmass: float | None = None, max_airmass: float | None = None
    ) -> Constraint:
        """
        Create an airmass constraint.

        Args:
            min_airmass: Minimum allowed airmass (≥1.0), optional
            max_airmass: Maximum allowed airmass (>0.0)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If airmass values are invalid
        """
        ...

    @staticmethod
    def moon_phase(
        min_illumination: float | None = None,
        max_illumination: float | None = None,
        min_distance: float | None = None,
        max_distance: float | None = None,
        enforce_when_below_horizon: bool = False,
        moon_visibility: str = "full",
    ) -> Constraint:
        """
        Create a Moon phase constraint.

        Args:
            min_illumination: Minimum allowed illumination fraction (0.0-1.0), optional
            max_illumination: Maximum allowed illumination fraction (0.0-1.0)
            min_distance: Minimum allowed Moon distance in degrees from target, optional
            max_distance: Maximum allowed Moon distance in degrees from target, optional
            enforce_when_below_horizon: Whether to enforce constraint when Moon is below horizon
            moon_visibility: Moon visibility requirement ("full" or "partial")

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If illumination or distance values are invalid
        """
        ...

    @staticmethod
    def saa(polygon: list[tuple[float, float]]) -> Constraint:
        """
        Create a South Atlantic Anomaly (SAA) constraint.

        Args:
            polygon: List of (longitude, latitude) pairs defining the region boundary in degrees

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If polygon has fewer than 3 vertices
        """
        ...

    @staticmethod
    def alt_az(
        min_altitude: float | None = None,
        max_altitude: float | None = None,
        min_azimuth: float | None = None,
        max_azimuth: float | None = None,
        polygon: list[tuple[float, float]] | None = None,
    ) -> Constraint:
        """
        Create an altitude/azimuth constraint.

        Args:
            min_altitude: Minimum allowed altitude in degrees (0-90), optional
            max_altitude: Maximum allowed altitude in degrees (0-90), optional
            min_azimuth: Minimum allowed azimuth in degrees (0-360), optional
            max_azimuth: Maximum allowed azimuth in degrees (0-360), optional
            polygon: List of (altitude, azimuth) pairs in degrees defining allowed region, optional.
                     If provided, the target must be inside this polygon to satisfy the constraint.

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range or polygon has fewer than 3 vertices
        """
        ...

    @staticmethod
    def orbit_ram(min_angle: float, max_angle: float | None = None) -> Constraint:
        """
        Create an orbit RAM direction constraint.

        Args:
            min_angle: Minimum allowed angular separation from spacecraft velocity vector in degrees (0-180)
            max_angle: Maximum allowed angular separation from spacecraft velocity vector in degrees (optional)

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range
        """
        ...

    @staticmethod
    def orbit_pole(
        min_angle: float, max_angle: float | None = None, earth_limb_pole: bool = False
    ) -> Constraint:
        """
        Create an orbit pole direction constraint.

        Args:
            min_angle: Minimum allowed angular separation from orbital poles in degrees (0-180)
            max_angle: Maximum allowed angular separation from orbital poles in degrees (optional)
            earth_limb_pole: If True, pole avoidance angle is earth_radius_deg + min_angle - 90

        Returns:
            A new Constraint instance

        Raises:
            ValueError: If angles are out of valid range
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
            {"type": "earth_limb", "min_angle": 10.0}
            {"type": "body", "body": "Mars", "min_angle": 45.0}
            {"type": "daytime", "twilight": "civil"}
            {"type": "airmass", "max_airmass": 2.0}
            {"type": "moon_phase", "max_illumination": 0.5}
            {"type": "saa", "polygon": [[-60, -20], [-30, -60], [0, -60], [0, -20]]}
            {"type": "alt_az", "min_altitude": 10.0}
            {"type": "and", "constraints": [ ... ]}
            {"type": "or", "constraints": [ ... ]}
            {"type": "xor", "constraints": [ ... ]}
            {"type": "not", "constraint": { ... }}
        """
        ...

    def evaluate(
        self,
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> Any:
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

    def in_constraint_batch(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float],
        target_decs: list[float],
        times: datetime | list[datetime] | None = None,
        indices: int | list[int] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """
        Check if targets are in-constraint for multiple RA/Dec positions (vectorized).

        This method is more efficient than calling in_constraint() multiple times
        when you need to check many target positions.

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, OEMEphemeris, or GroundEphemeris
            target_ras: List of target right ascensions in degrees (ICRS/J2000)
            target_decs: List of target declinations in degrees (ICRS/J2000)
            times: Optional specific time(s) to evaluate. Can be a single datetime
                   or list of datetimes. If provided, only these times will be
                   evaluated (must exist in the ephemeris).
            indices: Optional specific time index/indices to evaluate. Can be a
                     single index or list of indices into the ephemeris timestamp array.

        Returns:
            2D numpy boolean array of shape (n_targets, n_times) where True indicates
            constraint violation at that target/time combination.

        Raises:
            ValueError: If target_ras and target_decs have different lengths,
                       or if both times and indices are provided, or if times/indices
                       are not found in the ephemeris
            TypeError: If ephemeris type is not supported

        Note:
            Only one of `times` or `indices` should be provided. If neither is
            provided, all ephemeris times are evaluated.
        """
        ...

    def in_constraint(
        self,
        time: datetime | list[datetime] | npt.NDArray[np.datetime64],
        ephemeris: Ephemeris,
        target_ra: float,
        target_dec: float,
    ) -> bool | list[bool]:
        """
        Check if the target is in-constraint at given time(s).

        Args:
            time: The time(s) to check (must exist in ephemeris timestamps).
                  Can be a single datetime, list of datetimes, or numpy array of datetimes.
            ephemeris: One of TLEEphemeris, SPICEEphemeris, or GroundEphemeris
            target_ra: Target right ascension in degrees (ICRS/J2000)
            target_dec: Target declination in degrees (ICRS/J2000)

        Returns:
            True if constraint is violated at the given time(s). Returns a single bool
            for a single time, or a list of bools for multiple times.

        Raises:
            ValueError: If time is not found in ephemeris timestamps
            TypeError: If ephemeris type is not supported
        """
        ...

    def evaluate_moving_body(
        self,
        ephemeris: Ephemeris,
        target_ras: list[float] | None = None,
        target_decs: list[float] | None = None,
        times: datetime | list[datetime] | None = None,
        body: str | None = None,
        use_horizons: bool = False,
        spice_kernel: str | None = None,
    ) -> MovingBodyResult:
        """
        Evaluate constraint for a moving body (varying RA/Dec over time).

        This method evaluates the constraint for a body whose position changes over time,
        such as a comet, asteroid, or planet. It returns detailed results including
        per-timestamp violation status, visibility windows, and the body's coordinates.

        There are two ways to specify the body's position:
        1. Explicit coordinates: Provide `target_ras`, `target_decs`, and optionally `times`
        2. Body lookup: Provide `body` name/ID and optionally `use_horizons` to query positions

        Args:
            ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
            target_ras: Array of right ascensions in degrees (ICRS/J2000)
            target_decs: Array of declinations in degrees (ICRS/J2000)
            times: Specific times to evaluate (must match ras/decs length)
            body: Body identifier (NAIF ID or name like "Jupiter", "90004910")
            use_horizons: If True, query JPL Horizons for body positions (default: False)
            spice_kernel: Path or URL to a SPICE kernel file for body positions.

        Returns:
            MovingBodyResult containing:
                - violations: List of ConstraintViolation windows
                - all_satisfied: bool indicating if constraint was never violated
                - constraint_name: string name of the constraint
                - timestamp: list of datetime objects
                - ras: list of right ascensions in degrees
                - decs: list of declinations in degrees
                - constraint_array: list of bools (True = violated)
                - visibility: list of VisibilityWindow objects

        Performance:
            The constraint evaluation itself is highly optimized (~0.3 µs per timestamp).
            However, when using `body=`, each call fetches positions from SPICE which can
            take ~80ms for 10,000 timestamps. For best performance when evaluating multiple
            constraints against the same body, pre-fetch the coordinates once::

                # FAST: Pre-fetch coordinates once, reuse for multiple constraints
                skycoord = ephem.get_body("Jupiter")
                ras, decs = list(skycoord.ra.deg), list(skycoord.dec.deg)
                result1 = sun_constraint.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)
                result2 = moon_constraint.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)

                # SLOWER: Each call re-fetches positions from SPICE
                result1 = sun_constraint.evaluate_moving_body(ephem, body="Jupiter")
                result2 = moon_constraint.evaluate_moving_body(ephem, body="Jupiter")

        Raises:
            ValueError: If neither body nor target_ras/target_decs are provided
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

class TLEEphemeris(Ephemeris):
    """Ephemeris calculator using Two-Line Element (TLE) data"""

    def __init__(
        self,
        tle1: str | None = None,
        tle2: str | None = None,
        begin: datetime | None = None,
        end: datetime | None = None,
        step_size: int = 60,
        *,
        polar_motion: bool = False,
        tle: str | TLELike | None = None,
        norad_id: int | None = None,
        norad_name: str | None = None,
        spacetrack_username: str | None = None,
        spacetrack_password: str | None = None,
        epoch_tolerance_days: float | None = None,
    ) -> None:
        """
        Initialize TLE ephemeris from various TLE sources.

        Args:
            tle1: First line of TLE (legacy method, use with tle2)
            tle2: Second line of TLE (legacy method, use with tle1)
            tle: Path to TLE file, URL to download TLE from, or a TLERecord object.
                When passing a TLERecord (or any object with line1, line2, and epoch
                attributes), it will be used directly without fetching.
            norad_id: NORAD catalog ID to fetch TLE. If Space-Track.org credentials
                are available (via parameters, environment variables, or .env file),
                Space-Track.org is tried first with automatic failover to Celestrak.
                Otherwise, Celestrak is used directly.
            norad_name: Satellite name to fetch TLE from Celestrak
            spacetrack_username: Space-Track.org username (or set SPACETRACK_USERNAME env var)
            spacetrack_password: Space-Track.org password (or set SPACETRACK_PASSWORD env var)
            epoch_tolerance_days: For Space-Track cache: how many days TLE epoch can differ
                from target epoch (default: 4.0 days)
            begin: Start time (naive datetime treated as UTC, required)
            end: End time (naive datetime treated as UTC, required)
            step_size: Time step in seconds (default: 60)
            polar_motion: Whether to apply polar motion correction (default: False)

        Note:
            Must provide exactly one of: (tle1, tle2), tle, norad_id, or norad_name.
            begin and end parameters are required.

            When using norad_id with Space-Track.org credentials available:
            - Credentials can be provided via parameters, environment variables
              (SPACETRACK_USERNAME, SPACETRACK_PASSWORD), or a .env file
            - Space-Track will fetch a TLE with epoch closest to the begin time
            - If Space-Track fails, automatically falls back to Celestrak
            - Results are cached; cache is used if TLE epoch is within
              epoch_tolerance_days of the requested begin time

        Example:
            >>> # Using fetch_tle to get TLE, then pass to TLEEphemeris
            >>> from rust_ephem import fetch_tle, TLEEphemeris
            >>> tle_record = fetch_tle(norad_id=25544)
            >>> ephem = TLEEphemeris(tle=tle_record, begin=begin, end=end, step_size=60)
        """
        ...

    @property
    def tle1(self) -> str:
        """First line of the TLE"""
        ...

    @property
    def tle2(self) -> str:
        """Second line of the TLE"""
        ...

    @property
    def begin(self) -> datetime:
        """Start time of the ephemeris"""
        ...

    @property
    def end(self) -> datetime:
        """End time of the ephemeris"""
        ...

    @property
    def step_size(self) -> int:
        """Time step in seconds"""
        ...

    @property
    def polar_motion(self) -> bool:
        """Whether polar motion correction is applied"""
        ...

    @property
    def tle_epoch(self) -> datetime:
        """Epoch timestamp extracted from the TLE (UTC datetime)"""
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
    def latitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic latitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def latitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def latitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic longitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def longitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def height(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic height above the WGS84 ellipsoid as an astropy Quantity array (meters), one per timestamp"""
        ...

    @property
    def height_m(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in meters (one per timestamp)"""
        ...

    @property
    def height_km(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in kilometers (one per timestamp)"""
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
    def sun_pv(self) -> PositionVelocityData:
        """Sun position and velocity in GCRS frame"""
        ...

    @property
    def moon_pv(self) -> PositionVelocityData:
        """Moon position and velocity in GCRS frame"""
        ...

    @property
    def timestamp(self) -> npt.NDArray[np.datetime64]:
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

    @property
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in degrees.

        Convenient property that extracts RA/Dec from the sun SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame relative to the
            spacecraft position.

        Example:
            >>> eph = TLEEphemeris(...)
            >>> sun_coords = eph.sun_ra_dec_deg
            >>> sun_ra = sun_coords[:, 0]   # All RA values
            >>> sun_dec = sun_coords[:, 1]  # All Dec values
            >>> # Find when Sun is above horizon (Dec > 0)
            >>> above_horizon = sun_dec > 0
        """
        ...

    @property
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in degrees.

        Convenient property that extracts RA/Dec from the moon SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame relative to the
            spacecraft position.

        Example:
            >>> eph = TLEEphemeris(...)
            >>> moon_coords = eph.moon_ra_dec_deg
            >>> moon_ra = moon_coords[:, 0]   # All RA values
            >>> moon_dec = moon_coords[:, 1]  # All Dec values
        """
        ...

    @property
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in degrees.

        Convenient property that extracts RA/Dec from the earth SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in degrees [0, 360)
                - Column 1: Declination in degrees [-90, 90]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame and represent
            Earth's position relative to the spacecraft.

        Example:
            >>> eph = TLEEphemeris(...)
            >>> earth_coords = eph.earth_ra_dec_deg
            >>> earth_ra = earth_coords[:, 0]   # All RA values
            >>> earth_dec = earth_coords[:, 1]  # All Dec values
        """
        ...

    @property
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in radians.

        Convenient property that extracts RA/Dec from the sun SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame relative to the
            spacecraft position.

        Example:
            >>> import numpy as np
            >>> eph = TLEEphemeris(...)
            >>> sun_coords = eph.sun_ra_dec_rad
            >>> sun_ra = sun_coords[:, 0]   # All RA values in radians
            >>> sun_dec = sun_coords[:, 1]  # All Dec values in radians
            >>> # Convert to degrees if needed
            >>> sun_ra_deg = np.degrees(sun_ra)
        """
        ...

    @property
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in radians.

        Convenient property that extracts RA/Dec from the moon SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame relative to the
            spacecraft position.

        Example:
            >>> eph = TLEEphemeris(...)
            >>> moon_coords = eph.moon_ra_dec_rad
            >>> moon_ra = moon_coords[:, 0]   # All RA values in radians
            >>> moon_dec = moon_coords[:, 1]  # All Dec values in radians
        """
        ...

    @property
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in radians.

        Convenient property that extracts RA/Dec from the earth SkyCoord and combines
        them into a single Nx2 NumPy array for efficient numerical operations.

        Returns:
            Nx2 NumPy array where:
                - Column 0: Right Ascension in radians [0, 2π)
                - Column 1: Declination in radians [-π/2, π/2]
                - N is the number of timestamps in the ephemeris

        Note:
            This property is cached for performance. The coordinates are in the
            GCRS (Geocentric Celestial Reference System) frame and represent
            Earth's position relative to the spacecraft.

        Example:
            >>> eph = TLEEphemeris(...)
            >>> earth_coords = eph.earth_ra_dec_rad
            >>> earth_ra = earth_coords[:, 0]   # All RA values in radians
            >>> earth_dec = earth_coords[:, 1]  # All Dec values in radians
        """
        ...

    def get_body_pv(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> PositionVelocityData:
        """
        Get position and velocity of a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            Position and velocity data for the requested body
        """
        ...

    def get_body(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> Any:  # Returns astropy.coordinates.SkyCoord
        """
        Get SkyCoord for a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

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

    def moon_illumination(self, time_indices: list[int] | None = None) -> list[float]:
        """
        Calculate Moon illumination fraction for all ephemeris times.

        Returns the fraction of the Moon's illuminated surface as seen from the
        spacecraft observer (0.0 = new moon, 1.0 = full moon).

        Args:
            time_indices: Optional indices into ephemeris times (default: all times)

        Returns:
            List of Moon illumination fractions
        """
        ...

    @property
    def obsgeoloc(
        self,
    ) -> npt.NDArray[np.float64]:  # Returns NumPy array or None if unavailable
        """
        Observer geocentric location (GCRS position).

        Returns position in km, compatible with astropy's GCRS frame obsgeoloc parameter.
        """
        ...

    @property
    def obsgeovel(self) -> npt.NDArray[np.float64]:  # Returns astropy quantity array
        """Observer geocentric velocity (alias for GCRS velocity)"""
        ...

    def radec_to_altaz(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Topocentric altitude/azimuth for given RA/Dec (deg) at selected times."""
        ...

    def calculate_airmass(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> list[float]:
        """Calculate airmass for given RA/Dec (deg) at selected times.

        Returns airmass values (1.0 at zenith, ~2.0 at 30° altitude, infinity below horizon).
        Accounts for observer height using atmospheric scale height correction.
        """
        ...

class SPICEEphemeris(Ephemeris):
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
    def spk_path(self) -> str:
        """Path to SPICE SPK kernel file"""
        ...

    @property
    def naif_id(self) -> int:
        """NAIF ID of the target body"""
        ...

    @property
    def center_id(self) -> int:
        """NAIF ID of the observer/center body"""
        ...

    @property
    def begin(self) -> datetime:
        """Start time of the ephemeris"""
        ...

    @property
    def end(self) -> datetime:
        """End time of the ephemeris"""
        ...

    @property
    def step_size(self) -> int:
        """Time step in seconds"""
        ...

    @property
    def polar_motion(self) -> bool:
        """Whether polar motion correction is applied"""
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
    def latitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic latitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def latitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def latitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic longitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def longitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def height(self) -> Any:  # Returns astropy.units.Quantity
        """Height above the WGS84 ellipsoid as an astropy Quantity array (meters), one per timestamp"""
        ...

    @property
    def height_m(self) -> npt.NDArray[np.float64]:
        """Height above the WGS84 ellipsoid as a raw NumPy array in meters (one per timestamp)"""
        ...

    @property
    def height_km(self) -> npt.NDArray[np.float64]:
        """Height above the WGS84 ellipsoid as a raw NumPy array in kilometers (one per timestamp)"""
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
    def timestamp(self) -> npt.NDArray[np.datetime64]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
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

    def radec_to_altaz(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Topocentric altitude/azimuth for given RA/Dec (deg) at selected times."""
        ...

    def calculate_airmass(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> list[float]:
        """Calculate airmass for given RA/Dec (deg) at selected times.

        Returns airmass values (1.0 at zenith, ~2.0 at 30° altitude, infinity below horizon).
        Accounts for observer height using atmospheric scale height correction.
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

    @property
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
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

    def moon_illumination(self, time_indices: list[int] | None = None) -> list[float]:
        """
        Calculate Moon illumination fraction for all ephemeris times.

        Returns the fraction of the Moon's illuminated surface as seen from the
        spacecraft observer (0.0 = new moon, 1.0 = full moon).

        Args:
            time_indices: Optional indices into ephemeris times (default: all times)

        Returns:
            List of Moon illumination fractions
        """
        ...

    def get_body_pv(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> PositionVelocityData:
        """
        Get position and velocity of a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            Position and velocity data for the requested body
        """
        ...

    def get_body(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> Any:  # Returns astropy.coordinates.SkyCoord
        """
        Get SkyCoord for a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            astropy.coordinates.SkyCoord object
        """
        ...

class OEMEphemeris(Ephemeris):
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
    def oem_path(self) -> str:
        """Path to the CCSDS OEM file"""
        ...

    @property
    def begin(self) -> datetime:
        """Start time of ephemeris"""
        ...

    @property
    def end(self) -> datetime:
        """End time of ephemeris"""
        ...

    @property
    def step_size(self) -> int:
        """Time step in seconds"""
        ...

    @property
    def polar_motion(self) -> bool:
        """Whether polar motion correction is applied"""
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
    def latitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic latitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def latitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def latitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic longitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def longitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def height(self) -> Any:  # Returns astropy.units.Quantity
        """Height above the WGS84 ellipsoid as an astropy Quantity array (meters), one per timestamp"""
        ...

    @property
    def height_m(self) -> npt.NDArray[np.float64]:
        """Height above the WGS84 ellipsoid as a raw NumPy array in meters (one per timestamp)"""
        ...

    @property
    def height_km(self) -> npt.NDArray[np.float64]:
        """Height above the WGS84 ellipsoid as a raw NumPy array in kilometers (one per timestamp)"""
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
    def timestamp(self) -> npt.NDArray[np.datetime64]:
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

    def radec_to_altaz(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Topocentric altitude/azimuth for given RA/Dec (deg) at selected times."""
        ...

    def calculate_airmass(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> list[float]:
        """Calculate airmass for given RA/Dec (deg) at selected times.

        Returns airmass values (1.0 at zenith, ~2.0 at 30° altitude, infinity below horizon).
        Accounts for observer height using atmospheric scale height correction.
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

    @property
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
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

    def moon_illumination(self, time_indices: list[int] | None = None) -> list[float]:
        """
        Calculate Moon illumination fraction for all ephemeris times.

        Returns the fraction of the Moon's illuminated surface as seen from the
        spacecraft observer (0.0 = new moon, 1.0 = full moon).

        Args:
            time_indices: Optional indices into ephemeris times (default: all times)

        Returns:
            List of Moon illumination fractions
        """
        ...

    def get_body_pv(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> PositionVelocityData:
        """
        Get position and velocity of a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            Position and velocity data for the requested body
        """
        ...

    def get_body(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> Any:  # Returns astropy.coordinates.SkyCoord
        """
        Get SkyCoord for a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            astropy.coordinates.SkyCoord object
        """
        ...

class GroundEphemeris(Ephemeris):
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
    def input_latitude(self) -> float:
        """Input geodetic latitude in degrees"""
        ...

    @property
    def input_longitude(self) -> float:
        """Input geodetic longitude in degrees"""
        ...

    @property
    def input_height(self) -> float:
        """Input altitude in meters above WGS84 ellipsoid"""
        ...

    @property
    def begin(self) -> datetime:
        """Start time of ephemeris"""
        ...

    @property
    def end(self) -> datetime:
        """End time of ephemeris"""
        ...

    @property
    def step_size(self) -> int:
        """Time step in seconds"""
        ...

    @property
    def polar_motion(self) -> bool:
        """Whether polar motion correction is applied"""
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

    def radec_to_altaz(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Topocentric altitude/azimuth for given RA/Dec (deg) at selected times."""
        ...

    def calculate_airmass(
        self,
        ra_deg: float,
        dec_deg: float,
        time_indices: list[int] | None = None,
    ) -> list[float]:
        """Calculate airmass for given RA/Dec (deg) at selected times.

        Returns airmass values (1.0 at zenith, ~2.0 at 30° altitude, infinity below horizon).
        Accounts for observer height using atmospheric scale height correction.
        """
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
    def timestamp(self) -> npt.NDArray[np.datetime64]:
        """
        Array of timestamps for the ephemeris.

        Returns a NumPy array of datetime objects (not a list) for efficient indexing.
        This property is cached for performance - repeated access is ~90x faster.
        """
        ...

    @property
    def obsgeoloc(self) -> npt.NDArray[np.float64]:  # Returns astropy quantity array
        """Observatory geocentric location for astropy"""
        ...

    @property
    def obsgeovel(self) -> npt.NDArray[np.float64]:  # Returns astropy quantity array
        """Observatory geocentric velocity for astropy"""
        ...

    @property
    def latitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic latitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def latitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def latitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic latitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic longitude as an astropy Quantity array (degrees), one per timestamp"""
        ...

    @property
    def longitude_deg(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in degrees as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def longitude_rad(self) -> npt.NDArray[np.float64]:
        """Geodetic longitude in radians as a raw NumPy array (one per timestamp)"""
        ...

    @property
    def height(self) -> Any:  # Returns astropy.units.Quantity
        """Geodetic height above the WGS84 ellipsoid as an astropy Quantity array (meters), one per timestamp"""
        ...

    @property
    def height_m(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in meters (one per timestamp)"""
        ...

    @property
    def height_km(self) -> npt.NDArray[np.float64]:
        """Geodetic height above the WGS84 ellipsoid as a raw NumPy array in kilometers (one per timestamp)"""
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

    @property
    def sun_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_deg(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in degrees.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def sun_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Sun in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def moon_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Moon in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
        This property is cached for performance.
        """
        ...

    @property
    def earth_ra_dec_rad(self) -> npt.NDArray[np.float64]:
        """
        Right Ascension and Declination of the Earth in radians.

        Returns an Nx2 NumPy array where column 0 is RA and column 1 is Dec.
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
            >>> eph = GroundEphemeris(...)
            >>> target_time = datetime(2024, 1, 15, 12, 0, 0)
            >>> idx = eph.index(target_time)
            >>> sun_position = eph.sun_pv.position[idx]
        """
        ...

    def moon_illumination(self, time_indices: list[int] | None = None) -> list[float]:
        """
        Calculate Moon illumination fraction for all ephemeris times.

        Returns the fraction of the Moon's illuminated surface as seen from the
        spacecraft observer (0.0 = new moon, 1.0 = full moon).

        Args:
            time_indices: Optional indices into ephemeris times (default: all times)

        Returns:
            List of Moon illumination fractions
        """
        ...

    def get_body_pv(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> PositionVelocityData:
        """
        Get position and velocity of a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            Position and velocity data for the requested body
        """
        ...

    def get_body(
        self, body: str, spice_kernel: str | None = ..., use_horizons: bool = ...
    ) -> Any:  # Returns astropy.coordinates.SkyCoord
        """
        Get SkyCoord for a celestial body.

        Args:
            body: Name of the body (e.g., 'sun', 'moon', 'earth')
            spice_kernel: Optional path to SPICE kernel
            use_horizons: If True, fall back to JPL Horizons API when SPICE fails

        Returns:
            astropy.coordinates.SkyCoord object
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

def fetch_tle(
    *,
    tle: str | None = None,
    norad_id: int | None = None,
    norad_name: str | None = None,
    epoch: datetime | None = None,
    spacetrack_username: str | None = None,
    spacetrack_password: str | None = None,
    epoch_tolerance_days: float | None = None,
    enforce_source: str | None = None,
) -> dict[str, Any]:
    """
    Fetch a TLE from various sources (file, URL, Celestrak, Space-Track.org).

    This is the low-level Rust function. For a higher-level API with Pydantic
    models, use `rust_ephem.fetch_tle()` which returns a `TLERecord` object.

    Args:
        tle: Path to TLE file or URL to download TLE from
        norad_id: NORAD catalog ID to fetch TLE. If Space-Track credentials
            are available, Space-Track is tried first with failover to Celestrak.
        norad_name: Satellite name to fetch TLE from Celestrak
        epoch: Target epoch for Space-Track queries. If not specified,
            current time is used. Space-Track will fetch the TLE with epoch
            closest to this time.
        spacetrack_username: Space-Track.org username (or use SPACETRACK_USERNAME env var)
        spacetrack_password: Space-Track.org password (or use SPACETRACK_PASSWORD env var)
        epoch_tolerance_days: For Space-Track cache: how many days TLE epoch can
            differ from target epoch (default: 4.0 days)
        enforce_source: Enforce use of specific source without failover.
            Must be "celestrak", "spacetrack", or None (default behavior with failover)

    Returns:
        Dict with keys: line1, line2, name (optional), epoch (datetime), source

    Raises:
        ValueError: If no valid TLE source is specified or fetching fails
    """
    ...
