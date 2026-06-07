"""Type stubs for the TLE module"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

WGS72_EARTH_MU_M3_S2: float

def true_anomaly_from_mean_anomaly(
    mean_anomaly_deg: float, eccentricity: float
) -> float: ...

class TLERecord(BaseModel):
    """
    A Two-Line Element (TLE) record with optional metadata.

    This model can be passed directly to TLEEphemeris via the `tle` parameter.
    It supports JSON serialization for storage and transmission.

    Attributes:
        line1: First line of the TLE (starts with '1')
        line2: Second line of the TLE (starts with '2')
        name: Optional satellite name (from 3-line TLE format)
        epoch: TLE epoch timestamp (extracted from line1)
        source: Source of the TLE data (e.g., 'celestrak', 'spacetrack', 'file', 'url')
    """

    line1: str
    line2: str
    name: str | None = None
    epoch: datetime = ...
    source: str | None = None

    @property
    def norad_id(self) -> int:
        """Extract NORAD catalog ID from line1."""
        ...

    @property
    def classification(self) -> str:
        """Extract classification from line1 (U=unclassified, C=classified, S=secret)."""
        ...

    @property
    def international_designator(self) -> str:
        """Extract international designator from line1."""
        ...

    @property
    def inclination_deg(self) -> float:
        """Extract inclination (deg) from line2."""
        ...

    @property
    def right_ascension_deg(self) -> float:
        """Extract RAAN (deg) from line2."""
        ...

    @property
    def eccentricity(self) -> float:
        """Extract eccentricity from line2."""
        ...

    @property
    def arg_periapsis_deg(self) -> float:
        """Extract argument of periapsis (deg) from line2."""
        ...

    @property
    def mean_anomaly_deg(self) -> float:
        """Extract mean anomaly (deg) from line2."""
        ...

    @property
    def mean_motion_rev_per_day(self) -> float:
        """Extract mean motion (rev/day) from line2."""
        ...

    @property
    def true_anomaly_deg(self) -> float:
        """Derived true anomaly (deg) from mean anomaly and eccentricity."""
        ...

    @property
    def semimajor_axis_m(self) -> float:
        """Derived semimajor axis (m) using WGS72 Earth's gravitational parameter."""
        ...

    def to_tle_string(self) -> str:
        """
        Convert to a TLE string format.

        Returns:
            2-line or 3-line TLE string depending on whether name is set.
        """
        ...

    def classical_elements(self, mu_m3_s2: float = ...) -> dict[str, Any]:
        """Return TLE mean classical elements at the TLE epoch."""
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
) -> TLERecord:
    """
    Fetch a TLE from various sources.

    This function provides a unified interface for retrieving TLE data from:
    - Local files (2-line or 3-line TLE format)
    - URLs (with automatic caching)
    - Celestrak (by NORAD ID or satellite name)
    - Space-Track.org (by NORAD ID, requires credentials)

    When Space-Track.org credentials are available (via parameters, environment
    variables, or .env file), NORAD ID queries will try Space-Track first with
    automatic failover to Celestrak.

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

    Returns:
        TLERecord containing the TLE data and metadata

    Raises:
        ValueError: If no valid TLE source is specified or fetching fails

    Examples:
        >>> # Fetch from Celestrak by NORAD ID
        >>> tle = fetch_tle(norad_id=25544)  # ISS
        >>> print(tle.name)

        >>> # Fetch from file
        >>> tle = fetch_tle(tle="path/to/satellite.tle")

        >>> # Fetch from Space-Track with explicit credentials
        >>> tle = fetch_tle(
        ...     norad_id=25544,
        ...     spacetrack_username="user",
        ...     spacetrack_password="pass",
        ...     epoch=datetime(2020, 1, 1, tzinfo=timezone.utc)
        ... )
    """
    ...
