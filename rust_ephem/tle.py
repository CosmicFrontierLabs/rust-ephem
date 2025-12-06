"""
TLE (Two-Line Element) data models and fetching utilities.

This module provides Pydantic models for TLE data and a fetch_tle function
that can retrieve TLEs from various sources (files, URLs, Celestrak, Space-Track.org).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from ._rust_ephem import fetch_tle as _fetch_tle


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

    line1: str = Field(..., description="First line of the TLE")
    line2: str = Field(..., description="Second line of the TLE")
    name: str | None = Field(None, description="Optional satellite name")
    epoch: datetime = Field(..., description="TLE epoch timestamp")
    source: str | None = Field(None, description="Source of the TLE data")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def norad_id(self) -> int:
        """Extract NORAD catalog ID from line1."""
        return int(self.line1[2:7].strip())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classification(self) -> str:
        """Extract classification from line1 (U=unclassified, C=classified, S=secret)."""
        return self.line1[7]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def international_designator(self) -> str:
        """Extract international designator from line1."""
        return self.line1[9:17].strip()

    def to_tle_string(self) -> str:
        """
        Convert to a TLE string format.

        Returns:
            2-line or 3-line TLE string depending on whether name is set.
        """
        if self.name:
            return f"{self.name}\n{self.line1}\n{self.line2}"
        return f"{self.line1}\n{self.line2}"

    model_config = {"frozen": True}


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
    # Call the Rust function
    result = _fetch_tle(
        tle=tle,
        norad_id=norad_id,
        norad_name=norad_name,
        epoch=epoch,
        spacetrack_username=spacetrack_username,
        spacetrack_password=spacetrack_password,
        epoch_tolerance_days=epoch_tolerance_days,
    )

    # Convert the result dict to TLERecord
    return TLERecord(
        line1=result["line1"],
        line2=result["line2"],
        name=result.get("name"),
        epoch=result["epoch"],
        source=result.get("source"),
    )
