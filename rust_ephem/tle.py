"""
TLE (Two-Line Element) data models and fetching utilities.

This module provides Pydantic models for TLE data and a fetch_tle function
that can retrieve TLEs from various sources (files, URLs, Celestrak, Space-Track.org).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

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

    @model_validator(mode="before")
    def _validate_tle_lines(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that line1 and line2 conform to TLE format."""
        line1 = values.get("line1", "")
        line2 = values.get("line2", "")

        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            raise ValueError(
                "Invalid TLE format: line1 must start with '1 ' and line2 with '2 '"
            )

        if not values.get("epoch") and len(line1) >= 19:
            # Extract epoch from line1 if not provided
            epoch_str = line1[18:32].strip()
            try:
                epoch_year = int(epoch_str[:2])
                epoch_day = float(epoch_str[2:])
                epoch_year += 2000 if epoch_year < 57 else 1900  # TLE epoch year cutoff
                epoch_datetime = datetime(epoch_year, 1, 1) + timedelta(
                    days=epoch_day - 1
                )
                values["epoch"] = epoch_datetime
            except Exception as exc:
                raise ValueError(f"Failed to parse epoch from line1: {exc}") from exc

        return values

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
    enforce_source: str | None = None,
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
        enforce_source: Enforce use of specific source without failover.
            Must be "celestrak", "spacetrack", or None (default behavior with failover)

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
    try:
        result = _fetch_tle(
            tle=tle,
            norad_id=norad_id,
            norad_name=norad_name,
            epoch=epoch,
            spacetrack_username=spacetrack_username,
            spacetrack_password=spacetrack_password,
            epoch_tolerance_days=epoch_tolerance_days,
            enforce_source=enforce_source,
        )
    except ValueError as exc:
        # Surface a clearer message when the upstream source returned no usable TLE
        message = str(exc)
        parse_failure = "Invalid TLE" in message
        timeout_failure = "timeout" in message.lower()

        parts = []
        if norad_id is not None:
            parts.append(f"NORAD ID {norad_id}")
        if norad_name:
            parts.append(f"satellite name '{norad_name}'")
        if tle:
            parts.append(f"source '{tle}'")
        context = ", ".join(parts) if parts else "the requested source"

        if timeout_failure:
            hint = (
                f"TLE fetch timed out while retrieving {context}. "
                "The upstream service (Space-Track.org or Celestrak) may be slow or "
                "temporarily unavailable. Try again later, or check your network connection."
            )
            raise ValueError(hint) from exc

        if parse_failure:
            hint = (
                "No TLE data was returned from "
                f"{context}; the response was not in TLE format. "
                "The satellite may not exist, may not have public TLE data, or the upstream "
                "service may be temporarily unavailable."
            )
            raise ValueError(hint) from exc

        raise

    # Convert the result dict to TLERecord
    return TLERecord(
        line1=result["line1"],
        line2=result["line2"],
        name=result.get("name"),
        epoch=result["epoch"],
        source=result.get("source"),
    )
