"""
TLE (Two-Line Element) data models and fetching utilities.

This module provides Pydantic models for TLE data and a fetch_tle function
that can retrieve TLEs from various sources (files, URLs, Celestrak, Space-Track.org).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, cast

from pydantic import BaseModel, Field, computed_field, model_validator

from ._rust_ephem import fetch_tle as _fetch_tle

_SECONDS_PER_DAY = 86400.0
WGS72_EARTH_MU_M3_S2 = 398600.8e9


def _normalize_degrees(value: float) -> float:
    return value % 360.0


def _solve_eccentric_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(
            f"Only elliptical TLE eccentricities are supported; got {eccentricity}"
        )
    eccentric_anomaly = mean_anomaly_rad if eccentricity < 0.8 else math.pi
    for _ in range(50):
        denominator = 1.0 - eccentricity * math.cos(eccentric_anomaly)
        delta = (
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - mean_anomaly_rad
        ) / denominator
        eccentric_anomaly -= delta
        if abs(delta) < 1e-14:
            break
    return eccentric_anomaly


def true_anomaly_from_mean_anomaly(
    mean_anomaly_deg: float, eccentricity: float
) -> float:
    """Convert mean anomaly to true anomaly for an elliptical orbit."""
    mean_anomaly_rad = math.radians(_normalize_degrees(mean_anomaly_deg))
    eccentric_anomaly = _solve_eccentric_anomaly(mean_anomaly_rad, eccentricity)
    true_anomaly_rad = math.atan2(
        math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly),
        math.cos(eccentric_anomaly) - eccentricity,
    )
    return _normalize_degrees(math.degrees(true_anomaly_rad))


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

    line1: str = Field(
        ..., description="First line of the TLE", min_length=69, max_length=69
    )
    line2: str = Field(
        ..., description="Second line of the TLE", min_length=69, max_length=69
    )
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inclination_deg(self) -> float:
        """Extract inclination (deg) from line2."""
        return float(self.line2[8:16])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def right_ascension_deg(self) -> float:
        """Extract RAAN (deg) from line2."""
        return float(self.line2[17:25])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eccentricity(self) -> float:
        """Extract eccentricity from line2."""
        return float("0." + self.line2[26:33])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def arg_periapsis_deg(self) -> float:
        """Extract argument of periapsis (deg) from line2."""
        return float(self.line2[34:42])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_anomaly_deg(self) -> float:
        """Extract mean anomaly (deg) from line2."""
        return float(self.line2[43:51])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_motion_rev_per_day(self) -> float:
        """Extract mean motion (rev/day) from line2."""
        return float(self.line2[52:63])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def true_anomaly_deg(self) -> float:
        """Derived true anomaly (deg) from mean anomaly and eccentricity."""
        return true_anomaly_from_mean_anomaly(self.mean_anomaly_deg, self.eccentricity)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def semimajor_axis_m(self) -> float:
        """Derived semimajor axis (m) using WGS72 Earth's gravitational parameter."""
        mean_motion_rev_per_day = cast(float, self.mean_motion_rev_per_day)
        mean_motion_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / _SECONDS_PER_DAY
        return (WGS72_EARTH_MU_M3_S2 / mean_motion_rad_s**2) ** (1.0 / 3.0)

    def to_tle_string(self) -> str:
        """
        Convert to a TLE string format.

        Returns:
            2-line or 3-line TLE string depending on whether name is set.
        """
        if self.name:
            return f"{self.name}\n{self.line1}\n{self.line2}"
        return f"{self.line1}\n{self.line2}"

    def classical_elements(
        self, mu_m3_s2: float = WGS72_EARTH_MU_M3_S2
    ) -> dict[str, Any]:
        """Return TLE mean classical elements at the TLE epoch.

        Extracts inclination, RAAN, eccentricity, argument of perigee, mean
        anomaly, and mean motion from TLE line 2 using fixed-width column
        positions. Semimajor axis and true anomaly are derived from those
        mean elements.

        These are TLE mean elements at the TLE epoch, not propagated
        osculating elements.
        """
        eccentricity = self.eccentricity
        mean_anomaly_deg = self.mean_anomaly_deg
        mean_motion_rev_per_day = self.mean_motion_rev_per_day
        mean_motion_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / _SECONDS_PER_DAY
        semimajor_axis_m = (mu_m3_s2 / mean_motion_rad_s**2) ** (1.0 / 3.0)
        return {
            "SemimajorAxis_m": semimajor_axis_m,
            "Eccentricity": eccentricity,
            "Inclination_deg": self.inclination_deg,
            "RightAscension_deg": self.right_ascension_deg,
            "ArgPeriapsis_deg": self.arg_periapsis_deg,
            "TrueAnomaly_deg": true_anomaly_from_mean_anomaly(
                mean_anomaly_deg, eccentricity
            ),
            "MeanAnomaly_deg": mean_anomaly_deg,
            "MeanMotion_rev_per_day": mean_motion_rev_per_day,
            "GravitationalParameter_m3_s2": mu_m3_s2,
        }

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
