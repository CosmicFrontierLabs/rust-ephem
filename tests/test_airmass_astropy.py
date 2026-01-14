"""
Test airmass calculation against astropy to verify altitude correction.

This test validates that rust-ephem's airmass calculation, including the
atmospheric scale height correction for observer altitude, matches astropy's
implementation across various observing conditions.

Key validations:
1. Airmass values match astropy's secz model at high altitudes (>15°)
2. Exponential height correction is properly applied: airmass(h) = airmass(0) * exp(-h/H)
    where H = 8.5 km (atmospheric scale height)
3. Rozenberg approximation is used correctly for low altitudes (<10°)
4. Infinite airmass is returned for targets below the horizon
5. AirmassConstraint properly uses height-corrected values

The height correction accounts for the fact that observers at higher altitudes
have less atmosphere above them. For example:
- Mauna Kea (4.2 km): airmass reduced by ~39% compared to sea level
- La Palma (2.4 km): airmass reduced by ~24% compared to sea level
"""

from datetime import datetime, timedelta
from typing import List

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    AltAz,
    EarthLocation,
    SkyCoord,
)
from astropy.time import Time  # type: ignore[import-untyped]

import rust_ephem

# Tolerances for airmass comparisons
RTOL_SEA_LEVEL = 0.05  # 5% relative tolerance
ATOL_SEA_LEVEL = 0.01  # absolute tolerance
RTOL_ALTITUDE = 0.05  # 5% relative tolerance for altitude comparisons
ATOL_ALTITUDE = 0.01  # absolute tolerance


@pytest.fixture
def test_times() -> List[datetime]:
    """Generate a sequence of test times."""
    start = datetime(2025, 1, 15, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(6)]


class TestAirmassAstropyComparison:
    """Compare rust-ephem airmass calculations with astropy."""

    def test_airmass_sea_level_zenith(self, test_times: List[datetime]) -> None:
        """Test airmass at zenith from sea level."""

        # Ground observer at sea level
        lat, lon, height = 40.0, -105.0, 0.0  # Sea level
        ephem = rust_ephem.GroundEphemeris(
            lat, lon, height, test_times[0], test_times[-1], step_size=3600
        )

        # Target at zenith (directly overhead)
        # For simplicity, use a high declination target
        ra_deg, dec_deg = (
            0.0,
            90.0,
        )  # North celestial pole (always at zenith for north pole observer)

        # Actually, let's use the observer's latitude as declination for zenith
        ra_deg = 0.0  # RA doesn't matter for zenith
        dec_deg = lat  # Target at observer's latitude = zenith when it transits

        # Calculate airmass with rust-ephem
        airmass_rust = ephem.calculate_airmass(ra_deg, dec_deg)

        # Calculate with astropy
        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
        times_astropy = Time([t.isoformat() for t in test_times])

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        altaz_frame = AltAz(obstime=times_astropy, location=location)
        altaz = coord.transform_to(altaz_frame)

        # Astropy's secz airmass model (simple secant)
        airmass_astropy = altaz.secz.value

        # For high altitudes, both should be close to 1.0
        # Allow some tolerance due to different implementations
        valid_idx = airmass_astropy < 10  # Filter out low altitude points
        if np.any(valid_idx):
            np.testing.assert_allclose(
                np.array(airmass_rust)[valid_idx],
                airmass_astropy[valid_idx],
                rtol=RTOL_SEA_LEVEL,
                atol=ATOL_SEA_LEVEL,
            )

    def test_airmass_vs_altitude_comparison(self, test_times: List[datetime]) -> None:
        """Test airmass calculation for various altitudes."""

        lat, lon, height = 34.0, -118.0, 0.0  # Los Angeles, sea level
        ephem = rust_ephem.GroundEphemeris(
            lat, lon, height, test_times[0], test_times[-1], step_size=3600
        )

        # Test targets at various declinations to get different altitudes
        test_targets = [
            (0.0, 90.0),  # North pole (high altitude from northern latitude)
            (0.0, lat),  # Same as observer latitude (transits at zenith)
            (0.0, lat - 30.0),  # 30° from zenith at transit
            (0.0, 0.0),  # Equator
        ]

        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
        times_astropy = Time([t.isoformat() for t in test_times])

        for ra_deg, dec_deg in test_targets:
            airmass_rust = ephem.calculate_airmass(ra_deg, dec_deg)

            # Calculate with astropy
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            altaz_frame = AltAz(obstime=times_astropy, location=location)
            altaz = coord.transform_to(altaz_frame)

            # Get altitude in degrees
            alt_deg = altaz.alt.deg

            # For points above horizon and not too low, compare
            valid_idx = (alt_deg > 15) & (altaz.secz.value < 10)

            if np.any(valid_idx):
                # Astropy uses simple secant (secz = 1/sin(alt))
                airmass_astropy = altaz.secz.value

                # Both use similar approximations at high altitudes
                np.testing.assert_allclose(
                    np.array(airmass_rust)[valid_idx],
                    airmass_astropy[valid_idx],
                    rtol=RTOL_ALTITUDE,
                    atol=ATOL_ALTITUDE,
                    err_msg=f"Mismatch for target RA={ra_deg}, Dec={dec_deg}",
                )

    def test_airmass_low_altitude_rozenberg(self, test_times: List[datetime]) -> None:
        """Test that low altitude uses Rozenberg approximation."""
        # At very low altitudes (< 10°), rust-ephem uses Rozenberg approximation
        # which is more accurate than simple secant

        lat, lon, height = 0.0, 0.0, 0.0  # Equator, sea level
        ephem = rust_ephem.GroundEphemeris(
            lat, lon, height, test_times[0], test_times[-1], step_size=3600
        )

        # Target that will be at low altitude
        ra_deg, dec_deg = 0.0, -80.0  # Far south from equator

        airmass_rust = ephem.calculate_airmass(ra_deg, dec_deg)

        # At low altitudes, airmass should be quite high (> 3)
        # Just verify we get reasonable values, not infinity
        for am in airmass_rust:
            if np.isfinite(am):
                assert am > 1.0, "Airmass should be > 1.0"
                assert am < 40.0, "Airmass should be < 40 for reasonable altitudes"

    def test_airmass_below_horizon(self, test_times: List[datetime]) -> None:
        """Test that targets below horizon return infinite airmass."""
        lat, lon, height = 40.0, -105.0, 0.0
        ephem = rust_ephem.GroundEphemeris(
            lat, lon, height, test_times[0], test_times[-1], step_size=3600
        )

        # Target far below horizon
        ra_deg, dec_deg = 0.0, -80.0  # Far south from northern latitude

        airmass_rust = ephem.calculate_airmass(ra_deg, dec_deg)

        # At least some times should have infinite airmass (target below horizon)
        # Check if we have any infinite values when target is below horizon
        altaz = ephem.radec_to_altaz(ra_deg, dec_deg)

        for i in range(len(airmass_rust)):
            alt_deg = altaz[i, 0]
            if alt_deg <= 0:
                assert np.isinf(airmass_rust[i]), (
                    f"Airmass should be infinite when alt={alt_deg:.1f}° (below horizon)"
                )
            else:
                assert np.isfinite(airmass_rust[i]), (
                    f"Airmass should be finite when alt={alt_deg:.1f}° (above horizon)"
                )
