#!/usr/bin/env python3
"""
Integration test for earth property.

This test verifies that:
1. earth property exists and works
2. The returned SkyCoord objects have GCRS frame
3. The positions and velocities are the negative of spacecraft GCRS data

Requirements:
    pip install astropy numpy

Build and install the Rust module first:
    maturin build --release
    pip install target/wheels/*.whl
"""

import numpy as np
import pytest

from rust_ephem._rust_ephem import TLEEphemeris

try:
    from astropy.coordinates import GCRS, SkyCoord  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Position/velocity tolerance
TOLERANCE = 1e-9


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestEarthSkyCoordBasic:
    """Test basic properties of earth SkyCoord."""

    def test_earth_returns_skycoord(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that earth property returns a SkyCoord object."""
        earth_skycoord = multi_point_ephem.earth
        assert isinstance(earth_skycoord, SkyCoord)

    def test_earth_correct_length(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that earth has correct number of points."""
        earth_skycoord = multi_point_ephem.earth
        expected_len: int = len(multi_point_ephem.timestamp)
        assert len(earth_skycoord) == expected_len

    def test_earth_has_gcrs_frame(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that earth uses GCRS frame."""
        earth_skycoord = multi_point_ephem.earth
        assert isinstance(earth_skycoord.frame, GCRS)

    def test_earth_has_velocity(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that earth includes velocity information."""
        earth_skycoord = multi_point_ephem.earth
        assert earth_skycoord.velocity is not None


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestEarthPositionAccuracy:
    """Test Earth position accuracy for all time points."""

    @pytest.mark.parametrize("i", range(31))
    def test_earth_position_is_negative_of_spacecraft(
        self, multi_point_ephem: TLEEphemeris, i: int
    ) -> None:
        """Test that Earth position is negative of spacecraft position at time point i."""
        earth_skycoord = multi_point_ephem.earth
        expected_pos = -multi_point_ephem.gcrs_pv.position[i]
        actual_pos = earth_skycoord[i].cartesian.xyz.to("km").value
        assert np.allclose(expected_pos, actual_pos, rtol=TOLERANCE)


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestEarthVelocityAccuracy:
    """Test Earth velocity accuracy for all time points."""

    @pytest.mark.parametrize("i", range(31))
    def test_earth_velocity_is_negative_of_spacecraft(
        self, multi_point_ephem: TLEEphemeris, i: int
    ) -> None:
        """Test that Earth velocity is negative of spacecraft velocity at time point i."""
        earth_skycoord = multi_point_ephem.earth
        expected_vel = -multi_point_ephem.gcrs_pv.velocity[i]
        actual_vel = earth_skycoord[i].velocity.d_xyz.to("km/s").value
        assert np.allclose(expected_vel, actual_vel, rtol=TOLERANCE)


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestEarthObsGeo:
    """Test obsgeoloc and obsgeovel properties of earth."""

    def test_obsgeoloc_is_set(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that obsgeoloc is set in earth frame."""
        earth_skycoord = multi_point_ephem.earth
        assert earth_skycoord.frame.obsgeoloc is not None

    def test_obsgeovel_is_set(self, multi_point_ephem: TLEEphemeris) -> None:
        """Test that obsgeovel is set in earth frame."""
        earth_skycoord = multi_point_ephem.earth
        assert earth_skycoord.frame.obsgeovel is not None

    @pytest.mark.parametrize("i", range(31))
    def test_obsgeoloc_matches_spacecraft_position(
        self, multi_point_ephem: TLEEphemeris, i: int
    ) -> None:
        """Test that obsgeoloc matches spacecraft position at time point i."""
        earth_skycoord = multi_point_ephem.earth
        expected_pos = multi_point_ephem.gcrs_pv.position[i]
        actual_pos = np.array(
            [
                earth_skycoord.frame.obsgeoloc[i].x.to("km").value,
                earth_skycoord.frame.obsgeoloc[i].y.to("km").value,
                earth_skycoord.frame.obsgeoloc[i].z.to("km").value,
            ]
        )
        assert np.allclose(expected_pos, actual_pos, rtol=TOLERANCE)

    @pytest.mark.parametrize("i", range(31))
    def test_obsgeovel_matches_spacecraft_velocity(
        self, multi_point_ephem: TLEEphemeris, i: int
    ) -> None:
        """Test that obsgeovel matches spacecraft velocity at time point i."""
        earth_skycoord = multi_point_ephem.earth
        expected_vel = multi_point_ephem.gcrs_pv.velocity[i]
        actual_vel = np.array(
            [
                earth_skycoord.frame.obsgeovel[i].x.to("km/s").value,
                earth_skycoord.frame.obsgeovel[i].y.to("km/s").value,
                earth_skycoord.frame.obsgeovel[i].z.to("km/s").value,
            ]
        )
        assert np.allclose(expected_vel, actual_vel, rtol=TOLERANCE)


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestEarthSpacecraftRelationship:
    """Test relationship between Earth and spacecraft SkyCoord objects."""

    def test_earth_position_is_negative_of_spacecraft_single_point(
        self, single_point_ephem: TLEEphemeris
    ) -> None:
        """Test that Earth position equals negative of spacecraft position (single point)."""
        spacecraft_skycoord = single_point_ephem.gcrs
        earth_skycoord = single_point_ephem.earth
        sc_pos = spacecraft_skycoord[0].cartesian.xyz.to("km").value
        earth_pos = earth_skycoord[0].cartesian.xyz.to("km").value
        expected_earth = -sc_pos
        assert np.allclose(earth_pos, expected_earth, rtol=TOLERANCE)

    def test_earth_velocity_is_negative_of_spacecraft_single_point(
        self, single_point_ephem: TLEEphemeris
    ) -> None:
        """Test that Earth velocity equals negative of spacecraft velocity (single point)."""
        spacecraft_skycoord = single_point_ephem.gcrs
        earth_skycoord = single_point_ephem.earth
        sc_vel = spacecraft_skycoord[0].velocity.d_xyz.to("km/s").value
        earth_vel = earth_skycoord[0].velocity.d_xyz.to("km/s").value
        expected_earth_vel = -sc_vel
        assert np.allclose(earth_vel, expected_earth_vel, rtol=TOLERANCE)
