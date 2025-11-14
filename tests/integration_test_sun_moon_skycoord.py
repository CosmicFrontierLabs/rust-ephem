#!/usr/bin/env python3
"""
Integration test for sun and moon properties.

This test verifies that:
1. sun and moon properties exist and work
2. The returned SkyCoord objects have obsgeoloc/obsgeovel set correctly
3. The celestial body positions and velocities match expected values
4. obsgeoloc/obsgeovel match the spacecraft GCRS position/velocity

Requirements:
    pip install astropy numpy

Build and install the Rust module first:
    maturin build --release
    pip install target/wheels/*.whl
"""

import sys
from datetime import datetime, timezone

import numpy as np
import pytest

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False

try:
    from astropy.coordinates import (  # type: ignore[import-untyped]
        GCRS,
        SkyCoord,
        get_body,
    )
    from astropy.time import Time  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Test TLE for NORAD ID 28485 (2004-047A)
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test times
BEGIN = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 10, 14, 0, 30, 0, tzinfo=timezone.utc)
STEP_SIZE = 60  # 1 minute

# Position/velocity tolerance
TOLERANCE = 1e-9


@pytest.fixture
def ephem():
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)


@pytest.fixture
def sun(ephem):
    return ephem.sun


@pytest.fixture
def moon(ephem):
    return ephem.moon


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_sun_type(sun):
    """Test that sun property returns SkyCoord."""
    assert isinstance(sun, SkyCoord), f"Expected SkyCoord, got {type(sun)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_sun_length(sun, ephem):
    """Test that sun has correct length."""
    expected_len = len(ephem.timestamp)
    assert len(sun) == expected_len, f"Expected {expected_len} coords, got {len(sun)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_sun_frame(sun):
    """Test that sun is in GCRS frame."""
    assert isinstance(sun.frame, GCRS), f"Expected GCRS frame, got {type(sun.frame)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_sun_obsgeoloc_set(sun):
    """Test that sun obsgeoloc is set."""
    assert sun.frame.obsgeoloc is not None, "obsgeoloc not set"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_sun_obsgeovel_set(sun):
    """Test that sun obsgeovel is set."""
    assert sun.frame.obsgeovel is not None, "obsgeovel not set"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_sun_obsgeoloc_match(sun, ephem, i):
    """Test that sun obsgeoloc matches spacecraft position at index i."""
    expected_pos = ephem.gcrs_pv.position[i]
    actual_pos = np.array(
        [
            sun.frame.obsgeoloc[i].x.to("km").value,
            sun.frame.obsgeoloc[i].y.to("km").value,
            sun.frame.obsgeoloc[i].z.to("km").value,
        ]
    )
    assert np.allclose(expected_pos, actual_pos, rtol=TOLERANCE), (
        f"obsgeoloc mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_sun_obsgeovel_match(sun, ephem, i):
    """Test that sun obsgeovel matches spacecraft velocity at index i."""
    expected_vel = ephem.gcrs_pv.velocity[i]
    actual_vel = np.array(
        [
            sun.frame.obsgeovel[i].x.to("km/s").value,
            sun.frame.obsgeovel[i].y.to("km/s").value,
            sun.frame.obsgeovel[i].z.to("km/s").value,
        ]
    )
    assert np.allclose(expected_vel, actual_vel, rtol=TOLERANCE), (
        f"obsgeovel mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_sun_position_match(sun, ephem, i):
    """Test that sun position matches expected at index i."""
    expected_sun_pos = ephem.sun_pv.position[i] - ephem.gcrs_pv.position[i]
    actual_sun_pos = sun[i].cartesian.xyz.to("km").value
    assert np.allclose(expected_sun_pos, actual_sun_pos, rtol=TOLERANCE), (
        f"Sun position mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_sun_velocity_match(sun, ephem, i):
    """Test that sun velocity matches expected at index i."""
    expected_sun_vel = ephem.sun_pv.velocity[i] - ephem.gcrs_pv.velocity[i]
    actual_sun_vel = sun[i].velocity.d_xyz.to("km/s").value
    assert np.allclose(expected_sun_vel, actual_sun_vel, rtol=TOLERANCE), (
        f"Sun velocity mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_moon_type(moon):
    """Test that moon property returns SkyCoord."""
    assert isinstance(moon, SkyCoord), f"Expected SkyCoord, got {type(moon)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_moon_length(moon, ephem):
    """Test that moon has correct length."""
    expected_len = len(ephem.timestamp)
    assert len(moon) == expected_len, f"Expected {expected_len} coords, got {len(moon)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_moon_frame(moon):
    """Test that moon is in GCRS frame."""
    assert isinstance(moon.frame, GCRS), f"Expected GCRS frame, got {type(moon.frame)}"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_moon_obsgeoloc_set(moon):
    """Test that moon obsgeoloc is set."""
    assert moon.frame.obsgeoloc is not None, "obsgeoloc not set"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_moon_obsgeovel_set(moon):
    """Test that moon obsgeovel is set."""
    assert moon.frame.obsgeovel is not None, "obsgeovel not set"


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_moon_obsgeoloc_match(moon, ephem, i):
    """Test that moon obsgeoloc matches spacecraft position at index i."""
    expected_pos = ephem.gcrs_pv.position[i]
    actual_pos = np.array(
        [
            moon.frame.obsgeoloc[i].x.to("km").value,
            moon.frame.obsgeoloc[i].y.to("km").value,
            moon.frame.obsgeoloc[i].z.to("km").value,
        ]
    )
    assert np.allclose(expected_pos, actual_pos, rtol=TOLERANCE), (
        f"obsgeoloc mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_moon_obsgeovel_match(moon, ephem, i):
    """Test that moon obsgeovel matches spacecraft velocity at index i."""
    expected_vel = ephem.gcrs_pv.velocity[i]
    actual_vel = np.array(
        [
            moon.frame.obsgeovel[i].x.to("km/s").value,
            moon.frame.obsgeovel[i].y.to("km/s").value,
            moon.frame.obsgeovel[i].z.to("km/s").value,
        ]
    )
    assert np.allclose(expected_vel, actual_vel, rtol=TOLERANCE), (
        f"obsgeovel mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_moon_position_match(moon, ephem, i):
    """Test that moon position matches expected at index i."""
    expected_moon_pos = ephem.moon_pv.position[i] - ephem.gcrs_pv.position[i]
    actual_moon_pos = moon[i].cartesian.xyz.to("km").value
    assert np.allclose(expected_moon_pos, actual_moon_pos, rtol=TOLERANCE), (
        f"Moon position mismatch at index {i}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))  # 31 timestamps based on BEGIN, END, STEP_SIZE
def test_moon_velocity_match(moon, ephem, i):
    """Test that moon velocity matches expected at index i."""
    expected_moon_vel = ephem.moon_pv.velocity[i] - ephem.gcrs_pv.velocity[i]
    actual_moon_vel = moon[i].velocity.d_xyz.to("km/s").value
    assert np.allclose(expected_moon_vel, actual_moon_vel, rtol=TOLERANCE), (
        f"Moon velocity mismatch at index {i}"
    )


# Find Moon position using astropy get_body and compare to ephemeris
# value_chain
@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))
def test_moon_ra_compare_to_astropy(moon, ephem, i):
    """Test that moon RA matches expected using astropy at index i."""
    expected_moon_pos = ephem.moon[i]
    actual_moon_pos = get_body(
        "moon", Time(ephem.timestamp[i]), location=ephem.itrs.earth_location[i]
    )
    assert np.allclose(expected_moon_pos.ra.value, actual_moon_pos.ra.value, rtol=1e-3)


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))
def test_moon_dec_compare_to_astropy(moon, ephem, i):
    """Test that moon Dec matches expected using astropy at index i."""
    expected_moon_pos = ephem.moon[i]
    actual_moon_pos = get_body(
        "moon", Time(ephem.timestamp[i]), location=ephem.itrs.earth_location[i]
    )
    assert np.allclose(
        expected_moon_pos.dec.value, actual_moon_pos.dec.value, rtol=1e-3
    )


# Find Sun position using astropy get_body and compare to ephemeris
@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))
def test_sun_ra_compare_to_astropy(sun, ephem, i):
    """Test that sun RA matches expected using astropy at index i."""
    expected_sun_pos = ephem.sun[i]
    actual_sun_pos = get_body(
        "sun", Time(ephem.timestamp[i]), location=ephem.itrs.earth_location[i]
    )
    assert np.allclose(expected_sun_pos.ra.value, actual_sun_pos.ra.value, rtol=1e-4)


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.parametrize("i", range(31))
def test_sun_dec_compare_to_astropy(sun, ephem, i):
    """Test that sun Dec matches expected using astropy at index i."""
    expected_sun_pos = ephem.sun[i]
    actual_sun_pos = get_body(
        "sun", Time(ephem.timestamp[i]), location=ephem.itrs.earth_location[i]
    )
    assert np.allclose(expected_sun_pos.dec.value, actual_sun_pos.dec.value, rtol=1e-3)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Sun and Moon gcrs_to_skycoord() Integration Tests")
    print("=" * 80)

    # Since the tests are now pytest-based, we can run them via pytest
    # For standalone execution, we can use pytest.main()
    import pytest

    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
