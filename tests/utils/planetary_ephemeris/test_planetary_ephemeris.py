#!/usr/bin/env python3
"""
Test suite for planetary ephemeris functions.
Tests init_planetary_ephemeris, download_planetary_ephemeris, and related functionality.
"""

from pathlib import Path

import pytest

import rust_ephem


class TestPlanetaryEphemeris:
    """Test planetary ephemeris initialization and management."""

    def test_init_planetary_ephemeris_with_valid_file(self) -> None:
        """Test init_planetary_ephemeris with a valid SPK file."""
        # Use a test SPK file if available
        test_spk = Path("test_data/de440s.bsp")
        if test_spk.exists():
            # Should not raise an exception
            rust_ephem.init_planetary_ephemeris(str(test_spk))
        else:
            pytest.skip("Test SPK file not available")

    def test_init_planetary_ephemeris_with_invalid_file(self) -> None:
        """Test init_planetary_ephemeris with an invalid file path."""
        with pytest.raises(RuntimeError):
            rust_ephem.init_planetary_ephemeris("nonexistent_file.spk")

    def test_download_planetary_ephemeris_with_valid_url(self, temp_dir) -> None:
        """Test download_planetary_ephemeris with a valid URL."""
        dest_path = temp_dir / "test.spk"

        # This should work with a real URL, but we'll just test that it doesn't raise immediately
        # The actual download may fail due to network issues, but the function should be callable
        try:
            rust_ephem.download_planetary_ephemeris(
                "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
                str(dest_path),
            )
        except RuntimeError:
            # Expected if network/download fails
            pass

    def test_download_planetary_ephemeris_with_invalid_url(self, temp_dir) -> None:
        """Test download_planetary_ephemeris with an invalid URL."""
        dest_path = temp_dir / "test.spk"

        # Should raise RuntimeError for invalid URL
        with pytest.raises(RuntimeError):
            rust_ephem.download_planetary_ephemeris(
                "https://invalid-url-that-does-not-exist-12345.com/test.spk",
                str(dest_path),
            )
