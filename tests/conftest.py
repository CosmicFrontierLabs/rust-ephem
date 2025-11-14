"""Pytest configuration for rust-ephem tests."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_astropy: Tests that require astropy library"
    )
    config.addinivalue_line(
        "markers", "requires_spice: Tests that require SPICE kernel data files"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require optional dependencies if they're not available."""
    # Check for astropy
    try:
        import astropy  # type: ignore[import-untyped]  # noqa: F401

        has_astropy = True
    except ImportError:
        has_astropy = False

    # Mark tests based on module imports
    skip_astropy = pytest.mark.skip(reason="astropy not installed")

    for item in items:
        # Check if test file imports astropy
        if not has_astropy and "skycoord" in item.nodeid.lower():
            item.add_marker(skip_astropy)
        if not has_astropy and "gcrs" in item.nodeid.lower():
            item.add_marker(skip_astropy)
        if not has_astropy and "itrs" in item.nodeid.lower():
            item.add_marker(skip_astropy)
        if not has_astropy and "sun_moon" in item.nodeid.lower():
            item.add_marker(skip_astropy)
