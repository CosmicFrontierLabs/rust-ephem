"""Pytest configuration for rust-ephem tests."""

import os
from datetime import datetime, timezone
from typing import Any

import pytest
from _pytest.config import Config

# Common test constants used across multiple test modules
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)


# Common test constants used across multiple test modules
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


@pytest.fixture(scope="module")
def ensure_planetary_data() -> None:
    """Ensure planetary ephemeris is loaded once for all tests"""
    import rust_ephem

    test_data_path = os.path.join(os.path.dirname(__file__), "test_data", "de440s.bsp")
    if os.path.exists(test_data_path):
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=False
        )
    else:
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=True
        )


@pytest.fixture
def tle_ephemeris(ensure_planetary_data: Any) -> Any:
    """Create a standard TLEEphemeris instance for testing"""
    import rust_ephem

    return rust_ephem.TLEEphemeris(
        VALID_TLE1, VALID_TLE2, BEGIN_TIME, END_TIME, STEP_SIZE
    )


@pytest.fixture
def ground_ephemeris(ensure_planetary_data: Any) -> Any:
    """Create a standard GroundEphemeris instance for testing"""
    import rust_ephem

    return rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def temp_dir() -> Any:
    """Create a temporary directory for testing."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def rust_ephem_available() -> bool:
    """Check if rust_ephem is available."""
    try:
        import rust_ephem  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_astropy: Tests that require astropy library"
    )
    config.addinivalue_line(
        "markers", "requires_spice: Tests that require SPICE kernel data files"
    )


def pytest_collection_modifyitems(config: Config, items: list[Any]) -> None:
    """Skip tests that require optional dependencies if they're not available."""
    # Check for astropy
    try:
        import astropy  # type: ignore[import-untyped]  # noqa: F401

        has_astropy = True
    except ImportError:
        has_astropy = False

    # Check for rust_ephem
    try:
        import rust_ephem  # noqa: F401

        has_rust_ephem = True
    except ImportError:
        has_rust_ephem = False

    # Mark tests based on module imports
    skip_astropy: pytest.MarkDecorator = pytest.mark.skip(
        reason="astropy not installed"
    )
    skip_rust_ephem: pytest.MarkDecorator = pytest.mark.skip(
        reason="rust_ephem extension not built"
    )

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

        # Check if test requires rust_ephem
        if not has_rust_ephem and "rust_ephem" in str(item.fspath):
            item.add_marker(skip_rust_ephem)
