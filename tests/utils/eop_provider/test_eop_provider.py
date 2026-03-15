#!/usr/bin/env python3
"""
Test suite for EOP (Earth Orientation Parameters) provider functions.
Tests init_eop_provider, is_eop_available, and related functionality.
"""

import rust_ephem


class TestEOPProvider:
    """Test EOP provider initialization and availability."""

    def test_init_eop_provider_returns_bool(self) -> None:
        """Test that init_eop_provider returns a boolean."""
        result = rust_ephem.init_eop_provider()
        assert isinstance(result, bool)

    def test_is_eop_available_returns_bool(self) -> None:
        """Test that is_eop_available returns a boolean."""
        result = rust_ephem.is_eop_available()
        assert isinstance(result, bool)

    def test_is_eop_available_after_init(self) -> None:
        """Test that is_eop_available reflects initialization state."""
        # Initialize EOP provider
        init_result = rust_ephem.init_eop_provider()

        # Check availability - should be True if initialization succeeded
        available = rust_ephem.is_eop_available()

        # If initialization succeeded, EOP should be available
        if init_result:
            assert available
        # If initialization failed, availability might still be False
        # (we don't assert this since it depends on system state)
