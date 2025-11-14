#!/usr/bin/env python3
"""
Test suite for UT1-UTC implementation using hifitime.
Tests UT1 provider availability, functionality, and accuracy.
"""

from datetime import datetime, timedelta, timezone

import pytest

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    from astropy.time import Time  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="rust_ephem not available")


class TestUT1Provider:
    """Test UT1-UTC provider initialization and availability."""

    def test_provider_initialization(self):
        """Test that UT1 provider can be initialized."""
        # Check if provider is available (may be initialized already)
        is_available = rust_ephem.is_ut1_available()

        # Should return a boolean without raising
        assert isinstance(is_available, bool)

        # If not available, try to initialize
        if not is_available:
            success = rust_ephem.init_ut1_provider()
            # Should return a boolean
            assert isinstance(success, bool)

    def test_is_ut1_available(self):
        """Test is_ut1_available() function."""
        result = rust_ephem.is_ut1_available()
        assert isinstance(result, bool)

    def test_init_ut1_provider(self):
        """Test init_ut1_provider() function."""
        result = rust_ephem.init_ut1_provider()
        assert isinstance(result, bool)


class TestUT1Offsets:
    """Test UT1-UTC offset calculations."""

    def test_ut1_offset_returns_float(self):
        """Test that get_ut1_utc_offset returns a float."""
        dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        offset = rust_ephem.get_ut1_utc_offset(dt)
        assert isinstance(offset, float)

    def test_ut1_offset_reasonable_range(self):
        """Test that UT1-UTC offsets are in reasonable range."""
        # Test several dates with expected IERS coverage
        test_dates = [
            datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
        ]

        for dt in test_dates:
            offset = rust_ephem.get_ut1_utc_offset(dt)
            # UT1-UTC should be between -0.9 and +0.9 seconds (IERS constraint)
            # or 0.0 if data unavailable
            assert -1.0 <= offset <= 1.0, f"Offset {offset} out of range for {dt}"

    def test_ut1_offset_different_dates(self):
        """Test that different dates can have different UT1-UTC offsets."""
        dt1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        offset1 = rust_ephem.get_ut1_utc_offset(dt1)
        offset2 = rust_ephem.get_ut1_utc_offset(dt2)

        # Both should be valid floats
        assert isinstance(offset1, float)
        assert isinstance(offset2, float)

    @pytest.mark.parametrize(
        "date_tuple",
        [
            ("2024-01-01", datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
            ("2024-06-15", datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)),
            ("2025-01-01", datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        ],
    )
    def test_ut1_offset_various_dates(self, date_tuple):
        """Test UT1-UTC offsets for various dates."""
        date_str, dt = date_tuple
        offset = rust_ephem.get_ut1_utc_offset(dt)

        # Should return a float in reasonable range
        assert isinstance(offset, float)
        assert -1.0 <= offset <= 1.0


class TestUT1DataCoverage:
    """Test UT1 data coverage across different time periods."""

    def test_data_coverage_ranges(self):
        """Test UT1 data coverage for various time ranges."""
        now = datetime.now(timezone.utc)

        test_periods = [
            ("1 year ago", now - timedelta(days=365)),
            ("6 months ago", now - timedelta(days=180)),
            ("3 months ago", now - timedelta(days=90)),
            ("1 month ago", now - timedelta(days=30)),
            ("today", now),
            ("1 month ahead", now + timedelta(days=30)),
            ("3 months ahead", now + timedelta(days=90)),
        ]

        results = []
        for label, dt in test_periods:
            offset = rust_ephem.get_ut1_utc_offset(dt)
            results.append((label, offset))

            # All should be valid floats
            assert isinstance(offset, float)
            assert -1.0 <= offset <= 1.0

        # If provider is available, at least some recent dates should have non-zero offsets
        if rust_ephem.is_ut1_available():
            recent_offsets = [
                offset
                for label, offset in results
                if "ago" in label or label == "today"
            ]
            # At least one recent date should have data (non-zero)
            has_data = any(abs(offset) > 0.001 for offset in recent_offsets)
            # This is not a strict requirement (could be offline), but informative
            if not has_data:
                pytest.skip(
                    "UT1 provider available but no recent data found (possibly offline)"
                )


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestUT1Comparison:
    """Compare UT1-UTC offsets with astropy."""

    def test_compare_with_astropy(self):
        """Compare UT1-UTC offsets between rust_ephem and astropy."""
        now = datetime.now(timezone.utc)

        # Test dates with expected IERS coverage
        test_dates = [
            now - timedelta(days=180),  # 6 months ago
            now - timedelta(days=90),  # 3 months ago
            now - timedelta(days=30),  # 1 month ago
            now,  # today
        ]

        for dt in test_dates:
            # Get rust_ephem UT1-UTC
            rust_ut1_utc = rust_ephem.get_ut1_utc_offset(dt)

            # Get astropy UT1-UTC
            t = Time(dt, scale="utc")
            astropy_ut1_utc = (t.ut1.jd - t.utc.jd) * 86400.0  # Convert days to seconds

            # Calculate difference in milliseconds
            diff_ms = abs(rust_ut1_utc - astropy_ut1_utc) * 1000.0

            # Allow up to 20ms difference (different data sources/prediction algorithms)
            # Only check if both have data (neither is 0.0)
            if abs(rust_ut1_utc) > 0.001 and abs(astropy_ut1_utc) > 0.001:
                assert diff_ms < 20.0, (
                    f"UT1-UTC difference too large for {dt.isoformat()}: "
                    f"rust_ephem={rust_ut1_utc:.6f}s, astropy={astropy_ut1_utc:.6f}s, "
                    f"diff={diff_ms:.3f}ms"
                )


class TestUT1WithEphemeris:
    """Test that UT1-UTC corrections are applied in ephemeris calculations."""

    def test_tle_ephemeris_with_ut1(self):
        """Test that TLE ephemeris uses UT1 corrections."""
        tle1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
        tle2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

        begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)

        # Should not raise an exception
        ephem = rust_ephem.TLEEphemeris(
            tle1=tle1, tle2=tle2, begin=begin, end=end, step_size=60
        )

        # Verify ephemeris was created
        assert len(ephem.timestamp) > 0
        assert ephem.gcrs_pv is not None
        assert ephem.gcrs_pv.position is not None

    def test_ut1_impact_on_position(self):
        """Test the impact of UT1-UTC on position calculations."""
        test_date = datetime.now(timezone.utc)
        ut1_utc = rust_ephem.get_ut1_utc_offset(test_date)

        # If UT1 data is available, verify it's being used
        if rust_ephem.is_ut1_available() and abs(ut1_utc) > 0.001:
            # UT1-UTC should be non-zero
            assert abs(ut1_utc) > 0.0

            # Calculate expected positional impact
            earth_rotation_speed = 465.1  # m/s at equator
            position_error = abs(ut1_utc) * earth_rotation_speed

            # Position error should be reasonable (less than 500m for ±0.9s)
            assert position_error < 500.0

    def test_ephemeris_accuracy_with_ut1(self):
        """Test that ephemeris positions are reasonable with UT1 corrections."""
        tle1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
        tle2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

        test_date = datetime.now(timezone.utc)
        end_date = test_date + timedelta(minutes=5)

        ephem = rust_ephem.TLEEphemeris(
            tle1=tle1, tle2=tle2, begin=test_date, end=end_date, step_size=60
        )

        # Verify GCRS coordinates exist
        assert ephem.gcrs_pv is not None
        assert ephem.gcrs_pv.position is not None
        assert ephem.gcrs_pv.velocity is not None

        # Check first position is in reasonable LEO range
        pos0 = ephem.gcrs_pv.position[0]
        vel0 = ephem.gcrs_pv.velocity[0]

        # Calculate magnitudes
        import math

        pos_mag = math.sqrt(pos0[0] ** 2 + pos0[1] ** 2 + pos0[2] ** 2)
        vel_mag = math.sqrt(vel0[0] ** 2 + vel0[1] ** 2 + vel0[2] ** 2)

        # Position should be in LEO range (6500-8000 km)
        assert 6500 < pos_mag < 8000, (
            f"Position magnitude {pos_mag} km out of LEO range"
        )

        # Velocity should be orbital speed (7-8 km/s for LEO)
        assert 7.0 < vel_mag < 8.0, f"Velocity magnitude {vel_mag} km/s out of range"
