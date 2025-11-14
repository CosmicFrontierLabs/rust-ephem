#!/usr/bin/env python3
"""
Test suite for leap second implementation.
Compares rust_ephem's TAI-UTC offsets against astropy.
"""

import datetime
from datetime import timezone

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


class TestTAIUTCOffsets:
    """Tests for TAI-UTC offset calculations"""

    @pytest.mark.parametrize(
        "date,expected_offset",
        [
            (datetime.datetime(1972, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 10.0),
            (datetime.datetime(1980, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 19.0),
            (datetime.datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 25.0),
            (datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 32.0),
            (datetime.datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 34.0),
            (datetime.datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 37.0),
            (datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 37.0),
        ],
    )
    def test_known_tai_utc_offsets(self, date, expected_offset):
        """Test TAI-UTC offsets for dates with known leap seconds"""
        tai_utc = rust_ephem.get_tai_utc_offset(date)
        assert tai_utc is not None, f"TAI-UTC offset should not be None for {date}"
        assert abs(tai_utc - expected_offset) < 0.001, (
            f"Expected {expected_offset}, got {tai_utc}"
        )

    def test_tai_utc_offset_returns_value(self):
        """Test that TAI-UTC offset returns a numeric value"""
        dt = datetime.datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        tai_utc = rust_ephem.get_tai_utc_offset(dt)
        assert tai_utc is not None, "TAI-UTC offset should not be None"
        assert isinstance(tai_utc, (int, float)), "TAI-UTC offset should be numeric"
        assert tai_utc > 0, "TAI-UTC offset should be positive"

    def test_tai_utc_offset_increases_with_time(self):
        """Test that TAI-UTC offset increases or stays constant over time"""
        dates = [
            datetime.datetime(1980, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ]

        offsets = [rust_ephem.get_tai_utc_offset(dt) for dt in dates]
        assert all(o is not None for o in offsets), "All offsets should be non-None"

        for i in range(len(offsets) - 1):
            assert offsets[i] <= offsets[i + 1], (
                f"TAI-UTC should be non-decreasing: {offsets[i]} > {offsets[i + 1]}"
            )

    def test_tai_utc_constant_between_leap_seconds(self):
        """Test that TAI-UTC remains constant between leap second insertions"""
        # Between 2017-01-01 and 2024-01-01, no leap seconds were added
        dt1 = datetime.datetime(2017, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        offset1 = rust_ephem.get_tai_utc_offset(dt1)
        offset2 = rust_ephem.get_tai_utc_offset(dt2)

        assert offset1 == offset2, (
            f"TAI-UTC should be constant between leap seconds: {offset1} != {offset2}"
        )

    def test_tai_utc_before_1972(self):
        """Test TAI-UTC offset for dates before 1972 (pre-leap second era)"""
        dt = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        tai_utc = rust_ephem.get_tai_utc_offset(dt)
        # Before 1972, behavior may vary - just check it returns something reasonable
        if tai_utc is not None:
            assert tai_utc >= 0, "TAI-UTC should be non-negative"


class TestTTUTCConversion:
    """Tests for TT-UTC offset calculations (TAI-UTC + 32.184)"""

    @pytest.mark.parametrize(
        "date",
        [
            datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2024, 11, 11, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_tt_utc_relationship(self, date):
        """Test that TT-UTC = TAI-UTC + 32.184"""
        tai_utc = rust_ephem.get_tai_utc_offset(date)
        assert tai_utc is not None, f"TAI-UTC should not be None for {date}"

        tt_utc = tai_utc + 32.184
        assert tt_utc > 32.184, "TT-UTC should be greater than 32.184"
        assert tt_utc < 100.0, "TT-UTC should be reasonable (< 100 seconds)"

    def test_tt_utc_consistency(self):
        """Test TT-UTC offset consistency across multiple calls"""
        dt = datetime.datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        offsets = [rust_ephem.get_tai_utc_offset(dt) for _ in range(5)]
        assert all(o is not None for o in offsets), "All offsets should be non-None"
        assert len(set(offsets)) == 1, "TT-UTC should be consistent across calls"


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
class TestAstropyComparison:
    """Tests comparing rust_ephem with astropy"""

    @pytest.mark.parametrize(
        "date",
        [
            datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2010, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            datetime.datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            datetime.datetime(2024, 11, 11, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_tt_utc_matches_astropy(self, date):
        """Test that TT-UTC matches astropy within 1ms"""
        # Get rust_ephem offset
        tai_utc_rust = rust_ephem.get_tai_utc_offset(date)
        assert tai_utc_rust is not None, "rust_ephem should return TAI-UTC offset"
        tt_utc_rust = tai_utc_rust + 32.184

        # Get astropy offset
        t = Time(date, scale="utc")
        tt_jd = t.tt.jd
        utc_jd = t.utc.jd
        tt_utc_astropy = (tt_jd - utc_jd) * 86400.0  # Convert days to seconds

        diff_ms = abs(tt_utc_rust - tt_utc_astropy) * 1000
        assert diff_ms < 1.0, f"TT-UTC difference should be < 1ms, got {diff_ms:.3f}ms"

    def test_tai_utc_matches_astropy(self):
        """Test that TAI-UTC matches astropy's delta_tai_utc"""
        date = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        tai_utc_rust = rust_ephem.get_tai_utc_offset(date)
        assert tai_utc_rust is not None

        t = Time(date, scale="utc")
        # Get TAI-UTC from astropy
        tai_jd = t.tai.jd
        utc_jd = t.utc.jd
        tai_utc_astropy = (tai_jd - utc_jd) * 86400.0

        diff = abs(tai_utc_rust - tai_utc_astropy)
        assert diff < 0.001, f"TAI-UTC should match astropy, difference: {diff:.6f}s"


class TestEphemerisWithLeapSeconds:
    """Tests for ephemeris calculations with leap second corrections"""

    @pytest.fixture
    def tle_lines(self):
        """Provide TLE data for testing"""
        return (
            "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995",
            "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530",
        )

    def test_ephemeris_creation_with_leap_seconds(self, tle_lines):
        """Test that ephemeris can be created with leap second corrections"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 3600

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        assert ephem is not None, "Ephemeris should be created successfully"

    def test_ephemeris_propagation_with_leap_seconds(self, tle_lines):
        """Test that ephemeris propagation works with leap second corrections"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 3600

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        ephem.propagate_to_teme()

        assert ephem.teme_pv is not None, "TEME data should be available"
        assert ephem.teme_pv.position is not None, "TEME position should be available"

    def test_ephemeris_gcrs_transformation_with_leap_seconds(self, tle_lines):
        """Test GCRS transformation with leap second corrections"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 3600

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        ephem.propagate_to_teme()
        ephem.teme_to_gcrs()

        assert ephem.gcrs_pv is not None, "GCRS data should be available"
        assert ephem.gcrs_pv.position is not None, "GCRS position should be available"
        assert ephem.gcrs_pv.velocity is not None, "GCRS velocity should be available"

    def test_ephemeris_position_magnitude(self, tle_lines):
        """Test that GCRS position magnitude is reasonable for LEO"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 3600

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        ephem.propagate_to_teme()
        ephem.teme_to_gcrs()

        pos = ephem.gcrs_pv.position[0]
        pos_mag = (pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5

        assert 6500 < pos_mag < 8000, (
            f"Position magnitude should be in LEO range, got {pos_mag:.3f} km"
        )

    def test_ephemeris_velocity_magnitude(self, tle_lines):
        """Test that GCRS velocity magnitude is reasonable for LEO"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
        step_size = 3600

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        ephem.propagate_to_teme()
        ephem.teme_to_gcrs()

        vel = ephem.gcrs_pv.velocity[0]
        vel_mag = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5

        assert 7.0 < vel_mag < 8.0, (
            f"Velocity magnitude should be orbital speed, got {vel_mag:.6f} km/s"
        )

    def test_ephemeris_with_multiple_timesteps(self, tle_lines):
        """Test ephemeris with multiple time steps"""
        tle1, tle2 = tle_lines
        begin = datetime.datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2025, 10, 14, 6, 0, 0, tzinfo=timezone.utc)
        step_size = 3600  # 1 hour steps

        ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
        ephem.propagate_to_teme()
        ephem.teme_to_gcrs()

        assert len(ephem.gcrs_pv.position) == 7, "Should have 7 positions (0-6 hours)"
        assert len(ephem.gcrs_pv.velocity) == 7, "Should have 7 velocities (0-6 hours)"

        # Check all positions are reasonable
        for pos in ephem.gcrs_pv.position:
            pos_mag = (pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5
            assert 6500 < pos_mag < 8000, (
                f"All positions should be in LEO range, got {pos_mag:.3f} km"
            )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_leap_second_boundary(self):
        """Test TAI-UTC offset at leap second boundaries"""
        # Right before 2017-01-01 leap second
        before = datetime.datetime(2016, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        # Right after 2017-01-01 leap second
        after = datetime.datetime(2017, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        offset_before = rust_ephem.get_tai_utc_offset(before)
        offset_after = rust_ephem.get_tai_utc_offset(after)

        assert offset_before is not None and offset_after is not None
        assert offset_after == offset_before + 1.0, (
            "Offset should increase by 1 second across leap second"
        )

    def test_future_date_extrapolation(self):
        """Test TAI-UTC offset for future dates"""
        future = datetime.datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        tai_utc = rust_ephem.get_tai_utc_offset(future)

        # Should either return current offset or None
        if tai_utc is not None:
            assert tai_utc >= 37.0, "Future offset should be at least current value"

    def test_microsecond_precision(self):
        """Test that leap seconds work with microsecond precision"""
        dt = datetime.datetime(2020, 6, 15, 12, 30, 45, 123456, tzinfo=timezone.utc)
        tai_utc = rust_ephem.get_tai_utc_offset(dt)

        assert tai_utc is not None, "Should handle microsecond precision"
        assert tai_utc == 37.0, "Microseconds should not affect leap second offset"
