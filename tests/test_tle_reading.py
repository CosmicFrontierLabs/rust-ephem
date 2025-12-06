# mypy: ignore-errors
"""
Tests for TLE reading enhancements in TLEEphemeris.

This module tests the new TLE reading functionality including:
- File reading (2-line and 3-line formats)
- URL downloading with caching
- Celestrak fetching by NORAD ID and name
- TLE epoch extraction
- Backward compatibility with tle1/tle2 parameters
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

import rust_ephem

# Test TLE data
TLE_2LINE = """1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995
2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"""

TLE_3LINE = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""

TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test time range
BEGIN = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 60


class TestLegacyTLEMethod:
    """Test backward compatibility with original tle1/tle2 parameters."""

    def test_legacy_tle1_tle2(self):
        """Test that the legacy tle1/tle2 method still works."""
        ephem = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)
        assert ephem is not None
        assert ephem.timestamp is not None
        assert len(ephem.timestamp) == 61  # 0 to 60 minutes inclusive

    def test_legacy_with_tle_epoch(self):
        """Test that tle_epoch is available with legacy method."""
        ephem = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)
        assert ephem.tle_epoch is not None
        # TLE epoch should be Oct 14, 2025 (day 287)
        assert ephem.tle_epoch.year == 2025
        assert ephem.tle_epoch.month == 10
        assert ephem.tle_epoch.day == 14
        # Check it has timezone info
        assert ephem.tle_epoch.tzinfo is not None


class TestFileReading:
    """Test reading TLEs from files."""

    def test_read_2line_tle_file(self):
        """Test reading a 2-line TLE file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_2LINE)
            f.flush()
            filepath = f.name

        try:
            ephem = rust_ephem.TLEEphemeris(
                tle=filepath, begin=BEGIN, end=END, step_size=STEP_SIZE
            )
            assert ephem is not None
            assert ephem.timestamp is not None
            assert len(ephem.timestamp) == 61
            assert ephem.tle_epoch is not None
            assert ephem.tle_epoch.year == 2025
        finally:
            os.unlink(filepath)

    def test_read_3line_tle_file(self):
        """Test reading a 3-line TLE file (with satellite name)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            ephem = rust_ephem.TLEEphemeris(
                tle=filepath, begin=BEGIN, end=END, step_size=STEP_SIZE
            )
            assert ephem is not None
            assert ephem.timestamp is not None
            assert len(ephem.timestamp) == 61
            # This TLE is from 2008
            assert ephem.tle_epoch is not None
            assert ephem.tle_epoch.year == 2008
        finally:
            os.unlink(filepath)

    def test_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(ValueError, match="Failed to read TLE from file"):
            rust_ephem.TLEEphemeris(
                tle="/nonexistent/file.tle", begin=BEGIN, end=END, step_size=STEP_SIZE
            )

    def test_invalid_tle_in_file(self):
        """Test error handling with invalid TLE data in file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write("This is not a valid TLE\nAnother invalid line\n")
            f.flush()
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Invalid TLE"):
                rust_ephem.TLEEphemeris(
                    tle=filepath, begin=BEGIN, end=END, step_size=STEP_SIZE
                )
        finally:
            os.unlink(filepath)


class TestTLEEpoch:
    """Test TLE epoch extraction."""

    def test_tle_epoch_format(self):
        """Test that tle_epoch returns a proper datetime object."""
        ephem = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)
        epoch = ephem.tle_epoch

        assert epoch is not None
        assert isinstance(epoch, datetime)
        assert epoch.tzinfo is not None
        assert epoch.year == 2025
        assert epoch.month == 10
        assert epoch.day == 14

    def test_tle_epoch_different_tle(self):
        """Test epoch extraction for different TLE."""
        # ISS TLE from 2008, day 264
        tle1_iss = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        )
        tle2_iss = (
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        )

        ephem = rust_ephem.TLEEphemeris(tle1_iss, tle2_iss, BEGIN, END, STEP_SIZE)
        epoch = ephem.tle_epoch

        # 2008 was a leap year, so day 264 is September 20 (31+29+31+30+31+30+31+31+20=264)
        assert epoch.year == 2008
        assert epoch.month == 9
        assert epoch.day == 20


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_missing_begin_end_parameters(self):
        """Test that begin and end are required."""
        with pytest.raises(ValueError, match="begin parameter is required"):
            rust_ephem.TLEEphemeris(TLE1, TLE2, None, END, STEP_SIZE)

        with pytest.raises(ValueError, match="end parameter is required"):
            rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, None, STEP_SIZE)

    def test_no_tle_parameters_provided(self):
        """Test error when no TLE source is provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            rust_ephem.TLEEphemeris(begin=BEGIN, end=END, step_size=STEP_SIZE)

    def test_conflicting_parameters(self):
        """Test that only one TLE source should be used (documented behavior)."""
        # The constructor should use the first method it finds
        # Priority: tle1/tle2, then tle, then norad_id, then norad_name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_2LINE)
            f.flush()
            filepath = f.name

        try:
            # This should work - uses tle1/tle2 and ignores tle parameter
            ephem = rust_ephem.TLEEphemeris(
                tle1=TLE1,
                tle2=TLE2,
                tle=filepath,  # This should be ignored
                begin=BEGIN,
                end=END,
                step_size=STEP_SIZE,
            )
            assert ephem is not None
        finally:
            os.unlink(filepath)


class TestDataConsistency:
    """Test that all methods produce consistent results."""

    def test_legacy_vs_file_consistency(self):
        """Test that legacy and file methods produce the same results."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_2LINE)
            f.flush()
            filepath = f.name

        try:
            ephem1 = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)
            ephem2 = rust_ephem.TLEEphemeris(
                tle=filepath, begin=BEGIN, end=END, step_size=STEP_SIZE
            )

            # Check that epochs match
            assert ephem1.tle_epoch == ephem2.tle_epoch

            # Check that they produce the same number of timestamps
            assert len(ephem1.timestamp) == len(ephem2.timestamp)

            # Check GCRS positions match (allowing for small numerical differences)
            import numpy as np

            pos1 = ephem1.gcrs_pv.position
            pos2 = ephem2.gcrs_pv.position
            assert np.allclose(pos1, pos2, rtol=1e-10)
        finally:
            os.unlink(filepath)


class TestPolarMotionParameter:
    """Test that polar_motion parameter works with new TLE methods."""

    def test_polar_motion_with_file(self):
        """Test polar_motion parameter with file reading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_2LINE)
            f.flush()
            filepath = f.name

        try:
            ephem = rust_ephem.TLEEphemeris(
                tle=filepath,
                begin=BEGIN,
                end=END,
                step_size=STEP_SIZE,
                polar_motion=True,
            )
            assert ephem is not None
            assert ephem.timestamp is not None
        finally:
            os.unlink(filepath)


# Note: URL and Celestrak tests require network access and would need to be
# integration tests or use mocking. For now, they're documented but not implemented
# in the test file since the environment doesn't have internet access.


class TestURLDownloading:
    """Test URL downloading (requires network access - placeholder for documentation)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_download_from_url(self):
        """Test downloading TLE from URL (placeholder)."""
        # Example: Test downloading from a valid TLE URL
        # url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
        # ephem = rust_ephem.TLEEphemeris(tle=url, begin=BEGIN, end=END, step_size=STEP_SIZE)
        # assert ephem.tle_epoch is not None
        pass

    @pytest.mark.skip(reason="Requires network access")
    def test_url_caching(self):
        """Test that URL downloads are cached (placeholder)."""
        # Test that subsequent calls use cache within TTL
        pass


class TestCelestrakIntegration:
    """Test Celestrak API integration (requires network access - placeholder)."""

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_by_norad_id(self):
        """Test fetching TLE from Celestrak by NORAD ID (placeholder)."""
        # Example: Fetch ISS TLE
        # ephem = rust_ephem.TLEEphemeris(norad_id=25544, begin=BEGIN, end=END, step_size=STEP_SIZE)
        # assert ephem.tle_epoch is not None
        pass

    @pytest.mark.skip(reason="Requires network access")
    def test_fetch_by_name(self):
        """Test fetching TLE from Celestrak by satellite name (placeholder)."""
        # Example: Fetch ISS by name
        # ephem = rust_ephem.TLEEphemeris(norad_name="ISS", begin=BEGIN, end=END, step_size=STEP_SIZE)
        # assert ephem.tle_epoch is not None
        pass


class TestSpaceTrackIntegration:
    """Test Space-Track.org API integration with automatic failover to Celestrak.

    When Space-Track.org credentials are available (via environment variables,
    .env file, or explicit parameters), norad_id will first try Space-Track,
    then failover to Celestrak on failure.

    Credentials can be set via environment variables:
    - SPACETRACK_USERNAME
    - SPACETRACK_PASSWORD

    Or via a .env file in the project root.
    """

    @pytest.mark.skip(reason="Requires network access and Space-Track.org credentials")
    def test_fetch_by_norad_id_with_spacetrack_credentials(self):
        """Test fetching TLE via norad_id using Space-Track.org when credentials available."""
        # Example: Fetch ISS TLE - will use Space-Track.org if credentials are set
        # Requires SPACETRACK_USERNAME and SPACETRACK_PASSWORD env vars
        ephem = rust_ephem.TLEEphemeris(
            norad_id=25544, begin=BEGIN, end=END, step_size=STEP_SIZE
        )
        assert ephem is not None
        assert ephem.tle_epoch is not None
        assert ephem.timestamp is not None
        # TLE epoch should be reasonably close to BEGIN time
        epoch_diff = abs((ephem.tle_epoch - BEGIN).total_seconds())
        assert epoch_diff < 4 * 24 * 3600  # Within 4 days (default tolerance)

    @pytest.mark.skip(reason="Requires network access and Space-Track.org credentials")
    def test_fetch_by_norad_id_with_explicit_credentials(self):
        """Test fetching TLE via norad_id with explicit Space-Track.org credentials."""
        # This test uses explicit credentials passed as parameters
        # You would replace these with actual test credentials
        ephem = rust_ephem.TLEEphemeris(
            norad_id=25544,
            spacetrack_username="your_username",
            spacetrack_password="your_password",
            begin=BEGIN,
            end=END,
            step_size=STEP_SIZE,
        )
        assert ephem is not None
        assert ephem.tle_epoch is not None

    @pytest.mark.skip(reason="Requires network access and Space-Track.org credentials")
    def test_fetch_with_custom_epoch_tolerance(self):
        """Test fetching TLE with custom epoch tolerance for caching."""
        # Test with 7 day tolerance instead of default 4 days
        ephem = rust_ephem.TLEEphemeris(
            norad_id=25544,
            begin=BEGIN,
            end=END,
            step_size=STEP_SIZE,
            epoch_tolerance_days=7.0,
        )
        assert ephem is not None
        assert ephem.tle_epoch is not None

    @pytest.mark.skip(reason="Requires network access and Space-Track.org credentials")
    def test_spacetrack_epoch_based_fetching(self):
        """Test that Space-Track fetches TLE closest to begin epoch."""
        # Use a historical date to test epoch-based fetching
        historical_begin = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        historical_end = datetime(2020, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

        ephem = rust_ephem.TLEEphemeris(
            norad_id=25544,
            begin=historical_begin,
            end=historical_end,
            step_size=STEP_SIZE,
        )
        assert ephem is not None
        # TLE epoch should be in January 2020, not current date
        assert ephem.tle_epoch.year == 2020
        assert ephem.tle_epoch.month == 1

    @pytest.mark.skip(reason="Requires network access to Celestrak")
    def test_norad_id_without_credentials_uses_celestrak(self):
        """Test that norad_id without credentials falls back to Celestrak."""
        # Temporarily clear any env vars
        import os

        old_username = os.environ.pop("SPACETRACK_USERNAME", None)
        old_password = os.environ.pop("SPACETRACK_PASSWORD", None)

        try:
            # This should succeed using Celestrak (no credentials = no Space-Track attempt)
            ephem = rust_ephem.TLEEphemeris(
                norad_id=25544, begin=BEGIN, end=END, step_size=STEP_SIZE
            )
            assert ephem is not None
            assert ephem.tle_epoch is not None
        finally:
            # Restore env vars if they existed
            if old_username is not None:
                os.environ["SPACETRACK_USERNAME"] = old_username
            if old_password is not None:
                os.environ["SPACETRACK_PASSWORD"] = old_password

    def test_spacetrack_partial_credentials_error(self):
        """Test that providing only username or password raises error."""
        with pytest.raises(ValueError) as excinfo:
            rust_ephem.TLEEphemeris(
                norad_id=25544,
                spacetrack_username="test_user",
                # Missing password
                begin=BEGIN,
                end=END,
                step_size=STEP_SIZE,
            )
        assert (
            "password" in str(excinfo.value).lower()
            or "together" in str(excinfo.value).lower()
        )


class TestFetchTLE:
    """Test the fetch_tle function and TLERecord class."""

    def test_fetch_tle_from_file(self):
        """Test fetch_tle from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            tle_record = rust_ephem.fetch_tle(tle=filepath)
            assert tle_record is not None
            assert tle_record.line1.startswith("1 ")
            assert tle_record.line2.startswith("2 ")
            assert tle_record.source == "file"
            assert tle_record.epoch is not None
            assert tle_record.name == "ISS (ZARYA)"
        finally:
            os.unlink(filepath)

    def test_tle_record_properties(self):
        """Test TLERecord computed properties."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            tle_record = rust_ephem.fetch_tle(tle=filepath)
            # Test computed properties
            assert tle_record.norad_id == 25544
            assert tle_record.classification == "U"  # Unclassified
            assert "98067A" in tle_record.international_designator
        finally:
            os.unlink(filepath)

    def test_tle_record_to_string(self):
        """Test TLERecord to_tle_string method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            tle_record = rust_ephem.fetch_tle(tle=filepath)
            tle_string = tle_record.to_tle_string()
            # Should have 3 lines (name + line1 + line2)
            lines = tle_string.strip().split("\n")
            assert len(lines) == 3
            assert lines[0] == "ISS (ZARYA)"
        finally:
            os.unlink(filepath)

    def test_tle_record_json_serialization(self):
        """Test TLERecord can be serialized to JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            tle_record = rust_ephem.fetch_tle(tle=filepath)
            json_str = tle_record.model_dump_json()
            assert "line1" in json_str
            assert "line2" in json_str
            assert "epoch" in json_str

            # Test round-trip
            from rust_ephem import TLERecord

            reconstructed = TLERecord.model_validate_json(json_str)
            assert reconstructed.line1 == tle_record.line1
            assert reconstructed.line2 == tle_record.line2
        finally:
            os.unlink(filepath)

    def test_tle_record_with_ephemeris(self):
        """Test passing TLERecord to TLEEphemeris."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            # Fetch TLE
            tle_record = rust_ephem.fetch_tle(tle=filepath)

            # Use TLERecord with TLEEphemeris
            ephem = rust_ephem.TLEEphemeris(
                tle=tle_record, begin=BEGIN, end=END, step_size=STEP_SIZE
            )
            assert ephem is not None
            assert ephem.timestamp is not None
            assert len(ephem.timestamp) == 61
            # Epoch should match
            assert ephem.tle_epoch.year == tle_record.epoch.year
        finally:
            os.unlink(filepath)

    def test_fetch_tle_missing_params(self):
        """Test that fetch_tle raises error with no params."""
        with pytest.raises(ValueError):
            rust_ephem.fetch_tle()

    def test_tle_record_immutable(self):
        """Test that TLERecord is immutable (frozen)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
            f.write(TLE_3LINE)
            f.flush()
            filepath = f.name

        try:
            tle_record = rust_ephem.fetch_tle(tle=filepath)
            # Should raise error when trying to modify
            from pydantic import ValidationError

            with pytest.raises(ValidationError):
                tle_record.line1 = "modified"
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
