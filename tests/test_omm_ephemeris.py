"""Unit tests for OMMEphemeris class."""

from datetime import datetime, timezone

import pytest

from rust_ephem import OMMEphemeris


class TestOMMEphemeris:
    """Test OMMEphemeris functionality."""

    @pytest.fixture
    def begin_time(self) -> datetime:
        """Test begin time."""
        return datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def end_time(self) -> datetime:
        """Test end time."""
        return datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def step_size(self) -> int:
        """Test step size."""
        return 120  # 2 minutes

    @pytest.fixture
    def mock_omm_json(self) -> str:
        """Mock OMM JSON data for testing without internet access."""
        return """[{
            "NORAD_CAT_ID": 25544,
            "OBJECT_NAME": "ISS (ZARYA)",
            "EPOCH": "2025-12-12T03:41:57.165504",
            "MEAN_MOTION": 15.496999999999998,
            "ECCENTRICITY": 0.0001,
            "INCLINATION": 51.6407,
            "RA_OF_ASC_NODE": 123.4567,
            "ARG_OF_PERICENTER": 234.5678,
            "MEAN_ANOMALY": 345.6789,
            "EPHEMERIS_TYPE": 0,
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 12345,
            "BSTAR": 0.0001,
            "MEAN_MOTION_DOT": 0.000001,
            "MEAN_MOTION_DDOT": 0.0
        }]"""

    def test_omm_ephemeris_creation_celestrak(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris creation with Celestrak source."""
        eph = OMMEphemeris(
            norad_id=25544,  # ISS
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            enforce_source="celestrak",
            omm=mock_omm_json,  # type: ignore
        )

        assert isinstance(eph, OMMEphemeris)
        assert eph.begin == begin_time
        assert eph.end == end_time
        assert eph.step_size == step_size
        assert hasattr(eph, "omm_epoch")

    def test_omm_ephemeris_creation_automatic(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris creation with automatic source selection."""
        eph = OMMEphemeris(
            norad_id=25544,  # ISS
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            omm=mock_omm_json,  # type: ignore
        )

        assert isinstance(eph, OMMEphemeris)
        assert eph.begin == begin_time
        assert eph.end == end_time
        assert eph.step_size == step_size
        assert hasattr(eph, "omm_epoch")

    def test_omm_ephemeris_properties(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris property access."""
        eph = OMMEphemeris(
            norad_id=25544,
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            enforce_source="celestrak",
            omm=mock_omm_json,  # type: ignore
        )

        # Test basic properties
        assert isinstance(eph.begin, datetime)
        assert isinstance(eph.end, datetime)
        assert isinstance(eph.step_size, int)
        assert isinstance(eph.polar_motion, bool)
        assert isinstance(eph.omm_epoch, datetime)

        # Test that OMM epoch is reasonable (not in future, not too old)
        now = datetime.now(timezone.utc)
        assert eph.omm_epoch < now  # Not in the future
        assert eph.omm_epoch > now.replace(year=now.year - 2)  # Not older than 2 years

    def test_omm_ephemeris_invalid_norad_id(
        self, begin_time: datetime, end_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris with invalid NORAD ID."""
        with pytest.raises(ValueError, match="parse_omm_json failed"):
            OMMEphemeris(
                norad_id=999999,  # Invalid NORAD ID
                begin=begin_time,
                end=end_time,
                step_size=step_size,
                enforce_source="celestrak",
                omm="{invalid json",  # type: ignore
            )

    def test_omm_ephemeris_missing_begin(
        self, end_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris creation without begin parameter."""
        with pytest.raises(ValueError, match="begin parameter is required"):
            OMMEphemeris(
                norad_id=25544,
                end=end_time,
                step_size=step_size,
                enforce_source="celestrak",
            )

    def test_omm_ephemeris_missing_end(
        self, begin_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris creation without end parameter."""
        with pytest.raises(ValueError, match="end parameter is required"):
            OMMEphemeris(
                norad_id=25544,
                begin=begin_time,
                step_size=step_size,
                enforce_source="celestrak",
            )

    def test_omm_ephemeris_missing_norad_id(
        self, begin_time: datetime, end_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris creation without norad_id parameter."""
        with pytest.raises(ValueError, match="norad_id required when enforce_source"):
            OMMEphemeris(
                begin=begin_time,
                end=end_time,
                step_size=step_size,
                enforce_source="celestrak",
            )

    def test_omm_ephemeris_invalid_enforce_source(
        self, begin_time: datetime, end_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris with invalid enforce_source parameter."""
        with pytest.raises(ValueError, match="Unknown enforce_source"):
            OMMEphemeris(
                norad_id=25544,
                begin=begin_time,
                end=end_time,
                step_size=step_size,
                enforce_source="invalid_source",
            )

    def test_omm_ephemeris_polar_motion(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris with polar motion enabled."""
        eph_no_polar = OMMEphemeris(
            norad_id=25544,
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            polar_motion=False,
            enforce_source="celestrak",
            omm=mock_omm_json,  # type: ignore
        )

        eph_with_polar = OMMEphemeris(
            norad_id=25544,
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            polar_motion=True,
            enforce_source="celestrak",
            omm=mock_omm_json,  # type: ignore
        )

        assert eph_no_polar.polar_motion is False
        assert eph_with_polar.polar_motion is True

    @pytest.mark.parametrize("step_size", [30, 60, 300, 3600])
    def test_omm_ephemeris_step_size_variations(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris with different step sizes."""
        eph = OMMEphemeris(
            norad_id=25544,
            begin=begin_time,
            end=end_time,
            step_size=step_size,
            enforce_source="celestrak",
            omm=mock_omm_json,  # type: ignore
        )

        assert eph.step_size == step_size
        assert isinstance(eph, OMMEphemeris)

    def test_omm_ephemeris_from_file(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris creation from OMM file."""
        import os
        import tempfile

        # Create a temporary file with OMM data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(mock_omm_json)
            temp_file = f.name

        try:
            eph = OMMEphemeris(
                begin=begin_time, end=end_time, step_size=step_size, omm=temp_file
            )

            assert isinstance(eph, OMMEphemeris)
            assert eph.begin == begin_time
            assert eph.end == end_time
            assert eph.step_size == step_size
            assert eph.norad_cat_id == 25544
        finally:
            os.unlink(temp_file)

    def test_omm_ephemeris_from_url(
        self, begin_time: datetime, end_time: datetime, step_size: int
    ) -> None:
        """Test OMMEphemeris creation from OMM URL."""
        # Use Celestrak URL for testing
        url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON"

        eph = OMMEphemeris(begin=begin_time, end=end_time, step_size=step_size, omm=url)

        assert isinstance(eph, OMMEphemeris)
        assert eph.begin == begin_time
        assert eph.end == end_time
        assert eph.step_size == step_size
        assert eph.norad_cat_id == 25544

    def test_omm_ephemeris_from_json_string(
        self,
        begin_time: datetime,
        end_time: datetime,
        step_size: int,
        mock_omm_json: str,
    ) -> None:
        """Test OMMEphemeris creation from direct JSON string."""
        eph = OMMEphemeris(
            begin=begin_time, end=end_time, step_size=step_size, omm=mock_omm_json
        )

        assert isinstance(eph, OMMEphemeris)
        assert eph.begin == begin_time
        assert eph.end == end_time
        assert eph.step_size == step_size
        assert eph.norad_cat_id == 25544
