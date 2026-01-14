"""
Refactored tests into pytest classes for binary search index() method.
Each assert is now its own test (parametrized where appropriate).
Pulled out fixtures for reuse.
"""

from datetime import datetime

import pytest

from rust_ephem import TLEEphemeris

TLE1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
TLE2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"


@pytest.fixture(scope="module")
def eph_small_range():
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return TLEEphemeris(TLE1, TLE2, begin, end, step_size=60)


@pytest.fixture(scope="module")
def eph_large_range():
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 2, 0, 0, 0)
    return TLEEphemeris(TLE1, TLE2, begin, end, step_size=10)


def _naive(dt):
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class TestBinarySearchSmallRange:
    @pytest.mark.parametrize("index", [0, 10, 30, 60])
    def test_exact_match_at_index(self, eph_small_range, index):
        timestamp = eph_small_range.timestamp[index]
        timestamp = _naive(timestamp)
        idx = eph_small_range.index(timestamp)
        assert idx == index, f"Expected index {index}, got {idx}"

    def test_between_timestamps_returns_either_previous_or_next(self, eph_small_range):
        target = datetime(2024, 1, 1, 0, 0, 30)
        idx = eph_small_range.index(target)
        assert idx in [0, 1], f"Expected index 0 or 1, got {idx}"

    def test_before_range_returns_first_index(self, eph_small_range):
        target = datetime(2023, 12, 31, 23, 0, 0)
        idx = eph_small_range.index(target)
        assert idx == 0, f"Expected index 0 for time before range, got {idx}"

    def test_after_range_returns_last_index(self, eph_small_range):
        target = datetime(2024, 1, 1, 2, 0, 0)
        idx = eph_small_range.index(target)
        assert idx == len(eph_small_range.timestamp) - 1, (
            f"Expected last index, got {idx}"
        )

    def test_finds_closest_first_for_10_seconds_after(self, eph_small_range):
        target = datetime(2024, 1, 1, 0, 0, 10)
        idx = eph_small_range.index(target)
        assert idx == 0

    def test_finds_closest_second_for_50_seconds_after(self, eph_small_range):
        target = datetime(2024, 1, 1, 0, 0, 50)
        idx = eph_small_range.index(target)
        assert idx == 1

    def test_finds_either_for_exact_middle(self, eph_small_range):
        target = datetime(2024, 1, 1, 0, 0, 30)
        idx = eph_small_range.index(target)
        assert idx in [0, 1]


class TestBinarySearchLargeRange:
    def test_length_of_timestamps(self, eph_large_range):
        assert len(eph_large_range.timestamp) == 8641

    @pytest.mark.parametrize(
        "target_time, expected_idx",
        [
            (datetime(2024, 1, 1, 0, 0, 0), 0),
            (datetime(2024, 1, 2, 0, 0, 0), 8640),
            (datetime(2024, 1, 1, 12, 0, 0), 4320),
            (datetime(2024, 1, 1, 6, 0, 0), 2160),
            (datetime(2024, 1, 1, 18, 0, 0), 6480),
        ],
    )
    def test_timestamp_positions(self, eph_large_range, target_time, expected_idx):
        idx = eph_large_range.index(target_time)
        assert idx == expected_idx, (
            f"For {target_time}, expected {expected_idx}, got {idx}"
        )
