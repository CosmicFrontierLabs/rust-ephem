"""Fixtures for test_ephemeris_type tests."""

from datetime import datetime, timezone

import pytest

from rust_ephem import OEMEphemeris, SPICEEphemeris

# Test data
BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


@pytest.fixture
def oem_ephemeris(sample_oem_file: str) -> OEMEphemeris:
    """OEMEphemeris instance for testing."""
    return OEMEphemeris(
        sample_oem_file, begin=BEGIN_TIME, end=END_TIME, step_size=STEP_SIZE
    )


@pytest.fixture
def spice_ephemeris(spk_path: str) -> SPICEEphemeris:
    """SPICEEphemeris instance for testing."""
    return SPICEEphemeris(
        spk_path,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
        naif_id=301,
        center_id=399,
    )
