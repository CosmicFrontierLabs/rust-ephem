"""Fixtures for test_spice_ephemeris tests."""

from datetime import datetime, timezone

import pytest

from rust_ephem import SPICEEphemeris

# Test data
BEGIN_TIME = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME_1H = datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
END_TIME_10M = datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc)
STEP_SIZE = 600  # 10 minutes


@pytest.fixture
def spice_ephemeris_1h(spk_path: str) -> SPICEEphemeris:
    """SPICEEphemeris instance for 1 hour duration."""
    return SPICEEphemeris(
        spk_path=spk_path,
        naif_id=301,  # Moon
        begin=BEGIN_TIME,
        end=END_TIME_1H,
        step_size=STEP_SIZE,
        center_id=399,  # Earth
    )


@pytest.fixture
def spice_ephemeris_10m(spk_path: str) -> SPICEEphemeris:
    """SPICEEphemeris instance for 10 minutes duration."""
    return SPICEEphemeris(
        spk_path=spk_path,
        naif_id=301,  # Moon
        begin=BEGIN_TIME,
        end=END_TIME_10M,
        step_size=STEP_SIZE,
        center_id=399,  # Earth
    )
