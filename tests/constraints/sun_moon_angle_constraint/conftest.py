"""Fixtures for sun_moon_angle_constraint tests."""

from datetime import datetime, timezone
from typing import Any, Generator

import pytest
from astropy.coordinates import SkyCoord  # type:ignore[import-untyped]

import rust_ephem
from rust_ephem.constraints import (
    EarthLimbConstraint,
    EclipseConstraint,
    MoonConstraint,
    SunConstraint,
)


@pytest.fixture
def tle() -> tuple[str, str]:
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    return tle1, tle2


@pytest.fixture
def begin_end_step_size() -> tuple[datetime, datetime, int]:
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
    step_s = 60  # 1 minute
    return begin, end, step_s


@pytest.fixture
def tle_ephem(
    tle: tuple[str, str], begin_end_step_size: tuple[datetime, datetime, int]
) -> Generator[rust_ephem.TLEEphemeris, None, None]:
    yield rust_ephem.TLEEphemeris(
        tle[0],
        tle[1],
        begin_end_step_size[0],
        begin_end_step_size[1],
        begin_end_step_size[2],
    )


@pytest.fixture
def sun(tle_ephem: rust_ephem.TLEEphemeris) -> SkyCoord:
    return tle_ephem.sun[183]


@pytest.fixture
def moon(tle_ephem: rust_ephem.TLEEphemeris) -> SkyCoord:
    return tle_ephem.moon[183]


@pytest.fixture
def earth(tle_ephem: rust_ephem.TLEEphemeris) -> SkyCoord:
    return tle_ephem.earth[183]


@pytest.fixture
def timestamp(tle_ephem: rust_ephem.TLEEphemeris) -> Any:
    return tle_ephem.timestamp[183]


@pytest.fixture
def sun_constraint() -> SunConstraint:
    return SunConstraint(min_angle=45)


@pytest.fixture
def moon_constraint() -> MoonConstraint:
    return MoonConstraint(min_angle=21)


@pytest.fixture
def earth_limb_constraint() -> EarthLimbConstraint:
    return EarthLimbConstraint(min_angle=28)


@pytest.fixture
def eclipse_constraint() -> EclipseConstraint:
    return EclipseConstraint()
