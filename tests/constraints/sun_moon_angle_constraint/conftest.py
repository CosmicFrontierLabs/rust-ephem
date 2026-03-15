"""Fixtures for sun_moon_angle_constraint tests."""

from datetime import datetime, timezone
from typing import Any, Generator

import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import SkyCoord  # type:ignore[import-untyped]

import rust_ephem
from rust_ephem.constraints import (
    EarthLimbConstraint,
    EclipseConstraint,
    MoonConstraint,
    SunConstraint,
)

# Constants
EARTH_RADIUS_KM = 6378.137
SUN_RADIUS_KM = 696000.0

MOON_CONSTRAINT = 21
EARTH_CONSTRAINT = 28
SUN_CONSTRAINT = 45


def eclipse_flags(
    obs_pos: npt.NDArray[np.float64], sun_pos: npt.NDArray[np.float64]
) -> tuple[bool, bool]:
    sun_dist = np.linalg.norm(sun_pos)
    if sun_dist <= 0.0:
        return False, False

    sun_unit = sun_pos / sun_dist
    dot = float(np.dot(obs_pos, sun_unit))
    if dot >= 0.0:
        return False, False

    s = -dot
    perp = obs_pos - sun_unit * dot
    dist_to_axis = np.linalg.norm(perp)

    l_umbra = EARTH_RADIUS_KM * sun_dist / (SUN_RADIUS_KM - EARTH_RADIUS_KM)
    l_penumbra = EARTH_RADIUS_KM * sun_dist / (SUN_RADIUS_KM + EARTH_RADIUS_KM)

    umbra_radius = EARTH_RADIUS_KM * (1.0 - s / l_umbra) if s <= l_umbra else 0.0
    penumbra_radius = EARTH_RADIUS_KM * (1.0 + s / l_penumbra)

    in_umbra = umbra_radius > 0.0 and dist_to_axis < umbra_radius
    in_penumbra = dist_to_axis < penumbra_radius

    return bool(in_umbra), bool(in_penumbra)


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
