"""Fixtures for swift_visibility tests."""

import datetime

import pytest

from rust_ephem._rust_ephem import TLEEphemeris
from rust_ephem.constraints import (
    EarthLimbConstraint,
    MoonConstraint,
    OrbitPoleConstraint,
    OrConstraint,
    SAAConstraint,
    SunConstraint,
)
from rust_ephem.tle import TLERecord


@pytest.fixture
def swift_saa_polygon() -> list[tuple[float, float]]:
    return [
        (39.0, -30.0),
        (36.0, -26.0),
        (28.0, -21.0),
        (6.0, -12.0),
        (-5.0, -6.0),
        (-21.0, 2.0),
        (-30.0, 3.0),
        (-45.0, 2.0),
        (-60.0, -2.0),
        (-75.0, -7.0),
        (-83.0, -10.0),
        (-87.0, -16.0),
        (-86.0, -23.0),
        (-83.0, -30.0),
    ]


@pytest.fixture
def swift_saa_constraint(swift_saa_polygon: list[tuple[float, float]]) -> SAAConstraint:
    return SAAConstraint(polygon=swift_saa_polygon)


@pytest.fixture
def swift_sun_constraint() -> SunConstraint:
    return SunConstraint(min_angle=46 + 1)


@pytest.fixture
def swift_moon_constraint() -> MoonConstraint:
    return MoonConstraint(min_angle=22 + 1)


@pytest.fixture
def swift_earth_constraint() -> EarthLimbConstraint:
    return EarthLimbConstraint(min_angle=28 + 5)


@pytest.fixture
def swift_pole_constraintt() -> OrbitPoleConstraint:
    return OrbitPoleConstraint(min_angle=28 + 5, earth_limb_pole=True)


@pytest.fixture()
def swift_tle() -> TLERecord:
    return TLERecord(
        line1="1 28485U 04047A   25342.87828629  .00042565  00000-0  70967-3 0  9992",
        line2="2 28485  20.5524 193.5008 0003262 214.0798 145.9435 15.50178289157119",
        name=None,
        epoch=datetime.datetime(2025, 12, 8, 21, 4, 43, tzinfo=datetime.timezone.utc),
        source="spacetrack",
    )


@pytest.fixture
def begin() -> datetime.datetime:
    return datetime.datetime(2025, 12, 9)


@pytest.fixture
def end() -> datetime.datetime:
    return datetime.datetime(2025, 12, 10)


@pytest.fixture
def swift_constraint(
    swift_sun_constraint: SunConstraint,
    swift_moon_constraint: MoonConstraint,
    swift_earth_constraint: EarthLimbConstraint,
    swift_saa_constraint: SAAConstraint,
) -> OrConstraint:
    return (
        swift_sun_constraint
        | swift_moon_constraint
        | swift_earth_constraint
        | swift_saa_constraint
    )


@pytest.fixture
def swift_ephemeris(
    swift_tle: TLERecord, begin: datetime.datetime, end: datetime.datetime
) -> TLEEphemeris:
    return TLEEphemeris(tle=swift_tle, begin=begin, end=end, step_size=60)
