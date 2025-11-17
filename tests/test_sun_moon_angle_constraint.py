from datetime import datetime, timezone

import numpy as np
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
def tle() -> tuple:
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    return tle1, tle2


@pytest.fixture
def begin_end_step_size() -> tuple[datetime, datetime, int]:
    # Time span and step
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
    step_s = 60  # 1 minute
    return begin, end, step_s


@pytest.fixture
def tle_ephem(tle, begin_end_step_size):
    yield rust_ephem.TLEEphemeris(
        tle[0],
        tle[1],
        begin_end_step_size[0],
        begin_end_step_size[1],
        begin_end_step_size[2],
    )


@pytest.fixture
def sun(tle_ephem) -> SkyCoord:
    return tle_ephem.sun[183]


@pytest.fixture
def moon(tle_ephem) -> SkyCoord:
    return tle_ephem.moon[183]


@pytest.fixture
def earth(tle_ephem) -> SkyCoord:
    return tle_ephem.earth[183]


@pytest.fixture
def timestamp(tle_ephem) -> datetime:
    return tle_ephem.timestamp[183]


@pytest.fixture
def sun_constraint() -> SunConstraint:
    return SunConstraint(min_angle=45)


@pytest.fixture
def moon_constraint() -> MoonConstraint:
    return MoonConstraint(min_angle=21)


@pytest.fixture
def earth_limb_constraint() -> EarthLimbConstraint:
    return EarthLimbConstraint(min_angle=21)


@pytest.fixture
def eclipse_constraint() -> EclipseConstraint:
    return EclipseConstraint()


class TestSunConstraints:
    def test_sun_is_near_zero_declination(self, sun):
        assert sun.dec.deg == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_sun_is_sun_in_constraint(
        self, sun_constraint, tle_ephem, timestamp, sun, offset
    ):
        not_vis = sun_constraint.in_constraint(
            ephemeris=tle_ephem,
            time=timestamp,
            target_ra=(sun.ra.deg + offset) % 360,
            target_dec=sun.dec.deg,
        )
        if offset < 45:
            assert not_vis is True, "Sun should be Sun Constrained"
        else:
            assert not_vis is False, "Sun should not be Sun Constrained"

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_sun_is_sun_evaluated_bad(
        self, sun_constraint, tle_ephem, timestamp, sun, offset
    ):
        not_vis = sun_constraint.evaluate(
            ephemeris=tle_ephem,
            times=[timestamp],
            target_ra=(sun.ra.deg + offset) % 360,
            target_dec=sun.dec.deg,
        ).constraint_array[0]
        if offset < 45:
            assert not_vis is True, "Sun should be Sun Constrained"
        else:
            assert not_vis is False, "Sun should not be Sun Constrained"

    def test_or_constraints_sun_angle_greater_than_90(self, sun):
        sun_angle = sun.separation(SkyCoord(10, 0, unit="deg")).deg
        assert sun_angle > 90, f"Sun is too close, {sun.ra} {sun.dec}"

    def test_or_constraints_sun_not_in_constraint(self, sun_constraint, tle_ephem):
        sun_cons = sun_constraint.evaluate(
            ephemeris=tle_ephem, target_ra=0, target_dec=0
        )
        assert True not in sun_cons.constraint_array, "Source is Sun constrained"


class TestMoonConstraints:
    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_moon_is_moon_in_constraint(
        self, moon_constraint, tle_ephem, timestamp, moon, offset
    ):
        not_vis = moon_constraint.in_constraint(
            ephemeris=tle_ephem,
            time=timestamp,
            target_ra=(moon.ra.deg + offset) % 360,
            target_dec=moon.dec.deg,
        )
        if offset < 21:
            assert not_vis is True, "Moon should be Moon Constrained"
        else:
            assert not_vis is False, "Moon should not be Moon Constrained"

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_moon_is_moon_in_constraint_vs_distance(
        self, moon_constraint, tle_ephem, timestamp, moon, offset
    ):
        not_vis = moon_constraint.in_constraint(
            ephemeris=tle_ephem,
            time=timestamp,
            target_ra=(moon.ra.deg + offset) % 360,
            target_dec=moon.dec.deg,
        )
        in_moon_cons = (
            SkyCoord(moon.ra.deg, moon.dec.deg, unit="deg")
            .separation(SkyCoord(moon.ra.deg + offset, moon.dec.deg, unit="deg"))
            .deg
            < 21
        )
        assert in_moon_cons == not_vis

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_moon_is_moon_evaluated_bad(
        self, moon_constraint, tle_ephem, timestamp, moon, offset
    ):
        not_vis = moon_constraint.evaluate(
            ephemeris=tle_ephem,
            times=[timestamp],
            target_ra=(moon.ra.deg + offset) % 360,
            target_dec=moon.dec.deg,
        ).constraint_array[0]
        moon_angle = (
            SkyCoord(moon.ra.deg, moon.dec.deg, unit="deg")
            .separation(SkyCoord(moon.ra.deg + offset, moon.dec.deg, unit="deg"))
            .deg
        )
        in_moon_cons = moon_angle < 21
        assert in_moon_cons == not_vis, f"Moon is at {moon_angle:.1f} degrees"


class TestEarthLimbConstraints:
    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_earth_is_earth_in_constraint_vs_distance(
        self, earth_limb_constraint, tle_ephem, timestamp, earth, offset
    ):
        not_vis = earth_limb_constraint.in_constraint(
            ephemeris=tle_ephem,
            time=timestamp,
            target_ra=(earth.ra.deg + offset) % 360,
            target_dec=earth.dec.deg,
        )
        earth_angle = (
            SkyCoord(earth.ra.deg, earth.dec.deg, unit="deg")
            .separation(SkyCoord(earth.ra.deg + offset, earth.dec.deg, unit="deg"))
            .deg
        )
        in_earth_cons = earth_angle < tle_ephem.earth_radius_deg[0] + 21
        assert in_earth_cons == not_vis, f"Earth is at {earth_angle:.1f} degrees"

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_earth_is_earth_evaluated_bad(
        self, earth_limb_constraint, tle_ephem, timestamp, earth, offset
    ):
        not_vis = earth_limb_constraint.evaluate(
            ephemeris=tle_ephem,
            times=[timestamp],
            target_ra=(earth.ra.deg + offset) % 360,
            target_dec=earth.dec.deg,
        ).constraint_array[0]
        earth_angle = (
            SkyCoord(earth.ra.deg, earth.dec.deg, unit="deg")
            .separation(SkyCoord(earth.ra.deg + offset, earth.dec.deg, unit="deg"))
            .deg
        )
        in_earth_cons = earth_angle < tle_ephem.earth_radius_deg[0] + 21
        assert in_earth_cons == not_vis, f"Earth is at {earth_angle:.1f} degrees"


class TestEclipseConstraints:
    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 100, 10)],
    )
    def test_eclipse_in_constraint_vs_distance(
        self, eclipse_constraint, tle_ephem, offset
    ):
        not_vis = eclipse_constraint.in_constraint(
            ephemeris=tle_ephem,
            time=tle_ephem.timestamp[offset],
            target_ra=0,
            target_dec=0,
        )
        earth_sun_angle = tle_ephem.earth[offset].separation(tle_ephem.sun[offset]).deg
        in_eclipse = earth_sun_angle < tle_ephem.earth_radius_deg[0]
        assert in_eclipse == not_vis, (
            f"Earth/Sun angle is at {earth_sun_angle:.1f} degrees"
        )

    @pytest.mark.parametrize(
        "offset",
        [x for x in range(0, 190, 10)],
    )
    def test_eclipse_evaluated_bad(self, eclipse_constraint, tle_ephem, offset):
        not_vis = eclipse_constraint.evaluate(
            ephemeris=tle_ephem,
            times=[tle_ephem.timestamp[offset]],
            target_ra=0,
            target_dec=0,
        ).constraint_array[0]
        earth_sun_angle = tle_ephem.earth[offset].separation(tle_ephem.sun[offset]).deg
        in_eclipse = earth_sun_angle < tle_ephem.earth_radius_deg[0]
        assert in_eclipse == not_vis, (
            f"Earth/Sun angle is at {earth_sun_angle:.1f} degrees"
        )


class TestOrConstraints:
    def test_or_constraints_constraint_array_truths_equal(
        self, earth_limb_constraint, sun_constraint, tle_ephem
    ):
        earth_cons = earth_limb_constraint.evaluate(
            ephemeris=tle_ephem, target_ra=250, target_dec=-23
        )
        sun_earth = earth_limb_constraint | sun_constraint
        sun_earth_cons = sun_earth.evaluate(
            ephemeris=tle_ephem, target_ra=0, target_dec=0
        )

        earth_any = np.array(earth_cons.constraint_array).any()
        sun_earth_any = np.array(sun_earth_cons.constraint_array).any()

        assert earth_any == sun_earth_any, "Both arrays should have a True in them"

    def test_or_constraints_visibility_length_equal(
        self, earth_limb_constraint, sun_constraint, tle_ephem
    ):
        earth_cons = earth_limb_constraint.evaluate(
            ephemeris=tle_ephem, target_ra=250, target_dec=-23
        )
        sun_earth = earth_limb_constraint | sun_constraint
        sun_earth_cons = sun_earth.evaluate(
            ephemeris=tle_ephem, target_ra=0, target_dec=0
        )

        assert len(earth_cons.visibility) == len(sun_earth_cons.visibility), (
            "Both arrays should be equal"
        )


class TestAndConstraints:
    def test_and_constraints_constraint_array_truths_equal(
        self, earth_limb_constraint, sun_constraint, tle_ephem
    ):
        """
        Test that a constraint that is always False makes one with True in
        it turn all False when ANDed together.
        """
        sun_cons = sun_constraint.evaluate(
            ephemeris=tle_ephem, target_ra=250, target_dec=-23
        )
        sun_earth = earth_limb_constraint & sun_constraint
        sun_earth_cons = sun_earth.evaluate(
            ephemeris=tle_ephem, target_ra=0, target_dec=0
        )

        sun_any = np.array(sun_cons.constraint_array).any()
        sun_earth_any = np.array(sun_earth_cons.constraint_array).any()

        assert sun_any == sun_earth_any, "Both arrays should have a True in them"

    def test_and_constraints_visibility_length_equal(
        self, earth_limb_constraint, sun_constraint, tle_ephem
    ):
        """Test and the constraint ANDed with itself produces the same result"""
        earth_cons = earth_limb_constraint.evaluate(
            ephemeris=tle_ephem, target_ra=250, target_dec=-23
        )
        earth_earth = earth_limb_constraint & earth_limb_constraint
        earth_earth_cons = earth_earth.evaluate(
            ephemeris=tle_ephem, target_ra=0, target_dec=0
        )

        assert len(earth_cons.visibility) == len(earth_earth_cons.visibility), (
            "Both arrays should be equal"
        )
