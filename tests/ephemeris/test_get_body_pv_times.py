"""Tests for ``get_body_pv(times=..., apparent=...)`` and ``apparent_target_at``:
the state of a body relative to the observer, and the directions of targets,
at arbitrary times."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    ICRS,
    CartesianRepresentation,
    SkyCoord,
    solar_system_ephemeris,
)
from astropy.coordinates.solar_system import (  # type: ignore[import-untyped]
    _get_apparent_body_position,
)
from astropy.time import Time  # type: ignore[import-untyped]

import rust_ephem
from rust_ephem.constraints import (
    EarthLimbConstraint,
    MoonConstraint,
    SunConstraint,
)

KERNEL = Path(__file__).parents[1] / "test_data" / "de440s.bsp"

VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
STEP = 60
EARTH_RADIUS_KM = 6378.137
# Between grid rows, on a grid row, and out of order.
TIMES = [
    BEGIN + timedelta(seconds=1830.5),
    BEGIN + timedelta(seconds=120),
    BEGIN + timedelta(seconds=90.25),
    END,
]
RAS = [10.0, 123.4, 279.5, 359.9]
DECS = [5.0, -62.0, 88.0, -0.1]

Vectors = npt.NDArray[np.float64]
Observer = (
    rust_ephem.TLEEphemeris | rust_ephem.GroundEphemeris | rust_ephem.FileEphemeris
)


@pytest.fixture
def tle(ensure_planetary_data: None) -> rust_ephem.TLEEphemeris:
    return rust_ephem.TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN, END, STEP)


@pytest.fixture
def ground(ensure_planetary_data: None) -> rust_ephem.GroundEphemeris:
    return rust_ephem.GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN,
        end=END,
        step_size=STEP,
    )


@pytest.fixture
def distant(tmp_path: Path, ensure_planetary_data: None) -> rust_ephem.FileEphemeris:
    """A straight, fast trajectory far from the Earth: interpolation between
    grid rows is exact, so off-grid results test the corrections alone."""
    r = np.array([150000.0, -80000.0, 60000.0])
    v = np.array([1.2, 6.8, -2.4])
    lines = [f"ScenarioEpoch {BEGIN.isoformat()}", "CoordinateSystem GCRS"]
    for seconds in (0, 3600):
        lines.append(f"{seconds} " + " ".join(map(str, [*(r + seconds * v), *v])))
    path = tmp_path / "observer.txt"
    path.write_text("\n".join(lines) + "\n")
    return rust_ephem.FileEphemeris(str(path), begin=BEGIN, end=END, step_size=STEP)


@pytest.fixture(params=["tle", "ground", "distant"])
def ephem(request: pytest.FixtureRequest) -> Observer:
    observer: Observer = request.getfixturevalue(request.param)
    return observer


def angle_arcsec(a: Vectors, b: Vectors) -> Vectors:
    """Angle between rows, accurate for nearly parallel vectors."""
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    arcsec: Vectors = np.degrees(np.arctan2(cross, np.sum(a * b, axis=1))) * 3600
    return arcsec


def astropy_frame(ephem: Observer, times: Sequence[datetime]) -> GCRS:
    """The GCRS frame of the observer's own interpolated states, so the
    comparison isolates the apparent-place corrections."""
    state = ephem.gcrs_pv_at(times)
    return GCRS(
        obstime=Time(times),
        obsgeoloc=CartesianRepresentation(state.position.T * u.km),
        obsgeovel=CartesianRepresentation(state.velocity.T * u.km / u.s),
    )


@pytest.mark.parametrize(
    "cls",
    [
        rust_ephem.TLEEphemeris,
        rust_ephem.SPICEEphemeris,
        rust_ephem.GroundEphemeris,
        rust_ephem.OEMEphemeris,
        rust_ephem.FileEphemeris,
        rust_ephem.ParquetEphemeris,
    ],
)
def test_every_ephemeris_type_has_the_methods(cls: type) -> None:
    assert hasattr(cls, "apparent_target_at")


def test_geometric_is_body_minus_observer(ephem: Observer) -> None:
    observer = ephem.gcrs_pv_at(TIMES)
    for body, held in [
        ("sun", ephem.sun_pv_at(TIMES)),
        ("moon", ephem.moon_pv_at(TIMES)),
    ]:
        state = ephem.get_body_pv(body, times=TIMES)
        np.testing.assert_array_equal(state.position, held.position - observer.position)
        np.testing.assert_array_equal(state.velocity, held.velocity - observer.velocity)
    earth = ephem.get_body_pv("earth", times=TIMES, apparent=False)
    np.testing.assert_array_equal(earth.position, -observer.position)
    np.testing.assert_array_equal(earth.velocity, -observer.velocity)


@pytest.mark.parametrize("body", ["sun", "moon", "earth", "5"])
@pytest.mark.parametrize("apparent", [False, True])
def test_grid_times_give_the_grid_result(
    ephem: Observer, body: str, apparent: bool
) -> None:
    on_grid = ephem.get_body_pv(body, apparent=apparent)
    at_times = ephem.get_body_pv(body, times=ephem.timestamp, apparent=apparent)
    np.testing.assert_allclose(at_times.position, on_grid.position, rtol=0, atol=1e-6)
    np.testing.assert_allclose(at_times.velocity, on_grid.velocity, rtol=0, atol=1e-9)


def test_explicit_kernel_is_used_at_arbitrary_times(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    for apparent in (False, True):
        default = tle.get_body_pv("moon", times=TIMES, apparent=apparent)
        explicit = tle.get_body_pv(
            "moon", spice_kernel=str(KERNEL), times=TIMES, apparent=apparent
        )
        np.testing.assert_allclose(
            explicit.position, default.position, rtol=0, atol=1e-3
        )
        np.testing.assert_allclose(
            explicit.velocity, default.velocity, rtol=0, atol=1e-4
        )


@pytest.mark.parametrize(
    "name,alias", [("sun", "10"), ("Moon", "301"), ("EARTH", "399")]
)
@pytest.mark.parametrize("apparent", [False, True])
def test_names_and_naif_ids_agree(
    tle: rust_ephem.TLEEphemeris, name: str, alias: str, apparent: bool
) -> None:
    np.testing.assert_array_equal(
        tle.get_body_pv(name, times=TIMES, apparent=apparent).position,
        tle.get_body_pv(alias, times=TIMES, apparent=apparent).position,
    )


def test_geometric_kernel_body_matches_get_body(tle: rust_ephem.TLEEphemeris) -> None:
    grid = [BEGIN + timedelta(seconds=STEP * i) for i in (0, 7, 60)]
    expected = tle.get_body("5").cartesian.xyz.to_value(u.km).T[[0, 7, 60]]
    np.testing.assert_allclose(
        tle.get_body_pv("5", times=grid).position, expected, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("body", ["sun", "moon", "earth", "5"])
def test_apparent_matches_get_body_on_the_grid(ephem: Observer, body: str) -> None:
    rows = [0, 7, 31, 60]
    grid = [BEGIN + timedelta(seconds=STEP * i) for i in rows]
    expected = ephem.get_body(body, apparent=True).cartesian.xyz.to_value(u.km).T[rows]
    actual = ephem.get_body_pv(body, times=grid, apparent=True).position
    # The two solve the light time differently (first order in the states held
    # by the ephemeris, against iterated kernel lookups): centimetres, which
    # for the nearby Earth is the larger part of the angle allowed here.
    assert np.max(angle_arcsec(actual, expected)) < 1e-4
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "body,reference_body",
    [("sun", "sun"), ("moon", "moon"), ("earth", "earth"), ("5", "jupiter")],
)
def test_apparent_matches_astropy_off_the_grid(
    ephem: Observer, body: str, reference_body: str
) -> None:
    pytest.importorskip("jplephem")
    frame = astropy_frame(ephem, TIMES)
    with solar_system_ephemeris.set(str(KERNEL)):
        emitted = _get_apparent_body_position(
            reference_body, frame.obstime, str(KERNEL), frame.obsgeoloc
        )
        expected = SkyCoord(ICRS(emitted).transform_to(frame))
    actual = ephem.get_body_pv(body, times=TIMES, apparent=True).position
    reference = expected.cartesian.xyz.to_value(u.km).T
    # At the Earth's distance from a ground site or low orbit, a metre of
    # difference between the two kernel readers is already a few
    # milliarcseconds; elsewhere the directions agree far more closely.
    limit = 2e-2 if body == "earth" else 1e-4
    assert np.max(angle_arcsec(actual, reference)) < limit
    np.testing.assert_allclose(
        np.linalg.norm(actual, axis=1),
        np.linalg.norm(reference, axis=1),
        rtol=0,
        atol=1e-3,
    )


def test_apparent_differs_from_geometric_by_aberration(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    shift = angle_arcsec(
        tle.get_body_pv("sun", times=TIMES, apparent=True).position,
        tle.get_body_pv("sun", times=TIMES).position,
    )
    assert np.all((shift > 10) & (shift < 30))


def test_apparent_target_matches_astropy(ephem: Observer) -> None:
    frame = astropy_frame(ephem, TIMES)
    expected = (
        SkyCoord(RAS * u.deg, DECS * u.deg, frame="icrs")
        .transform_to(frame)
        .cartesian.xyz.value.T
    )
    actual = ephem.apparent_target_at(RAS, DECS, TIMES)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, rtol=0, atol=1e-15)
    assert np.max(angle_arcsec(actual, expected)) < 1e-4


def test_angles_between_the_vectors_are_the_constraint_angles(
    ephem: Observer,
) -> None:
    targets = ephem.apparent_target_at(RAS, DECS, TIMES)

    def separation_deg(body: str) -> Vectors:
        position = ephem.get_body_pv(body, times=TIMES, apparent=True).position
        degrees: Vectors = angle_arcsec(position, targets) / 3600
        return degrees

    def reported(
        constraint: SunConstraint | MoonConstraint | EarthLimbConstraint, key: str
    ) -> Vectors:
        result = constraint.evaluate_elementwise(
            ephem, RAS, DECS, times=TIMES, apparent=True
        )
        return np.asarray(result.constraint_values[key], dtype=np.float64)

    np.testing.assert_allclose(
        separation_deg("sun"),
        reported(SunConstraint(min_angle=45), "sun_angle_deg"),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        separation_deg("moon"),
        reported(MoonConstraint(min_angle=10), "moon_angle_deg"),
        rtol=0,
        atol=1e-10,
    )
    # The limb angle is measured from the Earth's edge, sized from the
    # observer's geometric distance.
    distance = np.linalg.norm(ephem.get_body_pv("earth", times=TIMES).position, axis=1)
    radius_deg = np.degrees(np.arcsin(np.minimum(EARTH_RADIUS_KM / distance, 1.0)))
    np.testing.assert_allclose(
        separation_deg("earth") - radius_deg,
        reported(EarthLimbConstraint(min_angle=10), "earth_limb_angle_deg"),
        rtol=0,
        atol=1e-10,
    )


def test_time_inputs(tle: rust_ephem.TLEEphemeris) -> None:
    expected = tle.get_body_pv("sun", times=TIMES, apparent=True).position
    as_datetime64 = np.array(
        [t.replace(tzinfo=None) for t in TIMES], dtype="datetime64[us]"
    )
    np.testing.assert_array_equal(
        tle.get_body_pv("sun", times=as_datetime64, apparent=True).position, expected
    )
    single = tle.get_body_pv("sun", times=TIMES[0], apparent=True).position
    assert single.shape == (1, 3)
    # The Sun's barycentric velocity is interpolated across the span of the
    # request, so a row asked for alone agrees to rounding, not bit for bit.
    np.testing.assert_allclose(single[0], expected[0], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(
        tle.apparent_target_at(RAS, DECS, as_datetime64),
        tle.apparent_target_at(RAS, DECS, TIMES),
    )


def test_apparent_target_broadcasts_single_values(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    paired = tle.apparent_target_at(RAS, DECS, TIMES)
    one_target = tle.apparent_target_at(RAS[1], DECS[1], TIMES)
    assert one_target.shape == (4, 3)
    np.testing.assert_array_equal(one_target[1], paired[1])
    one_time = tle.apparent_target_at(np.array(RAS), np.array(DECS), TIMES[2])
    assert one_time.shape == (4, 3)
    np.testing.assert_array_equal(one_time[2], paired[2])
    assert tle.apparent_target_at(RAS[0], DECS[0], TIMES[0]).shape == (1, 3)


def test_invalid_requests(tle: rust_ephem.TLEEphemeris) -> None:
    with pytest.raises(ValueError, match="outside ephemeris range"):
        tle.get_body_pv("sun", times=BEGIN - timedelta(seconds=1))
    with pytest.raises(ValueError, match="outside ephemeris range"):
        tle.apparent_target_at(RAS, DECS, END + timedelta(seconds=1))
    with pytest.raises(ValueError, match="Unknown body"):
        tle.get_body_pv("not a body", times=TIMES)
    with pytest.raises(ValueError, match="Unknown body"):
        tle.get_body_pv("not a body", times=TIMES, apparent=True)
    with pytest.raises(ValueError, match="expected 1 or 4"):
        tle.apparent_target_at(RAS[:2], DECS, TIMES)
    with pytest.raises(ValueError, match="expected 1 or 4"):
        tle.apparent_target_at(RAS, DECS, TIMES[:3])


@pytest.mark.parametrize("dec", [90.0, -90.0])
def test_the_poles_are_valid_declinations(
    tle: rust_ephem.TLEEphemeris, dec: float
) -> None:
    target = tle.apparent_target_at(10.0, dec, TIMES)
    np.testing.assert_allclose(np.linalg.norm(target, axis=1), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("ras", "decs", "message"),
    [
        (5.0, 95.0, r"decs\[0\] is 95; it must be between -90 and 90 degrees"),
        ([5.0, 6.0], [10.0, -90.5], r"decs\[1\] is -90.5"),
        (5.0, float("nan"), r"decs\[0\] is NaN"),
        (float("nan"), 5.0, r"ras\[0\] is NaN; it must be finite"),
        ([1.0, float("inf")], 5.0, r"ras\[1\] is inf"),
    ],
)
def test_apparent_target_rejects_invalid_coordinates(
    tle: rust_ephem.TLEEphemeris,
    ras: float | list[float],
    decs: float | list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tle.apparent_target_at(ras, decs, TIMES[0])


def test_right_ascension_is_not_range_limited(tle: rust_ephem.TLEEphemeris) -> None:
    np.testing.assert_allclose(
        tle.apparent_target_at(370.0, 5.0, TIMES),
        tle.apparent_target_at(10.0, 5.0, TIMES),
        rtol=0,
        atol=1e-12,
    )


def test_more_than_one_dimension_is_rejected(tle: rust_ephem.TLEEphemeris) -> None:
    grid = np.full((2, 2), 5.0)
    naive = [time.replace(tzinfo=None) for time in TIMES]
    times_2d = np.array(naive, dtype="datetime64[us]").reshape(2, 2)
    with pytest.raises(ValueError, match="ras must be a single value or one-dim"):
        tle.apparent_target_at(grid, 5.0, TIMES)
    with pytest.raises(ValueError, match="decs must be a single value or one-dim"):
        tle.apparent_target_at(5.0, grid, TIMES)
    with pytest.raises(ValueError, match="times must be a single value or one-dim"):
        tle.apparent_target_at(5.0, 5.0, times_2d)
    with pytest.raises(ValueError, match="times must be a single value or one-dim"):
        tle.get_body_pv("sun", times=times_2d)
    with pytest.raises(ValueError, match="not 2-dimensional"):
        tle.get_body_pv("sun", times=times_2d, apparent=True)
    # Zero- and one-dimensional arrays are fine.
    assert tle.apparent_target_at(np.float64(5.0), np.array(5.0), TIMES).shape == (4, 3)
    assert tle.get_body_pv("sun", times=times_2d.ravel()).position.shape == (4, 3)


def test_get_body_pv_is_declared_on_the_base_class() -> None:
    assert "get_body_pv" in rust_ephem.Ephemeris.__abstractmethods__
    assert "apparent_target_at" in rust_ephem.Ephemeris.__abstractmethods__


# Mid-interval, so a one-second difference does not straddle a grid row, where
# the interpolated observer's acceleration steps.
INSIDE = [BEGIN + timedelta(seconds=x) for x in (30.0, 630.5, 1830.0, 3570.25)]


@pytest.mark.parametrize("body", ["sun", "moon", "earth", "5"])
def test_apparent_velocity_is_the_rate_of_the_apparent_position(
    ephem: Observer, body: str
) -> None:
    half = timedelta(seconds=1)
    later = ephem.get_body_pv(body, times=[t + half for t in INSIDE], apparent=True)
    earlier = ephem.get_body_pv(body, times=[t - half for t in INSIDE], apparent=True)
    expected = (later.position - earlier.position) / 2.0
    actual = ephem.get_body_pv(body, times=INSIDE, apparent=True).velocity
    # A one-second difference of a low orbit is itself only good to ~1 cm/s.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize("body", ["sun", "moon", "5"])
def test_apparent_velocity_matches_differenced_get_body(
    tmp_path: Path, ensure_planetary_data: None, body: str
) -> None:
    """Against the kernel-iterated apparent positions of ``get_body``,
    differenced along the grid, for an unaccelerated observer."""
    r = np.array([150000.0, -80000.0, 60000.0])
    v = np.array([1.2, 6.8, -2.4])
    step = 10
    end = BEGIN + timedelta(seconds=200)
    lines = [f"ScenarioEpoch {BEGIN.isoformat()}", "CoordinateSystem GCRS"]
    for seconds in (0, 200):
        lines.append(f"{seconds} " + " ".join(map(str, [*(r + seconds * v), *v])))
    path = tmp_path / "observer.txt"
    path.write_text("\n".join(lines) + "\n")
    ephem = rust_ephem.FileEphemeris(str(path), begin=BEGIN, end=end, step_size=step)
    position = ephem.get_body(body, apparent=True).cartesian.xyz.to_value(u.km).T
    expected = (position[2:] - position[:-2]) / (2.0 * step)
    actual = ephem.get_body_pv(body, apparent=True).velocity
    np.testing.assert_allclose(actual[1:-1], expected, rtol=0, atol=1e-6)
    # The one-sided stencils at the ends of the span continue the interior.
    np.testing.assert_allclose(actual[0], actual[1], rtol=0, atol=1e-4)
    np.testing.assert_allclose(actual[-1], actual[-2], rtol=0, atol=1e-4)


def test_apparent_velocity_is_not_the_geometric_velocity(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    """From low orbit the Sun's aberration shift (thousands of km at 1 au)
    swings round with the orbital velocity, at kilometres per second."""
    apparent = tle.get_body_pv("sun", times=INSIDE, apparent=True).velocity
    geometric = tle.get_body_pv("sun", times=INSIDE).velocity
    assert np.all(np.linalg.norm(apparent - geometric, axis=1) > 1.0)


def test_apparent_velocity_at_the_ends_of_the_span(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    """No sample may fall outside the span, so the ends use one-sided
    differences; check them against one-second ones."""
    for edge, sign in ((BEGIN, 1), (END, -1)):
        samples = [edge + timedelta(seconds=sign * k) for k in (0, 1, 2)]
        state = tle.get_body_pv("moon", times=samples, apparent=True)
        p0, p1, p2 = state.position
        expected = sign * (-3 * p0 + 4 * p1 - p2) / 2.0
        np.testing.assert_allclose(state.velocity[0], expected, rtol=0, atol=2e-5)


def test_apparent_rejects_horizons(tle: rust_ephem.TLEEphemeris) -> None:
    with pytest.raises(ValueError, match="use_horizons=True is not supported"):
        tle.get_body_pv("moon", use_horizons=True, apparent=True)


@pytest.mark.parametrize("apparent", [False, True])
@pytest.mark.parametrize("at_times", [False, True])
def test_several_bodies_are_each_what_one_call_returns(
    ephem: Observer, apparent: bool, at_times: bool
) -> None:
    """Held bodies and one read from the kernel, in an order that is not the
    order they are solved in."""
    names = ["Moon", "venus", "earth", "sun"]
    times = TIMES if at_times else None
    several = ephem.get_body_pv(names, times=times, apparent=apparent)
    assert list(several) == names
    for name in names:
        alone = ephem.get_body_pv(name, times=times, apparent=apparent)
        np.testing.assert_array_equal(several[name].position, alone.position)
        np.testing.assert_array_equal(
            several[name].velocity, alone.velocity, strict=True
        )
        assert several[name].position.shape == alone.position.shape
    as_tuple = ephem.get_body_pv(tuple(names), times=times, apparent=apparent)
    np.testing.assert_array_equal(as_tuple["sun"].position, several["sun"].position)


def test_several_bodies_with_a_named_kernel(tle: rust_ephem.TLEEphemeris) -> None:
    several = tle.get_body_pv(["sun", "moon"], str(KERNEL), times=TIMES)
    for name in ("sun", "moon"):
        alone = tle.get_body_pv(name, str(KERNEL), times=TIMES)
        np.testing.assert_array_equal(several[name].position, alone.position)
        np.testing.assert_array_equal(several[name].velocity, alone.velocity)


def test_one_body_in_a_list_is_still_a_dict(tle: rust_ephem.TLEEphemeris) -> None:
    result = tle.get_body_pv(["sun"], times=TIMES)
    assert list(result) == ["sun"]
    assert tle.get_body_pv([], times=TIMES) == {}
    with pytest.raises(TypeError, match="name or NAIF ID string, or a sequence"):
        tle.get_body_pv(10, times=TIMES)  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        tle.get_body_pv(["sun", "not a body"], times=TIMES)


@pytest.mark.parametrize("apparent", [False, True])
def test_ra_dec_and_distance_are_those_of_the_position(
    ephem: Observer, apparent: bool
) -> None:
    for name in ("sun", "moon", "earth"):
        state = ephem.get_body_pv(name, times=TIMES, apparent=apparent)
        position = state.position
        distance = np.linalg.norm(position, axis=1)
        np.testing.assert_allclose(state.distance, distance, rtol=1e-15)
        assert ((state.ra_deg >= 0.0) & (state.ra_deg < 360.0)).all()
        ra, dec = np.radians(state.ra_deg), np.radians(state.dec_deg)
        direction = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        np.testing.assert_allclose(
            direction, position / distance[:, None], rtol=0, atol=1e-14
        )
        assert state.ra_deg.shape == state.dec_deg.shape == (len(TIMES),)


def test_ra_dec_of_the_sun_match_the_ephemeris_properties(
    tle: rust_ephem.TLEEphemeris,
) -> None:
    """On the grid, ``get_body_pv("sun")`` is the Sun the ephemeris reports."""
    state = tle.get_body_pv("sun")
    np.testing.assert_allclose(state.ra_deg, tle.sun_ra_deg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(state.dec_deg, tle.sun_dec_deg, rtol=0, atol=1e-9)
