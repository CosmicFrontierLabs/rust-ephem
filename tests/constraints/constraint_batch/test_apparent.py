"""Apparent constraint angles against Astropy: the target and the body both in
the directions the moving observer receives their light from."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

import rust_ephem as re

pytest.importorskip("jplephem")
KERNEL = Path(__file__).parents[2] / "test_data" / "de440s.bsp"
BEGIN = datetime(2026, 3, 20, tzinfo=timezone.utc)
STEP = 600
N = 400


@pytest.fixture(scope="module")
def spacecraft(
    tmp_path_factory: pytest.TempPathFactory, ensure_planetary_data: None
) -> Any:
    """A fast, distant observer: 250,000 km out, moving at 3 km/s."""
    r, v = np.array([180000.0, -150000.0, 90000.0]), np.array([1.4, 2.2, -1.5])
    lines = [f"ScenarioEpoch {BEGIN.isoformat()}", "CoordinateSystem GCRS"]
    for seconds in (0, STEP * N):
        lines.append(f"{seconds} " + " ".join(map(str, [*(r + seconds * v), *v])))
    path = tmp_path_factory.mktemp("apparent") / "observer.txt"
    path.write_text("\n".join(lines) + "\n")
    end = BEGIN + timedelta(seconds=STEP * N)
    return re.FileEphemeris(str(path), begin=BEGIN, end=end, step_size=STEP)


def targets(n: int, seed: int = 5) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))


def reference(
    ephem: Any, ras: npt.NDArray[Any], decs: npt.NDArray[Any], rows: npt.NDArray[Any]
) -> dict[str, npt.NDArray[Any]]:
    """Astropy's apparent places for the exact observer states, read
    independently from the same kernel through jplephem."""
    times = Time(ephem.timestamp[rows])
    position = CartesianRepresentation(ephem.obsgeoloc[rows].T * u.km)
    velocity = CartesianRepresentation(ephem.obsgeovel[rows].T * u.km / u.s)
    frame = GCRS(obstime=times, obsgeoloc=position, obsgeovel=velocity)
    target = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs").transform_to(frame)
    out = {}
    with solar_system_ephemeris.set(str(KERNEL)):
        for body in ("sun", "moon", "earth", "jupiter"):
            emitted = _get_apparent_body_position(body, times, str(KERNEL), position)
            apparent = SkyCoord(ICRS(emitted).transform_to(frame))
            out[body] = target.separation(apparent).deg
            out[f"{body}_distance"] = apparent.distance.to_value(u.km)
    return out


def test_apparent_angles_match_astropy(spacecraft: Any) -> None:
    ras, decs = targets(N)
    rows = np.arange(N)
    times = spacecraft.timestamp[rows]
    expected = reference(spacecraft, ras, decs, rows)
    constraint = (
        re.SunConstraint(min_angle=45)
        | re.MoonConstraint(min_angle=10)
        | re.EarthLimbConstraint(min_angle=10)
    )
    result = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=times, apparent=True
    )
    values = result.constraint_values
    tolerance = 1e-3 / 3600  # 1 mas
    np.testing.assert_allclose(
        values["sun.sun_angle_deg"], expected["sun"], atol=tolerance, rtol=0
    )
    np.testing.assert_allclose(
        values["moon.moon_angle_deg"], expected["moon"], atol=tolerance, rtol=0
    )
    earth_radius = np.degrees(
        np.arcsin(6378.137 / np.linalg.norm(spacecraft.obsgeoloc[rows], axis=1))
    )
    np.testing.assert_allclose(
        values["earth_limb.earth_limb_angle_deg"],
        expected["earth"] - earth_radius,
        atol=tolerance,
        rtol=0,
    )

    # The geometric default differs by the aberration it leaves out.
    geometric = constraint.evaluate_elementwise(spacecraft, ras, decs, times=times)
    difference = (
        np.abs(geometric.constraint_values["sun.sun_angle_deg"] - expected["sun"])
        * 3600
    )
    assert 5 < difference.max() < 45


def test_apparent_off_grid_times(spacecraft: Any) -> None:
    """Off-grid times interpolate the observer, then apply the same correction."""
    ras, decs = targets(50, seed=9)
    base = spacecraft.timestamp[:50]
    times = [t + timedelta(seconds=217) for t in base]
    constraint = re.SunConstraint(min_angle=45)
    on = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=base, apparent=True
    ).constraint_values["sun_angle_deg"]
    off = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=times, apparent=True
    ).constraint_values["sun_angle_deg"]
    after = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=spacecraft.timestamp[1:51], apparent=True
    ).constraint_values["sun_angle_deg"]
    # The Sun moves 1 degree per day: 217 s of a 600 s step, linear to well under 1 mas.
    np.testing.assert_allclose(off, on + (after - on) * 217 / 600, atol=1e-6, rtol=0)


@pytest.mark.parametrize("fixed_roll", [False, True])
def test_apparent_batch_equals_elementwise(spacecraft: Any, fixed_roll: bool) -> None:
    ras, decs = targets(40, seed=2)
    rows = np.arange(0, N, 13)
    times = list(spacecraft.timestamp[rows])
    constraint = (
        re.SunConstraint(min_angle=60, max_angle=120)
        | re.MoonConstraint(min_angle=25)
        | re.EarthLimbConstraint(min_angle=20)
    )
    rolls = [15.0] * len(ras) if fixed_roll else None
    grid = constraint.in_constraint_batch(
        spacecraft,
        list(ras),
        list(decs),
        times=times,
        target_rolls=rolls,
        apparent=True,
    )
    assert grid.shape == (len(ras), len(times))
    for column, time in enumerate(times):
        paired = constraint.in_constraint_elementwise(
            spacecraft,
            ras,
            decs,
            times=[time] * len(ras),
            target_rolls=rolls,
            apparent=True,
        )
        np.testing.assert_array_equal(grid[:, column], paired)


def test_apparent_changes_a_boundary_case(spacecraft: Any) -> None:
    """A target a few arcseconds inside the geometric limit flips when the
    apparent angle is used, and only then."""
    ras, decs = targets(2000, seed=11)
    time = spacecraft.timestamp[10]
    probe = re.SunConstraint(min_angle=0.0)
    geometric = probe.evaluate_elementwise(
        spacecraft, ras, decs, times=[time] * len(ras)
    ).constraint_values["sun_angle_deg"]
    apparent = probe.evaluate_elementwise(
        spacecraft, ras, decs, times=[time] * len(ras), apparent=True
    ).constraint_values["sun_angle_deg"]
    i = int(np.argmax(np.abs(apparent - geometric)))
    limit = float((apparent[i] + geometric[i]) / 2)
    constraint = re.SunConstraint(min_angle=limit)

    def one(**kw: Any) -> bool:
        return bool(
            constraint.in_constraint_elementwise(
                spacecraft, [ras[i]], [decs[i]], times=[time], **kw
            )[0]
        )

    assert one() != one(apparent=True)
    assert one(apparent=True) == (apparent[i] < limit)


def test_apparent_defaults_to_geometric(spacecraft: Any) -> None:
    constraint = re.SunConstraint(min_angle=45)
    default = constraint.in_constraint_elementwise(
        spacecraft, [10.0], [20.0], times=[spacecraft.timestamp[0]]
    )
    explicit = constraint.in_constraint_elementwise(
        spacecraft, [10.0], [20.0], times=[spacecraft.timestamp[0]], apparent=False
    )
    np.testing.assert_array_equal(default, explicit)


def test_apparent_batch_kernels_match_astropy(spacecraft: Any) -> None:
    """The grid path corrects targets inside the Sun, Moon and Earth-limb
    kernels; check each against Astropy's apparent angle directly."""
    ras, decs = targets(60, seed=21)
    rows = np.arange(0, N, 7)
    times = list(spacecraft.timestamp[rows])
    margin = 5e-3 / 3600  # cells within 5 mas of a limit may fall either side
    cases = {
        "sun": (re.SunConstraint(min_angle=70, max_angle=115), 70.0, 115.0),
        "moon": (re.MoonConstraint(min_angle=40, max_angle=140), 40.0, 140.0),
        "earth": (re.EarthLimbConstraint(min_angle=30, max_angle=120), 30.0, 120.0),
    }
    for body, (constraint, low, high) in cases.items():
        grid = constraint.in_constraint_batch(
            spacecraft, list(ras), list(decs), times=times, apparent=True
        )
        geometric = constraint.in_constraint_batch(
            spacecraft, list(ras), list(decs), times=times
        )
        for i in range(len(ras)):
            expected = reference(
                spacecraft,
                np.full(len(rows), ras[i]),
                np.full(len(rows), decs[i]),
                rows,
            )
            angle = expected[body]
            if body == "earth":
                angle = angle - np.degrees(
                    np.arcsin(
                        6378.137 / np.linalg.norm(spacecraft.obsgeoloc[rows], axis=1)
                    )
                )
            decided = (np.abs(angle - low) > margin) & (np.abs(angle - high) > margin)
            violated = (angle < low) | (angle > high)
            np.testing.assert_array_equal(grid[i][decided], violated[decided])
        # The correction is the only difference from the geometric grid.
        assert grid.shape == geometric.shape


def test_apparent_evaluate_matches_astropy_and_elementwise(spacecraft: Any) -> None:
    ra, dec = 123.4, -21.0
    rows = np.arange(len(spacecraft.timestamp))
    n = len(rows)
    constraint = re.SunConstraint(min_angle=80) | re.MoonConstraint(min_angle=30)
    result = constraint.evaluate(spacecraft, ra, dec, apparent=True)
    expected = reference(spacecraft, np.full(n, ra), np.full(n, dec), rows)
    values = result.constraint_values
    np.testing.assert_allclose(
        values["sun.sun_angle_deg"], expected["sun"], atol=1e-3 / 3600, rtol=0
    )
    np.testing.assert_allclose(
        values["moon.moon_angle_deg"], expected["moon"], atol=1e-3 / 3600, rtol=0
    )
    paired = constraint.evaluate_elementwise(
        spacecraft,
        np.full(n, ra),
        np.full(n, dec),
        times=spacecraft.timestamp,
        apparent=True,
    )
    np.testing.assert_array_equal(result.constraint_array, paired.constraint_array)
    assert len(result.timestamps) == n
    # The default still reports the geometric angles.
    geometric = constraint.evaluate(spacecraft, ra, dec).constraint_values
    assert np.abs(geometric["sun.sun_angle_deg"] - expected["sun"]).max() > 1 / 3600


@pytest.mark.parametrize("fixed_roll", [False, True])
def test_apparent_evaluate_batch_equals_evaluate(
    spacecraft: Any, fixed_roll: bool
) -> None:
    ras, decs = targets(6, seed=4)
    times = list(spacecraft.timestamp[::25])
    constraint = re.SunConstraint(min_angle=80) | re.EarthLimbConstraint(min_angle=25)
    rolls = [20.0] * len(ras) if fixed_roll else None
    batch = constraint.evaluate_batch(
        spacecraft,
        list(ras),
        list(decs),
        times=times,
        target_rolls=rolls,
        apparent=True,
    )
    assert len(batch) == len(ras)
    for i, result in enumerate(batch):
        single = constraint.evaluate(
            spacecraft,
            float(ras[i]),
            float(decs[i]),
            times=times,
            target_roll=20.0 if fixed_roll else None,
            apparent=True,
        )
        np.testing.assert_array_equal(result.constraint_array, single.constraint_array)
        for key, value in single.constraint_values.items():
            np.testing.assert_allclose(result.constraint_values[key], value, rtol=0)
        assert len(result.visibility) == len(single.visibility)


def test_apparent_body_constraint_matches_astropy(spacecraft: Any) -> None:
    """A body other than the Sun and Moon is placed where its light arrives from."""
    ras, decs = targets(N, seed=8)
    rows = np.arange(N)
    constraint = re.BodyConstraint(body="Jupiter barycenter", min_angle=30)
    expected = reference(spacecraft, ras, decs, rows)["jupiter"]
    apparent = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=spacecraft.timestamp[:N], apparent=True
    )
    (key,) = apparent.constraint_values
    np.testing.assert_allclose(
        apparent.constraint_values[key], expected, atol=1e-3 / 3600, rtol=0
    )
    geometric = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=spacecraft.timestamp[:N]
    ).constraint_values[key]
    assert np.abs(geometric - expected).max() > 5 / 3600

    # In a tree, the grid falls back to correcting targets per time and agrees.
    tree = constraint | re.SunConstraint(min_angle=60)
    times = list(spacecraft.timestamp[::40])
    grid = tree.in_constraint_batch(
        spacecraft, list(ras[:30]), list(decs[:30]), times=times, apparent=True
    )
    for column, time in enumerate(times):
        paired = tree.in_constraint_elementwise(
            spacecraft, ras[:30], decs[:30], times=[time] * 30, apparent=True
        )
        np.testing.assert_array_equal(grid[:, column], paired)


@pytest.mark.skipif(
    not hasattr(re, "MoonLimbConstraint"), reason="needs MoonLimbConstraint"
)
def test_apparent_moon_limb(spacecraft: Any) -> None:
    """The Moon's limb is measured from its apparent centre and light-time distance."""
    ras, decs = targets(N, seed=13)
    rows = np.arange(N)
    constraint = re.MoonLimbConstraint(min_angle=10)
    expected = reference(spacecraft, ras, decs, rows)
    radius = np.degrees(np.arcsin(1737.4 / expected["moon_distance"]))
    apparent = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=spacecraft.timestamp[:N], apparent=True
    )
    np.testing.assert_allclose(
        apparent.constraint_values["moon_limb_angle_deg"],
        expected["moon"] - radius,
        atol=1e-3 / 3600,
        rtol=0,
    )
    times = list(spacecraft.timestamp[::40])
    grid = constraint.in_constraint_batch(
        spacecraft, list(ras[:30]), list(decs[:30]), times=times, apparent=True
    )
    for column, time in enumerate(times):
        paired = constraint.in_constraint_elementwise(
            spacecraft, ras[:30], decs[:30], times=[time] * 30, apparent=True
        )
        np.testing.assert_array_equal(grid[:, column], paired)


@pytest.mark.parametrize("moving", [False, True], ids=["stationary", "leo"])
def test_apparent_earth_limb_near_earth(
    tmp_path: Path, ensure_planetary_data: None, moving: bool
) -> None:
    """Compare with geometric tangent rays aberrated by geocentric velocity.

    This reference never uses a retarded Earth-centre distance to size the disk.
    """
    seconds = np.arange(0, 6001, 120)
    omega = np.sqrt(398600.4418 / 7000**3) if moving else 0.0
    phase = omega * seconds
    positions = 7000 * np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros(len(phase)))
    )
    velocities = (
        7000
        * omega
        * np.column_stack((-np.sin(phase), np.cos(phase), np.zeros(len(phase))))
    )
    path = tmp_path / "leo.txt"
    lines = [f"ScenarioEpoch {BEGIN.isoformat()}", "CoordinateSystem GCRS"]
    for sec, r, v in zip(seconds, positions, velocities):
        lines.append(f"{sec} " + " ".join(map(str, [*r, *v])))
    path.write_text("\n".join(lines) + "\n")
    ephem = re.FileEphemeris(
        str(path), begin=BEGIN, end=BEGIN + timedelta(seconds=6000), step_size=120
    )
    rows = np.arange(len(ephem.timestamp))
    ras, decs = targets(len(rows), seed=35)
    with solar_system_ephemeris.set(str(KERNEL)):
        geocentric = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs").transform_to(
            GCRS(obstime=Time(ephem.timestamp))
        )
    target_units = geocentric.cartesian.xyz.value.T
    radius = np.arcsin(6378.137 / 7000)
    azimuth = np.linspace(0, 2 * np.pi, 32768, endpoint=False)

    def aberrate(vectors: npt.NDArray[Any], velocity: npt.NDArray[Any]) -> Any:
        beta = velocity / 299792.458
        bm1 = np.sqrt(1 - np.dot(beta, beta))
        dot = vectors @ beta
        return (bm1 * vectors + (1 + dot[..., None] / (1 + bm1)) * beta) / (
            1 + dot[..., None]
        )

    expected = []
    for r, v, target in zip(ephem.obsgeoloc, ephem.obsgeovel, target_units):
        centre = -r / np.linalg.norm(r)
        east = np.cross(centre, [0, 0, 1])
        east /= np.linalg.norm(east)
        north = np.cross(centre, east)
        limb = np.cos(radius) * centre + np.sin(radius) * (
            np.cos(azimuth)[:, None] * east + np.sin(azimuth)[:, None] * north
        )
        corrected_target = aberrate(target, v)
        separation = np.arccos(
            np.clip(aberrate(limb, v) @ corrected_target, -1, 1)
        ).min()
        # Signed distance from the disk, including targets behind the limb.
        inside = np.dot(target, centre) > np.cos(radius)
        expected.append(np.degrees(separation) * (-1 if inside else 1))
    constraint = re.EarthLimbConstraint(min_angle=0)
    result = constraint.evaluate_elementwise(
        ephem, ras, decs, times=ephem.timestamp, apparent=True
    )
    np.testing.assert_allclose(
        result.constraint_values["earth_limb_angle_deg"],
        expected,
        atol=(4.0 if moving else 0.05) / 3600,
        rtol=0,
    )
    # Exercise both the vectorised grid and the fixed-roll column fallback.
    for rolls in (None, [12.0] * len(ras)):
        grid = constraint.in_constraint_batch(
            ephem, list(ras), list(decs), target_rolls=rolls, apparent=True
        )
        np.testing.assert_array_equal(np.diag(grid), result.constraint_array)


def apparent_frame(ephem: Any, row: int) -> Any:
    """Astropy's GCRS frame of the observer state of one ephemeris row."""
    return GCRS(
        obstime=Time(ephem.timestamp[row]),
        obsgeoloc=CartesianRepresentation(ephem.obsgeoloc[row] * u.km),
        obsgeovel=CartesianRepresentation(ephem.obsgeovel[row] * u.km / u.s),
    )


def proper_direction(frame: Any, ras: Any, decs: Any) -> npt.NDArray[Any]:
    """Unit vectors in which the observer of `frame` sees catalogue directions."""
    seen = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs").transform_to(frame)
    return np.asarray(seen.cartesian.xyz.value).T


def apparent_sun(ephem: Any, row: int) -> npt.NDArray[Any]:
    """Unit vector of the apparent Sun for the observer state of one row."""
    frame = apparent_frame(ephem, row)
    position = CartesianRepresentation(ephem.obsgeoloc[row] * u.km)
    with solar_system_ephemeris.set(str(KERNEL)):
        emitted = _get_apparent_body_position(
            "sun", frame.obstime, str(KERNEL), position
        )
        seen = SkyCoord(ICRS(emitted).transform_to(frame))
    return np.asarray(seen.cartesian.xyz.value / seen.cartesian.norm().value)


def radec(vectors: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    vectors = np.atleast_2d(vectors)
    return (
        np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360,
        np.degrees(np.arcsin(np.clip(vectors[:, 2], -1, 1))),
    )


def catalogue_of(frame: Any, proper: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Catalogue unit vectors the observer of `frame` sees in `proper`."""
    proper = np.atleast_2d(proper)
    catalogue = proper.copy()
    for _ in range(4):
        catalogue += proper - proper_direction(frame, *radec(catalogue))
        catalogue /= np.linalg.norm(catalogue, axis=1, keepdims=True)
    return catalogue


STAR = (90.0, 0.0)
ROW = 10


def star_field() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Targets 4 arcsec inside and outside a 0.5 degree circle around STAR, on
    eight bearings: closer to the edge than the ~20 arcsec of aberration."""
    bearings = np.radians(np.arange(0, 360, 45))
    offsets = (0.5 + np.array([-4.0, 4.0])[:, None] / 3600) * np.ones_like(bearings)
    return (
        (STAR[0] + offsets * np.sin(bearings)).ravel(),
        (STAR[1] + offsets * np.cos(bearings)).ravel(),
    )


def test_apparent_bright_star_matches_astropy(spacecraft: Any) -> None:
    """Star and target are displaced together, so the star is in the field
    exactly when Astropy's apparent separation says so."""
    ras, decs = star_field()
    frame = apparent_frame(spacecraft, ROW)
    target = proper_direction(frame, ras, decs)
    star = proper_direction(frame, np.array([STAR[0]]), np.array([STAR[1]]))[0]
    separation = np.degrees(np.arccos(np.clip(target @ star, -1, 1)))
    expected = separation < 0.5
    assert expected.any() and not expected.all()

    constraint = re.BrightStarConstraint(stars=[STAR], fov_radius=0.5)
    time = spacecraft.timestamp[ROW]
    result = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=[time] * len(ras), apparent=True
    )
    np.testing.assert_array_equal(result.constraint_array, expected)
    np.testing.assert_allclose(
        result.constraint_values["nearest_bright_star_angle_deg"],
        separation,
        atol=0.5 / 3600,
        rtol=0,
    )

    # Had the target alone been corrected, the star left at its catalogue
    # place, several of these targets would be on the wrong side of the edge.
    catalogue_star = np.array(
        [np.cos(np.radians(STAR[0])), np.sin(np.radians(STAR[0])), 0.0]
    )
    uncorrected = np.degrees(np.arccos(np.clip(target @ catalogue_star, -1, 1))) < 0.5
    assert (uncorrected != expected).any()


@pytest.mark.parametrize(
    "method",
    [
        "evaluate",
        "evaluate_batch",
        "in_constraint_batch",
        "evaluate_elementwise",
        "in_constraint_elementwise",
        "in_constraint",
    ],
)
@pytest.mark.parametrize(
    "tree", ["leaf", "and", "or", "not", "boresight", "polygon", "polygon_roll"]
)
def test_apparent_bright_star_in_every_method(
    spacecraft: Any, method: str, tree: str
) -> None:
    """Every evaluation path accepts bright stars with apparent=True and gives
    the catalogue-frame answer at the edge of the field."""
    star = re.BrightStarConstraint(stars=[STAR], fov_radius=0.5)
    square = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    sun = re.SunConstraint(min_angle=1)  # never violated by these targets
    constraint: Any = {
        "leaf": star,
        "and": ~sun & star,
        "or": sun | star,
        "not": ~star,
        "boresight": (sun | star).boresight_offset(roll_deg=30),
        "polygon": re.BrightStarConstraint(stars=[STAR], fov_polygon=square),
        "polygon_roll": re.BrightStarConstraint(
            stars=[STAR], fov_polygon=square, roll_deg=0.0
        ),
    }[tree]
    time = spacecraft.timestamp[ROW]
    ras, decs = star_field()
    if tree.startswith("polygon"):
        # The same edge test for a square field: due north, east, south, west.
        ras, decs = ras[::2], decs[::2]

    def run(index: int, **kw: Any) -> bool:
        ra, dec = float(ras[index]), float(decs[index])
        if method == "evaluate":
            result = constraint.evaluate(spacecraft, ra, dec, times=[time], **kw)
            return bool(result.constraint_array[0])
        if method == "in_constraint":
            return bool(constraint.in_constraint(time, spacecraft, ra, dec, **kw))
        result = getattr(constraint, method)(
            spacecraft, [ra], [dec], times=[time], **kw
        )
        if method == "evaluate_batch":
            return bool(result[0].constraint_array[0])
        if method == "evaluate_elementwise":
            return bool(result.constraint_array[0])
        return bool(np.asarray(result).ravel()[0])

    geometric = [run(i) for i in range(len(ras))]
    assert len(set(geometric)) == 2
    assert [run(i, apparent=True) for i in range(len(ras))] == geometric


def test_apparent_in_constraint_matches_batch(spacecraft: Any) -> None:
    """in_constraint(apparent=True) is the row of the apparent batch grid, and
    flips a target that sits between the geometric and apparent limits."""
    ras, decs = targets(2000, seed=11)
    time = spacecraft.timestamp[ROW]
    probe = re.SunConstraint(min_angle=0.0)
    geometric = probe.evaluate_elementwise(
        spacecraft, ras, decs, times=[time] * len(ras)
    ).constraint_values["sun_angle_deg"]
    apparent = probe.evaluate_elementwise(
        spacecraft, ras, decs, times=[time] * len(ras), apparent=True
    ).constraint_values["sun_angle_deg"]
    i = int(np.argmax(np.abs(apparent - geometric)))
    limit = float((apparent[i] + geometric[i]) / 2)
    constraint = re.SunConstraint(min_angle=limit)
    ra, dec = float(ras[i]), float(decs[i])

    assert constraint.in_constraint(time, spacecraft, ra, dec) == (geometric[i] < limit)
    assert constraint.in_constraint(time, spacecraft, ra, dec, apparent=True) == (
        apparent[i] < limit
    )
    times = list(spacecraft.timestamp[:20])
    several = constraint.in_constraint(times, spacecraft, ra, dec, apparent=True)
    grid = constraint.in_constraint_batch(
        spacecraft, [ra], [dec], times=times, apparent=True
    )
    assert several == grid[0].tolist()


def test_apparent_field_of_regard_matches_astropy(spacecraft: Any) -> None:
    """The apparent field of regard is the sky clear of the apparent Sun."""
    n_points = 2_000_000
    k = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * k / n_points
    phi = np.arange(n_points) * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(1.0 - z * z)
    samples = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)

    constraint = re.SunConstraint(min_angle=60)
    time = spacecraft.timestamp[ROW]

    def clear(sun: npt.NDArray[Any]) -> int:
        return int(np.count_nonzero(samples @ sun <= np.cos(np.radians(60))))

    def samples_clear(**kw: Any) -> float:
        area = constraint.instantaneous_field_of_regard(
            spacecraft, time=time, n_points=n_points, **kw
        )
        return area / (4 * np.pi) * n_points

    expected = clear(apparent_sun(spacecraft, ROW))
    geometric_sun = spacecraft.sun_pv.position[ROW] - spacecraft.gcrs_pv.position[ROW]
    expected_geometric = clear(geometric_sun / np.linalg.norm(geometric_sun))
    # Moving the cap by ~20 arcsec swaps samples along its edge but hardly
    # changes how many are clear, so the counts are compared exactly.
    assert expected != expected_geometric
    assert round(samples_clear(apparent=True)) == expected
    assert round(samples_clear()) == expected_geometric
    by_index = constraint.instantaneous_field_of_regard(
        spacecraft, index=ROW, n_points=n_points, apparent=True
    )
    assert by_index == pytest.approx(
        samples_clear(apparent=True) / n_points * 4 * np.pi
    )


@pytest.mark.parametrize("margin_arcsec", [-3.0, 3.0])
def test_apparent_route_around_uses_apparent_sun(
    spacecraft: Any, margin_arcsec: float
) -> None:
    """A slew whose closest approach to the apparent Sun is 3 arcsec inside or
    outside the limit is blocked or clear accordingly."""
    limit = 50.0
    sun = apparent_sun(spacecraft, ROW)
    frame = apparent_frame(spacecraft, ROW)
    east = np.cross([0.0, 0.0, 1.0], sun)
    east /= np.linalg.norm(east)
    north = np.cross(sun, east)
    closest_angle = np.radians(limit + margin_arcsec / 3600)
    closest = np.cos(closest_angle) * sun + np.sin(closest_angle) * east
    # The great circle through `closest` that is tangent to the cone around the
    # Sun; with 50 samples the midpoint, the closest approach, is sampled.
    ends = np.array(
        [np.cos(a) * closest + np.sin(a) * north for a in np.radians([-25.0, 25.0])]
    )
    (start_ra, end_ra), (start_dec, end_dec) = radec(catalogue_of(frame, ends))
    constraint = re.SunConstraint(min_angle=limit)
    route = constraint.route_around(
        spacecraft,
        float(start_ra),
        float(start_dec),
        float(end_ra),
        float(end_dec),
        index=ROW,
        apparent=True,
    )
    assert route.violated == (margin_arcsec < 0)
    if route.violated:
        assert route.first_violation == pytest.approx(0.5)
        assert route.waypoint is not None
        # The waypoint is a catalogue position, clear when evaluated as one.
        assert not constraint.in_constraint(
            spacecraft.timestamp[ROW], spacecraft, *route.waypoint, apparent=True
        )


def test_apparent_moving_body(spacecraft: Any) -> None:
    """A named body is placed where its light arrives from; a supplied track is
    corrected like any other target."""
    rows = np.arange(N)
    constraint = re.SunConstraint(min_angle=30)
    result = constraint.evaluate_moving_body(
        spacecraft, body="Jupiter barycenter", apparent=True
    )
    times = Time(spacecraft.timestamp[rows])
    position = CartesianRepresentation(spacecraft.obsgeoloc[rows].T * u.km)
    velocity = CartesianRepresentation(spacecraft.obsgeovel[rows].T * u.km / u.s)
    frame = GCRS(obstime=times, obsgeoloc=position, obsgeovel=velocity)
    with solar_system_ephemeris.set(str(KERNEL)):
        seen = {
            body: SkyCoord(
                ICRS(
                    _get_apparent_body_position(body, times, str(KERNEL), position)
                ).transform_to(frame)
            )
            for body in ("sun", "jupiter")
        }
    tolerance = 1e-3 / 3600
    np.testing.assert_allclose(
        result.constraint_values["sun_angle_deg"][:N],
        seen["jupiter"].separation(seen["sun"]).deg,
        atol=tolerance,
        rtol=0,
    )
    apparent_place = SkyCoord(
        np.array(result.ras[:N]) * u.deg, np.array(result.decs[:N]) * u.deg
    )
    assert (
        apparent_place.separation(
            SkyCoord(seen["jupiter"].ra, seen["jupiter"].dec)
        ).arcsec.max()
        < 1e-3
    )
    geometric = constraint.evaluate_moving_body(spacecraft, body="Jupiter barycenter")
    assert (
        np.abs(
            np.array(geometric.constraint_values["sun_angle_deg"])
            - np.array(result.constraint_values["sun_angle_deg"])
        ).max()
        > 1 / 3600
    )
    with pytest.raises(ValueError, match="use_horizons"):
        constraint.evaluate_moving_body(
            spacecraft, body="Jupiter barycenter", use_horizons=True, apparent=True
        )

    # A supplied track: the same as pairing each position with its time.
    n = len(spacecraft.timestamp)
    ras, decs = targets(n, seed=3)
    track = constraint.evaluate_moving_body(
        spacecraft, target_ras=ras, target_decs=decs, apparent=True
    )
    paired = constraint.evaluate_elementwise(
        spacecraft, ras, decs, times=spacecraft.timestamp, apparent=True
    )
    np.testing.assert_array_equal(track.constraint_array, paired.constraint_array)
    np.testing.assert_allclose(
        track.constraint_values["sun_angle_deg"],
        paired.constraint_values["sun_angle_deg"],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(track.ras, ras)
