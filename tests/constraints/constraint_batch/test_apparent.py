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
    earth_radius = np.degrees(np.arcsin(6378.137 / expected["earth_distance"]))
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
                    np.arcsin(6378.137 / expected["earth_distance"])
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
