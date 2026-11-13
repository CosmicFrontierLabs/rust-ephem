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
        for body in ("sun", "moon", "earth"):
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
