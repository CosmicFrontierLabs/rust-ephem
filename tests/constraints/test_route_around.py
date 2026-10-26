"""Routing equivalence, sampled clearance, roll semantics, and degeneracies.

The independent NumPy reference follows coast-sim's
conops/common/vector.py:constraint_avoiding_waypoint, including its in-plane
candidate direction. Antipodes use the separately specified stable plane.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import (
    BodyConstraint,
    Constraint,
    DaytimeConstraint,
    EarthLimbConstraint,
    GroundEphemeris,
    RouteResult,
    SolarRollConstraint,
    SunConstraint,
)
from rust_ephem.constraints import RollReference, RustConstraintMixin

Vector = npt.NDArray[np.float64]
Check = Callable[[list[tuple[float, float]]], list[bool]]
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ephemeris(ensure_planetary_data: None) -> GroundEphemeris:
    return GroundEphemeris(35, -120, 0, BEGIN, BEGIN + timedelta(hours=6), 3600)


def unit(v: Vector) -> Vector:
    return v / np.linalg.norm(v)


def vec(ra: float, dec: float) -> Vector:
    r, d = np.radians([ra, dec])
    return unit(np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]))


def radec(v: Vector) -> tuple[float, float]:
    v = unit(v)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(
        np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    )


def angle(a: Vector, b: Vector) -> float:
    return float(np.arccos(np.clip(np.dot(a, b), -1, 1)))


def points(a: Vector, b: Vector, samples: int, endpoints: bool = True) -> list[Vector]:
    theta = angle(a, b)
    fractions = range(samples + 1) if endpoints else range(1, samples)
    return [
        unit(a + (i / samples) * (b - a))
        if theta < 0.01
        else (np.sin((1 - i / samples) * theta) * a + np.sin(i / samples * theta) * b)
        / np.sin(theta)
        for i in fractions
    ]


def reference(
    start: tuple[float, float],
    end: tuple[float, float],
    check: Check,
    samples: int = 50,
    margin: float = 5,
) -> tuple[float, float] | None:
    a, b = vec(*start), vec(*end)
    if angle(a, b) < 1e-6:
        return None
    n = unit(np.cross(a, b))
    direct = points(a, b, samples)
    flags = check([radec(v) for v in direct])
    if not any(flags):
        return None
    c = unit(direct[flags.index(True)])
    d = unit(np.cross(n, c))
    m = np.radians(margin)
    candidates = [unit(np.cos(m) * c + sign * np.sin(m) * d) for sign in (1, -1)]
    candidates.sort(key=lambda w: np.degrees(angle(a, w)) + np.degrees(angle(w, b)))
    for w in candidates:
        if check([radec(w)])[0]:
            continue
        if any(
            angle(x, y) > 1e-6
            and any(check([radec(v) for v in points(x, y, samples, False)]))
            for x, y in ((a, w), (w, b))
        ):
            continue
        return radec(w)
    return None


def checker(
    constraint: RustConstraintMixin,
    ephemeris: GroundEphemeris,
    index: int = 3,
    roll: float | None = None,
) -> Check:
    def check(coords: list[tuple[float, float]]) -> list[bool]:
        if not coords:
            return []
        mask = constraint.in_constraint_batch(
            ephemeris,
            [ra for ra, _ in coords],
            [dec for _, dec in coords],
            indices=[index],
            target_rolls=None if roll is None else [roll] * len(coords),
            n_roll_samples=12,
        )
        return [bool(v) for v in mask[:, 0]]

    return check


def random_pointings(count: int, seed: int = 84) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    return list(
        zip(
            rng.uniform(0, 360, count).tolist(),
            np.degrees(np.arcsin(rng.uniform(-1, 1, count))).tolist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("samples", [2, 50])
@pytest.mark.parametrize("min_angle", [2.0, 45.0])
def test_faithful_matches_numpy_reference(
    ephemeris: GroundEphemeris, samples: int, min_angle: float
) -> None:
    constraint = SunConstraint(min_angle=min_angle)
    check = checker(constraint, ephemeris)
    coords = random_pointings(240)
    for a, b in zip(coords[::2], coords[1::2], strict=True):
        expected = reference(a, b, check, samples)
        actual = constraint.route_around(ephemeris, *a, *b, index=3, samples=samples)
        flags = check([radec(v) for v in points(vec(*a), vec(*b), samples)])
        assert actual.violated == any(flags)
        assert actual.first_violation == (
            flags.index(True) / samples if any(flags) else None
        )
        if flags[0] or flags[-1]:
            assert actual.waypoint is None
        else:
            assert (
                actual.waypoint == pytest.approx(expected, abs=1e-9)
                if expected
                else actual.waypoint is None
            )
        assert (actual.detour_deg is None) == (actual.waypoint is None)


def assert_clear_route(
    result: RouteResult,
    a: tuple[float, float],
    b: tuple[float, float],
    check: Check,
    samples: int = 50,
) -> None:
    assert result.waypoint is not None
    assert 0 <= result.waypoint[0] < 360
    assert -90 <= result.waypoint[1] <= 90
    w = vec(*result.waypoint)
    legs = points(vec(*a), w, samples) + points(w, vec(*b), samples)
    assert not any(check([radec(v) for v in legs]))
    assert result.detour_deg == pytest.approx(
        np.degrees(angle(vec(*a), w)) + np.degrees(angle(w, vec(*b)))
    )


def test_growing_margin_routes_every_sampled_45_degree_cone_crossing(
    ephemeris: GroundEphemeris,
) -> None:
    constraint = SunConstraint(min_angle=45)
    check = checker(constraint, ephemeris)
    coords = random_pointings(2400)
    crossings = 0
    for a, b in zip(coords[::2], coords[1::2], strict=True):
        flags = check([radec(v) for v in points(vec(*a), vec(*b), 50)])
        if not any(flags) or flags[0] or flags[-1]:
            continue
        crossings += 1
        result = constraint.route_around(ephemeris, *a, *b, index=3, grow_margin=True)
        assert result.violated
        assert result.first_violation == flags.index(True) / 50
        assert_clear_route(result, a, b, check)
    assert crossings >= 50


def crossing(
    ephemeris: GroundEphemeris,
) -> tuple[tuple[float, float], tuple[float, float]]:
    # An arc through the Sun with both endpoints safely outside a 45-degree cone.
    sun = unit(ephemeris.sun_pv.position[3] - ephemeris.gcrs_pv.position[3])
    tangent = unit(np.cross(sun, np.array([0.0, 0.0, 1.0])))
    return radec(0.5 * sun + np.sqrt(0.75) * tangent), radec(
        0.5 * sun - np.sqrt(0.75) * tangent
    )


@pytest.mark.parametrize("samples", [2, 4])
def test_faithful_mode_can_return_the_reference_waypoint(
    ephemeris: GroundEphemeris, samples: int
) -> None:
    a, b = crossing(ephemeris)
    constraint = SunConstraint(min_angle=2)
    check = checker(constraint, ephemeris)
    expected = reference(a, b, check, samples)
    assert expected is not None
    result = constraint.route_around(ephemeris, *a, *b, index=3, samples=samples)
    assert result.waypoint == pytest.approx(expected, abs=1e-9)
    assert_clear_route(result, a, b, check, samples)


@pytest.mark.parametrize("margin", [0.01, 5.0, 89.0, 90.0])
def test_growing_margin_reaches_the_cap(
    ephemeris: GroundEphemeris, margin: float
) -> None:
    a, b = crossing(ephemeris)
    constraint = SunConstraint(min_angle=45)
    result = constraint.route_around(
        ephemeris, *a, *b, index=3, margin_deg=margin, grow_margin=True
    )
    assert_clear_route(result, a, b, checker(constraint, ephemeris))


@pytest.mark.parametrize("samples", [1, 2, 50])
def test_short_grids_and_growth_opt_in(
    ephemeris: GroundEphemeris, samples: int
) -> None:
    a, b = crossing(ephemeris)
    constraint = SunConstraint(min_angle=45)
    result = constraint.route_around(ephemeris, *a, *b, index=3, samples=samples)
    assert result.waypoint is None
    assert result.violated == (samples > 1)
    grown = constraint.route_around(
        ephemeris, *a, *b, index=3, samples=samples, grow_margin=True
    )
    if samples > 1:
        assert_clear_route(grown, a, b, checker(constraint, ephemeris), samples)
    else:
        assert not grown.violated


def test_blocked_endpoints_and_negligible_slews(ephemeris: GroundEphemeris) -> None:
    constraint = SunConstraint(min_angle=45)
    a, _ = crossing(ephemeris)
    sun = radec(ephemeris.sun_pv.position[3] - ephemeris.gcrs_pv.position[3])
    for start, end in [
        (sun, a),
        (a, sun),
        (sun, sun),
        (a, a),
        (sun, (sun[0] + 1e-7, sun[1])),
    ]:
        result = constraint.route_around(
            ephemeris, *start, *end, index=3, grow_margin=True
        )
        flags = checker(constraint, ephemeris)(
            [radec(v) for v in points(vec(*start), vec(*end), 50)]
        )
        assert result.violated == any(flags)
        assert result.first_violation == (
            flags.index(True) / 50 if any(flags) else None
        )
        assert result.waypoint is None
    # An end-point violation does not overwrite the earlier arc crossing.
    assert constraint.route_around(ephemeris, *a, *sun, index=3).first_violation == 0.26


@pytest.mark.parametrize("roll", [None, 0.0, 83.0])
def test_roll_context_matches_batch(
    ephemeris: GroundEphemeris, roll: float | None
) -> None:
    polygon = [(-15.0, 5.0), (15.0, 5.0), (15.0, 25.0), (-15.0, 25.0)]
    constraint = (
        SolarRollConstraint(tolerance_deg=25, roll_deg=17)
        | BodyConstraint(body="Moon", fov_polygon=polygon, roll_deg=39)
        | SunConstraint(min_angle=25).boresight_offset(
            pitch_deg=12, yaw_deg=7, roll_deg=33
        )
    )
    check = checker(constraint, ephemeris, roll=roll)
    coords = random_pointings(30, 63)
    for a, b in zip(coords[::2], coords[1::2], strict=True):
        result = constraint.route_around(
            ephemeris,
            *a,
            *b,
            index=3,
            target_roll=roll,
            n_roll_samples=12,
            grow_margin=True,
        )
        flags = check([radec(v) for v in points(vec(*a), vec(*b), 50)])
        assert result.violated == any(flags)
        assert result.first_violation == (
            flags.index(True) / 50 if any(flags) else None
        )
        if result.waypoint:
            assert_clear_route(result, a, b, check)


@pytest.mark.parametrize("start", [(0.0, 0.0), (0.0, 90.0), (43.0, -27.0)])
def test_antipodal_is_finite_and_repeatable(
    ephemeris: GroundEphemeris, start: tuple[float, float]
) -> None:
    end = ((start[0] + 180) % 360, -start[1])
    constraint = SunConstraint(min_angle=45)
    a = vec(*start)
    axis = np.eye(3)[np.argmin(np.abs(a))]
    n = unit(np.cross(a, axis))
    direct = [
        radec(np.cos(t * np.pi) * a + np.sin(t * np.pi) * np.cross(n, a))
        for t in np.linspace(0, 1, 51)
    ]
    flags = checker(constraint, ephemeris)(direct)
    result = constraint.route_around(ephemeris, *start, *end, index=3, grow_margin=True)
    assert result.violated == any(flags)
    assert result.first_violation == (flags.index(True) / 50 if any(flags) else None)
    again = constraint.route_around(ephemeris, *start, *end, index=3, grow_margin=True)
    assert (result.waypoint, result.detour_deg) == (again.waypoint, again.detour_deg)
    if result.waypoint:
        assert_clear_route(result, start, end, checker(constraint, ephemeris))


def test_time_floor_native_result_and_immutability(ephemeris: GroundEphemeris) -> None:
    a, b = crossing(ephemeris)
    constraint = SunConstraint(min_angle=45)
    expected = constraint.route_around(ephemeris, *a, *b, index=3, grow_margin=True)
    times: list[datetime | float] = [
        BEGIN + timedelta(hours=3),
        BEGIN + timedelta(hours=3, minutes=30),
        (BEGIN + timedelta(hours=3, minutes=30)).timestamp(),
        datetime(2023, 12, 31, 22, 30, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2024, 1, 1, 3, 30),
    ]
    for time in times:
        actual = constraint.route_around(ephemeris, *a, *b, time=time, grow_margin=True)
        assert isinstance(actual, RouteResult)
        assert actual.first_violation == expected.first_violation
        assert actual.waypoint == expected.waypoint
    native = Constraint.sun_proximity(45).route_around(
        ephemeris, *a, *b, index=3, grow_margin=True
    )
    assert native.waypoint == expected.waypoint
    with pytest.raises(AttributeError):
        setattr(native, "violated", False)


@pytest.mark.parametrize("index", [0, 3, 6])
def test_time_only_tree_has_no_route(ephemeris: GroundEphemeris, index: int) -> None:
    constraint = ~DaytimeConstraint()
    mask = checker(constraint, ephemeris, index)([(0, 0), (180, 0)])
    assert mask[0] == mask[1]
    result = constraint.route_around(
        ephemeris, 0, 0, 180, 0, index=index, grow_margin=True
    )
    assert result.violated == mask[0]
    assert result.first_violation == (0.0 if mask[0] else None)
    assert result.waypoint is None


@pytest.mark.parametrize("index", [0, 3, 6])
def test_snapshot_matches_full_ephemeris_and_mount_conventions(
    ephemeris: GroundEphemeris, index: int
) -> None:
    constraint = (
        SunConstraint(min_angle=45).boresight_offset(
            roll_deg=17,
            pitch_deg=8,
            yaw_deg=4,
            roll_clockwise=True,
            roll_reference=RollReference.SUN,
        )
        | EarthLimbConstraint(min_angle=5)
        | BodyConstraint(body="Moon", min_angle=10)
    )
    a, b = crossing(ephemeris)
    for roll in [None, 47.0]:
        check = checker(constraint, ephemeris, index, roll)
        flags = check([radec(v) for v in points(vec(*a), vec(*b), 50)])
        result = constraint.route_around(
            ephemeris,
            *a,
            *b,
            index=index,
            target_roll=roll,
            n_roll_samples=12,
            grow_margin=True,
        )
        assert result.violated == any(flags)
        assert result.first_violation == (
            flags.index(True) / 50 if any(flags) else None
        )
        if result.waypoint:
            assert_clear_route(result, a, b, check)


def test_time_after_ephemeris_uses_final_sample(ephemeris: GroundEphemeris) -> None:
    constraint = SunConstraint(min_angle=45)
    a, b = crossing(ephemeris)
    last = constraint.route_around(ephemeris, *a, *b, index=6, grow_margin=True)
    after = constraint.route_around(
        ephemeris, *a, *b, time=BEGIN + timedelta(days=1), grow_margin=True
    )
    assert after.waypoint == last.waypoint
    assert after.first_violation == last.first_violation


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"time": BEGIN, "index": 0},
        {"index": 7},
        {"time": BEGIN - timedelta(seconds=1)},
        {"time": float("nan")},
        {"index": 0, "samples": 0},
        {"index": 0, "margin_deg": 0},
        {"index": 0, "margin_deg": 91},
        {"index": 0, "margin_deg": float("nan")},
        {"index": 0, "target_roll": float("inf")},
        {"index": 0, "n_roll_samples": 0},
    ],
)
def test_invalid_options(ephemeris: GroundEphemeris, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SunConstraint(min_angle=45).route_around(ephemeris, 0, 0, 90, 0, **kwargs)


@pytest.mark.parametrize(
    "coords", [(float("nan"), 0, 90, 0), (0, 91, 90, 0), (0, 0, float("inf"), 0)]
)
def test_invalid_pointings(
    ephemeris: GroundEphemeris, coords: tuple[float, float, float, float]
) -> None:
    with pytest.raises(ValueError):
        SunConstraint(min_angle=45).route_around(ephemeris, *coords, index=0)
