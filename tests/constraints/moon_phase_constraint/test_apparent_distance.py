"""The Moon distance a ``MoonPhaseConstraint`` reports with ``apparent=True``
must be the apparent one from every method, whether or not the rule has a
distance limit."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import MoonConstraint, MoonPhaseConstraint, TLEEphemeris

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
ROWS = [3, 50, 100]
KEY = "moon_distance_deg"

Floats = npt.NDArray[np.float64]


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> TLEEphemeris:
    return TLEEphemeris(TLE1, TLE2, BEGIN, BEGIN + timedelta(hours=2), 60)


def targets(count: int) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(21)
    ras = rng.uniform(0.0, 360.0, count)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count)))
    return ras.tolist(), decs.tolist()


def moon_angles(
    ephem: TLEEphemeris, ras: list[float], decs: list[float], apparent: bool
) -> Floats:
    """The angle a ``MoonConstraint`` measures, targets x times."""
    times = [ephem.timestamp[row] for row in ROWS]
    grid = MoonConstraint(min_angle=30).evaluate_grid(
        ephem, ras, decs, times=times, apparent=apparent
    )
    return grid.constraint_values["moon_angle_deg"]


RULES = [
    MoonPhaseConstraint(max_illumination=0.5),
    MoonPhaseConstraint(max_illumination=0.5, min_distance=40.0),
]


@pytest.mark.parametrize("rule", RULES, ids=["phase alone", "with a distance limit"])
@pytest.mark.parametrize("count", [2, 7], ids=["long grid", "wide grid"])
@pytest.mark.parametrize("apparent", [False, True])
def test_every_method_reports_the_same_moon_distance(
    ephem: TLEEphemeris, rule: MoonPhaseConstraint, count: int, apparent: bool
) -> None:
    ras, decs = targets(count)
    times = [ephem.timestamp[row] for row in ROWS]
    expected = moon_angles(ephem, ras, decs, apparent)

    grid = rule.evaluate_grid(ephem, ras, decs, times=times, apparent=apparent)
    np.testing.assert_allclose(grid.constraint_values[KEY], expected, atol=1e-9)

    batch = rule.evaluate_batch(ephem, ras, decs, times=times, apparent=apparent)
    for row, result in enumerate(batch):
        np.testing.assert_allclose(
            result.constraint_values[KEY], expected[row], atol=1e-9
        )

    single = rule.evaluate(ephem, ras[0], decs[0], times=times, apparent=apparent)
    np.testing.assert_allclose(single.constraint_values[KEY], expected[0], atol=1e-9)

    cell_ras = np.repeat(ras, len(times))
    cell_decs = np.repeat(decs, len(times))
    paired = rule.evaluate_elementwise(
        ephem, cell_ras, cell_decs, times=times * count, apparent=apparent
    )
    np.testing.assert_allclose(
        paired.constraint_values[KEY].reshape(expected.shape), expected, atol=1e-9
    )


def test_the_apparent_distance_differs_from_the_geometric(
    ephem: TLEEphemeris,
) -> None:
    """Otherwise the test above could not tell them apart."""
    ras, decs = targets(7)
    shift = moon_angles(ephem, ras, decs, True) - moon_angles(ephem, ras, decs, False)
    assert np.abs(shift).max() > 1e-3


@pytest.mark.parametrize("roll", [None, 0.0, 25.0])
@pytest.mark.parametrize("apparent", [False, True])
def test_a_boresight_offset_keeps_the_moon_distance_consistent(
    ephem: TLEEphemeris, roll: float | None, apparent: bool
) -> None:
    """The offset is applied to the target as the observer sees it, in every
    method, including for targets near a pole."""
    ras, decs = targets(7)
    ras, decs = [*ras, 40.0, 220.0], [*decs, 89.9, -89.95]
    count = len(ras)
    times = [ephem.timestamp[row] for row in ROWS]
    rolls = None if roll is None else [roll] * count
    expected = (
        MoonConstraint(min_angle=30)
        .boresight_offset(roll_deg=0.0, pitch_deg=10.0)
        .evaluate_grid(
            ephem, ras, decs, times=times, target_rolls=rolls, apparent=apparent
        )
        .constraint_values["moon_angle_deg"]
    )
    rule = MoonPhaseConstraint(max_illumination=0.5).boresight_offset(
        roll_deg=0.0, pitch_deg=10.0
    )

    grid = rule.evaluate_grid(
        ephem, ras, decs, times=times, target_rolls=rolls, apparent=apparent
    )
    np.testing.assert_allclose(grid.constraint_values[KEY], expected, atol=1e-9)

    batch = rule.evaluate_batch(
        ephem, ras, decs, times=times, target_rolls=rolls, apparent=apparent
    )
    for row, result in enumerate(batch):
        np.testing.assert_allclose(
            result.constraint_values[KEY], expected[row], atol=1e-9
        )

    paired = rule.evaluate_elementwise(
        ephem,
        np.repeat(ras, len(times)),
        np.repeat(decs, len(times)),
        times=times * count,
        target_rolls=None if roll is None else [roll] * (count * len(times)),
        apparent=apparent,
    )
    np.testing.assert_allclose(
        paired.constraint_values[KEY].reshape(expected.shape), expected, atol=1e-9
    )
