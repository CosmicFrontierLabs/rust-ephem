"""Labels for results: the first violated rule of each row or cell, and the
names of the stray-light status values."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import (
    EarthLimbConstraint,
    MoonConstraint,
    MoonLimbConstraint,
    SAAConstraint,
    StrayLightConstraint,
    SunConstraint,
    TLEEphemeris,
)
from rust_ephem.constraints import ConstraintConfig, StrayLightStatus

from ..stray_light_constraint.test_stray_light_constraint import CONVERSION, make_table

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = 60
N_ROWS = 181
N = 400

Floats = npt.NDArray[np.float64]
Times = npt.NDArray[np.datetime64]


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> TLEEphemeris:
    end = BEGIN + timedelta(seconds=STEP * (N_ROWS - 1))
    return TLEEphemeris(TLE1, TLE2, BEGIN, end, STEP)


def elements() -> tuple[Floats, Floats, Times]:
    rng = np.random.default_rng(13)
    ras = rng.uniform(0.0, 360.0, N)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, N)))
    start = np.datetime64(BEGIN.replace(tzinfo=None), "us")
    micros = rng.integers(0, STEP * (N_ROWS - 1) * 1_000_000, N)
    times: Times = start + micros.astype("timedelta64[us]")
    return ras, decs, times


SUN = SunConstraint(min_angle=45)
WIDE_SUN = SunConstraint(min_angle=100)
EARTH = EarthLimbConstraint(min_angle=15)
MOON = MoonConstraint(min_angle=30)
MOON_LIMB = MoonLimbConstraint(min_angle=60)
SAA = SAAConstraint(polygon=[(-90.0, -50.0), (40.0, -50.0), (40.0, 0.0), (-90.0, 0.0)])


def trees() -> list[tuple[ConstraintConfig, list[str], list[ConstraintConfig]]]:
    """A tree, its rules' tags in the order written, and the rule behind each
    tag. Repeated rules are numbered as they are met."""
    return [
        (SUN, ["sun"], [SUN]),
        (EARTH | SUN | MOON, ["earth_limb", "sun", "moon"], [EARTH, SUN, MOON]),
        (MOON | SUN | EARTH, ["moon", "sun", "earth_limb"], [MOON, SUN, EARTH]),
        (
            SUN | (EARTH & WIDE_SUN) | SAA | MOON,
            ["sun", "earth_limb", "sun_2", "saa", "moon"],
            [SUN, EARTH, WIDE_SUN, SAA, MOON],
        ),
        (
            (MOON_LIMB ^ WIDE_SUN) | SUN | MOON,
            ["moon", "sun", "sun_2", "moon_2"],
            [MOON_LIMB, WIDE_SUN, SUN, MOON],
        ),
        (
            EARTH.boresight_offset(pitch_deg=0, yaw_deg=0) | SUN,
            ["earth_limb", "sun"],
            [EARTH, SUN],
        ),
    ]


@pytest.mark.parametrize("case", range(len(trees())))
def test_component_order_is_the_order_written(ephem: TLEEphemeris, case: int) -> None:
    tree, order, rules = trees()[case]
    ras, decs, times = elements()
    result = tree.evaluate_elementwise(ephem, ras, decs, times=times)
    assert result.component_order == order
    assert set(result.component_violated) == set(order)
    # Each tag is the mask of the rule at its place in the tree.
    for tag, rule in zip(order, rules, strict=True):
        alone = rule.in_constraint_elementwise(ephem, ras, decs, times=times)
        np.testing.assert_array_equal(result.component_violated[tag], alone, tag)
    grid = tree.evaluate_grid(ephem, ras[:7], decs[:7], indices=[0, 50, 180])
    assert grid.component_order == order


@pytest.mark.parametrize("case", range(len(trees())))
def test_first_violated_is_the_first_flagged_rule(
    ephem: TLEEphemeris, case: int
) -> None:
    tree, order, _ = trees()[case]
    ras, decs, times = elements()
    result = tree.evaluate_elementwise(ephem, ras, decs, times=times)
    flags = [result.component_violated[tag] & result.constraint_array for tag in order]
    expected = np.select(flags, order, default="")
    labels = result.first_violated
    assert labels.shape == (N,) and labels.dtype.kind == "U"
    np.testing.assert_array_equal(labels, expected)
    assert (labels[~result.constraint_array] == "").all()
    assert len(set(labels.tolist()) - {""}) > 1 or len(order) == 1

    grid = tree.evaluate_grid(ephem, ras[:30], decs[:30])
    flags = [grid.component_violated[tag] & grid.constraint_array for tag in order]
    cells = grid.first_violated
    assert cells.shape == grid.shape
    # Read-only, as the other arrays of a grid are.
    with pytest.raises(ValueError, match="read-only"):
        cells[0, 0] = "sun"
    np.testing.assert_array_equal(cells, np.select(flags, order, default=""))


def test_an_or_labels_every_violated_row(ephem: TLEEphemeris) -> None:
    ras, decs, times = elements()
    result = (EARTH | SUN | MOON).evaluate_elementwise(ephem, ras, decs, times=times)
    labels = result.first_violated
    np.testing.assert_array_equal(labels != "", result.constraint_array)
    # A row both the Earth and the Sun rule out is the Earth's: it is first.
    both = result.component_violated["earth_limb"] & result.component_violated["sun"]
    assert both.any() and (labels[both] == "earth_limb").all()


def test_a_negated_rule_is_named_with_its_prefix(ephem: TLEEphemeris) -> None:
    ras, decs, times = elements()
    result = (~SUN | EARTH).evaluate_elementwise(ephem, ras, decs, times=times)
    assert result.component_order == ["not.sun", "earth_limb"]
    assert set(result.first_violated.tolist()) <= {"", "not.sun", "earth_limb"}


def test_no_rows(ephem: TLEEphemeris) -> None:
    empty = np.array([], dtype=np.float64)
    result = (SUN | EARTH).evaluate_elementwise(
        ephem, empty, empty, times=np.array([], dtype="datetime64[us]")
    )
    assert result.first_violated.shape == (0,)
    assert result.component_order == ["sun", "earth_limb"]


def test_status_labels() -> None:
    codes = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])
    labels = StrayLightStatus.labels(codes)
    assert labels.shape == codes.shape
    assert labels.tolist() == [
        ["ok", "dark", "in_field"],
        ["first_cell", "unconverged", ""],
    ]
    assert StrayLightStatus.labels([7.0, -1.0, 0.5]).tolist() == ["", "", ""]
    assert StrayLightStatus.labels([]).shape == (0,)
    for status in StrayLightStatus:
        assert status.label == status.name.lower()
        assert StrayLightStatus.labels(status.value).item() == status.label


def test_status_labels_of_a_result(ephem: TLEEphemeris) -> None:
    ras, decs, times = elements()
    constraint = StrayLightConstraint(pst=make_table(), conversion_factor=CONVERSION)
    values = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times
    ).constraint_values
    for body in ("sun", "earth", "moon"):
        codes = values[f"{body}_status"]
        labels = StrayLightStatus.labels(codes)
        assert (labels != "").all()
        for status in StrayLightStatus:
            np.testing.assert_array_equal(labels == status.label, codes == status)
    assert len(set(StrayLightStatus.labels(values["earth_status"]).tolist())) > 1
