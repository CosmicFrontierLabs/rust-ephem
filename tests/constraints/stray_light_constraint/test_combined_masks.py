"""Mask-only evaluation of rules combined with a stray-light constraint.

A mask call evaluates the cheap members of an ``|`` or ``&`` first and gives
the stray-light integral only the cells they leave undecided, and an apparent
grid is evaluated as paired cells. Neither may change a single cell."""

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import pytest

import rust_ephem as re
from rust_ephem import (
    EarthLimbConstraint,
    MoonConstraint,
    StrayLightConstraint,
    SunConstraint,
)
from rust_ephem.constraints import ConstraintConfig

from .test_stray_light_constraint import BEGIN, CONVERSION, STEP, N, make_table

Mask = npt.NDArray[np.bool_]

ROWS = [0, 130, 250, 377, 600, 811, 1200]
N_TARGETS = 24


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> re.FileEphemeris:
    """An eccentric high orbit, 30,000 to 400,000 km from the Earth."""
    seconds = np.arange(N) * float(STEP)
    rate = np.sqrt(398600.4418 / 215000.0**3)
    radius = 215000.0 * (1.0 - 0.86 * np.cos(rate * seconds))
    angle = 1.3 * rate * seconds
    position = np.stack(
        [
            radius * np.cos(angle),
            0.9 * radius * np.sin(angle),
            0.43 * radius * np.sin(angle),
        ],
        axis=1,
    )
    velocity = np.gradient(position, float(STEP), axis=0)
    times = [BEGIN + timedelta(seconds=float(s)) for s in seconds]
    return re.FileEphemeris.from_arrays(times, position, velocity, step_size=STEP)


@pytest.fixture(scope="module")
def stray() -> StrayLightConstraint:
    return StrayLightConstraint(
        pst=make_table(), conversion_factor=CONVERSION, max_ratio=40.0
    )


@pytest.fixture(scope="module")
def targets() -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(7)
    ras = rng.uniform(0.0, 360.0, N_TARGETS)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, N_TARGETS)))
    return ras.tolist(), decs.tolist()


@pytest.fixture(scope="module")
def times(ephem: re.FileEphemeris) -> list[datetime]:
    return [ephem.timestamp[row] for row in ROWS]


def rules() -> ConstraintConfig:
    return SunConstraint(min_angle=70) | MoonConstraint(min_angle=40)


def grid(
    constraint: ConstraintConfig,
    ephem: re.FileEphemeris,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
    rolls: list[float] | None = None,
) -> Mask:
    return np.asarray(
        constraint.in_constraint_batch(
            ephem, *targets, times=times, target_rolls=rolls, apparent=apparent
        ),
        dtype=np.bool_,
    )


def paired(
    constraint: ConstraintConfig,
    ephem: re.FileEphemeris,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
    rolls: list[float | None] | None = None,
) -> Mask:
    """The grid's cells as paired elements, reshaped to the grid."""
    ras = np.repeat(targets[0], len(times))
    decs = np.repeat(targets[1], len(times))
    cell_times = times * len(targets[0])
    mask = constraint.in_constraint_elementwise(
        ephem, ras, decs, times=cell_times, target_rolls=rolls, apparent=apparent
    )
    return np.asarray(mask, dtype=np.bool_).reshape(len(targets[0]), len(times))


@pytest.mark.parametrize("apparent", [False, True])
def test_members_split_the_sample(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
) -> None:
    """The cases below only test the short circuit if the cheap rules decide
    some cells and the stray light decides some of the rest."""
    cheap = grid(rules(), ephem, targets, times, apparent)
    expensive = grid(stray, ephem, targets, times, apparent)
    assert cheap.any() and not cheap.all()
    assert (expensive & ~cheap).any() and (~expensive & ~cheap).any()
    assert (expensive & cheap).any() and (~expensive & cheap).any()


@pytest.mark.parametrize("apparent", [False, True])
def test_combined_masks_are_the_combination_of_the_member_masks(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
) -> None:
    cheap = grid(rules(), ephem, targets, times, apparent)
    expensive = grid(stray, ephem, targets, times, apparent)
    limb = EarthLimbConstraint(min_angle=15)
    limb_mask = grid(limb, ephem, targets, times, apparent)
    cases: list[tuple[ConstraintConfig, Mask]] = [
        (rules() | stray, cheap | expensive),
        (stray | rules(), cheap | expensive),
        (rules() & stray, cheap & expensive),
        (stray & rules(), cheap & expensive),
        (~(rules() | stray), ~(cheap | expensive)),
        ((rules() | stray) & limb, (cheap | expensive) & limb_mask),
        (limb | (rules() & stray), limb_mask | (cheap & expensive)),
        (rules() ^ stray, cheap ^ expensive),
    ]
    for constraint, expected in cases:
        np.testing.assert_array_equal(
            grid(constraint, ephem, targets, times, apparent), expected
        )
        np.testing.assert_array_equal(
            paired(constraint, ephem, targets, times, apparent), expected
        )


@pytest.mark.parametrize("apparent", [False, True])
def test_stray_light_grid_is_its_paired_cells(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
) -> None:
    np.testing.assert_array_equal(
        grid(stray, ephem, targets, times, apparent),
        paired(stray, ephem, targets, times, apparent),
    )


@pytest.mark.parametrize("apparent", [False, True])
def test_mask_calls_agree_with_value_calls(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
) -> None:
    """Value calls still evaluate every member on every cell."""
    combined = rules() | stray
    mask = grid(combined, ephem, targets, times, apparent)
    result = combined.evaluate_grid(ephem, *targets, times=times, apparent=apparent)
    np.testing.assert_array_equal(result.constraint_array, mask)
    ratio = np.asarray(result.constraint_values["stray_light.stray_light_ratio"])
    cheap = grid(rules(), ephem, targets, times, apparent)
    assert np.isfinite(ratio[cheap]).any()


@pytest.mark.parametrize("apparent", [False, True])
def test_combined_masks_at_explicit_rolls(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
    apparent: bool,
) -> None:
    rng = np.random.default_rng(11)
    rolls = rng.uniform(0.0, 360.0, N_TARGETS).tolist()
    # A limit at the median ratio, so that turning the telescope flips cells.
    ratios = stray.evaluate_grid(ephem, *targets, times=times).constraint_values
    stray = StrayLightConstraint(
        pst=make_table(),
        conversion_factor=CONVERSION,
        max_ratio=float(np.nanmedian(ratios["stray_light_ratio"])),
    )
    cheap = grid(rules(), ephem, targets, times, apparent, rolls)
    expensive = grid(stray, ephem, targets, times, apparent, rolls)
    nominal = grid(stray, ephem, targets, times, apparent)
    assert (expensive != nominal).any()
    combined = rules() | stray
    np.testing.assert_array_equal(
        grid(combined, ephem, targets, times, apparent, rolls), cheap | expensive
    )
    cell_rolls: list[float | None] = np.repeat(rolls, len(times)).tolist()
    np.testing.assert_array_equal(
        paired(combined, ephem, targets, times, apparent, cell_rolls),
        cheap | expensive,
    )
    # Some elements with a roll and some without.
    mixed = [roll if i % 2 else None for i, roll in enumerate(cell_rolls)]
    with_roll = np.array([roll is not None for roll in mixed]).reshape(cheap.shape)
    expected = cheap | np.where(with_roll, expensive, nominal)
    np.testing.assert_array_equal(
        paired(combined, ephem, targets, times, apparent, mixed), expected
    )


def test_single_time_and_single_target_grids(
    ephem: re.FileEphemeris,
    stray: StrayLightConstraint,
    targets: tuple[list[float], list[float]],
    times: list[datetime],
) -> None:
    combined = rules() | stray
    whole = grid(combined, ephem, targets, times, True)
    np.testing.assert_array_equal(
        grid(combined, ephem, targets, times[2:3], True), whole[:, 2:3]
    )
    one = ([targets[0][5]], [targets[1][5]])
    np.testing.assert_array_equal(grid(combined, ephem, one, times, True), whole[5:6])
    assert bool(combined.in_constraint(times[4], ephem, one[0][0], one[1][0])) == bool(
        grid(combined, ephem, one, times[4:5], False)[0, 0]
    )
