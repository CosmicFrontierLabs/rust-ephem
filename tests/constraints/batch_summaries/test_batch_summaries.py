"""Per-target summaries of the target x time mask: the violated count and the
visibility windows. Both must be what the mask of ``in_constraint_batch``
gives, without the mask being built."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import (
    EarthLimbConstraint,
    MoonConstraint,
    SolarRollConstraint,
    SunConstraint,
    TLEEphemeris,
)
from rust_ephem.constraints import ConstraintConfig

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = 60
N_ROWS = 361
N_TARGETS = 40

Mask = npt.NDArray[np.bool_]
Floats = npt.NDArray[np.float64]


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> TLEEphemeris:
    end = BEGIN + timedelta(seconds=STEP * (N_ROWS - 1))
    return TLEEphemeris(TLE1, TLE2, BEGIN, end, STEP)


def targets(n: int = N_TARGETS) -> tuple[Floats, Floats]:
    rng = np.random.default_rng(21)
    ras = rng.uniform(0.0, 360.0, n)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return ras, decs


def constraints() -> dict[str, ConstraintConfig]:
    sun = SunConstraint(min_angle=45)
    earth = EarthLimbConstraint(min_angle=15)
    moon = MoonConstraint(min_angle=20)
    return {
        "earth_limb": earth,
        "or": sun | earth | moon,
        "and": earth & ~moon,
        "solar_roll": SolarRollConstraint(tolerance_deg=40.0) | earth,
    }


def runs(row: Mask) -> list[list[int]]:
    """First and last index of each run of ``False`` in ``row``."""
    found: list[list[int]] = []
    for index, violated in enumerate(row.tolist()):
        if violated:
            continue
        if found and found[-1][1] == index - 1:
            found[-1][1] = index
        else:
            found.append([index, index])
    return found


def off_grid_times() -> npt.NDArray[np.datetime64]:
    start = np.datetime64(BEGIN.replace(tzinfo=None), "us")
    micros = (np.arange(90) * 173_250_000) + 7_000_000
    times: npt.NDArray[np.datetime64] = start + micros.astype("timedelta64[us]")
    return times


@pytest.mark.parametrize("name", list(constraints()))
@pytest.mark.parametrize("apparent", [False, True])
@pytest.mark.parametrize("selection", ["all", "indices", "times"])
def test_summaries_are_those_of_the_mask(
    ephem: TLEEphemeris, name: str, apparent: bool, selection: str
) -> None:
    constraint = constraints()[name]
    ras, decs = targets()
    indices = [3, 4, 5, 6, 40, 41, 200, 360, 0] if selection == "indices" else None
    times = off_grid_times() if selection == "times" else None
    mask = constraint.in_constraint_batch(
        ephem,
        ras.tolist(),
        decs.tolist(),
        times=times,
        indices=indices,
        apparent=apparent,
    )
    counts = constraint.violated_count(
        ephem, ras, decs, times=times, indices=indices, apparent=apparent
    )
    assert counts.dtype == np.int64 and counts.shape == (N_TARGETS,)
    np.testing.assert_array_equal(counts, mask.sum(axis=1))
    fractions = constraint.violated_count(
        ephem, ras, decs, times=times, indices=indices, apparent=apparent, fraction=True
    )
    np.testing.assert_array_equal(fractions, mask.sum(axis=1) / mask.shape[1])
    windows = constraint.visibility_windows(
        ephem, ras, decs, times=times, indices=indices, apparent=apparent
    )
    assert len(windows) == N_TARGETS
    for row, found in zip(mask, windows, strict=True):
        assert found.dtype == np.int64 and found.shape[1:] == (2,)
        assert found.tolist() == runs(row)
    # The sample has targets that are always, never and sometimes visible.
    if name == "or" and selection == "all":
        assert any(len(found) > 1 for found in windows)
        assert mask.all(axis=1).any() or (~mask).all(axis=1).any()


def test_rolls_per_target(ephem: TLEEphemeris) -> None:
    """Fixed rolls, free rolls and a mixture, on a roll-dependent tree."""
    constraint = constraints()["solar_roll"]
    ras, decs = targets()
    rng = np.random.default_rng(4)
    fixed: list[float | None] = rng.uniform(0.0, 360.0, N_TARGETS).tolist()
    mixed = [roll if i % 3 else None for i, roll in enumerate(fixed)]
    for rolls in (fixed, mixed, [None] * N_TARGETS):
        mask = constraint.in_constraint_batch(
            ephem, ras.tolist(), decs.tolist(), target_rolls=rolls, n_roll_samples=24
        )
        counts = constraint.violated_count(
            ephem, ras, decs, target_rolls=rolls, n_roll_samples=24
        )
        np.testing.assert_array_equal(counts, mask.sum(axis=1))
        windows = constraint.visibility_windows(
            ephem, ras, decs, target_rolls=rolls, n_roll_samples=24
        )
        for row, found in zip(mask, windows, strict=True):
            assert found.tolist() == runs(row)
    assert (
        constraint.violated_count(ephem, ras, decs, target_rolls=fixed)
        != constraint.violated_count(ephem, ras, decs)
    ).any()


@pytest.mark.parametrize("apparent", [False, True])
def test_many_targets_are_evaluated_in_blocks(
    ephem: TLEEphemeris, apparent: bool
) -> None:
    """More cells than one block holds give the same summaries as the mask."""
    constraint = constraints()["or"]
    ras, decs = targets(12_500)
    assert len(ras) * N_ROWS > 1 << 22
    mask = constraint.in_constraint_batch(
        ephem, ras.tolist(), decs.tolist(), apparent=apparent
    )
    counts = constraint.violated_count(ephem, ras, decs, apparent=apparent)
    np.testing.assert_array_equal(counts, mask.sum(axis=1))
    windows = constraint.visibility_windows(ephem, ras, decs, apparent=apparent)
    assert sum(len(found) for found in windows) > len(ras)
    lengths = np.concatenate([found[:, 1] - found[:, 0] + 1 for found in windows])
    assert lengths.sum() == (~mask).sum()
    for i in range(0, len(ras), 997):
        assert windows[i].tolist() == runs(mask[i])


def test_windows_index_the_requested_times(ephem: TLEEphemeris) -> None:
    constraint = constraints()["earth_limb"]
    ras, decs = targets()
    times = off_grid_times()
    windows = constraint.visibility_windows(ephem, ras, decs, times=times)
    mask = constraint.in_constraint_batch(
        ephem, ras.tolist(), decs.tolist(), times=times
    )
    some = next(i for i, found in enumerate(windows) if len(found) > 1)
    starts, ends = times[windows[some][:, 0]], times[windows[some][:, 1]]
    assert (starts <= ends).all() and (starts[1:] > ends[:-1]).all()
    assert not mask[some][windows[some][:, 0]].any()
    assert not mask[some][windows[some][:, 1]].any()


def test_edge_cases(ephem: TLEEphemeris) -> None:
    never = SunConstraint(min_angle=0)
    always = SunConstraint(min_angle=180)
    ras, decs = targets(3)
    assert never.violated_count(ephem, ras, decs).tolist() == [0, 0, 0]
    assert always.violated_count(ephem, ras, decs).tolist() == [N_ROWS] * 3
    assert [w.tolist() for w in never.visibility_windows(ephem, ras, decs)] == [
        [[0, N_ROWS - 1]]
    ] * 3
    for found in always.visibility_windows(ephem, ras, decs):
        assert found.shape == (0, 2)
    # No targets, and one time.
    empty = np.array([], dtype=np.float64)
    assert never.violated_count(ephem, empty, empty).shape == (0,)
    assert never.visibility_windows(ephem, empty, empty) == []
    one = never.visibility_windows(ephem, ras, decs, indices=7)
    assert [w.tolist() for w in one] == [[[0, 0]]] * 3
    assert always.violated_count(ephem, ras, decs, indices=7).tolist() == [1, 1, 1]
    assert np.isnan(
        never.violated_count(ephem, ras, decs, indices=[], fraction=True)
    ).all()


def test_invalid_requests_are_rejected(ephem: TLEEphemeris) -> None:
    constraint = constraints()["or"]
    ras, decs = targets(4)
    for call in (constraint.violated_count, constraint.visibility_windows):
        with pytest.raises(ValueError, match="same length"):
            call(ephem, ras, decs[:3])
        with pytest.raises(ValueError, match="target_rolls must have the same length"):
            call(ephem, ras, decs, target_rolls=[0.0])
        with pytest.raises(ValueError, match="n_roll_samples"):
            call(ephem, ras, decs, n_roll_samples=0)
        with pytest.raises(ValueError, match="both 'times' and 'indices'"):
            call(ephem, ras, decs, times=BEGIN, indices=0)
        with pytest.raises(IndexError):
            call(ephem, ras, decs, indices=[N_ROWS])


def test_coordinates_follow_the_rule_of_the_mask(ephem: TLEEphemeris) -> None:
    """Text is refused, and numbers held as objects are read, as
    ``in_constraint_batch`` and ``evaluate_grid`` do."""
    constraint = constraints()["or"]
    ras, decs = targets(3)
    expected = constraint.violated_count(ephem, ras, decs)
    decimals = np.array([Decimal(float(ra)) for ra in ras], dtype=object)
    fractions = np.array([Fraction(float(dec)) for dec in decs], dtype=object)
    np.testing.assert_array_equal(
        constraint.violated_count(ephem, decimals, fractions), expected
    )
    np.testing.assert_array_equal(
        constraint.violated_count(ephem, ras.astype(object), decs.astype(object)),
        expected,
    )
    text = np.array(["10", "20", "30"])
    mixed = np.array([10.0, "20", 30.0], dtype=object)
    missing = np.array([1.0, 2.0, None], dtype=object)
    for call in (constraint.violated_count, constraint.visibility_windows):
        with pytest.raises(TypeError, match="target_ras must be a 1-D sequence"):
            call(ephem, text, decs)
        with pytest.raises(TypeError, match="target_decs must be a 1-D sequence"):
            call(ephem, ras, text)
        with pytest.raises(TypeError, match=r"target_ras\[1\] is a str"):
            call(ephem, mixed, decs)
        with pytest.raises(TypeError, match=r"target_decs\[2\] is a NoneType"):
            call(ephem, ras, missing)
    # The grid refuses the same inputs with the same words.
    with pytest.raises(TypeError, match="target_ras must be a 1-D sequence"):
        constraint.evaluate_grid(ephem, text, decs)
    with pytest.raises(TypeError, match=r"target_ras\[1\] is a str"):
        constraint.evaluate_grid(ephem, mixed, decs)
