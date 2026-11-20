"""Per-element roll offsets measured from the nominal roll.

``roll_offsets[i]`` is evaluated as the explicit roll ``nominal_i +
roll_offsets[i]``, where the nominal roll turns body +Y toward the Sun. One
call must give what reading the nominal roll and passing it back gives."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

import rust_ephem as re
from rust_ephem import (
    EarthLimbConstraint,
    SolarRollConstraint,
    StrayLightConstraint,
    SunConstraint,
)
from rust_ephem.constraints import ConstraintConfig

from ..stray_light_constraint.test_stray_light_constraint import CONVERSION, make_table

Floats = npt.NDArray[np.float64]
Times = npt.NDArray[np.datetime64]

BEGIN = datetime(2024, 3, 1, tzinfo=timezone.utc)
STEP = 300
N_ROWS = 289
N = 60


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> re.FileEphemeris:
    """A circular orbit 60,000 km from the Earth's centre."""
    seconds = np.arange(N_ROWS) * float(STEP)
    radius = 60000.0
    rate = np.sqrt(398600.4418 / radius**3)
    angle = rate * seconds
    position = radius * np.stack(
        [np.cos(angle), 0.8 * np.sin(angle), 0.6 * np.sin(angle)], axis=1
    )
    velocity = (
        radius
        * rate
        * np.stack([-np.sin(angle), 0.8 * np.cos(angle), 0.6 * np.cos(angle)], axis=1)
    )
    times = [BEGIN + timedelta(seconds=float(s)) for s in seconds]
    return re.FileEphemeris.from_arrays(times, position, velocity, step_size=STEP)


def stray(
    roll_offset_deg: float = 0.0, max_ratio: float = 40.0
) -> StrayLightConstraint:
    return StrayLightConstraint(
        pst=make_table(),
        conversion_factor=CONVERSION,
        max_ratio=max_ratio,
        sun_azimuth_deg=180,
        roll_offset_deg=roll_offset_deg,
    )


def elements(on_grid: bool) -> tuple[Floats, Floats, Times]:
    rng = np.random.default_rng(3)
    ras = rng.uniform(0.0, 360.0, N)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, N)))
    start = np.datetime64(BEGIN.replace(tzinfo=None), "us")
    if on_grid:
        micros = rng.integers(0, N_ROWS, N) * STEP * 1_000_000
    else:
        micros = rng.integers(0, STEP * (N_ROWS - 1) * 1_000_000, N)
    times: Times = start + micros.astype("timedelta64[us]")
    return ras, decs, times


def offsets() -> list[float | None]:
    return [float(x) for x in np.random.default_rng(9).uniform(-180.0, 180.0, N)]


@pytest.mark.parametrize("apparent", [False, True])
@pytest.mark.parametrize("on_grid", [True, False])
def test_offsets_equal_the_nominal_roll_plus_the_offset(
    ephem: re.FileEphemeris, apparent: bool, on_grid: bool
) -> None:
    """What two calls gave: read ``roll_deg``, pass it back plus the offset."""
    constraint = stray()
    ras, decs, times = elements(on_grid)
    nominal = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, apparent=apparent
    )
    rolls = (nominal.constraint_values["roll_deg"] + np.array(offsets())).tolist()
    expected = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, target_rolls=rolls, apparent=apparent
    )
    result = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=offsets(), apparent=apparent
    )
    np.testing.assert_array_equal(result.constraint_array, expected.constraint_array)
    assert result.constraint_values.keys() == expected.constraint_values.keys()
    for key, values in expected.constraint_values.items():
        # The nominal roll read back is reduced to 0-360, so the roll may
        # differ from the one worked out directly by a rounding error.
        np.testing.assert_allclose(
            result.constraint_values[key], values, rtol=1e-9, atol=1e-9, err_msg=key
        )
    turned = (
        result.constraint_values["roll_deg"] - nominal.constraint_values["roll_deg"]
    )
    np.testing.assert_allclose(
        (turned - np.array(offsets()) + 180.0) % 360.0 - 180.0, 0.0, atol=1e-9
    )
    # The offsets turn the telescope far enough to change the answer.
    assert (
        result.constraint_values["stray_light_ratio"]
        != nominal.constraint_values["stray_light_ratio"]
    ).any()


@pytest.mark.parametrize("apparent", [False, True])
def test_a_uniform_offset_is_the_roll_offset_of_the_constraint(
    ephem: re.FileEphemeris, apparent: bool
) -> None:
    ras, decs, times = elements(on_grid=False)
    expected = stray(roll_offset_deg=37.5).evaluate_elementwise(
        ephem, ras, decs, times=times, apparent=apparent
    )
    result = stray().evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=[37.5] * N, apparent=apparent
    )
    np.testing.assert_array_equal(result.constraint_array, expected.constraint_array)
    for key, values in expected.constraint_values.items():
        np.testing.assert_array_equal(
            result.constraint_values[key], values, err_msg=key
        )


def test_an_offset_takes_the_place_of_the_roll_offset_of_the_constraint(
    ephem: re.FileEphemeris,
) -> None:
    ras, decs, times = elements(on_grid=True)
    expected = stray(roll_offset_deg=-20.0).evaluate_elementwise(
        ephem, ras, decs, times=times
    )
    result = stray(roll_offset_deg=65.0).evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=[-20.0] * N
    )
    np.testing.assert_array_equal(
        result.constraint_values["stray_light_ratio"],
        expected.constraint_values["stray_light_ratio"],
    )


@pytest.mark.parametrize("apparent", [False, True])
def test_mask_call_agrees_with_value_call(
    ephem: re.FileEphemeris, apparent: bool
) -> None:
    ras, decs, times = elements(on_grid=False)
    probe = stray().evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=offsets(), apparent=apparent
    )
    limit = float(np.nanmedian(probe.constraint_values["stray_light_ratio"]))
    trees: list[ConstraintConfig] = [
        stray(max_ratio=limit),
        SunConstraint(min_angle=60) | stray(max_ratio=limit),
        EarthLimbConstraint(min_angle=15) & stray(max_ratio=limit),
    ]
    for constraint in trees:
        result = constraint.evaluate_elementwise(
            ephem, ras, decs, times=times, roll_offsets=offsets(), apparent=apparent
        )
        mask = constraint.in_constraint_elementwise(
            ephem, ras, decs, times=times, roll_offsets=offsets(), apparent=apparent
        )
        np.testing.assert_array_equal(mask, result.constraint_array)
    # The limit is the median ratio, so the stray light decides both ways.
    alone = trees[0].in_constraint_elementwise(
        ephem, ras, decs, times=times, roll_offsets=offsets(), apparent=apparent
    )
    assert alone.any() and not alone.all()


def test_elements_without_an_offset_keep_the_nominal_attitude(
    ephem: re.FileEphemeris,
) -> None:
    constraint = stray()
    ras, decs, times = elements(on_grid=True)
    mixed = [offset if i % 3 else None for i, offset in enumerate(offsets())]
    nominal = constraint.evaluate_elementwise(ephem, ras, decs, times=times)
    rolled = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=offsets()
    )
    result = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, roll_offsets=mixed
    )
    without = np.array([offset is None for offset in mixed])
    for key in ("stray_light_ratio", "roll_deg"):
        expected = np.where(
            without, nominal.constraint_values[key], rolled.constraint_values[key]
        )
        np.testing.assert_array_equal(result.constraint_values[key], expected)


@pytest.mark.parametrize("apparent", [False, True])
def test_the_nominal_roll_is_the_solar_optimal_roll(
    ephem: re.FileEphemeris, apparent: bool
) -> None:
    """A panel facing +Y is best lit at offset zero, and is off its best roll
    by exactly the offset."""
    constraint = SolarRollConstraint(tolerance_deg=30.0)
    ras, decs, times = elements(on_grid=False)
    for offset, violated in [(0.0, False), (29.0, False), (31.0, True), (-170.0, True)]:
        result = constraint.evaluate_elementwise(
            ephem, ras, decs, times=times, roll_offsets=[offset] * N, apparent=apparent
        )
        assert (result.constraint_array == violated).all(), offset


def test_invalid_offsets_are_rejected(ephem: re.FileEphemeris) -> None:
    constraint = stray()
    ras, decs, times = elements(on_grid=True)
    with pytest.raises(ValueError, match="both 'target_rolls' and 'roll_offsets'"):
        constraint.evaluate_elementwise(
            ephem,
            ras,
            decs,
            times=times,
            target_rolls=[0.0] * N,
            roll_offsets=[0.0] * N,
        )
    with pytest.raises(ValueError, match="roll_offsets must have the same length"):
        constraint.in_constraint_elementwise(
            ephem, ras, decs, times=times, roll_offsets=[0.0] * (N - 1)
        )
    with pytest.raises(ValueError, match="roll_offsets must be finite"):
        constraint.in_constraint_elementwise(
            ephem, ras, decs, times=times, roll_offsets=[float("nan")] * N
        )
