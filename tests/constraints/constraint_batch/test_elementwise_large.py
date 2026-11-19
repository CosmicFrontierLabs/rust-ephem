"""Elementwise evaluation of many elements.

Above a few thousand elements the time lookup, the apparent target correction
and the paired kernels are spread across the cores. The result must be what
the same elements give in small pieces."""

from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt
import pytest

from rust_ephem import (
    EarthLimbConstraint,
    MoonConstraint,
    MoonLimbConstraint,
    SunConstraint,
    TLEEphemeris,
)
from rust_ephem.constraints import ConstraintConfig

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = 60
N_ROWS = 61
# Well above the size at which the work is shared, and not a multiple of the
# piece size used for the reference.
N_ELEMENTS = 20_011
PIECE = 1_000

Times = npt.NDArray[np.datetime64]
Floats = npt.NDArray[np.float64]


@pytest.fixture(scope="module")
def ephem(ensure_planetary_data: None) -> TLEEphemeris:
    end = BEGIN + timedelta(seconds=STEP * (N_ROWS - 1))
    return TLEEphemeris(TLE1, TLE2, BEGIN, end, STEP)


def constraints() -> dict[str, ConstraintConfig]:
    sun = SunConstraint(min_angle=45, max_angle=150)
    moon = MoonConstraint(min_angle=20)
    moon_limb = MoonLimbConstraint(min_angle=10)
    earth = EarthLimbConstraint(min_angle=15)
    return {
        "sun": sun,
        "moon": moon,
        "moon_limb": moon_limb,
        "earth_limb": earth,
        "or": sun | earth | moon_limb,
        "and": sun & earth,
        "nested": ~(sun ^ moon) | (earth & moon_limb),
    }


def elements(on_grid: bool) -> tuple[Floats, Floats, Times]:
    rng = np.random.default_rng(5)
    ras = rng.uniform(0.0, 360.0, N_ELEMENTS)
    decs = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, N_ELEMENTS)))
    start = np.datetime64(BEGIN.replace(tzinfo=None), "us")
    if on_grid:
        seconds = rng.integers(0, N_ROWS, N_ELEMENTS) * STEP
        offsets = (seconds * 1_000_000).astype("timedelta64[us]")
    else:
        # Mostly between rows, some exactly on one, in no order.
        micros = rng.integers(0, STEP * (N_ROWS - 1) * 1_000_000, N_ELEMENTS)
        micros[::7] = rng.integers(0, N_ROWS, len(micros[::7])) * STEP * 1_000_000
        offsets = micros.astype("timedelta64[us]")
    times: Times = start + offsets
    return ras, decs, times


@pytest.mark.parametrize("name", list(constraints()))
@pytest.mark.parametrize("apparent", [False, True])
@pytest.mark.parametrize("on_grid", [True, False])
def test_many_elements_equal_their_pieces(
    ephem: TLEEphemeris, name: str, apparent: bool, on_grid: bool
) -> None:
    constraint = constraints()[name]
    ras, decs, times = elements(on_grid)
    mask = constraint.in_constraint_elementwise(
        ephem, ras, decs, times=times, apparent=apparent
    )
    result = constraint.evaluate_elementwise(
        ephem, ras, decs, times=times, apparent=apparent
    )
    assert mask.shape == (N_ELEMENTS,)
    assert mask.any() and not mask.all()
    np.testing.assert_array_equal(result.constraint_array, mask)
    for start in range(0, N_ELEMENTS, PIECE):
        piece = slice(start, start + PIECE)
        expected = constraint.evaluate_elementwise(
            ephem, ras[piece], decs[piece], times=times[piece], apparent=apparent
        )
        np.testing.assert_array_equal(mask[piece], expected.constraint_array)
        assert result.constraint_values.keys() == expected.constraint_values.keys()
        # The Sun's barycentric velocity behind the apparent correction is
        # interpolated across the span of a request, so a piece asked for
        # alone agrees with the whole to rounding, not bit for bit.
        rtol = 1e-12 if apparent else 0.0
        for key, values in expected.constraint_values.items():
            np.testing.assert_allclose(
                result.constraint_values[key][piece], values, rtol=rtol, atol=0
            )
        for key, flags in expected.component_violated.items():
            np.testing.assert_array_equal(result.component_violated[key][piece], flags)


def test_many_times_sample_the_states_of_their_pieces(ephem: TLEEphemeris) -> None:
    _, _, times = elements(on_grid=False)
    whole = ephem.gcrs_pv_at(times)
    for start in range(0, N_ELEMENTS, 5 * PIECE):
        piece = slice(start, start + 5 * PIECE)
        expected = ephem.gcrs_pv_at(times[piece])
        np.testing.assert_array_equal(whole.position[piece], expected.position)
        np.testing.assert_array_equal(whole.velocity[piece], expected.velocity)


def test_errors_in_a_long_input_name_the_first_bad_entry(ephem: TLEEphemeris) -> None:
    constraint = SunConstraint(min_angle=45)
    ras, decs, times = elements(on_grid=True)
    late = np.datetime64("2024-01-02T00:00:00", "us")
    early = np.datetime64("2023-12-31T00:00:00", "us")
    outside = times.copy()
    outside[[15_000, 9_000]] = [late, early]
    with pytest.raises(ValueError, match="2023-12-31.* outside ephemeris range"):
        constraint.in_constraint_elementwise(ephem, ras, decs, times=outside)
    with pytest.raises(ValueError, match="2023-12-31.* outside ephemeris range"):
        ephem.gcrs_pv_at(outside)
    missing = times.copy()
    missing[[12_345, 18_000]] = np.datetime64("NaT")
    with pytest.raises(ValueError, match=r"times\[12345\] is NaT"):
        constraint.in_constraint_elementwise(ephem, ras, decs, times=missing)
