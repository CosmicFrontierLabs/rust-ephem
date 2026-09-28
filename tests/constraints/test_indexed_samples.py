"""Indexed evaluation must match full-array results, including query ordering."""

from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from rust_ephem import Constraint


@pytest.fixture(params=["sun", "moon", "earth", "eclipse", "moon_phase", "mounted"])
def constraint(request: pytest.FixtureRequest) -> Constraint:
    if request.param == "sun":
        return Constraint.sun_proximity(80.0)
    if request.param == "moon":
        return Constraint.moon_proximity(20.0)
    if request.param == "earth":
        return Constraint.earth_limb(20.0)
    if request.param == "eclipse":
        return Constraint.eclipse()
    if request.param == "moon_phase":
        return Constraint.moon_phase(min_illumination=0.2, max_illumination=0.8)
    return Constraint.or_(
        Constraint.and_(
            Constraint.sun_proximity(80.0), Constraint.not_(Constraint.eclipse())
        ),
        Constraint.boresight_offset(Constraint.earth_limb(20.0), yaw_deg=90.0),
    )


@pytest.mark.parametrize("indices", [[0], [3, 0, 3], [4, 1, 4, 0, 2]])
def test_indexed_batch_equals_full(
    tle_ephemeris: Any, constraint: Constraint, indices: list[int]
) -> None:
    ras = [0.0, 87.0, 180.0, 271.0]
    decs = [-60.0, -15.0, 10.0, 70.0]
    rolls = [0.0, 37.0, 90.0, 181.0]
    expected = constraint.in_constraint_batch(
        tle_ephemeris, ras, decs, target_rolls=rolls
    )[:, indices]
    by_index = constraint.in_constraint_batch(
        tle_ephemeris, ras, decs, indices=indices, target_rolls=rolls
    )
    times = [tle_ephemeris.timestamp[i] for i in indices]
    by_time = constraint.in_constraint_batch(
        tle_ephemeris, ras, decs, times=times, target_rolls=rolls
    )
    np.testing.assert_array_equal(by_index, expected)
    np.testing.assert_array_equal(by_time, expected)
    if len(indices) == 1:
        scalar_time = constraint.in_constraint_batch(
            tle_ephemeris, ras, decs, times=times[0], target_rolls=rolls
        )
        np.testing.assert_array_equal(scalar_time, expected)


def test_missing_timestamp_still_rejected(tle_ephemeris: Any) -> None:
    with pytest.raises(ValueError, match="not found in ephemeris timestamps"):
        Constraint.sun_proximity(80.0).in_constraint_batch(
            tle_ephemeris,
            [0.0],
            [0.0],
            times=tle_ephemeris.timestamp[0] + timedelta(microseconds=1),
        )
