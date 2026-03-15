import math

import pytest

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import SunConstraint


def test_field_of_regard_bounds_constraint_wrapper(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem
    c = Constraint.sun_proximity(45.0)
    field_sr = c.instantaneous_field_of_regard(ephem, index=0, n_points=8000)
    assert 0.0 <= field_sr <= 4.0 * math.pi


def test_field_of_regard_stricter_constraint_reduces_visible_area(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem
    loose = SunConstraint(min_angle=10.0)
    strict = SunConstraint(min_angle=90.0)

    loose_sr = loose.instantaneous_field_of_regard(ephem, index=0, n_points=8000)
    strict_sr = strict.instantaneous_field_of_regard(ephem, index=0, n_points=8000)

    assert strict_sr < loose_sr


def test_field_of_regard_index_out_of_range_raises_value_error(
    tle_ephem: rust_ephem.TLEEphemeris,
) -> None:
    ephem = tle_ephem
    c = Constraint.sun_proximity(45.0)

    with pytest.raises(ValueError):
        _ = c.instantaneous_field_of_regard(
            ephem,
            index=len(ephem.timestamp),
            n_points=2000,
        )
