import math
from datetime import datetime, timezone

import rust_ephem
from rust_ephem import Constraint
from rust_ephem.constraints import SunConstraint


def _make_ephem() -> rust_ephem.TLEEphemeris:
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 23, 2, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, 300)


def test_field_of_regard_bounds_constraint_wrapper() -> None:
    ephem = _make_ephem()
    c = Constraint.sun_proximity(45.0)
    field_sr = c.instantaneous_field_of_regard(ephem, index=0, n_points=8000)
    assert 0.0 <= field_sr <= 4.0 * math.pi


def test_field_of_regard_stricter_constraint_reduces_visible_area() -> None:
    ephem = _make_ephem()
    loose = SunConstraint(min_angle=10.0)
    strict = SunConstraint(min_angle=90.0)

    loose_sr = loose.instantaneous_field_of_regard(ephem, index=0, n_points=8000)
    strict_sr = strict.instantaneous_field_of_regard(ephem, index=0, n_points=8000)

    assert strict_sr < loose_sr
