"""A timezone offset must denote the same instant across public APIs."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rust_ephem import Constraint, GroundEphemeris, SunConstraint


@pytest.mark.parametrize("offset", [-5.0, 5.5, 9.75])
def test_aware_instants_agree_across_routing_evaluation_and_indexing(
    ensure_planetary_data: None, offset: float
) -> None:
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ephem = GroundEphemeris(35, -120, 0, begin, begin + timedelta(hours=6), 3600)
    instant = begin + timedelta(hours=3)
    local = instant.astimezone(timezone(timedelta(hours=offset)))
    constraint = SunConstraint(min_angle=45)
    assert ephem.index(local) == ephem.index(instant) == 3
    assert constraint.in_constraint(local, ephem, 0, 0) == constraint.in_constraint(
        instant, ephem, 0, 0
    )
    expected = constraint.evaluate(ephem, 0, 0, times=instant)
    actual = constraint.evaluate(ephem, 0, 0, times=local)
    np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
    assert actual.constraint_array == expected.constraint_array
    native = Constraint.sun_proximity(45)
    backends: list[SunConstraint | Constraint] = [constraint, native]
    for backend in backends:
        utc_route = backend.route_around(ephem, 220, -10, 340, -10, time=instant)
        local_route = backend.route_around(ephem, 220, -10, 340, -10, time=local)
        assert local_route.waypoint == utc_route.waypoint
        assert local_route.first_violation == utc_route.first_violation


def test_constructor_timezone_offsets_and_naive_utc_are_consistent(
    ensure_planetary_data: None,
) -> None:
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = begin + timedelta(hours=1)
    zone = timezone(timedelta(hours=5, minutes=30))
    reference = GroundEphemeris(35, -120, 0, begin, end, 3600)
    for start, finish in [
        (begin.astimezone(zone), end.astimezone(zone)),
        (begin.replace(tzinfo=None), end.replace(tzinfo=None)),
    ]:
        ephem = GroundEphemeris(35, -120, 0, start, finish, 3600)
        np.testing.assert_array_equal(ephem.timestamp, reference.timestamp)
        np.testing.assert_allclose(
            ephem.gcrs_pv.position, reference.gcrs_pv.position, rtol=0, atol=0
        )
