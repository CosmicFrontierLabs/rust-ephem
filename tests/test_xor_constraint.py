import pytest

from rust_ephem import Constraint


def test_xor_config_json():
    c1 = Constraint.sun_proximity(10.0)
    c2 = Constraint.moon_proximity(15.0)
    xor_c = Constraint.xor_(c1, c2)
    js = xor_c.to_json()
    assert '"type": "xor"' in js or '"type":"xor"' in js
    # Should include both sub-constraint configs
    assert js.count('"min_angle"') >= 2


def test_xor_requires_two():
    c1 = Constraint.sun_proximity(5.0)
    with pytest.raises(ValueError):
        Constraint.xor_(c1)  # type: ignore[arg-type]


def test_at_least_config_json():
    c1 = Constraint.sun_proximity(10.0)
    c2 = Constraint.moon_proximity(15.0)
    c3 = Constraint.eclipse()
    at_least_c = Constraint.at_least(2, [c1, c2, c3])
    js = at_least_c.to_json()
    assert '"type": "at_least"' in js or '"type":"at_least"' in js
    assert '"min_violated": 2' in js or '"min_violated":2' in js


def test_at_least_requires_nonzero_threshold():
    c1 = Constraint.sun_proximity(5.0)
    with pytest.raises(ValueError):
        Constraint.at_least(0, [c1])


def test_at_least_threshold_must_not_exceed_count():
    c1 = Constraint.sun_proximity(5.0)
    c2 = Constraint.moon_proximity(10.0)
    with pytest.raises(ValueError):
        Constraint.at_least(3, [c1, c2])


def test_at_least_from_json_threshold_validation():
    with pytest.raises(ValueError):
        Constraint.from_json(
            '{"type":"at_least","min_violated":2,"constraints":[{"type":"sun","min_angle":10.0}]}'
        )
