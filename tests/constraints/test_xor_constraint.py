import pytest

from rust_ephem import Constraint


def test_xor_config_json() -> None:
    c1 = Constraint.sun_proximity(10.0)
    c2 = Constraint.moon_proximity(15.0)
    xor_c = Constraint.xor_(c1, c2)
    js = xor_c.to_json()
    assert '"type": "xor"' in js or '"type":"xor"' in js
    # Should include both sub-constraint configs
    assert js.count('"min_angle"') >= 2


def test_xor_requires_two() -> None:
    c1 = Constraint.sun_proximity(5.0)
    with pytest.raises(ValueError):
        Constraint.xor_(c1)
