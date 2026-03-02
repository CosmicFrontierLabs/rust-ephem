import json

import pytest

from rust_ephem import Constraint


class TestXorConstraint:
    def test_xor_config_json(self) -> None:
        c1 = Constraint.sun_proximity(10.0)
        c2 = Constraint.moon_proximity(15.0)
        xor_c = Constraint.xor_(c1, c2)
        js = xor_c.to_json()
        assert '"type": "xor"' in js or '"type":"xor"' in js
        assert js.count('"min_angle"') >= 2

    def test_xor_requires_two(self) -> None:
        c1 = Constraint.sun_proximity(5.0)
        with pytest.raises(ValueError):
            Constraint.xor_(c1)  # type: ignore[arg-type]


class TestAtLeastConstraint:
    def test_at_least_config_json(self) -> None:
        c1 = Constraint.sun_proximity(10.0)
        c2 = Constraint.moon_proximity(15.0)
        c3 = Constraint.eclipse()
        at_least_c = Constraint.at_least(2, [c1, c2, c3])
        js = at_least_c.to_json()
        payload = json.loads(js)
        assert payload["type"] == "at_least"
        assert payload["min_violated"] == 2

    def test_at_least_requires_nonzero_threshold(self) -> None:
        c1 = Constraint.sun_proximity(5.0)
        with pytest.raises(ValueError):
            Constraint.at_least(0, [c1])

    def test_at_least_threshold_must_not_exceed_count(self) -> None:
        c1 = Constraint.sun_proximity(5.0)
        c2 = Constraint.moon_proximity(10.0)
        with pytest.raises(ValueError):
            Constraint.at_least(3, [c1, c2])

    def test_at_least_from_json_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            Constraint.from_json(
                '{"type":"at_least","min_violated":2,"constraints":[{"type":"sun","min_angle":10.0}]}'
            )
