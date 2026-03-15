from typing import Any

from rust_ephem.constraints import SunConstraint


class TestOperatorCombinators:
    def test_and_type(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_and: Any = sun & moon

        assert combined_and.type == "and"

    def test_or_type(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_or: Any = sun | moon

        assert combined_or.type == "or"

    def test_xor_type(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_xor: Any = sun ^ moon

        assert combined_xor.type == "xor"

    def test_not_type(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)

        inverted: Any = ~sun

        assert inverted.type == "not"

    def test_and_constraints_first(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_and: Any = sun & moon

        assert combined_and.constraints[0] is sun

    def test_or_constraints_second(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_or: Any = sun | moon

        assert combined_or.constraints[1] is moon

    def test_xor_constraints_first(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)
        moon: SunConstraint = SunConstraint(min_angle=20.0)

        combined_xor: Any = sun ^ moon

        assert combined_xor.constraints[0] is sun

    def test_not_constraint(self) -> None:
        sun: SunConstraint = SunConstraint(min_angle=10.0)

        inverted: Any = ~sun

        assert inverted.constraint is sun
