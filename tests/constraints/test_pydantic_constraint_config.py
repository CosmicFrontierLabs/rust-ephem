"""
Test suite for RustConstraintMixin functionality.

Tests the base constraint mixin methods including evaluation, logical operators,
and operator overloads.
"""

from typing import Any

import pytest

from rust_ephem.constraints import (
    AndConstraint,
    BodyConstraint,
    EarthLimbConstraint,
    EclipseConstraint,
    MoonConstraint,
    NotConstraint,
    OrConstraint,
    SunConstraint,
    XorConstraint,
)


class TestRustConstraintMixin:
    """Test RustConstraintMixin base functionality."""

    def test_evaluate_creates_rust_constraint_initially_no_attr(
        self: Any, sun_constraint: Any, mock_ephem: Any
    ) -> None:
        config = sun_constraint
        assert not hasattr(config, "_rust_constraint")

    def test_evaluate_creates_rust_constraint_after_call(
        self: Any, sun_constraint: Any, mock_ephem: Any
    ) -> None:
        config = sun_constraint
        try:
            config.evaluate(mock_ephem, 0.0, 0.0)
        except Exception:
            pass  # We expect this to fail, we just want to check constraint creation
        assert hasattr(config, "_rust_constraint")

    def test_evaluate_uses_cached_constraint(
        self: Any, sun_constraint: Any, mock_ephem: Any
    ) -> None:
        config = sun_constraint
        try:
            config.evaluate(mock_ephem, 0.0, 0.0)
        except Exception:
            pass
        cached_constraint = config._rust_constraint
        try:
            config.evaluate(mock_ephem, 0.0, 0.0)
        except Exception:
            pass
        assert config._rust_constraint is cached_constraint

    def test_operator_precedence_expr_is_or(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert isinstance(expr, OrConstraint)

    def test_operator_precedence_first_constraint_is_and(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert isinstance(expr.constraints[0], AndConstraint)

    def test_operator_precedence_second_constraint_is_eclipse(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert expr.constraints[1] is eclipse


class TestConstraints:
    """Test individual constraint configuration classes."""

    def test_sun_constraint_config_type(self: Any, sun_constraint: Any) -> None:
        config = sun_constraint
        assert config.type == "sun"

    def test_sun_constraint_config_min_angle(self: Any, sun_constraint: Any) -> None:
        config = sun_constraint
        assert config.min_angle == 45.0

    def test_sun_constraint_config_max_angle_default(
        self: Any, sun_constraint: Any
    ) -> None:
        config = sun_constraint
        assert config.max_angle is None

    def test_sun_constraint_config_max_angle(self: Any) -> None:
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        assert config.max_angle == 90.0

    def test_sun_constraint_config_validation_max_angle_below_minimum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            SunConstraint(min_angle=45.0, max_angle=-10.0)

    def test_sun_constraint_config_validation_max_angle_above_maximum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            SunConstraint(min_angle=45.0, max_angle=200.0)

    def test_sun_constraint_config_validation_min_angle_below_minimum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            SunConstraint(min_angle=-10.0)

    def test_sun_constraint_config_validation_min_angle_above_maximum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            SunConstraint(min_angle=200.0)

    def test_moon_constraint_config_type(self: Any, moon_constraint: Any) -> None:
        config = moon_constraint
        assert config.type == "moon"

    def test_moon_constraint_config_min_angle(self: Any, moon_constraint: Any) -> None:
        config = moon_constraint
        assert config.min_angle == 30.0

    def test_moon_constraint_config_max_angle_default(
        self: Any, moon_constraint: Any
    ) -> None:
        config = moon_constraint
        assert config.max_angle is None

    def test_moon_constraint_config_max_angle(self: Any) -> None:
        config = MoonConstraint(min_angle=30.0, max_angle=60.0)
        assert config.max_angle == 60.0

    def test_moon_constraint_config_validation_max_angle_below_minimum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            MoonConstraint(min_angle=30.0, max_angle=-10.0)

    def test_moon_constraint_config_validation_max_angle_above_maximum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            MoonConstraint(min_angle=30.0, max_angle=200.0)

    def test_earth_limb_constraint_config_type(
        self: Any, earth_limb_constraint: Any
    ) -> None:
        config = earth_limb_constraint
        assert config.type == "earth_limb"

    def test_earth_limb_constraint_config_min_angle(
        self: Any, earth_limb_constraint: Any
    ) -> None:
        config = earth_limb_constraint
        assert config.min_angle == 10.0

    def test_earth_limb_constraint_config_max_angle_default(
        self: Any, earth_limb_constraint: Any
    ) -> None:
        config = earth_limb_constraint
        assert config.max_angle is None

    def test_earth_limb_constraint_config_max_angle(self: Any) -> None:
        config = EarthLimbConstraint(min_angle=10.0, max_angle=45.0)
        assert config.max_angle == 45.0

    def test_earth_limb_constraint_config_validation_max_angle_below_minimum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            EarthLimbConstraint(min_angle=10.0, max_angle=-10.0)

    def test_earth_limb_constraint_config_validation_max_angle_above_maximum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            EarthLimbConstraint(min_angle=10.0, max_angle=200.0)

    def test_body_constraint_config_type(self: Any, body_constraint: Any) -> None:
        config = body_constraint
        assert config.type == "body"

    def test_body_constraint_config_body(self: Any, body_constraint: Any) -> None:
        config = body_constraint
        assert config.body == "Mars"

    def test_body_constraint_config_min_angle(self: Any, body_constraint: Any) -> None:
        config = body_constraint
        assert config.min_angle == 15.0

    def test_body_constraint_config_max_angle_default(
        self: Any, body_constraint: Any
    ) -> None:
        config = body_constraint
        assert config.max_angle is None

    def test_body_constraint_config_max_angle(self: Any) -> None:
        config = BodyConstraint(body="Mars", min_angle=15.0, max_angle=75.0)
        assert config.max_angle == 75.0

    def test_body_constraint_config_validation_max_angle_below_minimum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            BodyConstraint(body="Mars", min_angle=15.0, max_angle=-10.0)

    def test_body_constraint_config_validation_max_angle_above_maximum(
        self: Any,
    ) -> None:
        with pytest.raises(ValueError):
            BodyConstraint(body="Mars", min_angle=15.0, max_angle=200.0)

    def test_eclipse_constraint_config_type(self: Any, eclipse_constraint: Any) -> None:
        config = eclipse_constraint
        assert config.type == "eclipse"

    def test_eclipse_constraint_config_umbra_only(
        self: Any, eclipse_constraint: Any
    ) -> None:
        config = eclipse_constraint
        assert config.umbra_only is True

    def test_eclipse_constraint_config_default_umbra_only(self: Any) -> None:
        config2 = EclipseConstraint()
        assert config2.umbra_only is True

    def test_and_constraint_config_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        config = AndConstraint(constraints=[sun, moon])
        assert config.type == "and"

    def test_and_constraint_config_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        config = AndConstraint(constraints=[sun, moon])
        assert len(config.constraints) == 2

    def test_and_constraint_config_validation_empty_list(self: Any) -> None:
        with pytest.raises(ValueError):
            AndConstraint(constraints=[])

    def test_or_constraint_config_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        config = OrConstraint(constraints=[sun, moon])
        assert config.type == "or"

    def test_or_constraint_config_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        config = OrConstraint(constraints=[sun, moon])
        assert len(config.constraints) == 2

    def test_not_constraint_config_type(self: Any, sun_constraint: Any) -> None:
        sun = sun_constraint
        config = NotConstraint(constraint=sun)
        assert config.type == "not"

    def test_not_constraint_config_constraint(self: Any, sun_constraint: Any) -> None:
        sun = sun_constraint
        config = NotConstraint(constraint=sun)
        assert config.constraint is sun


class TestConstraintSerialization:
    """Test JSON serialization/deserialization of constraints."""

    def test_sun_constraint_serialization_type_in_json(
        self: Any, sun_constraint: Any
    ) -> None:
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"type":"sun"' in json_str

    def test_sun_constraint_serialization_min_angle_in_json(
        self: Any, sun_constraint: Any
    ) -> None:
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"min_angle":45.0' in json_str

    def test_sun_constraint_serialization_max_angle_in_json(self: Any) -> None:
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        json_str = config.model_dump_json()
        assert '"max_angle":90.0' in json_str

    def test_sun_constraint_serialization_max_angle_none_in_json(
        self: Any, sun_constraint: Any
    ) -> None:
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"max_angle":null' in json_str

    def test_sun_constraint_deserialization_type(
        self: Any, sun_constraint: Any
    ) -> None:
        config = sun_constraint
        json_str = config.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored, SunConstraint)

    def test_sun_constraint_deserialization_min_angle(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        config = sun_constraint
        json_str = config.model_dump_json()

        restored = SunConstraint.model_validate_json(json_str)
        assert restored.min_angle == 45.0

    def test_sun_constraint_deserialization_max_angle(self: Any) -> None:
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        json_str = config.model_dump_json()

        restored = SunConstraint.model_validate_json(json_str)
        assert restored.max_angle == 90.0

    def test_complex_constraint_serialization_type_in_json(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        assert '"type":"or"' in json_str

    def test_complex_constraint_deserialization_type(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored, OrConstraint)

    def test_complex_constraint_deserialization_length(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert len(restored.constraints) == 2  # type: ignore[union-attr]

    def test_complex_constraint_deserialization_first_is_and(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored.constraints[0], AndConstraint)  # type: ignore[union-attr]

    def test_complex_constraint_deserialization_second_is_not(
        self: Any, sun_constraint: Any, moon_constraint: Any, eclipse_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored.constraints[1], NotConstraint)  # type: ignore[union-attr]


class TestLogicalOperators:
    """Test logical operator methods and overloads."""

    def test_and_method_creates_and_constraint_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert isinstance(combined, AndConstraint)

    def test_and_method_creates_and_constraint_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert len(combined.constraints) == 2

    def test_and_method_creates_and_constraint_first(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert combined.constraints[0] is sun

    def test_and_method_creates_and_constraint_second(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert combined.constraints[1] is moon

    def test_or_method_creates_or_constraint_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert isinstance(combined, OrConstraint)

    def test_or_method_creates_or_constraint_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert len(combined.constraints) == 2

    def test_or_method_creates_or_constraint_first(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert combined.constraints[0] is sun

    def test_or_method_creates_or_constraint_second(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert combined.constraints[1] is moon

    def test_not_method_creates_not_constraint_type(
        self: Any, sun_constraint: Any
    ) -> None:
        sun = sun_constraint
        negated = sun.not_()
        assert isinstance(negated, NotConstraint)

    def test_not_method_creates_not_constraint_constraint(
        self: Any, sun_constraint: Any
    ) -> None:
        sun = sun_constraint
        negated = sun.not_()
        assert negated.constraint is sun

    def test_and_operator_overload_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun & moon
        assert isinstance(combined, AndConstraint)

    def test_and_operator_overload_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun & moon
        assert len(combined.constraints) == 2

    def test_or_operator_overload_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun | moon
        assert isinstance(combined, OrConstraint)

    def test_or_operator_overload_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = sun | moon
        assert len(combined.constraints) == 2

    def test_invert_operator_overload_type(self: Any, sun_constraint: Any) -> None:
        sun = sun_constraint
        negated = ~sun
        assert isinstance(negated, NotConstraint)

    def test_invert_operator_overload_constraint(
        self: Any, sun_constraint: Any
    ) -> None:
        sun = sun_constraint
        negated = ~sun
        assert negated.constraint is sun

    def test_operator_chaining_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert isinstance(combined, OrConstraint)

    def test_operator_chaining_length(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert len(combined.constraints) == 2

    def test_operator_chaining_first_is_and(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert isinstance(combined.constraints[0], AndConstraint)

    def test_operator_chaining_second_is_sun(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert combined.constraints[1] is sun

    def test_nested_logical_operations_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        nested = ~(sun & moon)
        assert isinstance(nested, NotConstraint)

    def test_nested_logical_operations_constraint_type(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        sun = sun_constraint
        moon = moon_constraint
        nested = ~(sun & moon)
        assert isinstance(nested.constraint, AndConstraint)

    def test_xor_operator_overload(
        self: Any, sun_constraint: Any, moon_constraint: Any
    ) -> None:
        result = sun_constraint ^ moon_constraint
        assert isinstance(result, XorConstraint)
        assert len(result.constraints) == 2
