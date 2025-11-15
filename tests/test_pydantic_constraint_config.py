"""
Test suite for RustConstraintMixin functionality.

Tests the base constraint mixin methods including evaluation, logical operators,
and operator overloads.
"""

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
)


@pytest.fixture
def mock_ephem():
    """Fixture for a mock ephemeris object."""

    class MockEphemeris:
        pass

    return MockEphemeris()


@pytest.fixture
def sun_constraint():
    """Fixture for a SunConstraint instance."""
    return SunConstraint(min_angle=45.0)


@pytest.fixture
def moon_constraint():
    """Fixture for a MoonConstraint instance."""
    return MoonConstraint(min_angle=30.0)


@pytest.fixture
def eclipse_constraint():
    """Fixture for an EclipseConstraint instance."""
    return EclipseConstraint(umbra_only=True)


@pytest.fixture
def earth_limb_constraint():
    """Fixture for an EarthLimbConstraint instance."""
    return EarthLimbConstraint(min_angle=10.0)


@pytest.fixture
def body_constraint():
    """Fixture for a BodyConstraint instance."""
    return BodyConstraint(body="Mars", min_angle=15.0)


class TestRustConstraintMixin:
    """Test RustConstraintMixin base functionality."""

    def test_evaluate_creates_rust_constraint_initially_no_attr(
        self, sun_constraint, mock_ephem
    ):
        """Test that evaluate method initially has no _rust_constraint attribute."""
        config = sun_constraint
        assert not hasattr(config, "_rust_constraint")

    def test_evaluate_creates_rust_constraint_after_call(
        self, sun_constraint, mock_ephem
    ):
        """Test that evaluate method creates _rust_constraint after call."""
        config = sun_constraint
        try:
            config.evaluate(mock_ephem, 0.0, 0.0)
        except Exception:
            pass  # We expect this to fail, we just want to check constraint creation
        assert hasattr(config, "_rust_constraint")

    def test_evaluate_uses_cached_constraint(self, sun_constraint, mock_ephem):
        """Test that evaluate method uses cached _rust_constraint on second call."""
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
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test operator precedence: expression is OrConstraint."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert isinstance(expr, OrConstraint)

    def test_operator_precedence_first_constraint_is_and(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test operator precedence: first constraint in OR is AndConstraint."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert isinstance(expr.constraints[0], AndConstraint)

    def test_operator_precedence_second_constraint_is_eclipse(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test operator precedence: second constraint in OR is eclipse."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        expr = sun & moon | eclipse
        assert expr.constraints[1] is eclipse


class TestConstraints:
    """Test individual constraint configuration classes."""

    def test_sun_constraint_config_type(self, sun_constraint):
        """Test SunConstraint type."""
        config = sun_constraint
        assert config.type == "sun"

    def test_sun_constraint_config_min_angle(self, sun_constraint):
        """Test SunConstraint min_angle."""
        config = sun_constraint
        assert config.min_angle == 45.0

    def test_sun_constraint_config_max_angle_default(self, sun_constraint):
        """Test SunConstraint max_angle defaults to None."""
        config = sun_constraint
        assert config.max_angle is None

    def test_sun_constraint_config_max_angle(self):
        """Test SunConstraint max_angle can be set."""
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        assert config.max_angle == 90.0

    def test_sun_constraint_config_validation_max_angle_below_minimum(self):
        """Test SunConstraint validation for max_angle below minimum."""
        with pytest.raises(ValueError):
            SunConstraint(min_angle=45.0, max_angle=-10.0)

    def test_sun_constraint_config_validation_max_angle_above_maximum(self):
        """Test SunConstraint validation for max_angle above maximum."""
        with pytest.raises(ValueError):
            SunConstraint(min_angle=45.0, max_angle=200.0)

    def test_sun_constraint_config_validation_min_angle_below_minimum(self):
        """Test SunConstraint validation for min_angle below minimum."""
        with pytest.raises(ValueError):
            SunConstraint(min_angle=-10.0)

    def test_sun_constraint_config_validation_min_angle_above_maximum(self):
        """Test SunConstraint validation for min_angle above maximum."""
        with pytest.raises(ValueError):
            SunConstraint(min_angle=200.0)

    def test_moon_constraint_config_type(self, moon_constraint):
        """Test MoonConstraint type."""
        config = moon_constraint
        assert config.type == "moon"

    def test_moon_constraint_config_min_angle(self, moon_constraint):
        """Test MoonConstraint min_angle."""
        config = moon_constraint
        assert config.min_angle == 30.0

    def test_moon_constraint_config_max_angle_default(self, moon_constraint):
        """Test MoonConstraint max_angle defaults to None."""
        config = moon_constraint
        assert config.max_angle is None

    def test_moon_constraint_config_max_angle(self):
        """Test MoonConstraint max_angle can be set."""
        config = MoonConstraint(min_angle=30.0, max_angle=60.0)
        assert config.max_angle == 60.0

    def test_moon_constraint_config_validation_max_angle_below_minimum(self):
        """Test MoonConstraint validation for max_angle below minimum."""
        with pytest.raises(ValueError):
            MoonConstraint(min_angle=30.0, max_angle=-10.0)

    def test_moon_constraint_config_validation_max_angle_above_maximum(self):
        """Test MoonConstraint validation for max_angle above maximum."""
        with pytest.raises(ValueError):
            MoonConstraint(min_angle=30.0, max_angle=200.0)

    def test_earth_limb_constraint_config_type(self, earth_limb_constraint):
        """Test EarthLimbConstraint type."""
        config = earth_limb_constraint
        assert config.type == "earth_limb"

    def test_earth_limb_constraint_config_min_angle(self, earth_limb_constraint):
        """Test EarthLimbConstraint min_angle."""
        config = earth_limb_constraint
        assert config.min_angle == 10.0

    def test_earth_limb_constraint_config_max_angle_default(
        self, earth_limb_constraint
    ):
        """Test EarthLimbConstraint max_angle defaults to None."""
        config = earth_limb_constraint
        assert config.max_angle is None

    def test_earth_limb_constraint_config_max_angle(self):
        """Test EarthLimbConstraint max_angle can be set."""
        config = EarthLimbConstraint(min_angle=10.0, max_angle=45.0)
        assert config.max_angle == 45.0

    def test_earth_limb_constraint_config_validation_max_angle_below_minimum(self):
        """Test EarthLimbConstraint validation for max_angle below minimum."""
        with pytest.raises(ValueError):
            EarthLimbConstraint(min_angle=10.0, max_angle=-10.0)

    def test_earth_limb_constraint_config_validation_max_angle_above_maximum(self):
        """Test EarthLimbConstraint validation for max_angle above maximum."""
        with pytest.raises(ValueError):
            EarthLimbConstraint(min_angle=10.0, max_angle=200.0)

    def test_body_constraint_config_type(self, body_constraint):
        """Test BodyConstraint type."""
        config = body_constraint
        assert config.type == "body"

    def test_body_constraint_config_body(self, body_constraint):
        """Test BodyConstraint body."""
        config = body_constraint
        assert config.body == "Mars"

    def test_body_constraint_config_min_angle(self, body_constraint):
        """Test BodyConstraint min_angle."""
        config = body_constraint
        assert config.min_angle == 15.0

    def test_body_constraint_config_max_angle_default(self, body_constraint):
        """Test BodyConstraint max_angle defaults to None."""
        config = body_constraint
        assert config.max_angle is None

    def test_body_constraint_config_max_angle(self):
        """Test BodyConstraint max_angle can be set."""
        config = BodyConstraint(body="Mars", min_angle=15.0, max_angle=75.0)
        assert config.max_angle == 75.0

    def test_body_constraint_config_validation_max_angle_below_minimum(self):
        """Test BodyConstraint validation for max_angle below minimum."""
        with pytest.raises(ValueError):
            BodyConstraint(body="Mars", min_angle=15.0, max_angle=-10.0)

    def test_body_constraint_config_validation_max_angle_above_maximum(self):
        """Test BodyConstraint validation for max_angle above maximum."""
        with pytest.raises(ValueError):
            BodyConstraint(body="Mars", min_angle=15.0, max_angle=200.0)

    def test_eclipse_constraint_config_type(self, eclipse_constraint):
        """Test EclipseConstraint type."""
        config = eclipse_constraint
        assert config.type == "eclipse"

    def test_eclipse_constraint_config_umbra_only(self, eclipse_constraint):
        """Test EclipseConstraint umbra_only."""
        config = eclipse_constraint
        assert config.umbra_only is True

    def test_eclipse_constraint_config_default_umbra_only(self):
        """Test EclipseConstraint default umbra_only."""
        config2 = EclipseConstraint()
        assert config2.umbra_only is True

    def test_and_constraint_config_type(self, sun_constraint, moon_constraint):
        """Test AndConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        config = AndConstraint(constraints=[sun, moon])
        assert config.type == "and"

    def test_and_constraint_config_length(self, sun_constraint, moon_constraint):
        """Test AndConstraint constraints length."""
        sun = sun_constraint
        moon = moon_constraint
        config = AndConstraint(constraints=[sun, moon])
        assert len(config.constraints) == 2

    def test_and_constraint_config_validation_empty_list(self):
        """Test AndConstraint validation for empty constraints."""
        with pytest.raises(ValueError):
            AndConstraint(constraints=[])

    def test_or_constraint_config_type(self, sun_constraint, moon_constraint):
        """Test OrConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        config = OrConstraint(constraints=[sun, moon])
        assert config.type == "or"

    def test_or_constraint_config_length(self, sun_constraint, moon_constraint):
        """Test OrConstraint constraints length."""
        sun = sun_constraint
        moon = moon_constraint
        config = OrConstraint(constraints=[sun, moon])
        assert len(config.constraints) == 2

    def test_not_constraint_config_type(self, sun_constraint):
        """Test NotConstraint type."""
        sun = sun_constraint
        config = NotConstraint(constraint=sun)
        assert config.type == "not"

    def test_not_constraint_config_constraint(self, sun_constraint):
        """Test NotConstraint constraint."""
        sun = sun_constraint
        config = NotConstraint(constraint=sun)
        assert config.constraint is sun


class TestConstraintSerialization:
    """Test JSON serialization/deserialization of constraints."""

    def test_sun_constraint_serialization_type_in_json(self, sun_constraint):
        """Test SunConstraint JSON contains type."""
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"type":"sun"' in json_str

    def test_sun_constraint_serialization_min_angle_in_json(self, sun_constraint):
        """Test SunConstraint JSON contains min_angle."""
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"min_angle":45.0' in json_str

    def test_sun_constraint_serialization_max_angle_in_json(self):
        """Test SunConstraint JSON contains max_angle when set."""
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        json_str = config.model_dump_json()
        assert '"max_angle":90.0' in json_str

    def test_sun_constraint_serialization_max_angle_none_in_json(self, sun_constraint):
        """Test SunConstraint JSON contains max_angle as null when None."""
        config = sun_constraint
        json_str = config.model_dump_json()
        assert '"max_angle":null' in json_str

    def test_sun_constraint_deserialization_type(self, sun_constraint):
        """Test SunConstraint deserialization type."""
        config = sun_constraint
        json_str = config.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored, SunConstraint)

    def test_sun_constraint_deserialization_min_angle(self, sun_constraint):
        """Test SunConstraint deserialization min_angle."""
        config = sun_constraint
        json_str = config.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert restored.min_angle == 45.0

    def test_sun_constraint_deserialization_max_angle(self):
        """Test SunConstraint deserialization max_angle."""
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        json_str = config.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert restored.max_angle == 90.0

    def test_complex_constraint_serialization_type_in_json(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test complex constraint JSON contains type."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        assert '"type":"or"' in json_str

    def test_complex_constraint_deserialization_type(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test complex constraint deserialization type."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored, OrConstraint)

    def test_complex_constraint_deserialization_length(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test complex constraint deserialization constraints length."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert len(restored.constraints) == 2

    def test_complex_constraint_deserialization_first_is_and(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test complex constraint deserialization first constraint is And."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored.constraints[0], AndConstraint)

    def test_complex_constraint_deserialization_second_is_not(
        self, sun_constraint, moon_constraint, eclipse_constraint
    ):
        """Test complex constraint deserialization second constraint is Not."""
        sun = sun_constraint
        moon = moon_constraint
        eclipse = eclipse_constraint
        complex_constraint = (sun & moon) | ~eclipse
        json_str = complex_constraint.model_dump_json()
        from rust_ephem.constraints import CombinedConstraintConfig

        restored = CombinedConstraintConfig.validate_json(json_str)
        assert isinstance(restored.constraints[1], NotConstraint)


class TestLogicalOperators:
    """Test logical operator methods and overloads."""

    def test_and_method_creates_and_constraint_type(
        self, sun_constraint, moon_constraint
    ):
        """Test and_ method creates AndConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert isinstance(combined, AndConstraint)

    def test_and_method_creates_and_constraint_length(
        self, sun_constraint, moon_constraint
    ):
        """Test and_ method creates AndConstraint with correct length."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert len(combined.constraints) == 2

    def test_and_method_creates_and_constraint_first(
        self, sun_constraint, moon_constraint
    ):
        """Test and_ method creates AndConstraint with correct first constraint."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert combined.constraints[0] is sun

    def test_and_method_creates_and_constraint_second(
        self, sun_constraint, moon_constraint
    ):
        """Test and_ method creates AndConstraint with correct second constraint."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.and_(moon)
        assert combined.constraints[1] is moon

    def test_or_method_creates_or_constraint_type(
        self, sun_constraint, moon_constraint
    ):
        """Test or_ method creates OrConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert isinstance(combined, OrConstraint)

    def test_or_method_creates_or_constraint_length(
        self, sun_constraint, moon_constraint
    ):
        """Test or_ method creates OrConstraint with correct length."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert len(combined.constraints) == 2

    def test_or_method_creates_or_constraint_first(
        self, sun_constraint, moon_constraint
    ):
        """Test or_ method creates OrConstraint with correct first constraint."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert combined.constraints[0] is sun

    def test_or_method_creates_or_constraint_second(
        self, sun_constraint, moon_constraint
    ):
        """Test or_ method creates OrConstraint with correct second constraint."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun.or_(moon)
        assert combined.constraints[1] is moon

    def test_not_method_creates_not_constraint_type(self, sun_constraint):
        """Test not_ method creates NotConstraint type."""
        sun = sun_constraint
        negated = sun.not_()
        assert isinstance(negated, NotConstraint)

    def test_not_method_creates_not_constraint_constraint(self, sun_constraint):
        """Test not_ method creates NotConstraint with correct constraint."""
        sun = sun_constraint
        negated = sun.not_()
        assert negated.constraint is sun

    def test_and_operator_overload_type(self, sun_constraint, moon_constraint):
        """Test __and__ operator creates AndConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun & moon
        assert isinstance(combined, AndConstraint)

    def test_and_operator_overload_length(self, sun_constraint, moon_constraint):
        """Test __and__ operator creates AndConstraint with correct length."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun & moon
        assert len(combined.constraints) == 2

    def test_or_operator_overload_type(self, sun_constraint, moon_constraint):
        """Test __or__ operator creates OrConstraint type."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun | moon
        assert isinstance(combined, OrConstraint)

    def test_or_operator_overload_length(self, sun_constraint, moon_constraint):
        """Test __or__ operator creates OrConstraint with correct length."""
        sun = sun_constraint
        moon = moon_constraint
        combined = sun | moon
        assert len(combined.constraints) == 2

    def test_invert_operator_overload_type(self, sun_constraint):
        """Test __invert__ operator creates NotConstraint type."""
        sun = sun_constraint
        negated = ~sun
        assert isinstance(negated, NotConstraint)

    def test_invert_operator_overload_constraint(self, sun_constraint):
        """Test __invert__ operator creates NotConstraint with correct constraint."""
        sun = sun_constraint
        negated = ~sun
        assert negated.constraint is sun

    def test_operator_chaining_type(self, sun_constraint, moon_constraint):
        """Test chaining of logical operators creates OrConstraint."""
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert isinstance(combined, OrConstraint)

    def test_operator_chaining_length(self, sun_constraint, moon_constraint):
        """Test chaining of logical operators has correct length."""
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert len(combined.constraints) == 2

    def test_operator_chaining_first_is_and(self, sun_constraint, moon_constraint):
        """Test chaining of logical operators first constraint is And."""
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert isinstance(combined.constraints[0], AndConstraint)

    def test_operator_chaining_second_is_sun(self, sun_constraint, moon_constraint):
        """Test chaining of logical operators second constraint is sun."""
        sun = sun_constraint
        moon = moon_constraint
        combined = (sun & moon) | sun
        assert combined.constraints[1] is sun

    def test_nested_logical_operations_type(self, sun_constraint, moon_constraint):
        """Test nested logical operations creates NotConstraint."""
        sun = sun_constraint
        moon = moon_constraint
        nested = ~(sun & moon)
        assert isinstance(nested, NotConstraint)

    def test_nested_logical_operations_constraint_type(
        self, sun_constraint, moon_constraint
    ):
        """Test nested logical operations constraint is AndConstraint."""
        sun = sun_constraint
        moon = moon_constraint
        nested = ~(sun & moon)
        assert isinstance(nested.constraint, AndConstraint)
