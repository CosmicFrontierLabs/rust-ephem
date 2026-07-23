"""Test suite for annotated TOML export/import of constraint configs."""

from pathlib import Path
from typing import Any

import pytest

from rust_ephem.constraint_toml import (
    load_constraint_toml,
    parse_constraint_toml,
    write_constraint_toml,
)
from rust_ephem.constraints import (
    AirmassConstraint,
    AltAzConstraint,
    AndConstraint,
    AtLeastConstraint,
    BodyConstraint,
    BoresightOffsetConstraint,
    BrightStarConstraint,
    ConstraintConfig,
    DaytimeConstraint,
    EarthLimbConstraint,
    EclipseConstraint,
    MoonConstraint,
    MoonPhaseConstraint,
    NotConstraint,
    OrbitPoleConstraint,
    OrbitRamConstraint,
    OrConstraint,
    SAAConstraint,
    SolarRollConstraint,
    SunConstraint,
    XorConstraint,
)


class TestLeafConstraintToToml:
    """A single, non-combinator constraint renders as one flat table."""

    def test_type_key_present(self: Any, sun_constraint: SunConstraint) -> None:
        assert 'type = "sun"' in sun_constraint.to_toml()

    def test_required_field_value_written(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        assert "min_angle = 45.0" in sun_constraint.to_toml()

    def test_description_rendered_as_comment(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        assert (
            "# min_angle -- Minimum angle from Sun in degrees"
            in sun_constraint.to_toml()
        )

    def test_bounds_rendered_in_comment(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        assert ">= 0.0, <= 180.0" in sun_constraint.to_toml()

    def test_unset_optional_field_not_written_as_key(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        toml_str = sun_constraint.to_toml()
        assert "\nmax_angle" not in toml_str

    def test_unset_optional_field_documented_as_comment(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        assert "# max_angle (not set) --" in sun_constraint.to_toml()

    def test_set_optional_field_written_as_key(self: Any) -> None:
        config = SunConstraint(min_angle=45.0, max_angle=90.0)
        assert "max_angle = 90.0" in config.to_toml()

    def test_docstring_summary_rendered_as_comment(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        assert "# Sun proximity constraint" in sun_constraint.to_toml()

    def test_no_definitions_or_expression_for_plain_leaf(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        toml_str = sun_constraint.to_toml()
        assert "definitions" not in toml_str
        assert "\nexpression" not in toml_str


class TestCombinatorToToml:
    """Boolean combinators flatten into named definitions plus an expression
    string instead of nested TOML tables."""

    def test_or_combinator_has_no_nested_tables(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = OrConstraint(constraints=[sun_constraint, moon_constraint])
        toml_str = config.to_toml()
        assert "[[" not in toml_str
        assert "[constraint]" not in toml_str

    def test_or_combinator_expression(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = OrConstraint(constraints=[sun_constraint, moon_constraint])
        assert 'expression = "sun | moon"' in config.to_toml()

    def test_and_combinator_expression(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = AndConstraint(constraints=[sun_constraint, moon_constraint])
        assert 'expression = "sun & moon"' in config.to_toml()

    def test_not_combinator_expression(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        config = NotConstraint(constraint=sun_constraint)
        assert 'expression = "~sun"' in config.to_toml()

    def test_at_least_combinator_expression(
        self: Any,
        sun_constraint: SunConstraint,
        moon_constraint: MoonConstraint,
        eclipse_constraint: EclipseConstraint,
    ) -> None:
        config = AtLeastConstraint(
            min_violated=2,
            constraints=[sun_constraint, moon_constraint, eclipse_constraint],
        )
        assert 'expression = "at_least(2, sun, moon, eclipse)"' in config.to_toml()

    def test_and_of_or_needs_no_parens(
        self: Any,
        sun_constraint: SunConstraint,
        moon_constraint: MoonConstraint,
        eclipse_constraint: EclipseConstraint,
    ) -> None:
        # AND binds tighter than OR, so this reads left-to-right with no parens.
        config = OrConstraint(
            constraints=[
                AndConstraint(constraints=[sun_constraint, moon_constraint]),
                NotConstraint(constraint=eclipse_constraint),
            ]
        )
        assert 'expression = "sun & moon | ~eclipse"' in config.to_toml()

    def test_not_of_or_needs_parens(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = NotConstraint(
            constraint=OrConstraint(constraints=[sun_constraint, moon_constraint])
        )
        assert 'expression = "~(sun | moon)"' in config.to_toml()

    def test_each_child_gets_its_own_flat_definition(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = AndConstraint(constraints=[sun_constraint, moon_constraint])
        toml_str = config.to_toml()
        assert "[definitions.sun]" in toml_str
        assert "[definitions.moon]" in toml_str

    def test_duplicate_leaf_types_get_unique_names(self: Any) -> None:
        config = OrConstraint(
            constraints=[SunConstraint(min_angle=1.0), SunConstraint(min_angle=2.0)]
        )
        toml_str = config.to_toml()
        assert "[definitions.sun]" in toml_str
        assert "[definitions.sun_2]" in toml_str
        assert 'expression = "sun | sun_2"' in toml_str

    def test_expression_legend_comment_present(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        config = OrConstraint(constraints=[sun_constraint, moon_constraint])
        assert "Python's own precedence" in config.to_toml()


class TestBoresightOffsetToToml:
    """BoresightOffsetConstraint keeps its own scalar fields flat and expresses
    its wrapped constraint as a ``constraint_expression`` string."""

    def test_root_wrapper_has_no_nested_table(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        config = BoresightOffsetConstraint(constraint=sun_constraint, roll_deg=5.0)
        toml_str = config.to_toml()
        assert "[constraint]" not in toml_str
        assert "constraint_expression" in toml_str

    def test_root_wrapper_scalar_fields_at_top_level(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        config = BoresightOffsetConstraint(constraint=sun_constraint, roll_deg=5.0)
        toml_str = config.to_toml()
        assert 'type = "boresight_offset"' in toml_str
        assert "roll_deg = 5.0" in toml_str

    def test_wrapped_constraint_becomes_a_definition(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        config = BoresightOffsetConstraint(constraint=sun_constraint, roll_deg=5.0)
        toml_str = config.to_toml()
        assert "[definitions.sun]" in toml_str
        assert 'constraint_expression = "sun"' in toml_str

    def test_wrapper_inside_combinator_becomes_its_own_definition(
        self: Any, sun_constraint: SunConstraint, moon_constraint: MoonConstraint
    ) -> None:
        wrapped = BoresightOffsetConstraint(constraint=sun_constraint, roll_deg=5.0)
        config = OrConstraint(constraints=[wrapped, moon_constraint])
        toml_str = config.to_toml()
        assert "[definitions.boresight_offset]" in toml_str
        assert 'expression = "boresight_offset | moon"' in toml_str


class TestConstraintTomlRoundTrip:
    """Test that TOML output parses back into an equivalent constraint config."""

    @pytest.mark.parametrize(
        "config",
        [
            SunConstraint(min_angle=45.0),
            SunConstraint(min_angle=45.0, max_angle=90.0),
            EarthLimbConstraint(min_angle=15.0, include_refraction=True),
            BodyConstraint(body="jupiter", min_angle=5.0),
            BodyConstraint(
                body="jupiter",
                fov_polygon=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
                roll_deg=10.0,
            ),
            SAAConstraint(polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            AirmassConstraint(min_airmass=1.0, max_airmass=2.0),
            AirmassConstraint(max_airmass=2.0),
            MoonPhaseConstraint(max_illumination=0.5),
            MoonPhaseConstraint(
                min_illumination=0.1,
                max_illumination=0.5,
                min_distance=10.0,
                max_distance=90.0,
                enforce_when_below_horizon=True,
                moon_visibility="partial",
            ),
            DaytimeConstraint(),
            DaytimeConstraint(twilight="astronomical"),
            AltAzConstraint(min_altitude=10.0, max_altitude=80.0),
            AltAzConstraint(polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
            SolarRollConstraint(tolerance_deg=15.0),
            SolarRollConstraint(tolerance_deg=15.0, panel_normal=(0.0, 0.0, 1.0)),
            OrbitRamConstraint(min_angle=5.0, max_angle=90.0),
            OrbitPoleConstraint(min_angle=5.0, earth_limb_pole=True),
            BrightStarConstraint(stars=[(10.0, 20.0), (30.0, 40.0)], fov_radius=1.0),
            BrightStarConstraint(
                stars=[(10.0, 20.0)],
                fov_polygon=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
                roll_deg=15.0,
            ),
            AndConstraint(
                constraints=[
                    SunConstraint(min_angle=1.0),
                    MoonConstraint(min_angle=2.0),
                ]
            ),
            OrConstraint(
                constraints=[
                    SunConstraint(min_angle=1.0),
                    MoonConstraint(min_angle=2.0),
                ]
            ),
            XorConstraint(
                constraints=[
                    SunConstraint(min_angle=1.0),
                    MoonConstraint(min_angle=2.0),
                ]
            ),
            AtLeastConstraint(
                min_violated=1,
                constraints=[
                    SunConstraint(min_angle=1.0),
                    MoonConstraint(min_angle=2.0),
                ],
            ),
            NotConstraint(constraint=SunConstraint(min_angle=1.0)),
            NotConstraint(
                constraint=OrConstraint(
                    constraints=[
                        SunConstraint(min_angle=1.0),
                        MoonConstraint(min_angle=2.0),
                    ]
                )
            ),
            OrConstraint(
                constraints=[
                    AndConstraint(
                        constraints=[
                            SunConstraint(min_angle=1.0),
                            MoonConstraint(min_angle=2.0),
                        ]
                    ),
                    NotConstraint(constraint=EclipseConstraint(umbra_only=False)),
                ]
            ),
            AtLeastConstraint(
                min_violated=2,
                constraints=[
                    SunConstraint(min_angle=1.0),
                    NotConstraint(constraint=MoonConstraint(min_angle=2.0)),
                    EclipseConstraint(),
                ],
            ),
            BoresightOffsetConstraint(
                constraint=SunConstraint(min_angle=1.0), roll_deg=5.0
            ),
            BoresightOffsetConstraint(
                constraint=AndConstraint(
                    constraints=[
                        SunConstraint(min_angle=1.0),
                        MoonConstraint(min_angle=2.0),
                    ]
                ),
                roll_deg=5.0,
            ),
            OrConstraint(
                constraints=[
                    BoresightOffsetConstraint(
                        constraint=SunConstraint(min_angle=1.0), roll_deg=1.0
                    ),
                    BoresightOffsetConstraint(
                        constraint=SunConstraint(min_angle=2.0), roll_deg=2.0
                    ),
                ]
            ),
        ],
    )
    def test_round_trip_equals_original(self: Any, config: ConstraintConfig) -> None:
        restored = parse_constraint_toml(config.to_toml())
        assert restored == config

    def test_round_trip_preserves_type(
        self: Any, sun_constraint: SunConstraint
    ) -> None:
        restored = parse_constraint_toml(sun_constraint.to_toml())
        assert isinstance(restored, SunConstraint)

    def test_file_round_trip(
        self: Any, sun_constraint: SunConstraint, tmp_path: Path
    ) -> None:
        path = tmp_path / "sun.toml"
        write_constraint_toml(sun_constraint, path)
        assert load_constraint_toml(path) == sun_constraint

    def test_to_toml_file_method(
        self: Any, sun_constraint: SunConstraint, tmp_path: Path
    ) -> None:
        path = tmp_path / "sun.toml"
        sun_constraint.to_toml_file(path)
        assert load_constraint_toml(path) == sun_constraint


class TestExpressionErrors:
    """Malformed constraint_expression strings raise clear errors."""

    @pytest.mark.parametrize(
        "expression",
        [
            "sun &",
            "sun | )",
            "at_least(sun, moon)",
            "unknown_name",
            "sun & (moon",
        ],
    )
    def test_invalid_expression_raises(self: Any, expression: str) -> None:
        with pytest.raises(ValueError):
            parse_constraint_toml(f'expression = "{expression}"\n')
