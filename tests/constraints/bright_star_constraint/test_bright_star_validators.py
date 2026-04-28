"""Pydantic validation tests for BrightStarConstraint."""

import pytest

import rust_ephem
from rust_ephem.constraints import BrightStarConstraint

_STARS = [(10.0, 20.0), (30.0, 40.0)]
_POLYGON = [(-0.25, -0.15), (0.25, -0.15), (0.25, 0.15), (-0.25, 0.15)]


class TestBrightStarPydanticValidators:
    def test_valid_circle_fov(self) -> None:
        c = BrightStarConstraint(stars=_STARS, fov_radius=0.5)
        assert c.fov_radius == 0.5
        assert c.fov_polygon is None

    def test_valid_polygon_fov(self) -> None:
        c = BrightStarConstraint(stars=_STARS, fov_polygon=_POLYGON)
        assert c.fov_polygon is not None
        assert c.fov_radius is None

    def test_valid_polygon_with_roll(self) -> None:
        c = BrightStarConstraint(stars=_STARS, fov_polygon=_POLYGON, roll_deg=45.0)
        assert c.roll_deg == 45.0

    def test_neither_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="fov_radius or fov_polygon"):
            BrightStarConstraint(stars=_STARS)

    def test_both_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            BrightStarConstraint(stars=_STARS, fov_radius=0.5, fov_polygon=_POLYGON)

    def test_polygon_too_few_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            BrightStarConstraint(stars=_STARS, fov_polygon=[(0.0, 0.0), (1.0, 0.0)])

    def test_roll_with_circle_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="roll_deg"):
            BrightStarConstraint(stars=_STARS, fov_radius=0.5, roll_deg=45.0)

    def test_roll_none_with_polygon_is_valid(self) -> None:
        c = BrightStarConstraint(stars=_STARS, fov_polygon=_POLYGON, roll_deg=None)
        assert c.roll_deg is None


class TestBrightStarRustValidators:
    """Tests for validation in the Rust-side Constraint.bright_star() factory."""

    def test_valid_circle(self) -> None:
        c = rust_ephem.Constraint.bright_star(stars=_STARS, fov_radius=0.5)
        assert c is not None

    def test_valid_polygon(self) -> None:
        c = rust_ephem.Constraint.bright_star(stars=_STARS, fov_polygon=_POLYGON)
        assert c is not None

    def test_empty_stars_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            rust_ephem.Constraint.bright_star(stars=[], fov_radius=0.5)

    def test_neither_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="fov_radius or fov_polygon"):
            rust_ephem.Constraint.bright_star(stars=_STARS)

    def test_both_fov_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            rust_ephem.Constraint.bright_star(
                stars=_STARS, fov_radius=0.5, fov_polygon=_POLYGON
            )

    def test_polygon_too_few_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            rust_ephem.Constraint.bright_star(
                stars=_STARS, fov_polygon=[(0.0, 0.0), (1.0, 0.0)]
            )

    def test_roll_with_circle_raises(self) -> None:
        with pytest.raises(ValueError, match="roll_deg"):
            rust_ephem.Constraint.bright_star(
                stars=_STARS, fov_radius=0.5, roll_deg=45.0
            )

    def test_fov_radius_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            rust_ephem.Constraint.bright_star(stars=_STARS, fov_radius=200.0)

    def test_roll_with_polygon(self) -> None:
        c = rust_ephem.Constraint.bright_star(
            stars=_STARS, fov_polygon=_POLYGON, roll_deg=0.0
        )
        assert c is not None
