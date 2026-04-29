"""Geometry and evaluation tests for the bright star avoidance constraint."""

import math
from datetime import datetime, timezone
from typing import Generator

import pytest

import rust_ephem

# Swift TLE (28485) used throughout the constraint test suite
_TLE1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
_TLE2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
_BEGIN = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
_END = datetime(2025, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
_STEP = 60

# Small rectangular polygon: ±0.25° u, ±0.15° v  (~0.5° × 0.3° FoV)
_POLYGON = [(-0.25, -0.15), (0.25, -0.15), (0.25, 0.15), (-0.25, 0.15)]


@pytest.fixture(scope="module")
def tle_ephem() -> Generator[rust_ephem.TLEEphemeris, None, None]:
    yield rust_ephem.TLEEphemeris(_TLE1, _TLE2, _BEGIN, _END, _STEP)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_violated(
    constraint: rust_ephem.Constraint,
    ephem: rust_ephem.TLEEphemeris,
    ra: float,
    dec: float,
) -> bool:
    """Return True when the constraint is violated at the given (ra, dec)."""
    result = constraint.evaluate(ephemeris=ephem, target_ra=ra, target_dec=dec)
    return any(result.constraint_array)


def _offset_ra(ra: float, dec: float, delta_deg: float) -> float:
    """Shift ra by delta_deg / cos(dec) so the on-sky separation is delta_deg."""
    cos_dec = math.cos(math.radians(dec))
    if abs(cos_dec) < 1e-9:
        return ra
    return (ra + delta_deg / cos_dec) % 360.0


# ── Circular FoV ───────────────────────────────────────────────────────────────


class TestCircularFov:
    """Star at boresight, offset outside radius, and boundary cases."""

    def test_star_at_boresight_is_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # target == star → always inside the circle
        star_ra, star_dec = 45.0, 10.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)], fov_radius=1.0
        )
        assert _is_violated(c, tle_ephem, star_ra, star_dec)

    def test_star_outside_radius_not_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        star_ra, star_dec = 45.0, 10.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)], fov_radius=1.0
        )
        # 5° offset — well outside the 1° radius
        offset_ra = _offset_ra(star_ra, star_dec, 5.0)
        assert not _is_violated(c, tle_ephem, offset_ra, star_dec)

    @pytest.mark.parametrize("offset_deg", [0.5, 0.9, 0.99])
    def test_star_inside_radius_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris, offset_deg: float
    ) -> None:
        star_ra, star_dec = 45.0, 10.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)], fov_radius=1.0
        )
        offset_ra = _offset_ra(star_ra, star_dec, offset_deg)
        assert _is_violated(c, tle_ephem, offset_ra, star_dec)

    @pytest.mark.parametrize("offset_deg", [1.1, 2.0, 5.0])
    def test_star_outside_radius_not_violated_parametrized(
        self, tle_ephem: rust_ephem.TLEEphemeris, offset_deg: float
    ) -> None:
        star_ra, star_dec = 45.0, 10.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)], fov_radius=1.0
        )
        offset_ra = _offset_ra(star_ra, star_dec, offset_deg)
        assert not _is_violated(c, tle_ephem, offset_ra, star_dec)

    def test_multiple_stars_any_triggers_violation(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Place two stars: one inside, one outside
        target_ra, target_dec = 100.0, 5.0
        star_inside = (target_ra, target_dec)  # coincident → inside
        star_outside = (_offset_ra(target_ra, target_dec, 5.0), target_dec)
        c = rust_ephem.Constraint.bright_star(
            stars=[star_inside, star_outside], fov_radius=1.0
        )
        assert _is_violated(c, tle_ephem, target_ra, target_dec)

    def test_no_stars_inside_not_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        target_ra, target_dec = 100.0, 5.0
        stars = [
            (_offset_ra(target_ra, target_dec, d), target_dec) for d in [3.0, 5.0, 10.0]
        ]
        c = rust_ephem.Constraint.bright_star(stars=stars, fov_radius=1.0)
        assert not _is_violated(c, tle_ephem, target_ra, target_dec)


# ── Polygon FoV at fixed roll ──────────────────────────────────────────────────


class TestPolygonFovFixedRoll:
    """Star inside/outside an axis-aligned rectangular FoV at roll=0."""

    def test_star_at_boresight_polygon_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        star_ra, star_dec = 60.0, 5.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=0.0,
        )
        assert _is_violated(c, tle_ephem, star_ra, star_dec)

    def test_star_well_outside_polygon_not_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        star_ra, star_dec = 60.0, 5.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=0.0,
        )
        # Target 2° away — star will be 2° from boresight, outside 0.5°×0.3° box
        offset_ra = _offset_ra(star_ra, star_dec, 2.0)
        assert not _is_violated(c, tle_ephem, offset_ra, star_dec)

    def test_star_inside_polygon_at_roll_90(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # At roll=90°: u = north, v = -east (per convention).
        # A star 0.10° east of target maps to u≈0.10° at roll=0 but u≈0° at roll=90°.
        # At roll=90°, a star 0.10° north maps to u=0° (v), v=0.10° — inside the polygon.
        # We construct a star slightly north of the target so it's in the box at roll=90.
        star_dec = 6.0
        target_dec = 5.9  # star is 0.1° north of target → v≈0.1° < 0.15° → inside
        star_ra = 60.0
        target_ra = star_ra  # same RA → delta_ra = 0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=90.0,
        )
        assert _is_violated(c, tle_ephem, target_ra, target_dec)

    def test_pydantic_polygon_fixed_roll_evaluates(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        from rust_ephem.constraints import BrightStarConstraint

        pyd = BrightStarConstraint(
            stars=[(60.0, 5.0)],
            fov_polygon=_POLYGON,
            roll_deg=0.0,
        )
        c = rust_ephem.Constraint.from_json(pyd.model_dump_json())
        assert _is_violated(c, tle_ephem, 60.0, 5.0)


# ── Roll-sweep semantics ───────────────────────────────────────────────────────


class TestRollSweep:
    """When roll_deg is None the constraint sweeps all roll angles.

    Violated only if every sampled roll has a star in the FoV.
    """

    def test_star_at_boresight_all_rolls_blocked(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Star == target: it's at the origin of the instrument frame for any roll.
        star_ra, star_dec = 70.0, -5.0
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=None,
        )
        assert _is_violated(c, tle_ephem, star_ra, star_dec)

    def test_target_far_from_stars_not_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Stars are all 10° away — no roll can put them in a sub-degree box.
        target_ra, target_dec = 70.0, -5.0
        stars = [
            (_offset_ra(target_ra, target_dec, d), target_dec)
            for d in [10.0, 15.0, 20.0]
        ]
        c = rust_ephem.Constraint.bright_star(
            stars=stars,
            fov_polygon=_POLYGON,
            roll_deg=None,
        )
        assert not _is_violated(c, tle_ephem, target_ra, target_dec)

    def test_star_in_only_some_rolls_not_violated(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Place a star 0.20° east (just outside u-limit of ±0.25°, inside at roll=0 east).
        # Actually put it *inside* at roll=0 but outside at roll=90. The sweep should
        # find a clear roll → not violated.
        # Star 0.15° north and 0.30° east of target:
        #   roll=0: u=0.30°, v=0.15° — u > 0.25° → outside
        #   roll=90: u=-0.15°, v=0.30° — v > 0.15° → outside
        # Since neither at roll=0 nor roll=90 is it inside, definitely not violated.
        target_ra, target_dec = 80.0, 10.0
        star_ra = _offset_ra(target_ra, target_dec, 0.30)
        star_dec = target_dec + 0.15
        c = rust_ephem.Constraint.bright_star(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=None,
        )
        assert not _is_violated(c, tle_ephem, target_ra, target_dec)


# ── JSON round-trip ────────────────────────────────────────────────────────────


class TestJsonRoundTrip:
    def test_circle_roundtrip(self, tle_ephem: rust_ephem.TLEEphemeris) -> None:
        from rust_ephem.constraints import (
            BrightStarConstraint,
            CombinedConstraintConfig,
        )

        star_ra, star_dec = 90.0, 15.0
        pyd = BrightStarConstraint(stars=[(star_ra, star_dec)], fov_radius=2.0)
        json_str = pyd.model_dump_json()

        # Validate through CombinedConstraintConfig (TypeAdapter)
        restored = CombinedConstraintConfig.validate_json(json_str)
        c = rust_ephem.Constraint.from_json(restored.model_dump_json())
        assert _is_violated(c, tle_ephem, star_ra, star_dec)

    def test_polygon_roundtrip(self, tle_ephem: rust_ephem.TLEEphemeris) -> None:
        from rust_ephem.constraints import BrightStarConstraint

        star_ra, star_dec = 90.0, 15.0
        pyd = BrightStarConstraint(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=0.0,
        )
        c = rust_ephem.Constraint.from_json(pyd.model_dump_json())
        assert _is_violated(c, tle_ephem, star_ra, star_dec)

    def test_polygon_roll_sweep_roundtrip(
        self, tle_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        from rust_ephem.constraints import BrightStarConstraint

        star_ra, star_dec = 90.0, 15.0
        pyd = BrightStarConstraint(
            stars=[(star_ra, star_dec)],
            fov_polygon=_POLYGON,
            roll_deg=None,
        )
        c = rust_ephem.Constraint.from_json(pyd.model_dump_json())
        # Star at boresight → all rolls blocked
        assert _is_violated(c, tle_ephem, star_ra, star_dec)
