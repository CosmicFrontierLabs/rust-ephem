"""Regression tests for the ``target_rolls=None`` roll-sweep batch path.

The Rust ``in_constraint_batch_constrained_at_every_roll`` entrypoint is
overridden by ``BrightStar`` (cached projections), ``SolarRoll`` (closed-form
optimum), and the ``AND``/``OR`` combinators (roll-independent child
hoisting).  Each override should agree with a simpler "AND across
``n_roll_samples`` fixed-roll batches" oracle, since both compute "violated
at every sampled roll" over the same uniformly-spaced roll grid.

The grid is kept small (3 times, 4-5 targets, 12 rolls) so the
``O(n_roll_samples)`` oracle stays cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import rust_ephem
from rust_ephem.constraints import (
    BrightStarConstraint,
    SolarRollConstraint,
    SunConstraint,
)

# 12 samples → 30° grid step, fine enough to mix True/False without making the
# oracle's per-step batch the dominant cost in the suite.
_N_ROLL_SAMPLES = 12
_INDICES = [0, 20, 40]

# ±0.25° u, ±0.15° v polygon: inscribed radius 0.15°, circumradius ≈0.292°.
_POLYGON = [(-0.25, -0.15), (0.25, -0.15), (0.25, 0.15), (-0.25, 0.15)]


def _per_roll_and(
    constraint: Any,
    ephem: rust_ephem.TLEEphemeris,
    ras: list[float],
    decs: list[float],
    n_roll_samples: int,
    indices: list[int],
) -> npt.NDArray[np.bool_]:
    """AND-reduce per-roll fixed-roll batches across the same grid the override uses."""
    n_targets = len(ras)
    step = 360.0 / n_roll_samples
    acc: npt.NDArray[np.bool_] | None = None
    for i in range(n_roll_samples):
        roll = i * step
        result = constraint.in_constraint_batch(
            ephem,
            ras,
            decs,
            indices=indices,
            target_rolls=[roll] * n_targets,
        )
        acc = result if acc is None else (acc & result)
    assert acc is not None
    return acc


def _swept(
    constraint: Any,
    ephem: rust_ephem.TLEEphemeris,
    ras: list[float],
    decs: list[float],
    n_roll_samples: int,
    indices: list[int],
) -> npt.NDArray[np.bool_]:
    result: npt.NDArray[np.bool_] = constraint.in_constraint_batch(
        ephem,
        ras,
        decs,
        indices=indices,
        target_rolls=None,
        n_roll_samples=n_roll_samples,
    )
    return result


def _offset_ra(ra: float, dec: float, delta_deg: float) -> float:
    cos_dec = np.cos(np.radians(dec))
    if abs(cos_dec) < 1e-9:
        return ra
    return float((ra + delta_deg / cos_dec) % 360.0)


class TestBareBrightStarSweep:
    """Polygon mode with ``roll_deg=None`` drives the cached-projection override."""

    def test_swept_matches_oracle(self, tle_ephemeris: rust_ephem.TLEEphemeris) -> None:
        # One always-blocked star (within inscribed radius), one sometimes-blocked
        # (between inscribed and circumradius), one never-blocked (beyond
        # circumradius), one with no nearby star.  Mix guarantees the swept
        # result has both True and False rows.
        targets = [(10.0, 5.0), (40.0, -15.0), (80.0, 30.0), (200.0, 0.0)]
        ras = [t[0] for t in targets]
        decs = [t[1] for t in targets]
        stars = [
            (_offset_ra(ras[0], decs[0], 0.05), decs[0]),
            (_offset_ra(ras[1], decs[1], 0.20), decs[1]),
            (_offset_ra(ras[2], decs[2], 0.40), decs[2]),
        ]
        c = BrightStarConstraint(stars=stars, fov_polygon=_POLYGON, roll_deg=None)

        swept = _swept(c, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)
        oracle = _per_roll_and(c, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)

        assert swept.shape == (len(ras), len(_INDICES))
        np.testing.assert_array_equal(swept, oracle)
        # Sanity: mixed True/False is required for the equality to be a real
        # test of the override rather than a degenerate all-same result.
        assert swept.any() and not swept.all()


class TestBareSolarRollSweep:
    """``SolarRoll`` with ``roll_deg=None`` drives its closed-form override."""

    @pytest.mark.parametrize(
        "panel_normal",
        [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    )
    def test_swept_matches_oracle(
        self,
        tle_ephemeris: rust_ephem.TLEEphemeris,
        panel_normal: tuple[float, float, float],
    ) -> None:
        # tolerance_deg=5° with a 30° grid: violated when the optimum's phase
        # mod 30° lies in (5°, 25°), giving mixed True/False over the target
        # grid below.
        c = SolarRollConstraint(tolerance_deg=5.0, panel_normal=panel_normal)
        ras = [10.0, 90.0, 170.0, 250.0, 330.0]
        decs = [0.0, 30.0, -30.0, 60.0, -60.0]

        swept = _swept(c, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)
        oracle = _per_roll_and(c, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)

        assert swept.shape == (len(ras), len(_INDICES))
        np.testing.assert_array_equal(swept, oracle)
        assert swept.any() and not swept.all()


class TestAndMixedHoisting:
    """``AND`` of a roll-dependent and roll-independent child exercises
    ``AndEvaluator``'s override: the sun child is hoisted out of the sweep
    loop and AND-combined once with the per-step solar-roll result."""

    def test_solar_roll_and_sun(self, tle_ephemeris: rust_ephem.TLEEphemeris) -> None:
        roll_dep = SolarRollConstraint(tolerance_deg=5.0)
        # min_angle=90° → roughly the sunward hemisphere is violated, giving a
        # ~50% mix that overlaps the solar-roll pattern below.
        roll_indep = SunConstraint(min_angle=90.0)
        combined = roll_dep & roll_indep

        ras = [10.0, 90.0, 170.0, 250.0, 330.0]
        decs = [0.0, 30.0, -30.0, 60.0, -60.0]

        swept = _swept(combined, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)
        oracle = _per_roll_and(
            combined, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES
        )

        np.testing.assert_array_equal(swept, oracle)
        assert swept.any() and not swept.all()


class TestOrMixedHoisting:
    """``OR`` of a roll-dependent and roll-independent child exercises
    ``OrEvaluator``'s override: the indep result is OR'd in once after the
    AND-across-rolls of the dep result."""

    def test_solar_roll_or_sun(self, tle_ephemeris: rust_ephem.TLEEphemeris) -> None:
        roll_dep = SolarRollConstraint(tolerance_deg=5.0)
        # min_angle=90° gives a ~50% mix that exercises the
        # `result |= v_indep` post-step in the override (and is neither all-True
        # — which would mask any sweep bug — nor all-False).
        roll_indep = SunConstraint(min_angle=90.0)
        combined = roll_dep | roll_indep

        ras = [10.0, 90.0, 170.0, 250.0, 330.0]
        decs = [0.0, 30.0, -30.0, 60.0, -60.0]

        swept = _swept(combined, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES)
        oracle = _per_roll_and(
            combined, tle_ephemeris, ras, decs, _N_ROLL_SAMPLES, _INDICES
        )

        np.testing.assert_array_equal(swept, oracle)
        assert swept.any() and not swept.all()
