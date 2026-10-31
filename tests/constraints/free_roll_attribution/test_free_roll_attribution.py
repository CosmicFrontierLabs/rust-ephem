"""Tests for free-roll (``target_roll=None``) evaluation and its cause attribution.

A free-roll evaluation asks whether *some* spacecraft orientation clears the
constraint tree, which makes two things easy to get wrong:

* the sweep must happen inside Rust, or the result carries no visibility windows
  and therefore no ``start_cause``/``end_cause`` at all; and
* the sweep must be *coordinated* — one shared candidate roll applied to every
  boresight offset at once, with each instrument's mounting angle preserved.
  Evaluating each offset independently lets two instruments drift into
  orientations the spacecraft could never hold simultaneously.

The oracle throughout is "AND across fixed-roll evaluations on the same roll
grid", which is the definition of "violated at every roll" spelled out directly.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import rust_ephem
from rust_ephem.constraints import (
    BoresightOffsetConstraint,
    EarthLimbConstraint,
    SolarRollConstraint,
    SunConstraint,
)

# 24 samples → 15° grid.  Fine enough for the geometries below to resolve a
# clear roll, coarse enough that the O(n_roll_samples) oracle stays cheap.
_N_ROLL_SAMPLES = 24

_TLE1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
_TLE2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"


@pytest.mark.parametrize("combination", ["xor", "at_least"])
@pytest.mark.parametrize("subset", [False, True])
def test_empty_altaz_roll_combination(
    ensure_planetary_data: Any, combination: str, subset: bool
) -> None:
    """Empty masks retain their time axis; release panics must not kill pytest."""
    script = dedent("""
        import sys
        from datetime import datetime, timedelta, timezone
        import rust_ephem
        from rust_ephem.constraints import (
            AltAzConstraint, AtLeastConstraint, SolarRollConstraint,
        )

        rust_ephem.ensure_planetary_ephemeris(
            py_path=sys.argv[1], download_if_missing=False,
        )
        begin = datetime(2025, 9, 23, tzinfo=timezone.utc)
        ephem = rust_ephem.GroundEphemeris(
            latitude=34, longitude=-118, height=100,
            begin=begin, end=begin + timedelta(minutes=10), step_size=300,
        )
        altaz = AltAzConstraint(min_altitude=20)
        roll = SolarRollConstraint(tolerance_deg=20)
        constraint = (
            altaz ^ roll if sys.argv[2] == "xor" else
            AtLeastConstraint(min_violated=1, constraints=[altaz, roll])
        )
        kwargs = {"indices": [0, 2]} if sys.argv[3] == "True" else {}
        n_times = 2 if kwargs else 3
        assert constraint.evaluate_batch(
            ephem, [], [], n_roll_samples=12, **kwargs,
        ) == []
        assert constraint.in_constraint_batch(
            ephem, [], [], n_roll_samples=12, **kwargs,
        ).shape == (0, n_times)
        assert altaz.in_constraint_batch(ephem, [], [], **kwargs).shape == (0, n_times)
    """)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parents[2] / "test_data" / "de440s.bsp"),
            combination,
            str(subset),
        ],
        # Match pytest's imported package; the checkout may lack the wheel's extension.
        cwd=Path(rust_ephem.__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture
def short_ephem(ensure_planetary_data: Any) -> rust_ephem.TLEEphemeris:
    """One hour at 1-minute cadence: enough samples to form windows, still fast."""
    return rust_ephem.TLEEphemeris(
        _TLE1,
        _TLE2,
        datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 9, 23, 1, 0, 0, tzinfo=timezone.utc),
        60,
    )


@pytest.fixture
def month_ephem(ensure_planetary_data: Any) -> rust_ephem.TLEEphemeris:
    """A month at 1-hour cadence, so the Sun sweeps ~30° past the target.

    The complementary-roll-range scenario needs the *combined* result to change
    over time while each individual instrument's own result does not; only the
    Sun's annual motion moves the geometry that far.
    """
    return rust_ephem.TLEEphemeris(
        _TLE1,
        _TLE2,
        datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 10, 23, 0, 0, 0, tzinfo=timezone.utc),
        3600,
    )


def _sun_relative_target(
    ephem: rust_ephem.TLEEphemeris, offset_deg: float
) -> tuple[float, float]:
    """A target ``offset_deg`` away from the Sun in RA at the first timestamp."""
    sun = ephem.sun[0]
    return (sun.ra.deg + offset_deg) % 360.0, sun.dec.deg


def _fixed_roll_and(
    constraint: Any,
    ephem: rust_ephem.TLEEphemeris,
    target_ra: float,
    target_dec: float,
    n_roll_samples: int = _N_ROLL_SAMPLES,
) -> npt.NDArray[np.bool_]:
    """Oracle: violated at every roll on the grid the free-roll sweep uses."""
    step = 360.0 / n_roll_samples
    acc: npt.NDArray[np.bool_] | None = None
    for i in range(n_roll_samples):
        row = np.asarray(
            constraint.evaluate(
                ephemeris=ephem,
                target_ra=target_ra,
                target_dec=target_dec,
                target_roll=i * step,
            ).constraint_array,
            dtype=bool,
        )
        acc = row if acc is None else (acc & row)
    assert acc is not None
    return acc


def _boresight(pitch_deg: float, roll_deg: float, inner: Any) -> Any:
    return BoresightOffsetConstraint(
        pitch_deg=pitch_deg, yaw_deg=0.0, roll_deg=roll_deg, constraint=inner
    )


class TestFreeRollResultsAreRustBacked:
    """A free-roll result must be a real evaluation result, not a bare mask."""

    def test_visibility_windows_are_populated(
        self, short_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Regression: the free-roll path used to combine per-roll boolean masks in
        # Python and return a result with no backing Rust result, so `visibility`
        # was always empty and no window ever carried a cause.
        target_ra, target_dec = _sun_relative_target(short_ephem, 60.0)
        constraint = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0))

        result = constraint.evaluate(
            ephemeris=short_ephem,
            target_ra=target_ra,
            target_dec=target_dec,
            n_roll_samples=_N_ROLL_SAMPLES,
        )

        assert not any(result.constraint_array), "scenario should be fully visible"
        assert len(result.visibility) == 1
        assert result.cause_value_keys

    def test_partially_visible_windows_carry_causes(
        self, short_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        target_ra, target_dec = _sun_relative_target(short_ephem, 10.0)
        constraint = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0)) & (
            EarthLimbConstraint(min_angle=25.0)
        )

        result = constraint.evaluate(
            ephemeris=short_ephem,
            target_ra=target_ra,
            target_dec=target_dec,
            n_roll_samples=_N_ROLL_SAMPLES,
        )

        violated = np.asarray(result.constraint_array, dtype=bool)
        assert violated.any() and not violated.all(), "need a real window boundary"

        timestamps = list(result.timestamps)
        interior_boundaries = 0
        for window in result.visibility:
            if timestamps.index(window.start_time) > 0:
                interior_boundaries += 1
                assert window.start_cause, "interior window start needs a cause"
            if timestamps.index(window.end_time) < len(timestamps) - 1:
                interior_boundaries += 1
                assert window.end_cause, "interior window end needs a cause"
        assert interior_boundaries > 0

    def test_cause_tags_resolve_through_cause_value_keys(
        self, short_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        target_ra, target_dec = _sun_relative_target(short_ephem, 10.0)
        constraint = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0)) & (
            EarthLimbConstraint(min_angle=25.0)
        )

        result = constraint.evaluate(
            ephemeris=short_ephem,
            target_ra=target_ra,
            target_dec=target_dec,
            n_roll_samples=_N_ROLL_SAMPLES,
        )

        for window in result.visibility:
            for tag in (window.start_cause or []) + (window.end_cause or []):
                assert tag in result.cause_value_keys
                for value_key in result.cause_value_keys[tag]:
                    assert value_key in result.constraint_values


class TestFreeRollApisAgree:
    """``evaluate``/``evaluate_batch``/``in_constraint``/``in_constraint_batch``.

    These used to disagree: the batch entry points never saw a free roll at all,
    because the cached Rust constraint carried a concrete ``roll_deg`` and so
    looked roll-independent to the sweep, which returned the fixed roll-0 answer.
    """

    @pytest.mark.parametrize("offset_deg", [10.0, 30.0, 60.0])
    def test_all_entry_points_match_the_oracle(
        self, short_ephem: rust_ephem.TLEEphemeris, offset_deg: float
    ) -> None:
        target_ra, target_dec = _sun_relative_target(short_ephem, offset_deg)
        constraint = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0)) & (
            EarthLimbConstraint(min_angle=25.0)
        )
        oracle = _fixed_roll_and(constraint, short_ephem, target_ra, target_dec)

        evaluated = np.asarray(
            constraint.evaluate(
                ephemeris=short_ephem,
                target_ra=target_ra,
                target_dec=target_dec,
                n_roll_samples=_N_ROLL_SAMPLES,
            ).constraint_array,
            dtype=bool,
        )
        batched = np.asarray(
            constraint.evaluate_batch(
                short_ephem,
                [target_ra],
                [target_dec],
                n_roll_samples=_N_ROLL_SAMPLES,
            )[0].constraint_array,
            dtype=bool,
        )
        mask = np.asarray(
            constraint.in_constraint_batch(
                short_ephem,
                [target_ra],
                [target_dec],
                n_roll_samples=_N_ROLL_SAMPLES,
            ),
            dtype=bool,
        )[0]
        pointwise = np.asarray(
            constraint.in_constraint(
                list(short_ephem.timestamp),
                short_ephem,
                target_ra,
                target_dec,
                n_roll_samples=_N_ROLL_SAMPLES,
            ),
            dtype=bool,
        )

        np.testing.assert_array_equal(evaluated, oracle)
        np.testing.assert_array_equal(batched, oracle)
        np.testing.assert_array_equal(mask, oracle)
        np.testing.assert_array_equal(pointwise, oracle)


class TestFreeRollMatchesOracleAcrossTreeShapes:
    """Every combinator and nesting shape must route the candidate roll downwards.

    A node that forgets to forward the swept roll to its children degrades to the
    fixed roll-0 answer without any error, so each shape is checked against the
    fixed-roll oracle rather than only against itself.
    """

    @pytest.mark.parametrize(
        "tree_name",
        [
            "solar_roll",
            "solar_roll_and_earth",
            "boresight_and_solar_roll",
            "not_boresight",
            "boresight_xor_earth",
            "nested_boresight",
        ],
    )
    @pytest.mark.parametrize("offset_deg", [25.0, 70.0])
    def test_matches_fixed_roll_oracle(
        self,
        short_ephem: rust_ephem.TLEEphemeris,
        tree_name: str,
        offset_deg: float,
    ) -> None:
        sun = SunConstraint(min_angle=45.0)
        earth = EarthLimbConstraint(min_angle=25.0)
        solar_roll = SolarRollConstraint(tolerance_deg=20.0)
        trees = {
            "solar_roll": solar_roll,
            "solar_roll_and_earth": solar_roll & earth,
            "boresight_and_solar_roll": _boresight(30.0, 0.0, sun) & solar_roll,
            "not_boresight": ~_boresight(30.0, 0.0, sun),
            "boresight_xor_earth": _boresight(30.0, 0.0, sun) ^ earth,
            # An offset nested inside another offset: the outer node rotates the
            # target, then must hand the same candidate roll to the inner one.
            "nested_boresight": _boresight(10.0, 45.0, _boresight(30.0, 0.0, sun)),
        }
        constraint = trees[tree_name]
        target_ra, target_dec = _sun_relative_target(short_ephem, offset_deg)

        free_roll = np.asarray(
            constraint.evaluate(
                ephemeris=short_ephem,
                target_ra=target_ra,
                target_dec=target_dec,
                n_roll_samples=_N_ROLL_SAMPLES,
            ).constraint_array,
            dtype=bool,
        )

        np.testing.assert_array_equal(
            free_roll,
            _fixed_roll_and(constraint, short_ephem, target_ra, target_dec),
        )


class TestCoordinatedRollSweep:
    """Every boresight offset must see the *same* candidate roll, mounting included."""

    def test_instrument_mounting_angles_stay_relative(
        self, short_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # Two instruments 180° apart on the same 30° cone.  A target 20° from the
        # Sun leaves each instrument its own clear rolls, but they are disjoint:
        # whenever one points away from the Sun the other points at it, so no
        # single spacecraft orientation clears both.  Collapsing both mountings
        # onto the same swept angle would make them identical and report the
        # target as reachable.
        target_ra, target_dec = _sun_relative_target(short_ephem, 20.0)
        instrument_a = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0))
        instrument_b = _boresight(30.0, 180.0, SunConstraint(min_angle=45.0))
        either_blocked = instrument_a | instrument_b

        def free_roll(constraint: Any) -> npt.NDArray[np.bool_]:
            return np.asarray(
                constraint.evaluate(
                    ephemeris=short_ephem,
                    target_ra=target_ra,
                    target_dec=target_dec,
                    n_roll_samples=_N_ROLL_SAMPLES,
                ).constraint_array,
                dtype=bool,
            )

        assert not free_roll(instrument_a).any(), "A alone has a clear roll"
        assert not free_roll(instrument_b).any(), "B alone has a clear roll"
        assert free_roll(either_blocked).all(), "no roll clears both instruments"
        np.testing.assert_array_equal(
            free_roll(either_blocked),
            _fixed_roll_and(either_blocked, short_ephem, target_ra, target_dec),
        )

    def test_boundary_gets_a_cause_when_no_leaf_changes_alone(
        self, month_ephem: rust_ephem.TLEEphemeris
    ) -> None:
        # The case the per-leaf-independent scheme could not attribute: the two
        # instruments' blocked roll ranges stop overlapping the full circle as the
        # Sun moves away, so the combined window opens while *neither* instrument
        # is ever blocked at every roll on its own.  Asking each leaf "are you
        # blocked at every roll?" gives a constant answer across the boundary and
        # therefore no cause; the witness-roll sweep reports the leaf whose state
        # changed at the orientation actually selected.
        target_ra, target_dec = _sun_relative_target(month_ephem, -20.0)
        instrument_a = _boresight(30.0, 0.0, SunConstraint(min_angle=45.0))
        instrument_b = _boresight(30.0, 180.0, SunConstraint(min_angle=45.0))
        either_blocked = instrument_a | instrument_b

        for leaf in (instrument_a, instrument_b):
            own = np.asarray(
                leaf.evaluate(
                    ephemeris=month_ephem,
                    target_ra=target_ra,
                    target_dec=target_dec,
                    n_roll_samples=_N_ROLL_SAMPLES,
                ).constraint_array,
                dtype=bool,
            )
            assert not own.any(), "each instrument alone always has a clear roll"

        result = either_blocked.evaluate(
            ephemeris=month_ephem,
            target_ra=target_ra,
            target_dec=target_dec,
            n_roll_samples=_N_ROLL_SAMPLES,
        )
        violated = np.asarray(result.constraint_array, dtype=bool)
        assert violated.any() and not violated.all(), "need a real window boundary"

        timestamps = list(result.timestamps)
        interior_starts = [
            window
            for window in result.visibility
            if timestamps.index(window.start_time) > 0
        ]
        assert interior_starts, "expected a window opening mid-span"
        for window in interior_starts:
            assert window.start_cause, "coordinated sweep must attribute the opening"
            for tag in window.start_cause:
                assert tag in result.cause_value_keys
