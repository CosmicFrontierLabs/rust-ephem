from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from rust_ephem.constraints import (
    ConstraintResult,
    MovingVisibilityResult,
    SunConstraint,
)


class TestRustConstraintMixin:
    def test_evaluate_batch_returns_constraint_results(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: Any = SunConstraint(min_angle=10.0)
        results = constraint.evaluate_batch(
            dummy_ephemeris,
            target_ras=[1.0, 3.0],
            target_decs=[2.0, 4.0],
        )

        assert len(results) == 2
        assert all(isinstance(result, ConstraintResult) for result in results)

    def test_evaluate_batch_reuses_cached_backend(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: Any = SunConstraint(min_angle=10.0)
        _ = constraint.evaluate_batch(
            dummy_ephemeris,
            target_ras=[1.0, 3.0],
            target_decs=[2.0, 4.0],
        )

        assert patched_constraint.created == 1
        backend = constraint._rust_constraint
        assert len(backend.evaluate_batch_calls) == 1

    def test_evaluate_creates_backend_once_created_count(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=1.0, target_dec=2.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)
        assert patched_constraint.created == 1

    def test_evaluate_creates_backend_once_first_result_type(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        first: ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=1.0, target_dec=2.0
        )
        assert isinstance(first, ConstraintResult)

    def test_evaluate_creates_backend_once_second_result_type(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        second: ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=3.0, target_dec=4.0
        )
        assert isinstance(second, ConstraintResult)

    def test_evaluate_creates_backend_once_evaluate_calls_length(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=1.0, target_dec=2.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)
        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_calls) == 2

    def test_batch_and_single_created_count(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        _ = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        _ = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        assert patched_constraint.created == 1

    def test_batch_and_single_batch_shape(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        batch: npt.NDArray[np.bool_] = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        _ = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        assert batch.shape == (2, 1)

    def test_batch_and_single_batch_calls_exist(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        _ = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        _ = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert backend.batch_calls

    def test_batch_and_single_single_calls_exist(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        _ = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        _ = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert backend.single_calls

    def test_batch_and_single_single_result(
        self, patched_constraint: Any, dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        _ = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        single = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        assert single == "single-result"  # type: ignore[comparison-overlap]


class TestEvaluateMovingBodyMethod:
    def test_with_body_calls_backend_once(
        self,
        patched_constraint: Any,
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body calls backend once"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_body_passes_body_argument(
        self,
        patched_constraint: Any,
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body passes body argument correctly"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[4] == "499"  # body

    def test_with_body_returns_correct_type(
        self,
        patched_constraint: Any,
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body returns MovingVisibilityResult"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        result = constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        assert isinstance(result, MovingVisibilityResult)

    def test_with_explicit_coords_calls_backend_once(
        self,
        patched_constraint: Any,
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body with coords calls backend once"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_explicit_coords_passes_ras(
        self,
        patched_constraint: Any,
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body passes target_ras correctly"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[1] == ras  # target_ras

    def test_with_explicit_coords_passes_decs(
        self,
        patched_constraint: Any,
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body passes target_decs correctly"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[2] == decs  # target_decs

    def test_with_explicit_coords_returns_correct_type(
        self,
        patched_constraint: Any,
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body with coords returns MovingVisibilityResult"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        result = constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        assert isinstance(result, MovingVisibilityResult)
