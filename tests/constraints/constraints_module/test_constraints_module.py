from datetime import datetime, timezone
from typing import Any, Tuple, Type

import numpy as np
import pytest

from rust_ephem.constraints import (
    AirmassConstraint,
    ConstraintResult,
    MoonPhaseConstraint,
    MovingVisibilityResult,
    SunConstraint,
)

from .conftest import (
    DummyConstraintBackend,
    DummyRustResult,
)


class TestEvaluateMovingBody:
    def test_timestamps(
        self,
        constraint_result_with_rust_ref: Tuple[ConstraintResult, DummyRustResult],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.timestamps == rust_result.timestamp

    def test_constraint_array(
        self,
        constraint_result_with_rust_ref: Tuple[ConstraintResult, DummyRustResult],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.constraint_array == rust_result.constraint_array

    def test_visibility(
        self,
        constraint_result_with_rust_ref: Tuple[ConstraintResult, DummyRustResult],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.visibility == rust_result.visibility  # type: ignore[comparison-overlap]

    def test_in_constraint(
        self,
        constraint_result_with_rust_ref: Tuple[ConstraintResult, DummyRustResult],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.in_constraint(result.timestamps[0]) is True

    def test_total_violation_duration(
        self, constraint_result_without_rust_ref: ConstraintResult
    ) -> None:
        result = constraint_result_without_rust_ref
        assert result.total_violation_duration() == 10.0

    def test_repr(self, constraint_result_without_rust_ref: ConstraintResult) -> None:
        result = constraint_result_without_rust_ref
        assert "ConstraintResult" in repr(result)

    def test_without_rust_ref_timestamps(self) -> None:
        result: ConstraintResult = ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.timestamps == []

    def test_without_rust_ref_constraint_array(self) -> None:
        result: ConstraintResult = ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.constraint_array == []

    def test_without_rust_ref_visibility(self) -> None:
        result: ConstraintResult = ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.visibility == []

    def test_without_rust_ref_in_constraint_raises(self) -> None:
        result: ConstraintResult = ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        with pytest.raises(ValueError):
            result.in_constraint(datetime.now(timezone.utc))


class TestRustConstraintMixin:
    def test_evaluate_creates_backend_once_created_count(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _: ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=1.0, target_dec=2.0
        )
        _ = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)
        assert patched_constraint.created == 1

    def test_evaluate_creates_backend_once_first_result_type(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        first: ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=1.0, target_dec=2.0
        )
        assert isinstance(first, ConstraintResult)

    def test_evaluate_creates_backend_once_second_result_type(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        second: ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=3.0, target_dec=4.0
        )
        assert isinstance(second, ConstraintResult)

    def test_evaluate_creates_backend_once_evaluate_calls_length(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=1.0, target_dec=2.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)
        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_calls) == 2

    def test_batch_and_single_created_count(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
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
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        batch: np.ndarray = constraint.in_constraint_batch(
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
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
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
        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert backend.batch_calls

    def test_batch_and_single_single_calls_exist(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
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
        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert backend.single_calls

    def test_batch_and_single_single_result(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
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


class TestValidators:
    def test_airmass_raises_on_invalid(self) -> None:
        with pytest.raises(ValueError):
            AirmassConstraint(min_airmass=2.0, max_airmass=1.5)

    def test_airmass_max_airmass(self) -> None:
        valid_airmass: AirmassConstraint = AirmassConstraint(
            min_airmass=1.0, max_airmass=2.0
        )
        assert valid_airmass.max_airmass == 2.0

    def test_moon_phase_raises_on_invalid_illumination(self) -> None:
        with pytest.raises(ValueError):
            MoonPhaseConstraint(
                min_illumination=0.8, max_illumination=0.5, max_distance=1.0
            )

    def test_moon_phase_raises_on_invalid_distance(self) -> None:
        with pytest.raises(ValueError):
            MoonPhaseConstraint(
                min_distance=5.0, max_distance=4.0, max_illumination=0.9
            )

    def test_moon_phase_max_distance(self) -> None:
        valid_phase: MoonPhaseConstraint = MoonPhaseConstraint(
            max_illumination=0.9, min_distance=1.0, max_distance=2.0
        )
        assert valid_phase.max_distance == 2.0


class TestEvaluateMovingBodyMethod:
    def test_with_body_calls_backend_once(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body calls backend once"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_body_passes_body_argument(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body passes body argument correctly"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[4] == "499"  # body

    def test_with_body_returns_correct_type(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body returns MovingVisibilityResult"""
        constraint = SunConstraint(min_angle=10.0)
        constraint._rust_constraint = patched_constraint()  # type: ignore[attr-defined]
        result = constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        assert isinstance(result, MovingVisibilityResult)

    def test_with_explicit_coords_calls_backend_once(
        self,
        patched_constraint: Type[DummyConstraintBackend],
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

        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_explicit_coords_passes_ras(
        self,
        patched_constraint: Type[DummyConstraintBackend],
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

        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[1] == ras  # target_ras

    def test_with_explicit_coords_passes_decs(
        self,
        patched_constraint: Type[DummyConstraintBackend],
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

        backend: DummyConstraintBackend = constraint._rust_constraint  # type: ignore[attr-defined]
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[2] == decs  # target_decs

    def test_with_explicit_coords_returns_correct_type(
        self,
        patched_constraint: Type[DummyConstraintBackend],
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
