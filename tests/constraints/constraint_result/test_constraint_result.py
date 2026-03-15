from datetime import datetime, timezone
from typing import Any

import pytest

from rust_ephem.constraints import ConstraintResult


class TestConstraintResult:
    def test_timestamps(
        self,
        constraint_result_with_rust_ref: tuple[ConstraintResult, Any],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.timestamps == rust_result.timestamp

    def test_constraint_array(
        self,
        constraint_result_with_rust_ref: tuple[ConstraintResult, Any],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.constraint_array == rust_result.constraint_array

    def test_visibility(
        self,
        constraint_result_with_rust_ref: tuple[ConstraintResult, Any],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.visibility == rust_result.visibility

    def test_in_constraint(
        self,
        constraint_result_with_rust_ref: tuple[ConstraintResult, Any],
    ) -> None:
        result, _ = constraint_result_with_rust_ref
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
