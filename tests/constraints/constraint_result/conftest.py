from datetime import datetime, timedelta

import pytest

from rust_ephem.constraints import ConstraintResult, ConstraintViolation


class DummyRustResult:
    def __init__(self) -> None:
        base: datetime = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamp: list[datetime] = [base, base + timedelta(seconds=1)]
        self.constraint_array: list[bool] = [True, False]
        self.visibility: list[str] = ["window"]
        self.all_satisfied: bool = False
        self.constraint_name: str = "DummyConstraint"
        self._in_constraint_calls: list[datetime] = []
        self._in_constraint_return: bool = True

    def in_constraint(self, time: datetime) -> bool:
        self._in_constraint_calls.append(time)
        return self._in_constraint_return


@pytest.fixture
def dummy_rust_result() -> DummyRustResult:
    return DummyRustResult()


@pytest.fixture
def constraint_result_with_rust_ref(
    dummy_rust_result: DummyRustResult,
) -> tuple[ConstraintResult, DummyRustResult]:
    result = ConstraintResult(
        violations=[
            ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 0),
                end_time=datetime(2024, 1, 1, 0, 0, 5),
                max_severity=1.0,
                description="v1",
            ),
            ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 10),
                end_time=datetime(2024, 1, 1, 0, 0, 15),
                max_severity=1.0,
                description="v2",
            ),
        ],
        all_satisfied=False,
        constraint_name="test",
        _rust_result_ref=dummy_rust_result,  # type: ignore
    )
    return result, dummy_rust_result


@pytest.fixture
def constraint_result_without_rust_ref() -> ConstraintResult:
    result = ConstraintResult(
        violations=[
            ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 0),
                end_time=datetime(2024, 1, 1, 0, 0, 5),
                max_severity=1.0,
                description="v1",
            ),
            ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 10),
                end_time=datetime(2024, 1, 1, 0, 0, 15),
                max_severity=1.0,
                description="v2",
            ),
        ],
        all_satisfied=False,
        constraint_name="test",
    )
    return result
