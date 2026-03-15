"""Fixtures for constraints_module tests."""

from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from rust_ephem.constraints import ConstraintResult, ConstraintViolation


class DummyRustResult:
    def __init__(self) -> None:
        base: datetime = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamp: List[datetime] = [base, base + timedelta(seconds=1)]
        self.constraint_array: List[bool] = [True, False]
        self.visibility: List[str] = ["window"]
        self.all_satisfied: bool = False
        self.constraint_name: str = "DummyConstraint"
        self._in_constraint_calls: List[datetime] = []
        self._in_constraint_return: bool = True
        self.violations: List[Any] = [
            type(
                "Violation",
                (),
                {
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-01-01T00:00:10Z",
                    "max_severity": 1.0,
                    "description": "test",
                },
            )(),
            type(
                "Violation",
                (),
                {
                    "start_time": "2024-01-01T00:00:20Z",
                    "end_time": "2024-01-01T00:00:25Z",
                    "max_severity": 0.5,
                    "description": "test2",
                },
            )(),
        ]

    def in_constraint(self, time: datetime) -> bool:
        self._in_constraint_calls.append(time)
        return self._in_constraint_return


class DummyMovingBodyResult:
    def __init__(self) -> None:
        base = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamp: List[datetime] = [base, base + timedelta(seconds=1)]
        self.ras: List[float] = [1.0, 2.0]
        self.decs: List[float] = [3.0, 4.0]
        self.constraint_array: List[bool] = [True, False]
        self.visibility: List[str] = []
        self.all_satisfied: bool = False
        self.constraint_name: str = "DummyMovingBody"


class DummyConstraintBackend:
    created: int = 0

    def __init__(self, payload: str = "") -> None:
        self.payload: str = payload
        self.evaluate_calls: List[
            Tuple[Any, float, float, List[datetime], Optional[List[int]]]
        ] = []
        self.batch_calls: List[
            Tuple[
                Any,
                Tuple[float, ...],
                Tuple[float, ...],
                List[datetime],
                Optional[List[int]],
            ]
        ] = []
        self.single_calls: List[Tuple[datetime, Any, float, float]] = []
        self.evaluate_moving_body_calls: List[
            Tuple[
                Any,
                Optional[List[float]],
                Optional[List[float]],
                Optional[List[datetime]],
                Optional[str],
                bool,
                Optional[str],
            ]
        ] = []

    @classmethod
    def from_json(cls, payload: str) -> "DummyConstraintBackend":
        cls.created += 1
        return cls(payload)

    def evaluate(
        self,
        ephemeris: Any,
        target_ra: float,
        target_dec: float,
        times: List[datetime],
        indices: Optional[List[int]],
    ) -> DummyRustResult:
        self.evaluate_calls.append((ephemeris, target_ra, target_dec, times, indices))
        return DummyRustResult()

    def in_constraint_batch(
        self,
        ephemeris: Any,
        target_ras: List[float],
        target_decs: List[float],
        times: List[datetime],
        indices: Optional[List[int]],
    ) -> np.ndarray:
        self.batch_calls.append(
            (ephemeris, tuple(target_ras), tuple(target_decs), times, indices)
        )
        return np.array([[True, False], [False, True]])

    def in_constraint(
        self, time: datetime, ephemeris: Any, target_ra: float, target_dec: float
    ) -> str:
        self.single_calls.append((time, ephemeris, target_ra, target_dec))
        return "single-result"

    def evaluate_moving_body(
        self,
        ephemeris: Any,
        ras: Optional[List[float]],
        decs: Optional[List[float]],
        times: Optional[List[datetime]],
        body: Optional[str],
        use_horizons: bool,
        spice_kernel: Optional[str],
    ) -> DummyMovingBodyResult:
        self.evaluate_moving_body_calls.append(
            (ephemeris, ras, decs, times, body, use_horizons, spice_kernel)
        )
        return DummyMovingBodyResult()


class DummyAngle:
    def __init__(self, deg: float | list[float]) -> None:
        self.deg: float | list[float] = deg


class DummySkyCoord:
    def __init__(self, ra: float | list[float], dec: float | list[float]) -> None:
        self.ra: DummyAngle = DummyAngle(ra)
        self.dec: DummyAngle = DummyAngle(dec)


class DummyEphemeris:
    def __init__(self, timestamps: List[datetime]) -> None:
        self.timestamp: List[datetime] = timestamps
        self.body_requests: List[str] = []

    def get_body(
        self,
        body_id: str,
        spice_kernel: Optional[str] = None,
        use_horizons: bool = False,
    ) -> DummySkyCoord:
        self.body_requests.append(str(body_id))
        return DummySkyCoord(ra=[1.0, 2.0], dec=[3.0, 4.0])


class SequenceConstraint:
    def __init__(self, results: List[bool]) -> None:
        self.results: List[bool] = results
        self.calls: List[Tuple[datetime, float, float]] = []

    def in_constraint(
        self, time: datetime, ephemeris: Any, target_ra: float, target_dec: float
    ) -> bool:
        idx: int = len(self.calls)
        self.calls.append((time, target_ra, target_dec))
        return self.results[idx]


@pytest.fixture
def dummy_rust_result() -> DummyRustResult:
    return DummyRustResult()


@pytest.fixture
def constraint_result_with_rust_ref(
    dummy_rust_result: DummyRustResult,
) -> Tuple[ConstraintResult, DummyRustResult]:
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
