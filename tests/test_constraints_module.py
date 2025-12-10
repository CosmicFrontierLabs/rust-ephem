from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import pytest

import rust_ephem
import rust_ephem.constraints as constraints
from rust_ephem.constraints import (
    AirmassConstraint,
    MoonPhaseConstraint,
    SunConstraint,
)


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

    def __init__(self, payload: str) -> None:
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


@pytest.fixture
def dummy_rust_result() -> DummyRustResult:
    return DummyRustResult()


@pytest.fixture
def constraint_result_with_rust_ref(
    dummy_rust_result: DummyRustResult,
) -> Tuple[constraints.ConstraintResult, DummyRustResult]:
    result = constraints.ConstraintResult(
        violations=[
            constraints.ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 0),
                end_time=datetime(2024, 1, 1, 0, 0, 5),
                max_severity=1.0,
                description="v1",
            ),
            constraints.ConstraintViolation(
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
def constraint_result_without_rust_ref() -> constraints.ConstraintResult:
    result = constraints.ConstraintResult(
        violations=[
            constraints.ConstraintViolation(
                start_time=datetime(2024, 1, 1, 0, 0, 0),
                end_time=datetime(2024, 1, 1, 0, 0, 5),
                max_severity=1.0,
                description="v1",
            ),
            constraints.ConstraintViolation(
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


@pytest.fixture
def patched_constraint(monkeypatch: pytest.MonkeyPatch) -> Type[DummyConstraintBackend]:
    DummyConstraintBackend.created = 0
    monkeypatch.setattr(rust_ephem, "Constraint", DummyConstraintBackend)
    return DummyConstraintBackend


@pytest.fixture
def dummy_ephemeris() -> Any:
    return object()


@pytest.fixture
def mock_ephemeris_with_body() -> Any:
    """Mock ephemeris that supports body lookup"""

    class MockEphemeris:
        def get_body(
            self,
            body: Any,
            spice_kernel: Optional[Any] = None,
            use_horizons: bool = False,
        ) -> Any:
            return type(
                "SkyCoord",
                (),
                {
                    "ra": type("Angle", (), {"deg": [1.0, 2.0]})(),
                    "dec": type("Angle", (), {"deg": [3.0, 4.0]})(),
                },
            )()

        @property
        def timestamp(self) -> List[datetime]:
            return [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 1)]

    return MockEphemeris()


@pytest.fixture
def mock_ephemeris_simple() -> Any:
    """Simple mock ephemeris for coordinate tests"""

    class MockEphemeris:
        @property
        def timestamp(self) -> List[datetime]:
            return [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 1)]

    return MockEphemeris()


@pytest.fixture
def mock_ephemeris_single() -> Any:
    """Mock ephemeris with single timestamp"""

    class MockEphemeris:
        @property
        def timestamp(self) -> List[datetime]:
            return [datetime(2024, 1, 1, 0, 0, 0)]

    return MockEphemeris()


class TestEvaluateMovingBody:
    def test_timestamps(
        self,
        constraint_result_with_rust_ref: Tuple[
            constraints.ConstraintResult, DummyRustResult
        ],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.timestamps == rust_result.timestamp

    def test_constraint_array(
        self,
        constraint_result_with_rust_ref: Tuple[
            constraints.ConstraintResult, DummyRustResult
        ],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.constraint_array == rust_result.constraint_array

    def test_visibility(
        self,
        constraint_result_with_rust_ref: Tuple[
            constraints.ConstraintResult, DummyRustResult
        ],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.visibility == rust_result.visibility

    def test_in_constraint(
        self,
        constraint_result_with_rust_ref: Tuple[
            constraints.ConstraintResult, DummyRustResult
        ],
    ) -> None:
        result, rust_result = constraint_result_with_rust_ref
        assert result.in_constraint(result.timestamps[0]) is True

    def test_total_violation_duration(
        self, constraint_result_without_rust_ref: constraints.ConstraintResult
    ) -> None:
        result = constraint_result_without_rust_ref
        assert result.total_violation_duration() == 10.0

    def test_repr(
        self, constraint_result_without_rust_ref: constraints.ConstraintResult
    ) -> None:
        result = constraint_result_without_rust_ref
        assert "ConstraintResult" in repr(result)

    def test_without_rust_ref_timestamps(self) -> None:
        result: constraints.ConstraintResult = constraints.ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.timestamps == []

    def test_without_rust_ref_constraint_array(self) -> None:
        result: constraints.ConstraintResult = constraints.ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.constraint_array == []

    def test_without_rust_ref_visibility(self) -> None:
        result: constraints.ConstraintResult = constraints.ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        assert result.visibility == []

    def test_without_rust_ref_in_constraint_raises(self) -> None:
        result: constraints.ConstraintResult = constraints.ConstraintResult(
            violations=[], all_satisfied=True, constraint_name="empty"
        )
        with pytest.raises(ValueError):
            result.in_constraint(datetime.now(timezone.utc))


class TestRustConstraintMixin:
    def test_evaluate_creates_backend_once_created_count(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _: constraints.ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=1.0, target_dec=2.0
        )
        _: constraints.ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=3.0, target_dec=4.0
        )
        assert DummyConstraintBackend.created == 1

    def test_evaluate_creates_backend_once_first_result_type(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        first: constraints.ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=1.0, target_dec=2.0
        )
        assert isinstance(first, constraints.ConstraintResult)

    def test_evaluate_creates_backend_once_second_result_type(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        second: constraints.ConstraintResult = constraint.evaluate(
            dummy_ephemeris, target_ra=3.0, target_dec=4.0
        )
        assert isinstance(second, constraints.ConstraintResult)

    def test_evaluate_creates_backend_once_evaluate_calls_length(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=10.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=1.0, target_dec=2.0)
        _ = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)
        backend: DummyConstraintBackend = constraint._rust_constraint
        assert len(backend.evaluate_calls) == 2

    def test_batch_and_single_created_count(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
        _: np.ndarray = constraint.in_constraint_batch(
            dummy_ephemeris,
            target_ras=[1.0, 2.0],
            target_decs=[3.0, 4.0],
            times=[datetime(2024, 1, 1)],
            indices=None,
        )
        _dog = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        assert DummyConstraintBackend.created == 1

    def test_batch_and_single_batch_shape(
        self, patched_constraint: Type[DummyConstraintBackend], dummy_ephemeris: Any
    ) -> None:
        constraint: SunConstraint = SunConstraint(min_angle=5.0)
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
        assert batch.shape == (2, 2)

    def test_batch_and_single_batch_calls_exist(
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
        backend: DummyConstraintBackend = constraint._rust_constraint
        assert backend.batch_calls

    def test_batch_and_single_single_calls_exist(
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
        backend: DummyConstraintBackend = constraint._rust_constraint
        assert backend.single_calls

    def test_batch_and_single_single_result(
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
        single: str = constraint.in_constraint(
            time=datetime(2024, 1, 1),
            ephemeris=dummy_ephemeris,
            target_ra=1.0,
            target_dec=2.0,
        )
        assert single == "single-result"


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


class DummyAngle:
    def __init__(self, deg: float) -> None:
        self.deg: float = deg


class DummySkyCoord:
    def __init__(self, ra: float, dec: float) -> None:
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


class TestEvaluateMovingBodyMethod:
    def test_with_body_calls_backend_once(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body calls backend once"""
        constraint = SunConstraint(min_angle=10.0)
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend: DummyConstraintBackend = constraint._rust_constraint
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_body_passes_body_argument(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body passes body argument correctly"""
        constraint = SunConstraint(min_angle=10.0)
        constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        backend: DummyConstraintBackend = constraint._rust_constraint
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[4] == "499"  # body

    def test_with_body_returns_correct_type(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_with_body: Any,
    ) -> None:
        """Test that evaluate_moving_body with body returns MovingVisibilityResult"""
        constraint = SunConstraint(min_angle=10.0)
        result = constraint.evaluate_moving_body(mock_ephemeris_with_body, body=499)

        assert isinstance(result, constraints.MovingVisibilityResult)

    def test_with_explicit_coords_calls_backend_once(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body with coords calls backend once"""
        constraint = SunConstraint(min_angle=10.0)
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend: DummyConstraintBackend = constraint._rust_constraint
        assert len(backend.evaluate_moving_body_calls) == 1

    def test_with_explicit_coords_passes_ras(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body passes target_ras correctly"""
        constraint = SunConstraint(min_angle=10.0)
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend: DummyConstraintBackend = constraint._rust_constraint
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[1] == ras  # target_ras

    def test_with_explicit_coords_passes_decs(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body passes target_decs correctly"""
        constraint = SunConstraint(min_angle=10.0)
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        backend: DummyConstraintBackend = constraint._rust_constraint
        call_args = backend.evaluate_moving_body_calls[0]
        assert call_args[2] == decs  # target_decs

    def test_with_explicit_coords_returns_correct_type(
        self,
        patched_constraint: Type[DummyConstraintBackend],
        mock_ephemeris_simple: Any,
    ) -> None:
        """Test that evaluate_moving_body with coords returns MovingVisibilityResult"""
        constraint = SunConstraint(min_angle=10.0)
        ras = [1.0, 2.0]
        decs = [3.0, 4.0]

        result = constraint.evaluate_moving_body(
            mock_ephemeris_simple, target_ras=ras, target_decs=decs
        )

        assert isinstance(result, constraints.MovingVisibilityResult)
