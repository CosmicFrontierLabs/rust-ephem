from datetime import datetime, timedelta

import numpy as np
import pytest

import rust_ephem.constraints as constraints
from rust_ephem.constraints import (
    AirmassConstraint,
    MoonPhaseConstraint,
    SunConstraint,
    _build_visibility_windows,
    _convert_single_datetime,
    _to_datetime_list,
    moving_body_visibility,
)


class DummyRustResult:
    def __init__(self) -> None:
        base = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamp = [base, base + timedelta(seconds=1)]
        self.constraint_array = [True, False]
        self.visibility = ["window"]
        self.all_satisfied = False
        self.constraint_name = "DummyConstraint"
        self._in_constraint_calls: list[datetime] = []
        self._in_constraint_return = True
        self.violations = [
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


class DummyConstraintBackend:
    created = 0

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.evaluate_calls: list[tuple] = []
        self.batch_calls: list[tuple] = []
        self.single_calls: list[tuple] = []

    @classmethod
    def from_json(cls, payload: str) -> "DummyConstraintBackend":
        cls.created += 1
        return cls(payload)

    def evaluate(self, ephemeris, target_ra, target_dec, times, indices):
        self.evaluate_calls.append((ephemeris, target_ra, target_dec, times, indices))
        return DummyRustResult()

    def in_constraint_batch(self, ephemeris, target_ras, target_decs, times, indices):
        self.batch_calls.append(
            (ephemeris, tuple(target_ras), tuple(target_decs), times, indices)
        )
        return np.array([[True, False], [False, True]])

    def in_constraint(self, time, ephemeris, target_ra, target_dec):
        self.single_calls.append((time, ephemeris, target_ra, target_dec))
        return "single-result"


def test_constraint_result_lazy_accessors_and_total_duration():
    rust_result = DummyRustResult()
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
        _rust_result_ref=rust_result,
    )

    assert result.timestamps == rust_result.timestamp
    assert result.constraint_array == rust_result.constraint_array
    assert result.visibility == rust_result.visibility
    assert result.in_constraint(result.timestamps[0]) is True
    assert result.total_violation_duration() == 10.0
    assert "ConstraintResult" in repr(result)


def test_constraint_result_without_rust_ref():
    result = constraints.ConstraintResult(
        violations=[], all_satisfied=True, constraint_name="empty"
    )

    assert result.timestamps == []
    assert result.constraint_array == []
    assert result.visibility == []
    with pytest.raises(ValueError):
        result.in_constraint(datetime.utcnow())


def test_rust_constraint_mixin_evaluate_creates_backend_once(monkeypatch):
    DummyConstraintBackend.created = 0
    monkeypatch.setattr(constraints.rust_ephem, "Constraint", DummyConstraintBackend)
    constraint = SunConstraint(min_angle=10.0)
    dummy_ephemeris = object()

    first = constraint.evaluate(dummy_ephemeris, target_ra=1.0, target_dec=2.0)
    second = constraint.evaluate(dummy_ephemeris, target_ra=3.0, target_dec=4.0)

    assert DummyConstraintBackend.created == 1
    assert isinstance(first, constraints.ConstraintResult)
    assert isinstance(second, constraints.ConstraintResult)
    backend: DummyConstraintBackend = constraint._rust_constraint
    assert len(backend.evaluate_calls) == 2


def test_rust_constraint_mixin_batch_and_single(monkeypatch):
    DummyConstraintBackend.created = 0
    monkeypatch.setattr(constraints.rust_ephem, "Constraint", DummyConstraintBackend)
    constraint = SunConstraint(min_angle=5.0)
    dummy_ephemeris = object()

    batch = constraint.in_constraint_batch(
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

    backend: DummyConstraintBackend = constraint._rust_constraint
    assert DummyConstraintBackend.created == 1
    assert batch.shape == (2, 2)
    assert backend.batch_calls
    assert backend.single_calls
    assert single == "single-result"


def test_operator_combinators():
    sun = SunConstraint(min_angle=10.0)
    moon = SunConstraint(min_angle=20.0)

    combined_and = sun & moon
    combined_or = sun | moon
    combined_xor = sun ^ moon
    inverted = ~sun

    assert combined_and.type == "and"
    assert combined_or.type == "or"
    assert combined_xor.type == "xor"
    assert inverted.type == "not"
    assert combined_and.constraints[0] is sun
    assert combined_or.constraints[1] is moon
    assert combined_xor.constraints[0] is sun
    assert inverted.constraint is sun


def test_validators_airmass_and_moon_phase():
    with pytest.raises(ValueError):
        AirmassConstraint(min_airmass=2.0, max_airmass=1.5)
    valid_airmass = AirmassConstraint(min_airmass=1.0, max_airmass=2.0)
    assert valid_airmass.max_airmass == 2.0

    with pytest.raises(ValueError):
        MoonPhaseConstraint(
            min_illumination=0.8, max_illumination=0.5, max_distance=1.0
        )
    with pytest.raises(ValueError):
        MoonPhaseConstraint(min_distance=5.0, max_distance=4.0, max_illumination=0.9)
    valid_phase = MoonPhaseConstraint(
        max_illumination=0.9, min_distance=1.0, max_distance=2.0
    )
    assert valid_phase.max_distance == 2.0


def test_to_datetime_list_and_convert_single_datetime():
    dt = datetime(2024, 1, 1)
    np_dt = np.datetime64("2024-01-02T00:00:00")
    array = np.array([np_dt])

    assert _to_datetime_list(dt) == [dt]
    assert _to_datetime_list(array)[0] == datetime.fromisoformat("2024-01-02T00:00:00")
    assert _to_datetime_list([dt, np_dt])[1] == datetime.fromisoformat(
        "2024-01-02T00:00:00"
    )
    with pytest.raises(TypeError):
        _convert_single_datetime(123)


class DummyAngle:
    def __init__(self, deg):
        self.deg = deg


class DummySkyCoord:
    def __init__(self, ra, dec):
        self.ra = DummyAngle(ra)
        self.dec = DummyAngle(dec)


class DummyEphemeris:
    def __init__(self, timestamps):
        self.timestamp = timestamps
        self.body_requests: list[str] = []

    def get_body(self, body_id, kernel_spec=None, use_horizons=False):
        self.body_requests.append(str(body_id))
        return DummySkyCoord(ra=[1.0, 2.0], dec=[3.0, 4.0])


class SequenceConstraint:
    def __init__(self, results):
        self.results = results
        self.calls: list[tuple[datetime, float, float]] = []

    def in_constraint(self, time, ephemeris, target_ra, target_dec):
        idx = len(self.calls)
        self.calls.append((time, target_ra, target_dec))
        return self.results[idx]


def test_moving_body_visibility_with_body():
    timestamps = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 1)]
    ephem = DummyEphemeris(timestamps)
    constraint = SequenceConstraint([True, False])

    result = moving_body_visibility(constraint, ephem, body=499)

    assert ephem.body_requests == ["499"]
    assert result.constraint_array == [True, False]
    assert result.visibility_flags == [False, True]
    assert result.all_satisfied is False
    assert len(result.visibility) == 1
    assert result.constraint_name == "SequenceConstraint"


def test_moving_body_visibility_requires_arrays_when_no_body():
    constraint = SequenceConstraint([False])
    with pytest.raises(ValueError):
        moving_body_visibility(constraint, DummyEphemeris([]))


def test_moving_body_visibility_shape_and_length_validation():
    constraint = SequenceConstraint([False, False])
    timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 1)]

    with pytest.raises(ValueError):
        moving_body_visibility(
            constraint,
            DummyEphemeris(timestamps),
            ras=[1.0],
            decs=[1.0],
            timestamps=timestamps,
        )
    with pytest.raises(ValueError):
        moving_body_visibility(
            constraint,
            DummyEphemeris(timestamps),
            ras=[1.0, 2.0],
            decs=[1.0, 2.0, 3.0],
            timestamps=timestamps,
        )
    with pytest.raises(ValueError):
        moving_body_visibility(
            constraint,
            DummyEphemeris(timestamps),
            ras=[1.0, 2.0],
            decs=[1.0, 2.0],
            timestamps=[datetime(2024, 1, 1)],
        )


def test_build_visibility_windows_merges_ranges():
    timestamps = [
        datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=i) for i in range(5)
    ]
    visibility = [False, True, True, False, True]

    windows = _build_visibility_windows(timestamps, visibility)
    # Window 1: indices 1,2 are True. End time is last visible (idx 2), not first False
    # Window 2: index 4 is True until end
    assert len(windows) == 2
    assert windows[0].start_time == timestamps[1]
    assert (
        windows[0].end_time == timestamps[2]
    )  # Last visible timestamp, not first False
    assert windows[1].start_time == timestamps[4]

    assert _build_visibility_windows([], []) == []
