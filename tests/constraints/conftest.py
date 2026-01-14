from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pytest

from rust_ephem import (
    BodyConstraint,
    ConstraintResult,
    ConstraintViolation,
    EarthLimbConstraint,
    EclipseConstraint,
    GroundEphemeris,
    MoonConstraint,
    SunConstraint,
    TLEEphemeris,
)


class DummyRustResult:
    def __init__(self) -> None:
        from datetime import datetime, timedelta

        base: datetime = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamp: list[datetime] = [base, base + timedelta(seconds=1)]
        self.constraint_array: list[bool] = [True, False]
        self.visibility: list[object] = [
            type(
                "VisibilityWindow",
                (),
                {
                    "start_time": base,
                    "end_time": base + timedelta(seconds=1),
                    "duration_seconds": 1.0,
                },
            )()
        ]
        self.all_satisfied: bool = False
        self.constraint_name: str = "DummyConstraint"
        self._in_constraint_calls: list[datetime] = []
        self._in_constraint_return: bool = True
        self.ras: list[float] = [1.0, 2.0]
        self.decs: list[float] = [3.0, 4.0]
        self.violations: list = [
            type(
                "Violation",
                (),
                {
                    "start_time": base,
                    "end_time": base + timedelta(seconds=1),
                    "max_severity": 1.0,
                    "description": "test violation",
                },
            )()
        ]
        self.ras: list[float] = [1.0, 2.0]
        self.decs: list[float] = [3.0, 4.0]

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


@pytest.fixture
def patched_constraint(monkeypatch: pytest.MonkeyPatch):
    import rust_ephem

    DummyConstraintBackend.created = 0
    monkeypatch.setattr(
        rust_ephem.Constraint, "from_json", lambda json_str: DummyConstraintBackend()
    )
    return DummyConstraintBackend


@pytest.fixture
def dummy_ephemeris():
    return object()


@pytest.fixture
def sun_constraint():
    """Fixture for a SunConstraint instance."""
    return SunConstraint(min_angle=45.0)


@pytest.fixture
def moon_constraint():
    """Fixture for a MoonConstraint instance."""
    return MoonConstraint(min_angle=30.0)


@pytest.fixture
def eclipse_constraint():
    """Fixture for an EclipseConstraint instance."""
    return EclipseConstraint(umbra_only=True)


@pytest.fixture
def earth_limb_constraint():
    """Fixture for an EarthLimbConstraint instance."""
    return EarthLimbConstraint(min_angle=10.0)


@pytest.fixture
def body_constraint():
    """Fixture for a BodyConstraint instance."""
    return BodyConstraint(body="Mars", min_angle=15.0)


@pytest.fixture
def ground_ephemeris():
    return GroundEphemeris(
        latitude=34.0,
        longitude=-118.0,
        height=100.0,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


@pytest.fixture
def tle_ephemeris():
    return TLEEphemeris(
        tle1=VALID_TLE1,
        tle2=VALID_TLE2,
        begin=BEGIN_TIME,
        end=END_TIME,
        step_size=STEP_SIZE,
    )


class DummyConstraintBackend:
    created = 0

    def __init__(self) -> None:
        DummyConstraintBackend.created += 1
        self.evaluate_calls = []
        self.batch_calls = []
        self.single_calls = []
        self.evaluate_moving_body_calls = []

    @classmethod
    def from_json(cls, json_str):
        return cls()

    def evaluate(
        self,
        ephemeris,
        target_ra,
        target_dec,
        times,
        indices,
    ):
        self.evaluate_calls.append((ephemeris, target_ra, target_dec, times, indices))
        return DummyRustResult()

    def in_constraint_batch(
        self,
        ephemeris,
        target_ras,
        target_decs,
        times,
        indices,
    ):
        self.batch_calls.append((ephemeris, target_ras, target_decs, times, indices))
        return np.array([[True], [False]])

    def in_constraint(self, time, ephemeris, target_ra, target_dec):
        self.single_calls.append((time, ephemeris, target_ra, target_dec))
        return "single-result"

    def evaluate_moving_body(
        self,
        ephemeris,
        ras,
        decs,
        times,
        body,
        use_horizons: bool,
        spice_kernel: Optional[str],
    ):
        self.evaluate_moving_body_calls.append(
            (ephemeris, ras, decs, times, body, use_horizons, spice_kernel)
        )
        return DummyRustResult()


@pytest.fixture
def mock_ephem():
    """Fixture for a mock ephemeris object."""

    class MockEphemeris:
        pass

    return MockEphemeris()


@pytest.fixture
def mock_ephemeris_with_body():
    """Mock ephemeris that supports body lookup"""

    class MockEphemeris:
        def get_body(
            self,
            body,
            spice_kernel=None,
            use_horizons: bool = False,
        ):
            return type(
                "SkyCoord",
                (),
                {
                    "ra": type("Angle", (), {"deg": [1.0, 2.0]})(),
                    "dec": type("Angle", (), {"deg": [3.0, 4.0]})(),
                },
            )()

        @property
        def timestamp(self):
            from datetime import datetime

            return [datetime(2024, 1, 1, 0, 0, 0)]

    return MockEphemeris()


# Test data
VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

BEGIN_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
STEP_SIZE = 120  # 2 minutes


@pytest.fixture
def tle() -> tuple:
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    return tle1, tle2


@pytest.fixture
def begin_end_step_size() -> tuple[datetime, datetime, int]:
    # Time span and step
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
    step_s = 60  # 1 minute
    return begin, end, step_s


@pytest.fixture
def tle_ephem(tle, begin_end_step_size):
    import rust_ephem

    yield rust_ephem.TLEEphemeris(
        tle[0],
        tle[1],
        begin_end_step_size[0],
        begin_end_step_size[1],
        begin_end_step_size[2],
    )


@pytest.fixture
def sun(tle_ephem):
    return tle_ephem.sun[183]


@pytest.fixture
def moon(tle_ephem):
    return tle_ephem.moon[183]


@pytest.fixture
def earth(tle_ephem):
    return tle_ephem.earth[183]


@pytest.fixture
def timestamp(tle_ephem) -> datetime:
    return tle_ephem.timestamp[183]


@pytest.fixture
def dummy_rust_result():
    class DummyRustResult:
        def __init__(self):
            self.evaluate_calls = []
            self.evaluate_moving_body_calls = []

        def evaluate(
            self,
            ephemeris,
            ras,
            decs,
            times,
            use_horizons: bool,
            spice_kernel: Optional[str],
        ):
            self.evaluate_calls.append(
                (ephemeris, ras, decs, times, use_horizons, spice_kernel)
            )
            return DummyRustResult()

        def evaluate_moving_body(
            self,
            ephemeris,
            ras,
            decs,
            times,
            body,
            use_horizons: bool,
            spice_kernel: Optional[str],
        ):
            self.evaluate_moving_body_calls.append(
                (ephemeris, ras, decs, times, body, use_horizons, spice_kernel)
            )
            return DummyRustResult()

    return DummyRustResult()


@pytest.fixture
def constraint_result_with_rust_ref(
    dummy_rust_result,
) -> Tuple[ConstraintResult, object]:
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
        ],
        all_satisfied=True,
        constraint_name="test",
    )
    return result


@pytest.fixture
def saa_polygon() -> list[tuple[float, float]]:
    """Simple rectangular SAA polygon for testing."""
    return [
        (-90.0, -50.0),  # Southwest
        (-40.0, -50.0),  # Southeast
        (-40.0, 0.0),  # Northeast
        (-90.0, 0.0),  # Northwest
    ]


@pytest.fixture
def mock_ephemeris_simple():
    """Simple mock ephemeris for coordinate tests"""

    class MockEphemeris:
        @property
        def timestamp(self):
            return [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 1)]

    return MockEphemeris()


@pytest.fixture
def mock_ephemeris_single():
    """Mock ephemeris with single timestamp"""

    class MockEphemeris:
        @property
        def timestamp(self):
            return [datetime(2024, 1, 1, 0, 0, 0)]

    return MockEphemeris()
