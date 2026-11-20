"""Results of ``evaluate`` / ``evaluate_batch`` are built without running
pydantic validation; they must still behave as ordinary models."""

import itertools
from datetime import datetime, timezone
from typing import Any

import pytest

import rust_ephem
from rust_ephem.constraints import ConstraintResult, ConstraintViolation

VALID_TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
VALID_TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)


@pytest.fixture
def results(ensure_planetary_data: Any) -> list[ConstraintResult]:
    ephem = rust_ephem.TLEEphemeris(VALID_TLE1, VALID_TLE2, BEGIN, END, 60)
    constraint = rust_ephem.SunConstraint(
        min_angle=45
    ) | rust_ephem.EarthLimbConstraint(min_angle=20)
    ras = [float(ra) for ra in range(0, 360, 30)]
    return constraint.evaluate_batch(ephem, ras, [10.0] * len(ras))


def test_fields_match_a_validated_model(results: list[ConstraintResult]) -> None:
    assert any(r.violations for r in results)
    for result in results:
        validated = ConstraintResult(
            violations=result.violations,
            all_satisfied=result.all_satisfied,
            constraint_name=result.constraint_name,
        )
        assert result.model_dump() == validated.model_dump()
        assert result.model_fields_set == validated.model_fields_set
        assert all(isinstance(v, ConstraintViolation) for v in result.violations)
        assert result.all_satisfied == (not result.violations)
        assert repr(result) == repr(validated)


def test_model_api_works(results: list[ConstraintResult]) -> None:
    result = next(r for r in results if r.violations)
    assert ConstraintResult.model_validate_json(result.model_dump_json()) == (
        ConstraintResult.model_validate(result.model_dump())
    )
    copy = result.model_copy()
    assert copy.violations == result.violations
    # The lazy views of the Rust result survive a copy.
    assert len(copy.timestamps) == len(result.timestamps) == 181
    assert list(copy.constraint_array) == list(result.constraint_array)
    assert result.total_violation_duration() > 0


def test_shared_batch_data_is_per_result(results: list[ConstraintResult]) -> None:
    """Times and cause keys are shared inside Rust but each result reports its own."""
    first, second = results[0], results[1]
    assert list(first.timestamps) == list(second.timestamps)
    assert first.cause_value_keys == second.cause_value_keys
    assert set(first.cause_value_keys) == {"sun", "earth_limb"}
    keys = first.cause_value_keys
    keys["sun"] = []
    assert first.cause_value_keys["sun"] != []
    assert any(
        list(a.constraint_array) != list(b.constraint_array)
        for a, b in itertools.pairwise(results)
    )
