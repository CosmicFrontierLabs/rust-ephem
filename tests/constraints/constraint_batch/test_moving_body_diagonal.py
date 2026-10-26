"""Paired moving-body results must match the Cartesian reference geometry."""

from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from rust_ephem import (
    AirmassConstraint,
    AltAzConstraint,
    BodyConstraint,
    BrightStarConstraint,
    Constraint,
    DaytimeConstraint,
    EarthLimbConstraint,
    EclipseConstraint,
    GroundEphemeris,
    MoonConstraint,
    MoonPhaseConstraint,
    OrbitPoleConstraint,
    OrbitRamConstraint,
    SAAConstraint,
    SolarRollConstraint,
    SunConstraint,
    TLEEphemeris,
)
from rust_ephem.constraints import RollReference, RustConstraintMixin


def moving_constraints() -> dict[str, RustConstraintMixin]:
    sun = SunConstraint(min_angle=30, max_angle=150)
    moon = MoonConstraint(min_angle=10, max_angle=150)
    earth = EarthLimbConstraint(
        min_angle=5, max_angle=150, include_refraction=True, horizon_dip=True
    )
    eclipse = EclipseConstraint(umbra_only=False)
    coast = sun | moon | earth | (SunConstraint(min_angle=45, max_angle=135) & ~eclipse)
    return {
        "sun": sun,
        "moon": moon,
        "earth": earth,
        "eclipse": eclipse,
        "umbra": EclipseConstraint(umbra_only=True),
        "daytime": DaytimeConstraint(),
        "airmass": AirmassConstraint(max_airmass=3, min_airmass=1),
        "alt_az": AltAzConstraint(min_altitude=10, max_altitude=70),
        "saa": SAAConstraint(polygon=[(-80.0, -40.0), (0.0, -40.0), (0.0, 0.0)]),
        "orbit_pole": OrbitPoleConstraint(min_angle=30, earth_limb_pole=True),
        "orbit_ram": OrbitRamConstraint(min_angle=10, max_angle=120),
        "moon_phase": MoonPhaseConstraint(max_illumination=0.4, min_distance=20),
        "bright_circle": BrightStarConstraint(
            stars=[(0.0, 0.0), (50.0, 30.0)], fov_radius=8
        ),
        "bright_polygon": BrightStarConstraint(
            stars=[(0.0, 0.0), (50.0, 30.0)],
            fov_polygon=[(-12.0, 8.0), (12.0, 8.0), (12.0, 20.0), (-12.0, 20.0)],
            roll_deg=31,
        ),
        "solar_roll": SolarRollConstraint(tolerance_deg=20, roll_deg=30),
        "body_circle": BodyConstraint(body="Earth", min_angle=10),
        "body_polygon": BodyConstraint(
            body="Moon",
            fov_polygon=[(-10.0, -4.0), (10.0, -4.0), (0.0, 10.0)],
            roll_deg=27,
        ),
        "coast": coast,
        "nested": ~(sun ^ moon) | (earth & eclipse),
        "threshold": sun.at_least(2, moon, earth, eclipse),
        "duplicates": sun | SunConstraint(min_angle=50) | ~sun,
        "north_offset": coast.boresight_offset(
            roll_deg=17,
            pitch_deg=11,
            yaw_deg=4,
            roll_clockwise=True,
            roll_reference=RollReference.NORTH,
        ),
        "sun_offset": coast.boresight_offset(
            roll_deg=-23, pitch_deg=8, yaw_deg=3, roll_reference=RollReference.SUN
        ),
    }


@pytest.mark.parametrize("name", list(moving_constraints()))
@pytest.mark.parametrize("backend_kind", ["python", "rust"])
def test_moving_body_matches_cartesian_diagonal(
    ground_ephem_5h: GroundEphemeris, name: str, backend_kind: str
) -> None:
    constraint = moving_constraints()[name]
    reference = Constraint.from_json(constraint.model_dump_json())
    backend: Any = constraint if backend_kind == "python" else reference
    n = len(ground_ephem_5h.timestamp)
    ras = np.linspace(0, 300, n).tolist()
    decs = np.linspace(-60, 60, n).tolist()
    expected_mask = np.diag(reference.in_constraint_batch(ground_ephem_5h, ras, decs))
    reference_results = reference.evaluate_batch(ground_ephem_5h, ras, decs)
    actual = backend.evaluate_moving_body(
        ground_ephem_5h, target_ras=ras, target_decs=decs
    )
    np.testing.assert_array_equal(actual.constraint_array, expected_mask)
    assert actual.ras == ras and actual.decs == decs
    assert actual.all_satisfied == (not expected_mask.any())
    assert actual.constraint_name == reference_results[0].constraint_name
    assert (
        actual.constraint_values.keys() == reference_results[0].constraint_values.keys()
    )
    for key, values in actual.constraint_values.items():
        expected = [
            result.constraint_values[key][i]
            for i, result in enumerate(reference_results)
        ]
        np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-8)
    assert {key: sorted(values) for key, values in actual.cause_value_keys.items()} == {
        key: sorted(values)
        for key, values in reference_results[0].cause_value_keys.items()
    }
    for window in actual.visibility:
        for causes in [window.start_cause, window.end_cause]:
            if causes is not None:
                assert set(causes) <= actual.cause_value_keys.keys()


@pytest.mark.parametrize("name", ["sun", "coast", "sun_offset", "moon_phase"])
def test_selected_timestamps_use_matching_ephemeris_rows(
    ground_ephem_5h: GroundEphemeris, name: str
) -> None:
    constraint = moving_constraints()[name]
    reference = Constraint.from_json(constraint.model_dump_json())
    indices = [5, 1, 5]
    times = [ground_ephem_5h.timestamp[i] for i in indices]
    ras, decs = [30.0, 90.0, 180.0], [10.0, -40.0, 50.0]
    actual = constraint.evaluate_moving_body(ground_ephem_5h, ras, decs, times=times)
    expected = np.diag(
        reference.in_constraint_batch(ground_ephem_5h, ras, decs, indices=indices)
    )
    np.testing.assert_array_equal(actual.constraint_array, expected)
    assert actual.timestamps == times
    full = reference.evaluate_batch(ground_ephem_5h, ras, decs, indices=indices)
    for key in actual.constraint_values:
        np.testing.assert_allclose(
            actual.constraint_values[key],
            [result.constraint_values[key][i] for i, result in enumerate(full)],
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.parametrize("body", ["Moon", "Sun"])
def test_body_lookup_matches_explicit_track(
    ground_ephem_5h: GroundEphemeris, body: str
) -> None:
    constraint = moving_constraints()["coast"]
    skycoord = ground_ephem_5h.get_body(body)
    explicit = constraint.evaluate_moving_body(
        ground_ephem_5h, skycoord.ra.deg.tolist(), skycoord.dec.deg.tolist()
    )
    lookup = constraint.evaluate_moving_body(ground_ephem_5h, body=body)
    assert lookup == explicit


@pytest.mark.parametrize("name", ["sun", "coast", "alt_az", "sun_offset"])
def test_empty_track_preserves_schema(
    ground_ephem_5h: GroundEphemeris, name: str
) -> None:
    constraint = moving_constraints()[name]
    empty = constraint.evaluate_moving_body(ground_ephem_5h, [], [], times=[])
    populated = constraint.evaluate_moving_body(
        ground_ephem_5h, [30.0], [10.0], times=[ground_ephem_5h.timestamp[0]]
    )
    assert empty.constraint_array == []
    assert empty.visibility == []
    assert empty.all_satisfied
    assert empty.constraint_values == {key: [] for key in populated.constraint_values}
    assert {key: sorted(values) for key, values in empty.cause_value_keys.items()} == {
        key: sorted(values) for key, values in populated.cause_value_keys.items()
    }


def test_invalid_track_times_raise(ground_ephem_5h: GroundEphemeris) -> None:
    constraint = SunConstraint(min_angle=30)
    with pytest.raises(ValueError, match="same length"):
        constraint.evaluate_moving_body(ground_ephem_5h, [0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="timestamps length"):
        constraint.evaluate_moving_body(ground_ephem_5h, [0.0], [0.0])
    with pytest.raises(ValueError):
        constraint.evaluate_moving_body(
            ground_ephem_5h,
            [0.0],
            [0.0],
            times=[ground_ephem_5h.timestamp[0] - timedelta(seconds=1)],
        )


@pytest.mark.parametrize("name", ["coast", "sun_offset", "moon_phase"])
def test_spacecraft_track_matches_batch(tle_ephemeris: TLEEphemeris, name: str) -> None:
    constraint = moving_constraints()[name]
    reference = Constraint.from_json(constraint.model_dump_json())
    indices = [0, 2, 5, 11]
    ras, decs = [30.0, 90.0, 180.0, 310.0], [10.0, -40.0, 50.0, 0.0]
    times = [tle_ephemeris.timestamp[i] for i in indices]
    actual = constraint.evaluate_moving_body(tle_ephemeris, ras, decs, times=times)
    expected = np.diag(
        reference.in_constraint_batch(tle_ephemeris, ras, decs, indices=indices)
    )
    np.testing.assert_array_equal(actual.constraint_array, expected)
    rows = reference.evaluate_batch(tle_ephemeris, ras, decs, indices=indices)
    for key in actual.constraint_values:
        np.testing.assert_allclose(
            actual.constraint_values[key],
            [row.constraint_values[key][i] for i, row in enumerate(rows)],
            rtol=1e-8,
            atol=1e-8,
        )


def test_constant_track_preserves_visibility_causes(
    ground_ephem_5h: GroundEphemeris,
) -> None:
    constraint = moving_constraints()["coast"]
    n = len(ground_ephem_5h.timestamp)
    moving = constraint.evaluate_moving_body(ground_ephem_5h, [90.0] * n, [10.0] * n)
    sidereal = constraint.evaluate(ground_ephem_5h, 90.0, 10.0)
    assert moving.constraint_array == sidereal.constraint_array
    assert len(moving.visibility) == len(sidereal.visibility)
    for a, b in zip(moving.visibility, sidereal.visibility):
        assert (a.start_time, a.end_time, a.start_cause, a.end_cause) == (
            b.start_time,
            b.end_time,
            b.start_cause,
            b.end_cause,
        )


@pytest.mark.parametrize(
    "name", ["coast", "sun_offset", "solar_roll", "bright_polygon"]
)
@pytest.mark.parametrize("roll", [0.0, 37.0])
def test_fixed_roll_moving_track_matches_batch(
    ground_ephem_5h: GroundEphemeris, name: str, roll: float
) -> None:
    constraint = moving_constraints()[name]
    n = len(ground_ephem_5h.timestamp)
    ras, decs = np.linspace(0, 300, n).tolist(), np.linspace(-60, 60, n).tolist()
    moving = constraint.evaluate_moving_body(
        ground_ephem_5h, ras, decs, target_roll=roll
    )
    rows = constraint.evaluate_batch(
        ground_ephem_5h, ras, decs, target_rolls=[roll] * n
    )
    expected = np.diag(
        constraint.in_constraint_batch(
            ground_ephem_5h, ras, decs, target_rolls=[roll] * n
        )
    )
    np.testing.assert_array_equal(moving.constraint_array, expected)
    assert moving.constraint_name == rows[0].constraint_name
    for key in moving.constraint_values:
        np.testing.assert_allclose(
            moving.constraint_values[key],
            [row.constraint_values[key][i] for i, row in enumerate(rows)],
            rtol=1e-8,
            atol=1e-8,
        )
