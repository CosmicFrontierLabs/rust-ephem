"""Apparent directions against independent JPL/ERFA calculations in Astropy."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    GCRS,
    ICRS,
    CartesianRepresentation,
    SkyCoord,
    solar_system_ephemeris,
)
from astropy.coordinates.solar_system import (  # type: ignore[import-untyped]
    _get_apparent_body_position,
)
from astropy.time import Time  # type: ignore[import-untyped]

import rust_ephem as re

pytest.importorskip("jplephem")
KERNEL = Path(__file__).parents[2] / "test_data" / "de440s.bsp"


def reference(ephem: Any, body: str) -> SkyCoord:
    """Use the exact observer states to isolate apparent-place corrections
    from differences between the installed Earth-orientation tables.
    Astropy reads the same kernel independently through jplephem.
    """
    times = Time(ephem.timestamp)
    position = CartesianRepresentation(ephem.obsgeoloc.T * u.km)
    velocity = CartesianRepresentation(ephem.obsgeovel.T * u.km / u.s)
    with solar_system_ephemeris.set(str(KERNEL)):
        emitted = _get_apparent_body_position(body, times, str(KERNEL), position)
        return SkyCoord(
            ICRS(emitted).transform_to(
                GCRS(obstime=times, obsgeoloc=position, obsgeovel=velocity)
            )
        )


@pytest.mark.parametrize(
    "body,reference_body", [("sun", "sun"), ("301", "moon"), ("5", "jupiter")]
)
@pytest.mark.parametrize("date", ["2008-09-20", "2025-11-01", "2026-06-21"])
@pytest.mark.parametrize("observer", ["ground", "geocentre", "spacecraft"])
def test_apparent_matches_astropy(
    tmp_path: Path,
    ensure_planetary_data: None,
    body: str,
    reference_body: str,
    date: str,
    observer: str,
) -> None:
    begin = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    end = begin + timedelta(hours=12)
    ephem: Any
    if observer == "ground":
        ephem = re.GroundEphemeris(19.8207, -155.4681, 4207, begin, end, 3600)
    else:
        path = tmp_path / "observer.txt"
        # A straight, moving trajectory isolates dependence on the observer's
        # velocity, including satellite aberration, without orbit-model error.
        r = (
            np.zeros(3)
            if observer == "geocentre"
            else np.array([7000.0, -1200.0, 1800.0])
        )
        v = np.zeros(3) if observer == "geocentre" else np.array([1.2, 6.8, -2.4])
        lines = [f"ScenarioEpoch {begin.isoformat()}", "CoordinateSystem GCRS"]
        for seconds in (0, 43200):
            lines.append(f"{seconds} " + " ".join(map(str, [*(r + seconds * v), *v])))
        path.write_text("\n".join(lines) + "\n")
        ephem = re.FileEphemeris(str(path), begin=begin, end=end, step_size=3600)
    actual = ephem.get_body(body, apparent=True)
    expected = reference(ephem, reference_body)
    assert np.max(actual.separation(expected).arcsec) < 1e-4
    np.testing.assert_allclose(
        actual.distance.to_value(u.km),
        expected.distance.to_value(u.km),
        # ANISE and jplephem already differ at sub-metre scale in the
        # instantaneous planetary states. Bound that backend difference
        # separately from the much tighter angular correction check.
        atol=1e-3,
        rtol=0,
    )
    # This query returns positions only; never attach geometric or zero
    # velocities to an apparent position as if they were its derivative.
    assert not actual.data.differentials


@pytest.mark.parametrize("body", ["sun", "moon"])
def test_tle_apparent_matches_astropy(tle_ephemeris: Any, body: str) -> None:
    actual = tle_ephemeris.get_body(body, apparent=True)
    assert np.max(actual.separation(reference(tle_ephemeris, body)).arcsec) < 1e-4


def test_sun_offset_is_removed_without_changing_defaults(ground_ephemeris: Any) -> None:
    ephem = ground_ephemeris
    before = ephem.sun.cartesian.xyz.to_value(u.km).copy()
    geometric = ephem.get_body("sun")
    corrected = ephem.get_body("sun", apparent=True)
    expected = reference(ephem, "sun")
    assert np.min(geometric.separation(expected).arcsec) > 15
    assert np.max(corrected.separation(expected).arcsec) < 1e-4
    np.testing.assert_array_equal(
        ephem.get_body("sun", apparent=False).cartesian.xyz.to_value(u.km),
        geometric.cartesian.xyz.to_value(u.km),
    )
    np.testing.assert_array_equal(ephem.sun.cartesian.xyz.to_value(u.km), before)
    assert geometric.data.differentials


def test_apparent_uses_explicit_kernel(ground_ephemeris: Any) -> None:
    actual = ground_ephemeris.get_body("moon", spice_kernel=str(KERNEL), apparent=True)
    assert np.max(actual.separation(reference(ground_ephemeris, "moon")).arcsec) < 1e-4


def test_apparent_reports_invalid_requests(
    ground_ephemeris: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="use_horizons"):
        ground_ephemeris.get_body("moon", use_horizons=True, apparent=True)
    with pytest.raises(ValueError, match="Unknown body"):
        ground_ephemeris.get_body("not a body", apparent=True)
    with pytest.raises(ValueError, match="Failed to initialize"):
        ground_ephemeris.get_body(
            "sun", spice_kernel=str(tmp_path / "missing.bsp"), apparent=True
        )
    with pytest.raises(ValueError, match="SPICE cannot compute apparent"):
        ground_ephemeris.get_body("999999", apparent=True)


def test_apparent_rejects_zero_distance_observer(ensure_planetary_data: None) -> None:
    begin = datetime(2025, 11, 1, tzinfo=timezone.utc)
    moon_observer = re.SPICEEphemeris(str(KERNEL), 301, begin, begin, step_size=60)
    with pytest.raises(ValueError, match="undefined"):
        moon_observer.get_body("moon", apparent=True)
