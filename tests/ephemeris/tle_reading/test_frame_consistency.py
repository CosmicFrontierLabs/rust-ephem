"""TLE frame routes must agree with each other and the Vallado/ERFA reference."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import erfa  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]
from numpy.typing import NDArray

from rust_ephem import FileEphemeris, TLEEphemeris, get_polar_motion, get_ut1_utc_offset

TLES = [
    (
        datetime(2008, 9, 20, 23, 59, 37, 250000, tzinfo=timezone.utc),
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    ),
    (
        datetime(2025, 10, 14, 23, 59, 37, 250000, tzinfo=timezone.utc),
        "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995",
        "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530",
    ),
]


@pytest.mark.parametrize("begin,tle1,tle2", TLES)
@pytest.mark.parametrize("polar_motion", [False, True])
@pytest.mark.parametrize("step,duration", [(2, 600), (60, 600), (600, 1800), (60, 60)])
def test_tle_frame_references_and_routes(
    tmp_path: Path,
    begin: datetime,
    tle1: str,
    tle2: str,
    polar_motion: bool,
    step: int,
    duration: int,
) -> None:
    """Cover dense, sparse and tiny batches across midnight and cache boundaries.

    EOP inputs match rust-ephem; ERFA independently defines Vallado TEME->ITRS
    (gmst82, no s') and IAU 2006/2000A GCRS->ITRS. Velocity references use a
    fourth-order 2-second stencil, distinct from the production derivative.
    """
    end = begin + timedelta(seconds=duration)
    tle = TLEEphemeris(tle1, tle2, begin, end, step, polar_motion=polar_motion)

    for frame, source, expected in [
        ("GCRS", tle.gcrs_pv, tle.itrs_pv),
        ("ITRS", tle.itrs_pv, tle.gcrs_pv),
    ]:
        lines = [f"ScenarioEpoch {begin.isoformat()}", f"CoordinateSystem {frame}"]
        for dt, pos, vel in zip(
            tle.timestamp, source.position, source.velocity, strict=True
        ):
            state = " ".join(f"{x:.17g}" for x in [*pos, *vel])
            lines.append(f"{(dt - begin).total_seconds():.17g} {state}")
        path = tmp_path / f"{frame}.txt"
        path.write_text("\n".join(lines) + "\n")
        converted = FileEphemeris(
            str(path), begin=begin, end=end, step_size=step, polar_motion=polar_motion
        )
        actual = converted.itrs_pv if frame == "GCRS" else converted.gcrs_pv
        np.testing.assert_allclose(
            actual.position, expected.position, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            actual.velocity, expected.velocity, rtol=0, atol=1e-9
        )

    for index, dt in enumerate(tle.timestamp):
        time = Time(dt)
        time.delta_ut1_utc = get_ut1_utc_offset(dt)
        xp, yp = get_polar_motion(dt) if polar_motion else (0.0, 0.0)
        xp, yp = np.deg2rad(np.array([xp, yp]) / 3600.0)
        position = tle.teme_pv.position[index]
        velocity = tle.teme_pv.velocity[index]

        def transformed(seconds: float, to_gcrs: bool) -> NDArray[np.float64]:
            tt = (time.tt.jd1, time.tt.jd2 + seconds / 86400.0)
            ut1 = (time.ut1.jd1, time.ut1.jd2 + seconds / 86400.0)
            teme_itrs = erfa.c2tcio(
                np.eye(3), erfa.gmst82(*ut1), erfa.pom00(xp, yp, 0.0)
            )
            matrix = (
                erfa.c2t06a(*tt, *ut1, xp, yp).T @ teme_itrs if to_gcrs else teme_itrs
            )
            return np.asarray(matrix @ (position + seconds * velocity))

        for to_gcrs, actual in [(False, tle.itrs_pv), (True, tle.gcrs_pv)]:
            expected_position = transformed(0.0, to_gcrs)
            expected_velocity = (
                transformed(-4.0, to_gcrs)
                - 8 * transformed(-2.0, to_gcrs)
                + 8 * transformed(2.0, to_gcrs)
                - transformed(4.0, to_gcrs)
            ) / 24.0
            np.testing.assert_allclose(
                actual.position[index], expected_position, rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(
                actual.velocity[index], expected_velocity, rtol=0, atol=1e-9
            )
