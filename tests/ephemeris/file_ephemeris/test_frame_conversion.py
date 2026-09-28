"""Independent ERFA regressions for the shared GCRS <-> ITRS conversion."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import erfa  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]
from numpy.typing import NDArray

from rust_ephem import FileEphemeris, get_polar_motion, get_ut1_utc_offset


@pytest.mark.parametrize("year", [2006, 2024])
@pytest.mark.parametrize("polar_motion", [False, True])
@pytest.mark.parametrize("frame", ["GCRS", "ITRS"])
@pytest.mark.parametrize("stationary", [False, True])
def test_position_and_velocity_against_erfa(
    tmp_path: Path, year: int, polar_motion: bool, frame: str, stationary: bool
) -> None:
    """Test actual file ingestion, both directions, and non-orbital velocities.

    Use the same EOP inputs in both libraries so results do not depend on which
    EOP table is installed. ERFA supplies the independent transformation; a
    five-point position difference checks velocity, including frame motion.
    """
    begin = datetime(year, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = begin + timedelta(seconds=60)
    position = np.array([7000.0, -1200.0, 1800.0])
    velocity = np.zeros(3) if stationary else np.array([1.2, 6.8, -2.4])
    path = tmp_path / "state.txt"
    lines = [f"ScenarioEpoch {begin.isoformat()}", f"CoordinateSystem {frame}"]
    for seconds in (0, 60):
        state = np.concatenate((position + seconds * velocity, velocity))
        lines.append(f"{seconds} " + " ".join(str(x) for x in state))
    path.write_text("\n".join(lines) + "\n")
    ephem = FileEphemeris(
        str(path), begin=begin, end=end, step_size=60, polar_motion=polar_motion
    )
    actual = ephem.itrs_pv if frame == "GCRS" else ephem.gcrs_pv

    for index, dt in enumerate((begin, end)):
        time = Time(dt)
        time.delta_ut1_utc = get_ut1_utc_offset(dt)
        xp, yp = get_polar_motion(dt) if polar_motion else (0.0, 0.0)
        xp, yp = np.deg2rad(np.array([xp, yp]) / 3600.0)
        center = position + index * 60 * velocity

        def transformed(seconds: float) -> NDArray[np.float64]:
            # Hold EOP fixed locally; advance TT and UT1, not UTC. This also
            # avoids introducing leap-second steps into the derivative.
            matrix = erfa.c2t06a(
                time.tt.jd1,
                time.tt.jd2 + seconds / 86400.0,
                time.ut1.jd1,
                time.ut1.jd2 + seconds / 86400.0,
                xp,
                yp,
            )
            if frame == "ITRS":
                matrix = matrix.T
            return np.asarray(matrix @ (center + seconds * velocity))

        expected_position = transformed(0.0)
        # Fourth-order stencil with a different interval than the Rust code.
        expected_velocity = (
            transformed(-4.0)
            - 8 * transformed(-2.0)
            + 8 * transformed(2.0)
            - transformed(4.0)
        ) / 24.0
        # 1 mm in position, 1 micrometre/s in velocity (km and km/s).
        np.testing.assert_allclose(
            actual.position[index], expected_position, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            actual.velocity[index], expected_velocity, rtol=0, atol=1e-9
        )
