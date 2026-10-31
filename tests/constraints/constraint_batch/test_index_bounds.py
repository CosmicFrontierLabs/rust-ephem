"""Invalid timestamp selections must raise Python errors, even in release builds."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

import rust_ephem


@pytest.mark.parametrize("api", ["evaluate", "evaluate_batch", "in_constraint_batch"])
def test_out_of_bounds_indices_do_not_abort(
    ensure_planetary_data: None, api: str
) -> None:
    # Run outside pytest's process because panic=abort cannot be caught by pytest.
    script = dedent("""
        import sys
        from datetime import datetime, timedelta, timezone
        import pytest
        import rust_ephem
        from rust_ephem.constraints import (
            AltAzConstraint, BrightStarConstraint, SolarRollConstraint,
        )

        rust_ephem.ensure_planetary_ephemeris(
            py_path=sys.argv[1], download_if_missing=False,
        )
        begin = datetime(2025, 9, 23, tzinfo=timezone.utc)
        ephem = rust_ephem.GroundEphemeris(
            34, -118, 100, begin, begin + timedelta(minutes=10), 300,
        )
        altaz = AltAzConstraint(min_altitude=20)
        solar = SolarRollConstraint(tolerance_deg=20)
        # Include a time-invariant leaf and nested, cached attributed evaluation.
        constraints = [
            altaz, solar, altaz ^ solar,
            BrightStarConstraint(stars=[], fov_radius=10),
        ]
        api = sys.argv[2]
        for constraint in constraints:
            for indices in [3, [3], [0, 3], [2**64 - 1]]:
                for fixed in [False, True]:
                    for empty in ([False] if api == "evaluate" else [False, True]):
                        kwargs = {"indices": indices, "n_roll_samples": 3}
                        if api == "evaluate":
                            targets = (90., 30.)
                            if fixed:
                                kwargs["target_roll"] = 15.
                        else:
                            targets = ([], []) if empty else ([90.], [30.])
                            if fixed:
                                kwargs["target_rolls"] = [] if empty else [15.]
                        with pytest.raises(IndexError, match="index .* out of bounds"):
                            getattr(constraint, api)(ephem, *targets, **kwargs)
    """)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parents[2] / "test_data" / "de440s.bsp"),
            api,
        ],
        # Match pytest's imported package; the checkout may lack the wheel's extension.
        cwd=Path(rust_ephem.__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
