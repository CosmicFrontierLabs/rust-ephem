"""Sun and Moon positions without an initialised planetary kernel.

The kernel is process-wide state that other tests initialise, so each case
runs in a fresh interpreter whose cache directory (``~/.cache/rust_ephem``) is
a temporary one.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

KERNEL = Path(__file__).parents[2] / "test_data" / "de440s.bsp"

SCRIPT = textwrap.dedent(
    """
    import warnings
    from datetime import datetime, timezone

    import rust_ephem

    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            rust_ephem.GroundEphemeris(
                latitude=34.0, longitude=-118.0, height=100.0,
                begin=begin, end=end, step_size=600,
            )
    print("initialized", rust_ephem.is_planetary_ephemeris_initialized())
    for warning in caught:
        print("warning", warning.category.__name__, warning.message)
    """
)


def run(home: Path) -> str:
    env = dict(os.environ, HOME=str(home))
    # Keep using the developer's (or CI's) EOP cache rather than downloading.
    env.setdefault("RUST_EPHEM_EOP_CACHE", str(Path.home() / ".cache" / "rust_ephem"))
    # Run from outside the repository: with ``-c`` the working directory is on
    # sys.path, and the source tree has no compiled extension when the package
    # was installed from a wheel.
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        cwd=home,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_warns_when_no_kernel_is_available(tmp_path: Path) -> None:
    out = run(tmp_path)
    assert "initialized False" in out
    warnings = [line for line in out.splitlines() if line.startswith("warning")]
    assert len(warnings) == 2
    assert "UserWarning" in warnings[0]
    assert "analytic models" in warnings[0]
    assert "ensure_planetary_ephemeris" in warnings[0]


@pytest.mark.skipif(not KERNEL.exists(), reason="needs the DE440s test kernel")
def test_loads_a_cached_kernel_on_demand(tmp_path: Path) -> None:
    cache = tmp_path / ".cache" / "rust_ephem"
    cache.mkdir(parents=True)
    try:
        os.symlink(KERNEL, cache / "de440s.bsp")
    except OSError:
        shutil.copy(KERNEL, cache / "de440s.bsp")
    out = run(tmp_path)
    assert "initialized True" in out
    assert "warning" not in out
