"""An ephemeris built before a planetary kernel is loaded holds analytic Sun
and Moon states. ``get_body_pv`` with ``times=`` or ``apparent=True`` uses
them, and says so once a kernel is loaded."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

KERNEL = Path(__file__).parents[2] / "test_data" / "de440s.bsp"

# Runs with an empty HOME, so no cached kernel is loaded on demand.
SCRIPT = textwrap.dedent(
    """
    import sys
    import warnings
    from datetime import datetime, timedelta, timezone

    import numpy as np

    import rust_ephem

    KERNEL = sys.argv[1]
    TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = begin + timedelta(hours=1)
    times = [begin + timedelta(seconds=90.5), begin + timedelta(seconds=600)]


    def build():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rust_ephem.TLEEphemeris(TLE1, TLE2, begin, end, 60)


    def stale_warnings(call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = call()
        hits = [w for w in caught if "before a planetary kernel" in str(w.message)]
        assert all(w.category is UserWarning for w in hits)
        return len(hits) > 0, result


    old = build()
    print("no kernel yet:", stale_warnings(lambda: old.get_body_pv("moon", times=times))[0])

    rust_ephem.ensure_planetary_ephemeris(KERNEL, download_if_missing=False)
    for label, call in [
        ("moon times", lambda: old.get_body_pv("moon", times=times)),
        ("sun times", lambda: old.get_body_pv("sun", times=times)),
        ("moon apparent", lambda: old.get_body_pv("moon", apparent=True)),
        ("sun apparent times", lambda: old.get_body_pv("10", times=times, apparent=True)),
        ("earth times", lambda: old.get_body_pv("earth", times=times)),
        ("jupiter times", lambda: old.get_body_pv("jupiter barycenter", times=times)),
        ("moon kernel", lambda: old.get_body_pv("moon", KERNEL, times=times)),
        ("moon grid", lambda: old.get_body_pv("moon")),
        ("rebuilt moon times", lambda: build().get_body_pv("moon", times=times)),
    ]:
        print(f"{label}:", stale_warnings(call)[0])

    held = old.get_body_pv("moon", times=times).position
    mine = old.moon_pv_at(times).position - old.gcrs_pv_at(times).position
    kernel = old.get_body_pv("moon", KERNEL, times=times).position
    fresh = build().get_body_pv("moon", times=times).position
    print("held is moon_pv_at:", bool(np.array_equal(held, mine)))
    print("held differs from kernel:", bool(np.linalg.norm(held - kernel, axis=1).min() > 1.0))
    print("rebuilt matches kernel:", bool(np.abs(fresh - kernel).max() < 1e-6))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            old.get_body_pv("moon", times=times)
            print("as error: no")
        except UserWarning:
            print("as error: yes")
    """
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    if not KERNEL.exists():
        pytest.skip("planetary kernel not available")
    home = tmp_path_factory.mktemp("home")
    env = dict(os.environ, HOME=str(home))
    # Keep using the developer's (or CI's) EOP cache rather than downloading.
    env.setdefault("RUST_EPHEM_EOP_CACHE", str(Path.home() / ".cache" / "rust_ephem"))
    # Run from outside the repository: with ``-c`` the working directory is on
    # sys.path, and the source tree has no compiled extension when the package
    # was installed from a wheel.
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(KERNEL)],
        env=env,
        cwd=home,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    lines = (line.split(": ", 1) for line in done.stdout.splitlines() if ": " in line)
    return {key: value for key, value in lines}


def test_no_warning_while_no_kernel_is_loaded(report: dict[str, str]) -> None:
    assert report["no kernel yet"] == "False"


@pytest.mark.parametrize(
    "call", ["moon times", "sun times", "moon apparent", "sun apparent times"]
)
def test_warns_when_held_states_are_analytic_and_a_kernel_is_loaded(
    report: dict[str, str], call: str
) -> None:
    assert report[call] == "True"


@pytest.mark.parametrize(
    "call",
    ["earth times", "jupiter times", "moon kernel", "moon grid", "rebuilt moon times"],
)
def test_no_warning_when_the_held_states_are_not_used_or_not_analytic(
    report: dict[str, str], call: str
) -> None:
    assert report[call] == "False"


def test_the_held_states_are_still_what_is_returned(report: dict[str, str]) -> None:
    assert report["held is moon_pv_at"] == "True"
    assert report["held differs from kernel"] == "True"


def test_rebuilding_gives_the_kernel(report: dict[str, str]) -> None:
    assert report["rebuilt matches kernel"] == "True"


def test_the_warning_can_be_made_an_error(report: dict[str, str]) -> None:
    assert report["as error"] == "yes"
