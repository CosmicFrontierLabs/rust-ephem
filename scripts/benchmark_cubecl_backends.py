"""Benchmark rust-ephem's constraint compute backends through the public Python API.

Use a release-built installation. With a cached planetary SPK, run:
    python scripts/benchmark_cubecl_backends.py
    python scripts/benchmark_cubecl_backends.py --list
    python scripts/benchmark_cubecl_backends.py --cases 'day-*' --backends cpu metal
    python scripts/benchmark_cubecl_backends.py --kernel /path/to/de440s.bsp --output timings.json

Each selected backend runs the same deterministic inputs. The native Rust baseline
is always included. First-use Sun-mask probes are reported separately. The table
shows warmed median milliseconds and speedup over native, including NumPy output
materialization. Grid cases return the violation mask and all named angle arrays.
The full-day Sun grid returns 475 MB of angles alone. Use --cases to select a subset.
Set RAYON_NUM_THREADS before starting the process to control native parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
from functools import cache
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import TypedDict

import numpy as np
import numpy.typing as npt

import rust_ephem as re

BACKENDS = ("native", "cpu", "wgpu", "metal", "cuda", "hip")
STARS = 16_384
Arrays = dict[str, npt.NDArray[np.bool_] | npt.NDArray[np.float64]]


class Timing(TypedDict):
    median_ms: float
    speedup: float
    runs_ms: list[float]
    route: str


@cache
def sky(
    count: int, *, fibonacci: bool = False, seed: int = 92
) -> tuple[list[float], list[float]]:
    if fibonacci:
        i = np.arange(count, dtype=float)
        ra = np.remainder(i * 180 * (3 - np.sqrt(5)), 360)
        z = 1 - 2 * (i + 0.5) / count
    else:
        rng = np.random.default_rng(seed)
        ra = rng.uniform(0, 360, count)
        z = rng.uniform(-1, 1, count)
    return ra.tolist(), np.rad2deg(np.arcsin(z)).tolist()


@cache
def observer(satellite: bool, epochs: int) -> re.GroundEphemeris | re.TLEEphemeris:
    start = datetime(2025, 10, 14, tzinfo=timezone.utc)
    end = start + timedelta(minutes=epochs - 1)
    if satellite:
        return re.TLEEphemeris(
            "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995",
            "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530",
            start,
            end,
            60,
        )
    return re.GroundEphemeris(40, -74, 100, start, end, 60)


@dataclass(frozen=True)
class Case:
    name: str
    rule: re.ConstraintConfig
    targets: int = 4096
    epochs: int = 1024
    grid: bool = False
    apparent: bool = False
    satellite: bool = False
    fibonacci: bool = False

    def prepare(self) -> Callable[[], Arrays]:
        # The one-epoch catalogue search uses the first row of the batch ephemeris.
        eph = observer(self.satellite, 1024 if self.epochs == 1 else self.epochs)
        ra, dec = sky(self.targets, fibonacci=self.fibonacci)
        indices = [0] if self.epochs == 1 else None

        def run() -> Arrays:
            if self.grid:
                grid = self.rule.evaluate_grid(
                    eph, ra, dec, apparent=self.apparent, indices=indices
                )
                return {"mask": grid.constraint_array, **grid.constraint_values}
            return {
                "mask": self.rule.in_constraint_batch(
                    eph, ra, dec, apparent=self.apparent, indices=indices
                )
            }

        return run


def scenarios() -> list[Case]:
    sun = re.SunConstraint(min_angle=40, max_angle=140)
    moon = re.MoonConstraint(min_angle=25, max_angle=140)
    limb = re.MoonLimbConstraint(min_angle=45)
    earth = re.EarthLimbConstraint(min_angle=25, max_angle=85)
    stars = list(zip(*sky(STARS, seed=93)))
    narrow = re.BrightStarConstraint(stars=stars, fov_radius=0.1)
    wide = re.BrightStarConstraint(stars=stars, fov_radius=10)
    rules: dict[str, re.ConstraintConfig] = {
        "sun": sun,
        "moon": moon,
        "moon-limb": limb,
        "earth-limb": earth,
        "orbit-ram": re.OrbitRamConstraint(min_angle=30, max_angle=120),
        "orbit-pole": re.OrbitPoleConstraint(min_angle=15, max_angle=65),
        "earth-limb-pole": re.OrbitPoleConstraint(
            min_angle=15, max_angle=65, earth_limb_pole=True
        ),
        "body-circle": re.BodyConstraint(body="Moon", min_angle=35, max_angle=110),
        "body-polygon": re.BodyConstraint(
            body="Moon", fov_polygon=[(-20, -5), (10, -10), (5, 15)], roll_deg=45
        ),
        "airmass": re.AirmassConstraint(min_airmass=1.1, max_airmass=2.5),
        "altaz-limits": re.AltAzConstraint(
            min_altitude=10, max_altitude=70, min_azimuth=310, max_azimuth=40
        ),
        "altaz-azimuth-only": re.AltAzConstraint(min_azimuth=20, max_azimuth=310),
        "altaz-polygon": re.AltAzConstraint(polygon=[(10, 20), (25, 80), (70, 35)]),
        "moon-phase-distance": re.MoonPhaseConstraint(
            max_illumination=1,
            min_distance=25,
            max_distance=140,
            enforce_when_below_horizon=True,
        ),
        "fixed-solar-roll": re.SolarRollConstraint(
            roll_deg=37, tolerance_deg=35, panel_normal=(0, 1, 2)
        ),
    }
    cases = [Case(f"{name}-mask", rule) for name, rule in rules.items()]
    for name in ("sun", "moon", "moon-limb", "earth-limb"):
        cases.extend(
            (
                Case(f"apparent-{name}-mask", rules[name], apparent=True),
                Case(f"{name}-grid", rules[name], grid=True),
                Case(f"apparent-{name}-grid", rules[name], apparent=True, grid=True),
            )
        )
    cases.extend(
        (
            Case("small-sun-mask", sun, targets=64, epochs=128),
            Case("small-sun-grid", sun, targets=64, epochs=128, grid=True),
            Case("satellite-sun-moon-earth-mask", sun | moon | earth, satellite=True),
            Case(
                "satellite-sun-moon-earth-grid",
                sun | moon | earth,
                satellite=True,
                grid=True,
            ),
            Case("bright-star-hits", narrow, epochs=1),
            Case("bright-star-wide-hits", wide, epochs=1),
            Case("bright-star-nearest-grid", narrow, epochs=1, grid=True),
        )
    )
    # One Fibonacci direction per square degree or quarter square degree.
    for spacing in (1, 0.5):
        count = round(4 * np.pi * (180 / np.pi) ** 2 / spacing**2)
        cases.append(
            Case(f"day-sun-mask-{spacing}deg", sun, count, 1440, fibonacci=True)
        )
        if spacing == 1:
            cases.append(
                Case("day-sun-grid-1deg", sun, count, 1440, grid=True, fibonacci=True)
            )
    return cases


def select(backend: str) -> None:
    os.environ["RUST_EPHEM_GPU"] = "0" if backend == "native" else "1"
    os.environ["RUST_EPHEM_CUBECL_BACKEND"] = backend


def timed(call: Callable[[], Arrays]) -> tuple[Arrays, float]:
    start = perf_counter_ns()
    result = call()
    return result, (perf_counter_ns() - start) / 1e6


def check(result: Arrays, reference: Arrays, case: Case, backend: str) -> None:
    if case.grid and len(result) < 2:
        raise ValueError(f"{case.name}/{backend}: grid returned no angle values")
    if result.keys() != reference.keys():
        raise ValueError(f"{case.name}/{backend}: different result keys")
    for name, values in result.items():
        expected = reference[name]
        context = f"{case.name}/{backend}/{name}"
        dtype = np.bool_ if name == "mask" else np.float64
        if values.shape != (case.targets, case.epochs) or values.dtype != dtype:
            raise ValueError(f"{context}: unexpected {values.shape} {values.dtype}")
        # Bound temporary memory even for the quarter-square-degree sky mask.
        rows = max(1, (1 << 20) // case.epochs)
        for start in range(0, case.targets, rows):
            actual = values[start : start + rows]
            ref = expected[start : start + rows]
            if name == "mask":
                np.testing.assert_array_equal(actual, ref, err_msg=context)
            else:
                if not np.isfinite(actual).all():
                    raise ValueError(f"{context}: non-finite angles")
                np.testing.assert_allclose(
                    actual, ref, rtol=0, atol=1e-9, err_msg=context
                )


def probe_backends(
    requested: list[str], probe: Case
) -> tuple[dict[str, float], dict[str, str]]:
    call = probe.prepare()
    select("native")
    native, elapsed = timed(call)
    first_ms = {"native": elapsed}
    unavailable = {}
    for backend in requested:
        if backend == "native":
            continue
        select(backend)
        try:
            result, elapsed = timed(call)
        except RuntimeError as error:
            message = str(error)
            if not any(
                reason in message
                for reason in (
                    "requires a build",
                    "requires an Apple target",
                    "requires a hardware GPU adapter",
                    "requires a Metal 3 capable device",
                    "has no such device",
                )
            ):
                raise
            unavailable[backend] = message
            print(f"Skipping {backend}: {message}", flush=True)
            continue
        check(result, native, probe, backend)
        first_ms[backend] = elapsed
    return first_ms, unavailable


def measure(case: Case, backends: list[str], repeats: int) -> dict[str, Timing]:
    call = case.prepare()
    select("native")
    reference = call()
    check(reference, reference, case, "native")
    # Probe the actual routing decision outside the timed region. Geometry and
    # field width can trigger native fallback even when a batch is large.
    dispatch = False
    if len(backends) > 1:
        select("benchmark-routing-probe")
        try:
            call()
        except RuntimeError as error:
            if "Unknown RUST_EPHEM_CUBECL_BACKEND" not in str(error):
                raise
            dispatch = True

    samples: dict[str, list[float]] = {b: [] for b in backends}
    for rep in range(repeats + 1):
        offset = rep % len(backends)
        for backend in backends[offset:] + backends[:offset]:
            select(backend)
            result, elapsed = timed(call)
            check(result, reference, case, backend)
            del result
            if rep:
                samples[backend].append(elapsed)

    timings: dict[str, Timing] = {}
    baseline = median(samples["native"])
    for backend, runs in samples.items():
        route = "native"
        if backend != "native":
            route = "CubeCL eligible" if dispatch else "native fallback"
            if (
                case.grid
                and not case.apparent
                and isinstance(case.rule, re.SunConstraint)
                and dispatch
            ):
                route = (
                    "native fallback"
                    if backend in {"wgpu", "metal"}
                    else "CubeCL if FP64 supported"
                )
        ms = median(runs)
        timings[backend] = {
            "median_ms": ms,
            "speedup": baseline / ms,
            "runs_ms": runs,
            "route": route,
        }
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="list scenario names and sizes"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[*BACKENDS, "rocm"],
        default=BACKENDS,
        help="runtimes to try (native is always the baseline; rocm is an alias for hip)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["*"],
        help="scenario names or quoted shell-style patterns",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed calls per scenario/backend after warmup",
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        help="local planetary SPK (default: rust-ephem's cached kernel)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write raw samples, medians, speedups and run metadata to JSON",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases = scenarios()
    for pattern in args.cases:
        if not any(fnmatchcase(c.name, pattern) for c in cases):
            parser.error(f"no scenarios matched {pattern!r}; use --list")
    selected = [c for c in cases if any(fnmatchcase(c.name, p) for p in args.cases)]
    if args.list:
        for c in selected:
            print(f"{c.name:34} {c.targets:>7,} targets × {c.epochs:>5,} epochs")
        return

    try:
        re.ensure_planetary_ephemeris(
            py_path=str(args.kernel) if args.kernel else None, download_if_missing=False
        )
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(
            1,
            f"Could not load planetary ephemeris. Supply --kernel /path/to/de440s.bsp.\n{error}\n",
        )
    requested = list(dict.fromkeys("hip" if b == "rocm" else b for b in args.backends))
    first_ms, unavailable = probe_backends(
        requested, next(c for c in cases if c.name == "sun-mask")
    )
    backends = list(first_ms)
    metadata = {
        "rust_ephem": version("rust-ephem"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rayon_num_threads": os.getenv("RAYON_NUM_THREADS"),
        "kernel": str(args.kernel.resolve()) if args.kernel else "package cache",
        "repeats": args.repeats,
        "first_sun_mask_ms": first_ms,
        "unavailable": unavailable,
    }
    print(
        "First eligible Sun mask (ms):",
        ", ".join(f"{b} {t:.2f}" for b, t in first_ms.items()),
    )
    print("Warmed median in ms (speedup vs native). * = native fallback.")
    print(f"{'Scenario':34} " + " ".join(f"{b:>19}" for b in backends), flush=True)
    records = {}
    for case in selected:
        timings = measure(case, backends, args.repeats)
        cells = []
        for t in timings.values():
            mark = "*" if t["route"] == "native fallback" else ""
            cells.append(f"{t['median_ms']:.2f} ({t['speedup']:.2f}x){mark}")
        print(f"{case.name:34} " + " ".join(s.rjust(19) for s in cells), flush=True)
        records[case.name] = {
            "targets": case.targets,
            "epochs": case.epochs,
            "stars": STARS if case.name.startswith("bright-star") else None,
            "timings": timings,
        }
    if args.output:
        args.output.write_text(
            json.dumps({"metadata": metadata, "cases": records}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
