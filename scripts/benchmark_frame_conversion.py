"""Benchmark complete ephemeris construction on identical synthetic inputs.

Run against each release-built checkout, with the same environment/EOP cache:
python scripts/benchmark_frame_conversion.py --repeats 5 --steps 2 60
An untimed warmup per case excludes provider initialization and first-use costs.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import rust_ephem


def benchmark_queries(repeats: int) -> list[dict[str, object]]:
    """Include filtered celestial queries with tiny, gapped and reordered batches."""
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ephem = rust_ephem.GroundEphemeris(
        40, -74, 100, begin, begin + timedelta(days=1), 60
    )
    records: list[dict[str, object]] = []
    for name, indices in [
        ("tiny", [0]),
        ("gap", list(range(600)) + list(range(900, 1441))),
        ("shuffled", list(range(0, 1441, 2)) + list(range(1, 1441, 2))),
    ]:
        timings = []
        for repeat in range(repeats + 1):
            started = perf_counter()
            ephem.radec_to_altaz(180, 45, time_indices=indices)
            elapsed = perf_counter() - started
            if repeat:
                timings.append(elapsed)
        record = {"kind": name, "samples": len(indices), "median_s": median(timings)}
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 60])
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["ground", "GCRS", "ITRS", "TLE"],
        default=["ground", "GCRS", "ITRS", "TLE"],
    )
    parser.add_argument(
        "--queries", action="store_true", help="also time filtered alt/az queries"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(s <= 0 or 86400 % s for s in args.steps):
        parser.error("positive repeats and steps dividing 86400 are required")
    records = []
    with TemporaryDirectory() as directory:
        for kind in args.kinds:
            begin = (
                datetime(2025, 10, 14, tzinfo=timezone.utc)
                if kind == "TLE"
                else datetime(2024, 1, 1, tzinfo=timezone.utc)
            )
            end = begin + timedelta(days=1)
            path = Path(directory) / "state.txt"
            if kind in ("GCRS", "ITRS"):
                rows = [
                    f"ScenarioEpoch {begin.isoformat()}",
                    f"CoordinateSystem {kind}",
                ]
                for elapsed in (0, 86400):
                    rows.append(
                        f"{elapsed} {7000 + 0.001 * elapsed} "
                        f"{-1200 + 0.002 * elapsed} {1800 - 0.001 * elapsed} .001 .002 -.001"
                    )
                path.write_text("\n".join(rows) + "\n")
            for step in args.steps:
                timings = []
                for repeat in range(args.repeats + 1):
                    started = perf_counter()
                    if kind == "ground":
                        ephem = rust_ephem.GroundEphemeris(
                            40, -74, 100, begin, end, step
                        )
                    elif kind == "TLE":
                        ephem = rust_ephem.TLEEphemeris(
                            "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995",
                            "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530",
                            begin,
                            end,
                            step,
                        )
                    else:
                        ephem = rust_ephem.FileEphemeris(
                            str(path), begin=begin, end=end, step_size=step
                        )
                    elapsed_s = perf_counter() - started
                    assert len(ephem.timestamp) == 86400 // step + 1
                    del ephem
                    if repeat:
                        timings.append(elapsed_s)
                record = {
                    "kind": kind,
                    "step_s": step,
                    "median_s": median(timings),
                    "samples_s": timings,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.queries:
        records.extend(benchmark_queries(args.repeats))
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
