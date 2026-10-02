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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 60])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(s <= 0 or 86400 % s for s in args.steps):
        parser.error("positive repeats and steps dividing 86400 are required")
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = begin + timedelta(days=1)
    records = []
    with TemporaryDirectory() as directory:
        for kind in ("ground", "GCRS", "ITRS"):
            path = Path(directory) / "state.txt"
            if kind != "ground":
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
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
