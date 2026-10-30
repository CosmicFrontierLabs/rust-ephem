#!/usr/bin/env python
"""Profile visibility getter performance.

Tests whether visibility recomputes constraint_vec on each access.
"""

import time
from datetime import datetime, timedelta, timezone

from rust_ephem import GroundEphemeris, SunConstraint


def profile_visibility(step_size: int, duration_hours: int = 12) -> dict:
    """Profile visibility getter at a given step_size."""
    results = {}

    begin = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = begin + timedelta(hours=duration_hours)

    ephem = GroundEphemeris(
        latitude=40.0,
        longitude=-74.0,
        height=0.0,
        begin=begin,
        end=end,
        step_size=step_size,
    )
    n_points = len(ephem.timestamp)
    results["step_size"] = step_size
    results["points"] = n_points

    constraint = SunConstraint(min_angle=45.0)
    result = constraint.evaluate(ephem, target_ra=180.0, target_dec=45.0)

    # Profile constraint_array access (should be cached after first call)
    times_constraint_array = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = result.constraint_array
        t1 = time.perf_counter()
        times_constraint_array.append((t1 - t0) * 1000)
    results["constraint_array_ms"] = times_constraint_array

    # Profile visibility access (should this be cached?)
    times_visibility = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = result.visibility
        t1 = time.perf_counter()
        times_visibility.append((t1 - t0) * 1000)
    results["visibility_ms"] = times_visibility

    return results


def main():
    print("=" * 80)
    print("Profiling visibility getter vs constraint_array")
    print("=" * 80)
    print()

    for step_size in [60, 10, 1]:
        print(f"step_size={step_size}s:")
        results = profile_visibility(step_size)
        print(f"  Points: {results['points']}")
        print(
            f"  constraint_array (5 calls): {[f'{t:.3f}' for t in results['constraint_array_ms']]} ms"
        )
        print(
            f"  visibility (5 calls):       {[f'{t:.3f}' for t in results['visibility_ms']]} ms"
        )

        # Check if first call is slower (computation) vs subsequent (cached)
        ca = results["constraint_array_ms"]
        vis = results["visibility_ms"]
        print(
            f"  constraint_array: first={ca[0]:.3f}ms, avg_rest={sum(ca[1:]) / 4:.3f}ms"
        )
        print(
            f"  visibility:       first={vis[0]:.3f}ms, avg_rest={sum(vis[1:]) / 4:.3f}ms"
        )
        print()


if __name__ == "__main__":
    main()
