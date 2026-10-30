#!/usr/bin/env python
"""Profile in_constraint scaling behavior with different step_size values.

This script measures:
1. Ephemeris creation time
2. Constraint evaluation time (creating ConstraintResult)
3. Per-call in_constraint() lookup time
4. Batch in_constraint_batch() time

To assess the claim about array cloning overhead.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from rust_ephem import SunConstraint

logger = logging.getLogger(__name__)


def profile_step_size(step_size: int, duration_hours: int = 12) -> dict:
    """Profile constraint operations at a given step_size."""
    from rust_ephem import GroundEphemeris

    results = {}

    # Setup time range
    begin = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = begin + timedelta(hours=duration_hours)

    # Calculate expected points
    expected_points = (duration_hours * 3600) // step_size + 1
    results["step_size"] = step_size
    results["expected_points"] = expected_points

    # 1. Profile ephemeris creation (using GroundEphemeris to avoid TLE issues)
    t0 = time.perf_counter()
    ephem = GroundEphemeris(
        latitude=40.0,
        longitude=-74.0,
        height=0.0,
        begin=begin,
        end=end,
        step_size=step_size,
    )
    t1 = time.perf_counter()
    results["ephemeris_creation_ms"] = (t1 - t0) * 1000
    results["actual_points"] = len(ephem.timestamp)

    # 2. Profile constraint evaluation (creating ConstraintResult)
    constraint = SunConstraint(min_angle=45.0)

    t0 = time.perf_counter()
    result = constraint.evaluate(ephem, target_ra=180.0, target_dec=45.0)
    t1 = time.perf_counter()
    results["constraint_eval_ms"] = (t1 - t0) * 1000

    # 3. Profile in_constraint() per-call (sample 100 random timestamps)
    timestamps = list(ephem.timestamp)
    n_samples = min(100, len(timestamps))
    sample_indices = list(
        range(0, len(timestamps), max(1, len(timestamps) // n_samples))
    )[:n_samples]
    sample_times = [timestamps[i] for i in sample_indices]

    t0 = time.perf_counter()
    for ts in sample_times:
        _ = result.in_constraint(ts)
    t1 = time.perf_counter()
    results["in_constraint_total_ms"] = (t1 - t0) * 1000
    results["in_constraint_per_call_us"] = ((t1 - t0) * 1_000_000) / n_samples
    results["n_samples"] = n_samples

    # 4. Profile batch evaluation (evaluating 10 targets at once)
    ras = [180.0 + i * 10 for i in range(10)]
    decs = [45.0 + i * 5 for i in range(10)]

    t0 = time.perf_counter()
    batch_result = constraint.in_constraint_batch(
        ephem, target_ras=ras, target_decs=decs
    )
    t1 = time.perf_counter()
    results["batch_eval_ms"] = (t1 - t0) * 1000
    results["batch_shape"] = batch_result.shape

    # 5. Profile multiple sequential constraint evaluations
    # This is where the claim says array cloning overhead occurs
    constraints = [
        SunConstraint(min_angle=45.0),
        SunConstraint(min_angle=30.0),
        SunConstraint(min_angle=60.0),
    ]

    t0 = time.perf_counter()
    for c in constraints:
        _ = c.evaluate(ephem, target_ra=180.0, target_dec=45.0)
    t1 = time.perf_counter()
    results["multi_constraint_eval_ms"] = (t1 - t0) * 1000
    results["per_constraint_eval_ms"] = ((t1 - t0) * 1000) / len(constraints)

    return results


def main():
    print("=" * 80)
    print("Profiling in_constraint scaling with different step_size values")
    print("=" * 80)
    print()

    # Test different step sizes
    step_sizes = [60, 30, 10, 5, 1]

    all_results = []
    for step_size in step_sizes:
        print(f"Testing step_size={step_size}s...")
        try:
            results = profile_step_size(step_size)
            all_results.append(results)
            print(f"  Points: {results['actual_points']}")
            print(f"  Ephemeris creation: {results['ephemeris_creation_ms']:.1f} ms")
            print(f"  Constraint eval: {results['constraint_eval_ms']:.3f} ms")
            print(
                f"  in_constraint per call: {results['in_constraint_per_call_us']:.1f} µs"
            )
            print(f"  Batch eval (10 targets): {results['batch_eval_ms']:.3f} ms")
            print(
                f"  Multi-constraint (3x): {results['multi_constraint_eval_ms']:.3f} ms"
            )
            print()
        except Exception:
            logger.exception("Profiling failed for step_size=%s", step_size)

    # Print summary table
    print()
    print("Summary Table:")
    print("-" * 100)
    print(
        f"{'step_size':>10} {'points':>10} {'ephem_ms':>12} {'eval_ms':>12} "
        f"{'in_const_µs':>12} {'batch_ms':>12} {'multi_ms':>12}"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['step_size']:>10} {r['actual_points']:>10} "
            f"{r['ephemeris_creation_ms']:>12.1f} {r['constraint_eval_ms']:>12.3f} "
            f"{r['in_constraint_per_call_us']:>12.1f} {r['batch_eval_ms']:>12.3f} "
            f"{r['multi_constraint_eval_ms']:>12.3f}"
        )

    # Calculate scaling factors
    if len(all_results) >= 2:
        print()
        print("Scaling Analysis (relative to step_size=60s):")
        print("-" * 80)
        baseline = all_results[0]  # 60s
        for r in all_results[1:]:
            point_ratio = r["actual_points"] / baseline["actual_points"]
            eval_ratio = r["constraint_eval_ms"] / baseline["constraint_eval_ms"]
            in_const_ratio = (
                r["in_constraint_per_call_us"] / baseline["in_constraint_per_call_us"]
            )
            print(
                f"step_size={r['step_size']:>3}s: "
                f"points={point_ratio:.1f}x, "
                f"eval_time={eval_ratio:.1f}x, "
                f"in_constraint={in_const_ratio:.1f}x"
            )

    # Memory bandwidth estimate (if claim is true)
    print()
    print("Theoretical memory bandwidth analysis (if claim is true):")
    print("-" * 80)
    for r in all_results:
        points = r["actual_points"]
        # Each clone: N x 3 x 8 bytes
        clone_size_mb = (points * 3 * 8) / (1024 * 1024)
        # Two clones per constraint call (sun + gcrs)
        per_call_mb = clone_size_mb * 2
        print(
            f"step_size={r['step_size']:>3}s: "
            f"{points:>6} points, "
            f"array clone={clone_size_mb:.3f} MB, "
            f"per constraint call={per_call_mb:.3f} MB"
        )


if __name__ == "__main__":
    main()
