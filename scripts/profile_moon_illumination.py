"""Profile moon_illumination to demonstrate O(N^2) vs O(N) scaling."""

import time
from datetime import datetime, timedelta, timezone

from rust_ephem import TLEEphemeris

TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
BEGIN = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bench(n_points: int, step_size: int) -> float:
    """Create ephemeris with n_points timesteps and time moon_illumination."""
    end = BEGIN + timedelta(seconds=step_size * n_points)
    ephem = TLEEphemeris(
        tle1=TLE1, tle2=TLE2, begin=BEGIN, end=end, step_size=step_size
    )

    # Warm up
    ephem.moon_illumination()

    # Time it
    n_runs = 5
    start = time.perf_counter()
    for _ in range(n_runs):
        ephem.moon_illumination()
    elapsed = (time.perf_counter() - start) / n_runs
    return elapsed


print(f"{'N points':>10}  {'Time (ms)':>10}  {'µs/point':>10}")
print("-" * 36)

results = []
for n in [100, 500, 1000, 2000, 5000, 10000]:
    t = bench(n, 60)
    us_per = t * 1e6 / n
    results.append((n, t, us_per))
    print(f"{n:>10}  {t * 1000:>10.3f}  {us_per:>10.3f}")

# Check scaling: if O(N^2), doubling N quadruples time (ratio ~4)
# If O(N), doubling N doubles time (ratio ~2)
print("\nScaling analysis (time ratio when N doubles):")
for i in range(1, len(results)):
    n_prev, t_prev, _ = results[i - 1]
    n_curr, t_curr, _ = results[i]
    n_ratio = n_curr / n_prev
    t_ratio = t_curr / t_prev
    expected_linear = n_ratio
    expected_quadratic = n_ratio**2
    print(
        f"  N: {n_prev:>5} -> {n_curr:>5} ({n_ratio:.1f}x), "
        f"Time: {t_ratio:.2f}x  "
        f"(linear={expected_linear:.1f}x, quadratic={expected_quadratic:.1f}x)"
    )
