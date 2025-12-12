Using Constraints
=================

This example shows how to evaluate observational constraints against ephemeris data
to determine when targets are visible.

Basic Constraint Evaluation
---------------------------

.. code-block:: python

    import datetime as dt
    import rust_ephem as re
    from rust_ephem.constraints import SunConstraint, MoonConstraint, EclipseConstraint

    # Ensure planetary ephemeris is available for Sun/Moon positions
    re.ensure_planetary_ephemeris()

    # Create ephemeris
    tle1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    tle2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    begin = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    ephem = re.TLEEphemeris(tle1, tle2, begin, end, 300)

    # Target coordinates (Crab Nebula)
    target_ra = 83.6333   # degrees
    target_dec = 22.0145  # degrees

    # Create and evaluate a single constraint
    sun_constraint = SunConstraint(min_angle=45.0)
    result = sun_constraint.evaluate(ephem, target_ra, target_dec)

    print(f"All satisfied: {result.all_satisfied}")
    print(f"Number of violations: {len(result.violations)}")
    print(f"Total violation duration: {result.total_violation_duration()} seconds")

Combining Constraints
---------------------

Use Python operators to combine constraints logically:

.. code-block:: python

    # Method 1: Using operators (recommended)
    combined = (
        SunConstraint(min_angle=45.0) &    # AND
        MoonConstraint(min_angle=10.0) &   # AND
        ~EclipseConstraint(umbra_only=True)  # NOT (avoid eclipses)
    )

    result = combined.evaluate(ephem, target_ra, target_dec)

    # Method 2: Using Constraint class directly
    constraint = re.Constraint.and_(
        re.Constraint.sun_proximity(45.0),
        re.Constraint.moon_proximity(10.0),
        re.Constraint.not_(re.Constraint.eclipse(umbra_only=True))
    )

    result = constraint.evaluate(ephem, target_ra, target_dec)

Vectorized Batch Evaluation
---------------------------

Evaluate multiple targets efficiently using vectorized operations:

.. code-block:: python

    import numpy as np

    # Create 100 random targets
    target_ras = np.random.uniform(0, 360, 100)   # degrees
    target_decs = np.random.uniform(-90, 90, 100) # degrees

    # Create constraint
    constraint = SunConstraint(min_angle=45.0) & MoonConstraint(min_angle=10.0)

    # Batch evaluate (returns 2D boolean array)
    # Shape: (n_targets, n_times)
    # True = constraint violated, False = satisfied
    violations = constraint.in_constraint_batch(ephem, target_ras, target_decs)

    print(f"Shape: {violations.shape}")  # (100, n_times)

    # Find targets that are always visible
    always_visible = ~violations.any(axis=1)  # No violations at any time
    print(f"Always visible targets: {always_visible.sum()}")

    # Find visibility fraction for each target
    visibility_fraction = (~violations).sum(axis=1) / violations.shape[1]
    print(f"Target 0 visibility: {visibility_fraction[0]*100:.1f}%")

Working with Results
--------------------

.. code-block:: python

    result = constraint.evaluate(ephem, target_ra, target_dec)

    # Access violations
    for violation in result.violations:
        print(f"Violation: {violation.start_time} to {violation.end_time}")
        print(f"  Severity: {violation.max_severity:.2f}")
        print(f"  Description: {violation.description}")

    # Access visibility windows
    for window in result.visibility:
        print(f"Visible: {window.start_time} to {window.end_time}")
        print(f"  Duration: {window.duration_seconds:.0f} seconds")

    # Check specific times efficiently
    constraint_array = result.constraint_array  # Boolean array (cached)
    for i, is_satisfied in enumerate(constraint_array):
        if is_satisfied:
            print(f"Visible at {result.timestamp[i]}")

Available Constraint Types
--------------------------

**Proximity Constraints**

.. code-block:: python

    # Sun proximity (min/max angles in degrees)
    sun = SunConstraint(min_angle=45.0, max_angle=135.0)

    # Moon proximity
    moon = MoonConstraint(min_angle=10.0)

    # Generic body proximity (requires planetary ephemeris)
    from rust_ephem.constraints import BodyConstraint
    mars = BodyConstraint(body="Mars", min_angle=15.0)

**Earth Limb Constraint**

.. code-block:: python

    from rust_ephem.constraints import EarthLimbConstraint

    # Basic earth limb avoidance
    earth_limb = EarthLimbConstraint(min_angle=28.0)

    # With atmospheric refraction (for ground observers)
    earth_limb_refracted = EarthLimbConstraint(
        min_angle=28.0,
        include_refraction=True,
        horizon_dip=True
    )

**Eclipse Constraint**

.. code-block:: python

    # Avoid umbra only
    eclipse_umbra = EclipseConstraint(umbra_only=True)

    # Avoid umbra and penumbra
    eclipse_both = EclipseConstraint(umbra_only=False)

**Logical Combinations**

.. code-block:: python

    from rust_ephem.constraints import (
        SunConstraint, MoonConstraint, EclipseConstraint,
        AndConstraint, OrConstraint, NotConstraint, XorConstraint
    )

    # Using operators
    combined = SunConstraint(min_angle=45) & MoonConstraint(min_angle=10)
    either = SunConstraint(min_angle=45) | MoonConstraint(min_angle=10)
    not_eclipse = ~EclipseConstraint()

    # Using explicit classes
    combined_explicit = AndConstraint(constraints=[
        SunConstraint(min_angle=45),
        MoonConstraint(min_angle=10)
    ])

JSON Serialization
------------------

Constraints can be serialized to/from JSON for configuration files:

.. code-block:: python

    # Serialize to JSON
    constraint = SunConstraint(min_angle=45.0) & MoonConstraint(min_angle=10.0)
    json_str = constraint.model_dump_json()
    print(json_str)
    # {"type": "and", "constraints": [{"type": "sun", "min_angle": 45.0, ...}, ...]}

    # Load from JSON
    rust_constraint = re.Constraint.from_json(json_str)
    result = rust_constraint.evaluate(ephem, target_ra, target_dec)

Performance Tips
----------------

1. **Use batch evaluation** for multiple targets — 3-50x faster than loops
2. **Reuse constraint objects** — they cache internal Rust objects
3. **Access ``constraint_array``** property for efficient iteration over times
4. **Use ``times`` or ``indices``** parameters to evaluate only specific times

.. code-block:: python

    # Evaluate only at specific times
    specific_times = [
        dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 1, 18, 0, tzinfo=dt.timezone.utc)
    ]
    result = constraint.evaluate(ephem, ra, dec, times=specific_times)

    # Or use indices
    result = constraint.evaluate(ephem, ra, dec, indices=[0, 10, 20])

Tracking Moving Bodies with Horizons
-------------------------------------

Use the ``Constraint.evaluate_moving_body()`` method to track solar system bodies (asteroids,
comets, spacecraft) with automatic JPL Horizons fallback:

.. code-block:: python

    # Constraint for observation planning
    constraint = SunConstraint(min_angle=30) & MoonConstraint(min_angle=15)

    # Track Ceres (asteroid 1)
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="1",  # Ceres
        use_horizons=True  # Enable JPL Horizons fallback
    )

    print(f"Visibility windows: {len(result.visibility)}")
    for window in result.visibility:
        duration = (window.end_time - window.start_time).total_seconds()
        print(f"  {window.start_time} to {window.end_time} ({duration:.0f}s)")

The ``use_horizons=True`` flag enables automatic fallback to NASA's JPL Horizons
system when a body is not found in local SPICE kernels. This allows tracking of
asteroids, comets, and spacecraft without requiring additional configuration.

**Key Features:**

- **SPICE-first lookup** — Uses fast cached SPICE kernels when available
- **Automatic fallback** — Queries JPL Horizons only when SPICE lacks the body
- **Constraint integration** — Works with all constraint types and combinations
- **Full accuracy** — Returns observer-relative positions with proper frame conversions

For detailed Horizons documentation including asteroid tracking examples,
constraint combinations, and troubleshooting, see :doc:`ephemeris_horizons`.
