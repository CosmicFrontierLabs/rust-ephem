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

    # Equivalent explicit construction with named intermediate constraints
    sun = SunConstraint(min_angle=45.0)
    moon = MoonConstraint(min_angle=10.0)
    eclipse = EclipseConstraint(umbra_only=True)

    constraint = (
        sun
        & moon
        & ~eclipse
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
        AndConstraint, OrConstraint, NotConstraint, XorConstraint, AtLeastConstraint
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

    # Threshold: violated when at least k sub-constraints are violated
    k_of_n = AtLeastConstraint(
        min_violated=2,
        constraints=[
            SunConstraint(min_angle=45),
            MoonConstraint(min_angle=10),
            EclipseConstraint(umbra_only=True),
        ],
    )

    # Convenience helper from any constraint instance
    k_of_n_helper = SunConstraint(min_angle=45).at_least(
        2,
        MoonConstraint(min_angle=10),
        EclipseConstraint(umbra_only=True),
    )

Threshold semantics:

- Constraints evaluate to ``True`` when blocked/not visible.
- ``min_violated=1`` is equivalent to OR over violations.
- ``min_violated=len(constraints)`` is equivalent to AND over violations.

Shared-Axis Multi-Instrument Planning
------------------------------------

Use boresight offsets when multiple instruments share the same mount axis but
have different fixed pointing directions relative to the primary boresight.

The pattern is:

1. Define the primary instrument constraint(s)
2. Define each secondary/tertiary instrument constraint
3. Wrap secondary/tertiary constraints with a boresight offset
4. Combine all with logical OR (blocked if any instrument is blocked)

Constraints are ``True`` when a target is **not visible**. For multi-instrument
planning, combine with ``|`` so the commanded target is marked blocked if either
the primary or any offset secondary/tertiary instrument is blocked.

.. code-block:: python

    import rust_ephem as re
    from rust_ephem.constraints import SunConstraint, MoonConstraint

    re.ensure_planetary_ephemeris()

    # Primary instrument constraint
    primary = SunConstraint(min_angle=45.0)

    # Secondary instrument constraint at fixed offset from primary boresight.
    # roll_deg defaults to 0.0 (instrument aligned with spacecraft frame).
    # instantaneous_field_of_regard sweeps all spacecraft roll angles by default
    # to compute the accessible sky fraction over all orientations.
    secondary_offset = MoonConstraint(min_angle=12.0).boresight_offset(
        pitch_deg=1.2,
        yaw_deg=-0.8,
    )

    # Commanded pointing is blocked if either instrument is blocked
    combined = primary | secondary_offset
    result = combined.evaluate(ephem, target_ra, target_dec)

    print(result.all_satisfied)

Equivalent Pydantic configuration:

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, MoonConstraint

    combined = SunConstraint(min_angle=45.0) | (
        MoonConstraint(min_angle=12.0).boresight_offset(
            pitch_deg=1.2,
            yaw_deg=-0.8,
        )
    )

Euler angles are specified in degrees as ``roll_deg`` (+X), ``pitch_deg`` (+Y),
and ``yaw_deg`` (+Z).

Spacecraft roll is separate from fixed instrument offsets. Pass spacecraft roll
at evaluation time:

.. code-block:: python

    result = combined.evaluate(
        ephem,
        target_ra,
        target_dec,
        target_roll=95.0,
    )

This lets you keep one fixed boresight definition while evaluating different
commanded roll states for the same RA/Dec pointing.

When ``target_roll`` is omitted (or ``None``) and the constraint contains a
boresight offset with non-zero pitch/yaw, all three evaluation methods
(``evaluate``, ``in_constraint``, ``in_constraint_batch``) automatically sweep
roll angles and report a target as blocked only when **every** possible roll is
blocked — i.e., no valid spacecraft orientation exists.  The sweep resolution is
controlled by the ``n_roll_samples`` parameter (default
:data:`~rust_ephem.constraints.DEFAULT_N_ROLL_SAMPLES` = 72 ≈ 5° resolution).
This is the conservative "is there any viable roll?" check.  Pass an explicit
``target_roll`` value to evaluate against a single commanded roll.

Instantaneous Field of Regard (steradians)
------------------------------------------

For any constraint (single or combined), you can compute instantaneous visible
sky area (solid angle) at one timestamp.

The result is returned in steradians and always lies in ``[0, 4π]``.

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, MoonConstraint, DEFAULT_N_POINTS

    constraint = SunConstraint(min_angle=45.0) | MoonConstraint(min_angle=12.0)

    # Fastest path: evaluate at an ephemeris index
    field_sr = constraint.instantaneous_field_of_regard(
        ephemeris=ephem,
        index=0,
        n_points=DEFAULT_N_POINTS,
    )

    visible_fraction = field_sr / (4.0 * 3.141592653589793)
    print(f"Field of regard: {field_sr:.3f} sr ({visible_fraction:.2%} of full sky)")

You can also evaluate by datetime:

.. code-block:: python

    t0 = ephem.timestamp[0]
    field_sr = constraint.instantaneous_field_of_regard(
        ephemeris=ephem,
        time=t0,
        n_points=DEFAULT_N_POINTS,
    )

Notes:

- Exactly one of ``time`` or ``index`` must be provided.
- ``n_points`` controls integration accuracy vs speed (higher = more accurate, slower).
- ``n_roll_samples`` controls how finely spacecraft roll is swept when ``target_roll`` is
  not specified (default ``DEFAULT_N_ROLL_SAMPLES`` = 72 ≈ 5° resolution). Reduce to speed
  up at the cost of accuracy; ignored when ``target_roll`` is given or when no pitch/yaw
  offset is present.
- Constraints are ``True`` when blocked/not visible, so field of regard integrates where constraint is ``False``.
- For boresight-offset constraints with non-zero pitch/yaw, the sky is sampled at 72
  evenly-spaced spacecraft roll angles when ``target_roll`` is not specified.  A direction
  is counted accessible if *any* roll angle satisfies the inner constraint, modelling a
  spacecraft that can rotate about its pointing axis.  The evaluation scales with
  ``n_roll_samples``; the default 72 is ~72× slower than a single-roll evaluation at the
  same ``n_points``.  Pass ``target_roll`` to pin spacecraft roll and recover the faster
  single-pass evaluation.

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
