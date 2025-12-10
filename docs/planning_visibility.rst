Calculating Visibility Windows
==============================

This guide shows how to use the constraint system to calculate visibility
windows for astronomical observations. Visibility windows are time periods
when a target is observable—that is, when all observational constraints
are satisfied.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

When planning observations from satellites or ground stations, you need to
determine when targets are actually observable. This requires checking multiple
constraints:

- **Sun avoidance**: Target must be far enough from the Sun
- **Moon avoidance**: Target must be far enough from the Moon
- **Earth limb**: Target must not be blocked by Earth (for spacecraft)
- **Eclipse avoidance**: Spacecraft must not be in Earth's shadow
- **Bright object avoidance**: Target must avoid planets or other bright objects

The ``rust-ephem`` constraint system lets you combine these constraints with
logical operators and efficiently evaluate them to find visibility windows.

Basic Setup
-----------

First, import the required modules and create an ephemeris:

.. code-block:: python

   import rust_ephem
   from rust_ephem.constraints import (
       SunConstraint,
       MoonConstraint,
       EarthLimbConstraint,
       EclipseConstraint,
       BodyConstraint,
   )
   from datetime import datetime, timezone

   # Initialize planetary ephemeris (required for constraint calculations)
   rust_ephem.ensure_planetary_ephemeris()

   # Create ephemeris for the ISS over a 24-hour period
   begin = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
   end = datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)

   ephem = rust_ephem.TLEEphemeris(
       norad_id=25544,  # ISS
       begin=begin,
       end=end,
       step_size=60  # 1-minute resolution
   )

   # Define target coordinates (Crab Nebula / M1)
   target_ra = 83.63   # degrees
   target_dec = 22.01  # degrees


Simple Constraint Evaluation
----------------------------

Creating a Single Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and evaluate a simple Sun avoidance constraint:

.. code-block:: python

   # Target must be at least 45° from the Sun
   sun_constraint = SunConstraint(min_angle=45.0)

   # Evaluate the constraint
   result = sun_constraint.evaluate(ephem, target_ra, target_dec)

   # Check results
   print(f"Constraint: {result.constraint_name}")
   print(f"All times satisfied: {result.all_satisfied}")
   print(f"Number of violations: {len(result.violations)}")
   print(f"Number of visibility windows: {len(result.visibility)}")

Understanding the Result
^^^^^^^^^^^^^^^^^^^^^^^^

The ``ConstraintResult`` object contains:

.. code-block:: python

   # Visibility windows - when the target IS observable
   for window in result.visibility:
       duration_hours = window.duration_seconds / 3600
       print(f"Visible: {window.start_time} to {window.end_time}")
       print(f"  Duration: {duration_hours:.2f} hours")

   # Violation windows - when the target is NOT observable
   for violation in result.violations:
       print(f"Violation: {violation.start_time} to {violation.end_time}")
       print(f"  Severity: {violation.max_severity:.2f}")
       print(f"  Reason: {violation.description}")

   # Access per-timestamp data
   # constraint_array[i] = True means constraint satisfied at timestamp[i]
   satisfied_count = result.constraint_array.sum()
   total_count = len(result.timestamp)
   print(f"Satisfied: {satisfied_count}/{total_count} timestamps")


Combining Constraints
---------------------

Real observations typically require multiple constraints. Combine them using
Python operators.

Using the AND Operator (&)
^^^^^^^^^^^^^^^^^^^^^^^^^^

All constraints must be satisfied:

.. code-block:: python

   # Target must satisfy ALL of these:
   # - At least 45° from Sun
   # - At least 10° from Moon
   # - Not in eclipse

   constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       ~EclipseConstraint()  # ~ means NOT (require NOT in eclipse)
   )

   result = constraint.evaluate(ephem, target_ra, target_dec)

   print(f"Combined visibility windows: {len(result.visibility)}")
   for window in result.visibility:
       print(f"  {window.start_time} to {window.end_time}")

Using the OR Operator (|)
^^^^^^^^^^^^^^^^^^^^^^^^^

At least one constraint must be satisfied:

.. code-block:: python

   # Observation is blocked if EITHER:
   # - Too close to Sun OR
   # - In eclipse

   blocking_constraint = (
       ~SunConstraint(min_angle=45.0) |  # Violated when < 45° from Sun
       EclipseConstraint()                # Violated when in eclipse
   )

   # Invert to get visibility (when NOT blocked)
   visibility_constraint = ~blocking_constraint

Using the NOT Operator (~)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Invert any constraint:

.. code-block:: python

   # EclipseConstraint is violated when IN eclipse
   # ~EclipseConstraint is violated when NOT in eclipse

   eclipse = EclipseConstraint()
   not_eclipse = ~eclipse

   # This is useful for requiring conditions:
   # "Require NOT in eclipse" = ~EclipseConstraint()

Using the XOR Operator (^)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Exclusive OR - violated when exactly one sub-constraint is violated:

.. code-block:: python

   # XOR example: violated when exactly one condition fails
   constraint = SunConstraint(min_angle=30.0) ^ MoonConstraint(min_angle=15.0)

   # This is less common but useful for mutually exclusive conditions


Spacecraft Observation Planning
-------------------------------

For spacecraft like Hubble or JWST, you typically need comprehensive constraints:

.. code-block:: python

   def create_spacecraft_constraint(
       sun_angle: float = 45.0,
       moon_angle: float = 10.0,
       earth_limb_angle: float = 20.0,
       avoid_eclipse: bool = True,
   ):
       """Create a typical spacecraft observation constraint.

       Args:
           sun_angle: Minimum angle from Sun (degrees)
           moon_angle: Minimum angle from Moon (degrees)
           earth_limb_angle: Minimum angle above Earth's limb (degrees)
           avoid_eclipse: If True, avoid observations during eclipse

       Returns:
           Combined constraint for spacecraft observations
       """
       constraint = (
           SunConstraint(min_angle=sun_angle) &
           MoonConstraint(min_angle=moon_angle) &
           EarthLimbConstraint(min_angle=earth_limb_angle)
       )

       if avoid_eclipse:
           constraint = constraint & ~EclipseConstraint()

       return constraint

   # Create constraint with typical values
   spacecraft_constraint = create_spacecraft_constraint(
       sun_angle=50.0,
       moon_angle=15.0,
       earth_limb_angle=25.0,
       avoid_eclipse=True
   )

   # Evaluate for target
   result = spacecraft_constraint.evaluate(ephem, target_ra, target_dec)

   # Calculate total visibility time
   total_visibility_seconds = sum(w.duration_seconds for w in result.visibility)
   total_visibility_hours = total_visibility_seconds / 3600

   print(f"Total visibility: {total_visibility_hours:.2f} hours over 24 hours")
   print(f"Visibility efficiency: {total_visibility_hours/24*100:.1f}%")


Ground-Based Observation Planning
---------------------------------

For ground observatories, constraints are similar but Earth limb becomes
horizon elevation:

.. code-block:: python

   # Create ground station ephemeris (Mauna Kea Observatory)
   ground = rust_ephem.GroundEphemeris(
       latitude=19.8207,     # degrees N
       longitude=-155.4681,  # degrees W
       height=4207,          # meters
       begin=begin,
       end=end,
       step_size=60
   )

   # Ground-based constraints
   # - Sun must be below horizon (night time) - use max_angle
   # - Target must be above horizon with margin
   # - Moon avoidance (less strict for ground)

   ground_constraint = (
       SunConstraint(min_angle=90.0) &              # Sun below horizon
       EarthLimbConstraint(min_angle=20.0) &        # Target > 20° elevation
       MoonConstraint(min_angle=5.0)                # Minimal moon avoidance
   )

   result = ground_constraint.evaluate(ground, target_ra, target_dec)

   print(f"Observable windows tonight: {len(result.visibility)}")
   for window in result.visibility:
       print(f"  {window.start_time.strftime('%H:%M')} - {window.end_time.strftime('%H:%M')} UTC")


Avoiding Bright Planets
-----------------------

Use ``BodyConstraint`` to avoid bright planets:

.. code-block:: python

   # Avoid Mars, Jupiter, and Saturn
   planet_avoidance = (
       BodyConstraint(body="Mars", min_angle=10.0) &
       BodyConstraint(body="Jupiter barycenter", min_angle=15.0) &
       BodyConstraint(body="Saturn barycenter", min_angle=15.0)
   )

   # Combine with other constraints
   full_constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       planet_avoidance &
       ~EclipseConstraint()
   )

   result = full_constraint.evaluate(ephem, target_ra, target_dec)

.. note::
   Planet availability depends on loaded SPICE kernels. The default ``de440s.bsp``
   includes planetary barycenters (use "Jupiter barycenter", "Saturn barycenter", etc.)
   but not individual planet centers.


Working with Multiple Targets
-----------------------------

For survey observations or target selection, evaluate many targets efficiently:

.. code-block:: python

   import numpy as np

   # Define a catalog of targets
   targets = [
       {"name": "Crab Nebula", "ra": 83.63, "dec": 22.01},
       {"name": "Orion Nebula", "ra": 83.82, "dec": -5.39},
       {"name": "Andromeda", "ra": 10.68, "dec": 41.27},
       {"name": "Whirlpool", "ra": 202.47, "dec": 47.20},
       {"name": "Sombrero", "ra": 189.99, "dec": -11.62},
   ]

   # Extract coordinates
   target_ras = [t["ra"] for t in targets]
   target_decs = [t["dec"] for t in targets]

   # Create constraint
   constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       ~EclipseConstraint()
   )

   # Batch evaluation (much faster than loop)
   violations = constraint.in_constraint_batch(ephem, target_ras, target_decs)
   # violations.shape = (n_targets, n_times)

   # Analyze results
   for i, target in enumerate(targets):
       # Count satisfied timestamps
       satisfied = ~violations[i]  # Invert: True = satisfied
       visibility_fraction = satisfied.sum() / len(satisfied)

       print(f"{target['name']}: {visibility_fraction*100:.1f}% observable")

Large Target Catalogs
^^^^^^^^^^^^^^^^^^^^^

For thousands of targets, use vectorized operations:

.. code-block:: python

   # Generate 10,000 random sky positions
   np.random.seed(42)
   n_targets = 10000
   target_ras = np.random.uniform(0, 360, n_targets)
   target_decs = np.random.uniform(-90, 90, n_targets)

   # Single batch evaluation (3-50x faster than loop)
   violations = constraint.in_constraint_batch(ephem, target_ras, target_decs)

   # Find best targets (least violations)
   violation_counts = violations.sum(axis=1)
   best_indices = np.argsort(violation_counts)[:10]

   print("Top 10 most observable targets:")
   for idx in best_indices:
       print(f"  RA={target_ras[idx]:.1f}°, Dec={target_decs[idx]:.1f}°: "
             f"{violation_counts[idx]} violations")


Finding Optimal Observation Times
---------------------------------

Analyze when a target is most observable:

.. code-block:: python

   result = constraint.evaluate(ephem, target_ra, target_dec)

   # Find longest visibility window
   if result.visibility:
       longest = max(result.visibility, key=lambda w: w.duration_seconds)
       print(f"Longest visibility window:")
       print(f"  Start: {longest.start_time}")
       print(f"  End: {longest.end_time}")
       print(f"  Duration: {longest.duration_seconds/60:.1f} minutes")

   # Find optimal time (middle of longest window)
   if result.visibility:
       from datetime import timedelta
       mid_duration = timedelta(seconds=longest.duration_seconds / 2)
       optimal_time = longest.start_time + mid_duration
       print(f"  Optimal observation time: {optimal_time}")

Visibility Statistics
^^^^^^^^^^^^^^^^^^^^^

Calculate detailed statistics:

.. code-block:: python

   def analyze_visibility(result):
       """Analyze visibility windows and return statistics."""
       if not result.visibility:
           return {
               "total_hours": 0,
               "window_count": 0,
               "avg_window_minutes": 0,
               "max_window_minutes": 0,
               "min_window_minutes": 0,
           }

       durations = [w.duration_seconds for w in result.visibility]

       return {
           "total_hours": sum(durations) / 3600,
           "window_count": len(durations),
           "avg_window_minutes": (sum(durations) / len(durations)) / 60,
           "max_window_minutes": max(durations) / 60,
           "min_window_minutes": min(durations) / 60,
       }

   stats = analyze_visibility(result)

   print(f"Visibility Statistics:")
   print(f"  Total observable time: {stats['total_hours']:.2f} hours")
   print(f"  Number of windows: {stats['window_count']}")
   print(f"  Average window: {stats['avg_window_minutes']:.1f} minutes")
   print(f"  Longest window: {stats['max_window_minutes']:.1f} minutes")
   print(f"  Shortest window: {stats['min_window_minutes']:.1f} minutes")


Saving and Loading Constraints
------------------------------

Constraints can be serialized to JSON for storage or configuration:

.. code-block:: python

   # Create a complex constraint
   constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       EarthLimbConstraint(min_angle=25.0) &
       ~EclipseConstraint()
   )

   # Serialize to JSON
   json_str = constraint.model_dump_json(indent=2)
   print(json_str)

   # Save to file
   with open("observation_constraint.json", "w") as f:
       f.write(json_str)

   # Load from file
   with open("observation_constraint.json", "r") as f:
       json_str = f.read()

   # Reconstruct constraint
   from rust_ephem.constraints import CombinedConstraintConfig
   loaded_constraint = CombinedConstraintConfig.validate_json(json_str)

   # Or use the Rust backend directly
   rust_constraint = rust_ephem.Constraint.from_json(json_str)


Constraint Templates
--------------------

Define reusable constraint templates for different observation modes:

.. code-block:: python

   from rust_ephem.constraints import (
       SunConstraint, MoonConstraint, EarthLimbConstraint,
       EclipseConstraint, BodyConstraint
   )

   # Template for UV observations (strict Sun/Moon avoidance)
   UV_OBSERVATION = (
       SunConstraint(min_angle=70.0) &      # Very strict Sun avoidance
       MoonConstraint(min_angle=30.0) &     # Strict Moon avoidance
       EarthLimbConstraint(min_angle=20.0) &
       ~EclipseConstraint()
   )

   # Template for infrared observations (less strict)
   IR_OBSERVATION = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       EarthLimbConstraint(min_angle=15.0)
       # Eclipse OK for IR
   )

   # Template for radio observations (minimal constraints)
   RADIO_OBSERVATION = (
       SunConstraint(min_angle=20.0) &
       EarthLimbConstraint(min_angle=10.0)
   )

   # Use templates
   uv_result = UV_OBSERVATION.evaluate(ephem, target_ra, target_dec)
   ir_result = IR_OBSERVATION.evaluate(ephem, target_ra, target_dec)

   print(f"UV observation windows: {len(uv_result.visibility)}")
   print(f"IR observation windows: {len(ir_result.visibility)}")


Real-World Example: Multi-Day Survey
------------------------------------

Plan observations over multiple days:

.. code-block:: python

   from datetime import timedelta

   # Create 7-day ephemeris
   begin = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
   end = begin + timedelta(days=7)

   ephem = rust_ephem.TLEEphemeris(
       norad_id=25544,
       begin=begin,
       end=end,
       step_size=300  # 5-minute steps for week-long analysis
   )

   # Survey targets
   survey_targets = [
       {"name": "NGC 1275", "ra": 49.95, "dec": 41.51},
       {"name": "NGC 4151", "ra": 182.64, "dec": 39.41},
       {"name": "NGC 5548", "ra": 214.50, "dec": 25.14},
       {"name": "Mrk 421", "ra": 166.11, "dec": 38.21},
       {"name": "Mrk 501", "ra": 253.47, "dec": 39.76},
   ]

   constraint = (
       SunConstraint(min_angle=45.0) &
       MoonConstraint(min_angle=10.0) &
       EarthLimbConstraint(min_angle=20.0) &
       ~EclipseConstraint()
   )

   # Analyze each target
   print("7-Day Survey Visibility Analysis")
   print("=" * 50)

   for target in survey_targets:
       result = constraint.evaluate(ephem, target["ra"], target["dec"])

       total_hours = sum(w.duration_seconds for w in result.visibility) / 3600
       avg_window = (total_hours * 60 / len(result.visibility)) if result.visibility else 0

       print(f"\n{target['name']}:")
       print(f"  Total visibility: {total_hours:.1f} hours")
       print(f"  Windows: {len(result.visibility)}")
       print(f"  Avg window: {avg_window:.1f} minutes")

       # Show best windows (longest 3)
       if result.visibility:
           best_windows = sorted(result.visibility,
                                key=lambda w: w.duration_seconds,
                                reverse=True)[:3]
           print("  Best windows:")
           for w in best_windows:
               print(f"    {w.start_time.strftime('%Y-%m-%d %H:%M')} "
                     f"({w.duration_seconds/60:.0f} min)")


Moving Target Visibility
------------------------

For targets that move across the sky (asteroids, comets, spacecraft), use the
``Constraint.evaluate_moving_body()`` method with JPL Horizons support:

Basic Moving Target Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Track any solar system body by name or NAIF ID:

.. code-block:: python

    # Define constraint
    constraint = SunConstraint(min_angle=30) & MoonConstraint(min_angle=15)

    # Track Mars (using SPICE kernel)
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="Mars"
    )

    print(f"Mars visibility windows: {len(result.visibility)}")
    for window in result.visibility:
        print(f"  {window.start_time} to {window.end_time}")

Using JPL Horizons for Extended Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable JPL Horizons to access asteroids, comets, and spacecraft beyond SPICE kernels:

.. code-block:: python

    # Track Ceres asteroid
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="1",  # Ceres NAIF ID
        use_horizons=True  # ← Enable Horizons fallback
    )

    # Track Apophis asteroid
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="99942",  # Apophis
        use_horizons=True
    )

    # Track by name (also works with Horizons)
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="Apophis",
        use_horizons=True
    )

**How Horizons Integration Works:**

1. **SPICE first** — Checks local SPICE kernels for the body
2. **Horizons fallback** — If not in SPICE and ``use_horizons=True``, queries NASA's JPL Horizons
3. **Frame conversion** — Horizons returns heliocentric coordinates; automatically converted to observer-relative
4. **Constraint evaluation** — Body position evaluated against all constraints over the time range

Custom RA/Dec Arrays
^^^^^^^^^^^^^^^^^^^^^

For custom target paths (non-solar system objects), provide explicit RA/Dec arrays:

.. code-block:: python

    import numpy as np

    # Define moving target path
    times = ephem.timestamp
    ras = np.linspace(100, 110, len(times))      # RA moves from 100° to 110°
    decs = np.linspace(10, 20, len(times))       # Dec moves from 10° to 20°

    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        target_ras=list(ras),
        target_decs=list(decs),
    )

    print(f"Visibility windows: {len(result.visibility)}")

Asteroid Close Approach Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Monitor visibility during an asteroid's close approach:

.. code-block:: python

    from datetime import datetime, timedelta, timezone

    # Apophis close approach period (April 2029)
    begin = datetime(2029, 4, 1, tzinfo=timezone.utc)
    end = datetime(2029, 4, 14, tzinfo=timezone.utc)

    # Ground observation site
    obs = rust_ephem.GroundEphemeris(
        latitude=40.0,    # Your observatory
        longitude=-105.0,
        height=1600,
        begin=begin,
        end=end,
        step_size=3600  # Hourly resolution
    )

    # Stringent constraints for close approach observation
    constraint = (
        SunConstraint(min_angle=10) &  # Not too close to Sun
        MoonConstraint(min_angle=20) &  # Away from Moon
        EarthLimbConstraint(min_angle=5)  # Clear of Earth limb
    )

    result = constraint.evaluate_moving_body(
        ephemeris=obs,
        body="99942",  # Apophis
        use_horizons=True
    )

    print(f"Apophis observable for {len(result.visibility)} period(s)")
    for window in result.visibility:
        hours = window.duration_seconds / 3600
        print(f"  {window.start_time} → {window.end_time} ({hours:.1f} hours)")

Performance Tips
----------------

1. **Use batch evaluation** for multiple targets:

   .. code-block:: python

      # FAST: Single batch call
      violations = constraint.in_constraint_batch(ephem, ras, decs)

      # SLOW: Loop over targets
      for ra, dec in zip(ras, decs):
          result = constraint.evaluate(ephem, ra, dec)  # Avoid!

2. **Choose appropriate step size**:

   .. code-block:: python

      # Coarse: 5 minutes (good for week-long planning)
      ephem = rust_ephem.TLEEphemeris(..., step_size=300)

      # Fine: 1 minute (good for precise scheduling)
      ephem = rust_ephem.TLEEphemeris(..., step_size=60)

      # Very fine: 10 seconds (only if needed)
      ephem = rust_ephem.TLEEphemeris(..., step_size=10)

3. **Use cached properties** when iterating:

   .. code-block:: python

      # FAST: Access cached arrays
      for i, (time, satisfied) in enumerate(zip(result.timestamp, result.constraint_array)):
          if satisfied:
              # Process visible time
              pass

      # SLOW: Call method repeatedly
      for time in result.timestamp:
          if result.in_constraint(time):  # Lookups on each call
              pass

4. **Evaluate specific times** if you only need certain times:

   .. code-block:: python

      # Only evaluate specific indices
      result = constraint.evaluate(ephem, ra, dec, indices=[0, 100, 200])

      # Only evaluate specific times
      specific_times = [datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)]
      result = constraint.evaluate(ephem, ra, dec, times=specific_times)
