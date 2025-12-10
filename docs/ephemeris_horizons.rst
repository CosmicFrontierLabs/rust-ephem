JPL Horizons Integration
=========================

``rust-ephem`` includes support for NASA's JPL Horizons system, providing access to
ephemerides for solar system bodies not available in SPICE kernels. This is particularly
useful for querying asteroids, comets, spacecraft, and other minor bodies.

Overview
--------

The JPL Horizons system is NASA's comprehensive solar system ephemeris service. It provides
high-accuracy position and velocity data for:

- **Planets and moons** (when not in SPICE kernels)
- **Asteroids** (including named and numbered asteroids)
- **Comets** (periodic and non-periodic)
- **Spacecraft** (natural and artificial satellites, space probes)
- **Interplanetary objects** (Voyager, New Horizons, etc.)

When you set ``use_horizons=True`` in ``get_body()`` or ``Constraint.evaluate_moving_body()``,
``rust-ephem`` automatically falls back to JPL Horizons if the requested body is not
found in your local SPICE kernels. This enables seamless querying of a much broader
range of bodies without requiring large kernel files or pre-configuration.

Setup
-----

No additional setup is required beyond installing ``rust-ephem``. The Horizons feature
is built-in and uses NASA's public HTTP API.

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    # Load default planetary ephemeris (still useful for Sun/Moon/planets)
    re.ensure_planetary_ephemeris()

    # Create any ephemeris type
    begin = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 2, tzinfo=timezone.utc)
    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)

Basic Usage
-----------

Enable Horizons queries by setting ``use_horizons=True``:

.. code-block:: python

    # Query asteroid Ceres
    ceres = ephem.get_body("1", use_horizons=True)
    print(f"Ceres RA: {ceres[0].ra}, Dec: {ceres[0].dec}")

    # Query by name (case-insensitive)
    ceres = ephem.get_body("Ceres", use_horizons=True)

    # Query position/velocity data
    ceres_pv = ephem.get_body_pv("1", use_horizons=True)
    print(f"Ceres distance: {ceres_pv.position[0]}")  # km
    print(f"Ceres velocity: {ceres_pv.velocity[0]}")  # km/s

Fallback Behavior
~~~~~~~~~~~~~~~~~

When ``use_horizons=True``, the lookup process is:

1. **Check SPICE kernels first** — If the body is found in your kernel, use it
2. **Fall back to Horizons** — If not found in SPICE, query JPL Horizons
3. **Raise error if not found** — If neither source has the body, raise an exception

This approach gives you the best of both worlds: fast, cached SPICE lookups for
frequently-used bodies, with automatic fallback to Horizons for less common objects.

.. code-block:: python

    # These are equivalent when Mars is in SPICE kernel:
    mars_spice = ephem.get_body("Mars")           # Uses SPICE (no network)
    mars_either = ephem.get_body("Mars", use_horizons=True)  # Prefers SPICE

    # But Horizons is required for lesser-known bodies:
    apophis = ephem.get_body("99942", use_horizons=True)  # Asteroid 99942 (Apophis)
    # This will fail if use_horizons=False and Apophis isn't in SPICE kernel


Body Identifiers
----------------

JPL Horizons accepts many types of body identifiers:

Common Objects (NAIF IDs)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Name
     - NAIF ID
     - Notes
   * - Sun
     - 10
     - Solar center
   * - Moon
     - 301
     - Earth's moon
   * - Mercury
     - 199
     - Planet center
   * - Venus
     - 299
     - Planet center
   * - Earth
     - 399
     - Planet center
   * - Mars
     - 499
     - Planet center
   * - Jupiter
     - 599
     - Planet center (5 = barycenter)
   * - Saturn
     - 699
     - Planet center (6 = barycenter)
   * - Uranus
     - 799
     - Planet center (7 = barycenter)
   * - Neptune
     - 899
     - Planet center (8 = barycenter)

Asteroids
~~~~~~~~~

Asteroids can be referenced by:

- **NAIF ID** (integer): ``ephem.get_body("1", use_horizons=True)`` → Ceres
- **Minor planet number** (integer): ``ephem.get_body("433", use_horizons=True)`` → Eros
- **Name** (string): ``ephem.get_body("Ceres", use_horizons=True)``

.. code-block:: python

    # Common asteroids
    ceres = ephem.get_body("1", use_horizons=True)        # Ceres (dwarf planet)
    vesta = ephem.get_body("4", use_horizons=True)        # Vesta
    juno = ephem.get_body("3", use_horizons=True)         # Juno
    eros = ephem.get_body("433", use_horizons=True)       # Eros
    apophis = ephem.get_body("99942", use_horizons=True)  # Apophis
    bennu = ephem.get_body("101955", use_horizons=True)   # Bennu

Comets
~~~~~~

Comets are referenced by name:

.. code-block:: python

    # Some well-known comets
    halley = ephem.get_body("Halley", use_horizons=True)
    neowise = ephem.get_body("C/2020 F3", use_horizons=True)  # NEOWISE
    leone = ephem.get_body("67P", use_horizons=True)  # Churyumov-Gerasimenko (short form)

Spacecraft
~~~~~~~~~~

Many space probes and satellites are available:

.. code-block:: python

    # Natural and artificial objects
    voyager1 = ephem.get_body("-31", use_horizons=True)   # Voyager 1
    voyager2 = ephem.get_body("-32", use_horizons=True)   # Voyager 2
    newhorizons = ephem.get_body("-98", use_horizons=True)  # New Horizons probe
    parker = ephem.get_body("-96", use_horizons=True)      # Parker Solar Probe
    juno = ephem.get_body("-61", use_horizons=True)        # Juno orbiter

**Note:** Spacecraft are referenced by negative NAIF IDs. Consult JPL's list of spacecraft IDs
at `Horizons System <https://ssd.jpl.nasa.gov/horizons/>`_ for a comprehensive list.

Working with Constraints
------------------------

Horizons integration is fully supported in the constraint system, enabling
observation planning for any Horizons-accessible body:

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, MoonConstraint

    # Set up ephemeris
    begin = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 2, tzinfo=timezone.utc)
    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)

    # Define constraint: body must be 45° from Sun AND 10° from Moon
    constraint = SunConstraint(min_angle=45) & MoonConstraint(min_angle=10)

    # Get visibility for Ceres
    visibility = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body="1",  # Ceres
        use_horizons=True  # ← Enable Horizons fallback
    )

    # Print visibility windows
    for window in visibility.visibility:
        print(f"Visible: {window.start_time} to {window.end_time}")

    # Check satisfaction statistics
    print(f"Total satisfied: {visibility.all_satisfied}")
    print(f"Per-sample satisfied: {visibility.constraint_array[:5]}")

Advanced Examples
-----------------

Asteroid Visibility During Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track an asteroid approaching Earth using Horizons:

.. code-block:: python

    import numpy as np
    from datetime import datetime, timedelta, timezone

    # Define extended time range for close approach event
    begin = datetime(2029, 4, 1, tzinfo=timezone.utc)
    end = datetime(2029, 4, 14, tzinfo=timezone.utc)

    # Ground observatory (e.g., Arecibo)
    obs = re.GroundEphemeris(
        latitude=18.3461,
        longitude=-66.7527,
        height=496,
        begin=begin,
        end=end,
        step_size=3600  # Hourly steps
    )

    # Define constraints for Apophis observation
    constraint = SunConstraint(min_angle=10) & MoonConstraint(min_angle=20)

    # Query Apophis during approach
    result = constraint.evaluate_moving_body(
        ephemeris=obs,
        body="99942",  # Apophis
        use_horizons=True
    )

    print(f"Apophis observable for {len(result.visibility)} window(s)")
    for window in result.visibility:
        print(f"  {window.start_time} to {window.end_time}")

Comparing Asteroid Positions Across Observers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare how an asteroid's position changes from different ground stations:

.. code-block:: python

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    begin = datetime(2024, 9, 15, tzinfo=timezone.utc)
    end = datetime(2024, 9, 16, tzinfo=timezone.utc)

    # Two observatories
    keck = re.GroundEphemeris(
        latitude=19.8267, longitude=-155.4730, height=4207,
        begin=begin, end=end, step_size=60
    )
    vlt = re.GroundEphemeris(
        latitude=-24.6276, longitude=-70.4035, height=2635,
        begin=begin, end=end, step_size=60
    )

    # Get Ceres from each location
    ceres_keck = keck.get_body("1", use_horizons=True)
    ceres_vlt = vlt.get_body("1", use_horizons=True)

    # Calculate parallax effect
    sep = ceres_keck.separation(ceres_vlt)
    print(f"Maximum parallax: {sep[0].max():.4f} degrees")

Tracking Multiple Asteroids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor visibility for a set of potentially hazardous asteroids:

.. code-block:: python

    # PHAs (Potentially Hazardous Asteroids)
    phas = {
        "433": "Eros",
        "1862": "Apollo",
        "2062": "Aten",
        "3122": "Florence",
        "99942": "Apophis",
    }

    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)
    constraint = SunConstraint(min_angle=30)

    for naif_id, name in phas.items():
        result = constraint.evaluate_moving_body(
            ephemeris=ephem,
            body=naif_id,
            use_horizons=True
        )

        window_count = len(result.visibility)
        print(f"{name:15s} ({naif_id:5s}): {window_count} visibility window(s)")

Performance Considerations
--------------------------

Network Requirements
~~~~~~~~~~~~~~~~~~~~

Horizons queries require an active internet connection. Each query makes an HTTP
request to NASA's servers. Queries are typically fast (< 1 second), but:

- **Network latency** affects query time
- **Large time ranges** may take longer to compute
- **Query caching** is not implemented (each call hits the network)

For repeated queries of the same body and time range, consider caching the results:

.. code-block:: python

    # Cache a lookup
    ceres_pv = ephem.get_body_pv("1", use_horizons=True)

    # Reuse the cached result multiple times
    sun_dist = np.linalg.norm(ceres_pv.position[0])
    print(f"Distance: {sun_dist:.0f} km")

Time Range Limitations
~~~~~~~~~~~~~~~~~~~~~~

Horizons has limitations on how far into the past and future it can compute:

- **Well-established bodies** (planets, Moon, major asteroids): ±thousands of years
- **Recently discovered objects** (comets, new asteroids): Much shorter ranges
- **Spacecraft**: Limited by mission duration and tracking data

If you query beyond the supported range, Horizons will raise an error. Start with
smaller time ranges and expand if successful.

Accuracy
~~~~~~~~

Horizons positions are typically accurate to within a few kilometers for solar system
bodies, with uncertainty increasing for:

- Objects far in the past or future
- Recently discovered bodies with fewer observations
- Comets with uncertain orbital parameters

For mission-critical applications, compare Horizons results with other sources
or use higher-order accuracy models.

Troubleshooting
---------------

Body Not Found
~~~~~~~~~~~~~~

If you get a "body not found" error, verify:

1. **Check the body identifier** — Use JPL's `Horizons browser <https://ssd.jpl.nasa.gov/horizons/>`_
   to find the correct NAIF ID or name
2. **Check time range** — The body may not be computable during your time range
3. **Network connectivity** — Ensure your system has internet access
4. **Horizons service** — NASA's servers may occasionally be unavailable

.. code-block:: python

    try:
        body = ephem.get_body("99999999", use_horizons=True)
    except Exception as e:
        print(f"Query failed: {e}")

Slow Queries
~~~~~~~~~~~~

If Horizons queries are slow:

1. **Network latency** — Check your internet connection speed
2. **Large time range** — Reduce the step size or query shorter periods
3. **Server load** — Horizons may be experiencing high traffic; retry later

For production applications querying many bodies, consider batching queries or
using periodic pre-computation to avoid real-time network dependencies.

Type Stub Support
-----------------

Full type hints are provided for Horizons methods:

.. code-block:: python

    from typing import Optional
    from rust_ephem import Ephemeris

    def track_asteroid(
        ephem: Ephemeris,
        asteroid_id: str,
        use_horizons: bool = True
    ) -> None:
        """Track an asteroid using SPICE or Horizons."""
        body = ephem.get_body(asteroid_id, use_horizons=use_horizons)
        print(f"Position: {body[0]}")

The ``.pyi`` stub files include ``use_horizons`` parameter documentation for IDE
autocomplete and static type checkers (mypy, pyright, etc.).

Integration with Constraint System
-----------------------------------

Horizons support is seamlessly integrated into all constraint types:

.. code-block:: python

    from rust_ephem.constraints import (
        AirmassConstraint,
        EarthLimbConstraint,
        SunConstraint,
        MoonConstraint,
        MoonPhaseConstraint,
    )

    constraint = (
        SunConstraint(min_angle=30) &
        AirmassConstraint(max_airmass=2.0) &
        EarthLimbConstraint(min_angle=20)
    )

    # Works with any Horizons-accessible body
    result = constraint.evaluate_moving_body(
        ephemeris=ephem,
        body=\"2\",  # Pallas asteroid
        use_horizons=True
    )

Limitations and Caveats
-----------------------

1. **Network Required** — Unlike SPICE kernel queries, Horizons lookups require
   internet connectivity

2. **No Caching** — Results are not cached; repeated queries recompute
   Consider implementing application-level caching if needed

3. **Time Range Constraints** — Some bodies (especially recently discovered ones)
   have limited computable time ranges

4. **Accuracy Varies** — Position accuracy depends on observational data for the body
   Check Horizons documentation for specific body accuracy estimates

5. **API Changes** — JPL Horizons is maintained by NASA; future API changes could
   affect compatibility (current API assumed stable)

6. **Spacecraft Tracking** — Spacecraft positions become unavailable once missions end
   or tracking ceases; consult NASA's list of active tracking objects

Reference
---------

**JPL Horizons System**
  - Main site: https://ssd.jpl.nasa.gov/horizons/
  - Web interface: https://ssd.jpl.nasa.gov/horizons/basic.html
  - NAIF IDs: https://ssd.jpl.nasa.gov/?horizons
  - Body list: Search the Horizons database for any object

**rustephem Implementation**
  - Horizons module: ``src/utils/horizons.rs``
  - Rhorizons crate: https://crates.io/crates/rhorizons (async NASA JPL Horizons client)
  - Integration: ``src/utils/celestial.rs`` (``calculate_body_by_id_or_name`` function)

**Related Documentation**
  - :doc:`ephemeris_get_body` — Basic body lookups
  - :doc:`planning_constraints` — Constraint system overview
  - :doc:`planning_visibility` — Visibility calculations
