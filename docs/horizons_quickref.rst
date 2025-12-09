.. _horizons-quickstart:

JPL Horizons Quick Reference
=============================

5-Minute Quick Start
---------------------

Enable Horizons for body queries:

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    # Set up ephemeris
    begin = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 2, tzinfo=timezone.utc)
    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)

    # Query asteroid (requires Horizons)
    ceres = ephem.get_body("1", use_horizons=True)
    print(f"Ceres: {ceres[0].ra}, {ceres[0].dec}")

For visibility windows with moving bodies:

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, moving_body_visibility

    constraint = SunConstraint(min_angle=45)
    result = moving_body_visibility(
        constraint=constraint,
        ephemeris=ephem,
        body="99942",  # Apophis
        use_horizons=True
    )

    for window in result.visibility:
        print(f"{window.start_time} → {window.end_time}")

Common Body IDs
---------------

**Planets (NAIF IDs)**

::

    1-9      Planet barycenters (Mercury to Pluto)
    199/299/399/499/599/699/799/899  Planet centers
    301      Moon
    10       Sun
    399      Earth center

**Common Asteroids (NAIF IDs)**

::

    1        Ceres
    2        Pallas
    3        Juno
    4        Vesta
    433      Eros
    1862     Apollo
    2062     Aten
    3122     Florence
    99942    Apophis (2004 MN4)
    101955   Bennu

**Spacecraft (Negative NAIF IDs)**

::

    -31      Voyager 1
    -32      Voyager 2
    -96      Parker Solar Probe
    -98      New Horizons
    -61      Juno orbiter

**Querying Asteroids by Name (via Horizons)**

To use comet or asteroid names with Horizons, you can pass the body name directly
when ``use_horizons=True``. Note: For comets, you may need to use the NAIF ID for
best results. To find NAIF IDs for specific bodies, use the JPL Horizons browser:

https://ssd.jpl.nasa.gov/horizons/

Supported formats:

::

    "1"                NAIF ID (numeric string)
    "Ceres"            Asteroid name
    "C/2020 F3"        Comet designation
    "67P"              Comet common name (Churyumov-Gerasimenko)

Key Parameters
--------------

``get_body(body, kernel_spec=None, use_horizons=False)``

- ``body`` — Name or NAIF ID as string
- ``kernel_spec`` — Path/URL to custom SPICE kernel (optional)
- ``use_horizons`` — Enable Horizons fallback [default: False]

``get_body_pv(body, kernel_spec=None, use_horizons=False)``

- Same as ``get_body`` but returns position/velocity data instead of SkyCoord

``moving_body_visibility(constraint, ephemeris, body=None, ras=None, decs=None, timestamps=None, use_horizons=False)``

- ``constraint`` — Constraint instance or combination
- ``ephemeris`` — Ephemeris object
- ``body`` — Body name/ID for Horizons lookup
- ``ras/decs/timestamps`` — Custom coordinate arrays
- ``use_horizons`` — Enable Horizons fallback

Common Patterns
---------------

**Single Body Lookup**

.. code-block:: python

    asteroid = ephem.get_body("99942", use_horizons=True)

**Multiple Bodies**

.. code-block:: python

    asteroids = {}
    for body_id in ["1", "4", "2", "99942"]:
        asteroids[body_id] = ephem.get_body(
            body_id, use_horizons=True
        )

**With Constraints**

.. code-block:: python

    from rust_ephem.constraints import (
        SunConstraint,
        MoonConstraint,
        moving_body_visibility
    )

    constraint = SunConstraint(min_angle=30) & MoonConstraint(min_angle=10)
    result = moving_body_visibility(
        constraint=constraint,
        ephemeris=ephem,
        body="Apophis",
        use_horizons=True
    )

**Error Handling**

.. code-block:: python

    try:
        body = ephem.get_body("99999999", use_horizons=True)
    except Exception as e:
        print(f"Failed to query: {e}")

Performance Tips
----------------

1. **Cache results** — Don't query the same body twice
2. **Use SPICE for major bodies** — Faster than Horizons
3. **Set appropriate time ranges** — Shorter ranges are faster
4. **Batch visibility with ``moving_body_visibility()``** — More efficient than individual queries

Troubleshooting
---------------

**"Body not found" error**

- Check the body identifier on `JPL Horizons <https://ssd.jpl.nasa.gov/horizons/>`_
- Try a shorter time range
- Verify internet connection

**Slow queries**

- Horizons queries are network-bound (typically 0.5-2 seconds)
- Cache results to avoid repeated queries
- Use smaller step sizes for faster computations

**"No times provided" error**

- Ensure ephemeris has at least one timestamp
- Check that begin < end

Documentation Links
-------------------

- **Full Guide** → :doc:`ephemeris_horizons`
- **FAQ** → :doc:`horizons_faq`
- **Body Lookups** → :doc:`ephemeris_get_body`
- **Constraints** → :doc:`planning_constraints`
- **Visibility** → :doc:`planning_visibility`
- **JPL Horizons** → https://ssd.jpl.nasa.gov/horizons/
- **NAIF ID List** → https://ssd.jpl.nasa.gov/?horizons

Type Hints
----------

All Horizons parameters have full type hint support:

.. code-block:: python

    from rust_ephem import Ephemeris
    from rust_ephem.constraints import ConstraintResult

    def track_asteroid(
        ephem: Ephemeris,
        body_id: str,
        use_horizons: bool = True
    ) -> ConstraintResult:
        """Track an asteroid with Horizons support."""
        from rust_ephem.constraints import SunConstraint, moving_body_visibility

        constraint = SunConstraint(min_angle=30)
        return moving_body_visibility(
            constraint=constraint,
            ephemeris=ephem,
            body=body_id,
            use_horizons=use_horizons
        )
