.. _horizons-cheatsheet:

JPL Horizons Cheat Sheet
=========================

Common Tasks
------------

**Query an asteroid by NAIF ID**

.. code-block:: python

    asteroid = ephem.get_body("1", use_horizons=True)  # Ceres

**Query by asteroid name**

.. code-block:: python

    apophis = ephem.get_body("Apophis", use_horizons=True)

**Get position and velocity**

.. code-block:: python

    pv = ephem.get_body_pv("99942", use_horizons=True)
    position = pv.position  # km
    velocity = pv.velocity  # km/s

**Track with visibility constraints**

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, moving_body_visibility

    constraint = SunConstraint(min_angle=30)
    result = moving_body_visibility(
        constraint=constraint,
        ephemeris=ephem,
        body="Apophis",
        use_horizons=True
    )

**Find visibility windows**

.. code-block:: python

    for window in result.visibility:
        print(f"{window.start_time} to {window.end_time}")

Common Body IDs Quick Reference
-------------------------------

**Asteroids**

::

    "1"        Ceres
    "4"        Vesta
    "433"      Eros
    "99942"    Apophis
    "101955"   Bennu

**Planets (Centers)**

::

    "199"      Mercury
    "299"      Venus
    "399"      Earth
    "499"      Mars
    "599"      Jupiter
    "699"      Saturn
    "799"      Uranus
    "899"      Neptune

**Special Bodies**

::

    "10"       Sun
    "301"      Moon

**Spacecraft**

::

    "-31"      Voyager 1
    "-32"      Voyager 2
    "-98"      New Horizons
    "-96"      Parker Solar Probe

**Comets**

::

    "Halley"
    "C/2020 F3"
    "67P"

Common Patterns
---------------

**Error-safe query**

.. code-block:: python

    try:
        body = ephem.get_body("999999", use_horizons=True)
    except Exception as e:
        print(f"Failed: {e}")

**Cache results**

.. code-block:: python

    _cache = {}
    def get_cached(body_id):
        if body_id not in _cache:
            _cache[body_id] = ephem.get_body(body_id, use_horizons=True)
        return _cache[body_id]

**Multiple asteroids**

.. code-block:: python

    for body_id in ["1", "4", "99942"]:
        result = ephem.get_body(body_id, use_horizons=True)
        print(result)

**Constraint combinations**

.. code-block:: python

    from rust_ephem.constraints import SunConstraint, MoonConstraint

    constraint = (
        SunConstraint(min_angle=30) &
        MoonConstraint(min_angle=15)
    )

Parameter Reference
-------------------

**get_body(body, kernel_spec=None, use_horizons=False)**

- ``body`` (str) — Name or ID
- ``kernel_spec`` (str, optional) — SPICE kernel path
- ``use_horizons`` (bool) — Enable fallback [default: False]
- **Returns**: SkyCoord

**get_body_pv(body, kernel_spec=None, use_horizons=False)**

- Same as ``get_body`` but returns position/velocity
- **Returns**: PositionVelocityData

**moving_body_visibility(constraint, ephemeris, body=None, ras=None, decs=None, timestamps=None, use_horizons=False)**

- ``constraint`` — Constraint instance
- ``ephemeris`` — Ephemeris object
- ``body`` (str, optional) — Body ID/name
- ``ras/decs/timestamps`` (array, optional) — Custom coordinates
- ``use_horizons`` (bool) — Enable fallback [default: False]
- **Returns**: ConstraintResult

Key Points
----------

✓ **Enable with** ``use_horizons=True``

✓ **Try SPICE first** (faster, no network)

✓ **Falls back to Horizons** automatically

✓ **Works with constraints** (moving_body_visibility)

✓ **Network required** for Horizons queries

✓ **~0.5-2 seconds** per query (network-bound)

✗ **No automatic caching** (implement your own if needed)

✗ **Not for high-frequency** repeated queries (use SPICE)

Troubleshooting
---------------

**"Body not found"**

::

    Solution: Check JPL Horizons browser
    https://ssd.jpl.nasa.gov/horizons/

**Slow queries**

::

    Solution: Queries depend on network speed
    Use smaller time ranges
    Cache results if repeated

**Network error**

::

    Solution: Check internet connection
    Verify JPL Horizons service availability

**Out of time range**

::

    Solution: Try shorter time period
    Some bodies have limited computability ranges

Common Gotchas
--------------

**❌ Forgetting to set use_horizons=True**

.. code-block:: python

    # This fails if not in SPICE:
    body = ephem.get_body("Apophis")

.. code-block:: python

    # ✓ This works:
    body = ephem.get_body("Apophis", use_horizons=True)

**❌ Querying same body repeatedly**

.. code-block:: python

    # Slow: hits network 10 times
    for i in range(10):
        body = ephem.get_body("1", use_horizons=True)

.. code-block:: python

    # ✓ Fast: hits network once
    body = ephem.get_body("1", use_horizons=True)
    for i in range(10):
        use_result(body)  # Reuse cached result

**❌ Using wrong NAIF ID**

.. code-block:: python

    # Wrong (Mars center, not available in DE440S):
    mars = ephem.get_body("499")  # Fails without use_horizons

.. code-block:: python

    # ✓ Correct (Mars or barycenter):
    mars = ephem.get_body("4", use_horizons=True)  # Works
    mars = ephem.get_body("Mars", use_horizons=True)  # Also works

Body ID Formats
---------------

**Integer string**

::

    ephem.get_body("1")  ✓
    ephem.get_body(1)    ✗ (must be string)

**Name (case-insensitive)**

::

    ephem.get_body("Ceres")    ✓
    ephem.get_body("ceres")    ✓
    ephem.get_body("CERES")    ✓

**Negative IDs (spacecraft)**

::

    ephem.get_body("-31")  # Voyager 1 ✓

**Comet designations**

::

    ephem.get_body("Halley")         ✓
    ephem.get_body("C/2020 F3")      ✓
    ephem.get_body("67P")            ✓

Type Hints
----------

.. code-block:: python

    from typing import Optional
    from rust_ephem import Ephemeris
    from astropy.coordinates import SkyCoord

    def get_asteroid(
        ephem: Ephemeris,
        body_id: str,
    ) -> Optional[SkyCoord]:
        """Get asteroid position with Horizons fallback."""
        try:
            return ephem.get_body(body_id, use_horizons=True)
        except Exception:
            return None

Performance Baseline
--------------------

**SPICE query**: < 1 ms (cached)

**Horizons query**: 0.5-2 seconds

**Frame conversion**: 1-10 ms

**Constraint evaluation**: 1-100 ms

**Batch evaluation** (100 targets): 50-500 ms

Related Documentation
---------------------

- 📖 :doc:`horizons_quickref` — Quick lookup
- 📗 :doc:`ephemeris_horizons` — Full guide
- ❓ :doc:`horizons_faq` — Q&A
- 🔧 :doc:`horizons_implementation` — Architecture

External Links
---------------

- `JPL Horizons <https://ssd.jpl.nasa.gov/horizons/>`_
- `NAIF ID List <https://ssd.jpl.nasa.gov/?horizons>`_
- `rhorizons Crate <https://crates.io/crates/rhorizons>`_
- `GitHub Issues <https://github.com/CosmicFrontierLabs/rust-ephem/issues>`_
