.. _horizons-faq:

JPL Horizons FAQ and Best Practices
====================================

Frequently Asked Questions
--------------------------

**Q: How is JPL Horizons different from SPICE kernels?**

A: SPICE kernels like DE440S are comprehensive, pre-computed ephemeris files cached
locally for fast access. JPL Horizons is NASA's online service that computes positions
on-demand. Key differences:

+-------------------+------------------+-----------------+
| Feature           | SPICE Kernels    | JPL Horizons    |
+===================+==================+=================+
| Speed             | Instant (cached) | Network-bound   |
+-------------------+------------------+-----------------+
| Setup             | One-time download| None required   |
+-------------------+------------------+-----------------+
| Body coverage     | Limited          | Comprehensive   |
+-------------------+------------------+-----------------+
| Network required  | No               | Yes             |
+-------------------+------------------+-----------------+
| Accuracy          | High             | High            |
+-------------------+------------------+-----------------+

**Q: When should I use Horizons?**

A: Use Horizons when you need to track:

- **Asteroids** (numbered and unnumbered)
- **Comets** (periodic and non-periodic)
- **Spacecraft** (Voyager, New Horizons, Parker Solar Probe, etc.)
- **Minor planets and dwarf planets** (Ceres, Pluto, Eris, etc.)
- **Planets** when SPICE kernels don't have the data you need

Use SPICE kernels for the fastest performance on major bodies (Sun, Moon, planets).

**Q: Why do I need ``use_horizons=True``?**

A: This parameter is explicit to avoid surprise network requests. By default,
``use_horizons=False`` means no network queries are made—the code stays fast and
predictable. Set ``use_horizons=True`` only when you explicitly want Horizons fallback.

.. code-block:: python

    # Fast, predictable, no network
    mars = ephem.get_body("Mars")

    # May hit network, but gets broader coverage
    ceres = ephem.get_body("Ceres", use_horizons=True)

**Q: Does Horizons fallback affect performance?**

A: Only when a body is queried that requires Horizons:

- **SPICE bodies**: Same speed as before (cached)
- **Horizons bodies**: ~0.5-2 seconds per query (network-dependent)
- **Cached results**: If you query the same body multiple times, implement
  application-level caching

**Q: Can I batch query multiple asteroids?**

A: Yes, but note each ``get_body()`` call is independent:

.. code-block:: python

    asteroids = ["1", "4", "2", "3122", "99942"]

    positions = {}
    for body_id in asteroids:
        positions[body_id] = ephem.get_body(body_id, use_horizons=True)

    # Each call may hit the network; consider caching if you repeat this

For visibility windows, use ``moving_body_visibility()`` which internally batches
constraint evaluation efficiently.

**Q: What body identifiers does Horizons accept?**

A: JPL Horizons recognizes:

1. **NAIF IDs** (integers): ``"1"`` (Ceres), ``"499"`` (Mars), ``"-98"`` (New Horizons)
2. **Minor planet numbers**: ``"433"`` (Eros), ``"99942"`` (Apophis)
3. **Names**: ``"Ceres"``, ``"Apophis"``, ``"Halley"``, ``"C/2020 F3"`` (comets)
4. **Spacecraft names**: ``"New Horizons"``, ``"Voyager 1"`` (with NAIF ID preferred)

When in doubt, use JPL's `Horizons browser <https://ssd.jpl.nasa.gov/horizons/>`_
to find the exact identifier.

**Q: What if Horizons returns "body not found"?**

A: The body may not exist in Horizons, or the time range is outside its supported range:

1. **Check the identifier** on JPL's Horizons website
2. **Try a shorter time range** (recently discovered objects have limited data)
3. **Use a SPICE kernel** if the body is major (planets, moons)
4. **Check Horizons service status** at https://ssd.jpl.nasa.gov/

**Q: How accurate is Horizons data?**

A: Accuracy varies by body:

- **Planets and major moons**: Meters to km level
- **Asteroids**: Typically km level, depends on observational arc
- **Comets**: Highly variable, can be 100s of km near perihelion
- **Spacecraft**: Accurate while tracked, unavailable after tracking ends

For mission-critical work, verify Horizons accuracy against other sources
or use specialized propagation models.

**Q: Can I use Horizons for prediction far in the future?**

A: It depends on the body:

- **Planets**: Accurate for many millennia
- **Asteroids**: Generally good for ±100 years from current epoch
- **Comets**: Often only ±decades (uncertain orbital parameters)
- **Spacecraft**: Only while tracking data is available

Start with your desired time range; if Horizons can't compute it, try nearby dates.

Best Practices
--------------

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Cache Horizons queries in your application:**

.. code-block:: python

    _body_cache = {}

    def get_body_cached(ephem, body_id):
        if body_id not in _body_cache:
            _body_cache[body_id] = ephem.get_body(body_id, use_horizons=True)
        return _body_cache[body_id]

    # First call: hits network
    ceres = get_body_cached(ephem, "1")

    # Subsequent calls: instant
    ceres = get_body_cached(ephem, "1")

**2. Use SPICE for high-frequency lookups:**

.. code-block:: python

    # Sun/Moon/planets: use SPICE (fastest)
    sun = ephem.get_body("Sun")
    moon = ephem.get_body("Moon")

    # Asteroids: use Horizons (network)
    ceres = ephem.get_body("Ceres", use_horizons=True)

**3. Batch constraint evaluation:**

.. code-block:: python

    # Evaluate 100 targets efficiently
    constraint = SunConstraint(min_angle=45)
    violations = constraint.in_constraint_batch(ephem, ras, decs)

Reliability and Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Handle network failures gracefully:**

.. code-block:: python

    try:
        body = ephem.get_body("99942", use_horizons=True)
    except Exception as e:
        print(f"Horizons query failed: {e}")
        # Fallback to SPICE or skip this body
        body = None

**2. Verify results make sense:**

.. code-block:: python

    import numpy as np

    body_pv = ephem.get_body_pv("1", use_horizons=True)
    distance = np.linalg.norm(body_pv.position[0])

    # Sanity check: asteroid should be within ~10 AU
    if distance > 1500e6:  # > 10 AU in km
        print("Warning: Unexpectedly large distance")

**3. Test with smaller time ranges first:**

.. code-block:: python

    from datetime import datetime, timezone

    # Test: single day
    begin = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 2, tzinfo=timezone.utc)

    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)
    body = ephem.get_body("1", use_horizons=True)  # Should work

    # If that succeeds, expand time range
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)

Integration Patterns
~~~~~~~~~~~~~~~~~~~~

**Pattern 1: Multi-observer asteroid tracking**

.. code-block:: python

    from datetime import datetime, timezone

    # Define asteroid and observation sites
    asteroid_id = "99942"  # Apophis
    begin = datetime(2029, 4, 1, tzinfo=timezone.utc)
    end = datetime(2029, 4, 14, tzinfo=timezone.utc)

    sites = {
        "Keck": (19.8267, -155.4730, 4207),
        "VLT": (-24.6276, -70.4035, 2635),
        "Gemini": (-30.2408, -70.7345, 2737),
    }

    results = {}
    for site_name, (lat, lon, height) in sites.items():
        obs = re.GroundEphemeris(
            latitude=lat, longitude=lon, height=height,
            begin=begin, end=end, step_size=3600
        )

        constraint = SunConstraint(min_angle=10) & MoonConstraint(min_angle=15)
        visibility = moving_body_visibility(
            constraint=constraint,
            ephemeris=obs,
            body=asteroid_id,
            use_horizons=True
        )
        results[site_name] = visibility

    # Analyze observations from all sites
    for site, result in results.items():
        print(f"{site}: {len(result.visibility)} windows")

**Pattern 2: Periodic monitoring with caching**

.. code-block:: python

    import json
    from datetime import datetime, timezone
    from pathlib import Path

    CACHE_FILE = Path("horizons_cache.json")

    def get_body_with_cache(ephem, body_id, use_cache=True):
        """Query body, optionally using disk cache"""
        cache_key = f"{body_id}_{ephem.begin}_{ephem.end}"

        # Check disk cache
        if use_cache and CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                cache = json.load(f)
                if cache_key in cache:
                    print(f"Cache hit for {body_id}")
                    return cache[cache_key]

        # Query Horizons
        print(f"Querying Horizons for {body_id}")
        body = ephem.get_body(body_id, use_horizons=True)

        # Update disk cache
        if use_cache:
            cache = json.load(open(CACHE_FILE)) if CACHE_FILE.exists() else {}
            # Store simplified result (e.g., coordinates at sample times)
            cache[cache_key] = str(body)  # Simplified for demo
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        return body

**Pattern 3: Graceful degradation**

.. code-block:: python

    def get_body_safe(ephem, body_id, use_spice=True, use_horizons=True):
        """Try SPICE first, then Horizons, then fail gracefully"""

        # Try SPICE
        if use_spice:
            try:
                return ephem.get_body(body_id)  # Fast, no network
            except Exception:
                pass

        # Try Horizons
        if use_horizons:
            try:
                return ephem.get_body(body_id, use_horizons=True)  # Network query
            except Exception as e:
                print(f"Horizons query failed: {e}")

        # Fall back to None
        print(f"Could not find {body_id}")
        return None

    # Usage
    for body_id in ["Sun", "Mars", "Ceres", "Apophis", "Unknown"]:
        result = get_body_safe(ephem, body_id)
        if result:
            print(f"{body_id}: RA={result[0].ra}, Dec={result[0].dec}")

Common Pitfalls
---------------

**Pitfall 1: Assuming cached behavior**

.. code-block:: python

    # ❌ WRONG: Each call hits the network
    for i in range(10):
        ceres = ephem.get_body("1", use_horizons=True)  # Network query each time!

    # ✅ RIGHT: Cache the result
    ceres = ephem.get_body("1", use_horizons=True)  # Network query once
    for i in range(10):
        # Use cached result
        print(ceres)

**Pitfall 2: Forgetting about network dependencies**

.. code-block:: python

    # ❌ WRONG: Fails silently on network error
    body = ephem.get_body("Apophis", use_horizons=True)  # What if network is down?

    # ✅ RIGHT: Handle network failures
    try:
        body = ephem.get_body("Apophis", use_horizons=True)
    except Exception as e:
        print(f"Network error: {e}; using fallback")
        body = None

**Pitfall 3: Querying beyond supported time ranges**

.. code-block:: python

    # ❌ WRONG: May fail for newly discovered comets
    from datetime import datetime, timezone
    begin = datetime(1900, 1, 1, tzinfo=timezone.utc)
    end = datetime(2100, 1, 1, tzinfo=timezone.utc)
    ephem = re.TLEEphemeris(..., begin=begin, end=end)
    comet = ephem.get_body("C/2024 A1", use_horizons=True)  # May fail!

    # ✅ RIGHT: Verify time range support first
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ephem = re.TLEEphemeris(..., begin=begin, end=end)
    comet = ephem.get_body("C/2024 A1", use_horizons=True)

Related Documentation
---------------------

- :doc:`ephemeris_horizons` — Comprehensive JPL Horizons guide
- :doc:`ephemeris_get_body` — Basic body lookup documentation
- :doc:`planning_constraints` — Constraint system with Horizons examples
- :doc:`planning_visibility` — Visibility windows for moving targets
- `JPL Horizons Homepage <https://ssd.jpl.nasa.gov/horizons/>`_
- `NAIF ID List <https://ssd.jpl.nasa.gov/?horizons>`_
