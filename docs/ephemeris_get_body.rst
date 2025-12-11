Using get_body for Celestial Bodies
====================================

The ``get_body()`` and ``get_body_pv()`` methods provide a flexible way to
query any supported solar system body's position relative to your observer
(satellite, spacecraft, or ground station).

Setup
-----

Before using ``get_body()``, ensure the planetary ephemeris is loaded:

.. code-block:: python

    import rust_ephem

    # Load planetary ephemeris (downloads DE440S if needed)
    rust_ephem.ensure_planetary_ephemeris()

Basic Usage
-----------

.. code-block:: python

    import datetime as dt
    import rust_ephem as re

    # Create any ephemeris type
    tle1 = "1 25544U 98067A   20344.91777778  .00002182  00000-0  46906-4 0  9991"
    tle2 = "2 25544  51.6460  44.6055 0002398  79.4451  23.5248 15.49364984256518"
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)

    ephem = re.TLEEphemeris(tle1, tle2, begin, end, step_size=60)

    # Get body position/velocity data
    sun_pv = ephem.get_body_pv("Sun")
    moon_pv = ephem.get_body_pv("Moon")
    mars_pv = ephem.get_body_pv("Mars")

    print(f"Sun distance: {np.linalg.norm(sun_pv.position[0]):.0f} km")
    print(f"Moon distance: {np.linalg.norm(moon_pv.position[0]):.0f} km")
    print(f"Mars distance: {np.linalg.norm(mars_pv.position[0]):.0f} km")

    # Get body as astropy SkyCoord (includes observer location)
    sun_sc = ephem.get_body("Sun")
    print(f"Sun RA/Dec: {sun_sc[0].ra}, {sun_sc[0].dec}")

Body Identifiers
----------------

Bodies can be specified by name or NAIF ID:

.. code-block:: python

    # By name (case-insensitive)
    ephem.get_body("Sun")
    ephem.get_body("moon")
    ephem.get_body("Mars")
    ephem.get_body("Jupiter barycenter")

    # By NAIF ID as string
    ephem.get_body("10")    # Sun
    ephem.get_body("301")   # Moon
    ephem.get_body("499")   # Mars (requires full DE440 kernel)
    ephem.get_body("5")     # Jupiter barycenter

**Note:** The default DE440S kernel includes:

- Sun (10), Moon (301), Earth (399)
- Planetary barycenters (1-9): Mercury, Venus, Earth-Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto

For planet center IDs (like 499 Mars, 599 Jupiter), use the full DE440 kernel.

Ground Observatory Example
--------------------------

.. code-block:: python

    # Create ground observatory
    obs = re.GroundEphemeris(
        latitude=19.8207,    # Mauna Kea
        longitude=-155.468,
        height=4205.0,
        begin=begin,
        end=end,
        step_size=60
    )

    # Get bodies from observatory perspective
    moon = obs.get_body("Moon")
    jupiter = obs.get_body("Jupiter barycenter")

    # The SkyCoord includes observer location for proper parallax
    print(f"Observer location set: {moon.frame.obsgeoloc[0]}")

Calculating Separations
-----------------------

.. code-block:: python

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Get positions
    moon = ephem.get_body("Moon")
    mars = ephem.get_body("Mars")

    # Calculate angular separation
    separation = moon.separation(mars)
    print(f"Moon-Mars separation: {separation[0].to(u.deg):.2f}")

    # Define a target and check separation from Sun
    target = SkyCoord(ra=83.63*u.deg, dec=22.01*u.deg, frame='icrs')
    sun = ephem.get_body("Sun")
    sun_sep = target.separation(sun)
    print(f"Target-Sun separation: {sun_sep[0].to(u.deg):.2f}")

JPL Horizons Fallback
---------------------

For solar system bodies not in your SPICE kernel (asteroids, comets, spacecraft),
enable JPL Horizons with ``use_horizons=True``:

.. code-block:: python

    # Query asteroid Ceres (not in default DE440S kernel)
    ceres = ephem.get_body("1", use_horizons=True)
    print(f"Ceres RA/Dec: {ceres[0].ra}, {ceres[0].dec}")

    # Or by name
    apophis = ephem.get_body("Apophis", use_horizons=True)

    # Get position/velocity data
    ceres_pv = ephem.get_body_pv("1", use_horizons=True)
    print(f"Distance: {np.linalg.norm(ceres_pv.position[0]):.0f} km")

Horizons is automatically queried only when:

1. The body is not found in SPICE kernels
2. ``use_horizons=True`` is explicitly set
3. An internet connection is available

This provides the best of both worlds: fast SPICE lookups when available, with
automatic fallback to Horizons for broader body coverage. For a comprehensive
guide to Horizons usage including asteroid tracking, constraint integration, and
troubleshooting, see :doc:`ephemeris_horizons`.
