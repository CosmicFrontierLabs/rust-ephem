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

Geometric and Apparent Positions
--------------------------------

By default, ``get_body()`` returns the body's geometric position at the
observation time. For the direction in which received light appears to arrive,
request an apparent position explicitly:

.. code-block:: python

    sun = ephem.get_body("sun", apparent=True)
    moon = ephem.get_body("moon", apparent=True)
    print(sun.ra.deg, sun.dec.deg)

This option is available on every ephemeris type. It uses the observer's
position and velocity, including a spacecraft's motion, and applies:

* Converged reception light-travel time, solved in barycentric coordinates.
* Solar light deflection with finite source distance.
* Relativistic aberration using the observer's barycentric velocity.

The result is an observer-centred GCRS ``SkyCoord`` with apparent direction
and light-time distance, using the same convention as Astropy's
``get_body()``. The distance is the source-to-observer distance along the
retarded light path, rather than the instantaneous geometric distance.
The correction is calculated in Rust; Astropy is used to package the result.
Apparent queries return positions only, without velocity differentials.
``get_body_pv()`` continues to return geometric states.

An apparent query requires a planetary SPICE kernel covering both reception
and emission times, with barycentric states for the target, Earth, and Sun.
Use ``ensure_planetary_ephemeris()`` as above or pass
``spice_kernel="/path/to/de440s.bsp"``. Apparent queries do not use the
approximate analytic Sun/Moon models or the Horizons fallback;
``apparent=True, use_horizons=True`` raises ``ValueError``.

Existing ``sun``, ``moon``, ``sun_pv``, ``moon_pv``, RA/Dec properties, and
constraint calculations retain their geometric convention. Calling
``get_body(..., apparent=True)`` does not change those cached values.
Astropy's ``get_body()`` defaults to an apparent position, so comparisons
should select ``apparent=True`` and use the same planetary kernel and
observer states. Remaining differences can reflect kernel or
Earth-orientation inputs.

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
