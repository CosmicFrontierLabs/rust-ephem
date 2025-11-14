Planetary ephemeris (SPICE)
===========================

``SPICEEphemeris`` provides access to celestial body positions via SPK files (e.g.
DE440S). The class computes positions and velocities for any SPICE-supported body
in GCRS and ITRS coordinate frames.

Key functions
-------------

**Planetary Ephemeris Management**

``ensure_planetary_ephemeris(py_path=None, download_if_missing=True, spk_url=None)``
    Download (if needed) and initialize the planetary SPK. If no path is provided,
    uses the default cache location for de440s.bsp.

``init_planetary_ephemeris(py_path)``
    Initialize an already-downloaded planetary SPK file.

``download_planetary_ephemeris(url, dest)``
    Explicitly download a planetary SPK file from a URL to a destination path.

``is_planetary_ephemeris_initialized()``
    Check if the planetary ephemeris is initialized and ready. Returns ``bool``.

Usage example
-------------

.. code-block:: python

    import datetime as dt
    import numpy as np
    import rust_ephem as re

    # Ensure planetary ephemeris is available (downloads if missing)
    re.ensure_planetary_ephemeris()

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create SPICE ephemeris for Moon (NAIF ID 301) relative to Earth (399)
    moon_ephem = re.SPICEEphemeris(
        spk_path="path/to/moon.bsp",  # Your SPK file
        naif_id=301,                   # Moon
        begin=begin,
        end=end,
        step_size=step_size,
        center_id=399,                 # Earth center
        polar_motion=False
    )

    # Access pre-computed positions
    gcrs = moon_ephem.gcrs  # Position/velocity in GCRS
    itrs = moon_ephem.itrs  # Position/velocity in ITRS

    # Access Sun and Moon positions from planetary ephemeris
    sun = moon_ephem.sun
    moon = moon_ephem.moon

    # Access timestamps
    times = moon_ephem.timestamp

    print("Moon GCRS position (km):", gcrs.position[0])
    print("Moon distance (km):", np.linalg.norm(gcrs.position[0]))

    # Access astropy SkyCoord objects (requires astropy)
    gcrs_sc = moon_ephem.gcrs_sc
    earth_sc = moon_ephem.earth_sc
    sun_sc = moon_ephem.sun_sc

NAIF ID Reference
-----------------

Common NAIF IDs for celestial bodies:

- 10: Sun
- 199: Mercury
- 299: Venus
- 301: Moon
- 399: Earth
- 499: Mars
- 599: Jupiter
- 699: Saturn
- 799: Uranus
- 899: Neptune

SPK Files
---------

The library supports any SPICE SPK (ephemeris) file. Common options:

- **de440s.bsp** — Compact planetary ephemeris (1849-2150)
- **de440.bsp** — Full planetary ephemeris (1550-2650)
- Custom mission SPK files for specific spacecraft

Performance notes
-----------------

- SPICE loading is done once during initialization; subsequent queries are fast
- All frames are pre-computed during object creation for efficiency
- Keep the SPK file on a fast local disk; network-mounted paths add latency
- Use appropriate time ranges to avoid loading unnecessary data

SPK Error Handling
------------------

.. code-block:: python

    import rust_ephem as re

    try:
        # This will raise an error if file not found and download disabled
        re.ensure_planetary_ephemeris(
            py_path="missing.bsp",
            download_if_missing=False
        )
    except FileNotFoundError as e:
        print(f"SPK file not found: {e}")

    # Always check before creating ephemeris objects
    if re.is_planetary_ephemeris_initialized():
        print("Ready to create SPICEEphemeris objects")

See also: :doc:`examples/usage_spice` and :doc:`accuracy_precision`.


