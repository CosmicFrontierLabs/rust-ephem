SPICE Ephemeris
===============

``SPICEEphemeris`` provides spacecraft ephemeris from SPICE SPK (Spacecraft and
Planet Kernel) files. This is ideal for missions that distribute their trajectory
data in SPICE format, which is the standard for NASA and ESA missions.

.. note::
   ``SPICEEphemeris`` is for **spacecraft** ephemeris from SPK files, not for
   querying planetary positions. To get Sun, Moon, and planet positions relative
   to any observer, use the ``get_body()`` and ``get_body_pv()`` methods available
   on all ephemeris types after calling ``ensure_planetary_ephemeris()``.

Planetary Ephemeris Setup
-------------------------

Before using Sun/Moon positions or the ``get_body()`` method with any ephemeris
type, you must initialize the planetary ephemeris:

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

    # Ensure planetary ephemeris is available for Sun/Moon positions
    re.ensure_planetary_ephemeris()

    # Define time range
    begin = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    end = dt.datetime(2024, 1, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    step_size = 60  # seconds

    # Create SPICE ephemeris for a spacecraft
    # The SPK file contains your spacecraft's trajectory
    spacecraft_ephem = re.SPICEEphemeris(
        spk_path="path/to/spacecraft.bsp",  # Your spacecraft SPK file
        naif_id=-12345,                      # Your spacecraft's NAIF ID
        begin=begin,
        end=end,
        step_size=step_size,
        center_id=399,                       # Earth center (default)
        polar_motion=False
    )

    # Access spacecraft positions
    gcrs = spacecraft_ephem.gcrs  # Position/velocity in GCRS as SkyCoord
    itrs = spacecraft_ephem.itrs  # Position/velocity in ITRS as SkyCoord

    # Access Sun and Moon positions relative to spacecraft
    sun = spacecraft_ephem.sun
    moon = spacecraft_ephem.moon

    # Access timestamps
    times = spacecraft_ephem.timestamp

    print("Spacecraft GCRS position (km):", spacecraft_ephem.gcrs_pv.position[0])
    print("Distance from Earth (km):", np.linalg.norm(spacecraft_ephem.gcrs_pv.position[0]))

NAIF ID Reference
-----------------

Spacecraft NAIF IDs are typically negative numbers assigned by NAIF or the
mission. Common spacecraft IDs include:

- -82: Cassini
- -98: New Horizons
- -143: Mars Reconnaissance Orbiter
- -236: MESSENGER
- -140: Deep Impact

For your spacecraft, check your SPK file documentation or the NAIF website.

Solar system body IDs (for ``center_id`` or ``get_body()``):

- 10: Sun
- 301: Moon
- 399: Earth
- 499: Mars
- 599: Jupiter
- 699: Saturn

SPK Files
---------

SPK (Spacecraft and Planet Kernel) files contain trajectory data. Sources include:

**Spacecraft SPK files** (for ``SPICEEphemeris``):

- Mission-specific files from NAIF: https://naif.jpl.nasa.gov/naif/data.html
- ESA SPICE Service: https://www.cosmos.esa.int/web/spice
- Mission websites and data archives

**Planetary SPK files** (for ``ensure_planetary_ephemeris()``):

- **de440s.bsp** — Compact planetary ephemeris (1849-2150), ~32 MB
- **de440.bsp** — Full planetary ephemeris (1550-2650), ~114 MB
- Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

SPK Loading Performance
-----------------------

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

See also: :doc:`ephemeris_spice` and :doc:`accuracy_precision`.


