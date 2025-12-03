Planetary Ephemeris
===================

To compute Sun, Moon, and planet positions relative to any observer, you need
to initialize the planetary ephemeris. This is required for:

- Accessing ``.sun`` and ``.moon`` attributes on any ephemeris object
- Using ``get_body()`` and ``get_body_pv()`` methods
- Evaluating constraints (``SunConstraint``, ``MoonConstraint``, etc.)

The planetary ephemeris uses NASA JPL's DE440S kernel, which provides high-precision
positions for the Sun, Moon, and planets from 1849-2150.

Setup Functions
---------------

``ensure_planetary_ephemeris(py_path=None, download_if_missing=True, spk_url=None)``
    Download (if needed) and initialize the planetary SPK. If no path is provided,
    uses the default cache location for de440s.bsp. This is the recommended function
    for most users.

``init_planetary_ephemeris(py_path)``
    Initialize an already-downloaded planetary SPK file. Use this if you have
    a custom SPK file or want to avoid automatic downloads.

``download_planetary_ephemeris(url, dest)``
    Explicitly download a planetary SPK file from a URL to a destination path.

``is_planetary_ephemeris_initialized()``
    Check if the planetary ephemeris is initialized and ready. Returns ``bool``.

Basic Usage
-----------

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    # Initialize planetary ephemeris (downloads DE440S on first use)
    re.ensure_planetary_ephemeris()

    # Create any ephemeris type
    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    ephem = re.TLEEphemeris(
        norad_id=25544,  # ISS
        begin=begin,
        end=end,
        step_size=60
    )

    # Now Sun and Moon positions are available
    sun_positions = ephem.sun    # SkyCoord of Sun from observer
    moon_positions = ephem.moon  # SkyCoord of Moon from observer

Using get_body()
----------------

The ``get_body()`` and ``get_body_pv()`` methods work with any ephemeris type:

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    re.ensure_planetary_ephemeris()

    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    # Works with TLEEphemeris
    tle_ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)
    mars_from_iss = tle_ephem.get_body("mars")

    # Works with GroundEphemeris
    ground = re.GroundEphemeris(
        lon=-122.0, lat=37.0, alt=0.0,
        begin=begin, end=end, step_size=60
    )
    jupiter_from_ground = ground.get_body("jupiter")

    # Works with SPICEEphemeris
    spice_ephem = re.SPICEEphemeris(
        spk_path="spacecraft.bsp",
        naif_id=-12345,
        begin=begin, end=end
    )
    saturn_from_spacecraft = spice_ephem.get_body("saturn")

Supported Bodies
----------------

The following body names are supported with ``get_body()``:

- ``"sun"`` — The Sun
- ``"moon"`` — Earth's Moon
- ``"mercury"`` — Mercury
- ``"venus"`` — Venus
- ``"earth"`` — Earth (useful for non-Earth-centered observers)
- ``"mars"`` — Mars
- ``"jupiter"`` — Jupiter
- ``"saturn"`` — Saturn
- ``"uranus"`` — Uranus
- ``"neptune"`` — Neptune

NAIF ID Reference
-----------------

Solar system body NAIF IDs (for advanced use):

- 10: Sun
- 301: Moon
- 199: Mercury
- 299: Venus
- 399: Earth
- 499: Mars
- 599: Jupiter
- 699: Saturn
- 799: Uranus
- 899: Neptune

Planetary SPK Files
-------------------

The default DE440S kernel is recommended for most applications:

- **de440s.bsp** — Compact planetary ephemeris (1849-2150), ~32 MB
- **de440.bsp** — Full planetary ephemeris (1550-2650), ~114 MB

Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

Custom SPK File
---------------

To use a different planetary SPK file:

.. code-block:: python

    import rust_ephem as re

    # Use a specific SPK file
    re.init_planetary_ephemeris("/path/to/de440.bsp")

    # Or download from a custom URL
    re.download_planetary_ephemeris(
        url="https://example.com/custom.bsp",
        dest="/path/to/custom.bsp"
    )
    re.init_planetary_ephemeris("/path/to/custom.bsp")

Error Handling
--------------

.. code-block:: python

    import rust_ephem as re
    from datetime import datetime, timezone

    begin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    # Check if already initialized
    if not re.is_planetary_ephemeris_initialized():
        try:
            re.ensure_planetary_ephemeris()
        except Exception as e:
            print(f"Failed to initialize planetary ephemeris: {e}")

    # Accessing Sun/Moon without initialization raises an error
    ephem = re.TLEEphemeris(norad_id=25544, begin=begin, end=end)
    
    if re.is_planetary_ephemeris_initialized():
        sun = ephem.sun  # Works
    else:
        print("Initialize planetary ephemeris first!")

Performance Notes
-----------------

- Planetary ephemeris is loaded once and cached in memory
- Sun/Moon positions are computed during ephemeris object creation
- The ``get_body()`` method is efficient for batch queries
- DE440S is optimized for size while maintaining high accuracy

See also: :doc:`ephemeris_get_body` for detailed examples of querying body positions.


