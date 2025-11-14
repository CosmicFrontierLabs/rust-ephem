Time systems
============

Time handling is critical for precise ephemerides. ``rust-ephem`` uses:

- UTC for external times passed from Python
- TT (Terrestrial Time) internally for some ERFA routines
- TAI (International Atomic Time) for leap second calculations
- UT1 (Universal Time) for Earth rotation calculations
- Julian Date (JD) split into two parts (JD1, JD2) to preserve precision

Time System Functions
---------------------

The library provides several functions to work with different time systems:

**Leap Seconds (TAI-UTC)**

.. code-block:: python

    import datetime as dt
    import rust_ephem as re

    when = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    tai_utc = re.get_tai_utc_offset(when)
    print(f"TAI-UTC offset: {tai_utc} seconds")

The TAI-UTC offset represents the number of leap seconds at a given time.

**UT1-UTC Offset**

.. code-block:: python

    import rust_ephem as re

    # Initialize UT1 provider (loads IERS data)
    if re.init_ut1_provider():
        print("UT1 data available")
        
        when = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        ut1_utc = re.get_ut1_utc_offset(when)
        print(f"UT1-UTC offset: {ut1_utc} seconds")
    
    # Check availability
    if re.is_ut1_available():
        print("UT1 data is loaded")

UT1-UTC varies continuously due to Earth's irregular rotation and requires
up-to-date IERS data for accurate values.

**Earth Orientation Parameters (EOP)**

.. code-block:: python

    import rust_ephem as re

    # Initialize EOP provider (loads polar motion data)
    if re.init_eop_provider():
        print("EOP data available")
        
        when = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        xp, yp = re.get_polar_motion(when)
        print(f"Polar motion: x_p = {xp} arcsec, y_p = {yp} arcsec")
    
    # Check availability
    if re.is_eop_available():
        print("EOP data is loaded")

Polar motion parameters are used for high-precision coordinate transformations.

Assumptions and approximations
------------------------------

When IERS data is not available:

- TT − UTC is approximated as 69.184 seconds (sufficient for many LEO tasks)
- UT1 − UTC is assumed to be zero
- Polar motion (x_p, y_p) is assumed to be zero

These approximations are typically sufficient for:

- Low Earth Orbit (LEO) satellite tracking
- General-purpose ephemeris calculations
- Applications requiring ~100m position accuracy

For higher accuracy requirements:

- Initialize UT1 provider with ``init_ut1_provider()``
- Initialize EOP provider with ``init_eop_provider()``
- Enable polar motion corrections with ``polar_motion=True`` parameter

Practical guidance
------------------

- Always pass timezone-aware datetimes (UTC) from Python
- Be consistent with time scales when comparing to external tools (e.g., astropy)
- For sub-arcsecond accuracy, use up-to-date IERS data and enable all corrections
- The library automatically handles leap seconds for dates with known TAI-UTC offsets
- UT1 and polar motion data must be explicitly loaded via init functions

See also: :doc:`frames` and :doc:`accuracy_precision`.

