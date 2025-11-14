Accuracy and precision
======================

Overview
--------

``rust-ephem`` achieves high accuracy through modern astronomical standards and
optional Earth orientation corrections:

- **GCRS positions**: ~10-20 meters compared to astropy (with all corrections enabled)
- **Default accuracy**: ~100 meters (without UT1/polar motion corrections)
- **LEO velocity magnitudes**: 7-8 km/s (typical for Low Earth Orbit)

Accuracy Levels
---------------

The library provides multiple accuracy levels depending on configuration:

**Basic (Default)**
  - Accuracy: ~100 meters
  - Uses embedded leap seconds (TAI-UTC)
  - Assumes UT1-UTC = 0
  - No polar motion correction
  - Sufficient for: General satellite tracking, educational purposes

**High Accuracy (UT1 enabled)**
  - Accuracy: ~20 meters
  - Uses IERS Earth Orientation Parameters (EOP) for UT1-UTC corrections
  - Automatic download from JPL
  - Sufficient for: Most operational satellite tracking, astronomical observations

**Maximum Accuracy (UT1 + Polar Motion)**
  - Accuracy: ~10-20 meters
  - Uses both UT1-UTC and polar motion corrections
  - Best achievable with SGP4 propagation model
  - Sufficient for: Scientific research, high-precision applications

Enabling High Accuracy
----------------------

Use the ``polar_motion`` parameter and initialize data providers:

.. code-block:: python

    import rust_ephem
    from datetime import datetime, timezone

    # Initialize UT1 and EOP data providers (downloads from JPL)
    rust_ephem.init_ut1_provider()
    rust_ephem.init_eop_provider()

    # Create ephemeris with polar motion correction
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    
    ephem = rust_ephem.TLEEphemeris(
        tle1, tle2, begin, end, step_size=60,
        polar_motion=True  # Enable polar motion corrections
    )
    
    # Now GCRS positions have ~10-20m accuracy

Time Scale Accuracy
-------------------

**Leap Seconds (TAI-UTC)**

The library includes embedded leap second data:

- **Coverage**: 1972 to present (28 leap seconds through 2017)
- **Accuracy**: Sub-microsecond time conversion accuracy
- **No dependencies**: Works offline without external files
- **Automatic**: No initialization required

.. code-block:: python

    import rust_ephem
    from datetime import datetime, timezone
    
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tai_utc = rust_ephem.get_tai_utc_offset(dt)  # Returns 32.0 seconds

**UT1-UTC (Earth Rotation)**

UT1-UTC corrections account for Earth's irregular rotation:

- **Data source**: IERS EOP2 from JPL
- **Automatic download**: First use or when cache expires
- **Coverage**: ~1 year historical + 6 months predicted
- **Impact**: Improves accuracy from ~100m to ~20m
- **Range**: ±0.9 seconds (updated daily by IERS)

.. code-block:: python

    # Check availability
    if rust_ephem.is_ut1_available():
        ut1_utc = rust_ephem.get_ut1_utc_offset(dt)
        print(f"UT1-UTC: {ut1_utc:.6f} seconds")

**Polar Motion (xp, yp)**

Polar motion describes Earth's rotation axis movement:

- **Data source**: IERS EOP2 from JPL
- **Correction magnitude**: ~10-20 meters
- **Parameters**: xp, yp in arcseconds
- **Optional**: Disabled by default for backward compatibility

.. code-block:: python

    # Get polar motion values
    if rust_ephem.is_eop_available():
        xp, yp = rust_ephem.get_polar_motion(dt)
        print(f"Polar motion: xp={xp:.6f}\", yp={yp:.6f}\"")

Data Caching
------------

IERS data is automatically cached locally:

- **Cache location**: ``$HOME/.cache/rust_ephem/latest_eop2.short``
- **TTL**: 24 hours (configurable)
- **Environment variables**:
  
  - ``RUST_EPHEM_EOP_CACHE``: Custom cache directory
  - ``RUST_EPHEM_EOP_CACHE_TTL``: Cache time-to-live in seconds

- **Fallback**: Returns zero offset if data unavailable (graceful degradation)

Moon Position Accuracy
----------------------

Two methods with different accuracy levels:

**Meeus Algorithm (Built-in)**
  - Accuracy: ~32 arcminutes (0.5 degrees)
  - No external data required
  - Sufficient for: Spacecraft applications, general astronomy
  - Automatically used when SPICE not initialized

**SPICE/ANISE (High Precision)**
  - Accuracy: <1 arcsecond
  - Requires: DE440 or DE441 kernel file
  - JPL-quality ephemeris
  - Required for: Scientific research, high-precision work

.. code-block:: python

    # Use high-precision SPICE-based Moon positions
    rust_ephem.ensure_planetary_ephemeris()  # Downloads DE440S if needed
    
    ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
    moon = ephem.moon  # Now uses SPICE if initialized

Coordinate Frame Transformations
---------------------------------

The library implements IAU-standard transformations:

- **ERFA library**: Essential Routines for Fundamental Astronomy
- **Precession-nutation**: IAU 2006 model (``pn_matrix_06a``)
- **Frame bias**: Proper ICRS/GCRS alignment
- **Earth rotation**: UT1-based when available

Transformation accuracy:

- **TEME → GCRS**: ~10-20m (with all corrections)
- **GCRS → ITRS**: ~10-20m (with polar motion)
- **ITRS → GCRS**: ~10-20m (with polar motion)

Error Sources
-------------

The remaining ~10-20m position error after all corrections is due to:

**SGP4 Model Limitations**
  - Simplified atmospheric drag model
  - Simplified Earth gravity model (J2-J4 only)
  - No solar radiation pressure
  - TLE orbital element uncertainties

**Frame Transformation Approximations**
  - Numerical precision limits
  - Higher-order corrections omitted
  - Ephemeris model differences

**For higher accuracy**: Use SPICE-based ephemeris with high-fidelity orbit
propagators (not SGP4) and precise orbit determination products.

Testing and Validation
-----------------------

Validation against astropy:

.. code-block:: bash

    # Run Python integration tests
    python tests/integration_test_gcrs.py
    python tests/integration_test_itrs_skycoord.py
    python tests/integration_test_sun_moon_skycoord.py

Expected results:

- Position errors: <20 meters (with UT1 + polar motion)
- Position magnitudes: 6500-8000 km for LEO satellites
- Velocity magnitudes: 7-8 km/s for LEO orbits
- GCRS and TEME magnitudes: Similar within ~100m

Rust unit tests:

.. code-block:: bash

    cargo test --verbose

All tests validate:

- Coordinate transformation accuracy
- Time scale conversions
- Sun/Moon position calculations
- Edge cases and error handling

See also: :doc:`time_systems` for time scale details and :doc:`frames` for coordinate system information.
