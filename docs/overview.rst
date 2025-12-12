Project Overview
================

``rust-ephem`` is a Rust library with Python bindings for high-performance
satellite and planetary ephemeris calculations. It propagates Two-Line Element
(TLE) data and SPICE kernels, outputs standard coordinate frames (ITRS, GCRS),
and integrates with astropy for Python workflows.

Why rust-ephem?
---------------

**Performance**: Built in Rust for speed, ``rust-ephem`` generates ephemerides for
thousands of time steps efficiently. Ideal for visibility calculators where speed
is critical (e.g., APIs serving many users) and large-scale ephemeris tasks where
it outperforms pure-Python libraries by an order of magnitude.

**Accuracy**: Achieves 10-20 meter accuracy for Low Earth Orbit (LEO) satellites
with proper time corrections (UT1, polar motion, leap seconds).

**Integration**: Outputs ``astropy`` ``SkyCoord`` objects directly, including Sun
and Moon positions with observer location and velocity. This correctly handles
effects like Moon parallax for LEO spacecraft observatories.

**Constraints**: A flexible constraint system enables observation planning with
Sun/Moon proximity, Earth limb avoidance, and eclipse detection. Logical operators
(AND, OR, NOT, XOR) allow combining constraints with Python's ``&``, ``|``, ``~``,
``^`` operators.

Core Capabilities
-----------------

- **TLE propagation** using the SGP4 algorithm
- **SPICE kernel access** for spacecraft ephemeris from SPK files
- **Ground observatory ephemeris** for fixed Earth locations
- **OEM file support** for CCSDS Orbit Ephemeris Messages
- **JPL Horizons integration** for asteroids, comets, and spacecraft
- **Coordinate transformations** between TEME, ITRS, and GCRS frames
- **Time system conversions** (TAI, UT1, UTC) with leap seconds
- **Earth Orientation Parameters** (EOP) for polar motion corrections
- **Constraint evaluation** for observation planning

Key Technologies
----------------

- **Language**: `Rust (2021 edition) <https://www.rust-lang.org/>`_
- **Python integration**: `PyO3 <https://pyo3.rs/>`_ with `maturin <https://www.maturin.rs>`_ wheels
- **Astronomy libraries**:

  - `ERFA <https://docs.rs/erfa/latest/erfa/index.html>`_ (IAU standards)
  - `SGP4 <https://github.com/neuromorphicsystems/sgp4>`_ (pure-Rust TLE propagation)
  - `ANISE <https://github.com/nyx-space/anise>`_ (SPICE kernel handling)
  - `rhorizons <https://crates.io/crates/rhorizons>`_ (JPL Horizons API client)
  - `astropy <https://astropy.org>`_ (SkyCoord output)

- **Time handling**: `hifitime <https://github.com/nyx-space/hifitime>`_ for high-precision time
- **Arrays**: NumPy integration for efficient data handling

Typical Workflows
-----------------

**Satellite Ephemeris (TLE)**

1. Create ``TLEEphemeris`` from TLE data (file, URL, or Celestrak query)
2. Access TEME, ITRS, or GCRS coordinates as ``SkyCoord`` objects
3. Query Sun/Moon positions relative to the satellite

**Spacecraft Ephemeris (SPICE)**

1. Obtain SPK file for your spacecraft (from mission provider or NAIF)
2. Create ``SPICEEphemeris`` with the SPK path and spacecraft NAIF ID
3. Access positions in GCRS or ITRS frames
4. Call ``ensure_planetary_ephemeris()`` if you need Sun/Moon positions

**Ground Observatory**

1. Create ``GroundEphemeris`` with latitude, longitude, height
2. Access observatory positions in ITRS/GCRS
3. Query Sun/Moon positions from the observatory

**Constraint Evaluation**

1. Configure constraints (Sun proximity, Moon avoidance, eclipse, etc.)
2. Combine constraints with ``&``, ``|``, ``~`` operators
3. Evaluate against ephemeris to find visibility windows
4. Use batch evaluation for multiple targets efficiently

Next Steps
----------

- :doc:`frames` — Coordinate frame details
- :doc:`time_systems` — Time scale handling
- :doc:`accuracy_precision` — Accuracy information
- :doc:`api` — Complete API reference
