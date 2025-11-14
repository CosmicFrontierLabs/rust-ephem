Project overview
================

``rust-ephem`` is a Rust library with Python bindings for high-performance
satellite and planetary ephemeris calculations. It propagates Two-Line Element
(TLE) data and SPICE kernels, outputs standard coordinate frames (ITRS, GCRS),
and integrates with astropy for Python workflows. It achieves meters-level
accuracy for Low Earth Orbit (LEO) satellites with proper time corrections. It
also supports ground-based observatory ephemerides.

Built for performance: generates ephemerides for thousands of time steps using
Rust's speed and efficient memory handling. Ideal for visibility calculators
where speed is critical (e.g. APIs serving many users) and large-scale
ephemeris tasks where it outperforms pure-Python libraries by an order of
magnitude.

``rust-ephem`` supports outputting calculated ephemerides as ``astropy`` ``SkyCoord``
objects, alleviating the need to manually convert raw ephemeris data into astropy
frames. This makes it easy to integrate into existing Python astronomy workflows
that rely on astropy for coordinate transformations and time system handling.

By default ephemeris calculation includes locations the Sun and Moon in
``SkyCoord``, with observatory location and velocity included, so calculations of
distance between Sun and Moon will correctly account for the observer's motion.
Therefore issues that arise with LEO spacecraft observatories such as Moon
parallax are properly handled. In addition it can calculate ephemerides for
other solar system bodies.

It provides:

- TLE propagation using the SGP4 algorithm
- Coordinate transformations between TEME, ITRS, and GCRS frames
- Ground-based observatory ephemeris for fixed Earth locations
- Access to planetary ephemerides (SPICE, e.g. DE440S)
- Time system conversions (TAI, UT1, UTC) with leap seconds
- Earth Orientation Parameters (EOP) for polar motion corrections
- A concise Python API for high-performance workflows

Key technologies
----------------

- Language: `Rust (2021 edition) <https://www.rust-lang.org/>`_
- Python integration: `PyO3 <https://pyo3.rs/>`_, distributed via `maturin <https://www.maturin.rs>`_ wheels
- Astronomy libraries:
  
  - `ERFA <https://docs.rs/erfa/latest/erfa/index.html>`_ (IAU standards)
  - `SGP4 <https://github.com/neuromorphicsystems/sgp4>`_ (pure-Rust TLE propogation)
  - `ANISE <https://github.com/nyx-space/anise>`_ (pure-Python SPICE kernel propogation)
  - `astropy <https://astropy.org>`_ - natively outputs astropy SkyCoord objects
- Timing libraries: 
  
  - `hifitime <https://github.com/nyx-space/hifitime>`_ for high-precision time
    handling and corrections.

- Arrays: NumPy integration


Typical workflow
----------------

**For satellite ephemeris (TLE):**

1. Parse a TLE and propagate with SGP4 to get TEME position/velocity
2. Transform TEME → ITRS → GCRS (or TEME → GCRS) using ERFA routines
3. Optionally query Sun/Moon positions in GCRS

**For planetary ephemeris (SPICE):**

1. Ensure planetary SPK file is available (automatically downloads if needed)
2. Create SPICEEphemeris for a specific celestial body
3. Access positions in GCRS or ITRS frames

**For ground observatory:**

1. Define observatory location (latitude, longitude, height)
2. Create GroundEphemeris for a time range
3. Access observatory positions in ITRS/GCRS and Sun/Moon positions

**For constraint evaluation:**

1. Configure constraints (Sun/Moon proximity, eclipse avoidance, etc.)
2. Combine constraints using logical operators (AND, OR, NOT)
3. Evaluate against ephemeris data to find violation windows
4. Analyze results for observation planning

See :doc:`api` for the Python surface and :doc:`examples/index` for practical code.
