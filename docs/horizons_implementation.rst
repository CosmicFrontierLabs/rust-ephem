.. _horizons-implementation:

JPL Horizons Implementation Guide
==================================

For developers who want to understand how JPL Horizons is integrated into ``rust-ephem``.

Architecture Overview
---------------------

The JPL Horizons integration consists of several layers:

::

    ┌─────────────────────────────────────────┐
    │  Python API (rust_ephem/constraints.py) │
    │  - moving_body_visibility()             │
    │  - use_horizons parameter               │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  PyO3 Bindings (Rust ↔ Python)          │
    │  - src/ephemeris/*.rs (#[pyo3])         │
    │  - Default use_horizons=false           │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  Rust Core (src/utils/celestial.rs)     │
    │  - calculate_body_by_id_or_name()       │
    │  - Tries SPICE first, falls back        │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  Horizons Module (src/utils/horizons.rs)│
    │  - query_horizons_body()                │
    │  - Uses tokio for async runtime         │
    │  - Depends on rhorizons crate           │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  NASA JPL Horizons API (HTTP)           │
    │  - https://ssd.jpl.nasa.gov/horizons/   │
    └─────────────────────────────────────────┘

Code Flow
---------

When you call ``ephem.get_body("1", use_horizons=True)``:

1. **Python → PyO3** — ``get_body()`` method receives parameters
2. **PyO3 Binding** — Calls Rust ``EphemerisBase::get_body_impl()`` with use_horizons flag
3. **Core Lookup** — ``calculate_body_by_id_or_name()`` is called
4. **SPICE First** — Attempts SPICE kernel lookup via ANISE
5. **Horizons Fallback** — If SPICE fails and use_horizons=true, calls ``query_horizons_body()``
6. **Frame Conversion** — Converts heliocentric to observer-relative coordinates
7. **Return** — SkyCoord with proper frame and observer location

Key Files
---------

**src/utils/horizons.rs** (82 lines)

Module for JPL Horizons queries:

.. code-block:: rust

    pub fn query_horizons_body(
        times: &[DateTime<Utc>],
        body_id: i32,
    ) -> Result<Array2<f64>, String>

- Creates Tokio runtime for async execution
- Calls ``rhorizons::ephemeris_vector()`` async function
- Interpolates results to requested times
- Returns (N, 6) array: [x, y, z, vx, vy, vz] in km/s

**src/utils/celestial.rs** (Updated)

Main body lookup function:

.. code-block:: rust

    pub fn calculate_body_by_id_or_name(
        times: &[DateTime<Utc>],
        body_id: i32,
        use_horizons: bool,
    ) -> Result<PositionVelocityData, String>

Flow:
1. Parse body_id from string input
2. Call ``calculate_body_positions_spice_result()`` (SPICE lookup)
3. If fails AND use_horizons=true:
   - Call ``query_horizons_body()``
   - Get Sun position for frame conversion
   - Subtract: heliocentric - sun_geocentric = body_geocentric

**src/ephemeris/ephemeris_common.rs** (Updated)

Trait definitions and implementations:

.. code-block:: rust

    pub trait EphemerisBase {
        fn get_body_impl(
            &self,
            body_id: &str,
            kernel_spec: Option<&str>,
            use_horizons: bool,
        ) -> Result<SkyCoord, String>;
    }

**src/ephemeris/tle_ephemeris.rs** etc (Updated)

All four ephemeris types updated:

.. code-block:: rust

    #[pyo3(signature = (body, kernel_spec=None, use_horizons=false))]
    fn get_body(&self, body: &str, kernel_spec: Option<&str>, use_horizons: bool) -> ...

- TLEEphemeris
- SPICEEphemeris
- GroundEphemeris
- OEMEphemeris (CCSDS)

**rust_ephem/constraints.py** (Updated)

Python constraint system:

.. code-block:: python

    def moving_body_visibility(
        constraint: Constraint,
        ephemeris: Ephemeris,
        body: Optional[Union[int, str]] = None,
        ras: Optional[np.ndarray] = None,
        decs: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        use_horizons: bool = False,  # ← New parameter
    ) -> ConstraintResult:

Dependencies
~~~~~~~~~~~~

**Cargo.toml additions:**

.. code-block:: toml

    rhorizons = "0.5.0"
    tokio = { version = "1", features = ["rt", "macros"] }

**Why these dependencies?**

- **rhorizons** — Async Rust client for JPL Horizons API
  - Handles HTTP requests to NASA servers
  - Returns structured ephemeris data
  - Requires async runtime

- **tokio** — Async runtime (already in project)
  - Enables async/await syntax
  - Manages network I/O
  - Provides runtime for sync-from-async conversion

Implementation Details
----------------------

Async to Sync Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

JPL Horizons queries are inherently async (network I/O), but the Python API expects
synchronous functions. The solution:

.. code-block:: rust

    // Create a single-threaded Tokio runtime
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("Failed to create Tokio runtime: {}", e))?;

    // Execute async code synchronously
    let ephemeris_data = rt.block_on(async {
        rhorizons::ephemeris_vector(body_id, start_time, end_time).await
    });

**Why single-threaded?**
- PyO3 has GIL restrictions
- Single-threaded avoids blocking issues
- Sufficient for I/O-bound network queries

Coordinate Frame Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Horizons returns heliocentric (Sun-centered) positions. We need observer-relative
(geocentric or satellite-relative) positions:

.. code-block:: rust

    // pseudocode
    let body_heliocentric = horizons_result;
    let sun_geocentric = calculate_sun_position();
    let body_geocentric = body_heliocentric - sun_geocentric;

This transformation:
- Moves origin from Sun to Earth/observer
- Enables integration with observer-based constraint system
- Maintains accuracy through vector subtraction

Time Interpolation
~~~~~~~~~~~~~~~~~~~

Horizons may not return data at exactly the requested times. Current implementation
uses nearest-neighbor interpolation:

.. code-block:: rust

    // Find closest ephemeris point to requested time
    let closest_idx = ephemeris_data
        .iter()
        .enumerate()
        .min_by_key(|(_, item)| {
            let diff = (item.time - time).num_seconds().abs();
            diff
        })
        .map(|(idx, _)| idx)?;

**Future improvements:**
- Linear interpolation between points
- Spline fitting for smooth curves
- Configurable interpolation method

Error Handling
~~~~~~~~~~~~~~

The implementation handles several error cases:

.. code-block:: rust

    // Empty time array
    if times.is_empty() {
        return Err("No times provided for Horizons query".to_string());
    }

    // Empty response
    if ephemeris_data.is_empty() {
        return Err(format!(
            "No ephemeris data returned from Horizons for body ID {}",
            body_id
        ));
    }

    // Missing data point
    if let Err(e) = time_lookup {
        return Err("No valid ephemeris data found".to_string());
    }

Testing
-------

Unit Tests
~~~~~~~~~~

The module includes a network-dependent test:

.. code-block:: rust

    #[test]
    #[ignore]  // Ignore by default
    fn test_query_horizons_mars() {
        let times = vec![...];
        let result = query_horizons_body(&times, 499);  // Mars

        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(data.shape(), &[2, 6]);

        // Sanity checks on values
        let pos_mag = (data[[0, 0]].powi(2) + ...).sqrt();
        assert!(pos_mag > 1e8 && pos_mag < 5e8);  // ~1-5 AU in km
    }

Mark tests with ``#[ignore]`` since they require:
- Network access to NASA servers
- API availability
- Valid time range for the body

Python Tests
~~~~~~~~~~~~

Integration tests in ``tests/test_constraints_module.py`` verify:

.. code-block:: python

    def test_moving_body_visibility_with_body():
        # Uses mock DummyEphemeris that bypasses real queries
        timestamps = [...]
        ephem = DummyEphemeris(timestamps)
        constraint = SequenceConstraint([True, False])

        result = moving_body_visibility(
            constraint=constraint,
            ephemeris=ephem,
            body=499
        )

        # Verify constraint evaluation works
        assert len(result.visibility) > 0

Type Hints
----------

Python type stubs (ephemeris.pyi) provide IDE support:

.. code-block:: python

    class Ephemeris:
        def get_body(
            self,
            body: str,
            kernel_spec: Optional[str] = None,
            use_horizons: bool = False,
        ) -> SkyCoord:
            """Get SkyCoord for a body, with optional Horizons fallback."""

Performance Characteristics
----------------------------

Operation Timing
~~~~~~~~~~~~~~~~

::

    SPICE lookup:     ~0.1 ms (cached)
    Horizons query:   ~0.5-2 s (network-dependent)
    Frame conversion: ~1-10 ms
    Constraint eval:  ~1-100 ms

Memory Usage
~~~~~~~~~~~~

.. code-block:: rust

    // query_horizons_body allocates:
    Array2::zeros((n_times, 6))  // ~48 bytes per time point
    // Plus Horizons response data (~1-10 KB for typical queries)

For 10,000 time points: ~500 KB + Horizons response

Scalability Notes
~~~~~~~~~~~~~~~~~

- **Network latency** is the limiting factor for Horizons queries
- **Batch constraint evaluation** is efficient (vectorized in Rust)
- **Multiple body queries** require separate network calls (not batched in Horizons API)
- **Time range size** doesn't significantly impact query time (Horizons computes analytically)

Future Enhancements
-------------------

Potential improvements to the implementation:

1. **Caching** — Store Horizons results locally to avoid repeated network calls
2. **Batch Horizons queries** — Use async to query multiple bodies in parallel
3. **Better interpolation** — Implement linear or spline interpolation
4. **Custom time steps** — Allow specifying Horizons step size
5. **Async API** — Expose async Horizons queries to Python (requires async support in PyO3)
6. **Error recovery** — Retry logic for transient network failures
7. **Horizons caching server** — Local cache to serve multiple processes

Integration with Other Components
----------------------------------

Constraint System
~~~~~~~~~~~~~~~~~

Horizons integration works seamlessly with all constraint types:

.. code-block:: python

    constraint = (
        SunConstraint(min_angle=30) &
        MoonConstraint(min_angle=15) &
        EarthLimbConstraint(min_angle=20)
    )

    result = moving_body_visibility(
        constraint=constraint,
        ephemeris=ephem,
        body="99942",
        use_horizons=True
    )

Ephemeris Types
~~~~~~~~~~~~~~~

All four ephemeris types support Horizons equally:

- **TLEEphemeris** — Satellites with Horizons for external bodies
- **SPICEEphemeris** — Spacecraft with Horizons fallback
- **GroundEphemeris** — Ground station viewing asteroids/comets
- **OEMEphemeris** — CCSDS orbit data with Horizons fallback

Contributing
------------

To extend JPL Horizons integration:

1. **Bug fixes** — Test with ``cargo test --release`` and unit tests
2. **Performance** — Profile with Horizons queries to identify bottlenecks
3. **New features** — Keep backward compatibility (use_horizons defaults to false)
4. **Documentation** — Update both Rust docs (///) and RST guides
5. **Testing** — Add tests with ``#[ignore]`` for network-dependent code

Related Documentation
---------------------

- **User Guide** → :doc:`ephemeris_horizons`
- **Quick Reference** → :doc:`horizons_quickref`
- **FAQ** → :doc:`horizons_faq`
- **Constraint System** → :doc:`planning_constraints`
- **rhorizons** → https://github.com/nyx-space/rhorizons
- **JPL Horizons** → https://ssd.jpl.nasa.gov/horizons/
