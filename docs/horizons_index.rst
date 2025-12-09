.. _horizons-docs-index:

JPL Horizons Documentation Index
=================================

``rust-ephem`` includes comprehensive documentation for JPL Horizons integration.
This page helps you find the right resource for your needs.

Getting Started
---------------

**New to Horizons?** Start here:

1. :doc:`horizons_quickref` — 5-minute quick start with code examples
2. :doc:`ephemeris_horizons` — Comprehensive feature guide

Planning Observations
---------------------

**Planning astronomical observations?** See:

1. :doc:`planning_constraints` — Constraint system with Horizons examples
2. :doc:`planning_visibility` — Visibility windows for moving targets
3. :doc:`horizons_faq` — Common questions and best practices

Body Lookups
------------

**Looking up body positions?** See:

1. :doc:`ephemeris_get_body` — Basic ``get_body()`` usage
2. :doc:`horizons_quickref` — Common body ID reference
3. :doc:`ephemeris_horizons` — Advanced body lookup patterns

Implementation Details
----------------------

**Contributing or extending Horizons?**

1. :doc:`horizons_implementation` — Architecture and code structure
2. Source code: ``src/utils/horizons.rs`` (82 lines)
3. Integration points: ``src/utils/celestial.rs``, ``src/ephemeris/*.rs``

Document Reference
-------------------

Main Horizons Documents
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Document
     - Purpose
     - Audience
   * - :doc:`ephemeris_horizons`
     - Complete guide to JPL Horizons integration
     - All users
   * - :doc:`horizons_quickref`
     - Quick reference with common patterns
     - Quick lookup
   * - :doc:`horizons_faq`
     - Frequently asked questions and best practices
     - Troubleshooting, patterns
   * - :doc:`horizons_implementation`
     - Implementation architecture and design
     - Developers, contributors

Related Documents
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Document
     - Purpose
     - Relevance to Horizons
   * - :doc:`ephemeris_get_body`
     - Body lookup methods
     - Basic ``get_body()`` usage
   * - :doc:`planning_constraints`
     - Constraint system
     - ``moving_body_visibility()`` with Horizons
   * - :doc:`planning_visibility`
     - Visibility window calculations
     - Moving target tracking
   * - :doc:`api`
     - Python API reference
     - Method signatures with ``use_horizons`` parameter

Feature Overview
----------------

Horizons Quick Facts
~~~~~~~~~~~~~~~~~~~~

- **Coverage**: Asteroids, comets, spacecraft, planets, moons
- **Access**: NASA JPL Horizons online service via HTTP API
- **Integration**: Fallback when SPICE kernels lack data
- **Network**: Requires internet connectivity
- **Speed**: ~0.5-2 seconds per query (network-dependent)
- **Accuracy**: km-level for most solar system objects

Key Parameters
~~~~~~~~~~~~~~

All ``get_body()`` variants support:

.. code-block:: python

    get_body(body, kernel_spec=None, use_horizons=False)
    get_body_pv(body, kernel_spec=None, use_horizons=False)

- ``body`` — Name or NAIF ID
- ``kernel_spec`` — Custom SPICE kernel path/URL
- ``use_horizons`` — Enable Horizons fallback

Common Use Cases
~~~~~~~~~~~~~~~~

1. **Asteroid tracking** → :doc:`horizons_quickref` → Example: Apophis
2. **Comet position** → :doc:`ephemeris_horizons` → Comet section
3. **Spacecraft ephemeris** → :doc:`horizons_quickref` → Spacecraft IDs
4. **Observation planning** → :doc:`planning_visibility` → Moving targets
5. **Multi-site tracking** → :doc:`horizons_faq` → Best practices

Performance & Reliability
--------------------------

Performance Tips
~~~~~~~~~~~~~~~~

See :doc:`horizons_faq` sections:
- "Performance Optimization"
- "Best Practices"

Reliability Patterns
~~~~~~~~~~~~~~~~~~~~

See :doc:`horizons_faq` sections:
- "Reliability and Error Handling"
- "Common Pitfalls"

Troubleshooting
~~~~~~~~~~~~~~~

See :doc:`horizons_faq` sections:
- "Common Pitfalls"
- "Troubleshooting" in :doc:`ephemeris_horizons`

API Reference
-------------

Function Signatures
~~~~~~~~~~~~~~~~~~~

``Ephemeris.get_body(body, kernel_spec=None, use_horizons=False) → SkyCoord``
  Get body position as astropy SkyCoord

``Ephemeris.get_body_pv(body, kernel_spec=None, use_horizons=False) → PositionVelocityData``
  Get body position and velocity

``moving_body_visibility(constraint, ephemeris, body=None, ras=None, decs=None, timestamps=None, use_horizons=False) → ConstraintResult``
  Track moving body with constraints

Full signatures in :doc:`api` and :doc:`constraints_api`

Examples by Use Case
--------------------

**Asteroid by NAIF ID**

See :doc:`horizons_quickref` or :doc:`ephemeris_horizons` → Body Identifiers

**Asteroid by Name**

See :doc:`horizons_quickref` → Common Asteroids or :doc:`ephemeris_horizons` → Asteroids

**Comet**

See :doc:`ephemeris_horizons` → Comets section

**Spacecraft**

See :doc:`horizons_quickref` → Spacecraft or :doc:`ephemeris_horizons` → Spacecraft section

**With Constraints**

See :doc:`planning_visibility` → Moving Target Visibility or :doc:`horizons_faq` → Integration Patterns

**Performance Optimization**

See :doc:`horizons_faq` → Performance Optimization or :doc:`ephemeris_horizons` → Performance Considerations

**Error Handling**

See :doc:`horizons_faq` → Common Pitfalls or :doc:`horizons_faq` → Reliability and Error Handling

Integration Examples
--------------------

Single Body Lookup
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    body = ephem.get_body("99942", use_horizons=True)

See :doc:`horizons_quickref` → Quick Start

Multiple Bodies
~~~~~~~~~~~~~~~

.. code-block:: python

    for body_id in ["1", "4", "99942"]:
        position = ephem.get_body(body_id, use_horizons=True)

See :doc:`horizons_faq` → Integration Patterns → Pattern 1

With Visibility Windows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rust_ephem.constraints import moving_body_visibility
    result = moving_body_visibility(
        constraint=constraint,
        ephemeris=ephem,
        body="Apophis",
        use_horizons=True
    )

See :doc:`planning_visibility` → Moving Target Visibility

Multi-Site Tracking
~~~~~~~~~~~~~~~~~~~

See :doc:`horizons_faq` → Integration Patterns → Pattern 1: Multi-observer asteroid tracking

Getting Help
------------

**Quick answers:** :doc:`horizons_quickref`

**Common issues:** :doc:`horizons_faq` → Common Pitfalls

**Best practices:** :doc:`horizons_faq` → Best Practices

**Implementation details:** :doc:`horizons_implementation`

**External resources:**
  - JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
  - NAIF IDs: https://ssd.jpl.nasa.gov/?horizons
  - rhorizons crate: https://crates.io/crates/rhorizons

Document Map
------------

Quick Navigation
~~~~~~~~~~~~~~~~

::

    Getting Started
    ├── horizons_quickref ..................... 5-min quick start
    └── ephemeris_horizons .................... Full guide

    Using Horizons
    ├── ephemeris_get_body .................... Basic lookups
    ├── planning_constraints .................. Constraint integration
    └── planning_visibility ................... Visibility windows

    Reference
    ├── horizons_quickref ..................... Body ID reference
    ├── horizons_faq .......................... FAQs and patterns
    └── api / constraints_api ................. Function signatures

    Development
    └── horizons_implementation ............... Architecture and code

Feedback
--------

For issues or suggestions about JPL Horizons integration:

1. Check the :doc:`horizons_faq` → Common Pitfalls
2. Review the appropriate guide document above
3. Report issues on `GitHub <https://github.com/CosmicFrontierLabs/rust-ephem/issues>`_

Related Topics
--------------

- :doc:`ephemeris_get_body` — General body lookups
- :doc:`ephemeris_skycoord` — SkyCoord output format
- :doc:`frames` — Coordinate frame systems
- :doc:`planning_constraints` — Constraint evaluation
- :doc:`planning_visibility` — Visibility window calculation
