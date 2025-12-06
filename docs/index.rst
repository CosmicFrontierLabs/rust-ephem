rust-ephem Documentation
========================

.. image:: _static/logo.png
   :alt: rust-ephem logo
   :align: center
   :width: 300px

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python 3.10 +

.. image:: https://img.shields.io/badge/rust-2021-orange.svg
   :alt: Rust 2021

**High-performance satellite and planetary ephemeris calculations for Python**

``rust-ephem`` is a Python library powered by Rust that provides fast, accurate
ephemeris calculations for satellites, spacecraft, and ground observatories.
It integrates seamlessly with `astropy <https://www.astropy.org/>`_ and achieves
**10-20 meter accuracy** for LEO satellites.

Key Features
------------

🚀 **Performance**
   Up to 84x faster than pure-Python solutions for coordinate conversions.
   Vectorized batch constraint evaluation provides 3-50x speedup over single-target loops.

🎯 **Accuracy**
   10-20 meter position accuracy with UT1 and polar motion corrections.
   Sub-arcsecond Moon positions using SPICE kernels.

🔗 **Astropy Integration**
   Direct ``SkyCoord`` output for GCRS, ITRS, Sun, Moon, and Earth frames.
   Compatible with astropy's coordinate transformation ecosystem.

📡 **Multiple Ephemeris Types**
   - **TLEEphemeris**: Satellite tracking from Two-Line Elements (SGP4)
   - **SPICEEphemeris**: Spacecraft ephemeris from SPICE kernel (SPK) files
   - **GroundEphemeris**: Fixed ground station positions
   - **OEMEphemeris**: CCSDS Orbit Ephemeris Message files

🎛️ **Flexible Constraints**
   Evaluate observational constraints (Sun/Moon avoidance, Earth limb, eclipses)
   with logical operators (AND, OR, NOT, XOR) for observation planning.

Quick Start
-----------

**Installation**

.. code-block:: bash

   pip install rust-ephem

**Basic Usage**

.. code-block:: python

   import rust_ephem
   from datetime import datetime, timezone

   # Define time range
   begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
   end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

   # Create satellite ephemeris from TLE
   ephem = rust_ephem.TLEEphemeris(
       norad_id=25544,  # ISS
       begin=begin,
       end=end,
       step_size=60
   )

   # Get positions as astropy SkyCoord
   satellite = ephem.gcrs  # Satellite position in GCRS frame
   sun = ephem.sun         # Sun position from satellite's perspective
   moon = ephem.moon       # Moon position from satellite's perspective

   # Access raw position/velocity data
   print(f"Position: {ephem.gcrs_pv.position[0]} km")
   print(f"Velocity: {ephem.gcrs_pv.velocity[0]} km/s")

**Constraint Evaluation**

Constraints are created by instantiating constraint classes and can be combined
using logical operators. The example below creates a constraint that requires
a target to be at least 45 degrees from the Sun **or** at least 10 degrees from
the Moon. Python logical operators are overloaded to allow combining
constraints. As we are evaluating constraints, and not visibilities, we
combined constraints with "or" (`|`), if we used "and" (`&`), the target would
only be constrained if it violated both Sun and Moon constraints simultaneously.

.. code-block:: python

   from rust_ephem.constraints import SunConstraint, MoonConstraint

   rust_ephem.ensure_planetary_ephemeris()

   # Create combined constraint, not the use of or operator, this means a
   # target is constrained if it's too close to either the Sun or the Moon.
   constraint = SunConstraint(min_angle=45.0) | MoonConstraint(min_angle=10.0)

   # Evaluate for a target (Crab Nebula)
   result = constraint.evaluate(ephem, target_ra=83.63, target_dec=22.01)

   print(f"Visibility windows: {len(result.visibility)}")
   for window in result.visibility:
       print(f"  {window.start_time} to {window.end_time}")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   frames
   time_systems
   accuracy_precision

.. toctree::
   :maxdepth: 2
   :caption: Ephemeris Generation

   ephemeris_tle
   ephemeris_spice
   ephemeris_ground
   ephemeris_oem
   ephemeris_skycoord
   ephemeris_get_body

.. toctree::
   :maxdepth: 2
   :caption: Observation Planning

   planning_constraints
   planning_visibility

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   constraints_api
   planetary_ephemeris

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
-----

* `GitHub Repository <https://github.com/CosmicFrontierLabs/rust-ephem>`_
* `Issue Tracker <https://github.com/CosmicFrontierLabs/rust-ephem/issues>`_
