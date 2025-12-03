Examples
========

Practical usage examples demonstrating key features of ``rust-ephem``.

.. toctree::
   :maxdepth: 1

   usage_tle
   usage_spice
   usage_ground
   usage_oem
   usage_skycoord
   usage_get_body
   usage_constraints

Getting Started
---------------

All examples assume you have ``rust-ephem`` installed:

.. code-block:: bash

   pip install rust-ephem

For examples using planetary ephemeris data (Sun/Moon positions), you'll need
to initialize the planetary ephemeris first:

.. code-block:: python

   import rust_ephem
   rust_ephem.ensure_planetary_ephemeris()

This automatically downloads and caches the DE440S ephemeris kernel on first use.
