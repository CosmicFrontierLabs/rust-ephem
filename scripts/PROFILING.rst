Constraint profiling scripts
============================

These standalone diagnostics measure constraint lookup, visibility access,
and Moon illumination scaling. Install rust-ephem in the active Python
environment, then run::

    python scripts/profile_in_constraint.py
    python scripts/profile_visibility.py
    python scripts/profile_moon_illumination.py

The scripts print timings and scaling ratios; they are not pass/fail performance
tests. Run comparisons on the same machine and record the installed rust-ephem
revision. Ephemeris initialization may obtain Earth-orientation or kernel data
when the local cache is empty.
