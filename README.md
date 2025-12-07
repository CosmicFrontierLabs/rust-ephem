<p align="center">
  <img src="docs/_static/logo.png" alt="rust-ephem logo" width="300">
</p>

# rust-ephem

[![PyPI version](https://img.shields.io/pypi/v/rust-ephem.svg)](https://pypi.org/project/rust-ephem/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rust-ephem.readthedocs.io/en/latest/)


**High-performance satellite and planetary ephemeris calculations for Python, powered by Rust.**

`rust-ephem` is a Rust library with Python bindings for high-performance
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

`rust-ephem` outputs ephemerides as `astropy` `SkyCoord` objects, eliminating
manual conversions and enabling seamless integration with astropy-based
workflows. By default, it includes Sun and Moon positions in `SkyCoord` with observer
location and velocity, correctly handling motion effects like Moon parallax in
LEO spacecraft. It also supports ephemerides for other solar system bodies.

`rust-ephem` also has a constraint system that enables flexible evaluation of
observational constraints for ephemeris planning, including Sun and Moon
proximity, Earth limb avoidance, and generic body proximity. It supports
logical operators (AND, OR, NOT, XOR) for combining constraints, with Python
operator overloading (`&`, `|`, `~`, `^`) for intuitive composition. Built on
Pydantic models, it allows JSON serialization and direct evaluation against
ephemeris objects for efficient visibility and planning calculations.

## Features

- **Rust for Speed**: Full featured Python module built on a Rust core for
  maximum efficiency
- **Multiple Ephemeris Types**: TLE (SGP4), SPICE kernels, ground
  observatories, CCSDS OEM files supported
- **Coordinate Frames**: TEME, ITRS, GCRS with automatic transformations
- **Astropy Integration**: Direct `SkyCoord` output for satellite, Sun, Moon,
  Earth and other solar system body positions
- **High Accuracy**: UT1-UTC and polar motion corrections using IERS EOP data
- **Constraint System**: Calculate target visibility with Sun/Moon avoidance,
  Earth limb, eclipses with logical operators (`&`, `|`, `~`)
- **Type Support**: strong type support for use with
  [mypy](https://mypy-lang.org), [pyright](https://github.com/microsoft/pyright) etc.


## Installation

```bash
pip install rust-ephem
```

Note that if a binary wheel for your operating system dos not exist on PyPI,
this will build from source. This will require your computer to have a valid
[Rust installation](https://rustup.rs), and other dependencies may be required.
If you require a specific architecture, please put in an [Issue](https://github.com/CosmicFrontierLabs/rust-ephem/issues/new).

## Quick Start

### Satellite Ephemeris from TLE

This example generates an ephemeris for the ISS from a TLE. The TLE is
downloaded automatically (and cached) from [Celestrak](https://celestrak.org).

```python
import rust_ephem
from datetime import datetime, timezone

# Define time range
begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

# Create ephemeris from NORAD ID (fetches from Celestrak)
ephem = rust_ephem.TLEEphemeris(
    norad_id=25544,  # ISS
    begin=begin,
    end=end,
    step_size=60
)

# Access positions as astropy SkyCoord
satellite = ephem.gcrs      # Satellite in GCRS frame
sun = ephem.sun             # Sun position
moon = ephem.moon           # Moon position

# Or access raw position/velocity data (km, km/s)
print(f"Position: {ephem.gcrs_pv.position[0]}")
print(f"Velocity: {ephem.gcrs_pv.velocity[0]}")
```

### Ground Observatory

Ground-based observatories can also have an ephemeris generated for them. This
is useful if you need a one-system-fits all approach that includes both space
and ground based telescopes.

```python
import rust_ephem
from datetime import datetime, timezone

begin = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

# Mauna Kea Observatory
obs = rust_ephem.GroundEphemeris(
    latitude=19.8207,
    longitude=-155.468,
    height=4205.0,  # meters
    begin=begin,
    end=end,
    step_size=60
)

# Sun and Moon positions from observatory
sun = obs.sun
moon = obs.moon
```

### Target Visibility

`rust-ephem` can calculate visibilities for celestial targets. It does this
by defining constraints, which are `True` when a target cannot be observed.
These constraints can be combined using logical operators, and then evaluated
using the telescope Ephemeris. Not the `|` (or) operator is used here to
combine the Sun and Moon constraint, as if we used `&` (and) the constraint
would only be true if the target was too close to the Sun _and_ Moon.

```python
import rust_ephem
from rust_ephem.constraints import SunConstraint, MoonConstraint

# Initialize planetary ephemeris (required for constraints)
rust_ephem.ensure_planetary_ephemeris()

# Create combined constraint: avoid Sun within 45° OR Moon within 10°
constraint = SunConstraint(min_angle=45.0) | MoonConstraint(min_angle=10.0)

# Evaluate against ephemeris for a target (Crab Nebula)
result = constraint.evaluate(ephem, target_ra=83.63, target_dec=22.01)

# Get visibility windows
for window in result.visibility:
    print(f"{window.start_time} to {window.end_time}")
```

## TLE Sources

`rust-ephem` supports multiple ways to fetch TLE data for the `TLEEphemeris` class:

```python
# Direct TLE strings
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# From file path
ephem = rust_ephem.TLEEphemeris(tle="path/to/satellite.tle", begin=begin, end=end)

# From URL (cached for 24 hours)
ephem = rust_ephem.TLEEphemeris(tle="https://celestrak.org/...", begin=begin, end=end)

# From NORAD ID (Celestrak, or Space-Track.org with credentials)
ephem = rust_ephem.TLEEphemeris(norad_id=25544, begin=begin, end=end)

# From satellite name
ephem = rust_ephem.TLEEphemeris(norad_name="ISS", begin=begin, end=end)

# Using fetch_tle
tle = fetch_tle(norad_id=25544)
ephem = rust_ephem.TLEEphemeris(tle=tle, begin=begin, end=end)
```

For Space-Track.org integration, set credentials via environment variables or `.env` file:

```bash
SPACETRACK_USERNAME=your_username
SPACETRACK_PASSWORD=your_password
```

## Documentation

For comprehensive documentation including:

- **[API Reference](https://rust-ephem.readthedocs.io/en/latest/api.html)** - Complete class and function documentation
- **[TLE Ephemeris Guide](https://rust-ephem.readthedocs.io/en/latest/ephemeris_tle.html)** - Detailed TLE usage and Space-Track integration
- **[SPICE Ephemeris Guide](https://rust-ephem.readthedocs.io/en/latest/ephemeris_spice.html)** - Using SPICE kernels
- **[Constraint System](https://rust-ephem.readthedocs.io/en/latest/planning_constraints.html)** - Observation planning with constraints
- **[Coordinate Frames](https://rust-ephem.readthedocs.io/en/latest/frames.html)** - TEME, ITRS, GCRS explained
- **[Accuracy & Precision](https://rust-ephem.readthedocs.io/en/latest/accuracy_precision.html)** - Time corrections and error analysis

Visit **[rust-ephem.readthedocs.io](https://rust-ephem.readthedocs.io/en/latest/)**

## Building from Source

A full Rust toolchain should be installed.

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop

# Or build a release wheel
maturin build --release
pip install target/wheels/*.whl
```

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or pull request.
