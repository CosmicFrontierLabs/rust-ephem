# rust-ephem

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

`rust-ephem` also has a constraint system, that enables flexible evaluation of
observational constraints for ephemeris planning, including Sun and Moon
proximity, Earth limb avoidance, and generic body proximity. It supports
logical operators (AND, OR, NOT) for combining constraints, with Python
operator overloading (`&`, `|`, `~`) for intuitive composition. Built on
Pydantic models, it allows JSON serialization and direct evaluation against
ephemeris objects for efficient visibility and planning calculations.

## Features

- **TLE Propagation (TLEEphemeris)**: Propagate satellite positions from Two-Line Element (TLE) sets using SGP4
- **SPICE Ephemeris (SPICEEphemeris)**: Query satellite positions from SPICE kernels (SPK files) using ANISE
- **Ground Observatory Ephemeris (GroundEphemeris)**: Calculate positions for ground-based observatories from geodetic coordinates (latitude, longitude, height)
- **Coordinate Transformations**:
  - TEME (True Equator Mean Equinox) - output from SGP4
  - ITRS (International Terrestrial Reference System) - Earth-fixed frame
  - GCRS (Geocentric Celestial Reference System) - celestial reference frame
- **Sun and Moon Positions**: Calculate Sun and Moon positions in GCRS frame
- **Astropy Integration**:
  - Direct SkyCoord properties (`gcrs`, `itrs`, `sun`, `moon`, `earth`) for instant SkyCoord objects
  - `obsgeoloc`, `obsgeovel` properties for GCRS frame construction
  - **84x faster** than manual Python loops for coordinate conversion
- **High Accuracy**:
  - GCRS positions accurate to ~10-20 meters compared to astropy
  - **UT1-UTC corrections** using IERS EOP data for improved Earth rotation accuracy
  - **Polar motion correction** (optional) for additional ~10-20m accuracy improvement
  - Automatically downloads and caches IERS data from JPL
- **Pure Rust**: Uses ERFA (Essential Routines for Fundamental Astronomy) and ANISE for transformations
- **Constraint System**:
  - Flexible constraint evaluation for observational planning
  - Sun/Moon proximity constraints with configurable minimum angles
  - Eclipse detection (umbra and penumbra)
  - Earth limb avoidance constraints
  - Generic body proximity constraints (planets, etc.)
  - Logical operators (AND, OR, NOT) for combining constraints
  - Python operator overloading (`&`, `|`, `~`) for intuitive constraint composition
  - Pydantic-based configuration with JSON serialization support

## Building

### Requirements

- Rust (latest stable)
- Python 3.7+ (for Python bindings)
- maturin (for building Python wheels)

### Build Python Module

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop

# Or build a wheel
maturin build --release
pip install target/wheels/*.whl
```

## Usage

### TLE-Based Ephemeris (TLEEphemeris)

```python
import rust_ephem
from datetime import datetime, timezone

# TLE data for a satellite
tle1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
tle2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Create ephemeris object with time range
begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
step_size = 60  # seconds

ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# Access timestamps
print("Timestamps:", ephem.timestamp)

# Get TEME coordinates
print("TEME Position:", ephem.teme.position)  # km
print("TEME Velocity:", ephem.teme.velocity)  # km/s

# Get GCRS coordinates (recommended for most applications)
print("GCRS Position:", ephem.gcrs.position)  # km
print("GCRS Velocity:", ephem.gcrs.velocity)  # km/s

# Can also get ITRS (Earth-fixed frame) coordinates
print("ITRS Position:", ephem.itrs.position)  # km

# Access Sun and Moon positions in GCRS frame
print("Sun Position:", ephem.sun.position)  # km
print("Sun Velocity:", ephem.sun.velocity)  # km/s
print("Moon Position:", ephem.moon.position)  # km
print("Moon Velocity:", ephem.moon.velocity)  # km/s

# For astropy compatibility: observer geocentric location and velocity
print("Observer Location (obsgeoloc):", ephem.obsgeoloc)  # km
print("Observer Velocity (obsgeovel):", ephem.obsgeovel)  # km/s
```

### SPICE Kernel-Based Ephemeris (SPICEEphemeris)

For high-precision ephemeris using SPICE kernels:

```python
import rust_ephem
from datetime import datetime, timezone

# Define time range
begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
step_size = 60  # seconds

# Create SPICE ephemeris object
# - spk_path: Path to SPK (SPICE kernel) file
# - naif_id: NAIF ID of the target body (e.g., -999 for a spacecraft)
# - begin/end/step_size: Time range to compute ephemeris
# - center_id: NAIF ID of the observer (optional, default: 399 for Earth)
ephem = rust_ephem.SPICEEphemeris(
    spk_path="path/to/spacecraft.bsp",
    naif_id=-999,  # Spacecraft NAIF ID
    begin=begin,
    end=end,
    step_size=step_size
    # center_id=399  # Optional: Observer center (default is Earth/399)
)

# Access is identical to TLEEphemeris
print("GCRS Position:", ephem.gcrs.position)  # km
print("GCRS Velocity:", ephem.gcrs.velocity)  # km/s

# Sun and Moon positions
print("Sun Position:", ephem.sun.position)  # km
print("Moon Position:", ephem.moon.position)  # km

# Observer location and velocity (for astropy compatibility)
print("Observer Location (obsgeoloc):", ephem.obsgeoloc)  # km
print("Observer Velocity (obsgeovel):", ephem.obsgeovel)  # km/s
```

**Note**: You need to obtain SPICE kernel files (SPK/BSP format) for your spacecraft. Common sources:

- NASA Navigation and Ancillary Information Facility (NAIF): <https://naif.jpl.nasa.gov/>
- Nyx Space public data: <http://public-data.nyxspace.com/anise/>

#### SPK preload and auto-download (central Almanac for planetary bodies)

To avoid repeated file loads for Sun/Moon/planet calculations and to make it easy to fetch kernels when missing, the library provides a central in-memory Almanac (for planetary ephemeris) and a few helper functions exposed to Python:

- `rust_ephem.init_planetary_ephemeris(path: str)` — preload a local SPK and initialize the central Almanac
- `rust_ephem.download_planetary_ephemeris(url: str, dest: str)` — download an SPK to disk
- `rust_ephem.ensure_planetary_ephemeris(path: str, download_if_missing: bool, spk_url: Optional[str])` — ensure the default SPK path (from `DE440S_PATH`, usually `test_data/de440s.bsp`) exists and initialize the central Almanac (used by Moon calculations); optionally download if missing
- `rust_ephem.is_planetary_ephemeris_initialized() -> bool` — check whether the central Almanac is loaded

You can pick any of these flows:

```python
# 1) Preload from a local file
rust_ephem.init_planetary_ephemeris("test_data/de440s.bsp")

# 2) Download then preload
rust_ephem.download_planetary_ephemeris("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp", "test_data/de440s.bsp")
rust_ephem.init_planetary_ephemeris("test_data/de440s.bsp")

# 3) Ensure default SPK in a single call (downloads if missing)
rust_ephem.ensure_planetary_ephemeris(download_if_missing=True, spk_url=None)  # uses default path and downloads if missing

```

Note: `SPICEEphemeris` always uses a spacecraft-specific Almanac loaded from its `spk_path` and does not use the global Almanac. The global Almanac is used for high-precision Sun/Moon and other supported planetary body calculations.

When a central Almanac is initialized, the planetary calculators use it automatically, avoiding repeated SPK loads. `SPICEEphemeris` always uses a spacecraft-specific Almanac loaded from its `spk_path` and does not use the global Almanac.

### Astropy-like body lookup: get_body_pv() and get_body()

The ephemeris classes expose methods analogous to astropy's `get_body`:

- `get_body_pv(body: str) -> PositionVelocityData` returns position/velocity of a solar-system body relative to the observer (spacecraft or ground station) in the GCRS frame.
- `get_body(body: str) -> SkyCoord` returns an astropy SkyCoord with the observer location set (obsgeoloc/obsgeovel) in the GCRS frame.

Body can be specified by name (case-insensitive) or NAIF ID string. Examples of valid identifiers:

- Names: `"Sun"`, `"Moon"`, `"Luna"`, `"Earth"`, `"Mars"`, `"Jupiter barycenter"`
- NAIF IDs as strings: `"10"` (Sun), `"301"` (Moon), `"399"` (Earth), `"5"` (Jupiter barycenter)

Requirements:

- Call `rust_ephem.ensure_planetary_ephemeris()` once per session to load the planetary SPK (e.g., de440s.bsp) into a central Almanac.
- Kernel coverage matters:
  - The small kernel `de440s.bsp` includes Sun (10), Moon (301), Earth (399), and the planetary barycenters (1..9). It does NOT include individual planet centers like `599` (Jupiter) or `499` (Mars).
  - To query planet-center IDs (e.g., `599` Jupiter), use a full kernel such as `de440.bsp` and initialize it with `ensure_planetary_ephemeris(path=...)`.

Examples

TLEEphemeris

```python
import rust_ephem
from datetime import datetime, timezone

rust_ephem.ensure_planetary_ephemeris()  # loads/caches de440s by default

tle1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
tle2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"
begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
eph = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=1800)

# Bodies by name or NAIF ID (strings)
sun = eph.get_body_pv("Sun")       # by name
moon = eph.get_body_pv("301")      # by NAIF ID
sun_sc = eph.get_body("Sun") # SkyCoord with observer location set
```

GroundEphemeris

```python
rust_ephem.ensure_planetary_ephemeris()
obs = rust_ephem.GroundEphemeris(19.8207, -155.4681, 4207, begin, end, step_size=1800)
moon = obs.get_body_pv("moon")
moon_sc = obs.get_body("Moon")
```

SPICEEphemeris (spacecraft as observer)

```python
rust_ephem.ensure_planetary_ephemeris()
spk_path = "path/to/spacecraft.bsp"  # your spacecraft kernel
eph = rust_ephem.SPICEEphemeris(spk_path, naif_id=301, begin=begin, end=end, step_size=1800)
sun = eph.get_body_pv("Sun")
sun_sc = eph.get_body("Sun")
```

Tip: With `de440s.bsp`, use planetary barycenter identifiers (names like "Jupiter barycenter" or IDs `"5"`) if you need giant-planet directions without the full kernel.

### Ground Observatory Ephemeris (GroundEphemeris)

For ground-based observatories at fixed locations on Earth:

```python
import rust_ephem
from datetime import datetime, timezone

# Define observatory location (e.g., Mauna Kea)
latitude = 19.8207   # degrees N
longitude = -155.468 # degrees W
height = 4205.0    # meters above WGS84 ellipsoid

# Define time range
begin = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
step_size = 60  # seconds

# Create GroundEphemeris object
obs = rust_ephem.GroundEphemeris(
    latitude, longitude, height,
    begin, end, step_size
)

# Access observatory coordinates
print(f"Latitude: {obs.latitude}°")
print(f"Longitude: {obs.longitude}°")
print(f"Altitude: {obs.height} m")

# Get ITRS position (Earth-fixed frame)
print("ITRS Position:", obs.itrs.position)  # km
print("ITRS Velocity:", obs.itrs.velocity)  # km/s (due to Earth rotation)

# Get GCRS position (inertial frame)
print("GCRS Position:", obs.gcrs.position)  # km
print("GCRS Velocity:", obs.gcrs.velocity)  # km/s

# Get Sun and Moon positions as seen from observatory
print("Sun Position:", obs.sun.position)  # km
print("Moon Position:", obs.moon.position)  # km

# Access SkyCoord objects
obs_skycoord = obs.gcrs
sun_skycoord = obs.sun
moon_skycoord = obs.moon
```

**Use cases**:

- Ground-based telescope pointing calculations
- Satellite visibility from specific observatories
- Sun/Moon position calculations for observation planning
- Atmospheric refraction modeling (future enhancement)

See `GROUND_EPHEMERIS.md` for detailed documentation and implementation details.

### Using with Astropy

#### Fast Method: Direct SkyCoord Conversion (Recommended)

The `gcrs` property provides the fastest way to get astropy SkyCoord objects (**84x faster** than manual loops):

```python
import rust_ephem
from datetime import datetime, timezone

# Create ephemeris
tle1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
tle2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 14, 1, 0, 0, tzinfo=timezone.utc)
step_size = 60  # 1 minute

ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# Get SkyCoord directly - includes positions, velocities, and GCRS frame
skycoord = ephem.gcrs
print(f"Satellite positions: {skycoord}")
print(f"First position: {skycoord[0].cartesian.xyz}")
print(f"First velocity: {skycoord[0].velocity.d_xyz}")

# Get ITRS (Earth-fixed) SkyCoord
itrs_skycoord = ephem.itrs
print(f"ITRS positions: {itrs_skycoord}")
print(f"First ITRS position: {itrs_skycoord[0].cartesian.xyz}")
```

#### Manual Method: Using obsgeoloc and obsgeovel

For more control, you can use `obsgeoloc` and `obsgeovel` properties:

```python
from astropy.coordinates import GCRS, SkyCoord, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u

# Create ephemeris
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# Manual vectorized creation (still much faster than loops)
positions = ephem.obsgeoloc * u.km
velocities = ephem.obsgeovel * u.km / u.s
times = Time(ephem.timestamp, scale='utc')

# Create CartesianDifferential for velocities
cart_diff = CartesianDifferential(
    d_x=velocities[:, 0],
    d_y=velocities[:, 1],
    d_z=velocities[:, 2]
)

# Create CartesianRepresentation with positions and velocities
cart_rep = CartesianRepresentation(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    differentials=cart_diff
)

# Create SkyCoord
skycoord = SkyCoord(cart_rep, frame=GCRS(obstime=times))
```

#### Sun and Moon SkyCoord

Get Sun and Moon positions as SkyCoord objects with spacecraft position/velocity as observer location:

```python
# Create ephemeris
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# Get Sun as SkyCoord (from spacecraft perspective)
sun_skycoord = ephem.sun
print(f"Sun position: {sun_skycoord[0].cartesian.xyz}")
print(f"Spacecraft observing from: {sun_skycoord.frame.obsgeoloc[0]}")

# Get Moon as SkyCoord (from spacecraft perspective)
moon_skycoord = ephem.moon
print(f"Moon position: {moon_skycoord[0].cartesian.xyz}")
print(f"Spacecraft observing from: {moon_skycoord.frame.obsgeoloc[0]}")
```

#### Earth SkyCoord

Get Earth position as a SkyCoord (Earth relative to spacecraft):

```python
# Create ephemeris
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)

# Get Earth as SkyCoord (Earth position relative to spacecraft)
earth_skycoord = ephem.earth
print(f"Earth position: {earth_skycoord[0].cartesian.xyz}")

# The Earth position is the negative of the spacecraft GCRS position
# This represents the Earth's position as seen from the spacecraft
```

The `sun`, `moon`, and `earth` properties automatically set:

- `obsgeoloc`: Spacecraft GCRS position (observer location)
- `obsgeovel`: Spacecraft GCRS velocity (observer velocity)

This is useful for calculating Sun/Moon angles, eclipse predictions, and other observations from the spacecraft.

### Moon Position Accuracy

The library provides two methods for calculating Moon positions with different accuracy levels:

#### Built-in: Meeus Algorithm (~32 arcminutes)

The built-in method uses the Meeus Chapter 47 algorithm for Moon positions. This provides:

- **Accuracy**: ~32 arcminutes (0.5 degrees)
- **Sufficient for**: Spacecraft applications, general astronomy
- **No external data required**: Works without SPICE kernels

```python
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
moon_pos = ephem.moon.position  # Meeus algorithm, ~32 arcmin accuracy
```

#### SPICE Kernel: JPL DE440/DE441 (<1 arcsecond)

Uses JPL's high-precision ephemeris data via ANISE/SPICE:

- **Accuracy**: Sub-arcsecond (JPL-quality ephemeris)
- **Required for**: Scientific research, high-precision work
- **Requires**: DE440 or DE441 SPICE kernel file

To use SPICE based planetary ephemeris requires the download of the
`de440s.bsp` file. This can be done in several ways. Firstly if the file is
already available on disk, you can load it with the `init_planetary_ephemeris()`
function, passing the path to the file:

```python
rust_ephem.init_planetary_ephemeris("test_data/de440s.bsp")
```

If you don't have a local copy, `rust_ephem` provides a method to download it
for you:

```python
rust_ephem.download_planetary_ephemeris("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp", "test_data/de440s.bsp")
rust_ephem.init_planetary_ephemeris("test_data/de440s.bsp")
```

Alternatively `rust_ephem` module provides a way to ensure that the file
exists locally, using the `ensure_planetary_ephemeris()` function.

**Recommendation**: Use `SPICEEphemeris` with DE440 kernels when high-precision Moon positions are required for scientific applications.

### Constraint Calculations

This library includes a flexible constraint-evaluation system for computing when
observational constraints (Sun/Moon proximity, Earth limb avoidance, eclipses,
and logical combinations) are violated for a given ephemeris.

Key Python exports (from `rust_ephem`):

- `Constraint` (class): factory methods and evaluation helpers.
- `ConstraintResult`, `ConstraintViolation`: structured result objects returned by evaluations.

Pydantic-based constraint models are provided in `rust_ephem.constraints`. These
models support Python operator combinators for building complex constraints:

- `SunConstraintConfig(min_angle=45.0)` — Sun proximity constraint
- `MoonConstraintConfig(min_angle=10.0)` — Moon proximity constraint
- `EclipseConstraintConfig(umbra_only=True)` — Eclipse constraint
- `EarthLimbConstraintConfig(min_angle=28.0)` — Earth limb avoidance
- `BodyConstraintConfig(body="Mars", min_angle=15.0)` — Generic body proximity
- `AndConstraintConfig(constraints=[...])` — Logical AND
- `OrConstraintConfig(constraints=[...])` — Logical OR
- `NotConstraintConfig(constraint=...)` — Logical NOT

Operator-based composition (recommended):

- `&` — logical AND
- `|` — logical OR
- `~` — logical NOT (unary)

The Pydantic models support these operators directly, allowing expressions such as:

```python
from rust_ephem.constraints import SunConstraintConfig, MoonConstraintConfig, EclipseConstraintConfig

# Build a compound constraint using operators
constraint = SunConstraintConfig(min_angle=45.0) & MoonConstraintConfig(min_angle=10.0) & ~EclipseConstraintConfig(umbra_only=True)
```

Each constraint model has an `evaluate()` method that lazily creates the corresponding
Rust constraint and evaluates it:

```python
import rust_ephem
from datetime import datetime, timezone
from rust_ephem.constraints import SunConstraintConfig, MoonConstraintConfig, EclipseConstraintConfig

rust_ephem.ensure_planetary_ephemeris()

tle1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
tle2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
begin = datetime(2024,1,1,tzinfo=timezone.utc)
end = datetime(2024,1,2,tzinfo=timezone.utc)
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, 300)

# Build and evaluate constraint directly
constraint = SunConstraintConfig(min_angle=45.0) & MoonConstraintConfig(min_angle=10.0)
result = constraint.evaluate(ephem, 83.6333, 22.0145)  # target RA/Dec in degrees
print(result)
```

For advanced usage, constraints can be serialized to JSON and loaded via `rust_ephem.Constraint.from_json()`:

```python
# Serialize to JSON
json_str = constraint.model_dump_json()

# Load and evaluate via Rust
rust_constraint = rust_ephem.Constraint.from_json(json_str)
result = rust_constraint.evaluate(ephem, 83.6333, 22.0145)
```

JSON format examples:

- `{"type": "sun", "min_angle": 45.0}`
- `{"type": "moon", "min_angle": 10.0}`
- `{"type": "eclipse", "umbra_only": true}`
- `{"type": "and", "constraints": [{"type": "sun", "min_angle": 45.0}, {"type": "moon", "min_angle": 10.0}]}`
- `{"type": "or", "constraints": [...]}`
- `{"type": "not", "constraint": {"type": "eclipse", "umbra_only": true}}`

This functionality is demonstrated in the `example_code/demo_constraints*.py` scripts.

## Testing

### Rust Tests

```bash
cargo test
```

### Python Integration Tests

The Python integration test compares our GCRS implementation against astropy:

```bash
# Requires astropy
pip install astropy

# Build and install the module
maturin build --release
pip install target/wheels/*.whl --force-reinstall

# Run integration test
python tests/integration_test_gcrs.py
```

Expected output: All tests pass with position errors < 100 meters.

## Coordinate Frames

- **TEME (True Equator Mean Equinox)**: The output frame from SGP4. Based on the true equator and mean equinox of date. Most directly compatible with TLE data.

- **ITRS (International Terrestrial Reference System)**: Earth-fixed coordinate system. Rotates with the Earth. Useful for ground station coordinates and geographic calculations.

- **GCRS (Geocentric Celestial Reference System)**: Modern celestial reference frame, essentially ICRS (International Celestial Reference System) centered at Earth. Does not rotate with Earth. Preferred for most astronomical calculations. Uses proper precession, nutation, and frame bias corrections.

## Accuracy

The TEME to GCRS transformation achieves position accuracy of approximately **20 meters** compared to astropy's implementation. This is achieved using:

1. **UT1-UTC corrections** using IERS EOP data (automatic download from JPL)
2. Proper precession-nutation matrix from ERFA (`pn_matrix_06a`)
3. Leap second corrections for accurate UTC to TT conversions
4. IAU 2006 precession model

### Time Scale Accuracy

As of the current version, the library uses **accurate leap second data** embedded directly in the module:

- **Embedded data**: 28 leap seconds from 1972 to 2017 compiled into the binary
- **Historical accuracy**: Correct TT-UTC offsets for all dates from 1972 onwards
- **Zero dependencies**: No need to download external files or network access
- **Sub-microsecond accuracy**: Matches astropy time conversions to within 0.014ms

Example of accuracy improvement:

- Year 2000: Old fixed offset had **5.0 second error**, now correct
- Year 1990: Old fixed offset had **12.0 second error**, now correct
- Year 2017+: Matches previous fixed offset (69.184 seconds)

The leap second data is always available and requires no initialization:

```python
import rust_ephem
from datetime import datetime, timezone

# Get TAI-UTC offset for any date since 1972
dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
tai_utc = rust_ephem.get_tai_utc_offset(dt)  # Returns 32.0 seconds
```

### UT1-UTC Corrections for Earth Rotation

The library now includes **automatic UT1-UTC corrections** using IERS Earth Orientation Parameters (EOP) data from JPL. This dramatically improves accuracy from ~100m to ~20m by properly accounting for Earth rotation irregularities.

**Features:**

- **Automatic data download**: IERS EOP2 data downloaded from JPL on first use
- **Data coverage**: ~1 year historical + ~6 months predicted (IERS EOP2 "short" file)
- **Offline fallback**: Returns 0.0 if data unavailable (graceful degradation)
- **Thread-safe caching**: Data persists across multiple ephemeris calculations

**Usage:**

```python
import rust_ephem
from datetime import datetime, timezone

# Check if UT1 provider is available
is_available = rust_ephem.is_ut1_available()
print(f"UT1 provider available: {is_available}")

# Get UT1-UTC offset for a specific date
dt = datetime(2025, 11, 11, tzinfo=timezone.utc)
ut1_utc = rust_ephem.get_ut1_utc_offset(dt)  # Returns offset in seconds
print(f"UT1-UTC offset: {ut1_utc:.6f} seconds")

# UT1 corrections are automatically applied in TLEEphemeris and GroundEphemeris
ephem = rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size)
# GCRS coordinates now include UT1-UTC corrections for ~20m accuracy
```

**Accuracy Impact:**

- **Without UT1**: ~100-400m error due to Earth rotation timing uncertainty
- **With UT1**: ~20m error (5x improvement)
- **UT1-UTC range**: ±0.9 seconds (updated daily by IERS)
- **Position error**: ~465 m/s × UT1-UTC offset at equator

The UT1-UTC data is automatically downloaded from `https://eop2-external.jpl.nasa.gov/eop2/latest_eop2.short` and cached locally by the [hifitime](https://github.com/nyx-space/hifitime) library. No manual configuration required.

### Polar Motion Correction (Optional)

As of the current version, polar motion correction is available as an optional parameter for all ephemeris classes. Polar motion describes the movement of Earth's rotation axis relative to Earth's crust, typically causing position corrections of 10-20 meters.

**Usage:**

```python
# Enable polar motion correction (disabled by default for backward compatibility)
ephem = rust_ephem.TLEEphemeris(
    tle1, tle2, begin, end, step_size,
    polar_motion=True  # Optional keyword argument
)

# Also available for SPICE and Ground ephemeris
spice_ephem = rust_ephem.SPICEEphemeris(
    spk_path, naif_id, begin, end, step_size,
    polar_motion=True
)

ground_ephem = rust_ephem.GroundEphemeris(
    latitude, longitude, height, begin, end, step_size,
    polar_motion=True
)
```

**Polar Motion Functions:**

The library also exposes EOP (Earth Orientation Parameters) functions for direct access:

```python
import rust_ephem
from datetime import datetime, timezone

# Initialize EOP provider (downloads IERS data if needed)
rust_ephem.init_eop_provider()

# Check if EOP data is available
if rust_ephem.is_eop_available():
    # Get polar motion values for a specific time
    dt = datetime(2024, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
    xp, yp = rust_ephem.get_polar_motion(dt)  # Returns values in arcseconds
    print(f"Polar motion: xp = {xp:.6f} arcsec, yp = {yp:.6f} arcsec")
```

**EOP Data:**

- Polar motion data (xp, yp) is automatically downloaded from JPL's EOP2 service
- Data is cached locally with configurable TTL (default: 24 hours)
- Falls back to zero polar motion if data is unavailable
- Cache location: `$HOME/.cache/rust_ephem/latest_eop2.short`
- Environment variables:
  - `RUST_EPHEM_EOP_CACHE`: Custom cache directory
  - `RUST_EPHEM_EOP_CACHE_TTL`: Cache time-to-live in seconds

**Accuracy Impact:**

- Polar motion correction: ~10-20m improvement
- Total error with UT1 + polar motion: ~10-20m (best achievable with SGP4)

### Remaining Error Sources

The remaining ~10-20m position error is primarily due to:

- GCRS/ITRS frame transformation approximations
- SGP4 propagation model limitations

## License

[Add your license here]

## Contributing

Contributions welcome! Please open an issue or pull request.
