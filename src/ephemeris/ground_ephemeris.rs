use ndarray::Array2;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{generate_timestamps, EphemerisBase, EphemerisData};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions::{self, Frame};
use crate::utils::to_skycoord::AstropyModules;

/// Ground-based observatory ephemeris
/// Represents a fixed point on Earth's surface specified by geodetic coordinates
#[pyclass]
pub struct GroundEphemeris {
    latitude: f64,  // degrees
    longitude: f64, // degrees
    height: f64,    // meters above WGS84 ellipsoid
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>,
    polar_motion: bool, // Whether to apply polar motion correction
    // Common ephemeris data
    common_data: EphemerisData,
}

#[pymethods]
impl GroundEphemeris {
    /// Create a new GroundEphemeris for a ground-based observatory
    ///
    /// # Arguments
    /// * `latitude` - Geodetic latitude in degrees (-90 to 90)
    /// * `longitude` - Geodetic longitude in degrees (-180 to 180)
    /// * `height` - Altitude in meters above WGS84 ellipsoid
    /// * `begin` - Start time (Python datetime)
    /// * `end` - End time (Python datetime)
    /// * `step_size` - Time step in seconds
    /// * `polar_motion` - Whether to apply polar motion correction (default: false)
    #[new]
    #[pyo3(signature = (latitude, longitude, height, begin, end, step_size=60, *, polar_motion=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        latitude: f64,
        longitude: f64,
        height: f64,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        polar_motion: bool,
    ) -> PyResult<Self> {
        // Validate latitude and longitude
        if !(-90.0..=90.0).contains(&latitude) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "latitude must be between -90 and 90 degrees",
            ));
        }
        if !(-180.0..=180.0).contains(&longitude) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "longitude must be between -180 and 180 degrees",
            ));
        }

        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size)?;

        // Create the GroundEphemeris object
        let mut ephemeris = GroundEphemeris {
            latitude,
            longitude,
            height,
            itrs: None,
            itrs_skycoord: OnceLock::new(),
            polar_motion,
            common_data: {
                let mut data = EphemerisData::new();
                data.times = Some(times);
                data
            },
        };

        // Pre-compute all frames
        ephemeris.compute_itrs_position()?;
        ephemeris.itrs_to_gcrs()?;
        ephemeris.calculate_sun_moon()?;

        // Note: SkyCoords are now created lazily on first access

        // Return the GroundEphemeris object
        Ok(ephemeris)
    }

    #[getter]
    fn gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_gcrs_pv(py)
    }

    #[getter]
    fn itrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_itrs_pv(py)
    }

    #[getter]
    fn itrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_itrs(py)
    }

    #[getter]
    fn gcrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_gcrs(py)
    }

    #[getter]
    fn earth(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth(py)
    }

    #[getter]
    fn sun(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun(py)
    }

    #[getter]
    fn moon(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon(py)
    }

    #[getter]
    fn sun_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_sun_pv(py)
    }

    #[getter]
    fn moon_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_moon_pv(py)
    }

    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
    }

    #[getter]
    fn obsgeoloc(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_obsgeoloc(py)
    }

    #[getter]
    fn obsgeovel(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_obsgeovel(py)
    }

    /// Get the latitude of the observatory in degrees
    #[getter]
    fn latitude(&self) -> f64 {
        self.latitude
    }

    /// Get the longitude of the observatory in degrees
    #[getter]
    fn longitude(&self) -> f64 {
        self.longitude
    }

    /// Get the height of the observatory in meters
    #[getter]
    fn height(&self) -> f64 {
        self.height
    }

    /// Get position and velocity for any solar system body
    ///
    /// Analogous to astropy's `get_body()` function. Returns position and velocity
    /// of the specified body relative to the observer (ground station).
    ///
    /// # Arguments
    /// * `body` - Body identifier: NAIF ID (as string) or name (e.g., "Jupiter", "mars", "301")
    ///
    /// # Returns
    /// `PositionVelocityData` containing position and velocity in km and km/s
    ///
    /// # Example
    /// ```python
    /// eph = GroundEphemeris(...)
    /// jupiter = eph.get_body_pv("Jupiter")  # By name
    /// mars = eph.get_body_pv("499")  # By NAIF ID
    /// print(jupiter.position)  # Position in km
    /// ```
    fn get_body_pv(&self, py: Python, body: String) -> PyResult<Py<PositionVelocityData>> {
        <Self as EphemerisBase>::get_body_pv(self, py, &body)
    }

    /// Get SkyCoord for any solar system body with observer location set
    ///
    /// Analogous to astropy's `get_body()` function but returns a SkyCoord object.
    /// The returned SkyCoord is in GCRS frame with obsgeoloc and obsgeovel set
    /// to the observer's position.
    ///
    /// # Arguments
    /// * `body` - Body identifier: NAIF ID (as string) or name (e.g., "Jupiter", "mars", "301")
    ///
    /// # Returns
    /// Astropy SkyCoord object in GCRS frame
    ///
    /// # Example
    /// ```python
    /// eph = GroundEphemeris(...)
    /// jupiter = eph.get_body("Jupiter")
    /// # Can now compute separations, altaz coordinates, etc.
    /// altaz = jupiter.transform_to(AltAz(obstime=..., location=...))
    /// ```
    fn get_body(&self, py: Python, body: String) -> PyResult<Py<PyAny>> {
        let modules = AstropyModules::import(py)?;
        <Self as EphemerisBase>::get_body(self, py, &modules, &body)
    }

    /// Get angular radius of the Sun with astropy units
    ///
    /// Returns an astropy Quantity array of angular radii for each timestamp (units: degrees).
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    #[getter]
    fn sun_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius(py)
    }

    /// Get angular radius of the Sun with astropy units
    ///
    /// Returns an astropy Quantity array of angular radii for each timestamp (units: degrees).
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    #[getter]
    fn sun_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_deg(py)
    }

    /// Get angular radius of the Moon with astropy units (degrees)
    ///
    /// Returns an astropy Quantity with units of degrees
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    #[getter]
    fn moon_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius(py)
    }

    /// Get angular radius of the Moon as seen from the ground station (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
    #[getter]
    fn moon_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius_deg(py)
    }

    /// Get angular radius of the Earth with astropy units (degrees)
    ///
    /// Returns an astropy Quantity with units of degrees
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    #[getter]
    fn earth_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius(py)
    }

    /// Get angular radius of the Earth as seen from the ground station (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
    #[getter]
    fn earth_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_deg(py)
    }

    /// Get angular radius of the Sun as seen from the ground station (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    #[getter]
    fn sun_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_rad(py)
    }

    /// Get angular radius of the Moon as seen from the ground station (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    #[getter]
    fn moon_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius_rad(py)
    }

    /// Get angular radius of the Earth as seen from the ground station (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    #[getter]
    fn earth_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_rad(py)
    }

    /// Find the index of the closest timestamp to the given datetime
    ///
    /// Returns the index in the ephemeris timestamp array that is closest to the provided time.
    /// This can be used to index into any of the ephemeris arrays (positions, velocities, etc.)
    ///
    /// # Arguments
    /// * `time` - Python datetime object to find the closest match for
    ///
    /// # Returns
    /// Index of the closest timestamp
    ///
    /// # Example
    /// ```python
    /// from datetime import datetime
    /// eph = GroundEphemeris(...)
    /// target_time = datetime(2024, 1, 15, 12, 0, 0)
    /// idx = eph.index(target_time)
    /// sun_position = eph.sun_pv.position[idx]
    /// ```
    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }
}

impl GroundEphemeris {
    /// Compute ITRS position and velocity for the ground station
    /// Position is computed from geodetic coordinates (lat, lon, alt)
    /// Velocity accounts for Earth's rotation
    fn compute_itrs_position(&mut self) -> PyResult<()> {
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "GroundEphemeris was not properly initialized with times.",
            )
        })?;

        let n_times = times.len();

        // Convert geodetic coordinates to ITRS Cartesian coordinates
        // Using WGS84 ellipsoid parameters
        let lat_rad = self.latitude.to_radians();
        let lon_rad = self.longitude.to_radians();

        // WGS84 parameters
        const A: f64 = 6378.137; // Semi-major axis in km
        const F: f64 = 1.0 / 298.257223563; // Flattening
        const E_SQ: f64 = 2.0 * F - F * F; // First eccentricity squared

        // Radius of curvature in the prime vertical
        let sin_lat = lat_rad.sin();
        let n = A / (1.0 - E_SQ * sin_lat * sin_lat).sqrt();

        // Convert height from meters to km
        let h_km = self.height / 1000.0;

        // ITRS Cartesian coordinates (km)
        let x = (n + h_km) * lat_rad.cos() * lon_rad.cos();
        let y = (n + h_km) * lat_rad.cos() * lon_rad.sin();
        let z = (n * (1.0 - E_SQ) + h_km) * lat_rad.sin();

        // Earth's rotation rate (rad/s)
        const OMEGA_EARTH: f64 = 7.292115e-5;

        // Velocity due to Earth's rotation (km/s)
        // v = omega × r, where omega is along z-axis
        let vx = -OMEGA_EARTH * y;
        let vy = OMEGA_EARTH * x;
        let vz = 0.0;

        // Create array with same position/velocity for all times
        // Shape: (n_times, 6) where columns are [x, y, z, vx, vy, vz]
        let mut itrs_array = Array2::<f64>::zeros((n_times, 6));
        for i in 0..n_times {
            itrs_array[[i, 0]] = x;
            itrs_array[[i, 1]] = y;
            itrs_array[[i, 2]] = z;
            itrs_array[[i, 3]] = vx;
            itrs_array[[i, 4]] = vy;
            itrs_array[[i, 5]] = vz;
        }

        self.itrs = Some(itrs_array);
        Ok(())
    }

    /// Convert ITRS positions to GCRS
    fn itrs_to_gcrs(&mut self) -> PyResult<()> {
        let times = self
            .common_data
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        let itrs_data = self.itrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No ITRS data available. Call compute_itrs_position first.",
            )
        })?;

        // Use generic frame conversion
        let gcrs_array = conversions::convert_frames(
            itrs_data,
            times,
            Frame::ITRS,
            Frame::GCRS,
            self.polar_motion,
        );

        self.common_data.gcrs = Some(gcrs_array);
        Ok(())
    }
}

// Implement the EphemerisBase trait for GroundEphemeris
impl EphemerisBase for GroundEphemeris {
    fn data(&self) -> &EphemerisData {
        &self.common_data
    }

    fn data_mut(&mut self) -> &mut EphemerisData {
        &mut self.common_data
    }

    fn get_itrs_data(&self) -> Option<&Array2<f64>> {
        self.itrs.as_ref()
    }

    fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>> {
        self.itrs_skycoord.get()
    }

    fn set_itrs_skycoord_cache(&self, skycoord: Py<PyAny>) -> Result<(), Py<PyAny>> {
        self.itrs_skycoord.set(skycoord)
    }
}
