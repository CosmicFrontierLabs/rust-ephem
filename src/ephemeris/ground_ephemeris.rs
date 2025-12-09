use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{generate_timestamps, EphemerisBase, EphemerisData};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::config::OMEGA_EARTH;
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

        // Pre-populate geodetic caches with exact input values to preserve precision
        if let Some(times_ref) = ephemeris.common_data.times.as_ref() {
            let n_times = times_ref.len();
            let lat_deg = Array1::from_vec(vec![latitude; n_times]);
            let lon_deg = Array1::from_vec(vec![longitude; n_times]);
            let lat_rad = lat_deg.mapv(|v| v.to_radians());
            let lon_rad = lon_deg.mapv(|v| v.to_radians());
            let h_m = Array1::from_vec(vec![height; n_times]);
            let h_km = h_m.mapv(|v| v / 1000.0);
            let _ = ephemeris.common_data.latitude_deg_cache.set(lat_deg);
            let _ = ephemeris.common_data.longitude_deg_cache.set(lon_deg);
            let _ = ephemeris.common_data.latitude_rad_cache.set(lat_rad);
            let _ = ephemeris.common_data.longitude_rad_cache.set(lon_rad);
            let _ = ephemeris.common_data.height_cache.set(h_m);
            let _ = ephemeris.common_data.height_km_cache.set(h_km);
        }

        // Note: SkyCoords are now created lazily on first access

        // Return the GroundEphemeris object
        Ok(ephemeris)
    }

    // ===== Type-specific getters =====

    /// Get the input latitude in degrees (constructor argument)
    #[getter]
    fn input_latitude(&self) -> f64 {
        self.latitude
    }

    /// Get the input longitude in degrees (constructor argument)
    #[getter]
    fn input_longitude(&self) -> f64 {
        self.longitude
    }

    /// Convert RA/Dec to Altitude/Azimuth for this ground site
    ///
    /// Returns a NumPy array with shape (N, 2) of [altitude_deg, azimuth_deg]
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn radec_to_altaz(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let result = <Self as crate::ephemeris::ephemeris_common::EphemerisBase>::radec_to_altaz(
            self,
            ra_deg,
            dec_deg,
            time_indices.as_deref(),
        );
        Ok(result.into_pyarray(py).into())
    }

    /// Calculate airmass for a target at given RA/Dec
    ///
    /// Airmass represents the relative path length through Earth's atmosphere compared to
    /// zenith observation. Lower values indicate better observing conditions.
    ///
    /// # Arguments
    /// * `ra_deg` - Right ascension in degrees (ICRS/J2000)
    /// * `dec_deg` - Declination in degrees (ICRS/J2000)
    /// * `time_indices` - Optional indices into ephemeris times (default: all times)
    ///
    /// # Returns
    /// List of airmass values:
    /// - 1.0 at zenith (directly overhead)
    /// - ~2.0 at 30° altitude
    /// - ~5.8 at 10° altitude
    /// - Infinity for targets below horizon
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn calculate_airmass(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Vec<f64>> {
        <Self as crate::ephemeris::ephemeris_common::EphemerisBase>::calculate_airmass(
            self,
            ra_deg,
            dec_deg,
            time_indices.as_deref(),
        )
    }

    /// Get the input height in meters (constructor argument)
    #[getter]
    fn input_height(&self) -> f64 {
        self.height
    }

    /// Get whether polar motion correction is applied
    #[getter]
    fn polar_motion(&self) -> bool {
        self.polar_motion
    }

    // ===== Common ephemeris getters (delegating to EphemerisBase trait) =====

    #[getter]
    fn begin(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_begin_time(&self.common_data.times, py)
    }

    #[getter]
    fn end(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_end_time(&self.common_data.times, py)
    }

    #[getter]
    fn step_size(&self) -> PyResult<i64> {
        crate::ephemeris::ephemeris_common::get_step_size(&self.common_data.times)
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
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
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
    fn obsgeoloc(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_obsgeoloc(py)
    }

    #[getter]
    fn obsgeovel(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_obsgeovel(py)
    }

    #[getter]
    fn latitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_latitude(py)
    }

    #[getter]
    fn latitude_deg(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_latitude_deg(py)
    }

    #[getter]
    fn latitude_rad(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_latitude_rad(py)
    }

    #[getter]
    fn longitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_longitude(py)
    }

    #[getter]
    fn longitude_deg(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_longitude_deg(py)
    }

    #[getter]
    fn longitude_rad(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_longitude_rad(py)
    }

    #[getter]
    fn height(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_height(py)
    }

    #[getter]
    fn height_m(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_height_m(py)
    }

    #[getter]
    fn height_km(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        // Override to use constant input height for ground stations
        let times = self.common_data.times.as_ref();
        if times.is_none() {
            return Ok(None);
        }
        let n = times.unwrap().len();
        let arr = ndarray::Array1::from_elem(n, self.height / 1000.0);
        Ok(Some(arr.into_pyarray(py).to_owned().into()))
    }

    #[getter]
    fn sun_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius(py)
    }

    #[getter]
    fn sun_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_deg(py)
    }

    #[getter]
    fn sun_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_rad(py)
    }

    #[getter]
    fn moon_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius(py)
    }

    #[getter]
    fn moon_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius_deg(py)
    }

    #[getter]
    fn moon_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius_rad(py)
    }

    #[getter]
    fn earth_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius(py)
    }

    #[getter]
    fn earth_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_deg(py)
    }

    #[getter]
    fn earth_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_rad(py)
    }

    /// Calculate Moon illumination fraction for all ephemeris times
    ///
    /// Returns the fraction of the Moon's illuminated surface as seen from the
    /// spacecraft observer (0.0 = new moon, 1.0 = full moon).
    ///
    /// # Arguments
    /// * `time_indices` - Optional indices into ephemeris times (default: all times)
    ///
    /// # Returns
    /// List of Moon illumination fractions
    #[pyo3(signature = (time_indices=None))]
    fn moon_illumination(&self, time_indices: Option<Vec<usize>>) -> PyResult<Vec<f64>> {
        EphemerisBase::moon_illumination(self, time_indices.as_deref())
    }

    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }

    #[pyo3(signature = (body, kernel_spec=None))]
    fn get_body_pv(
        &self,
        py: Python,
        body: &str,
        kernel_spec: Option<String>,
    ) -> PyResult<Py<PositionVelocityData>> {
        <Self as EphemerisBase>::get_body_pv(self, py, body, kernel_spec.as_deref())
    }

    #[pyo3(signature = (body, kernel_spec=None))]
    fn get_body(&self, py: Python, body: &str, kernel_spec: Option<String>) -> PyResult<Py<PyAny>> {
        let modules = AstropyModules::import(py)?;
        <Self as EphemerisBase>::get_body(self, py, &modules, body, kernel_spec.as_deref())
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

    fn radec_to_altaz(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<&[usize]>,
    ) -> Array2<f64> {
        crate::utils::celestial::radec_to_altaz(ra_deg, dec_deg, self, time_indices)
    }
}
