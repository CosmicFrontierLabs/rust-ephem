//! OMM (Orbital Mean-Elements Message) Ephemeris
//!
//! This module provides support for CCSDS Orbital Mean-Elements Messages (OMM)
//! which contain mean orbital elements that can be propagated using SGP4.

use chrono::{Datelike, Timelike};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};

use crate::ephemeris::ephemeris_common::{
    generate_timestamps, EphemerisBase, EphemerisData, SGP4EphemerisBase,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::omm_utils::{fetch_omm_unified, omm_to_elements, FetchedOMM};

#[pyclass]
pub struct OMMEphemeris {
    omm_data: crate::utils::omm_utils::OMMData,
    base: SGP4EphemerisBase, // Common SGP4 ephemeris functionality
}

#[pymethods]
impl OMMEphemeris {
    #[new]
    #[pyo3(signature = (norad_id=None, norad_name=None, begin=None, end=None, step_size=60, *, polar_motion=false, spacetrack_username=None, spacetrack_password=None, _epoch_tolerance_days=None, enforce_source=None, omm=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        norad_id: Option<u32>,
        norad_name: Option<String>,
        begin: Option<&Bound<'_, PyDateTime>>,
        end: Option<&Bound<'_, PyDateTime>>,
        step_size: i64,
        polar_motion: bool,
        spacetrack_username: Option<String>,
        spacetrack_password: Option<String>,
        _epoch_tolerance_days: Option<f64>,
        enforce_source: Option<String>,
        omm: Option<String>,
    ) -> PyResult<Self> {
        // Build credentials using helper function
        let credentials = crate::utils::tle_utils::build_credentials(
            spacetrack_username.as_deref(),
            spacetrack_password.as_deref(),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Determine target epoch from begin time
        let target_epoch =
            begin.and_then(|b| crate::utils::time_utils::python_datetime_to_utc(b).ok());

        // Fetch OMM data or use provided OMM data
        let fetched: FetchedOMM = if let Some(omm_source) = omm {
            // Check if omm_source is JSON data (starts with [ or {)
            let json_content =
                if omm_source.trim().starts_with('[') || omm_source.trim().starts_with('{') {
                    // Direct JSON data
                    omm_source.clone()
                } else if omm_source.starts_with("http://") || omm_source.starts_with("https://") {
                    // Fetch from URL
                    ureq::get(&omm_source)
                        .call()
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed to fetch OMM data from URL: {}",
                                e
                            ))
                        })?
                        .body_mut()
                        .read_to_string()
                        .map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed to read response body: {}",
                                e
                            ))
                        })?
                } else {
                    // Read from file
                    std::fs::read_to_string(&omm_source).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to read OMM file '{}': {}",
                            omm_source, e
                        ))
                    })?
                };

            let data = crate::utils::omm_utils::parse_omm_json(&json_content).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "parse_omm_json failed for OMM data: {}",
                    e
                ))
            })?;
            FetchedOMM { data }
        } else {
            fetch_omm_unified(
                norad_id,
                norad_name.as_deref(),
                target_epoch.as_ref(),
                credentials,
                enforce_source.as_deref(),
            )
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("fetch_omm_unified failed: {}", e))
            })?
        };

        // Convert OMM to SGP4 Elements directly
        omm_to_elements(&fetched.data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("omm_to_elements failed: {}", e))
        })?;

        // Check that begin and end are provided
        let begin = begin.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("begin parameter is required")
        })?;
        let end = end
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("end parameter is required"))?;

        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "generate_timestamps failed (begin/end/step_size): {}",
                e
            ))
        })?;

        let omm_data = fetched.data;

        // Create the OMMEphemeris object
        let mut ephemeris: OMMEphemeris = OMMEphemeris {
            omm_data,
            base: {
                let mut base = SGP4EphemerisBase::new(polar_motion);
                base.common_data.times = Some(times);
                base
            },
        };

        // Pre-compute all frames
        ephemeris.propagate_to_teme().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("propagate_to_teme failed: {}", e))
        })?;
        ephemeris.teme_to_itrs().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("teme_to_itrs failed: {}", e))
        })?;
        ephemeris.teme_to_gcrs().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("teme_to_gcrs failed: {}", e))
        })?;
        ephemeris.calculate_sun_moon().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("calculate_sun_moon failed: {}", e))
        })?;

        // Note: SkyCoords are now created lazily on first access

        // Return the OMMEphemeris object
        Ok(ephemeris)
    }

    /// Get the epoch of the OMM as a Python datetime object
    #[getter]
    fn epoch(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Convert chrono::DateTime<Utc> to Python datetime with UTC timezone
        let epoch = self.omm_data.epoch;

        let dt = pyo3::types::PyDateTime::new(
            py,
            epoch.year(),
            epoch.month() as u8,
            epoch.day() as u8,
            epoch.hour() as u8,
            epoch.minute() as u8,
            epoch.second() as u8,
            epoch.timestamp_subsec_micros(),
            None,
        )?;

        // Get UTC timezone and replace
        let datetime_mod = py.import("datetime")?;
        let utc_tz = datetime_mod.getattr("timezone")?.getattr("utc")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("tzinfo", utc_tz)?;
        let dt_with_tz = dt.call_method("replace", (), Some(&kwargs))?;

        Ok(dt_with_tz.into())
    }

    /// Get the NORAD catalog ID
    #[getter]
    fn norad_cat_id(&self) -> u32 {
        self.omm_data.norad_cat_id
    }

    /// Get the object name
    #[getter]
    fn object_name(&self) -> Option<&str> {
        self.omm_data.object_name.as_deref()
    }

    /// Get the mean motion (revolutions per day)
    #[getter]
    fn mean_motion(&self) -> f64 {
        self.omm_data.mean_motion
    }

    /// Get the eccentricity
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.omm_data.eccentricity
    }

    /// Get the inclination (degrees)
    #[getter]
    fn inclination(&self) -> f64 {
        self.omm_data.inclination
    }

    /// Get the right ascension of ascending node (degrees)
    #[getter]
    fn ra_of_asc_node(&self) -> f64 {
        self.omm_data.ra_of_asc_node
    }

    /// Get the argument of pericenter (degrees)
    #[getter]
    fn arg_of_pericenter(&self) -> f64 {
        self.omm_data.arg_of_pericenter
    }

    /// Get the mean anomaly (degrees)
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.omm_data.mean_anomaly
    }

    /// Get the B* drag term
    #[getter]
    fn bstar(&self) -> f64 {
        self.omm_data.bstar
    }

    /// Get the semimajor axis (km)
    #[getter]
    fn semimajor_axis(&self) -> Option<f64> {
        self.omm_data.semimajor_axis
    }

    /// Get the orbital period (minutes)
    #[getter]
    fn period(&self) -> Option<f64> {
        self.omm_data.period
    }

    /// Get the apoapsis altitude (km)
    #[getter]
    fn apoapsis(&self) -> Option<f64> {
        self.omm_data.apoapsis
    }

    /// Get the periapsis altitude (km)
    #[getter]
    fn periapsis(&self) -> Option<f64> {
        self.omm_data.periapsis
    }

    /// Get the start time of the ephemeris
    #[getter]
    fn begin(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_begin_time(&self.base.common_data.times, py)
    }

    /// Get the end time of the ephemeris
    #[getter]
    fn end(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_end_time(&self.base.common_data.times, py)
    }

    /// Get the step size in seconds
    #[getter]
    fn step_size(&self) -> PyResult<i64> {
        crate::ephemeris::ephemeris_common::get_step_size(&self.base.common_data.times)
    }

    /// Get the epoch of the OMM as a Python datetime object
    #[getter]
    fn omm_epoch(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Convert chrono::DateTime<Utc> to Python datetime with UTC timezone
        let epoch = self.omm_data.epoch;

        let dt = pyo3::types::PyDateTime::new(
            py,
            epoch.year(),
            epoch.month() as u8,
            epoch.day() as u8,
            epoch.hour() as u8,
            epoch.minute() as u8,
            epoch.second() as u8,
            epoch.timestamp_subsec_micros(),
            None,
        )?;

        // Get UTC timezone and replace
        let datetime_mod = py.import("datetime")?;
        let utc_tz = datetime_mod.getattr("timezone")?.getattr("utc")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("tzinfo", utc_tz)?;
        let dt_with_tz = dt.call_method("replace", (), Some(&kwargs))?;

        Ok(dt_with_tz.into())
    }

    /// Get whether polar motion correction is applied
    #[getter]
    fn polar_motion(&self) -> bool {
        self.base.polar_motion
    }

    /// Get the timestamp array
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
    }

    /// Get GCRS position and velocity data
    #[getter]
    fn gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_gcrs_pv(py)
    }

    /// Get ITRS position and velocity data
    #[getter]
    fn itrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_itrs_pv(py)
    }

    /// Get GCRS SkyCoord
    #[getter]
    fn gcrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_gcrs(py)
    }

    /// Get ITRS SkyCoord
    #[getter]
    fn itrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_itrs(py)
    }

    /// Get Earth SkyCoord (for Earth observation)
    #[getter]
    fn earth(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth(py)
    }

    /// Get Sun SkyCoord
    #[getter]
    fn sun(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun(py)
    }

    /// Get Moon SkyCoord
    #[getter]
    fn moon(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon(py)
    }

    /// Get angular radius of the Sun for all times
    #[getter]
    fn sun_angular_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius(py)
    }

    /// Get angular radius of the Moon for all times
    #[getter]
    fn moon_angular_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius(py)
    }

    /// Get angular radius of Earth as seen from satellite for all times
    #[getter]
    fn earth_angular_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius(py)
    }

    /// Get latitude array (degrees)
    #[getter]
    fn latitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_latitude(py)
    }

    /// Get longitude array (degrees)
    #[getter]
    fn longitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_longitude(py)
    }

    /// Get height array (km)
    #[getter]
    fn height(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_height(py)
    }

    /// Convert RA/Dec to Altitude/Azimuth
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn radec_to_altaz(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        use crate::utils::celestial::radec_to_altaz;
        let result = radec_to_altaz(ra_deg, dec_deg, self, time_indices.as_deref());
        Ok(result.into_pyarray(py).into())
    }

    /// Calculate airmass for a target at given RA/Dec
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn airmass(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let result = self.calculate_airmass(ra_deg, dec_deg, time_indices.as_deref())?;
        Ok(result.into_pyarray(py).into())
    }
}

// ===== Propagation methods =====

impl OMMEphemeris {
    /// Propagate the OMM data using SGP4 to generate TEME positions and velocities
    fn propagate_to_teme(&mut self) -> PyResult<()> {
        // Convert OMM data to SGP4 elements
        let elements = omm_to_elements(&self.omm_data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "OMM to elements conversion failed: {}",
                e
            ))
        })?;

        // Use the base implementation
        self.base.propagate_to_teme(&elements)
    }

    /// Convert TEME to ITRS coordinates
    fn teme_to_itrs(&mut self) -> PyResult<()> {
        self.base.teme_to_itrs()
    }

    /// Convert TEME to GCRS coordinates
    fn teme_to_gcrs(&mut self) -> PyResult<()> {
        self.base.teme_to_gcrs()
    }

    /// Calculate Sun and Moon positions
    fn calculate_sun_moon(&mut self) -> PyResult<()> {
        self.base.calculate_sun_moon()
    }
}

// ===== EphemerisBase trait implementation =====

impl EphemerisBase for OMMEphemeris {
    fn data(&self) -> &EphemerisData {
        &self.base.common_data
    }

    fn data_mut(&mut self) -> &mut EphemerisData {
        &mut self.base.common_data
    }

    fn get_itrs_data(&self) -> Option<&Array2<f64>> {
        self.base.itrs.as_ref()
    }

    fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>> {
        self.base.itrs_skycoord.get()
    }

    fn set_itrs_skycoord_cache(&self, skycoord: Py<PyAny>) -> Result<(), Py<PyAny>> {
        self.base.itrs_skycoord.set(skycoord)
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
