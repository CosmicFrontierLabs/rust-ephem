//! OMM (Orbital Mean-Elements Message) Ephemeris
//!
//! This module provides support for CCSDS Orbital Mean-Elements Messages (OMM)
//! which contain mean orbital elements that can be propagated using SGP4.

use chrono::{Datelike, Timelike};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use sgp4::Constants;
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{generate_timestamps, EphemerisBase, EphemerisData};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::omm_utils::{fetch_omm_unified, omm_to_elements, FetchedOMM};

#[pyclass]
pub struct OMMEphemeris {
    omm_data: crate::utils::omm_utils::OMMData,
    elements: sgp4::Elements, // SGP4 elements for direct propagation
    teme: Option<Array2<f64>>,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>, // Lazy-initialized cached SkyCoord object for ITRS
    polar_motion: bool,                 // Whether to apply polar motion correction
    // Common ephemeris data
    common_data: EphemerisData,
}

#[pymethods]
impl OMMEphemeris {
    #[new]
    #[pyo3(signature = (norad_id=None, norad_name=None, begin=None, end=None, step_size=60, *, polar_motion=false, spacetrack_username=None, spacetrack_password=None, _epoch_tolerance_days=None, enforce_source=None))]
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

        // Fetch OMM data
        let fetched: FetchedOMM = fetch_omm_unified(
            norad_id,
            norad_name.as_deref(),
            target_epoch.as_ref(),
            credentials,
            enforce_source.as_deref(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Convert OMM to SGP4 Elements directly
        let elements = omm_to_elements(&fetched.data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Check that begin and end are provided
        let begin = begin.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("begin parameter is required")
        })?;
        let end = end
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("end parameter is required"))?;

        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size)?;

        let omm_data = fetched.data;

        // Create the OMMEphemeris object
        let mut ephemeris: OMMEphemeris = OMMEphemeris {
            omm_data,
            elements,
            teme: None,
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
        ephemeris.propagate_to_teme()?;
        ephemeris.teme_to_itrs()?;
        ephemeris.teme_to_gcrs()?;
        ephemeris.calculate_sun_moon()?;

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
    fn begin_time(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_begin_time(&self.common_data.times, py)
    }

    /// Get the end time of the ephemeris
    #[getter]
    fn end_time(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_end_time(&self.common_data.times, py)
    }

    /// Get the step size in seconds
    #[getter]
    fn step_size(&self) -> PyResult<i64> {
        crate::ephemeris::ephemeris_common::get_step_size(&self.common_data.times)
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
        let times = self
            .common_data
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No times available"))?;

        // Create SGP4 constants directly from elements
        let constants = Constants::from_elements(&self.elements).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("SGP4 constants error: {e:?}"))
        })?;

        // Create output array for positions and velocities
        let n_times = times.len();
        let mut pv_data = Vec::with_capacity(n_times * 6);

        // Propagate for each timestamp
        for time in times {
            // Convert to NaiveDateTime for sgp4 compatibility
            let naive_dt = time.naive_utc();

            // Calculate minutes since epoch
            // Use unwrap() since time conversions should always succeed for valid timestamps
            let minutes_since_epoch = self
                .elements
                .datetime_to_minutes_since_epoch(&naive_dt)
                .unwrap();

            let result = constants.propagate(minutes_since_epoch).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("SGP4 propagation failed: {}", e))
            })?;

            // SGP4 returns position in km and velocity in km/s
            pv_data.extend_from_slice(&[
                result.position[0],
                result.position[1],
                result.position[2],
                result.velocity[0],
                result.velocity[1],
                result.velocity[2],
            ]);
        }

        // Convert to ndarray
        let pv_array = Array2::from_shape_vec((n_times, 6), pv_data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Array creation failed: {}", e))
        })?;

        self.teme = Some(pv_array);
        Ok(())
    }

    /// Convert TEME to ITRS coordinates
    fn teme_to_itrs(&mut self) -> PyResult<()> {
        if let Some(teme) = &self.teme {
            let times =
                self.common_data.times.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("No times available")
                })?;

            let itrs = conversions::convert_frames(
                teme,
                times,
                conversions::Frame::TEME,
                conversions::Frame::ITRS,
                self.polar_motion,
            );
            self.itrs = Some(itrs);
        }
        Ok(())
    }

    /// Convert TEME to GCRS coordinates
    fn teme_to_gcrs(&mut self) -> PyResult<()> {
        if let Some(teme) = &self.teme {
            let times =
                self.common_data.times.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("No times available")
                })?;

            let gcrs = conversions::convert_frames(
                teme,
                times,
                conversions::Frame::TEME,
                conversions::Frame::GCRS,
                false,
            );
            self.common_data.gcrs = Some(gcrs);
        }
        Ok(())
    }

    /// Calculate Sun and Moon positions
    fn calculate_sun_moon(&mut self) -> PyResult<()> {
        let times = self
            .common_data
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No times available"))?;

        let sun_gcrs = crate::utils::celestial::calculate_sun_positions(times);
        let moon_gcrs = crate::utils::celestial::calculate_moon_positions(times);
        self.common_data.sun_gcrs = Some(sun_gcrs);
        self.common_data.moon_gcrs = Some(moon_gcrs);

        Ok(())
    }
}

// ===== EphemerisBase trait implementation =====

impl EphemerisBase for OMMEphemeris {
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
