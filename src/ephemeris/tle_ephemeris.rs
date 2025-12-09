use chrono::{Datelike, Timelike};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use sgp4::{parse_2les, Constants};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{
    generate_timestamps, split_pos_vel, EphemerisBase, EphemerisData,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::tle_utils;
use crate::utils::to_skycoord::AstropyModules;

#[pyclass]
pub struct TLEEphemeris {
    tle1: String,
    tle2: String,
    tle_epoch: chrono::DateTime<chrono::Utc>, // TLE epoch timestamp
    teme: Option<Array2<f64>>,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>, // Lazy-initialized cached SkyCoord object for ITRS
    polar_motion: bool,                 // Whether to apply polar motion correction
    // Common ephemeris data
    common_data: EphemerisData,
}

#[pymethods]
impl TLEEphemeris {
    #[new]
    #[pyo3(signature = (tle1=None, tle2=None, begin=None, end=None, step_size=60, *, polar_motion=false, tle=None, norad_id=None, norad_name=None, spacetrack_username=None, spacetrack_password=None, epoch_tolerance_days=None, enforce_source=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        tle1: Option<String>,
        tle2: Option<String>,
        begin: Option<&Bound<'_, PyDateTime>>,
        end: Option<&Bound<'_, PyDateTime>>,
        step_size: i64,
        polar_motion: bool,
        tle: Option<&Bound<'_, pyo3::PyAny>>,
        norad_id: Option<u32>,
        norad_name: Option<String>,
        spacetrack_username: Option<String>,
        spacetrack_password: Option<String>,
        epoch_tolerance_days: Option<f64>,
        enforce_source: Option<String>,
    ) -> PyResult<Self> {
        // For Space-Track, we need begin time first to calculate target epoch
        let begin_for_epoch =
            begin.and_then(|b| crate::utils::time_utils::python_datetime_to_utc(b).ok());

        // Determine which method to use for getting TLE data
        let fetched = if let (Some(l1), Some(l2)) = (tle1, tle2) {
            // Legacy method: tle1 and tle2 parameters (direct TLE lines)
            tle_utils::FetchedTLE::from_lines(l1, l2, None, None)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else if let Some(tle_obj) = tle {
            // tle parameter: can be a string (file path/URL) or a TLERecord object
            if let Ok(tle_string) = tle_obj.extract::<String>() {
                // String: file path or URL - use unified function
                tle_utils::fetch_tle_unified(Some(&tle_string), None, None, None, None, None, None)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            } else if tle_obj.hasattr("line1")? && tle_obj.hasattr("line2")? {
                // Object with line1/line2 attributes (TLERecord or similar)
                let line1: String = tle_obj.getattr("line1")?.extract()?;
                let line2: String = tle_obj.getattr("line2")?.extract()?;
                let epoch = if tle_obj.hasattr("epoch")? {
                    let epoch_obj = tle_obj.getattr("epoch")?;
                    if let Ok(epoch_dt) = epoch_obj.downcast::<PyDateTime>() {
                        Some(crate::utils::time_utils::python_datetime_to_utc(epoch_dt)?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                tle_utils::FetchedTLE::from_lines(line1, line2, None, epoch)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "tle parameter must be a string (file path or URL) or an object with line1/line2 attributes"
                ));
            }
        } else if norad_id.is_some() || norad_name.is_some() {
            // Build credentials using helper function
            let credentials = tle_utils::build_credentials(
                spacetrack_username.as_deref(),
                spacetrack_password.as_deref(),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

            // For Space-Track with norad_id, we need a target epoch
            let target_epoch = if norad_id.is_some() && credentials.is_some() {
                Some(begin_for_epoch.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "begin parameter is required (used as target epoch for Space-Track.org TLE selection)"
                    )
                })?)
            } else {
                begin_for_epoch
            };

            tle_utils::fetch_tle_unified(
                None,
                norad_id,
                norad_name.as_deref(),
                target_epoch.as_ref(),
                credentials,
                epoch_tolerance_days,
                enforce_source.as_deref(),
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Must provide either (tle1, tle2), tle, norad_id, or norad_name parameters",
            ));
        };

        // Check that begin and end are provided
        let begin = begin.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("begin parameter is required")
        })?;
        let end = end
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("end parameter is required"))?;

        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size)?;

        // Create the TLEEphemeris object
        let mut ephemeris: TLEEphemeris = TLEEphemeris {
            tle1: fetched.line1,
            tle2: fetched.line2,
            tle_epoch: fetched.epoch,
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

        // Return the TLEEphemeris object
        Ok(ephemeris)
    }

    /// Get the epoch of the TLE as a Python datetime object
    #[getter]
    fn tle_epoch(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Convert chrono::DateTime<Utc> to Python datetime with UTC timezone
        let epoch = self.tle_epoch;

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

    /// Get the first TLE line
    #[getter]
    fn tle1(&self) -> &str {
        &self.tle1
    }

    /// Get the second TLE line
    #[getter]
    fn tle2(&self) -> &str {
        &self.tle2
    }

    /// Get the start time of the ephemeris
    #[getter]
    fn begin(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_begin_time(&self.common_data.times, py)
    }

    /// Get the end time of the ephemeris
    #[getter]
    fn end(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::ephemeris::ephemeris_common::get_end_time(&self.common_data.times, py)
    }

    /// Get the time step size in seconds
    #[getter]
    fn step_size(&self) -> PyResult<i64> {
        crate::ephemeris::ephemeris_common::get_step_size(&self.common_data.times)
    }

    /// Get whether polar motion correction is applied
    #[getter]
    fn polar_motion(&self) -> bool {
        self.polar_motion
    }

    #[getter]
    fn teme_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.teme
            .as_ref()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
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

    /// Convert RA/Dec to Altitude/Azimuth
    ///
    /// # Arguments
    /// * `ra_deg` - Right ascension in degrees
    /// * `dec_deg` - Declination in degrees
    /// * `time_indices` - Optional indices into ephemeris times (default: all times)
    ///
    /// # Returns
    /// Numpy array with shape (N, 2) containing [altitude_deg, azimuth_deg]
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
        EphemerisBase::calculate_airmass(self, ra_deg, dec_deg, time_indices.as_deref())
    }

    #[getter]
    fn gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_gcrs_pv(py)
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
        self.get_height_km(py)
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
    fn moon_phase(&self, time_indices: Option<Vec<usize>>) -> PyResult<Vec<f64>> {
        EphemerisBase::moon_phase(self, time_indices.as_deref())
    }

    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }

    fn get_body_pv(&self, py: Python, body: &str) -> PyResult<Py<PositionVelocityData>> {
        <Self as EphemerisBase>::get_body_pv(self, py, body)
    }

    fn get_body(&self, py: Python, body: &str) -> PyResult<Py<PyAny>> {
        let modules = AstropyModules::import(py)?;
        <Self as EphemerisBase>::get_body(self, py, &modules, body)
    }

    /// propagate_to_teme() -> np.ndarray
    ///
    /// Propagates the satellite to the times specified during initialization.
    /// Returns [x,y,z,vx,vy,vz] in TEME coordinates (km, km/s).
    fn propagate_to_teme(&mut self) -> PyResult<()> {
        // Get the internally stored times
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "TLEEphemeris object was not properly initialized. Please create a new TLEEphemeris instance with begin, end, and step_size parameters.",
            )
        })?;

        // Parse TLE - concatenate with newlines (parse_2les expects newline-separated format)
        let tle_string = format!("{}\n{}", self.tle1, self.tle2);
        let elements_vec = parse_2les(&tle_string).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("TLE parse error: {e:?}"))
        })?;
        // Use the first set of elements
        if elements_vec.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No elements parsed from TLE",
            ));
        }
        let elements = elements_vec.into_iter().next().unwrap();

        // Create SGP4 constants
        let constants = Constants::from_elements(&elements).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("SGP4 constants error: {e:?}"))
        })?;

        // Prepare output array
        let n = times.len();
        let mut out = Array2::<f64>::zeros((n, 6));

        for (i, dt) in times.iter().enumerate() {
            // Convert to NaiveDateTime for sgp4 compatibility
            let naive_dt = dt.naive_utc();

            // Calculate minutes since epoch
            // Use unwrap() since time conversions should always succeed for valid timestamps
            let minutes_since_epoch = elements.datetime_to_minutes_since_epoch(&naive_dt).unwrap();

            // Propagate to get position and velocity in TEME
            let pred = constants.propagate(minutes_since_epoch).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Propagation error: {e:?}"))
            })?;

            // Store results - use direct assignment for better performance
            let mut row = out.row_mut(i);
            row[0] = pred.position[0];
            row[1] = pred.position[1];
            row[2] = pred.position[2];
            row[3] = pred.velocity[0];
            row[4] = pred.velocity[1];
            row[5] = pred.velocity[2];
        }

        // Store results
        self.teme = Some(out);
        Ok(())
    }

    /// teme_to_itrs() -> np.ndarray
    ///
    /// Converts the stored TEME coordinates to ITRS (Earth-fixed) coordinates.
    /// Returns [x,y,z,vx,vy,vz] in ITRS frame (km, km/s).
    /// Requires propagate_to_teme to be called first.
    fn teme_to_itrs(&mut self) -> PyResult<()> {
        // Access stored TEME data
        let teme_data = self.teme.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No TEME data available. Call propagate_to_teme first.",
            )
        })?;
        // Use stored times
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No times available. Call propagate_to_teme first.",
            )
        })?;

        // Check lengths match
        if teme_data.nrows() != times.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of times must match number of TEME rows",
            ));
        }

        // Use the generic conversion function
        let itrs_result = conversions::convert_frames(
            teme_data,
            times,
            conversions::Frame::TEME,
            conversions::Frame::ITRS,
            self.polar_motion,
        );
        self.itrs = Some(itrs_result);
        Ok(())
    }

    /// teme_to_gcrs() -> np.ndarray
    ///
    /// Converts stored TEME coordinates directly to GCRS using proper transformations.
    /// This is the recommended method for TEME -> GCRS conversion.
    /// Returns [x,y,z,vx,vy,vz] in GCRS (km, km/s).
    /// Requires propagate_to_teme to be called first.
    fn teme_to_gcrs(&mut self) -> PyResult<()> {
        let teme_data = self.teme.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No TEME data available. Call propagate_to_teme first.",
            )
        })?;

        // Use stored times
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No times available. Call propagate_to_teme first.",
            )
        })?;

        if teme_data.nrows() != times.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of times must match number of TEME rows",
            ));
        }

        // Use the generic conversion function
        let gcrs_result = conversions::convert_frames(
            teme_data,
            times,
            conversions::Frame::TEME,
            conversions::Frame::GCRS,
            self.polar_motion,
        );
        self.common_data.gcrs = Some(gcrs_result);
        Ok(())
    }
}

// Implement the EphemerisBase trait for TLEEphemeris
impl EphemerisBase for TLEEphemeris {
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
