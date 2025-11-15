use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use ndarray::{s, Array2};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};

use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::celestial::{calculate_moon_positions, calculate_sun_positions};
use crate::utils::config::MAX_TIMESTAMPS;
use crate::utils::time_utils::python_datetime_to_utc;
use crate::utils::to_skycoord::{to_skycoord, AstropyModules, SkyCoordConfig};

/// Splits a stacked position+velocity (N x 6) array into a PositionVelocityData struct.
///
/// # Arguments
/// * `arr` - Reference to an N x 6 array where columns 0-2 are position (km) and 3-5 are velocity (km/s).
///
/// # Returns
/// `PositionVelocityData` containing separate position and velocity arrays (both N x 3).
pub(crate) fn split_pos_vel(arr: &Array2<f64>) -> PositionVelocityData {
    let position = arr.slice(s![.., 0..3]).to_owned();
    let velocity = arr.slice(s![.., 3..6]).to_owned();
    PositionVelocityData { position, velocity }
}

/// Generate a vector of timestamps from begin to end (inclusive) with step_size in seconds
/// This is common logic shared between TLEEphemeris and SPICEEphemeris constructors.
///
/// # Arguments
/// * `begin` - Python datetime for the start of the time range
/// * `end` - Python datetime for the end of the time range
/// * `step_size` - Step size in seconds between timestamps
///
/// # Returns
/// `Vec<DateTime<Utc>>` of generated timestamps
///
/// # Errors
/// Returns error if:
/// - begin > end
/// - step_size <= 0
/// - Expected timestamp count exceeds MAX_TIMESTAMPS
pub fn generate_timestamps(
    begin: &Bound<'_, PyDateTime>,
    end: &Bound<'_, PyDateTime>,
    step_size: i64,
) -> PyResult<Vec<DateTime<Utc>>> {
    // Convert Python datetime objects to Rust DateTime<Utc>
    let begin_dt = python_datetime_to_utc(begin)?;
    let end_dt = python_datetime_to_utc(end)?;

    // Validate inputs
    if begin_dt > end_dt {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "begin must be before or equal to end",
        ));
    }
    if step_size <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step_size must be positive",
        ));
    }

    // Calculate expected number of timestamps to prevent excessive memory allocation
    // Using ceiling division: (a + b - 1) / b to handle non-evenly divisible ranges
    let time_range_secs = (end_dt - begin_dt).num_seconds();
    let expected_count = (time_range_secs + step_size) / step_size;

    // Limit to prevent memory exhaustion
    if expected_count > MAX_TIMESTAMPS {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Time range would generate approximately {expected_count} timestamps (max: {MAX_TIMESTAMPS}). Use a larger step_size.")
        ));
    }

    // Generate timestamps from begin to end (inclusive) with step_size in seconds
    // Pre-allocate with expected capacity to avoid reallocations
    let mut times = Vec::with_capacity(expected_count as usize);
    let mut current = begin_dt;
    let step_duration = Duration::seconds(step_size);

    while current <= end_dt {
        times.push(current);
        current += step_duration;
    }

    Ok(times)
}

/// Common data structure for ephemeris objects
/// This holds the shared state between TLEEphemeris and SPICEEphemeris
pub struct EphemerisData {
    pub gcrs: Option<Array2<f64>>,
    pub times: Option<Vec<DateTime<Utc>>>,
    pub sun_gcrs: Option<Array2<f64>>,
    pub moon_gcrs: Option<Array2<f64>>,
    pub gcrs_skycoord: Option<Py<PyAny>>,
    pub earth_skycoord: Option<Py<PyAny>>,
    pub sun_skycoord: Option<Py<PyAny>>,
    pub moon_skycoord: Option<Py<PyAny>>,
}

impl EphemerisData {
    /// Create a new empty EphemerisData
    pub fn new() -> Self {
        EphemerisData {
            gcrs: None,
            times: None,
            sun_gcrs: None,
            moon_gcrs: None,
            gcrs_skycoord: None,
            earth_skycoord: None,
            sun_skycoord: None,
            moon_skycoord: None,
        }
    }
}

impl Default for EphemerisData {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait defining common behavior for ephemeris objects
pub trait EphemerisBase {
    /// Get a reference to the common ephemeris data
    fn data(&self) -> &EphemerisData;

    /// Get a mutable reference to the common ephemeris data
    fn data_mut(&mut self) -> &mut EphemerisData;

    /// Get a reference to ITRS data
    /// This must be implemented by each ephemeris type that supports ITRS
    fn get_itrs_data(&self) -> Option<&Array2<f64>>;

    /// Get a reference to cached ITRS SkyCoord
    /// This must be implemented by each ephemeris type that caches ITRS SkyCoord
    fn get_itrs_skycoord(&self) -> Option<&Py<PyAny>>;

    /// Get ITRS position and velocity in PositionVelocityData format
    fn get_itrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_itrs_data()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    /// Get cached ITRS SkyCoord object
    fn get_itrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self
            .get_itrs_skycoord()
            .map(|py_obj| py_obj.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "No cached ITRS SkyCoord available. Ensure it was computed during initialization.",
                )
            })
    }

    fn get_gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.data()
            .gcrs
            .as_ref()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    /// Get cached GCRS SkyCoord object
    fn get_gcrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        self
            .data()
            .gcrs_skycoord
            .as_ref()
            .map(|py_obj| py_obj.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "No cached GCRS SkyCoord available. Ensure it was computed during initialization.",
                )
            })
    }

    /// Get cached Earth SkyCoord object
    fn get_earth(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.data()
            .earth_skycoord
            .as_ref()
            .map(|py_obj| py_obj.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "No cached Earth SkyCoord available. Ensure it was computed during initialization.",
                )
            })
    }

    /// Get cached Sun SkyCoord object
    fn get_sun(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.data()
            .sun_skycoord
            .as_ref()
            .map(|py_obj| py_obj.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "No cached Sun SkyCoord available. Ensure it was computed during initialization.",
                )
            })
    }

    /// Get cached Moon SkyCoord object
    fn get_moon(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.data()
            .moon_skycoord
            .as_ref()
            .map(|py_obj| py_obj.clone_ref(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "No cached Moon SkyCoord available. Ensure it was computed during initialization.",
                )
            })
    }

    /// Get timestamps as Python datetime objects (for internal use by SkyCoordConfig)
    fn get_timestamp_vec(&self, py: Python) -> PyResult<Option<Vec<Py<PyDateTime>>>> {
        Ok(self.data().times.as_ref().map(|times| {
            times
                .iter()
                .map(|dt| {
                    PyDateTime::new(
                        py,
                        dt.year(),
                        dt.month() as u8,
                        dt.day() as u8,
                        dt.hour() as u8,
                        dt.minute() as u8,
                        dt.second() as u8,
                        dt.timestamp_subsec_micros(),
                        None,
                    )
                    .unwrap()
                    .into()
                })
                .collect()
        }))
    }

    /// Get timestamps as numpy array of Python datetime objects (optimized for property access)
    fn get_timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        use crate::utils::time_utils::utc_to_python_datetime;

        Ok(self.data().times.as_ref().map(|times| {
            // Import numpy
            let np = pyo3::types::PyModule::import(py, "numpy")
                .map_err(|_| pyo3::exceptions::PyImportError::new_err("numpy is required"))
                .unwrap();

            // Build list of Python datetime objects
            let py_list = pyo3::types::PyList::empty(py);
            for dt in times {
                let py_dt = utc_to_python_datetime(py, dt).unwrap();
                py_list.append(py_dt).unwrap();
            }

            // Convert to numpy array with dtype=object
            let np_array = np.getattr("array").unwrap().call1((py_list,)).unwrap();

            np_array.into()
        }))
    }

    /// Get Sun position and velocity in GCRS frame
    fn get_sun_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.data()
            .sun_gcrs
            .as_ref()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    /// Get Moon position and velocity in GCRS frame
    fn get_moon_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.data()
            .moon_gcrs
            .as_ref()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    // ========== Constraint helper methods ==========

    /// Get times as Vec<DateTime<Utc>> for constraint evaluation
    fn get_times(&self) -> PyResult<Vec<DateTime<Utc>>> {
        self.data()
            .times
            .as_ref()
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))
    }

    /// Get Sun positions in GCRS (N x 3 array, km) for constraint evaluation
    fn get_sun_positions(&self) -> PyResult<Array2<f64>> {
        let sun_data =
            self.data().sun_gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No Sun positions available")
            })?;

        // Extract only positions (first 3 columns)
        Ok(sun_data.slice(s![.., 0..3]).to_owned())
    }

    /// Get Moon positions in GCRS (N x 3 array, km) for constraint evaluation
    fn get_moon_positions(&self) -> PyResult<Array2<f64>> {
        let moon_data = self.data().moon_gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No Moon positions available")
        })?;

        // Extract only positions (first 3 columns)
        Ok(moon_data.slice(s![.., 0..3]).to_owned())
    }

    /// Get observer (spacecraft/satellite) positions in GCRS (N x 3 array, km) for constraint evaluation
    fn get_gcrs_positions(&self) -> PyResult<Array2<f64>> {
        let gcrs_data = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No GCRS positions available")
        })?;

        // Extract only positions (first 3 columns)
        Ok(gcrs_data.slice(s![.., 0..3]).to_owned())
    }

    // ========== End constraint helper methods ==========

    /// Get observer geocentric location (obsgeoloc) - alias for GCRS position
    fn get_obsgeoloc(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.data().gcrs.as_ref().map(|arr| {
            let position = arr.slice(s![.., 0..3]).to_owned();
            position.into_pyarray(py).to_owned().into()
        }))
    }

    /// Get observer geocentric velocity (obsgeovel) - alias for GCRS velocity
    fn get_obsgeovel(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.data().gcrs.as_ref().map(|arr| {
            let velocity = arr.slice(s![.., 3..6]).to_owned();
            velocity.into_pyarray(py).to_owned().into()
        }))
    }

    /// Helper to build SkyCoordConfig with common data retrieval pattern
    /// This eliminates duplication across all xxx_to_skycoord methods
    fn build_skycoord_config<'a>(
        &'a self,
        py: Python,
        data: &'a Array2<f64>,
        frame_name: &'a str,
        negate_vectors: bool,
        observer_data: Option<&'a Array2<f64>>,
    ) -> PyResult<SkyCoordConfig<'a>> {
        let time_objects = self
            .get_timestamp_vec(py)?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        Ok(SkyCoordConfig {
            data,
            time_objects,
            frame_name,
            negate_vectors,
            observer_data,
        })
    }

    /// Convert to astropy SkyCoord object with GCRS frame
    fn gcrs_to_skycoord(&self, py: Python, modules: &AstropyModules) -> PyResult<Py<PyAny>> {
        let gcrs_data = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        let config = self.build_skycoord_config(py, gcrs_data, "GCRS", false, None)?;
        to_skycoord(py, Some(modules), config)
    }

    /// Convert Earth position to astropy SkyCoord object (Earth relative to spacecraft)
    fn earth_to_skycoord(&self, py: Python, modules: &AstropyModules) -> PyResult<Py<PyAny>> {
        let gcrs_data = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        let config = self.build_skycoord_config(py, gcrs_data, "GCRS", true, Some(gcrs_data))?;
        to_skycoord(py, Some(modules), config)
    }

    /// Convert Sun positions to astropy SkyCoord object
    fn sun_to_skycoord(&self, py: Python, modules: &AstropyModules) -> PyResult<Py<PyAny>> {
        self.celestial_body_to_skycoord(py, modules, "sun")
    }

    /// Convert Moon positions to astropy SkyCoord object
    fn moon_to_skycoord(&self, py: Python, modules: &AstropyModules) -> PyResult<Py<PyAny>> {
        self.celestial_body_to_skycoord(py, modules, "moon")
    }

    /// Helper method to convert celestial body positions to SkyCoord
    fn celestial_body_to_skycoord(
        &self,
        py: Python,
        modules: &AstropyModules,
        body: &str,
    ) -> PyResult<Py<PyAny>> {
        let body_data =
            match body {
                "sun" => self.data().sun_gcrs.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("No Sun data available.")
                })?,
                "moon" => self.data().moon_gcrs.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("No Moon data available.")
                })?,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid body. Must be 'sun' or 'moon'.",
                    ))
                }
            };

        let gcrs_data = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        // Correct observer position and velocity to be that of the spacecraft
        let body_data_corr: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
            body_data - gcrs_data;

        let config =
            self.build_skycoord_config(py, &body_data_corr, "GCRS", false, Some(gcrs_data))?;
        to_skycoord(py, Some(modules), config)
    }

    /// Calculate Sun and Moon positions in GCRS frame for all timestamps
    fn calculate_sun_moon(&mut self) -> PyResult<()> {
        let times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        // Use batch calculations for better performance
        let sun_out = calculate_sun_positions(times);
        let moon_out = calculate_moon_positions(times);

        let data_mut = self.data_mut();
        data_mut.sun_gcrs = Some(sun_out);
        data_mut.moon_gcrs = Some(moon_out);
        Ok(())
    }

    /// Cache all SkyCoord objects for this ephemeris
    ///
    /// This is a helper function to reduce code duplication across different ephemeris types.
    /// It caches GCRS, Earth, Sun, and Moon SkyCoord objects in the common data structure,
    /// and returns the ITRS SkyCoord for types that need to cache it separately.
    ///
    /// # Arguments
    /// * `py` - Python interpreter state
    ///
    /// # Returns
    /// `PyResult<Option<Py<PyAny>>>` - The ITRS SkyCoord if computed, or None
    fn cache_skycoords(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        // Import astropy modules once for all SkyCoord creations
        let astropy_modules = AstropyModules::import(py)?;

        // Compute ITRS SkyCoord (may fail if not available for this ephemeris type)
        let itrs_skycoord = self.itrs_to_skycoord(py, &astropy_modules).ok();

        // Cache common SkyCoord objects - compute all first, then store
        let gcrs_skycoord = self.gcrs_to_skycoord(py, &astropy_modules).ok();
        let earth_skycoord = self.earth_to_skycoord(py, &astropy_modules).ok();
        let sun_skycoord = self.sun_to_skycoord(py, &astropy_modules).ok();
        let moon_skycoord = self.moon_to_skycoord(py, &astropy_modules).ok();

        // Now store them
        let data_mut = self.data_mut();
        data_mut.gcrs_skycoord = gcrs_skycoord;
        data_mut.earth_skycoord = earth_skycoord;
        data_mut.sun_skycoord = sun_skycoord;
        data_mut.moon_skycoord = moon_skycoord;

        Ok(itrs_skycoord)
    }

    /// Convert to astropy SkyCoord object with ITRS frame
    /// Returns a SkyCoord with ITRS (Earth-fixed) frame containing all time points
    /// This is much faster than creating SkyCoord objects in a Python loop
    fn itrs_to_skycoord(&self, py: Python, modules: &AstropyModules) -> PyResult<Py<PyAny>> {
        let itrs_data = self.get_itrs_data().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No ITRS data available. Ephemeris should compute ITRS during initialization.",
            )
        })?;

        let config = self.build_skycoord_config(py, itrs_data, "ITRS", false, None)?;
        to_skycoord(py, Some(modules), config)
    }

    /// Calculate positions for any body identified by NAIF ID or name relative to the observer
    ///
    /// This is analogous to astropy's `get_body()` function. Returns position and velocity
    /// vectors for the specified body in the observer's GCRS frame.
    ///
    /// # Arguments
    /// * `body_identifier` - NAIF ID (as string) or body name (e.g., "Jupiter", "mars", "301")
    ///
    /// # Returns
    /// `PositionVelocityData` containing position and velocity arrays in km and km/s
    ///
    /// # Example Python usage
    /// ```python
    /// eph = TLEEphemeris(...)
    /// jupiter = eph.get_body("Jupiter")  # By name
    /// mars = eph.get_body("499")  # By NAIF ID
    /// ```
    fn get_body_pv(&self, py: Python, body_identifier: &str) -> PyResult<Py<PositionVelocityData>> {
        use crate::utils::celestial::calculate_body_by_id_or_name;
        use crate::utils::config::EARTH_NAIF_ID;

        let times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        // Calculate body position relative to Earth center
        let body_geocentric = calculate_body_by_id_or_name(times, body_identifier, EARTH_NAIF_ID)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Get observer's geocentric position
        let observer_geocentric = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        // Calculate body position relative to observer: body - observer
        let body_observer_centric = &body_geocentric - observer_geocentric;

        Py::new(py, split_pos_vel(&body_observer_centric))
    }

    /// Get SkyCoord object for any body identified by NAIF ID or name
    ///
    /// This is analogous to astropy's `get_body()` function but returns a SkyCoord
    /// object with the observer location properly set. The returned SkyCoord is in
    /// the GCRS frame with obsgeoloc and obsgeovel set to the observer's position.
    ///
    /// # Arguments
    /// * `body_identifier` - NAIF ID (as string) or body name (e.g., "Jupiter", "mars", "301")
    ///
    /// # Returns
    /// Astropy SkyCoord object in GCRS frame with observer location set
    ///
    /// # Example Python usage
    /// ```python
    /// eph = TLEEphemeris(...)
    /// jupiter = eph.get_body("Jupiter")
    /// # Can now compute separations, altaz coordinates, etc.
    /// separation = jupiter.separation(target_sc)
    /// ```
    fn get_body(
        &self,
        py: Python,
        modules: &AstropyModules,
        body_identifier: &str,
    ) -> PyResult<Py<PyAny>> {
        use crate::utils::celestial::calculate_body_by_id_or_name;
        use crate::utils::config::EARTH_NAIF_ID;

        let times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        // Calculate body position relative to Earth center
        let body_geocentric = calculate_body_by_id_or_name(times, body_identifier, EARTH_NAIF_ID)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Get observer's geocentric position
        let observer_geocentric = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        // Calculate body position relative to observer: body - observer
        let body_observer_centric = &body_geocentric - observer_geocentric;

        // Create SkyCoord with observer location set
        let config = self.build_skycoord_config(
            py,
            &body_observer_centric,
            "GCRS",
            false,
            Some(observer_geocentric),
        )?;
        to_skycoord(py, Some(modules), config)
    }
}
