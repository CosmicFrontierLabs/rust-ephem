use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use ndarray::{s, Array2};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

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
    /// Lazy-initialized SkyCoord caches - created on first access
    pub gcrs_skycoord: OnceLock<Py<PyAny>>,
    pub earth_skycoord: OnceLock<Py<PyAny>>,
    pub sun_skycoord: OnceLock<Py<PyAny>>,
    pub moon_skycoord: OnceLock<Py<PyAny>>,
    /// Cached Python timestamp array (NumPy array of datetime objects)
    pub timestamp_cache: OnceLock<Py<PyAny>>,
    /// Cached angular radius arrays
    pub sun_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub moon_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub earth_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub sun_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
    pub moon_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
    pub earth_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
}

impl EphemerisData {
    /// Create a new empty EphemerisData
    pub fn new() -> Self {
        EphemerisData {
            gcrs: None,
            times: None,
            sun_gcrs: None,
            moon_gcrs: None,
            gcrs_skycoord: OnceLock::new(),
            earth_skycoord: OnceLock::new(),
            sun_skycoord: OnceLock::new(),
            moon_skycoord: OnceLock::new(),
            timestamp_cache: OnceLock::new(),
            sun_angular_radius_cache: OnceLock::new(),
            moon_angular_radius_cache: OnceLock::new(),
            earth_angular_radius_cache: OnceLock::new(),
            sun_angular_radius_rad_cache: OnceLock::new(),
            moon_angular_radius_rad_cache: OnceLock::new(),
            earth_angular_radius_rad_cache: OnceLock::new(),
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

    /// Get a reference to cached ITRS SkyCoord (if already created)
    /// This must be implemented by each ephemeris type that caches ITRS SkyCoord
    /// Returns None if not yet created (lazy initialization)
    fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>>;

    /// Set the ITRS SkyCoord cache
    /// This must be implemented by each ephemeris type to store in its OnceLock
    fn set_itrs_skycoord_cache(&self, skycoord: Py<PyAny>) -> Result<(), Py<PyAny>>;

    /// Get ITRS position and velocity in PositionVelocityData format
    fn get_itrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_itrs_data()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    /// Get cached ITRS SkyCoord object
    fn get_itrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check if already cached
        if let Some(cached) = self.get_itrs_skycoord_ref() {
            return Ok(cached.clone_ref(py));
        }

        // Lazy create and cache - implementation will handle caching
        self.itrs_to_skycoord_helper(py)
    }

    fn get_gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.data()
            .gcrs
            .as_ref()
            .map(|arr| Py::new(py, split_pos_vel(arr)).unwrap())
    }

    /// Get cached GCRS SkyCoord object
    fn get_gcrs(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Lazy initialization: create on first access
        if let Some(cached) = self.data().gcrs_skycoord.get() {
            return Ok(cached.clone_ref(py));
        }

        // Create the SkyCoord
        let modules = AstropyModules::import(py)?;
        let skycoord = self.gcrs_to_skycoord(py, &modules)?;

        // Try to cache it (may fail if another thread cached it first, which is fine)
        let _ = self.data().gcrs_skycoord.set(skycoord.clone_ref(py));

        Ok(skycoord)
    }

    /// Get cached Earth SkyCoord object
    fn get_earth(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Lazy initialization: create on first access
        if let Some(cached) = self.data().earth_skycoord.get() {
            return Ok(cached.clone_ref(py));
        }

        // Create the SkyCoord
        let modules = AstropyModules::import(py)?;
        let skycoord = self.earth_to_skycoord(py, &modules)?;

        // Try to cache it
        let _ = self.data().earth_skycoord.set(skycoord.clone_ref(py));

        Ok(skycoord)
    }

    /// Get cached Sun SkyCoord object
    fn get_sun(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Lazy initialization: create on first access
        if let Some(cached) = self.data().sun_skycoord.get() {
            return Ok(cached.clone_ref(py));
        }

        // Create the SkyCoord
        let modules = AstropyModules::import(py)?;
        let skycoord = self.sun_to_skycoord(py, &modules)?;

        // Try to cache it
        let _ = self.data().sun_skycoord.set(skycoord.clone_ref(py));

        Ok(skycoord)
    }

    /// Get cached Moon SkyCoord object
    fn get_moon(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Lazy initialization: create on first access
        if let Some(cached) = self.data().moon_skycoord.get() {
            return Ok(cached.clone_ref(py));
        }

        // Create the SkyCoord
        let modules = AstropyModules::import(py)?;
        let skycoord = self.moon_to_skycoord(py, &modules)?;

        // Try to cache it
        let _ = self.data().moon_skycoord.set(skycoord.clone_ref(py));

        Ok(skycoord)
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
            // Check cache first
            if let Some(cached) = self.data().timestamp_cache.get() {
                return cached.clone_ref(py);
            }

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
            let result: Py<PyAny> = np_array.into();

            // Cache the result
            let _ = self.data().timestamp_cache.set(result.clone_ref(py));

            result
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

    /// Get ITRS SkyCoord - helper for ephemeris types that support ITRS
    /// Default implementation provides lazy initialization with caching
    fn itrs_to_skycoord_helper(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.get_itrs_skycoord_ref() {
            return Ok(cached.clone_ref(py));
        }

        // Create the SkyCoord
        let modules = AstropyModules::import(py)?;
        let skycoord = self.itrs_to_skycoord(py, &modules)?;

        // Try to cache it (may fail if another thread cached it first, which is fine)
        let _ = self.set_itrs_skycoord_cache(skycoord.clone_ref(py));

        Ok(skycoord)
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

    /// Get angular radius of the Sun as seen from the observer (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
    fn get_sun_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().sun_angular_radius_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::SUN_RADIUS_KM;
        use numpy::PyArray1;

        let sun_pv_data = self
            .data()
            .sun_gcrs
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No Sun data available."))?;

        let n = sun_pv_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = sun_pv_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            let angular_radius_rad = (SUN_RADIUS_KM / distance).asin();
            angular_radii.push(angular_radius_rad.to_degrees());
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .sun_angular_radius_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Get angular radius of the Moon as seen from the observer (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
    fn get_moon_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().moon_angular_radius_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::MOON_RADIUS_KM;
        use numpy::PyArray1;

        let moon_pv_data =
            self.data().moon_gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No Moon data available.")
            })?;

        let n = moon_pv_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = moon_pv_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            let angular_radius_rad = (MOON_RADIUS_KM / distance).asin();
            angular_radii.push(angular_radius_rad.to_degrees());
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .moon_angular_radius_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Get angular radius of the Earth as seen from the observer (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
    fn get_earth_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().earth_angular_radius_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::EARTH_RADIUS_KM;
        use numpy::PyArray1;

        let gcrs_data =
            self.data().gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No GCRS data available.")
            })?;

        let n = gcrs_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = gcrs_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            // Angular radius: angle from observer to visible horizon
            // For ground observers: arcsin(R_earth / distance) < 90°
            // Clamp ratio to [0, 1] for numerical stability
            let ratio = (EARTH_RADIUS_KM / distance).min(1.0);
            let angular_radius_rad = ratio.asin();
            angular_radii.push(angular_radius_rad.to_degrees());
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .earth_angular_radius_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Get angular radius of the Sun with astropy units (degrees)
    ///
    /// Returns an astropy Quantity with units of degrees
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    fn get_sun_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        let angular_radii_array = self.get_sun_radius_deg(py)?;
        let astropy = py.import("astropy.units")?;
        let quantity_class = astropy.getattr("Quantity")?;
        let deg_unit = astropy.getattr("deg")?;

        Ok(quantity_class
            .call1((angular_radii_array, deg_unit))?
            .into())
    }

    /// Get angular radius of the Moon with astropy units (degrees)
    ///
    /// Returns an astropy Quantity with units of degrees
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    fn get_moon_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        let angular_radii_array = self.get_moon_radius_deg(py)?;
        let astropy = py.import("astropy.units")?;
        let quantity_class = astropy.getattr("Quantity")?;
        let deg_unit = astropy.getattr("deg")?;

        Ok(quantity_class
            .call1((angular_radii_array, deg_unit))?
            .into())
    }

    /// Get angular radius of the Earth with astropy units (degrees)
    ///
    /// Returns an astropy Quantity with units of degrees
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    fn get_earth_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        let angular_radii_array = self.get_earth_radius_deg(py)?;
        let astropy = py.import("astropy.units")?;
        let quantity_class = astropy.getattr("Quantity")?;
        let deg_unit = astropy.getattr("deg")?;

        Ok(quantity_class
            .call1((angular_radii_array, deg_unit))?
            .into())
    }

    /// Get angular radius of the Sun as seen from the observer (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    fn get_sun_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().sun_angular_radius_rad_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::SUN_RADIUS_KM;
        use numpy::PyArray1;

        let sun_pv_data = self
            .data()
            .sun_gcrs
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No Sun data available."))?;

        let n = sun_pv_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = sun_pv_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            let angular_radius_rad = (SUN_RADIUS_KM / distance).asin();
            angular_radii.push(angular_radius_rad);
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .sun_angular_radius_rad_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Get angular radius of the Moon as seen from the observer (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    fn get_moon_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().moon_angular_radius_rad_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::MOON_RADIUS_KM;
        use numpy::PyArray1;

        let moon_pv_data =
            self.data().moon_gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No Moon data available.")
            })?;

        let n = moon_pv_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = moon_pv_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            let angular_radius_rad = (MOON_RADIUS_KM / distance).asin();
            angular_radii.push(angular_radius_rad);
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .moon_angular_radius_rad_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Get angular radius of the Earth as seen from the observer (in radians)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in radians
    fn get_earth_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = self.data().earth_angular_radius_rad_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        use crate::utils::config::EARTH_RADIUS_KM;
        use numpy::PyArray1;

        let gcrs_data =
            self.data().gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No GCRS data available.")
            })?;

        let n = gcrs_data.nrows();
        let mut angular_radii = Vec::with_capacity(n);

        for i in 0..n {
            let row = gcrs_data.row(i);
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let distance = (x * x + y * y + z * z).sqrt();
            // Angular radius: angle from observer to visible horizon
            // For ground observers: arcsin(R_earth / distance) < π/2
            // Clamp ratio to [0, 1] for numerical stability
            let ratio = (EARTH_RADIUS_KM / distance).min(1.0);
            let angular_radius_rad = ratio.asin();
            angular_radii.push(angular_radius_rad);
        }

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .earth_angular_radius_rad_cache
            .set(result.clone_ref(py));

        Ok(result)
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
    /// `usize` - Index of the closest timestamp
    ///
    /// # Errors
    /// Returns error if:
    /// - No timestamps are available in the ephemeris
    /// - The provided datetime cannot be converted to UTC
    ///
    /// # Example Python usage
    /// ```python
    /// from datetime import datetime
    /// eph = TLEEphemeris(...)
    /// target_time = datetime(2024, 1, 15, 12, 0, 0)
    /// idx = eph.index(target_time)
    /// # Now you can use idx to access specific data points
    /// position = eph.gcrs_pv.position[idx]
    /// ```
    fn find_closest_index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        use crate::utils::time_utils::python_datetime_to_utc;

        let times = self.data().times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No timestamps available in ephemeris.")
        })?;

        if times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Ephemeris contains no timestamps.",
            ));
        }

        let target_time = python_datetime_to_utc(time)?;

        // Use binary search to find the closest timestamp - O(log n) complexity
        // Since timestamps are guaranteed to be sorted (generated in ascending order),
        // we can use binary_search_by to efficiently locate the insertion point
        match times.binary_search(&target_time) {
            // Exact match found
            Ok(idx) => Ok(idx),
            // Not found - idx is the insertion point
            Err(idx) => {
                // Handle edge cases
                if idx == 0 {
                    // Target is before all timestamps, return first index
                    Ok(0)
                } else if idx >= times.len() {
                    // Target is after all timestamps, return last index
                    Ok(times.len() - 1)
                } else {
                    // Target is between idx-1 and idx, find which is closer
                    let before = &times[idx - 1];
                    let after = &times[idx];

                    let diff_before = (target_time - *before).num_milliseconds().abs();
                    let diff_after = (*after - target_time).num_milliseconds().abs();

                    if diff_before <= diff_after {
                        Ok(idx - 1)
                    } else {
                        Ok(idx)
                    }
                }
            }
        }
    }
}
