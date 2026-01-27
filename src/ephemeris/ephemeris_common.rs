use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use ndarray::{s, Array2};
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::celestial::{calculate_moon_positions, calculate_sun_positions};
use crate::utils::config::MAX_TIMESTAMPS;
use crate::utils::conversions::{convert_frames, Frame};
use crate::utils::geo::{deg_to_rad_array, ecef_to_geodetic_deg};
use crate::utils::time_utils::{python_datetime_to_utc, utc_to_python_datetime};
use crate::utils::to_skycoord::{to_skycoord, AstropyModules, SkyCoordConfig};
use ndarray::Array1;

/// Helper function for getting begin time from ephemeris common data
#[inline]
pub fn get_begin_time(times: &Option<Vec<DateTime<Utc>>>, py: Python) -> PyResult<Py<PyAny>> {
    if let Some(times) = times {
        if let Some(first_time) = times.first() {
            return utc_to_python_datetime(py, first_time);
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "No times available",
    ))
}

/// Helper function for getting end time from ephemeris common data
#[inline]
pub fn get_end_time(times: &Option<Vec<DateTime<Utc>>>, py: Python) -> PyResult<Py<PyAny>> {
    if let Some(times) = times {
        if let Some(last_time) = times.last() {
            return utc_to_python_datetime(py, last_time);
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "No times available",
    ))
}

/// Helper function for getting step size from ephemeris common data
#[inline]
pub fn get_step_size(times: &Option<Vec<DateTime<Utc>>>) -> PyResult<i64> {
    if let Some(times) = times {
        if times.len() >= 2 {
            let step = times[1].signed_duration_since(times[0]);
            return Ok(step.num_seconds());
        }
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "Cannot compute step size",
    ))
}

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
    /// Cached Python timestamp vector (Vec of Python datetime objects for SkyCoord creation)
    pub timestamp_vec_cache: OnceLock<Vec<Py<PyDateTime>>>,
    /// Cached angular radius arrays
    pub sun_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub moon_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub earth_angular_radius_cache: OnceLock<Py<PyAny>>,
    pub sun_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
    pub latitude_qty_cache: OnceLock<Py<PyAny>>,
    pub longitude_qty_cache: OnceLock<Py<PyAny>>,
    pub height_qty_cache: OnceLock<Py<PyAny>>,
    pub latitude_deg_cache: OnceLock<Array1<f64>>,
    pub longitude_deg_cache: OnceLock<Array1<f64>>,
    pub latitude_rad_cache: OnceLock<Array1<f64>>,
    pub longitude_rad_cache: OnceLock<Array1<f64>>,
    pub height_km_cache: OnceLock<Array1<f64>>,
    pub height_cache: OnceLock<Array1<f64>>,
    pub moon_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
    pub earth_angular_radius_rad_cache: OnceLock<Py<PyAny>>,
    /// Cached Sun altitude angles (in degrees) for all times
    #[allow(dead_code)]
    pub sun_altitudes_cache: OnceLock<Array1<f64>>,
    /// Cached RA/Dec arrays (Nx2 arrays with RA in column 0, Dec in column 1)
    pub sun_ra_dec_deg_cache: OnceLock<Py<PyAny>>,
    pub moon_ra_dec_deg_cache: OnceLock<Py<PyAny>>,
    pub earth_ra_dec_deg_cache: OnceLock<Py<PyAny>>,
    pub sun_ra_dec_rad_cache: OnceLock<Py<PyAny>>,
    pub moon_ra_dec_rad_cache: OnceLock<Py<PyAny>>,
    pub earth_ra_dec_rad_cache: OnceLock<Py<PyAny>>,
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
            timestamp_vec_cache: OnceLock::new(),
            sun_angular_radius_cache: OnceLock::new(),
            moon_angular_radius_cache: OnceLock::new(),
            earth_angular_radius_cache: OnceLock::new(),
            sun_angular_radius_rad_cache: OnceLock::new(),
            latitude_qty_cache: OnceLock::new(),
            longitude_qty_cache: OnceLock::new(),
            height_qty_cache: OnceLock::new(),
            moon_angular_radius_rad_cache: OnceLock::new(),
            earth_angular_radius_rad_cache: OnceLock::new(),
            latitude_deg_cache: OnceLock::new(),
            longitude_deg_cache: OnceLock::new(),
            latitude_rad_cache: OnceLock::new(),
            longitude_rad_cache: OnceLock::new(),
            height_km_cache: OnceLock::new(),
            height_cache: OnceLock::new(),
            sun_altitudes_cache: OnceLock::new(),
            sun_ra_dec_deg_cache: OnceLock::new(),
            moon_ra_dec_deg_cache: OnceLock::new(),
            earth_ra_dec_deg_cache: OnceLock::new(),
            sun_ra_dec_rad_cache: OnceLock::new(),
            moon_ra_dec_rad_cache: OnceLock::new(),
            earth_ra_dec_rad_cache: OnceLock::new(),
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

    /// Convert RA/Dec to Altitude/Azimuth for this ephemeris
    ///
    /// This function calculates the topocentric altitude and azimuth of a celestial target
    /// as seen from the observer locations defined in this ephemeris.
    ///
    /// # Arguments
    /// * `ra_deg` - Right ascension in degrees
    /// * `dec_deg` - Declination in degrees
    /// * `time_indices` - Optional indices into ephemeris times to evaluate (default: all times)
    ///
    /// # Returns
    /// Array2 with shape (N, 2) containing [altitude_deg, azimuth_deg] for each time
    /// where N is the number of selected times
    fn radec_to_altaz(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<&[usize]>,
    ) -> Array2<f64>;

    /// Calculate airmass for a target at given RA/Dec
    ///
    /// Airmass represents the relative path length through Earth's atmosphere compared to
    /// zenith observation. This is a convenience method that combines altitude calculation
    /// with airmass computation, accounting for observer height above sea level.
    ///
    /// # Arguments
    /// * `ra_deg` - Right ascension in degrees (ICRS/J2000)
    /// * `dec_deg` - Declination in degrees (ICRS/J2000)
    /// * `time_indices` - Optional indices into ephemeris times to evaluate (default: all times)
    ///
    /// # Returns
    /// Vector of airmass values for each selected time:
    /// - 1.0 at zenith (directly overhead)
    /// - ~2.0 at 30° altitude
    /// - ~5.8 at 10° altitude
    /// - Infinity for targets below horizon
    ///
    /// # Example
    /// ```rust,ignore
    /// let airmass = ephemeris.calculate_airmass(180.0, 45.0, None)?;
    /// // Returns airmass for all ephemeris times
    /// ```
    fn calculate_airmass(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Vec<f64>> {
        use crate::utils::celestial::calculate_airmass_kasten;

        // Get altitudes
        let altaz = self.radec_to_altaz(ra_deg, dec_deg, time_indices);

        // Calculate airmass for each time using Kasten formula
        // Kasten & Czeplak (1980): accurate to ±0.02 airmass for zenith angles up to ~75°
        let airmass: Vec<f64> = (0..altaz.nrows())
            .map(|i| {
                let altitude_deg = altaz[[i, 0]];
                calculate_airmass_kasten(altitude_deg)
            })
            .collect();

        Ok(airmass)
    }

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
        // Check cache first
        if let Some(cached) = self.data().timestamp_vec_cache.get() {
            // Clone each Py<PyDateTime> using clone_ref
            let cloned: Vec<Py<PyDateTime>> = cached.iter().map(|dt| dt.clone_ref(py)).collect();
            return Ok(Some(cloned));
        }

        let result: Option<Vec<Py<PyDateTime>>> = self.data().times.as_ref().map(|times| {
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
        });

        // Cache the result if it exists (store by value since we own it)
        if let Some(vec) = &result {
            // Create a clone for caching - each Py<PyDateTime> uses clone_ref
            let cache_vec: Vec<Py<PyDateTime>> = vec.iter().map(|dt| dt.clone_ref(py)).collect();
            let _ = self.data().timestamp_vec_cache.set(cache_vec);
        }

        Ok(result)
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

    /// Calculate Moon illumination fraction for all ephemeris times
    ///
    /// Returns the fraction of the Moon's illuminated surface as seen from the
    /// spacecraft observer (0.0 = new moon, 1.0 = full moon).
    ///
    /// # Arguments
    /// * `time_indices` - Optional indices into ephemeris times to evaluate (default: all times)
    ///
    /// # Returns
    /// Vector of Moon illumination fractions for each selected time
    fn moon_illumination(&self, time_indices: Option<&[usize]>) -> PyResult<Vec<f64>> {
        use crate::utils::moon::calculate_moon_illumination_from_vectors;

        let n_times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))?
            .len();

        let indices: Vec<usize> = if let Some(indices) = time_indices {
            indices.to_vec()
        } else {
            (0..n_times).collect()
        };

        let n = indices.len();

        let observer_positions = match self.data().gcrs.as_ref() {
            Some(pos) => pos,
            None => return Ok(vec![0.5; n]),
        };

        let sun_positions = match self.data().sun_gcrs.as_ref() {
            Some(pos) => pos,
            None => return Ok(vec![0.5; n]),
        };

        let moon_positions = match self.data().moon_gcrs.as_ref() {
            Some(pos) => pos,
            None => return Ok(vec![0.5; n]),
        };

        let mut illuminations = Vec::with_capacity(n);

        for &i in indices.iter() {
            let obs_x = observer_positions[[i, 0]];
            let obs_y = observer_positions[[i, 1]];
            let obs_z = observer_positions[[i, 2]];

            let sun_rel = [
                sun_positions[[i, 0]] - obs_x,
                sun_positions[[i, 1]] - obs_y,
                sun_positions[[i, 2]] - obs_z,
            ];
            let moon_rel = [
                moon_positions[[i, 0]] - obs_x,
                moon_positions[[i, 1]] - obs_y,
                moon_positions[[i, 2]] - obs_z,
            ];

            illuminations.push(calculate_moon_illumination_from_vectors(sun_rel, moon_rel));
        }

        Ok(illuminations)
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

    /// Get observer geodetic latitude (astropy Quantity array). Uses EarthLocation
    /// from observer geocentric coords (obsgeoloc). Returns None if obsgeoloc is missing.
    fn get_latitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        // Return cached Quantity if present
        if let Some(qty_cache) = self.data().latitude_qty_cache.get() {
            return Ok(Some(qty_cache.clone_ref(py)));
        }
        if let Some(lats) = self.data().latitude_deg_cache.get() {
            // Convert to numpy array and wrap as astropy Quantity deg
            let arr = lats.clone().into_pyarray(py);
            let modules = AstropyModules::import(py)?;
            let deg_unit = modules.units.getattr("deg")?;
            let qty = deg_unit.call_method1("__rmul__", (arr,))?;
            let _ = self.data().latitude_qty_cache.set(qty.into());
            Ok(Some(
                self.data().latitude_qty_cache.get().unwrap().clone_ref(py),
            ))
        } else {
            Ok(None)
        }
    }

    /// Get observer geodetic longitude as Quantity array (degrees)
    fn get_longitude(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(qty_cache) = self.data().longitude_qty_cache.get() {
            return Ok(Some(qty_cache.clone_ref(py)));
        }
        if let Some(lons) = self.data().longitude_deg_cache.get() {
            let arr = lons.clone().into_pyarray(py);
            let modules = AstropyModules::import(py)?;
            let deg_unit = modules.units.getattr("deg")?;
            let qty = deg_unit.call_method1("__rmul__", (arr,))?;
            let _ = self.data().longitude_qty_cache.set(qty.into());
            Ok(Some(
                self.data().longitude_qty_cache.get().unwrap().clone_ref(py),
            ))
        } else {
            Ok(None)
        }
    }

    /// Return raw deg float arrays (latitude_deg)
    fn get_latitude_deg(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(lat_arr) = self.data().latitude_deg_cache.get() {
            let arr = lat_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    fn get_longitude_deg(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(lon_arr) = self.data().longitude_deg_cache.get() {
            let arr = lon_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    fn get_latitude_rad(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(lat_arr) = self.data().latitude_rad_cache.get() {
            let arr = lat_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    fn get_longitude_rad(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(lon_arr) = self.data().longitude_rad_cache.get() {
            let arr = lon_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    /// Get observer geodetic height as Quantity array (meters)
    fn get_height(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(qty_cache) = self.data().height_qty_cache.get() {
            return Ok(Some(qty_cache.clone_ref(py)));
        }
        if let Some(h_m) = self.data().height_cache.get() {
            let arr = h_m.clone().into_pyarray(py);
            let modules = AstropyModules::import(py)?;
            let m_unit = modules.units.getattr("m")?;
            let qty = m_unit.call_method1("__rmul__", (arr,))?;
            let _ = self.data().height_qty_cache.set(qty.into());
            Ok(Some(
                self.data().height_qty_cache.get().unwrap().clone_ref(py),
            ))
        } else {
            Ok(None)
        }
    }

    /// Return raw height array in meters
    fn get_height_m(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(h_arr) = self.data().height_cache.get() {
            let arr = h_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    /// Return raw height array in kilometers
    fn get_height_km(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.compute_latlon_caches()?;
        if let Some(h_arr) = self.data().height_km_cache.get() {
            let arr = h_arr.clone().into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else if let Some(h_m_arr) = self.data().height_cache.get() {
            // fallback: convert meters to kilometers if km cache missing
            let h_km = h_m_arr.mapv(|v| v / 1000.0);
            let arr = h_km.into_pyarray(py).to_owned();
            Ok(Some(arr.into()))
        } else {
            Ok(None)
        }
    }

    /// Get observer geocentric velocity (obsgeovel) - alias for GCRS velocity
    fn get_obsgeovel(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.data().gcrs.as_ref().map(|arr| {
            let velocity = arr.slice(s![.., 3..6]).to_owned();
            velocity.into_pyarray(py).to_owned().into()
        }))
    }

    /// Ensure latitude/longitude caches are computed using pure Rust.
    fn compute_latlon_caches(&self) -> PyResult<()> {
        // Already computed
        if self.data().latitude_deg_cache.get().is_some() {
            return Ok(());
        }

        // Get ITRS positions if available, otherwise convert from GCRS
        let positions_itrs_opt: Option<Array2<f64>> = if let Some(itrs) = self.get_itrs_data() {
            Some(itrs.slice(s![.., 0..3]).to_owned())
        } else if let Some(gcrs) = self.data().gcrs.as_ref() {
            // Convert to ITRS
            let times = self
                .data()
                .times
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))?;
            let itrs_array = convert_frames(gcrs, times, Frame::GCRS, Frame::ITRS, false);
            Some(itrs_array.slice(s![.., 0..3]).to_owned())
        } else {
            None
        };

        if let Some(positions_itrs) = positions_itrs_opt {
            let (lats_deg, lons_deg, h_km) = ecef_to_geodetic_deg(&positions_itrs);
            let lats_rad = deg_to_rad_array(&lats_deg);
            let lons_rad = deg_to_rad_array(&lons_deg);
            // Convert height from km (returned by ecef_to_geodetic_deg) to meters
            let h_m = h_km.mapv(|v| v * 1000.0);
            let _ = self.data().latitude_deg_cache.set(lats_deg);
            let _ = self.data().longitude_deg_cache.set(lons_deg);
            let _ = self.data().latitude_rad_cache.set(lats_rad);
            let _ = self.data().longitude_rad_cache.set(lons_rad);
            // Store both km and m caches
            let _ = self.data().height_km_cache.set(h_km);
            let _ = self.data().height_cache.set(h_m);
        }

        Ok(())
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
    fn get_body_pv(
        &self,
        py: Python,
        body_identifier: &str,
        spice_kernel: Option<&str>,
        use_horizons: bool,
    ) -> PyResult<Py<PositionVelocityData>> {
        use crate::utils::celestial::calculate_body_by_id_or_name;
        use crate::utils::config::EARTH_NAIF_ID;

        let times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        // Calculate body position relative to Earth center
        let body_geocentric = calculate_body_by_id_or_name(
            times,
            body_identifier,
            EARTH_NAIF_ID,
            spice_kernel,
            use_horizons,
        )
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
        spice_kernel: Option<&str>,
        use_horizons: bool,
    ) -> PyResult<Py<PyAny>> {
        use crate::utils::celestial::calculate_body_by_id_or_name;
        use crate::utils::config::EARTH_NAIF_ID;

        let times = self
            .data()
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available."))?;

        // Calculate body position relative to Earth center
        let body_geocentric = calculate_body_by_id_or_name(
            times,
            body_identifier,
            EARTH_NAIF_ID,
            spice_kernel,
            use_horizons,
        )
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

    fn body_observer_distances(&self, body_data: &Array2<f64>) -> PyResult<Vec<f64>> {
        let observer_data = self.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Ephemeris should compute GCRS during initialization.",
            )
        })?;

        if body_data.nrows() != observer_data.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Body and observer data must have the same number of rows.",
            ));
        }

        let n = body_data.nrows();
        let mut distances = Vec::with_capacity(n);

        for i in 0..n {
            let body_row = body_data.row(i);
            let obs_row = observer_data.row(i);
            let dx = body_row[0] - obs_row[0];
            let dy = body_row[1] - obs_row[1];
            let dz = body_row[2] - obs_row[2];
            distances.push((dx * dx + dy * dy + dz * dz).sqrt());
        }

        Ok(distances)
    }

    /// Compute angular radii in radians from distances and a physical radius (km)
    fn compute_angular_radii_rad(&self, radius_km: f64, distances: &[f64]) -> Vec<f64> {
        distances
            .iter()
            .map(|distance| {
                let ratio = (radius_km / distance).min(1.0);
                ratio.asin()
            })
            .collect()
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

        use numpy::{PyArray1, PyArrayMethods};

        // Get radians, then convert to degrees
        let radians_array = self.get_sun_radius_rad(py)?;
        let radians_bound = radians_array.downcast_bound::<PyArray1<f64>>(py)?;
        let radians_vec = radians_bound.readonly().as_slice()?.to_vec();

        let degrees_vec: Vec<f64> = radians_vec.iter().map(|r| r.to_degrees()).collect();
        let result: Py<PyAny> = PyArray1::from_vec(py, degrees_vec).to_owned().into();

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

        use numpy::{PyArray1, PyArrayMethods};

        // Get radians, then convert to degrees
        let radians_array = self.get_moon_radius_rad(py)?;
        let radians_bound = radians_array.downcast_bound::<PyArray1<f64>>(py)?;
        let radians_vec = radians_bound.readonly().as_slice()?.to_vec();

        let degrees_vec: Vec<f64> = radians_vec.iter().map(|r| r.to_degrees()).collect();
        let result: Py<PyAny> = PyArray1::from_vec(py, degrees_vec).to_owned().into();

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

        use numpy::{PyArray1, PyArrayMethods};

        // Get radians, then convert to degrees
        let radians_array = self.get_earth_radius_rad(py)?;
        let radians_bound = radians_array.downcast_bound::<PyArray1<f64>>(py)?;
        let radians_vec = radians_bound.readonly().as_slice()?.to_vec();

        let degrees_vec: Vec<f64> = radians_vec.iter().map(|r| r.to_degrees()).collect();
        let result: Py<PyAny> = PyArray1::from_vec(py, degrees_vec).to_owned().into();

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

        let sun_geocentric = self
            .data()
            .sun_gcrs
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No Sun data available."))?;

        let distances = self.body_observer_distances(sun_geocentric)?;
        let angular_radii = self.compute_angular_radii_rad(SUN_RADIUS_KM, &distances);

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

        let moon_geocentric =
            self.data().moon_gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No Moon data available.")
            })?;

        let distances = self.body_observer_distances(moon_geocentric)?;
        let angular_radii = self.compute_angular_radii_rad(MOON_RADIUS_KM, &distances);

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
        let distances: Vec<f64> = (0..n)
            .map(|i| {
                let row = gcrs_data.row(i);
                let x = row[0];
                let y = row[1];
                let z = row[2];
                (x * x + y * y + z * z).sqrt()
            })
            .collect();

        // Angular radius: angle from observer to visible horizon
        // For ground observers: arcsin(R_earth / distance) < π/2
        let angular_radii = self.compute_angular_radii_rad(EARTH_RADIUS_KM, &distances);

        let result: Py<PyAny> = PyArray1::from_vec(py, angular_radii).to_owned().into();

        // Cache the result
        let _ = self
            .data()
            .earth_angular_radius_rad_cache
            .set(result.clone_ref(py));

        Ok(result)
    }

    /// Helper method to extract RA/Dec from a SkyCoord object and create Nx2 array
    ///
    /// # Arguments
    /// * `py` - Python context
    /// * `cache` - Cache reference to check and store result
    /// * `skycoord` - The SkyCoord object
    /// * `unit` - Unit attribute name ("deg" or "rad")
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec
    fn build_ra_dec_array(
        &self,
        py: Python,
        cache: &OnceLock<Py<PyAny>>,
        skycoord: Py<PyAny>,
        unit: &str,
    ) -> PyResult<Py<PyAny>> {
        // Check cache first
        if let Some(cached) = cache.get() {
            return Ok(cached.clone_ref(py));
        }

        // Extract RA and Dec with specified unit
        let ra = skycoord.getattr(py, "ra")?.getattr(py, unit)?;
        let dec = skycoord.getattr(py, "dec")?.getattr(py, unit)?;

        // Convert to numpy arrays
        let ra_array = ra.bind(py);
        let dec_array = dec.bind(py);

        // Stack them into Nx2 array using numpy
        let np = py.import("numpy")?;
        let vstack = np.getattr("vstack")?;
        let stacked = vstack.call1(((ra_array, dec_array),))?;
        let result = stacked.getattr("T")?; // Transpose to get Nx2 instead of 2xN

        let result_py: Py<PyAny> = result.into();

        // Cache the result
        let _ = cache.set(result_py.clone_ref(py));

        Ok(result_py)
    }

    /// Get RA and Dec of the Sun in degrees as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in degrees and column 1 is Dec in degrees.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in degrees
    fn get_sun_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        let sun_skycoord = self.get_sun(py)?;
        self.build_ra_dec_array(py, &self.data().sun_ra_dec_deg_cache, sun_skycoord, "deg")
    }

    /// Get RA and Dec of the Moon in degrees as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in degrees and column 1 is Dec in degrees.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in degrees
    fn get_moon_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        let moon_skycoord = self.get_moon(py)?;
        self.build_ra_dec_array(py, &self.data().moon_ra_dec_deg_cache, moon_skycoord, "deg")
    }

    /// Get RA and Dec of the Earth in degrees as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in degrees and column 1 is Dec in degrees.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in degrees
    fn get_earth_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        let earth_skycoord = self.get_earth(py)?;
        self.build_ra_dec_array(
            py,
            &self.data().earth_ra_dec_deg_cache,
            earth_skycoord,
            "deg",
        )
    }

    /// Get RA and Dec of the Sun in radians as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in radians and column 1 is Dec in radians.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in radians
    fn get_sun_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        let sun_skycoord = self.get_sun(py)?;
        self.build_ra_dec_array(py, &self.data().sun_ra_dec_rad_cache, sun_skycoord, "rad")
    }

    /// Get RA and Dec of the Moon in radians as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in radians and column 1 is Dec in radians.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in radians
    fn get_moon_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        let moon_skycoord = self.get_moon(py)?;
        self.build_ra_dec_array(py, &self.data().moon_ra_dec_rad_cache, moon_skycoord, "rad")
    }

    /// Get RA and Dec of the Earth in radians as an Nx2 array
    ///
    /// Returns a NumPy array where column 0 is RA in radians and column 1 is Dec in radians.
    ///
    /// # Returns
    /// NumPy array of shape (N, 2) with RA and Dec in radians
    fn get_earth_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        let earth_skycoord = self.get_earth(py)?;
        self.build_ra_dec_array(
            py,
            &self.data().earth_ra_dec_rad_cache,
            earth_skycoord,
            "rad",
        )
    }

    /// Helper method to extract a column from an Nx2 array
    ///
    /// # Arguments
    /// * `ra_dec_array` - The Nx2 array to extract from
    /// * `column` - Column index (0 for RA, 1 for Dec)
    /// * `py` - Python context
    ///
    /// # Returns
    /// 1D NumPy array with the extracted column
    fn extract_column(
        &self,
        py: Python,
        ra_dec_array: Py<PyAny>,
        column: usize,
    ) -> PyResult<Py<PyAny>> {
        let ra_dec_bound = ra_dec_array.bind(py);
        let result = ra_dec_bound.call_method1("__getitem__", ((py.Ellipsis(), column),))?;
        Ok(result.into())
    }

    /// Get RA of the Sun in degrees as a 1D array
    ///
    /// Convenience method that extracts just the RA column from sun_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in degrees
    fn get_sun_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_sun_ra_dec_deg(py)?, 0)
    }

    /// Get Dec of the Sun in degrees as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from sun_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in degrees
    fn get_sun_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_sun_ra_dec_deg(py)?, 1)
    }

    /// Get RA of the Moon in degrees as a 1D array
    ///
    /// Convenience method that extracts just the RA column from moon_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in degrees
    fn get_moon_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_moon_ra_dec_deg(py)?, 0)
    }

    /// Get Dec of the Moon in degrees as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from moon_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in degrees
    fn get_moon_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_moon_ra_dec_deg(py)?, 1)
    }

    /// Get RA of the Earth in degrees as a 1D array
    ///
    /// Convenience method that extracts just the RA column from earth_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in degrees
    fn get_earth_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_earth_ra_dec_deg(py)?, 0)
    }

    /// Get Dec of the Earth in degrees as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from earth_ra_dec_deg.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in degrees
    fn get_earth_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_earth_ra_dec_deg(py)?, 1)
    }

    /// Get RA of the Sun in radians as a 1D array
    ///
    /// Convenience method that extracts just the RA column from sun_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in radians
    fn get_sun_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_sun_ra_dec_rad(py)?, 0)
    }

    /// Get Dec of the Sun in radians as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from sun_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in radians
    fn get_sun_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_sun_ra_dec_rad(py)?, 1)
    }

    /// Get RA of the Moon in radians as a 1D array
    ///
    /// Convenience method that extracts just the RA column from moon_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in radians
    fn get_moon_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_moon_ra_dec_rad(py)?, 0)
    }

    /// Get Dec of the Moon in radians as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from moon_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in radians
    fn get_moon_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_moon_ra_dec_rad(py)?, 1)
    }

    /// Get RA of the Earth in radians as a 1D array
    ///
    /// Convenience method that extracts just the RA column from earth_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with RA in radians
    fn get_earth_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_earth_ra_dec_rad(py)?, 0)
    }

    /// Get Dec of the Earth in radians as a 1D array
    ///
    /// Convenience method that extracts just the Dec column from earth_ra_dec_rad.
    ///
    /// # Returns
    /// NumPy array of shape (N,) with Dec in radians
    fn get_earth_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.extract_column(py, self.get_earth_ra_dec_rad(py)?, 1)
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
