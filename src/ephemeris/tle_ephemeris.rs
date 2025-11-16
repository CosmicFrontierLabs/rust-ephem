use ndarray::Array2;
use pyo3::{prelude::*, types::PyDateTime};
use sgp4::{parse_2les, Constants};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{
    generate_timestamps, split_pos_vel, EphemerisBase, EphemerisData,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::to_skycoord::AstropyModules;

#[pyclass]
pub struct TLEEphemeris {
    tle1: String,
    tle2: String,
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
    #[pyo3(signature = (tle1, tle2, begin, end, step_size=60, *, polar_motion=false))]
    fn new(
        _py: Python,
        tle1: String,
        tle2: String,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        polar_motion: bool,
    ) -> PyResult<Self> {
        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size)?;

        // Create the TLEEphemeris object
        let mut ephemeris: TLEEphemeris = TLEEphemeris {
            tle1,
            tle2,
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

    #[getter]
    fn gcrs_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_gcrs_pv(py)
    }

    // Getter for times but call it timestamp and convert to python datetime
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
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

    /// Get Sun position and velocity in GCRS frame
    #[getter]
    fn sun_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_sun_pv(py)
    }

    /// Get Moon position and velocity in GCRS frame
    #[getter]
    fn moon_pv(&self, py: Python) -> Option<Py<PositionVelocityData>> {
        self.get_moon_pv(py)
    }

    /// Get observer geocentric location (obsgeoloc) - alias for GCRS position
    /// This is compatible with astropy's GCRS frame obsgeoloc parameter
    #[getter]
    fn obsgeoloc(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_obsgeoloc(py)
    }

    /// Get observer geocentric velocity (obsgeovel) - alias for GCRS velocity
    /// This is compatible with astropy's GCRS frame obsgeovel parameter
    #[getter]
    fn obsgeovel(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        // Delegates to get_obsgeovel to return the observer's geocentric velocity.
        // Ensures compatibility with astropy's GCRS frame obsgeovel parameter.
        self.get_obsgeovel(py)
    }

    /// Get position and velocity for any solar system body
    ///
    /// Analogous to astropy's `get_body()` function. Returns position and velocity
    /// of the specified body relative to the observer (spacecraft).
    ///
    /// # Arguments
    /// * `body` - Body identifier: NAIF ID (as string) or name (e.g., "Jupiter", "mars", "301")
    ///
    /// # Returns
    /// `PositionVelocityData` containing position and velocity in km and km/s
    ///
    /// # Example
    /// ```python
    /// eph = TLEEphemeris(...)
    /// jupiter = eph.get_body("Jupiter")  # By name
    /// mars = eph.get_body("499")  # By NAIF ID
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
    /// eph = TLEEphemeris(...)
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
    /// Returns an astropy Quantity with units of degrees.
    ///
    /// # Returns
    /// astropy Quantity array with units of degrees
    #[getter]
    fn sun_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius(py)
    }

    /// Get angular radius of the Sun as seen from the spacecraft (in degrees)
    ///
    /// Returns a NumPy array of angular radii for each timestamp.
    /// Angular radius = arcsin(physical_radius / distance)
    ///
    /// # Returns
    /// NumPy array of angular radii in degrees
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

    /// Get angular radius of the Moon as seen from the spacecraft (in degrees)
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

    /// Get angular radius of the Earth as seen from the spacecraft (in degrees)
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

    /// Get angular radius of the Sun as seen from the spacecraft (in radians)
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

    /// Get angular radius of the Moon as seen from the spacecraft (in radians)
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

    /// Get angular radius of the Earth as seen from the spacecraft (in radians)
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
    /// eph = TLEEphemeris(...)
    /// target_time = datetime(2024, 1, 15, 12, 0, 0)
    /// idx = eph.index(target_time)
    /// position = eph.gcrs_pv.position[idx]
    /// ```
    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
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
}
