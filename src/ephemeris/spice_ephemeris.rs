use anise::prelude::*;
use chrono::{Datelike, Timelike};
use hifitime::Epoch as HifiEpoch;
use ndarray::Array2;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{generate_timestamps, EphemerisBase, EphemerisData};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::to_skycoord::AstropyModules;

#[pyclass]
pub struct SPICEEphemeris {
    spk_path: String,
    naif_id: i32,
    center_id: i32,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>, // Lazy-initialized cached SkyCoord object for ITRS
    polar_motion: bool,                 // Whether to apply polar motion correction
    // Common ephemeris data
    common_data: EphemerisData,
}

#[pymethods]
impl SPICEEphemeris {
    #[new]
    #[pyo3(signature = (spk_path, naif_id, begin, end, step_size=60, center_id=399, *, polar_motion=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        spk_path: String,
        naif_id: i32,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        center_id: i32,
        polar_motion: bool,
    ) -> PyResult<Self> {
        // Use common timestamp generation logic
        let times = generate_timestamps(begin, end, step_size)?;

        // Create the SPICEEphemeris object
        let mut ephemeris = SPICEEphemeris {
            spk_path,
            naif_id,
            center_id,
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
        ephemeris.propagate_to_gcrs()?;
        ephemeris.gcrs_to_itrs()?;
        ephemeris.calculate_sun_moon()?;

        // Note: SkyCoords are now created lazily on first access

        // Return the SPICEEphemeris object
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

    // Getter for times but call it timestamp and convert to python datetime
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
    }

    /// propagate_to_gcrs() -> np.ndarray
    ///
    /// Propagates the object using SPICE kernels to the times specified during initialization.
    /// Returns [x,y,z,vx,vy,vz] in GCRS coordinates (km, km/s).
    fn propagate_to_gcrs(&mut self) -> PyResult<()> {
        // Get the internally stored times
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "SPICEEphemeris object was not properly initialized. Please create a new SPICEEphemeris instance with begin, end, and step_size parameters.",
            )
        })?;

        // Load the SPK file locally from the provided path (spacecraft-specific Almanac)
        let spk = SPK::load(&self.spk_path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to load SPK file: {e:?}"))
        })?;
        let almanac = Almanac::default().with_spk(spk);

        // Prepare output array
        let n = times.len();
        let mut out = Array2::<f64>::zeros((n, 6));

        for (i, dt) in times.iter().enumerate() {
            // Convert DateTime<Utc> to hifitime Epoch
            // hifitime expects UTC time
            let epoch = HifiEpoch::from_gregorian_utc(
                dt.year(),
                dt.month() as u8,
                dt.day() as u8,
                dt.hour() as u8,
                dt.minute() as u8,
                dt.second() as u8,
                dt.timestamp_subsec_nanos(),
            );

            // Query the almanac for the state (position and velocity)
            // Create Frame objects from NAIF IDs
            // Use J2000 orientation which is essentially GCRS
            let target_frame = Frame::from_ephem_j2000(self.naif_id);
            let observer_frame = Frame::from_ephem_j2000(self.center_id);

            // almanac_holder is Arc<Almanac>, deref to Almanac for method call
            let state = almanac
                .translate_geometric(target_frame, observer_frame, epoch)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to translate state at {dt}: {e:?}"
                    ))
                })?;

            // Extract position and velocity
            // ANISE returns position in km and velocity in km/s
            let mut row = out.row_mut(i);
            row[0] = state.radius_km.x;
            row[1] = state.radius_km.y;
            row[2] = state.radius_km.z;
            row[3] = state.velocity_km_s.x;
            row[4] = state.velocity_km_s.y;
            row[5] = state.velocity_km_s.z;
        }

        // Store results
        self.common_data.gcrs = Some(out);
        Ok(())
    }

    /// gcrs_to_itrs() -> ()
    ///
    /// Converts the stored GCRS coordinates to ITRS (Earth-fixed) coordinates.
    /// Requires propagate_to_gcrs to be called first.
    fn gcrs_to_itrs(&mut self) -> PyResult<()> {
        // Access stored GCRS data
        let gcrs_data = self.common_data.gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No GCRS data available. Call propagate_to_gcrs first.",
            )
        })?;

        // Use stored times
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "No times available. Call propagate_to_gcrs first.",
            )
        })?;

        // Check lengths match
        if gcrs_data.nrows() != times.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of times must match number of GCRS rows",
            ));
        }

        // Use the generic conversion function
        let itrs_result = conversions::convert_frames(
            gcrs_data,
            times,
            conversions::Frame::GCRS,
            conversions::Frame::ITRS,
            self.polar_motion,
        );
        self.itrs = Some(itrs_result);
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
    /// eph = SPICEEphemeris(...)
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
    /// eph = SPICEEphemeris(...)
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
    /// Returns an astropy Quantity array with units of degrees for each timestamp.
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
    /// Returns an astropy Quantity array with units of degrees for each timestamp.
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

    /// Get angular radius of the Moon as seen from the observer (in degrees)
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

    /// Get angular radius of the Earth as seen from the observer (in degrees)
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

    /// Get angular radius of the Sun as seen from the observer (in radians)
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

    /// Get angular radius of the Moon as seen from the observer (in radians)
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

    /// Get angular radius of the Earth as seen from the observer (in radians)
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
    /// eph = SPICEEphemeris(...)
    /// target_time = datetime(2024, 1, 15, 12, 0, 0)
    /// idx = eph.index(target_time)
    /// position = eph.gcrs_pv.position[idx]
    /// ```
    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }
}

// Implement the EphemerisBase trait for SPICEEphemeris
impl EphemerisBase for SPICEEphemeris {
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
