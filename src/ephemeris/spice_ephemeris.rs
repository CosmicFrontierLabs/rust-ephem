use anise::prelude::*;
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyDateTime};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{generate_timestamps, EphemerisBase, EphemerisData};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::time_utils::chrono_to_epoch;
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

    // ===== Type-specific getters =====

    /// Get the SPK file path
    #[getter]
    fn spk_path(&self) -> &str {
        &self.spk_path
    }

    /// Get the NAIF ID of the target body
    #[getter]
    fn naif_id(&self) -> i32 {
        self.naif_id
    }

    /// Get the NAIF ID of the center body
    #[getter]
    fn center_id(&self) -> i32 {
        self.center_id
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

    /// Convert RA/Dec to Altitude/Azimuth for this SPICE ephemeris
    /// Returns NumPy array (N,2): [altitude_deg, azimuth_deg]
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn radec_to_altaz(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let arr = <Self as crate::ephemeris::ephemeris_common::EphemerisBase>::radec_to_altaz(
            self,
            ra_deg,
            dec_deg,
            time_indices.as_deref(),
        );
        Ok(arr.into_pyarray(py).into())
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
    fn moon_illumination(&self, time_indices: Option<Vec<usize>>) -> PyResult<Vec<f64>> {
        EphemerisBase::moon_illumination(self, time_indices.as_deref())
    }

    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }

    #[pyo3(signature = (body, spice_kernel=None, use_horizons=false))]
    fn get_body_pv(
        &self,
        py: Python,
        body: &str,
        spice_kernel: Option<String>,
        use_horizons: bool,
    ) -> PyResult<Py<PositionVelocityData>> {
        <Self as EphemerisBase>::get_body_pv(self, py, body, spice_kernel.as_deref(), use_horizons)
    }

    #[pyo3(signature = (body, spice_kernel=None, use_horizons=false))]
    fn get_body(
        &self,
        py: Python,
        body: &str,
        spice_kernel: Option<String>,
        use_horizons: bool,
    ) -> PyResult<Py<PyAny>> {
        let modules = AstropyModules::import(py)?;
        <Self as EphemerisBase>::get_body(
            self,
            py,
            &modules,
            body,
            spice_kernel.as_deref(),
            use_horizons,
        )
    }

    // ===== Type-specific methods =====

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
            let epoch = chrono_to_epoch(dt);

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

    fn radec_to_altaz(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<&[usize]>,
    ) -> Array2<f64> {
        crate::utils::celestial::radec_to_altaz(ra_deg, dec_deg, self, time_indices)
    }
}
