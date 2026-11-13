//! Generic file-based ephemeris support
//!
//! Reads spacecraft state vectors from a file containing timestamped
//! position and velocity data in an Earth-centred coordinate frame.
//!
//! ## Supported timestamp formats
//!
//! - **Numeric offset** (default): seconds (or days) elapsed since a reference epoch.
//!   The epoch is read from header keys such as `ScenarioEpoch`, `Epoch`, or `T0`,
//!   or may be supplied via the `epoch` constructor parameter.
//! - **ISO 8601**: `YYYY-MM-DDTHH:MM:SS[.ffffff][Z]` — parsed directly as UTC.
//! - **STK natural-language date**: `DD Mon YYYY HH:MM:SS[.ffffff]`
//!   (e.g. `15 Oct 2028 00:00:00.000000`).
//!
//! ## Supported coordinate frames
//!
//! - **GCRS-compatible** (J2000, EME2000, GCRF, GCRS, ICRF, ICRF2, ICRF3): data are
//!   stored in the GCRS frame and transformed to ITRS.
//! - **Earth-fixed** (ITRS, ECEF, ECF, FIXED, TERRESTRIAL): data are stored in ITRS
//!   and transformed to GCRS.
//!
//! ## Supported position/velocity units
//!
//! Position: `"km"` (default), `"m"`, `"cm"`.
//! Velocity: `"km/s"` (default), `"m/s"`, `"cm/s"`.
//!
//! The frame, epoch and units are auto-detected from the file header when possible
//! and may be overridden via constructor parameters.

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::{prelude::*, types::PyDateTime};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{
    generate_timestamps_utc, split_pos_vel, EphemerisBase, EphemerisData,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions::{self, Frame};
use crate::utils::interpolation::hermite_interpolate;
use crate::utils::time_utils::python_datetime_to_utc;
use crate::utils::to_skycoord::AstropyModules;

// ─── Internal parsing result ──────────────────────────────────────────────────

/// State vectors as read from their source (file or in-memory arrays), before
/// unit conversion, range validation, resampling and frame transforms.
struct ParsedStates {
    times: Vec<DateTime<Utc>>,
    /// Raw state vectors (N × 6), still in the source's original units.
    states: Array2<f64>,
    frame: String,
    position_unit: String,
    velocity_unit: String,
}

// ─── Public struct ────────────────────────────────────────────────────────────

#[pyclass]
pub struct FileEphemeris {
    /// Source file path; `None` when built from in-memory arrays.
    file_path: Option<String>,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>,
    polar_motion: bool,
    common_data: EphemerisData,
    /// Raw file timestamps (before resampling to the query grid).
    file_times: Vec<DateTime<Utc>>,
    /// Raw file state vectors in km / km/s (after unit conversion, before resampling).
    file_states: Array2<f64>,
    source_position_unit: String,
    source_velocity_unit: String,
    source_frame: String,
}

// ─── Python-visible methods ───────────────────────────────────────────────────

#[pymethods]
impl FileEphemeris {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (file_path, begin, end, step_size=60, *, polar_motion=false, position_unit=None, velocity_unit=None, frame=None, epoch=None, time_format=None))]
    fn new(
        _py: Python,
        file_path: String,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        polar_motion: bool,
        position_unit: Option<String>,
        velocity_unit: Option<String>,
        frame: Option<String>,
        epoch: Option<&Bound<'_, PyDateTime>>,
        time_format: Option<String>,
    ) -> PyResult<Self> {
        let path = Path::new(&file_path);

        // Parse an explicit override epoch if supplied.
        let override_epoch: Option<DateTime<Utc>> =
            epoch.map(|e| python_datetime_to_utc(e)).transpose()?;

        let time_fmt = time_format.as_deref().unwrap_or("auto");

        // ── Parse the file ──────────────────────────────────────────────────
        let mut parsed = Self::parse_file(path, override_epoch.as_ref(), time_fmt)?;

        // ── Explicit overrides win over header-detected values ──────────────
        if let Some(unit) = position_unit {
            parsed.position_unit = unit;
        }
        if let Some(unit) = velocity_unit {
            parsed.velocity_unit = unit;
        }
        if let Some(f) = frame {
            parsed.frame = f;
        }

        let begin_dt = python_datetime_to_utc(begin)?;
        let end_dt = python_datetime_to_utc(end)?;

        Self::from_parsed(
            Some(file_path),
            parsed,
            begin_dt,
            end_dt,
            step_size,
            polar_motion,
        )
    }

    /// Build a FileEphemeris from in-memory state vectors instead of a file.
    ///
    /// Shares the post-parse pipeline of the file constructor (unit
    /// conversion, range validation, Hermite resampling onto the
    /// ``begin``/``end``/``step_size`` grid, frame transforms, Sun/Moon).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (times, positions, velocities, begin=None, end=None, step_size=60, *, polar_motion=false, position_unit="km".to_string(), velocity_unit="km/s".to_string(), frame="J2000".to_string()))]
    fn from_arrays(
        py: Python,
        times: &Bound<'_, PyAny>,
        positions: &Bound<'_, PyAny>,
        velocities: &Bound<'_, PyAny>,
        begin: Option<&Bound<'_, PyDateTime>>,
        end: Option<&Bound<'_, PyDateTime>>,
        step_size: i64,
        polar_motion: bool,
        position_unit: String,
        velocity_unit: String,
        frame: String,
    ) -> PyResult<Self> {
        let times = extract_utc_times(py, times)?;
        let pos = extract_vectors(py, positions, "positions")?;
        let vel = extract_vectors(py, velocities, "velocities")?;

        let n = times.len();
        if pos.nrows() != n || vel.nrows() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Length mismatch: {n} times, {} positions, {} velocities",
                pos.nrows(),
                vel.nrows()
            )));
        }
        if n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "At least 2 samples are required for interpolation, got {n}"
            )));
        }
        if let Some(i) = (1..n).find(|&i| times[i] <= times[i - 1]) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "times must be strictly increasing: times[{i}] = {} is not after times[{}] = {}",
                times[i].to_rfc3339(),
                i - 1,
                times[i - 1].to_rfc3339()
            )));
        }

        let mut states = Array2::<f64>::zeros((n, 6));
        states.slice_mut(s![.., 0..3]).assign(&pos);
        states.slice_mut(s![.., 3..6]).assign(&vel);

        let begin_dt = begin
            .map(|b| python_datetime_to_utc(b))
            .transpose()?
            .unwrap_or(times[0]);
        let end_dt = end
            .map(|e| python_datetime_to_utc(e))
            .transpose()?
            .unwrap_or(times[n - 1]);

        Self::from_parsed(
            None,
            ParsedStates {
                times,
                states,
                frame,
                position_unit,
                velocity_unit,
            },
            begin_dt,
            end_dt,
            step_size,
            polar_motion,
        )
    }

    // ── Type-specific properties ─────────────────────────────────────────────

    /// Path to the source file, or ``None`` when built with ``from_arrays``.
    #[getter]
    fn file_path(&self) -> Option<&str> {
        self.file_path.as_deref()
    }

    /// Whether polar motion correction is applied.
    #[getter]
    fn polar_motion(&self) -> bool {
        self.polar_motion
    }

    /// Position unit as found/specified (before conversion to km).
    #[getter]
    fn source_position_unit(&self) -> &str {
        &self.source_position_unit
    }

    /// Velocity unit as found/specified (before conversion to km/s).
    #[getter]
    fn source_velocity_unit(&self) -> &str {
        &self.source_velocity_unit
    }

    /// Coordinate frame as found/specified.
    #[getter]
    fn source_frame(&self) -> &str {
        &self.source_frame
    }

    /// Raw position/velocity from the file (km, km/s) before resampling.
    #[getter]
    fn file_pv(&self, py: Python) -> Py<PositionVelocityData> {
        Py::new(py, split_pos_vel(&self.file_states)).unwrap()
    }

    /// Raw timestamps from the file before resampling.
    #[getter]
    fn file_timestamp(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        use pyo3::types::PyTzInfo;
        let utc_tz = PyTzInfo::utc(py)?;
        self.file_times
            .iter()
            .map(|dt| {
                let pydt = PyDateTime::from_timestamp(py, dt.timestamp() as f64, Some(&utc_tz))?;
                Ok(pydt.into_any().unbind())
            })
            .collect()
    }

    // ── Common getters (delegate to EphemerisBase) ───────────────────────────

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

    #[getter]
    fn sun_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_ra_dec_deg(py)
    }

    #[getter]
    fn moon_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_ra_dec_deg(py)
    }

    #[getter]
    fn earth_ra_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_ra_dec_deg(py)
    }

    #[getter]
    fn sun_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_ra_dec_rad(py)
    }

    #[getter]
    fn moon_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_ra_dec_rad(py)
    }

    #[getter]
    fn earth_ra_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_ra_dec_rad(py)
    }

    #[getter]
    fn sun_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_ra_deg(py)
    }

    #[getter]
    fn sun_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_dec_deg(py)
    }

    #[getter]
    fn moon_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_ra_deg(py)
    }

    #[getter]
    fn moon_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_dec_deg(py)
    }

    #[getter]
    fn earth_ra_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_ra_deg(py)
    }

    #[getter]
    fn earth_dec_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_dec_deg(py)
    }

    #[getter]
    fn sun_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_ra_rad(py)
    }

    #[getter]
    fn sun_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_dec_rad(py)
    }

    #[getter]
    fn moon_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_ra_rad(py)
    }

    #[getter]
    fn moon_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_dec_rad(py)
    }

    #[getter]
    fn earth_ra_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_ra_rad(py)
    }

    #[getter]
    fn earth_dec_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_dec_rad(py)
    }

    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }

    #[pyo3(signature = (time_indices=None))]
    fn moon_illumination(&self, time_indices: Option<Vec<usize>>) -> PyResult<Vec<f64>> {
        EphemerisBase::moon_illumination(self, time_indices.as_deref())
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

    /// Get a body in observer-centred GCRS. With apparent=True, apply reception
    /// light time, solar deflection, and observer aberration using SPICE.
    /// Apparent results contain positions only; the default is geometric.
    #[pyo3(signature = (body, spice_kernel=None, use_horizons=false, *, apparent=false))]
    fn get_body(
        &self,
        py: Python,
        body: &str,
        spice_kernel: Option<String>,
        use_horizons: bool,
        apparent: bool,
    ) -> PyResult<Py<PyAny>> {
        let modules = AstropyModules::import(py)?;
        <Self as EphemerisBase>::get_body(
            self,
            py,
            &modules,
            body,
            spice_kernel.as_deref(),
            use_horizons,
            apparent,
        )
    }

    /// Convert RA/Dec to Altitude/Azimuth. Returns a NumPy array (N, 2): [alt_deg, az_deg].
    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn radec_to_altaz(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let arr = <Self as EphemerisBase>::radec_to_altaz(
            self,
            ra_deg,
            dec_deg,
            time_indices.as_deref(),
        )?;
        Ok(arr.into_pyarray(py).into())
    }

    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn calculate_airmass(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Vec<f64>> {
        <Self as EphemerisBase>::calculate_airmass(self, ra_deg, dec_deg, time_indices.as_deref())
    }
}

// ─── Private implementation ───────────────────────────────────────────────────

impl FileEphemeris {
    // ── Shared post-parse pipeline ───────────────────────────────────────────

    /// Common construction path for the file and in-memory constructors:
    /// unit conversion, range validation, query-grid generation, Hermite
    /// resampling, frame transforms and Sun/Moon positions.
    fn from_parsed(
        file_path: Option<String>,
        parsed: ParsedStates,
        begin_dt: DateTime<Utc>,
        end_dt: DateTime<Utc>,
        step_size: i64,
        polar_motion: bool,
    ) -> PyResult<Self> {
        let ParsedStates {
            times: source_times,
            states,
            frame,
            position_unit,
            velocity_unit,
        } = parsed;

        let source_desc = if file_path.is_some() { "file" } else { "input" };
        let n_source = source_times.len();
        if n_source < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "At least 2 state vectors are required for interpolation; \
                {source_desc} data contains {n_source}"
            )));
        }

        // ── Convert units to km / km/s ─────────────────────────────────────
        let source_states = Self::apply_unit_conversion(&states, &position_unit, &velocity_unit)?;

        // ── Validate requested time range ───────────────────────────────────
        if begin_dt < source_times[0] || end_dt > source_times[n_source - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Requested time range [{}, {}] exceeds {source_desc} data range [{}, {}]",
                begin_dt,
                end_dt,
                source_times[0],
                source_times[n_source - 1]
            )));
        }

        // ── Build query time grid ───────────────────────────────────────────
        let times = generate_timestamps_utc(begin_dt, end_dt, step_size)?;

        let mut ephemeris = FileEphemeris {
            file_path,
            itrs: None,
            itrs_skycoord: OnceLock::new(),
            polar_motion,
            common_data: {
                let mut data = EphemerisData::new();
                data.times = Some(times);
                data
            },
            file_times: source_times,
            file_states: source_states,
            source_position_unit: position_unit,
            source_velocity_unit: velocity_unit,
            source_frame: frame.clone(),
        };

        // ── Interpolate and transform frames ────────────────────────────────
        let frame_upper = frame.to_uppercase();
        if Self::is_gcrs_compatible(&frame_upper) {
            ephemeris.interpolate_gcrs()?;
            ephemeris.gcrs_to_itrs()?;
        } else if Self::is_itrs_compatible(&frame_upper) {
            ephemeris.interpolate_itrs()?;
            ephemeris.itrs_to_gcrs()?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported coordinate frame '{frame}'. \
                GCRS-compatible: J2000, EME2000, GCRF, GCRS, ICRF. \
                Earth-fixed: ITRS, ECEF, ECF, FIXED, TERRESTRIAL."
            )));
        }

        ephemeris.calculate_sun_moon()?;

        Ok(ephemeris)
    }

    // ── Frame classification ─────────────────────────────────────────────────

    fn is_gcrs_compatible(frame_upper: &str) -> bool {
        matches!(
            frame_upper,
            "J2000" | "EME2000" | "GCRF" | "GCRS" | "ICRF" | "ICRF2" | "ICRF3"
        )
    }

    fn is_itrs_compatible(frame_upper: &str) -> bool {
        matches!(
            frame_upper,
            "ITRS" | "ECEF" | "ECF" | "FIXED" | "TERRESTRIAL" | "EARTH_FIXED"
        )
    }

    // ── Interpolation and frame transforms ───────────────────────────────────

    fn interpolate_gcrs(&mut self) -> PyResult<()> {
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No times available for interpolation")
        })?;
        let interpolated = hermite_interpolate(times, &self.file_times, &self.file_states);
        self.common_data.gcrs = Some(interpolated);
        Ok(())
    }

    fn interpolate_itrs(&mut self) -> PyResult<()> {
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No times available for interpolation")
        })?;
        let interpolated = hermite_interpolate(times, &self.file_times, &self.file_states);
        self.itrs = Some(interpolated);
        Ok(())
    }

    fn gcrs_to_itrs(&mut self) -> PyResult<()> {
        let gcrs_data = self
            .common_data
            .gcrs
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No GCRS data available"))?;
        let times = self
            .common_data
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))?;
        let itrs = conversions::convert_frames(
            gcrs_data,
            times,
            Frame::GCRS,
            Frame::ITRS,
            self.polar_motion,
        )?;
        self.itrs = Some(itrs);
        Ok(())
    }

    fn itrs_to_gcrs(&mut self) -> PyResult<()> {
        let itrs_data = self
            .itrs
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No ITRS data available"))?;
        let times = self
            .common_data
            .times
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times available"))?;
        let gcrs = conversions::convert_frames(
            itrs_data,
            times,
            Frame::ITRS,
            Frame::GCRS,
            self.polar_motion,
        )?;
        self.common_data.gcrs = Some(gcrs);
        Ok(())
    }

    // ── Unit conversion ───────────────────────────────────────────────────────

    fn apply_unit_conversion(
        states: &Array2<f64>,
        pos_unit: &str,
        vel_unit: &str,
    ) -> PyResult<Array2<f64>> {
        let pos_lower = pos_unit.to_lowercase();
        let pos_scale: f64 = match pos_lower.trim() {
            "km" | "kilometre" | "kilometres" | "kilometer" | "kilometers" => 1.0,
            "m" | "metre" | "metres" | "meter" | "meters" => 1.0e-3,
            "cm" | "centimetre" | "centimetres" | "centimeter" | "centimeters" => 1.0e-5,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown position unit '{other}'. Supported: 'km', 'm', 'cm'"
                )))
            }
        };

        let vel_lower = vel_unit.to_lowercase();
        let vel_scale: f64 = match vel_lower.trim() {
            "km/s" => 1.0,
            "m/s" => 1.0e-3,
            "cm/s" => 1.0e-5,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown velocity unit '{other}'. Supported: 'km/s', 'm/s', 'cm/s'"
                )))
            }
        };

        if pos_scale == 1.0 && vel_scale == 1.0 {
            return Ok(states.to_owned());
        }

        let mut result = states.to_owned();
        if pos_scale != 1.0 {
            for i in 0..result.nrows() {
                for j in 0..3 {
                    result[[i, j]] *= pos_scale;
                }
            }
        }
        if vel_scale != 1.0 {
            for i in 0..result.nrows() {
                for j in 3..6 {
                    result[[i, j]] *= vel_scale;
                }
            }
        }
        Ok(result)
    }

    // ── File parsing ──────────────────────────────────────────────────────────

    /// Read and parse the ephemeris file.
    ///
    /// The parser performs two logical passes in a single read:
    /// 1. **Header phase**: collect key-value metadata until the first data-like line
    ///    or an explicit data-section marker (`EphemerisTimePosVel`, `DATA_START`, etc.).
    /// 2. **Data phase**: parse lines with 7+ whitespace-separated tokens where the
    ///    columns 2–7 are all valid floating-point numbers.
    fn parse_file(
        path: &Path,
        override_epoch: Option<&DateTime<Utc>>,
        time_format: &str,
    ) -> PyResult<ParsedStates> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to open '{}': {e}",
                path.display()
            ))
        })?;
        let reader = BufReader::new(file);

        let mut detected_epoch: Option<DateTime<Utc>> = override_epoch.copied();
        let mut detected_frame = "J2000".to_string();
        let mut detected_position_unit = "km".to_string();
        let mut detected_velocity_unit = "km/s".to_string();

        let mut in_data_section = false;
        let mut data_times: Vec<DateTime<Utc>> = Vec::new();
        let mut data_rows: Vec<[f64; 6]> = Vec::new();

        for (line_num, raw) in reader.lines().enumerate() {
            let line = raw.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to read line {}: {e}",
                    line_num + 1
                ))
            })?;
            let trimmed = line.trim();

            // ── Skip blank lines and comment lines ──────────────────────────
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with("//")
                || trimmed.starts_with('!')
            {
                continue;
            }

            let upper = trimmed.to_uppercase();

            // ── Ignore file-type header lines ───────────────────────────────
            if upper.starts_with("STK.V")
                || upper.starts_with("BEGIN EPHEMERIS")
                || upper.starts_with("WRITTEN")
                || upper.starts_with("# WRITTEN")
            {
                continue;
            }

            // ── Data-section end markers ────────────────────────────────────
            if upper.starts_with("END EPHEMERIS")
                || upper == "DATA_STOP"
                || upper.starts_with("END DATA")
            {
                in_data_section = false;
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // ── Detect whether this line looks like a state-vector row ───────
            // Columns 2–7 (0-indexed 1..7) must all be parseable as f64.
            let cols_numeric =
                parts.len() >= 7 && parts[1..7].iter().all(|s| s.parse::<f64>().is_ok());

            if in_data_section || cols_numeric {
                if cols_numeric {
                    // Auto-enter data section on first numeric row
                    in_data_section = true;

                    match Self::parse_time_column(parts[0], detected_epoch.as_ref(), time_format)? {
                        Some(t) => {
                            let state = [
                                parts[1].parse::<f64>().unwrap(),
                                parts[2].parse::<f64>().unwrap(),
                                parts[3].parse::<f64>().unwrap(),
                                parts[4].parse::<f64>().unwrap(),
                                parts[5].parse::<f64>().unwrap(),
                                parts[6].parse::<f64>().unwrap(),
                            ];
                            data_times.push(t);
                            data_rows.push(state);
                        }
                        None => {
                            // Couldn't parse time column; skip row
                        }
                    }
                }
                // Lines inside data section that don't match the format are silently skipped
                // (e.g. inline metadata, trailing END lines handled above).
                continue;
            }

            // ── Check for explicit data-section start markers ───────────────
            let first_upper = parts[0].to_uppercase();
            let is_data_marker = matches!(
                first_upper.as_str(),
                "EPHEMERISTIMEPOSVEL" | "EPHEMERISTIMEPOS" | "DATA_START" | "BEGINDATA"
            ) || (first_upper == "BEGIN"
                && parts.len() > 1
                && parts[1].to_uppercase() == "DATA");

            if is_data_marker {
                in_data_section = true;
                continue;
            }

            // ── Header key-value parsing ─────────────────────────────────────
            if parts.len() >= 2 {
                let key = parts[0].to_lowercase();
                let value = parts[1..].join(" ");

                // Epoch / reference time
                if matches!(
                    key.as_str(),
                    "scenarioepoch"
                        | "epoch"
                        | "t0"
                        | "start_epoch"
                        | "begin_epoch"
                        | "reference_epoch"
                        | "begintime"
                        | "centralepoch"
                ) && override_epoch.is_none()
                    && detected_epoch.is_none()
                {
                    if let Some(dt) = Self::parse_epoch_string(&value) {
                        detected_epoch = Some(dt);
                    }
                }

                // Coordinate frame
                if matches!(
                    key.as_str(),
                    "coordinatesystem"
                        | "ref_frame"
                        | "frame"
                        | "coordinate_system"
                        | "coordsystem"
                        | "referencesystem"
                ) && !value.is_empty()
                {
                    detected_frame = value.clone();
                }

                // Units (rarely stated explicitly, but handle when present)
                if matches!(
                    key.as_str(),
                    "distanceunit" | "position_unit" | "positionunit" | "lengthunit"
                ) {
                    detected_position_unit = value.clone();
                }
                if matches!(key.as_str(), "velocityunit" | "velocity_unit" | "speedunit") {
                    detected_velocity_unit = value.clone();
                }
            }
        }

        if data_times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No state vector data found in file. \
                Expected lines with at least 7 whitespace-separated values: \
                <time> <x> <y> <z> <vx> <vy> <vz>",
            ));
        }

        // Build ndarray from collected rows
        let n = data_times.len();
        let mut states = Array2::<f64>::zeros((n, 6));
        for (i, row) in data_rows.iter().enumerate() {
            for j in 0..6 {
                states[[i, j]] = row[j];
            }
        }

        Ok(ParsedStates {
            times: data_times,
            states,
            frame: detected_frame,
            position_unit: detected_position_unit,
            velocity_unit: detected_velocity_unit,
        })
    }

    /// Parse the time column of a data row according to `time_format`.
    ///
    /// Returns `Ok(None)` when the value cannot be interpreted (row should be skipped).
    fn parse_time_column(
        s: &str,
        epoch: Option<&DateTime<Utc>>,
        time_format: &str,
    ) -> PyResult<Option<DateTime<Utc>>> {
        match time_format {
            "seconds" => {
                let offset = s.parse::<f64>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Cannot parse '{s}' as a numeric time offset in seconds"
                    ))
                })?;
                let ep = epoch.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "time_format='seconds' requires an epoch. \
                        Add a 'ScenarioEpoch' header or supply the 'epoch' parameter.",
                    )
                })?;
                Ok(Some(
                    *ep + chrono::Duration::microseconds((offset * 1e6) as i64),
                ))
            }
            "days" => {
                let offset = s.parse::<f64>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Cannot parse '{s}' as a numeric time offset in days"
                    ))
                })?;
                let ep = epoch.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "time_format='days' requires an epoch. \
                        Add a 'ScenarioEpoch' header or supply the 'epoch' parameter.",
                    )
                })?;
                Ok(Some(
                    *ep + chrono::Duration::microseconds((offset * 86400.0 * 1e6) as i64),
                ))
            }
            "iso8601" => match Self::parse_epoch_string(s) {
                Some(dt) => Ok(Some(dt)),
                None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot parse '{s}' as an ISO 8601 or STK datetime"
                ))),
            },
            _ => {
                // Auto-detect: try numeric offset first, then datetime string.
                if let Ok(offset) = s.parse::<f64>() {
                    let ep = epoch.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "File uses numeric time offsets (e.g. '{s}') but no epoch was \
                            found in the file header. Add a 'ScenarioEpoch' header or supply \
                            the 'epoch' parameter."
                        ))
                    })?;
                    Ok(Some(
                        *ep + chrono::Duration::microseconds((offset * 1e6) as i64),
                    ))
                } else if let Some(dt) = Self::parse_epoch_string(s) {
                    Ok(Some(dt))
                } else {
                    // Unrecognised format: skip this row
                    Ok(None)
                }
            }
        }
    }

    /// Try to parse a string as a UTC datetime using multiple common formats.
    ///
    /// Supported formats (tried in order):
    /// 1. STK: `DD Mon YYYY HH:MM:SS.ffffff` (e.g. `15 Oct 2028 00:00:00.000000`)
    /// 2. STK: `DD Mon YYYY HH:MM:SS`
    /// 3. STK date-only: `DD Mon YYYY`
    /// 4. RFC 3339 / ISO 8601 with timezone: `YYYY-MM-DDTHH:MM:SS±HH:MM`
    /// 5. ISO 8601 T-separator, no timezone: `YYYY-MM-DDTHH:MM:SS.ffffff`
    /// 6. ISO 8601 T-separator, no fractional: `YYYY-MM-DDTHH:MM:SS`
    /// 7. ISO 8601 space-separator: `YYYY-MM-DD HH:MM:SS.ffffff`
    /// 8. ISO 8601 space-separator, no fractional: `YYYY-MM-DD HH:MM:SS`
    fn parse_epoch_string(s: &str) -> Option<DateTime<Utc>> {
        let s = s.trim();

        // STK: "15 Oct 2028 00:00:00.000000"
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%d %b %Y %H:%M:%S%.f") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
        // STK: "15 Oct 2028 00:00:00"
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%d %b %Y %H:%M:%S") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
        // STK date-only: "15 Oct 2028"
        if let Ok(d) = NaiveDate::parse_from_str(s, "%d %b %Y") {
            if let Some(t) = NaiveTime::from_hms_opt(0, 0, 0) {
                let dt = NaiveDateTime::new(d, t);
                return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
            }
        }
        // RFC 3339 (with timezone)
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Some(dt.with_timezone(&Utc));
        }
        // ISO 8601 with T, fractional seconds
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
        // ISO 8601 with T, no fractional
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
        // ISO 8601 space-separator, fractional
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
        // ISO 8601 space-separator, no fractional
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }

        None
    }
}

// ─── In-memory input helpers ──────────────────────────────────────────────────

/// Convert a sequence of datetimes (naive = UTC) or a NumPy ``datetime64``
/// array to UTC timestamps.
fn extract_utc_times(py: Python, obj: &Bound<'_, PyAny>) -> PyResult<Vec<DateTime<Utc>>> {
    let is_datetime64 = obj
        .getattr("dtype")
        .and_then(|dt| dt.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "M")
        .unwrap_or(false);

    if is_datetime64 {
        let np = py.import("numpy")?;
        let micros = np
            .call_method1("asarray", (obj,))?
            .call_method1("astype", ("datetime64[us]",))?
            .call_method1("astype", ("int64",))?
            .call_method1("ravel", ())?;
        let arr = micros.cast::<PyArray1<i64>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Could not convert datetime64 times to int64")
        })?;
        return arr
            .readonly()
            .as_array()
            .iter()
            .enumerate()
            .map(|(i, &us)| {
                if us == i64::MIN {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "times[{i}] is NaT"
                    )));
                }
                DateTime::<Utc>::from_timestamp_micros(us).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("times[{i}] is out of range"))
                })
            })
            .collect();
    }

    let iter = pyo3::types::PyIterator::from_object(obj).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "times must be a sequence of datetime objects or a numpy datetime64 array",
        )
    })?;
    iter.enumerate()
        .map(|(i, item)| {
            python_datetime_to_utc(&item?).map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("times[{i}] is not a datetime: {e}"))
            })
        })
        .collect()
}

/// Convert an (N, 3) array-like to a finite float64 array.
fn extract_vectors(py: Python, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Array2<f64>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, "float64")).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name} could not be converted to a float array: {e}"
        ))
    })?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, 3), got {shape:?}"
        )));
    }
    let arr = arr.cast::<PyArray2<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("{name} is not a float64 array"))
    })?;
    let values = arr.readonly().as_array().to_owned();
    if let Some(((i, j), v)) = values.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} contains a non-finite value ({v}) at [{i}, {j}]"
        )));
    }
    Ok(values)
}

// ─── EphemerisBase trait implementation ───────────────────────────────────────

impl EphemerisBase for FileEphemeris {
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
    ) -> PyResult<Array2<f64>> {
        crate::utils::celestial::radec_to_altaz(ra_deg, dec_deg, self, time_indices)
    }
}
