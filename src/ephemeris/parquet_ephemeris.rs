//! Parquet-based ephemeris support via DuckDB.
//!
//! Loads spacecraft state vectors from a Parquet file (local or cloud-hosted)
//! using the Python `duckdb` package, which handles:
//!
//! - Local files: any `path/to/file.parquet` or glob like `path/sat_*.parquet`
//! - S3: `s3://bucket/key.parquet` (credentials via standard AWS env vars)
//! - DigitalOcean Spaces and other S3-compatibles: pass `s3_endpoint=...`
//! - HTTPS: `https://host/path.parquet` (public buckets)
//!
//! The Parquet file must contain at minimum a timestamp column and three
//! position + three velocity columns. Column names default to `time`,
//! `x/y/z`, `vx/vy/vz` and are overridable via constructor kwargs.
//!
//! Authentication uses DuckDB's `credential_chain` SECRET, which picks up
//! `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and
//! `AWS_SESSION_TOKEN` from the environment. For S3-compatible services
//! (e.g. DigitalOcean Spaces), pass `s3_endpoint`.

use chrono::{DateTime, TimeZone, Utc};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{
    prelude::*,
    types::{PyDateTime, PyDict, PyTuple},
};
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{
    generate_timestamps, split_pos_vel, EphemerisBase, EphemerisData,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions::{self, Frame};
use crate::utils::interpolation::hermite_interpolate;
use crate::utils::time_utils::python_datetime_to_utc;
use crate::utils::to_skycoord::AstropyModules;

// ─── Defaults ─────────────────────────────────────────────────────────────────

const DEFAULT_TIME_COL: &str = "time";
const DEFAULT_POS_COLS: [&str; 3] = ["x", "y", "z"];
const DEFAULT_VEL_COLS: [&str; 3] = ["vx", "vy", "vz"];

/// Seconds of margin to add on either side of [begin, end] when pre-filtering
/// the Parquet so Hermite interpolation has neighbours at the boundaries.
const TIME_FILTER_MARGIN_SECS: i64 = 3600;

// ─── Public struct ────────────────────────────────────────────────────────────

#[pyclass]
pub struct ParquetEphemeris {
    source: String,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>,
    polar_motion: bool,
    common_data: EphemerisData,
    /// Raw timestamps from the Parquet (after time-range filter, before resampling).
    file_times: Vec<DateTime<Utc>>,
    /// Raw state vectors in km / km/s (after unit conversion, before resampling).
    file_states: Array2<f64>,
    source_position_unit: String,
    source_velocity_unit: String,
    source_frame: String,
}

// ─── Python-visible methods ───────────────────────────────────────────────────

#[pymethods]
impl ParquetEphemeris {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        source, begin, end, step_size=60, *,
        polar_motion=false,
        time_col=None,
        pos_cols=None,
        vel_cols=None,
        position_unit=None,
        velocity_unit=None,
        frame=None,
        s3_endpoint=None,
        s3_region=None,
        where_clause=None,
    ))]
    fn new(
        py: Python,
        source: String,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        polar_motion: bool,
        time_col: Option<String>,
        pos_cols: Option<(String, String, String)>,
        vel_cols: Option<(String, String, String)>,
        position_unit: Option<String>,
        velocity_unit: Option<String>,
        frame: Option<String>,
        s3_endpoint: Option<String>,
        s3_region: Option<String>,
        where_clause: Option<String>,
    ) -> PyResult<Self> {
        // ── Resolve column names ────────────────────────────────────────────
        let time_col = time_col.unwrap_or_else(|| DEFAULT_TIME_COL.to_string());
        let pos_cols = pos_cols.unwrap_or_else(|| {
            (
                DEFAULT_POS_COLS[0].to_string(),
                DEFAULT_POS_COLS[1].to_string(),
                DEFAULT_POS_COLS[2].to_string(),
            )
        });
        let vel_cols = vel_cols.unwrap_or_else(|| {
            (
                DEFAULT_VEL_COLS[0].to_string(),
                DEFAULT_VEL_COLS[1].to_string(),
                DEFAULT_VEL_COLS[2].to_string(),
            )
        });

        // Validate column names to prevent SQL injection.
        validate_identifier(&time_col, "time_col")?;
        validate_identifier(&pos_cols.0, "pos_cols[0]")?;
        validate_identifier(&pos_cols.1, "pos_cols[1]")?;
        validate_identifier(&pos_cols.2, "pos_cols[2]")?;
        validate_identifier(&vel_cols.0, "vel_cols[0]")?;
        validate_identifier(&vel_cols.1, "vel_cols[1]")?;
        validate_identifier(&vel_cols.2, "vel_cols[2]")?;

        let begin_dt = python_datetime_to_utc(begin)?;
        let end_dt = python_datetime_to_utc(end)?;

        // ── Pull data via DuckDB ────────────────────────────────────────────
        let (raw_times, raw_states) = load_via_duckdb(
            py,
            &source,
            &time_col,
            &pos_cols,
            &vel_cols,
            &begin_dt,
            &end_dt,
            s3_endpoint.as_deref(),
            s3_region.as_deref(),
            where_clause.as_deref(),
        )?;

        if raw_times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Parquet source '{source}' returned no rows for [{begin_dt}, {end_dt}]"
            )));
        }

        // ── Resolve effective units and frame ───────────────────────────────
        let eff_pos_unit = position_unit.unwrap_or_else(|| "km".to_string());
        let eff_vel_unit = velocity_unit.unwrap_or_else(|| "km/s".to_string());
        let eff_frame = frame.unwrap_or_else(|| "GCRS".to_string());

        // ── Convert units to km / km/s ──────────────────────────────────────
        let file_states = apply_unit_conversion(&raw_states, &eff_pos_unit, &eff_vel_unit)?;

        // ── Validate requested time range against returned data ─────────────
        let n_file = raw_times.len();
        if begin_dt < raw_times[0] || end_dt > raw_times[n_file - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Requested time range [{}, {}] is not fully covered by Parquet data \
                 [{}, {}] (after applying time filter with {}s margin). Either widen \
                 the source data or shrink the requested range.",
                begin_dt,
                end_dt,
                raw_times[0],
                raw_times[n_file - 1],
                TIME_FILTER_MARGIN_SECS
            )));
        }

        // ── Build query time grid ───────────────────────────────────────────
        let times = generate_timestamps(begin, end, step_size)?;

        let mut ephemeris = ParquetEphemeris {
            source,
            itrs: None,
            itrs_skycoord: OnceLock::new(),
            polar_motion,
            common_data: {
                let mut data = EphemerisData::new();
                data.times = Some(times);
                data
            },
            file_times: raw_times,
            file_states,
            source_position_unit: eff_pos_unit,
            source_velocity_unit: eff_vel_unit,
            source_frame: eff_frame.clone(),
        };

        // ── Interpolate and transform frames ────────────────────────────────
        let frame_upper = eff_frame.to_uppercase();
        if is_gcrs_compatible(&frame_upper) {
            ephemeris.interpolate_gcrs()?;
            ephemeris.gcrs_to_itrs()?;
        } else if is_itrs_compatible(&frame_upper) {
            ephemeris.interpolate_itrs()?;
            ephemeris.itrs_to_gcrs()?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported coordinate frame '{eff_frame}'. \
                GCRS-compatible: J2000, EME2000, GCRF, GCRS, ICRF. \
                Earth-fixed: ITRS, ECEF, ECF, FIXED, TERRESTRIAL."
            )));
        }

        ephemeris.calculate_sun_moon()?;

        Ok(ephemeris)
    }

    // ── Type-specific properties ─────────────────────────────────────────────

    /// Source path or URI for the Parquet data.
    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    /// Alias for `source` for symmetry with `FileEphemeris.file_path`.
    #[getter]
    fn file_path(&self) -> &str {
        &self.source
    }

    #[getter]
    fn polar_motion(&self) -> bool {
        self.polar_motion
    }

    #[getter]
    fn source_position_unit(&self) -> &str {
        &self.source_position_unit
    }

    #[getter]
    fn source_velocity_unit(&self) -> &str {
        &self.source_velocity_unit
    }

    #[getter]
    fn source_frame(&self) -> &str {
        &self.source_frame
    }

    /// Raw position/velocity from the Parquet (km, km/s) before resampling.
    #[getter]
    fn file_pv(&self, py: Python) -> Py<PositionVelocityData> {
        Py::new(py, split_pos_vel(&self.file_states)).unwrap()
    }

    /// Raw timestamps from the Parquet before resampling.
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

    #[pyo3(signature = (ra_deg, dec_deg, time_indices=None))]
    fn radec_to_altaz(
        &self,
        py: Python,
        ra_deg: f64,
        dec_deg: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let arr =
            <Self as EphemerisBase>::radec_to_altaz(self, ra_deg, dec_deg, time_indices.as_deref());
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

impl ParquetEphemeris {
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
        );
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
        );
        self.common_data.gcrs = Some(gcrs);
        Ok(())
    }
}

// ─── EphemerisBase trait implementation ───────────────────────────────────────

impl EphemerisBase for ParquetEphemeris {
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

/// Validate that a string is a safe SQL identifier (alphanumeric + underscore).
///
/// Column names are interpolated into SQL (DuckDB doesn't support binding
/// identifiers via `?` placeholders), so we restrict them to a safe alphabet
/// to prevent injection.
fn validate_identifier(name: &str, label: &str) -> PyResult<()> {
    if name.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must not be empty"
        )));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} '{name}' is not a valid identifier (allowed: ASCII alphanumeric and '_')"
        )));
    }
    Ok(())
}

/// Quote a SQL identifier with double quotes (safe — already validated).
fn quote_ident(name: &str) -> String {
    format!("\"{name}\"")
}

/// Load Parquet data via the Python `duckdb` package.
///
/// Returns (timestamps, Nx6 state matrix in source units).
#[allow(clippy::too_many_arguments)]
fn load_via_duckdb(
    py: Python,
    source: &str,
    time_col: &str,
    pos_cols: &(String, String, String),
    vel_cols: &(String, String, String),
    begin: &DateTime<Utc>,
    end: &DateTime<Utc>,
    s3_endpoint: Option<&str>,
    s3_region: Option<&str>,
    where_clause: Option<&str>,
) -> PyResult<(Vec<DateTime<Utc>>, Array2<f64>)> {
    let duckdb = py.import("duckdb").map_err(|e| {
        pyo3::exceptions::PyImportError::new_err(format!(
            "Failed to import duckdb. Install with `pip install rust-ephem[parquet]` \
             or `pip install duckdb`. Underlying error: {e}"
        ))
    })?;

    let conn = duckdb.call_method0("connect")?;

    // Configure cloud access if the source looks remote.
    let is_remote = source.starts_with("s3://")
        || source.starts_with("https://")
        || source.starts_with("http://")
        || source.starts_with("gcs://")
        || source.starts_with("r2://");

    // Run all queries in UTC so TIMESTAMP/TIMESTAMPTZ comparisons are unambiguous.
    conn.call_method1("execute", ("SET TIMEZONE = 'UTC';",))?;

    if is_remote {
        conn.call_method1("execute", ("INSTALL httpfs;",))?;
        conn.call_method1("execute", ("LOAD httpfs;",))?;

        // Build a credential_chain SECRET that picks up AWS_* env vars.
        let mut secret_sql = String::from(
            "CREATE OR REPLACE SECRET rust_ephem_s3 (TYPE S3, PROVIDER credential_chain",
        );
        if let Some(endpoint) = s3_endpoint {
            // Strip protocol if present (DuckDB wants a bare host).
            let host = endpoint
                .trim_start_matches("https://")
                .trim_start_matches("http://");
            secret_sql.push_str(&format!(", ENDPOINT '{}'", escape_sql_literal(host)));
        }
        if let Some(region) = s3_region {
            secret_sql.push_str(&format!(", REGION '{}'", escape_sql_literal(region)));
        }
        secret_sql.push(')');
        conn.call_method1("execute", (secret_sql,))?;
    }

    // Build the SELECT.
    let t_q = quote_ident(time_col);
    let px = quote_ident(&pos_cols.0);
    let py_ = quote_ident(&pos_cols.1);
    let pz = quote_ident(&pos_cols.2);
    let vx = quote_ident(&vel_cols.0);
    let vy = quote_ident(&vel_cols.1);
    let vz = quote_ident(&vel_cols.2);

    let extra_where = match where_clause {
        Some(w) if !w.trim().is_empty() => format!(" AND ({w})"),
        _ => String::new(),
    };

    // Filter and select using epoch microseconds (BIGINT) so timezone-aware and
    // timezone-naive timestamp columns behave the same. `epoch_us` in DuckDB
    // accepts TIMESTAMP, TIMESTAMPTZ, and DATE.
    let sql = format!(
        "SELECT \
            CAST(epoch_us({t_q}) AS BIGINT) AS __t_us, \
            CAST({px} AS DOUBLE) AS __x, \
            CAST({py_} AS DOUBLE) AS __y, \
            CAST({pz} AS DOUBLE) AS __z, \
            CAST({vx} AS DOUBLE) AS __vx, \
            CAST({vy} AS DOUBLE) AS __vy, \
            CAST({vz} AS DOUBLE) AS __vz \
         FROM read_parquet(?) \
         WHERE epoch_us({t_q}) BETWEEN ? AND ?{extra_where} \
         ORDER BY __t_us"
    );

    // Pull a margin around [begin, end] so Hermite interpolation has neighbours.
    let margin = chrono::Duration::seconds(TIME_FILTER_MARGIN_SECS);
    let begin_us =
        (*begin - margin).timestamp() * 1_000_000 + (begin.timestamp_subsec_micros() as i64);
    let end_us = (*end + margin).timestamp() * 1_000_000 + (end.timestamp_subsec_micros() as i64);

    let params = PyTuple::new(
        py,
        [
            source.into_pyobject(py)?.into_any(),
            begin_us.into_pyobject(py)?.into_any(),
            end_us.into_pyobject(py)?.into_any(),
        ],
    )?;
    let result = conn.call_method1("execute", (sql, params))?;

    // fetchnumpy() returns a dict[str, np.ndarray] keyed by column alias.
    let np_dict_obj = result.call_method0("fetchnumpy")?;
    let np_dict = np_dict_obj.downcast::<PyDict>()?;

    let times = extract_int64_column(np_dict, "__t_us")?;
    let xs = extract_f64_column(np_dict, "__x")?;
    let ys = extract_f64_column(np_dict, "__y")?;
    let zs = extract_f64_column(np_dict, "__z")?;
    let vxs = extract_f64_column(np_dict, "__vx")?;
    let vys = extract_f64_column(np_dict, "__vy")?;
    let vzs = extract_f64_column(np_dict, "__vz")?;

    let n = times.len();
    if [
        xs.len(),
        ys.len(),
        zs.len(),
        vxs.len(),
        vys.len(),
        vzs.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "DuckDB returned columns of inconsistent length",
        ));
    }

    let mut datetimes: Vec<DateTime<Utc>> = Vec::with_capacity(n);
    for &us in &times {
        let dt = Utc.timestamp_micros(us).single().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Could not convert epoch microseconds {us} to datetime"
            ))
        })?;
        datetimes.push(dt);
    }

    let mut states = Array2::<f64>::zeros((n, 6));
    for i in 0..n {
        states[[i, 0]] = xs[i];
        states[[i, 1]] = ys[i];
        states[[i, 2]] = zs[i];
        states[[i, 3]] = vxs[i];
        states[[i, 4]] = vys[i];
        states[[i, 5]] = vzs[i];
    }

    Ok((datetimes, states))
}

fn extract_f64_column(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let item = dict.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("DuckDB result missing column '{key}'"))
    })?;
    let arr = item.downcast::<PyArray1<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "Column '{key}' is not a float64 array (DuckDB CAST may have failed)"
        ))
    })?;
    Ok(arr.readonly().as_slice()?.to_vec())
}

fn extract_int64_column(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<i64>> {
    let item = dict.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("DuckDB result missing column '{key}'"))
    })?;
    let arr = item.downcast::<PyArray1<i64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("Column '{key}' is not an int64 array"))
    })?;
    Ok(arr.readonly().as_slice()?.to_vec())
}

/// Escape a string for use as a single-quoted SQL literal.
fn escape_sql_literal(s: &str) -> String {
    s.replace('\'', "''")
}
