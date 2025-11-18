//! CCSDS Orbit Ephemeris Message (OEM) support
//!
//! This module provides support for CCSDS Orbit Ephemeris Messages (OEM)
//! which are standard formats for exchanging spacecraft orbit data.
//!
//! ## Reference Frame Requirements
//!
//! The OEM file must specify a reference frame that is compatible with GCRS
//! (Geocentric Celestial Reference System). Compatible frames include:
//! - J2000 / EME2000 (Earth Mean Equator and Equinox of J2000.0)
//! - GCRF (Geocentric Celestial Reference Frame)
//! - ICRF (International Celestial Reference Frame)
//!
//! If the OEM file uses a different reference frame, loading will fail with
//! an error indicating the incompatible frame.

use chrono::{DateTime, TimeZone, Utc};
use ndarray::Array2;
use pyo3::{prelude::*, types::PyDateTime};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::OnceLock;

use crate::ephemeris::ephemeris_common::{
    generate_timestamps, split_pos_vel, EphemerisBase, EphemerisData,
};
use crate::ephemeris::position_velocity::PositionVelocityData;
use crate::utils::conversions;
use crate::utils::interpolation::hermite_interpolate;
use crate::utils::time_utils::python_datetime_to_utc;

/// A simple OEM state vector record
#[derive(Debug, Clone)]
struct StateVectorRecord {
    epoch: DateTime<Utc>,
    x: f64,
    y: f64,
    z: f64,
    x_dot: f64,
    y_dot: f64,
    z_dot: f64,
}

#[pyclass]
pub struct OEMEphemeris {
    #[allow(dead_code)] // Stored for debugging/inspection purposes
    oem_path: String,
    itrs: Option<Array2<f64>>,
    itrs_skycoord: OnceLock<Py<PyAny>>, // Lazy-initialized cached SkyCoord object for ITRS
    polar_motion: bool,                 // Whether to apply polar motion correction
    // Common ephemeris data
    common_data: EphemerisData,
    // Store raw OEM data for reference
    oem_times: Vec<DateTime<Utc>>,
    oem_states: Array2<f64>,
}

#[pymethods]
impl OEMEphemeris {
    #[new]
    #[pyo3(signature = (oem_path, begin, end, step_size=60, *, polar_motion=false))]
    fn new(
        _py: Python,
        oem_path: String,
        begin: &Bound<'_, PyDateTime>,
        end: &Bound<'_, PyDateTime>,
        step_size: i64,
        polar_motion: bool,
    ) -> PyResult<Self> {
        // Load and parse the OEM file
        let path = Path::new(&oem_path);
        let records = Self::parse_oem_file(path)?;

        if records.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "OEM file contains no state vectors",
            ));
        }

        // Extract times and states from OEM records
        let (oem_times, oem_states) = Self::extract_oem_data(&records)?;

        // Validate time range
        let begin_dt = python_datetime_to_utc(begin)?;
        let end_dt = python_datetime_to_utc(end)?;

        if begin_dt < oem_times[0] || end_dt > oem_times[oem_times.len() - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Requested time range [{}, {}] exceeds OEM data range [{}, {}]",
                begin_dt,
                end_dt,
                oem_times[0],
                oem_times[oem_times.len() - 1]
            )));
        }

        // Generate query timestamps
        let times = generate_timestamps(begin, end, step_size)?;

        // Create the OEMEphemeris object
        let mut ephemeris = OEMEphemeris {
            oem_path,
            itrs: None,
            itrs_skycoord: OnceLock::new(),
            polar_motion,
            common_data: {
                let mut data = EphemerisData::new();
                data.times = Some(times);
                data
            },
            oem_times,
            oem_states,
        };

        // Pre-compute all frames
        ephemeris.interpolate_to_gcrs()?;
        ephemeris.gcrs_to_itrs()?;
        ephemeris.calculate_sun_moon()?;

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

    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.get_timestamp(py)
    }

    /// Get OEM raw data position and velocity
    ///
    /// Returns the raw state vectors from the OEM file without interpolation
    #[getter]
    fn oem_pv(&self, py: Python) -> Py<PositionVelocityData> {
        Py::new(py, split_pos_vel(&self.oem_states)).unwrap()
    }

    /// Get OEM raw data timestamps
    ///
    /// Returns the raw timestamps from the OEM file as Python datetime objects
    #[getter]
    fn oem_timestamp(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        use pyo3::types::PyTzInfo;
        let utc_tz = PyTzInfo::utc(py)?;
        self.oem_times
            .iter()
            .map(|dt| {
                let pydt = PyDateTime::from_timestamp(py, dt.timestamp() as f64, Some(&utc_tz))?;
                Ok(pydt.into_any().unbind())
            })
            .collect()
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

    /// Get angular radius of the Sun with astropy units (degrees)
    #[getter]
    fn sun_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius(py)
    }

    #[getter]
    fn sun_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_deg(py)
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
    fn earth_radius(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius(py)
    }

    #[getter]
    fn earth_radius_deg(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_deg(py)
    }

    #[getter]
    fn sun_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_sun_radius_rad(py)
    }

    #[getter]
    fn moon_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_moon_radius_rad(py)
    }

    #[getter]
    fn earth_radius_rad(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.get_earth_radius_rad(py)
    }

    /// Find the index of the closest timestamp to the given datetime
    fn index(&self, time: &Bound<'_, PyDateTime>) -> PyResult<usize> {
        self.find_closest_index(time)
    }
}

impl OEMEphemeris {
    /// Parse an OEM file and extract state vector records
    ///
    /// This parser handles basic OEM format with multiple segments and validates
    /// that the reference frame is compatible with GCRS (J2000/EME2000, GCRF, or ICRF)
    fn parse_oem_file(path: &Path) -> PyResult<Vec<StateVectorRecord>> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open OEM file: {}", e))
        })?;
        let reader = BufReader::new(file);

        let mut records = Vec::new();
        let mut in_data_section = false;
        let mut past_meta = false;
        let mut in_meta_section = false;
        let mut ref_frame_validated = false;

        for line in reader.lines() {
            let line = line.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to read OEM file: {}", e))
            })?;
            let trimmed = line.trim();

            // Skip comments and empty lines
            if trimmed.starts_with("COMMENT") || trimmed.is_empty() {
                continue;
            }

            // Handle new segment - reset flags to process next segment
            if trimmed == "META_START" {
                in_data_section = false;
                past_meta = false;
                in_meta_section = true;
                continue;
            }

            // Parse metadata fields while in META section
            if in_meta_section {
                // Check and validate REF_FRAME
                if trimmed.starts_with("REF_FRAME") {
                    if let Some(frame_value) = trimmed.split('=').nth(1) {
                        let frame = frame_value.trim();
                        // Validate that the frame is compatible with GCRS
                        if !Self::is_gcrs_compatible_frame(frame) {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                format!(
                                    "Unsupported reference frame '{}'. OEM file must use a GCRS-compatible frame such as J2000, EME2000, GCRF, or ICRF.",
                                    frame
                                )
                            ));
                        }
                        ref_frame_validated = true;
                    }
                }
            }

            // Track when we pass the metadata section
            if trimmed == "META_STOP" {
                in_meta_section = false;
                past_meta = true;
                in_data_section = true; // Start reading data after META_STOP
                continue;
            }

            // Check for explicit data section markers (if present)
            if trimmed == "DATA_START" {
                in_data_section = true;
                continue;
            }
            if trimmed == "DATA_STOP" {
                // Don't break - there might be more segments
                in_data_section = false;
                past_meta = false;
                continue;
            }

            // Parse state vector records in the data section
            // This includes both explicit DATA_START/STOP sections and
            // data that comes directly after META_STOP
            if in_data_section || past_meta {
                if let Some(record) = Self::parse_state_vector_line(trimmed)? {
                    records.push(record);
                }
            }
        }

        if records.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No state vectors found in OEM file",
            ));
        }

        if !ref_frame_validated {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "OEM file does not specify a REF_FRAME. A GCRS-compatible frame (J2000, EME2000, GCRF, or ICRF) is required.",
            ));
        }

        Ok(records)
    }

    /// Check if a reference frame string is compatible with GCRS
    ///
    /// GCRS-compatible frames include J2000, EME2000, GCRF, and ICRF variants
    fn is_gcrs_compatible_frame(frame: &str) -> bool {
        let frame_upper = frame.to_uppercase();
        matches!(
            frame_upper.as_str(),
            "J2000" | "EME2000" | "GCRF" | "ICRF" | "ICRF2" | "ICRF3"
        )
    }

    /// Parse a single state vector line
    ///
    /// Expected format: YYYY-MM-DDTHH:MM:SS.ffffff X Y Z VX VY VZ
    fn parse_state_vector_line(line: &str) -> PyResult<Option<StateVectorRecord>> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 7 {
            // Not a state vector line
            return Ok(None);
        }

        let epoch = Self::parse_ccsds_epoch(parts[0])?;
        let x = parts[1]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid X coordinate"))?;
        let y = parts[2]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid Y coordinate"))?;
        let z = parts[3]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid Z coordinate"))?;
        let x_dot = parts[4]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid VX velocity"))?;
        let y_dot = parts[5]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid VY velocity"))?;
        let z_dot = parts[6]
            .parse::<f64>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid VZ velocity"))?;

        Ok(Some(StateVectorRecord {
            epoch,
            x,
            y,
            z,
            x_dot,
            y_dot,
            z_dot,
        }))
    }

    /// Extract times and state vectors from OEM records
    ///
    /// Converts OEM state vector records into chrono DateTime and ndarray format
    fn extract_oem_data(
        records: &[StateVectorRecord],
    ) -> PyResult<(Vec<DateTime<Utc>>, Array2<f64>)> {
        let n = records.len();
        let mut times = Vec::with_capacity(n);
        let mut states = Array2::<f64>::zeros((n, 6));

        for (i, record) in records.iter().enumerate() {
            times.push(record.epoch);

            // Extract position and velocity
            // CCSDS OEM uses km for position and km/s for velocity
            states[[i, 0]] = record.x;
            states[[i, 1]] = record.y;
            states[[i, 2]] = record.z;
            states[[i, 3]] = record.x_dot;
            states[[i, 4]] = record.y_dot;
            states[[i, 5]] = record.z_dot;
        }

        Ok((times, states))
    }

    /// Parse CCSDS epoch string to DateTime<Utc>
    ///
    /// CCSDS uses ISO 8601 format: YYYY-MM-DDTHH:MM:SS.ffffff
    fn parse_ccsds_epoch(epoch_str: &str) -> PyResult<DateTime<Utc>> {
        // Remove 'Z' suffix if present
        let clean_str = epoch_str.trim_end_matches('Z');

        // Split into date and time parts
        let parts: Vec<&str> = clean_str.split('T').collect();
        if parts.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid CCSDS epoch format: {}",
                epoch_str
            )));
        }

        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_parts: Vec<&str> = parts[1].split(':').collect();

        if date_parts.len() != 3 || time_parts.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid CCSDS epoch format: {}",
                epoch_str
            )));
        }

        let year = date_parts[0]
            .parse::<i32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid year in epoch"))?;
        let month = date_parts[1]
            .parse::<u32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid month in epoch"))?;
        let day = date_parts[2]
            .parse::<u32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid day in epoch"))?;
        let hour = time_parts[0]
            .parse::<u32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid hour in epoch"))?;
        let minute = time_parts[1]
            .parse::<u32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid minute in epoch"))?;

        // Handle seconds with fractional part
        let sec_parts: Vec<&str> = time_parts[2].split('.').collect();
        let second = sec_parts[0]
            .parse::<u32>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid second in epoch"))?;

        let nanosecond = if sec_parts.len() > 1 {
            // Parse fractional seconds
            let frac_str = sec_parts[1];
            // Pad or truncate to 9 digits (nanoseconds)
            let padded = format!("{:0<9}", frac_str);
            let truncated = &padded[..9];
            truncated.parse::<u32>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("Invalid fractional seconds in epoch")
            })?
        } else {
            0
        };

        Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
            .earliest()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid datetime components: {}-{}-{} {}:{}:{}",
                    year, month, day, hour, minute, second
                ))
            })
            .map(|dt| dt + chrono::Duration::nanoseconds(nanosecond as i64))
    }

    /// Interpolate OEM data to requested timestamps in GCRS frame
    ///
    /// Uses Hermite interpolation for smooth position and velocity
    fn interpolate_to_gcrs(&mut self) -> PyResult<()> {
        let times = self.common_data.times.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No times available for interpolation")
        })?;

        // Interpolate using Hermite method
        let interpolated = hermite_interpolate(times, &self.oem_times, &self.oem_states);

        // Store in GCRS (assuming OEM data is in an inertial frame compatible with GCRS)
        // Note: CCSDS OEM typically uses J2000/GCRF which is essentially GCRS
        self.common_data.gcrs = Some(interpolated);

        Ok(())
    }

    /// Transform GCRS to ITRS coordinates
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

// Implement the EphemerisBase trait for OEMEphemeris
impl EphemerisBase for OEMEphemeris {
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
