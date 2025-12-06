//! Time utilities for astronomical calculations
//!
//! Provides conversions between chrono DateTime<Utc> and hifitime Epoch,
//! Julian Date calculations in various time scales, and Python datetime interop.

use chrono::{DateTime, Datelike, Timelike, Utc};
use hifitime::{Duration, Epoch};
use pyo3::prelude::*;

use crate::utils::ut1_provider;

// ============================================================================
// Chrono <-> hifitime conversions
// ============================================================================

/// Convert chrono `DateTime<Utc>` to hifitime `Epoch`
#[inline]
pub fn chrono_to_epoch(dt: &DateTime<Utc>) -> Epoch {
    let nanos = (dt.timestamp() as i128) * 1_000_000_000 + (dt.timestamp_subsec_nanos() as i128);
    Epoch::from_unix_duration(Duration::from_total_nanoseconds(nanos))
}

/// Get TAI-UTC offset in seconds (leap seconds) for a DateTime
#[inline]
pub fn get_tai_utc_offset(dt: &DateTime<Utc>) -> Option<f64> {
    chrono_to_epoch(dt).leap_seconds(true)
}

// ============================================================================
// Julian Date conversions for ERFA
// ============================================================================

/// Convert DateTime to two-part Julian Date in UTC for ERFA (JD1=2400000.5, JD2=MJD)
#[inline]
pub fn datetime_to_jd_utc(dt: &DateTime<Utc>) -> (f64, f64) {
    const JD1: f64 = 2400000.5;
    (JD1, chrono_to_epoch(dt).to_mjd_utc_days())
}

/// Convert DateTime to two-part Julian Date in TT for ERFA precession/nutation
/// Uses hifitime's native TT conversion for proper leap second handling.
#[inline]
pub fn datetime_to_jd_tt(dt: &DateTime<Utc>) -> (f64, f64) {
    const JD1: f64 = 2400000.5;
    let epoch = chrono_to_epoch(dt);
    (JD1, epoch.to_jde_tt_days() - JD1)
}

/// Convert DateTime to two-part Julian Date in UT1 time scale
#[inline]
pub fn datetime_to_jd_ut1(dt: &DateTime<Utc>) -> (f64, f64) {
    let (jd1, jd2) = datetime_to_jd_utc(dt);
    (jd1, jd2 + ut1_provider::get_ut1_utc_offset(dt) / 86400.0)
}

/// Convert DateTime to MJD in UTC
#[inline]
pub fn datetime_to_mjd(dt: &DateTime<Utc>) -> f64 {
    chrono_to_epoch(dt).to_mjd_utc_days()
}

// ============================================================================
// Python datetime interop
// ============================================================================

/// Convert Python datetime to chrono DateTime<Utc>
pub fn python_datetime_to_utc(py_dt: &Bound<PyAny>) -> PyResult<DateTime<Utc>> {
    let date = chrono::NaiveDate::from_ymd_opt(
        py_dt.getattr("year")?.extract()?,
        py_dt.getattr("month")?.extract()?,
        py_dt.getattr("day")?.extract()?,
    )
    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid date"))?;

    let time = chrono::NaiveTime::from_hms_micro_opt(
        py_dt.getattr("hour")?.extract()?,
        py_dt.getattr("minute")?.extract()?,
        py_dt.getattr("second")?.extract()?,
        py_dt.getattr("microsecond")?.extract()?,
    )
    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid time"))?;

    Ok(DateTime::from_naive_utc_and_offset(
        chrono::NaiveDateTime::new(date, time),
        Utc,
    ))
}

/// Convert chrono DateTime<Utc> to Python datetime (UTC timezone-aware)
pub fn utc_to_python_datetime(py: Python, dt: &DateTime<Utc>) -> PyResult<Py<PyAny>> {
    let datetime_mod = py.import("datetime")?;
    let tz_utc = datetime_mod.getattr("timezone")?.getattr("utc")?;
    Ok(datetime_mod
        .getattr("datetime")?
        .call1((
            dt.year(),
            dt.month(),
            dt.day(),
            dt.hour(),
            dt.minute(),
            dt.second(),
            dt.timestamp_subsec_micros(),
            tz_utc,
        ))?
        .into())
}
