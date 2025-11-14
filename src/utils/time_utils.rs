use crate::utils::config::*;
use crate::utils::leap_seconds;
use crate::utils::ut1_provider;
use chrono::{DateTime, Datelike, Timelike, Utc};
use pyo3::prelude::*;

/// Convert a DateTime<Utc> to a two-part Julian Date (JD1, JD2) for ERFA functions.
/// This splits the JD into an integer part (JD1) and fractional part (JD2) for precision.
#[inline]
pub fn datetime_to_jd(dt: &DateTime<Utc>) -> (f64, f64) {
    // Use shared time constants from config

    let timestamp_secs = dt.timestamp() as f64;
    let timestamp_nanos = dt.timestamp_subsec_nanos() as f64;

    // Use fused multiply-add for better precision and performance
    let days_since_epoch =
        timestamp_secs.mul_add(SECONDS_PER_DAY_RECIP, timestamp_nanos * NANOS_TO_DAYS);

    // Split into integer and fractional parts for better precision
    // Standard practice is to use JD1 = 2400000.5 + integer days, JD2 = fractional part
    let jd2 = JD_UNIX_EPOCH - JD1 + days_since_epoch;

    (JD1, jd2)
}

/// Get TT-UTC offset in days for a given UTC time.
/// Uses leap second data if available, otherwise falls back to approximation.
#[inline]
pub fn get_tt_offset_days(dt: &DateTime<Utc>) -> f64 {
    leap_seconds::get_tt_utc_offset_days(dt)
}

/// Convert a DateTime<Utc> to a two-part Julian Date in UT1 time scale.
/// This applies the UT1-UTC offset from IERS data via hifitime.
#[inline]
pub fn datetime_to_jd_ut1(dt: &DateTime<Utc>) -> (f64, f64) {
    let (jd1, jd2) = datetime_to_jd(dt);
    let ut1_utc_days = ut1_provider::get_ut1_utc_offset_days(dt);
    (jd1, jd2 + ut1_utc_days)
}

/// Helper function to convert Python datetime to Rust DateTime<Utc>
/// Accepts both naive and timezone-aware datetimes (naive datetimes are treated as UTC)
pub fn python_datetime_to_utc(py_dt: &Bound<PyAny>) -> PyResult<DateTime<Utc>> {
    let year = py_dt.getattr("year")?.extract::<i32>()?;
    let month = py_dt.getattr("month")?.extract::<u32>()?;
    let day = py_dt.getattr("day")?.extract::<u32>()?;
    let hour = py_dt.getattr("hour")?.extract::<u32>()?;
    let minute = py_dt.getattr("minute")?.extract::<u32>()?;
    let second = py_dt.getattr("second")?.extract::<u32>()?;
    let micro = py_dt.getattr("microsecond")?.extract::<u32>()?;

    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid date"))?;
    let time = chrono::NaiveTime::from_hms_micro_opt(hour, minute, second, micro)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid time"))?;
    let naive = chrono::NaiveDateTime::new(date, time);
    Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

/// Helper function to convert Rust DateTime<Utc> to Python datetime (UTC timezone-aware)
pub fn utc_to_python_datetime(py: Python, dt: &DateTime<Utc>) -> PyResult<Py<PyAny>> {
    let datetime_mod = py.import("datetime")?;
    let timezone_class = datetime_mod.getattr("timezone")?;
    let timezone_utc = timezone_class.getattr("utc")?;

    let py_dt = datetime_mod.getattr("datetime")?.call1((
        dt.year(),
        dt.month(),
        dt.day(),
        dt.hour(),
        dt.minute(),
        dt.second(),
        dt.timestamp_subsec_micros(),
        timezone_utc,
    ))?;
    Ok(py_dt.into())
}
