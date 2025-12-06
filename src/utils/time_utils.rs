//! Time utilities for astronomical calculations

use crate::utils::hifi_time;
use crate::utils::ut1_provider;
use chrono::{DateTime, Datelike, Timelike, Utc};
use pyo3::prelude::*;

/// Convert DateTime to two-part Julian Date for ERFA
#[inline]
pub fn datetime_to_jd(dt: &DateTime<Utc>) -> (f64, f64) {
    hifi_time::datetime_to_jd(dt)
}

/// Get TT-UTC offset in days
#[inline]
pub fn get_tt_offset_days(dt: &DateTime<Utc>) -> f64 {
    hifi_time::get_tt_utc_offset_seconds(dt) / 86400.0
}

/// Convert DateTime to two-part Julian Date in UT1 time scale
#[inline]
pub fn datetime_to_jd_ut1(dt: &DateTime<Utc>) -> (f64, f64) {
    let (jd1, jd2) = hifi_time::datetime_to_jd(dt);
    (jd1, jd2 + ut1_provider::get_ut1_utc_offset(dt) / 86400.0)
}

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
