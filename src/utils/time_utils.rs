//! Python datetime conversion utilities

use chrono::{DateTime, Datelike, Timelike, Utc};
use pyo3::prelude::*;

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
