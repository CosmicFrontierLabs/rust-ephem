//! Time utilities for astronomical calculations
//!
//! Provides conversions between chrono DateTime<Utc> and hifitime Epoch,
//! Julian Date calculations in various time scales, and Python datetime interop.

use chrono::{DateTime, Datelike, Timelike, Utc};
use hifitime::{Duration, Epoch};
use pyo3::prelude::*;

use crate::utils::config::{JD_EPOCH, SECONDS_PER_DAY};
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

/// Convert DateTime to two-part Julian Date in UTC for ERFA
#[inline]
pub fn datetime_to_jd_utc(dt: &DateTime<Utc>) -> (f64, f64) {
    (JD_EPOCH, chrono_to_epoch(dt).to_mjd_utc_days())
}

/// Split integer nanoseconds before converting to f64. Forming one large
/// floating-point Julian/MJD date first discards sub-microsecond precision.
fn split_julian_duration(duration: Duration) -> (f64, f64) {
    const NANOS_PER_DAY: i128 = 86_400_000_000_000;
    let nanos = duration.total_nanoseconds();
    (
        nanos.div_euclid(NANOS_PER_DAY) as f64,
        nanos.rem_euclid(NANOS_PER_DAY) as f64 / NANOS_PER_DAY as f64,
    )
}

/// Convert DateTime to two-part Julian Date in TT for ERFA precession/nutation
#[inline]
pub fn datetime_to_jd_tt(dt: &DateTime<Utc>) -> (f64, f64) {
    split_julian_duration(chrono_to_epoch(dt).to_jde_tt_duration())
}

/// Convert DateTime to two-part Julian Date in UT1 time scale
#[inline]
pub fn datetime_to_jd_ut1(dt: &DateTime<Utc>) -> (f64, f64) {
    let (jd1, jd2) = split_julian_duration(chrono_to_epoch(dt).to_jde_utc_duration());
    (
        jd1,
        jd2 + ut1_provider::get_ut1_utc_offset(dt) / SECONDS_PER_DAY,
    )
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

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn tt_and_ut1_retain_submicrosecond_offsets() {
        for stamp in [
            "1950-01-01T23:59:59.123456789Z",
            "2024-01-01T23:59:59.123456789Z",
            "2100-01-01T23:59:59.123456789Z",
        ] {
            let first = stamp.parse::<DateTime<Utc>>().unwrap();
            let second = first + chrono::Duration::nanoseconds(100);
            for convert in [datetime_to_jd_tt, datetime_to_jd_ut1] {
                let a = convert(&first);
                let b = convert(&second);
                let elapsed = ((b.0 - a.0) + (b.1 - a.1)) * SECONDS_PER_DAY;
                assert!((elapsed - 1e-7).abs() < 2e-11, "{elapsed}");
            }
        }
    }

    #[test]
    fn tt_retains_the_2016_leap_second() {
        let before = "2016-12-31T23:59:59Z".parse::<DateTime<Utc>>().unwrap();
        let after = "2017-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let a = datetime_to_jd_tt(&before);
        let b = datetime_to_jd_tt(&after);
        assert!((((b.0 - a.0) + (b.1 - a.1)) * SECONDS_PER_DAY - 2.0).abs() < 1e-10);
    }
}
