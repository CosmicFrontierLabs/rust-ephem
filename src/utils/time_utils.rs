//! Time utilities for astronomical calculations
//!
//! Provides conversions between chrono DateTime<Utc> and hifitime Epoch,
//! Julian Date calculations in various time scales, and Python datetime interop.

use chrono::{DateTime, Datelike, Timelike, Utc};
use hifitime::{Duration, Epoch};
use numpy::PyArrayMethods;
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

/// Convert a single Python datetime, or any iterable of them, to UTC instants,
/// preserving input order. Shared by constraint `times=` arguments and the
/// ephemeris `*_pv_at` methods.
///
/// NumPy `datetime64` input (an array of any shape, a scalar, or a sequence of
/// scalars) is accepted as well and read as UTC.
pub fn python_datetimes_to_utc(times: &Bound<PyAny>) -> PyResult<Vec<DateTime<Utc>>> {
    if let Some(converted) = datetime64_to_utc(times)? {
        return Ok(converted);
    }
    let Ok(iter) = pyo3::types::PyIterator::from_object(times) else {
        return Ok(vec![python_datetime_to_utc(times)?]);
    };
    iter.enumerate()
        .map(|(index, item)| python_datetime_to_utc_at(&item?, index))
        .collect()
}

/// `datetime64` entries above which the conversion is shared across the cores.
const DATETIME64_PARALLEL_MIN_LEN: usize = 8192;

/// Convert NumPy `datetime64` data (array or scalar, any unit) to UTC instants
/// in flattened (C) order. Returns `Ok(None)` when `obj` is not `datetime64`.
///
/// # Errors
/// `ValueError` for `NaT` entries and for instants outside the supported range.
pub fn datetime64_to_utc(obj: &Bound<PyAny>) -> PyResult<Option<Vec<DateTime<Utc>>>> {
    datetime64_to_utc_from(obj, 0)
}

/// [`datetime64_to_utc`] for data that starts at `first_index` of the caller's
/// input, so that an error names the entry the caller passed.
fn datetime64_to_utc_from(
    obj: &Bound<PyAny>,
    first_index: usize,
) -> PyResult<Option<Vec<DateTime<Utc>>>> {
    let is_datetime64 = obj
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract::<String>())
        .is_ok_and(|kind| kind == "M");
    if !is_datetime64 {
        return Ok(None);
    }
    let np = obj.py().import("numpy")?;
    let micros = np
        .call_method1("asarray", (obj,))?
        .call_method1("astype", ("datetime64[us]",))?
        .call_method1("astype", ("int64",))?
        .call_method1("ravel", ())?;
    let micros = micros
        .cast::<numpy::PyArray1<i64>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Could not convert datetime64 times to int64")
        })?
        .readonly();
    let micros = micros.as_array();
    // A long array is converted across the cores. If any entry is invalid,
    // the serial pass below reports the first one.
    if micros.len() >= DATETIME64_PARALLEL_MIN_LEN {
        use rayon::prelude::*;
        let valid = |us: &i64| {
            (*us != i64::MIN)
                .then(|| DateTime::<Utc>::from_timestamp_micros(*us))
                .flatten()
        };
        let converted: Option<Vec<DateTime<Utc>>> = match micros.as_slice() {
            Some(values) => values.par_iter().map(valid).collect(),
            None => None,
        };
        if converted.is_some() {
            return Ok(converted);
        }
    }
    micros
        .iter()
        .enumerate()
        .map(|(i, &us)| {
            let i = first_index + i;
            if us == i64::MIN {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "times[{i}] is NaT"
                )));
            }
            DateTime::<Utc>::from_timestamp_micros(us).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("times[{i}] is out of range"))
            })
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// Convert Python datetime to UTC, respecting its offset. Naive values mean UTC.
/// A `numpy.datetime64` scalar is accepted too and read as UTC.
pub fn python_datetime_to_utc(py_dt: &Bound<PyAny>) -> PyResult<DateTime<Utc>> {
    python_datetime_to_utc_at(py_dt, 0)
}

/// [`python_datetime_to_utc`] for entry `index` of a sequence of times: a
/// `datetime64` entry that cannot be read is reported at that index.
fn python_datetime_to_utc_at(py_dt: &Bound<PyAny>, index: usize) -> PyResult<DateTime<Utc>> {
    let year = match py_dt.getattr("year") {
        Ok(year) => year,
        Err(err) => {
            return match datetime64_to_utc_from(py_dt, index)?.as_deref() {
                Some([dt]) => Ok(*dt),
                _ => Err(err),
            }
        }
    };
    let date = chrono::NaiveDate::from_ymd_opt(
        year.extract()?,
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

    let wall_time =
        DateTime::from_naive_utc_and_offset(chrono::NaiveDateTime::new(date, time), Utc);
    let offset = py_dt.call_method0("utcoffset")?;
    if offset.is_none() {
        return Ok(wall_time);
    }
    let days: i64 = offset.getattr("days")?.extract()?;
    let seconds: i64 = offset.getattr("seconds")?.extract()?;
    let micros: i64 = offset.getattr("microseconds")?.extract()?;
    let delta = chrono::Duration::days(days)
        + chrono::Duration::seconds(seconds)
        + chrono::Duration::microseconds(micros);
    wall_time
        .checked_sub_signed(delta)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("UTC datetime is out of range"))
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
