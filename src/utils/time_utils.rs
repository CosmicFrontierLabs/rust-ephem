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
/// scalars) is accepted as well and read as UTC, and so is an astropy `Time`
/// (scalar or array, in any scale).
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

/// The UTC instants of an astropy `Time`, as `datetime64[us]` (a scalar or an
/// array, as the `Time` is). `None` for anything else.
///
/// The object is recognised by its attributes, so astropy is not imported. A
/// `Time` in any scale is converted to UTC. Its instants are rounded to the
/// microsecond: astropy holds a time as two doubles, and the nanoseconds it
/// reports carry rounding of that size.
///
/// The instants are computed from the two doubles of the UTC time (`jd1` and
/// `jd2`), not read from `Time.datetime64`. astropy builds that array from one
/// ISO string per element, about 1.7 us each, and keeps it on the object, so
/// only the first use of a `Time` pays for it. The result here is the one that
/// path gives, to the microsecond.
///
/// # Errors
/// `ValueError` for an instant that is not finite or lies within a leap
/// second, which a UTC `datetime64` cannot hold.
fn astropy_time_to_datetime64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !(obj.hasattr("jd1")? && obj.hasattr("scale")? && obj.hasattr("utc")?) {
        return Ok(None);
    }
    let py = obj.py();
    let np = py.import("numpy")?;
    let utc = obj.getattr("utc")?;
    let jd1 = utc.getattr("jd1")?;
    let shape = np.call_method1("shape", (&jd1,))?;
    let part = |value: Bound<'py, PyAny>| -> PyResult<Vec<f64>> {
        let flat = np
            .call_method1("asarray", (value, "float64"))?
            .call_method1("ravel", ())?;
        let flat = flat
            .cast::<numpy::PyArray1<f64>>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Could not read the Time's Julian dates")
            })?
            .readonly();
        Ok(flat.as_array().to_vec())
    };
    let jd1 = part(jd1)?;
    let jd2 = part(utc.getattr("jd2")?)?;
    if jd1
        .iter()
        .zip(&jd2)
        .any(|(a, b)| a + b < LEAP_SECOND_ERA_JD)
    {
        // Before 1972 UTC ran at a varying rate against atomic time and its
        // days had lengths of their own, which astropy takes from ERFA. Such
        // times are read from the strings, as all times once were.
        return astropy_time_to_datetime64_from_strings(&utc).map(Some);
    }
    let mut lengths = DayLengths::default();
    let micros = jd1
        .iter()
        .zip(&jd2)
        .enumerate()
        .map(|(i, (&a, &b))| {
            utc_jd_to_unix_micros(a, b, &mut lengths).map_err(|reason| {
                pyo3::exceptions::PyValueError::new_err(format!("times[{i}] {reason}"))
            })
        })
        .collect::<PyResult<Vec<i64>>>()?;
    numpy::PyArray1::from_vec(py, micros)
        .call_method1("reshape", (shape,))?
        .call_method1("astype", ("datetime64[us]",))
        .map(Some)
}

/// The UTC instants of an astropy `Time` already in UTC, read from its
/// `datetime64`, which astropy builds from one ISO string per element and
/// reports to the nanosecond; rounded to the microsecond.
fn astropy_time_to_datetime64_from_strings<'py>(
    utc: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = utc.py().import("numpy")?;
    let nanos = np
        .call_method1("asarray", (utc.getattr("datetime64")?,))?
        .call_method1("astype", ("datetime64[ns]",))?
        .call_method1("astype", ("int64",))?;
    let micros = np.call_method1(
        "floor_divide",
        (np.call_method1("add", (nanos, 500))?, 1000),
    )?;
    micros.call_method1("astype", ("datetime64[us]",))
}

/// Julian date of 1970-01-01T00:00:00 UTC.
const UNIX_EPOCH_JD: f64 = 2_440_587.5;
/// Julian date of 1972-01-01T00:00:00 UTC, from which UTC keeps the SI second
/// and differs from atomic time by whole leap seconds.
const LEAP_SECOND_ERA_JD: f64 = 2_441_317.5;
/// Days from 0001-01-01 (day 1 of the common era) to 1970-01-01.
const UNIX_EPOCH_DAYS_FROM_CE: i64 = 719_163;
const MICROS_PER_DAY: i64 = 86_400_000_000;
const NANOS_PER_SECOND: f64 = 1e9;

/// The length in SI seconds of UTC days, by days since 1970-01-01: 86,400,
/// or 86,401 for a day that ends with a leap second. Each day asked about is
/// kept, since the times of one array share a few days.
#[derive(Default)]
struct DayLengths {
    known: std::collections::HashMap<i64, f64>,
}

impl DayLengths {
    fn seconds(&mut self, day: i64) -> Option<f64> {
        if let Some(seconds) = self.known.get(&day) {
            return Some(*seconds);
        }
        let midnight = |day: i64| -> Option<Epoch> {
            let date = chrono::NaiveDate::from_num_days_from_ce_opt(
                i32::try_from(day.checked_add(UNIX_EPOCH_DAYS_FROM_CE)?).ok()?,
            )?;
            Some(Epoch::from_gregorian_utc_at_midnight(
                date.year(),
                date.month() as u8,
                date.day() as u8,
            ))
        };
        // TAI minus UTC at the midnight that ends the day, less that at the
        // midnight that begins it, is the leap second the day ends with.
        let leap = midnight(day.checked_add(1)?)?.leap_seconds(true)?
            - midnight(day)?.leap_seconds(true)?;
        let seconds = SECONDS_PER_DAY + leap;
        self.known.insert(day, seconds);
        Some(seconds)
    }
}

/// Microseconds from 1970-01-01T00:00:00 UTC, as `datetime64` counts them
/// (86,400 s to every day), of a UTC instant given as a two-part Julian date
/// in astropy's convention: on a day that ends with a leap second the day
/// fraction covers 86,401 s.
///
/// The arithmetic is that of ERFA's `jd2cal` and `d2dtf`, which astropy uses
/// to write the ISO string behind `Time.datetime64`: the day fraction is
/// summed from the two parts with a compensated sum, scaled to the day's
/// length, rounded to the nanosecond, and then to the microsecond. Each step
/// rounds as that path does, so the result is the same microsecond.
///
/// Returns the reason, for an error message, where the instant is not finite,
/// out of range, or within a leap second.
fn utc_jd_to_unix_micros(
    jd1: f64,
    jd2: f64,
    lengths: &mut DayLengths,
) -> Result<i64, &'static str> {
    if !(jd1.is_finite() && jd2.is_finite()) {
        return Err("is not a finite time");
    }
    // Whole days and the fraction of each part, in [-0.5, 0.5].
    let (whole1, whole2) = (jd1.round(), jd2.round());
    let mut julian_day = whole1 + whole2;
    // The fraction from midnight: the two fractions and the half day from
    // noon, where the Julian day begins, in a compensated sum.
    let mut sum = 0.5;
    let mut lost = 0.0;
    for part in [jd1 - whole1, jd2 - whole2] {
        let next = sum + part;
        lost += if sum.abs() >= part.abs() {
            (sum - next) + part
        } else {
            (part - next) + sum
        };
        sum = next;
        if sum >= 1.0 {
            julian_day += 1.0;
            sum -= 1.0;
        }
    }
    let mut fraction = sum + lost;
    if fraction < 0.0 {
        julian_day -= 1.0;
        fraction += 1.0;
    }
    if fraction >= 1.0 {
        julian_day += 1.0;
        fraction -= 1.0;
    }
    // The civil day begins half a day before the Julian day number.
    let days = julian_day - 0.5 - UNIX_EPOCH_JD;
    if days.abs() > f64::from(i32::MAX) {
        return Err("is out of range");
    }
    let day = days as i64;
    let seconds_in_day = lengths.seconds(day).ok_or("is out of range")?;
    if seconds_in_day != SECONDS_PER_DAY {
        fraction += fraction * (seconds_in_day - SECONDS_PER_DAY) / SECONDS_PER_DAY;
    }
    let nanos = (NANOS_PER_SECOND * (SECONDS_PER_DAY * fraction)).round() as i64;
    let micros = (nanos + 500).div_euclid(1000);
    let day_micros = (seconds_in_day * 1e6).round() as i64;
    let micros = if micros >= day_micros {
        // Rounded up to the end of the day: the first instant of the next.
        MICROS_PER_DAY
    } else if micros >= MICROS_PER_DAY {
        return Err("is within a leap second, which a UTC datetime cannot hold");
    } else {
        micros
    };
    day.checked_mul(MICROS_PER_DAY)
        .and_then(|start| start.checked_add(micros))
        .ok_or("is out of range")
}

/// `datetime64` entries above which the conversion is shared across the cores.
const DATETIME64_PARALLEL_MIN_LEN: usize = 16_384;
/// Fewest entries converted by one core: a conversion costs a few
/// nanoseconds, so a smaller share costs more to hand out than to convert.
const DATETIME64_PARALLEL_CHUNK: usize = 4096;

/// Convert NumPy `datetime64` data (array or scalar, any unit) to UTC instants
/// in flattened (C) order, and likewise an astropy `Time`. Returns `Ok(None)`
/// when `obj` is neither.
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
        return match astropy_time_to_datetime64(obj)? {
            Some(converted) => datetime64_to_utc(&converted),
            None => Ok(None),
        };
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
            Some(values) => values
                .par_iter()
                .with_min_len(DATETIME64_PARALLEL_CHUNK)
                .map(valid)
                .collect(),
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

#[cfg(test)]
mod astropy_time_tests {
    use super::*;

    /// Microseconds from the Unix epoch of a UTC date and time of day.
    fn micros(stamp: &str) -> i64 {
        stamp.parse::<DateTime<Utc>>().unwrap().timestamp_micros()
    }

    fn convert(jd1: f64, jd2: f64) -> Result<i64, &'static str> {
        utc_jd_to_unix_micros(jd1, jd2, &mut DayLengths::default())
    }

    #[test]
    fn a_day_with_a_leap_second_is_longer() {
        let mut lengths = DayLengths::default();
        let day = |stamp: &str| micros(stamp) / MICROS_PER_DAY;
        assert_eq!(lengths.seconds(day("2016-12-30T00:00:00Z")), Some(86_400.0));
        assert_eq!(lengths.seconds(day("2016-12-31T00:00:00Z")), Some(86_401.0));
        assert_eq!(lengths.seconds(day("2017-01-01T00:00:00Z")), Some(86_400.0));
    }

    #[test]
    fn the_two_parts_may_be_split_anywhere() {
        // 2029-01-01T06:00:00, as a Julian date and as an MJD after its epoch.
        let expected = micros("2029-01-01T06:00:00Z");
        assert_eq!(convert(2_462_137.75, 0.0), Ok(expected));
        assert_eq!(convert(2_462_138.0, -0.25), Ok(expected));
        assert_eq!(convert(2_400_000.5, 62_137.25), Ok(expected));
        // Before 1972 a day's length is not known here; the caller reads
        // such times another way.
        assert!(convert(UNIX_EPOCH_JD, 0.0).is_err());
        assert_eq!(
            convert(LEAP_SECOND_ERA_JD, 0.0),
            Ok(micros("1972-01-01T00:00:00Z"))
        );
    }

    #[test]
    fn a_time_is_rounded_to_the_microsecond() {
        let noon = micros("2029-01-01T12:00:00Z");
        let microsecond = 1e-6 / SECONDS_PER_DAY;
        assert_eq!(convert(2_462_138.0, 0.4 * microsecond), Ok(noon));
        assert_eq!(convert(2_462_138.0, 0.6 * microsecond), Ok(noon + 1));
        assert_eq!(convert(2_462_138.0, -0.6 * microsecond), Ok(noon - 1));
    }

    #[test]
    fn the_fraction_of_a_leap_second_day_spans_its_length() {
        // On 2016-12-31 a day fraction covers 86,401 s.
        let midnight = 2_457_753.5;
        let fraction = |seconds: f64| seconds / 86_401.0;
        assert_eq!(
            convert(midnight, fraction(43_200.0)),
            Ok(micros("2016-12-31T12:00:00Z"))
        );
        assert_eq!(
            convert(midnight, fraction(86_399.5)),
            Ok(micros("2016-12-31T23:59:59.5Z"))
        );
        // Within the leap second there is no UTC datetime.
        assert!(convert(midnight, fraction(86_400.5)).is_err());
        // The day after begins where the long day ends.
        assert_eq!(
            convert(midnight + 1.0, 0.0),
            Ok(micros("2017-01-01T00:00:00Z"))
        );
        assert_eq!(
            convert(midnight, 1.0 - 1e-13),
            Ok(micros("2017-01-01T00:00:00Z"))
        );
    }

    #[test]
    fn a_time_that_is_not_finite_is_refused() {
        assert!(convert(f64::NAN, 0.0).is_err());
        assert!(convert(2_462_138.0, f64::INFINITY).is_err());
        assert!(convert(1e300, 0.0).is_err());
    }
}
