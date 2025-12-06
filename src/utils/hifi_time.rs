//! Minimal hifitime utilities for chrono interop
//!
//! Provides only the conversions needed between chrono DateTime<Utc> and hifitime Epoch.
//! Uses hifitime's native time scale conversions for maximum accuracy.

use chrono::{DateTime, Utc};
use hifitime::{Duration, Epoch};

use crate::utils::ut1_provider;

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
    // to_jde_tt_days() returns Julian Date Ephemeris in TT scale
    // Subtract JD1 to get the MJD-like fractional part
    (JD1, epoch.to_jde_tt_days() - JD1)
}

/// Convert DateTime to MJD in UTC
#[inline]
pub fn datetime_to_mjd(dt: &DateTime<Utc>) -> f64 {
    chrono_to_epoch(dt).to_mjd_utc_days()
}

/// Convert DateTime to two-part Julian Date in UT1 time scale
#[inline]
pub fn datetime_to_jd_ut1(dt: &DateTime<Utc>) -> (f64, f64) {
    let (jd1, jd2) = datetime_to_jd_utc(dt);
    (jd1, jd2 + ut1_provider::get_ut1_utc_offset(dt) / 86400.0)
}
