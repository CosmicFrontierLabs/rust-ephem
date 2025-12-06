//! Minimal hifitime utilities for chrono interop
//!
//! Provides only the conversions needed between chrono DateTime<Utc> and hifitime Epoch.

use chrono::{DateTime, Utc};
use hifitime::{Duration, Epoch};

use crate::utils::config::TT_TAI_SECONDS;

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

/// Get TT-UTC offset in seconds (TT-TAI + TAI-UTC = 32.184 + leap_seconds)
#[inline]
pub fn get_tt_utc_offset_seconds(dt: &DateTime<Utc>) -> f64 {
    get_tai_utc_offset(dt).map_or(69.184, |tai_utc| TT_TAI_SECONDS + tai_utc)
}

/// Convert DateTime to two-part Julian Date for ERFA (JD1=2400000.5, JD2=MJD)
#[inline]
pub fn datetime_to_jd(dt: &DateTime<Utc>) -> (f64, f64) {
    const JD1: f64 = 2400000.5;
    (JD1, chrono_to_epoch(dt).to_mjd_utc_days())
}

/// Convert DateTime to MJD
#[inline]
pub fn datetime_to_mjd(dt: &DateTime<Utc>) -> f64 {
    chrono_to_epoch(dt).to_mjd_utc_days()
}
