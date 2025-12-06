/// Leap second management using hifitime
///
/// This module provides accurate TAI-UTC and TT-UTC offsets using hifitime's
/// built-in leap second data. This eliminates the need to manually maintain
/// leap second tables.
///
/// TT-UTC = TT-TAI + TAI-UTC = 32.184 + TAI-UTC
use chrono::{DateTime, Utc};
use hifitime::Epoch;

use crate::utils::config::TT_TAI_SECONDS;

/// Convert chrono DateTime<Utc> to hifitime Epoch
#[inline]
fn chrono_to_epoch(dt: &DateTime<Utc>) -> Epoch {
    let timestamp_secs = dt.timestamp();
    let timestamp_nanos = dt.timestamp_subsec_nanos();
    let total_nanos = (timestamp_secs as i128) * 1_000_000_000 + (timestamp_nanos as i128);
    Epoch::from_unix_duration(hifitime::Duration::from_total_nanoseconds(total_nanos))
}

/// Get TAI-UTC offset in seconds for a given UTC time
///
/// Returns None if the date is before 1960 (when UTC was defined)
pub fn get_tai_utc_offset(dt: &DateTime<Utc>) -> Option<f64> {
    let epoch = chrono_to_epoch(dt);
    // leap_seconds(true) returns IERS-only leap seconds (integer values since 1972)
    epoch.leap_seconds(true)
}

/// Get TT-UTC offset in seconds for a given UTC time
///
/// TT-UTC = TT-TAI + TAI-UTC = 32.184 + TAI-UTC
///
/// Falls back to 69.184 seconds if leap second data unavailable
pub fn get_tt_utc_offset_seconds(dt: &DateTime<Utc>) -> f64 {
    if let Some(tai_utc) = get_tai_utc_offset(dt) {
        TT_TAI_SECONDS + tai_utc
    } else {
        // Fallback to current approximation (as of 2017+)
        69.184
    }
}

/// Get TT-UTC offset in days (for ERFA functions)
pub fn get_tt_utc_offset_days(dt: &DateTime<Utc>) -> f64 {
    get_tt_utc_offset_seconds(dt) / 86400.0
}
