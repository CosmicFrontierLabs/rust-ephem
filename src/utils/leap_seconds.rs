/// Leap second management for accurate UTC to TT conversions
///
/// This module provides accurate TAI-UTC offsets for any date using embedded
/// IERS leap second data. The TT-UTC offset is then:
/// TT-UTC = TT-TAI + TAI-UTC = 32.184 + TAI-UTC
///
/// Data source: IERS/IETF leap seconds list
/// Last update: 2017-01-01 (37 leap seconds)
/// Next scheduled check: None announced as of November 2024
use chrono::{DateTime, Utc};

use crate::utils::config::{LEAP_SECONDS_DATA, NTP_UNIX_OFFSET, TT_TAI_SECONDS};

/// Get TAI-UTC offset in seconds for a given UTC time
///
/// Returns None if the date is before the first leap second (1972-01-01)
pub fn get_tai_utc_offset(dt: &DateTime<Utc>) -> Option<f64> {
    // Convert DateTime to NTP timestamp
    // NTP epoch: 1900-01-01, Unix epoch: 1970-01-01
    // Difference: 2208988800 seconds
    let ntp_timestamp = dt.timestamp() + NTP_UNIX_OFFSET;

    // Find the appropriate entry using binary search
    let idx = match LEAP_SECONDS_DATA.binary_search_by_key(&ntp_timestamp, |(ts, _)| *ts) {
        Ok(i) => i,            // Exact match
        Err(0) => return None, // Before first leap second
        Err(i) => i - 1,       // Use previous entry
    };

    Some(LEAP_SECONDS_DATA[idx].1)
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
