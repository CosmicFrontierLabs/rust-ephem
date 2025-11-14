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

use crate::utils::config::TT_TAI_SECONDS;

/// Embedded leap second data: (NTP timestamp, TAI-UTC offset in seconds)
/// NTP timestamps are seconds since 1900-01-01 00:00:00
/// Data from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
const LEAP_SECONDS_DATA: &[(i64, f64)] = &[
    (2272060800, 10.0), // 1 Jan 1972
    (2287785600, 11.0), // 1 Jul 1972
    (2303683200, 12.0), // 1 Jan 1973
    (2335219200, 13.0), // 1 Jan 1974
    (2366755200, 14.0), // 1 Jan 1975
    (2398291200, 15.0), // 1 Jan 1976
    (2429913600, 16.0), // 1 Jan 1977
    (2461449600, 17.0), // 1 Jan 1978
    (2492985600, 18.0), // 1 Jan 1979
    (2524521600, 19.0), // 1 Jan 1980
    (2571782400, 20.0), // 1 Jul 1981
    (2603318400, 21.0), // 1 Jul 1982
    (2634854400, 22.0), // 1 Jul 1983
    (2698012800, 23.0), // 1 Jul 1985
    (2776982400, 24.0), // 1 Jan 1988
    (2840140800, 25.0), // 1 Jan 1990
    (2871676800, 26.0), // 1 Jan 1991
    (2918937600, 27.0), // 1 Jul 1992
    (2950473600, 28.0), // 1 Jul 1993
    (2982009600, 29.0), // 1 Jul 1994
    (3029443200, 30.0), // 1 Jan 1996
    (3076704000, 31.0), // 1 Jul 1997
    (3124137600, 32.0), // 1 Jan 1999
    (3345062400, 33.0), // 1 Jan 2006
    (3439756800, 34.0), // 1 Jan 2009
    (3550089600, 35.0), // 1 Jul 2012
    (3644697600, 36.0), // 1 Jul 2015
    (3692217600, 37.0), // 1 Jan 2017
];

/// Get TAI-UTC offset in seconds for a given UTC time
///
/// Returns None if the date is before the first leap second (1972-01-01)
pub fn get_tai_utc_offset(dt: &DateTime<Utc>) -> Option<f64> {
    // Convert DateTime to NTP timestamp
    // NTP epoch: 1900-01-01, Unix epoch: 1970-01-01
    // Difference: 2208988800 seconds
    const NTP_UNIX_OFFSET: i64 = 2208988800;
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
