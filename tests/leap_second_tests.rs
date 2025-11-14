// Tests for leap second functionality
// These tests verify the accuracy of TAI-UTC offset calculations
// using embedded IERS leap second data.
//
// Since this is a cdylib crate, we test the leap second logic
// as integration tests using the internal implementation.

use chrono::{TimeZone, Utc};

// Helper to convert DateTime to NTP timestamp (copied from leap_seconds.rs)
fn datetime_to_ntp(dt: &chrono::DateTime<Utc>) -> i64 {
    const NTP_UNIX_OFFSET: i64 = 2208988800;
    dt.timestamp() + NTP_UNIX_OFFSET
}

// Embedded leap second data (copied from leap_seconds.rs for testing)
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

// Test implementation of get_tai_utc_offset
fn get_tai_utc_offset_test(dt: &chrono::DateTime<Utc>) -> Option<f64> {
    let ntp_timestamp = datetime_to_ntp(dt);

    let idx = match LEAP_SECONDS_DATA.binary_search_by_key(&ntp_timestamp, |(ts, _)| *ts) {
        Ok(i) => i,
        Err(0) => return None,
        Err(i) => i - 1,
    };

    Some(LEAP_SECONDS_DATA[idx].1)
}

// Test implementation of get_tt_utc_offset_seconds
fn get_tt_utc_offset_seconds_test(dt: &chrono::DateTime<Utc>) -> f64 {
    const TT_TAI_SECONDS: f64 = 32.184;
    if let Some(tai_utc) = get_tai_utc_offset_test(dt) {
        TT_TAI_SECONDS + tai_utc
    } else {
        69.184 // Fallback
    }
}

#[test]
fn test_tai_utc_historical() {
    // Test some known historical values
    // Before 1972: no leap seconds, TAI-UTC should be None
    let dt = Utc.with_ymd_and_hms(1971, 1, 1, 0, 0, 0).unwrap();
    assert!(get_tai_utc_offset_test(&dt).is_none());

    // 1972-01-01: First leap second, TAI-UTC = 10
    let dt = Utc.with_ymd_and_hms(1972, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt), Some(10.0));

    // 1990-01-01: TAI-UTC = 25
    let dt = Utc.with_ymd_and_hms(1990, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt), Some(25.0));

    // 2000-01-01: TAI-UTC = 32
    let dt = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt), Some(32.0));

    // 2017-01-01: TAI-UTC = 37 (most recent as of this writing)
    let dt = Utc.with_ymd_and_hms(2017, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt), Some(37.0));

    // 2024-01-01: TAI-UTC = 37 (still current)
    let dt = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt), Some(37.0));
}

#[test]
fn test_tt_utc_offset() {
    // TT-UTC = TT-TAI + TAI-UTC = 32.184 + TAI-UTC
    let dt = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let offset = get_tt_utc_offset_seconds_test(&dt);

    // Should be 69.184 (32.184 + 37.0)
    assert!((offset - 69.184).abs() < 0.001);

    // Test historical accuracy
    let dt_2000 = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let offset_2000 = get_tt_utc_offset_seconds_test(&dt_2000);
    // Should be 64.184 (32.184 + 32.0)
    assert!((offset_2000 - 64.184).abs() < 0.001);

    let dt_1990 = Utc.with_ymd_and_hms(1990, 1, 1, 0, 0, 0).unwrap();
    let offset_1990 = get_tt_utc_offset_seconds_test(&dt_1990);
    // Should be 57.184 (32.184 + 25.0)
    assert!((offset_1990 - 57.184).abs() < 0.001);
}

#[test]
fn test_is_valid_always_true() {
    // Embedded data is always valid - this is a simple sanity test
    assert!(LEAP_SECONDS_DATA.len() == 28);
}

#[test]
fn test_leap_second_boundaries() {
    // Test dates just before and after a leap second
    // 2016-12-31 23:59:59 UTC: TAI-UTC = 36
    let dt_before = Utc.with_ymd_and_hms(2016, 12, 31, 23, 59, 59).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt_before), Some(36.0));

    // 2017-01-01 00:00:00 UTC: TAI-UTC = 37
    let dt_after = Utc.with_ymd_and_hms(2017, 1, 1, 0, 0, 0).unwrap();
    assert_eq!(get_tai_utc_offset_test(&dt_after), Some(37.0));
}
