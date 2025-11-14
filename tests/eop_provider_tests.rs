//! Integration tests for EOP provider
use chrono::{DateTime, TimeZone, Utc};

// Import from the library - this requires EopProvider to be public
use rust_ephem::eop_provider::EopProvider;

#[test]
fn test_datetime_to_mjd() {
    // 2000-01-01 12:00:00 UTC = MJD 51544.5
    let dt = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap();
    let mjd = EopProvider::datetime_to_mjd(&dt);
    assert!((mjd - 51544.5).abs() < 1e-6);
}

#[test]
fn test_parse_eop2_data() {
    let data = r#"
# Comment header line
$ Another comment style
EOP2LBL='EOP2. LAST UTPM DATUM 2025-11-11.'
EOP2UT1='UT1'
60000.0, 123.456, 234.567, 37000.0, 0.05, 0.05, 0.01, 0.0, 0.0, 0.0, 0.123, 0.234
60001.0, 125.000, 235.000, 37100.0, 0.05, 0.05, 0.01, 0.0, 0.0, 0.0, 0.125, 0.235, $ 2024-11-13
60002.0, 126.000, 236.000, 37200.0, 0.05, 0.05, 0.01, 0.0, 0.0, 0.0, 0.126, 0.236
"#
    .to_string();

    let provider = EopProvider::from_eop2_data(data).unwrap();
    assert_eq!(provider.len(), 3);

    // Test we can query polar motion for a date in range
    let test_dt = Utc.with_ymd_and_hms(2023, 10, 15, 0, 0, 0).unwrap(); // MJD ~60233
    let (xp, yp) = provider.get_polar_motion(&test_dt);
    // Should return interpolated values (not zero since we have data)
    assert!(xp != 0.0 || yp != 0.0);
}

#[test]
fn test_polar_motion_interpolation() {
    let data = r#"
60000.0, 100.000, 200.000, 37000.0
60002.0, 120.000, 220.000, 37200.0
"#
    .to_string();
    let provider = EopProvider::from_eop2_data(data).unwrap();

    // MJD 60001.0 midpoint
    let target_mjd = 60001.0;
    let seconds_from_epoch = (target_mjd - 40587.0) * 86400.0;
    let dt_test = DateTime::from_timestamp(seconds_from_epoch as i64, 0)
        .unwrap()
        .with_timezone(&Utc);

    let (xp, yp) = provider.get_polar_motion(&dt_test);
    assert!(
        (xp - 0.110000).abs() < 1e-6,
        "xp interpolation failed: {xp}"
    );
    assert!(
        (yp - 0.210000).abs() < 1e-6,
        "yp interpolation failed: {yp}"
    );
}
