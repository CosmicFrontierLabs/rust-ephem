// Centralized configuration and astronomical constants
// Put shared constants here so they're defined in one place.

use once_cell::sync::Lazy;
use std::env;
use std::path::PathBuf;

/// Cache directory for rust_ephem data files
pub static CACHE_DIR: Lazy<PathBuf> = Lazy::new(|| {
    if let Some(home) = dirs::home_dir() {
        let mut p = home;
        p.push(".cache");
        p.push("rust_ephem");
        if !p.exists() {
            std::fs::create_dir_all(&p).expect("Failed to create cache directory");
        }
        p
    } else {
        // Fallback to current directory if home dir not available
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
});

/// Configuration for planetary ephemeris paths
pub static DEFAULT_DE440S_PATH: Lazy<PathBuf> = Lazy::new(|| CACHE_DIR.join("de440s.bsp"));
pub static DEFAULT_DE440_PATH: Lazy<PathBuf> = Lazy::new(|| CACHE_DIR.join("de440.bsp"));
pub const DE440S_URL: &str =
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp";
pub const DE440_URL: &str =
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp";

/// Configuration for Earth Orientation Parameters (EOP) data
pub static DEFAULT_EOP_PATH: Lazy<PathBuf> = Lazy::new(|| CACHE_DIR.join("latest_eop2.short"));
pub static DEFAULT_EOP_TTL: u64 = 86_400; // default 1 day in seconds
pub const EOP2_URL: &str = "https://eop2-external.jpl.nasa.gov/eop2/latest_eop2.short";

// Distance/time conversions
pub const AU_TO_KM: f64 = 149597870.7;
pub const SECONDS_PER_DAY: f64 = 86400.0;
pub const SECS_PER_DAY: f64 = SECONDS_PER_DAY;
pub const SECONDS_PER_DAY_RECIP: f64 = 1.0 / SECONDS_PER_DAY;
pub const NANOS_TO_DAYS: f64 = 1.0 / (1_000_000_000.0 * SECONDS_PER_DAY);
pub const AU_PER_DAY_TO_KM_PER_SEC: f64 = AU_TO_KM / SECONDS_PER_DAY;
pub const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;
pub const MJD_UNIX_EPOCH: f64 = 40587.0;
pub const NTP_UNIX_OFFSET: i64 = 2208988800;

// Time offsets
// TT-TAI is exactly 32.184 seconds by definition
pub const TT_TAI_SECONDS: f64 = 32.184;
// TT_OFFSET_DAYS is now deprecated, use leap_seconds module for accurate conversions
// This remains as a fallback approximation (assumes TAI-UTC = 37, valid since 2017)
#[allow(dead_code)]
pub const TT_OFFSET_DAYS: f64 = 69.184 / SECONDS_PER_DAY;
pub const JD_UNIX_EPOCH: f64 = 2440587.5;
pub const JD1: f64 = 2400000.5;

// Earth / orbital constants
pub const GM_EARTH: f64 = 398600.4418;
pub const JD_J2000: f64 = 2451545.0;
pub const DAYS_PER_CENTURY: f64 = 36525.0;
pub const OMEGA_EARTH: f64 = 7.292115e-5; // rad/s

// NAIF IDs
pub const MOON_NAIF_ID: i32 = 301;
pub const EARTH_NAIF_ID: i32 = 399;
pub const SUN_NAIF_ID: i32 = 10;

// Physical radii in kilometers
pub const SUN_RADIUS_KM: f64 = 696000.0; // Sun mean radius
pub const MOON_RADIUS_KM: f64 = 1737.4; // Moon mean radius
pub const EARTH_RADIUS_KM: f64 = 6378.137; // Earth equatorial radius (WGS84)

// Limits
pub const MAX_TIMESTAMPS: i64 = 100_000;

// GMST helper constants used in TLE calculations
pub const PI_OVER_43200: f64 = std::f64::consts::PI / 43200.0;
pub const GMST_COEFF_0: f64 = 67310.54841;
pub const GMST_COEFF_1: f64 = 876600.0 * 3600.0 + 8640184.812866;
pub const GMST_COEFF_2: f64 = 0.093104;
pub const GMST_COEFF_3: f64 = -6.2e-6;

/// Celestrak GP TLE API endpoint
pub const CELESTRAK_API_BASE: &str = "https://celestrak.org/NORAD/elements/gp.php";

/// Space-Track.org API endpoints
pub const SPACETRACK_API_BASE: &str = "https://www.space-track.org";
pub const SPACETRACK_LOGIN_URL: &str = "https://www.space-track.org/ajaxauth/login";

/// Environment variable names for Space-Track.org credentials
pub const SPACETRACK_USERNAME_ENV: &str = "SPACETRACK_USERNAME";
pub const SPACETRACK_PASSWORD_ENV: &str = "SPACETRACK_PASSWORD";

/// Default TLE epoch tolerance for Space-Track.org caching (4 days)
pub const DEFAULT_EPOCH_TOLERANCE_DAYS: f64 = 4.0;

/// TTL for cached TLE downloads (24 hours)
pub const TLE_CACHE_TTL: u64 = 86_400;

/// Embedded leap second data: (NTP timestamp, TAI-UTC offset in seconds)
/// NTP timestamps are seconds since 1900-01-01 00:00:00
/// Data from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
pub const LEAP_SECONDS_DATA: &[(i64, f64)] = &[
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
