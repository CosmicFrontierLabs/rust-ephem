// Centralized configuration and astronomical constants
// Put shared constants here so they're defined in one place.

use once_cell::sync::Lazy;
use std::env;
use std::path::PathBuf;

/// Cache directory for rust_ephem data files
pub static CACHE_DIR: Lazy<PathBuf> = Lazy::new(|| {
    if let Ok(home) = env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".cache");
        p.push("rust_ephem");
        if !p.exists() {
            std::fs::create_dir_all(&p).expect("Failed to create cache directory");
        }
        p
    } else {
        // Fallback to current directory if HOME not available
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
