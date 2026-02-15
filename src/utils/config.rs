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
pub const AU_PER_DAY_TO_KM_PER_SEC: f64 = AU_TO_KM / SECONDS_PER_DAY;
pub const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;

/// Standard epoch for two-part Julian Date (JD = JD_EPOCH + MJD)
pub const JD_EPOCH: f64 = 2400000.5;

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

/// Default TLE epoch tolerance for Space-Track.org caching (2 days)
pub const DEFAULT_EPOCH_TOLERANCE_DAYS: f64 = 2.0;

/// Maximum number of cached Space-Track TLEs per NORAD ID
pub const SPACETRACK_CACHE_MAX_ENTRIES: usize = 1000;

/// TTL for cached TLE downloads (24 hours)
pub const TLE_CACHE_TTL: u64 = 86_400;
