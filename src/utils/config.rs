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

/// Configuration for Earth Orientation Parameters (EOP) data.
///
/// The EOP2 file is cached as `latest_eop2.short` in the directory
/// `RUST_EPHEM_EOP_CACHE` names, or in the rust_ephem cache directory, and a
/// cached copy older than `RUST_EPHEM_EOP_CACHE_TTL` seconds (default one
/// day) is downloaded again. A run that must repeat exactly pins the file by
/// pointing `RUST_EPHEM_EOP_CACHE` at a directory holding the copy it was
/// made with and setting a TTL long enough that it is never refreshed.
pub const EOP_FILE_NAME: &str = "latest_eop2.short";
pub const EOP_CACHE_ENV: &str = "RUST_EPHEM_EOP_CACHE";
pub const EOP_CACHE_TTL_ENV: &str = "RUST_EPHEM_EOP_CACHE_TTL";
pub static DEFAULT_EOP_PATH: Lazy<PathBuf> =
    Lazy::new(|| eop_path_from(env::var(EOP_CACHE_ENV).ok(), &CACHE_DIR));
pub static DEFAULT_EOP_TTL: Lazy<u64> =
    Lazy::new(|| eop_ttl_from(env::var(EOP_CACHE_TTL_ENV).ok()));
pub const EOP2_URL: &str = "https://eop2-external.jpl.nasa.gov/eop2/latest_eop2.short";

/// The EOP2 cache file: in the directory the environment names, else in `cache_dir`.
pub fn eop_path_from(cache_env: Option<String>, cache_dir: &std::path::Path) -> PathBuf {
    match cache_env {
        Some(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()).join(EOP_FILE_NAME),
        _ => cache_dir.join(EOP_FILE_NAME),
    }
}

/// The EOP2 cache TTL in seconds: the environment's value, else one day.
pub fn eop_ttl_from(ttl_env: Option<String>) -> u64 {
    ttl_env
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(86_400)
}

// Distance/time conversions
pub const AU_TO_KM: f64 = 149597870.7;
pub const SECONDS_PER_DAY: f64 = 86400.0;
pub const AU_PER_DAY_TO_KM_PER_SEC: f64 = AU_TO_KM / SECONDS_PER_DAY;
pub const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;

/// Standard epoch for two-part Julian Date (JD = JD_EPOCH + MJD)
pub const JD_EPOCH: f64 = 2400000.5;

// Earth / orbital constants
pub const GM_EARTH: f64 = 398600.4418;
pub const JD_J2000: f64 = 2451545.0;
pub const DAYS_PER_CENTURY: f64 = 36525.0;

// NAIF IDs
pub const MOON_NAIF_ID: i32 = 301;
pub const EARTH_NAIF_ID: i32 = 399;
pub const SUN_NAIF_ID: i32 = 10;

// Physical radii in kilometers
pub const SUN_RADIUS_KM: f64 = 695700.0; // Sun mean radius (IAU 2015 Resolution B 3)
pub const MOON_RADIUS_KM: f64 = 1737.4; // Moon mean radius
pub const EARTH_RADIUS_KM: f64 = 6378.137; // Earth equatorial radius (WGS84 / IERS 2010)

// Limits
pub const MAX_TIMESTAMPS: i64 = 100_000;

/// Celestrak GP TLE API endpoint
pub const CELESTRAK_API_BASE: &str = "https://celestrak.org/NORAD/elements/gp.php";

/// Space-Track.org API endpoints
pub const SPACETRACK_API_BASE: &str = "https://www.space-track.org";
pub const SPACETRACK_LOGIN_URL: &str = "https://www.space-track.org/ajaxauth/login";

/// Space-Track.org credentials loaded from the environment, default to None if not set
pub static SPACETRACK_USERNAME_ENV: Lazy<Option<String>> =
    Lazy::new(|| env::var("SPACETRACK_USERNAME").ok());
pub static SPACETRACK_PASSWORD_ENV: Lazy<Option<String>> =
    Lazy::new(|| env::var("SPACETRACK_PASSWORD").ok());

/// Default TLE epoch tolerance for Space-Track.org caching (2 days)
///
/// Override by setting SPACETRACK_EPOCH_TOLERANCE_DAYS to a float value.
pub static DEFAULT_EPOCH_TOLERANCE_DAYS: Lazy<f64> = Lazy::new(|| {
    env::var("SPACETRACK_EPOCH_TOLERANCE_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0)
});

/// Maximum number of cached TLEs per key for Celestrak and URL caches
pub const TLE_CACHE_MAX_ENTRIES: usize = 10;

/// Maximum number of cached Space-Track TLEs per NORAD ID (one per historical epoch queried)
pub const SPACETRACK_CACHE_MAX_ENTRIES: usize = 1000;

/// TTL for cached TLE downloads (24 hours)
pub const TLE_CACHE_TTL: u64 = 86_400;

/// How close to "now" a target epoch must be for the /gp (current) endpoint to be used
/// instead of /gp_history. Intentionally short so requests for recent historical epochs
/// (hours to days ago) still go to /gp_history and retrieve the TLE nearest that time.
pub const GP_NOW_THRESHOLD_SECS: f64 = 172800.0; // 2 days

#[cfg(test)]
mod tests {
    use super::{eop_path_from, eop_ttl_from, EOP_FILE_NAME};
    use std::path::Path;

    #[test]
    fn eop_path_defaults_to_the_cache_directory() {
        let cache = Path::new("/cache");
        assert_eq!(eop_path_from(None, cache), cache.join(EOP_FILE_NAME));
        assert_eq!(
            eop_path_from(Some("  ".into()), cache),
            cache.join(EOP_FILE_NAME)
        );
    }

    #[test]
    fn eop_path_follows_the_environment() {
        let cache = Path::new("/cache");
        assert_eq!(
            eop_path_from(Some("/pinned/eop".into()), cache),
            Path::new("/pinned/eop").join(EOP_FILE_NAME)
        );
    }

    #[test]
    fn eop_ttl_defaults_to_a_day_unless_the_environment_parses() {
        assert_eq!(eop_ttl_from(None), 86_400);
        assert_eq!(eop_ttl_from(Some("not a number".into())), 86_400);
        assert_eq!(eop_ttl_from(Some("3153600000".into())), 3_153_600_000);
    }
}
