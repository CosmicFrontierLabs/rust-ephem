/// UT1-UTC offset provider using hifitime
///
/// This module provides UT1-UTC offsets using the hifitime library's built-in
/// IERS data support. Hifitime can download and parse IERS EOP2 data from JPL.
use chrono::{DateTime, Utc};
use hifitime::Epoch;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// Shared EOP2 caching
use crate::utils::eop_cache::load_or_download_eop2_text;
use crate::utils::leap_seconds;

// Re-export Ut1Provider for external use if needed
pub use hifitime::ut1::Ut1Provider;

/// Cached UT1 provider from hifitime
static UT1_PROVIDER: Lazy<Mutex<Option<Ut1Provider>>> = Lazy::new(|| {
    match load_or_download_ut1() {
        Ok(provider) => {
            eprintln!("UT1 provider initialized successfully (EOP2 short, cached)");
            Mutex::new(Some(provider))
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not initialize UT1 provider: {e}. UT1-UTC will default to 0.0"
            );
            eprintln!("This is expected if you don't have internet access or JPL servers are unreachable.");
            Mutex::new(None)
        }
    }
});

/// Load Ut1Provider using shared EOP2 cache logic
fn load_or_download_ut1() -> Result<Ut1Provider, Box<dyn std::error::Error>> {
    let text = load_or_download_eop2_text()?;
    Ok(Ut1Provider::from_eop_data(text)?)
}

/// Get UT1-UTC offset in seconds for a given UTC datetime
///
/// Returns the UT1-UTC offset in seconds, or 0.0 if data is not available.
/// The offset is looked up from IERS EOP2 data.
///
/// # Note
/// The offset returned is TAI - UT1, which for our purposes we need to apply as:
/// UT1 = UTC + (TAI-UTC) - (TAI-UT1) = UTC + offset
///
/// However, hifitime's ut1_offset returns TAI-UT1 directly.
pub fn get_ut1_utc_offset(dt: &DateTime<Utc>) -> f64 {
    let provider_lock = UT1_PROVIDER.lock().unwrap();

    if let Some(provider) = provider_lock.as_ref() {
        // Convert chrono DateTime to hifitime Epoch (in TAI scale)
        let epoch = chrono_to_hifitime_epoch(dt);

        // Get TAI-UT1 offset from provider
        // We need UT1-UTC = UT1 - UTC
        // We have: TAI-UT1 from provider
        // We have: TAI-UTC = 37.0 seconds (approximately, from leap seconds)
        // Therefore: UT1-UTC = (TAI-UTC) - (TAI-UT1) = 37 - (TAI-UT1)
        //
        // Actually, hifitime stores TAI-UT1, so to get UT1 we need:
        // UT1 = TAI - (TAI-UT1)
        // UTC = TAI - (TAI-UTC)
        // UT1-UTC = [TAI - (TAI-UT1)] - [TAI - (TAI-UTC)] = (TAI-UTC) - (TAI-UT1)

        if let Some(tai_minus_ut1) = epoch.ut1_offset(provider) {
            // TAI-UTC from leap seconds (we need to get this properly)
            let tai_minus_utc = leap_seconds::get_tai_utc_offset(dt).unwrap_or(37.0);

            // UT1-UTC = (TAI-UTC) - (TAI-UT1)
            tai_minus_utc - tai_minus_ut1.to_seconds()
        } else {
            0.0
        }
    } else {
        // No provider available, return 0.0 (equivalent to assuming UT1 = UTC)
        0.0
    }
}

/// Get UT1-UTC offset in days (for ERFA functions)
#[inline]
pub fn get_ut1_utc_offset_days(dt: &DateTime<Utc>) -> f64 {
    get_ut1_utc_offset(dt) / 86400.0
}

/// Convert chrono DateTime<Utc> to hifitime Epoch
fn chrono_to_hifitime_epoch(dt: &DateTime<Utc>) -> Epoch {
    // Get Unix timestamp with nanosecond precision
    let timestamp_secs = dt.timestamp();
    let timestamp_nanos = dt.timestamp_subsec_nanos();

    // Create hifitime Epoch from Unix timestamp
    // Epoch::from_unix_seconds expects seconds since Unix epoch
    let total_nanos = (timestamp_secs as i128) * 1_000_000_000 + (timestamp_nanos as i128);

    // Use hifitime's from_unix_duration which handles nanoseconds
    Epoch::from_unix_duration(hifitime::Duration::from_total_nanoseconds(total_nanos))
}

/// Initialize or re-initialize the UT1 provider
///
/// This can be called to force a refresh of IERS data. Returns true if successful.
pub fn init_ut1_provider() -> bool {
    let mut provider_lock = UT1_PROVIDER.lock().unwrap();

    match load_or_download_ut1() {
        Ok(provider) => {
            *provider_lock = Some(provider);
            true
        }
        Err(e) => {
            eprintln!("Error initializing UT1 provider: {e}");
            false
        }
    }
}

/// Check if UT1 provider is available
pub fn is_ut1_available() -> bool {
    UT1_PROVIDER.lock().unwrap().is_some()
}
