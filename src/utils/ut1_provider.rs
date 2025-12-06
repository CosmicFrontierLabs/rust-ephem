/// UT1-UTC offset provider using hifitime
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use std::sync::Mutex;

use crate::utils::eop_cache::load_or_download_eop2_text;
use crate::utils::time_utils::{chrono_to_epoch, get_tai_utc_offset};

pub use hifitime::ut1::Ut1Provider;

static UT1_PROVIDER: Lazy<Mutex<Option<Ut1Provider>>> = Lazy::new(|| {
    match load_or_download_eop2_text()
        .map_err(|e| e.to_string())
        .and_then(|t| Ut1Provider::from_eop_data(t).map_err(|e| e.to_string()))
    {
        Ok(p) => {
            eprintln!("UT1 provider initialized (EOP2 short, cached)");
            Mutex::new(Some(p))
        }
        Err(e) => {
            eprintln!("Warning: UT1 provider init failed: {e}");
            Mutex::new(None)
        }
    }
});

/// Get UT1-UTC offset in seconds (UT1-UTC = TAI-UTC - TAI-UT1)
pub fn get_ut1_utc_offset(dt: &DateTime<Utc>) -> f64 {
    let guard = UT1_PROVIDER.lock().unwrap();
    guard.as_ref().map_or(0.0, |provider| {
        let epoch = chrono_to_epoch(dt);
        epoch.ut1_offset(provider).map_or(0.0, |tai_ut1| {
            get_tai_utc_offset(dt).unwrap_or(37.0) - tai_ut1.to_seconds()
        })
    })
}

/// Initialize/refresh UT1 provider
pub fn init_ut1_provider() -> bool {
    let mut guard = UT1_PROVIDER.lock().unwrap();
    match load_or_download_eop2_text()
        .map_err(|e| e.to_string())
        .and_then(|t| Ut1Provider::from_eop_data(t).map_err(|e| e.to_string()))
    {
        Ok(p) => {
            *guard = Some(p);
            true
        }
        Err(e) => {
            eprintln!("UT1 init error: {e}");
            false
        }
    }
}

/// Check if UT1 provider is available
pub fn is_ut1_available() -> bool {
    UT1_PROVIDER.lock().unwrap().is_some()
}
