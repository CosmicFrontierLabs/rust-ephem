use std::path::Path;
use std::sync::{Arc, Mutex};

use anise::prelude::*;
use once_cell::sync::OnceCell;

/// Global almanac for planetary ephemeris (Moon, Sun, planets) loaded from DE440/DE440s SPK files.
/// This is NOT used for spacecraft-specific ephemeris.
static PLANETARY_EPHEMERIS: OnceCell<Mutex<Option<Arc<Almanac>>>> = OnceCell::new();

/// Initialize the planetary almanac with the SPK at `path`. If already initialized, replaces it.
/// This should be used for planetary ephemeris files like de440s.bsp, NOT spacecraft kernels.
pub fn init_planetary_ephemeris<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
    // Load SPK from provided path (as &str) and build the planetary almanac
    let spk_path = path.as_ref().to_str().ok_or_else(|| {
        format!(
            "Path {:?} contains invalid UTF-8 characters; cannot load SPK file",
            path.as_ref()
        )
    })?;
    let spk = SPK::load(spk_path)?;
    let almanac = Almanac::default().with_spk(spk);
    let cell = PLANETARY_EPHEMERIS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    *guard = Some(Arc::new(almanac));
    Ok(())
}

/// Try to get a clone of the planetary almanac if initialized
pub fn get_planetary_ephemeris() -> Option<Arc<Almanac>> {
    PLANETARY_EPHEMERIS
        .get()
        .and_then(|m| m.lock().unwrap().as_ref().cloned())
}

/// Check whether the planetary almanac is initialized
pub fn is_planetary_ephemeris_initialized() -> bool {
    PLANETARY_EPHEMERIS
        .get()
        .map(|m| m.lock().unwrap().is_some())
        .unwrap_or(false)
}

/// Optional helper to download a SPK file from a URL to a local path.
/// Uses ureq; failures return an io::Error or a ureq::Error.
pub fn download_planetary_ephemeris(
    url: &str,
    dest_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Simple HTTP GET request for our SPK file
    let resp = ureq::get(url).call()?;
    if resp.status() != 200 {
        return Err(format!("Failed to download {}: HTTP {}", url, resp.status()).into());
    }

    // Create parent directory if it doesn't exist
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write response body to file
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(dest_path)?;
    std::io::copy(&mut reader, &mut file)?;
    Ok(())
}

/// Ensure the planetary ephemeris SPK (e.g., de440.bsp or de440s.bsp) is present and initialize the
/// planetary almanac. This is for Moon/planet calculations, NOT spacecraft.
pub fn ensure_planetary_ephemeris(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Check if almanac is already initialized
    if is_planetary_ephemeris_initialized() {
        return Ok(());
    }

    // Try to load from the given path
    init_planetary_ephemeris(path)?;
    Ok(())
}

/// Choose the best available planetary SPK path on disk.
/// Always prefer full DE440 if present, otherwise fall back to DE440S.
pub fn best_available_planetary_path() -> Option<std::path::PathBuf> {
    use crate::utils::config::{DEFAULT_DE440S_PATH, DEFAULT_DE440_PATH};
    let full = DEFAULT_DE440_PATH.as_path();
    if full.exists() {
        return Some(full.to_path_buf());
    }
    let slim = DEFAULT_DE440S_PATH.as_path();
    if slim.exists() {
        return Some(slim.to_path_buf());
    }
    None
}
