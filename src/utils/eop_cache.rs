//! Shared EOP2 download and caching utilities
//!
//! Provides a single place to download the JPL EOP2 "short" file,
//! store it in a local cache, and read it back with a TTL.
//! Both UT1 and polar motion providers can consume the returned text.

use crate::utils::config::DEFAULT_EOP_PATH;
use crate::utils::config::DEFAULT_EOP_TTL;
use crate::utils::config::EOP2_URL;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime};

fn fetch_eop2_text() -> Result<String, Box<dyn Error>> {
    let response = ureq::get(EOP2_URL).call()?;
    Ok(response.into_string()?)
}

fn try_read_fresh_cache(path: &Path, ttl: Duration) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if let Ok(modified) = meta.modified() {
        if let Ok(age) = SystemTime::now().duration_since(modified) {
            if age <= ttl {
                if let Ok(mut f) = fs::File::open(path) {
                    let mut buf = String::new();
                    if f.read_to_string(&mut buf).is_ok() {
                        eprintln!(
                            "EOP2 text loaded from cache: {} (age: {}s)",
                            path.display(),
                            age.as_secs()
                        );
                        return Some(buf);
                    }
                }
            }
        }
    }
    None
}

fn save_cache(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = fs::File::create(path) {
        let _ = f.write_all(body.as_bytes());
    }
}

fn try_read_stale_cache(path: &Path) -> Option<String> {
    if path.exists() {
        if let Ok(mut f) = fs::File::open(path) {
            let mut buf = String::new();
            if f.read_to_string(&mut buf).is_ok() {
                eprintln!("EOP2 text loaded from STALE cache: {}", path.display());
                return Some(buf);
            }
        }
    }
    None
}

/// Load EOP2 text payload: use cache if fresh, or download and update cache.
/// Falls back to stale cache if network fails.
pub fn load_or_download_eop2_text() -> Result<String, Box<dyn Error>> {
    let path = DEFAULT_EOP_PATH.clone();
    let ttl = Duration::from_secs(DEFAULT_EOP_TTL);
    if let Some(text) = try_read_fresh_cache(&path, ttl) {
        return Ok(text);
    }
    match fetch_eop2_text() {
        Ok(text) => {
            save_cache(&path, &text);
            Ok(text)
        }
        Err(e) => {
            eprintln!("Warning: failed to download EOP2 after cache miss ({e}).");
            if let Some(text) = try_read_stale_cache(&path) {
                return Ok(text);
            }
            Err("Unable to load EOP2 from cache or network".into())
        }
    }
}
