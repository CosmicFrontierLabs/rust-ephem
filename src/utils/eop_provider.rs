//! Earth Orientation Parameters (EOP) Provider
//!
//! This module provides access to Earth Orientation Parameters (EOP) from IERS data,
//! including polar motion (xp, yp), UT1-UTC offsets, and celestial pole offsets.
//!
//! Data is downloaded from JPL's EOP2 service and cached for reuse.

use crate::utils::config::ARCSEC_TO_RAD;
use crate::utils::eop_cache::load_or_download_eop2_text;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// A single EOP data record
#[derive(Debug, Clone)]
pub struct EopRecord {
    /// Modified Julian Date (MJD)
    pub mjd: f64,
    /// X-component of polar motion in arcseconds
    pub xp: f64,
    /// Y-component of polar motion in arcseconds
    pub yp: f64,
}

/// Earth Orientation Parameters provider
#[derive(Debug, Clone)]
pub struct EopProvider {
    records: Vec<EopRecord>,
}

// EopRecord is a plain data holder; conversions are provided by EopProvider/free functions.

impl EopProvider {
    /// Load from cache if available/fresh; otherwise download and update cache
    pub fn load_or_download() -> Result<Self, Box<dyn std::error::Error>> {
        let text = load_or_download_eop2_text()?;
        Self::from_eop2_data(text)
    }

    /// Parse EOP2 CSV data format
    ///
    /// Format: MJD, PMx(mas), PMy(mas), TAI-UT1(ms), ... (additional columns ignored)
    /// Lines starting with '#' or '$' are comments
    pub fn from_eop2_data(data: String) -> Result<Self, Box<dyn std::error::Error>> {
        let mut records = Vec::new();

        for raw_line in data.lines() {
            // Trim and skip obvious comments / empties
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('$') {
                continue;
            }

            // The JPL EOP2 file contains metadata lines like EOP2LBL='...', EOP2UT1='UT1'
            // Robust filter: require first non-whitespace char to be a digit or '-'
            if let Some(first) = line.chars().find(|c| !c.is_whitespace()) {
                if !(first.is_ascii_digit() || first == '-') {
                    // Skip metadata / label line
                    continue;
                }
            } else {
                continue; // all whitespace (already handled but safe)
            }

            // Split CSV fields; some lines end with a date tag after a '$', but splitting by comma is safe.
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            // Need at least MJD, PMx(mas), PMy(mas)
            if parts.len() < 3 {
                eprintln!("EOP2 parse: skipping short line: {line}");
                continue;
            }

            // Parse numerics defensively; skip line on failure instead of failing entire provider
            let mjd = match parts[0].parse::<f64>() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "EOP2 parse: invalid MJD field '{}' in line: {}",
                        parts[0], line
                    );
                    continue;
                }
            };
            let xp_mas = match parts[1].parse::<f64>() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "EOP2 parse: invalid PMx field '{}' in line: {}",
                        parts[1], line
                    );
                    continue;
                }
            };
            let yp_mas = match parts[2].parse::<f64>() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "EOP2 parse: invalid PMy field '{}' in line: {}",
                        parts[2], line
                    );
                    continue;
                }
            };

            // Convert milliarcseconds -> arcseconds
            let xp = xp_mas / 1000.0;
            let yp = yp_mas / 1000.0;

            records.push(EopRecord { mjd, xp, yp });
        }

        // Sort by MJD for efficient lookup (ignore partial_cmp None since MJD parsed as f64)
        records.sort_by(|a, b| a.mjd.partial_cmp(&b.mjd).unwrap());

        Ok(EopProvider { records })
    }

    /// Convert DateTime<Utc> to Modified Julian Date
    /// Made public so integration tests can validate conversion.
    pub fn datetime_to_mjd(dt: &DateTime<Utc>) -> f64 {
        // MJD = JD - 2400000.5
        // JD for Unix epoch (1970-01-01 00:00:00) = 2440587.5
        // MJD for Unix epoch = 40587.0
        const MJD_UNIX_EPOCH: f64 = 40587.0;
        const SECONDS_PER_DAY: f64 = 86400.0;

        let unix_seconds = dt.timestamp() as f64;
        let unix_nanos = dt.timestamp_subsec_nanos() as f64;
        let total_seconds = unix_seconds + unix_nanos / 1e9;

        MJD_UNIX_EPOCH + total_seconds / SECONDS_PER_DAY
    }

    /// Get polar motion (xp, yp) for a given datetime
    ///
    /// Returns (xp, yp) in arcseconds, or (0.0, 0.0) if data is unavailable
    pub fn get_polar_motion(&self, dt: &DateTime<Utc>) -> (f64, f64) {
        let mjd = Self::datetime_to_mjd(dt);

        if self.records.is_empty() {
            return (0.0, 0.0);
        }

        // Find surrounding records for interpolation
        let idx = match self
            .records
            .binary_search_by(|record| record.mjd.partial_cmp(&mjd).unwrap())
        {
            Ok(i) => return (self.records[i].xp, self.records[i].yp),
            Err(i) => i,
        };

        // Check if we're outside the data range
        if idx == 0 {
            // Before first record
            return (self.records[0].xp, self.records[0].yp);
        }

        if idx >= self.records.len() {
            // After last record
            let last = &self.records[self.records.len() - 1];
            return (last.xp, last.yp);
        }

        // Linear interpolation between records
        let r0 = &self.records[idx - 1];
        let r1 = &self.records[idx];

        let t = (mjd - r0.mjd) / (r1.mjd - r0.mjd);
        let xp = r0.xp + t * (r1.xp - r0.xp);
        let yp = r0.yp + t * (r1.yp - r0.yp);

        (xp, yp)
    }

    /// Get polar motion in radians for a given datetime
    ///
    /// Returns (xp_rad, yp_rad) in radians
    pub fn get_polar_motion_rad(&self, dt: &DateTime<Utc>) -> (f64, f64) {
        let (xp, yp) = self.get_polar_motion(dt);
        (xp * ARCSEC_TO_RAD, yp * ARCSEC_TO_RAD)
    }

    /// Get the number of records
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if there are no records
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get the MJD range of available data
    pub fn mjd_range(&self) -> Option<(f64, f64)> {
        if self.records.is_empty() {
            None
        } else {
            Some((
                self.records[0].mjd,
                self.records[self.records.len() - 1].mjd,
            ))
        }
    }
}

/// Cached EOP provider from JPL
static EOP_PROVIDER: Lazy<Mutex<Option<EopProvider>>> = Lazy::new(|| {
    // Try to initialize by loading from cache or downloading
    match EopProvider::load_or_download() {
        Ok(provider) => {
            eprintln!(
                "EOP provider initialized successfully with {} records",
                provider.len()
            );
            if let Some((start, end)) = provider.mjd_range() {
                eprintln!("  MJD range: {start} to {end}");
            }
            Mutex::new(Some(provider))
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not initialize EOP provider: {e}. Polar motion will default to (0, 0)"
            );
            eprintln!("This is expected if you don't have internet access or JPL servers are unreachable.");
            Mutex::new(None)
        }
    }
});

/// Get polar motion (xp, yp) for a given UTC datetime
///
/// Returns (xp, yp) in arcseconds, or (0.0, 0.0) if data is not available.
pub fn get_polar_motion(dt: &DateTime<Utc>) -> (f64, f64) {
    let provider_lock = EOP_PROVIDER.lock().unwrap();

    if let Some(provider) = provider_lock.as_ref() {
        provider.get_polar_motion(dt)
    } else {
        (0.0, 0.0)
    }
}

/// Get polar motion in radians for a given UTC datetime
///
/// Returns (xp_rad, yp_rad) in radians, or (0.0, 0.0) if data is not available.
pub fn get_polar_motion_rad(dt: &DateTime<Utc>) -> (f64, f64) {
    let provider_lock = EOP_PROVIDER.lock().unwrap();

    if let Some(provider) = provider_lock.as_ref() {
        provider.get_polar_motion_rad(dt)
    } else {
        (0.0, 0.0)
    }
}

// UT1-UTC retrieval is intentionally omitted here; the crate uses the existing UT1 provider.

/// Initialize or re-initialize the EOP provider
///
/// This can be called to force a refresh of IERS data. Returns true if successful.
pub fn init_eop_provider() -> bool {
    let mut provider_lock = EOP_PROVIDER.lock().unwrap();

    match EopProvider::load_or_download() {
        Ok(provider) => {
            *provider_lock = Some(provider);
            true
        }
        Err(e) => {
            eprintln!("Error initializing EOP provider: {e}");
            false
        }
    }
}

/// Check if EOP provider is available
pub fn is_eop_available() -> bool {
    EOP_PROVIDER.lock().unwrap().is_some()
}

// Tests moved to tests/eop_provider_tests.rs
