//! TLE parsing and fetching utilities
//!
//! Provides utilities for:
//! - Parsing 2-line and 3-line TLE formats
//! - Reading TLEs from files
//! - Downloading TLEs from URLs with caching
//! - Fetching TLEs from Celestrak by NORAD ID or name
//! - Extracting TLE epoch information

use crate::utils::config::CACHE_DIR;
#[allow(unused_imports)]
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Celestrak GP TLE API endpoint
const CELESTRAK_API_BASE: &str = "https://celestrak.org/NORAD/elements/gp.php";

/// TTL for cached TLE downloads (24 hours)
const TLE_CACHE_TTL: u64 = 86_400;

/// Result of parsing a TLE - contains the two lines and optional satellite name
#[derive(Debug, Clone)]
pub struct TLEData {
    pub line1: String,
    pub line2: String,
    #[allow(dead_code)]
    pub name: Option<String>,
}

/// Parse TLE from a string that may be 2 or 3 lines
///
/// Supports:
/// - 2-line format: line1\nline2
/// - 3-line format: name\nline1\nline2
///
/// Lines can be separated by \n or \r\n
pub fn parse_tle_string(content: &str) -> Result<TLEData, Box<dyn Error>> {
    // Normalize line endings and split
    let normalized = content.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized
        .split('\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    match lines.len() {
        2 => {
            // 2-line format
            validate_tle_lines(lines[0], lines[1])?;
            Ok(TLEData {
                line1: lines[0].to_string(),
                line2: lines[1].to_string(),
                name: None,
            })
        }
        3 => {
            // 3-line format: first line is the name
            validate_tle_lines(lines[1], lines[2])?;
            Ok(TLEData {
                line1: lines[1].to_string(),
                line2: lines[2].to_string(),
                name: Some(lines[0].to_string()),
            })
        }
        _ => Err(format!(
            "Invalid TLE format: expected 2 or 3 lines, got {}",
            lines.len()
        )
        .into()),
    }
}

/// Validate that two lines are valid TLE lines
fn validate_tle_lines(line1: &str, line2: &str) -> Result<(), Box<dyn Error>> {
    // Check line 1
    if !line1.starts_with('1') || line1.len() < 69 {
        return Err(format!("Invalid TLE line 1: {}", line1).into());
    }

    // Check line 2
    if !line2.starts_with('2') || line2.len() < 69 {
        return Err(format!("Invalid TLE line 2: {}", line2).into());
    }

    // Check that both lines have the same satellite number
    let sat_num1 = line1
        .get(2..7)
        .ok_or("Invalid TLE line 1: missing satellite number")?
        .trim();
    let sat_num2 = line2
        .get(2..7)
        .ok_or("Invalid TLE line 2: missing satellite number")?
        .trim();

    if sat_num1 != sat_num2 {
        return Err(format!(
            "TLE line satellite numbers don't match: {} vs {}",
            sat_num1, sat_num2
        )
        .into());
    }

    Ok(())
}

/// Read TLE from a file
pub fn read_tle_file(path: &str) -> Result<TLEData, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    parse_tle_string(&content)
}

/// Download TLE from a URL
fn download_tle(url: &str) -> Result<String, Box<dyn Error>> {
    let response = ureq::get(url).call()?;
    Ok(response.into_string()?)
}

/// Get cache path for a URL
fn get_url_cache_path(url: &str) -> PathBuf {
    // Create a simple hash of the URL to use as filename
    let hash = format!("{:x}", md5::compute(url.as_bytes()));
    let mut path = CACHE_DIR.clone();
    path.push("tle_cache");
    path.push(format!("{}.tle", hash));
    path
}

/// Try to read TLE from cache if it's fresh
fn try_read_fresh_cache(path: &Path, ttl: Duration) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if let Ok(modified) = meta.modified() {
        if let Ok(age) = SystemTime::now().duration_since(modified) {
            if age <= ttl {
                if let Ok(content) = fs::read_to_string(path) {
                    // Only print debug info in debug builds
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "TLE loaded from cache: {} (age: {}s)",
                        path.display(),
                        age.as_secs()
                    );
                    return Some(content);
                }
            }
        }
    }
    None
}

/// Save TLE content to cache
fn save_to_cache(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            // Log error but don't fail - caching is optional
            #[cfg(debug_assertions)]
            eprintln!("Warning: Failed to create TLE cache directory: {}", e);
            return;
        }
    }
    if let Err(e) = fs::File::create(path).and_then(|mut f| f.write_all(content.as_bytes())) {
        // Log error but don't fail - caching is optional
        #[cfg(debug_assertions)]
        eprintln!("Warning: Failed to write TLE to cache: {}", e);
    }
}

/// Download TLE from URL with caching
pub fn download_tle_with_cache(url: &str) -> Result<TLEData, Box<dyn Error>> {
    let cache_path = get_url_cache_path(url);
    let ttl = Duration::from_secs(TLE_CACHE_TTL);

    // Try to use cached version
    if let Some(content) = try_read_fresh_cache(&cache_path, ttl) {
        return parse_tle_string(&content);
    }

    // Download fresh TLE
    let content = download_tle(url)?;
    save_to_cache(&cache_path, &content);
    parse_tle_string(&content)
}

/// Fetch TLE from Celestrak by NORAD ID
pub fn fetch_tle_by_norad_id(norad_id: u32) -> Result<TLEData, Box<dyn Error>> {
    let url = format!("{}?CATNR={}&FORMAT=TLE", CELESTRAK_API_BASE, norad_id);
    download_tle_with_cache(&url)
}

/// Fetch TLE from Celestrak by satellite name
pub fn fetch_tle_by_name(name: &str) -> Result<TLEData, Box<dyn Error>> {
    // Simple URL encoding for satellite names
    // Replace spaces and special characters
    let encoded_name = name
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('#', "%23");
    let url = format!("{}?NAME={}&FORMAT=TLE", CELESTRAK_API_BASE, encoded_name);
    download_tle_with_cache(&url)
}

/// Extract TLE epoch from TLE lines
///
/// Returns the epoch as a DateTime<Utc>
///
/// TLE year convention (as per TLE specification):
/// - Years 57-99 represent 1957-1999 (20th century)
/// - Years 00-56 represent 2000-2056 (21st century)
///   This convention will need updating after 2056
pub fn extract_tle_epoch(line1: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    // TLE epoch is in columns 19-32 (0-indexed 18-31)
    let epoch_str = line1
        .get(18..32)
        .ok_or("Invalid TLE line 1: missing epoch")?
        .trim();

    // Format: YYDDD.DDDDDDDD where YY is year, DDD is day of year, and .DDDDDDDD is fractional day
    let year_str = &epoch_str[0..2];
    let day_str = &epoch_str[2..];

    let year: i32 = year_str.parse()?;
    // Determine century: 57-99 = 1900s, 00-56 = 2000s (following TLE convention)
    let full_year = if year >= 57 { 1900 + year } else { 2000 + year };

    let day_of_year_with_frac: f64 = day_str.parse()?;
    let day_of_year = day_of_year_with_frac.floor() as u32;

    // Validate day of year
    if !(1..=366).contains(&day_of_year) {
        return Err(format!("Invalid day of year in TLE: {}", day_of_year).into());
    }

    let frac_day = day_of_year_with_frac.fract();

    // Convert day of year to date
    let base_date = NaiveDate::from_ymd_opt(full_year, 1, 1)
        .ok_or("Invalid year in TLE")?
        .and_hms_opt(0, 0, 0)
        .ok_or("Invalid time")?;

    // Add days (day_of_year - 1 because day 1 is Jan 1)
    let date = base_date + chrono::Duration::days((day_of_year - 1) as i64);

    // Add fractional day
    let seconds = (frac_day * 86400.0) as i64;
    let datetime = date + chrono::Duration::seconds(seconds);

    Ok(DateTime::from_naive_utc_and_offset(datetime, Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tle_2_lines() {
        let tle = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995\n2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530";
        let result = parse_tle_string(tle).unwrap();
        assert!(result.name.is_none());
        assert_eq!(result.line1.len(), 69);
        assert_eq!(result.line2.len(), 69);
    }

    #[test]
    fn test_parse_tle_3_lines() {
        let tle = "ISS (ZARYA)\n1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
        let result = parse_tle_string(tle).unwrap();
        assert_eq!(result.name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(result.line1.len(), 69);
        assert_eq!(result.line2.len(), 69);
    }

    #[test]
    fn test_extract_epoch() {
        let line1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995";
        let epoch = extract_tle_epoch(line1).unwrap();
        // Day 287 of 2025 is October 14
        assert_eq!(epoch.year(), 2025);
        assert_eq!(epoch.month(), 10);
        assert_eq!(epoch.day(), 14);
    }

    #[test]
    fn test_validate_tle_lines() {
        let line1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995";
        let line2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530";
        assert!(validate_tle_lines(line1, line2).is_ok());

        // Test mismatched satellite numbers
        let line2_bad = "2 99999  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530";
        assert!(validate_tle_lines(line1, line2_bad).is_err());
    }
}
