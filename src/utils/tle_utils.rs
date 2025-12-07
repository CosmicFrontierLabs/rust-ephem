//! TLE parsing and fetching utilities
//!
//! Provides utilities for:
//! - Parsing 2-line and 3-line TLE formats
//! - Reading TLEs from files
//! - Downloading TLEs from URLs with caching
//! - Fetching TLEs from Celestrak by NORAD ID or name
//! - Fetching TLEs from Space-Track.org by NORAD ID with epoch support
//! - Extracting TLE epoch information
//! - Unified TLE fetching from multiple sources

use crate::utils::config::{
    CACHE_DIR, CELESTRAK_API_BASE, DEFAULT_EPOCH_TOLERANCE_DAYS, SPACETRACK_API_BASE,
    SPACETRACK_LOGIN_URL, SPACETRACK_PASSWORD_ENV, SPACETRACK_USERNAME_ENV, TLE_CACHE_TTL,
};
#[allow(unused_imports)]
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
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
    let mut response = ureq::get(url).call()?;
    Ok(response.body_mut().read_to_string()?)
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
        if let Err(_e) = fs::create_dir_all(parent) {
            // Log error but don't fail - caching is optional
            #[cfg(debug_assertions)]
            eprintln!("Warning: Failed to create TLE cache directory: {}", _e);
            return;
        }
    }
    if let Err(_e) = fs::File::create(path).and_then(|mut f| f.write_all(content.as_bytes())) {
        // Log error but don't fail - caching is optional
        #[cfg(debug_assertions)]
        eprintln!("Warning: Failed to write TLE to cache: {}", _e);
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

// ============================================================================
// Space-Track.org API Support
// ============================================================================

/// Credentials for Space-Track.org authentication
#[derive(Clone)]
pub struct SpaceTrackCredentials {
    pub username: String,
    pub password: String,
}

impl std::fmt::Debug for SpaceTrackCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpaceTrackCredentials")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}
impl SpaceTrackCredentials {
    /// Create new credentials from explicit values
    pub fn new(username: String, password: String) -> Self {
        Self { username, password }
    }

    /// Load credentials from environment variables or .env file
    ///
    /// Checks in order:
    /// 1. Environment variables SPACETRACK_USERNAME and SPACETRACK_PASSWORD
    /// 2. .env file in the current directory
    /// 3. .env file in the user's home directory
    pub fn from_env() -> Result<Self, Box<dyn Error>> {
        // Try to load .env file (silently ignore if not found)
        let _ = dotenvy::dotenv();

        // Also try home directory .env
        if let Some(home_dir) = dirs::home_dir() {
            let home_env = home_dir.join(".env");
            let _ = dotenvy::from_path(home_env);
        }

        let username = std::env::var(SPACETRACK_USERNAME_ENV).map_err(|_| {
            format!(
                "Space-Track.org username not found. Set {} environment variable or pass credentials explicitly.",
                SPACETRACK_USERNAME_ENV
            )
        })?;

        let password = std::env::var(SPACETRACK_PASSWORD_ENV).map_err(|_| {
            format!(
                "Space-Track.org password not found. Set {} environment variable or pass credentials explicitly.",
                SPACETRACK_PASSWORD_ENV
            )
        })?;

        Ok(Self { username, password })
    }
}

/// Extended TLE data with epoch information for Space-Track caching
#[derive(Debug, Clone)]
pub struct TLEDataWithEpoch {
    pub tle: TLEData,
    pub epoch: DateTime<Utc>,
}

/// Get cache path for Space-Track TLE with NORAD ID
fn get_spacetrack_cache_path(norad_id: u32) -> PathBuf {
    let mut path = CACHE_DIR.clone();
    path.push("spacetrack_cache");
    path.push(format!("{}.tle", norad_id));
    path
}

/// Try to read Space-Track TLE from cache if epoch is within tolerance
///
/// # Arguments
/// * `path` - Path to the cached TLE file
/// * `target_epoch` - The target epoch we want the TLE for
/// * `tolerance_days` - How many days off the TLE epoch can be from target
///
/// # Returns
/// Some(TLEDataWithEpoch) if cache is valid, None otherwise
fn try_read_spacetrack_cache(
    path: &Path,
    target_epoch: &DateTime<Utc>,
    tolerance_days: f64,
) -> Option<TLEDataWithEpoch> {
    // Check if file exists and read content
    let content = fs::read_to_string(path).ok()?;

    // Parse the TLE
    let tle = parse_tle_string(&content).ok()?;

    // Extract the TLE epoch
    let tle_epoch = extract_tle_epoch(&tle.line1).ok()?;

    // Check if the TLE epoch is within tolerance of the target epoch
    let diff_seconds = (*target_epoch - tle_epoch).num_seconds().abs() as f64;
    let tolerance_seconds = tolerance_days * 86400.0;

    if diff_seconds <= tolerance_seconds {
        #[cfg(debug_assertions)]
        eprintln!(
            "Space-Track TLE loaded from cache: {} (epoch diff: {:.2} days)",
            path.display(),
            diff_seconds / 86400.0
        );
        Some(TLEDataWithEpoch {
            tle,
            epoch: tle_epoch,
        })
    } else {
        #[cfg(debug_assertions)]
        eprintln!(
            "Space-Track cache TLE epoch ({}) too far from target ({}): {:.2} days > {:.2} days tolerance",
            tle_epoch, target_epoch, diff_seconds / 86400.0, tolerance_days
        );
        None
    }
}

/// Save TLE content to Space-Track cache
fn save_to_spacetrack_cache(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        if let Err(_e) = fs::create_dir_all(parent) {
            #[cfg(debug_assertions)]
            eprintln!(
                "Warning: Failed to create Space-Track cache directory: {}",
                _e
            );
            return;
        }
    }
    if let Err(_e) = fs::File::create(path).and_then(|mut f| f.write_all(content.as_bytes())) {
        #[cfg(debug_assertions)]
        eprintln!("Warning: Failed to write TLE to Space-Track cache: {}", _e);
    }
}

/// Authenticate with Space-Track.org and create an authenticated agent
///
/// Space-Track uses cookie-based session authentication.
fn create_spacetrack_agent(
    credentials: &SpaceTrackCredentials,
) -> Result<ureq::Agent, Box<dyn Error>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .into();

    // Login to Space-Track.org
    let form_data = [
        ("identity", credentials.username.as_str()),
        ("password", credentials.password.as_str()),
    ];
    let mut login_response = agent.post(SPACETRACK_LOGIN_URL).send_form(form_data)?;

    if login_response.status() != 200 {
        return Err(format!(
            "Space-Track.org login failed with status: {}",
            login_response.status()
        )
        .into());
    }

    // Check the response body for login errors
    let body = login_response.body_mut().read_to_string()?;
    if body.contains("\"Login\":\"Failed\"") || body.contains("Login Failed") {
        return Err("Space-Track.org login failed: Invalid credentials".into());
    }

    Ok(agent)
}

/// Fetch TLE from Space-Track.org by NORAD ID for a specific epoch
///
/// This function queries Space-Track.org's GP history data to find the TLE
/// closest to the specified epoch.
///
/// # Arguments
/// * `norad_id` - NORAD catalog ID of the satellite
/// * `target_epoch` - The epoch for which to find the closest TLE
/// * `credentials` - Optional credentials (will try env vars if None)
/// * `epoch_tolerance_days` - How many days tolerance for cache matching (default: 4.0)
///
/// # Returns
/// TLEDataWithEpoch containing the TLE lines and epoch
pub fn fetch_tle_from_spacetrack(
    norad_id: u32,
    target_epoch: &DateTime<Utc>,
    credentials: Option<SpaceTrackCredentials>,
    epoch_tolerance_days: Option<f64>,
) -> Result<TLEDataWithEpoch, Box<dyn Error>> {
    let tolerance = epoch_tolerance_days.unwrap_or(DEFAULT_EPOCH_TOLERANCE_DAYS);
    let cache_path = get_spacetrack_cache_path(norad_id);

    // Try to use cached version if epoch is within tolerance
    if let Some(cached) = try_read_spacetrack_cache(&cache_path, target_epoch, tolerance) {
        return Ok(cached);
    }

    // Get credentials
    let creds = credentials
        .map(Ok)
        .unwrap_or_else(SpaceTrackCredentials::from_env)?;

    // Create authenticated agent
    let agent = create_spacetrack_agent(&creds)?;

    // Format the epoch range for the query
    // We search for TLEs within the tolerance window around the target epoch
    let start_epoch = *target_epoch - chrono::Duration::days(tolerance as i64);
    let end_epoch = *target_epoch + chrono::Duration::days(tolerance as i64);

    let start_str = start_epoch.format("%Y-%m-%d").to_string();
    let end_str = end_epoch.format("%Y-%m-%d").to_string();

    // Query the GP history class to find TLEs near the target epoch
    // We order by epoch descending and get the one closest to our target
    let query_url = format!(
        "{}/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{}/EPOCH/{}--{}/orderby/EPOCH%20desc/format/tle",
        SPACETRACK_API_BASE, norad_id, start_str, end_str
    );

    #[cfg(debug_assertions)]
    eprintln!("Space-Track query URL: {}", query_url);

    let mut response = agent.get(&query_url).call()?;

    if response.status() != 200 {
        return Err(format!(
            "Space-Track.org query failed with status: {}",
            response.status()
        )
        .into());
    }

    let body = response.body_mut().read_to_string()?;

    if body.trim().is_empty() {
        return Err(format!(
            "No TLE found on Space-Track.org for NORAD ID {} near epoch {}",
            norad_id, target_epoch
        )
        .into());
    }

    // The response may contain multiple TLEs, find the one closest to target epoch
    let best_tle = find_closest_tle_to_epoch(&body, target_epoch)?;

    // Save to cache
    let cache_content = format!("{}\n{}", best_tle.tle.line1, best_tle.tle.line2);
    save_to_spacetrack_cache(&cache_path, &cache_content);

    Ok(best_tle)
}

/// Parse multiple TLEs from Space-Track response and find the one closest to target epoch
fn find_closest_tle_to_epoch(
    content: &str,
    target_epoch: &DateTime<Utc>,
) -> Result<TLEDataWithEpoch, Box<dyn Error>> {
    let normalized = content.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized
        .split('\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let mut best_tle: Option<TLEDataWithEpoch> = None;
    let mut best_diff = i64::MAX;

    // Process lines in pairs (TLE line 1 and line 2)
    let mut i = 0;
    while i + 1 < lines.len() {
        let line1 = lines[i];
        let line2 = lines[i + 1];

        // Check if these are valid TLE lines
        if line1.starts_with('1') && line2.starts_with('2') {
            if let Ok(tle) = parse_tle_string(&format!("{}\n{}", line1, line2)) {
                if let Ok(epoch) = extract_tle_epoch(&tle.line1) {
                    let diff = (*target_epoch - epoch).num_seconds().abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_tle = Some(TLEDataWithEpoch { tle, epoch });
                    }
                }
            }
            i += 2;
        } else {
            // Skip non-TLE lines (might be a 3-line format with name)
            i += 1;
        }
    }

    best_tle.ok_or_else(|| "No valid TLE found in Space-Track response".into())
}

// ============================================================================
// TLE Fetching with Caching
// ============================================================================

/// Result from the unified TLE fetch operation
#[derive(Debug, Clone)]
pub struct FetchedTLE {
    pub line1: String,
    pub line2: String,
    pub name: Option<String>,
    pub epoch: DateTime<Utc>,
    pub source: &'static str,
}

impl FetchedTLE {
    /// Create a FetchedTLE from raw TLE lines
    ///
    /// This is useful when you have TLE lines directly (e.g., from a TLERecord object)
    /// and want to create a FetchedTLE without going through the fetch process.
    pub fn from_lines(
        line1: String,
        line2: String,
        name: Option<String>,
        epoch: Option<DateTime<Utc>>,
    ) -> Result<Self, Box<dyn Error>> {
        // Validate the TLE lines
        validate_tle_lines(&line1, &line2)?;

        // Extract epoch from line1 if not provided
        let epoch = match epoch {
            Some(e) => e,
            None => extract_tle_epoch(&line1)?,
        };

        Ok(FetchedTLE {
            line1,
            line2,
            name,
            epoch,
            source: "direct",
        })
    }
}

/// Build SpaceTrack credentials from optional username/password parameters
///
/// This helper consolidates the credential-building logic used in multiple places.
///
/// # Arguments
/// * `username` - Optional explicit username
/// * `password` - Optional explicit password
///
/// # Returns
/// - `Ok(Some(credentials))` if both username and password are provided
/// - `Ok(Some(credentials))` if neither provided but env vars are available
/// - `Ok(None)` if neither provided and env vars are not available
/// - `Err` if only one of username/password is provided
pub fn build_credentials(
    username: Option<&str>,
    password: Option<&str>,
) -> Result<Option<SpaceTrackCredentials>, &'static str> {
    match (username, password) {
        (Some(u), Some(p)) => Ok(Some(SpaceTrackCredentials::new(
            u.to_string(),
            p.to_string(),
        ))),
        (None, None) => Ok(SpaceTrackCredentials::from_env().ok()),
        _ => Err("Both spacetrack_username and spacetrack_password must be provided together, or omit both to use environment variables"),
    }
}

/// Unified TLE fetching from multiple sources
///
/// This function consolidates TLE fetching logic used by both `fetch_tle()` Python API
/// and `TLEEphemeris::new()`. It handles:
/// - File paths and URLs via the `tle` parameter
/// - NORAD ID lookups with Space-Track.org (with Celestrak failover)
/// - Satellite name lookups via Celestrak
///
/// # Arguments
/// * `tle_path` - Optional file path or URL to a TLE file
/// * `norad_id` - Optional NORAD catalog ID
/// * `norad_name` - Optional satellite name for Celestrak lookup
/// * `target_epoch` - Optional target epoch for Space-Track lookups
/// * `credentials` - Optional Space-Track.org credentials
/// * `epoch_tolerance_days` - Optional tolerance for Space-Track cache matching
/// * `enforce_source` - Optional source enforcement: "celestrak", "spacetrack", or None for default behavior
///
/// # Returns
/// `FetchedTLE` containing the TLE data, epoch, and source information
pub fn fetch_tle_unified(
    tle_path: Option<&str>,
    norad_id: Option<u32>,
    norad_name: Option<&str>,
    target_epoch: Option<&DateTime<Utc>>,
    credentials: Option<SpaceTrackCredentials>,
    epoch_tolerance_days: Option<f64>,
    enforce_source: Option<&str>,
) -> Result<FetchedTLE, Box<dyn Error>> {
    if let Some(tle_param) = tle_path {
        // tle parameter: file path or URL
        let (tle_data, src) =
            if tle_param.starts_with("http://") || tle_param.starts_with("https://") {
                let data = download_tle_with_cache(tle_param)?;
                (data, "url")
            } else {
                let data = read_tle_file(tle_param)?;
                (data, "file")
            };
        let epoch = extract_tle_epoch(&tle_data.line1)?;
        Ok(FetchedTLE {
            line1: tle_data.line1,
            line2: tle_data.line2,
            name: tle_data.name,
            epoch,
            source: src,
        })
    } else if let Some(nid) = norad_id {
        match enforce_source {
            Some("celestrak") => {
                // Enforce Celestrak only
                let tle_data = fetch_tle_by_norad_id(nid)?;
                let epoch = extract_tle_epoch(&tle_data.line1)?;
                Ok(FetchedTLE {
                    line1: tle_data.line1,
                    line2: tle_data.line2,
                    name: tle_data.name,
                    epoch,
                    source: "celestrak",
                })
            }
            Some("spacetrack") => {
                // Enforce Space-Track only
                let creds = credentials.ok_or(
                    "Space-Track.org credentials required when enforce_source='spacetrack'",
                )?;
                let target = target_epoch.cloned().unwrap_or_else(chrono::Utc::now);
                let tle_with_epoch =
                    fetch_tle_from_spacetrack(nid, &target, Some(creds), epoch_tolerance_days)?;
                Ok(FetchedTLE {
                    line1: tle_with_epoch.tle.line1,
                    line2: tle_with_epoch.tle.line2,
                    name: tle_with_epoch.tle.name,
                    epoch: tle_with_epoch.epoch,
                    source: "spacetrack",
                })
            }
            Some(other) => Err(format!(
                "Invalid enforce_source value: {}. Must be 'celestrak', 'spacetrack', or None",
                other
            )
            .into()),
            None => {
                // Default behavior: try Space-Track first, failover to Celestrak
                if let Some(creds) = credentials {
                    // Use provided target_epoch or current time
                    let target = target_epoch.cloned().unwrap_or_else(chrono::Utc::now);

                    match fetch_tle_from_spacetrack(nid, &target, Some(creds), epoch_tolerance_days)
                    {
                        Ok(tle_with_epoch) => Ok(FetchedTLE {
                            line1: tle_with_epoch.tle.line1,
                            line2: tle_with_epoch.tle.line2,
                            name: tle_with_epoch.tle.name,
                            epoch: tle_with_epoch.epoch,
                            source: "spacetrack",
                        }),
                        Err(_spacetrack_err) => {
                            // Failover to Celestrak
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "Space-Track.org fetch failed, falling back to Celestrak: {}",
                                _spacetrack_err
                            );
                            let tle_data = fetch_tle_by_norad_id(nid)?;
                            let epoch = extract_tle_epoch(&tle_data.line1)?;
                            Ok(FetchedTLE {
                                line1: tle_data.line1,
                                line2: tle_data.line2,
                                name: tle_data.name,
                                epoch,
                                source: "celestrak",
                            })
                        }
                    }
                } else {
                    // No Space-Track credentials, use Celestrak directly
                    let tle_data = fetch_tle_by_norad_id(nid)?;
                    let epoch = extract_tle_epoch(&tle_data.line1)?;
                    Ok(FetchedTLE {
                        line1: tle_data.line1,
                        line2: tle_data.line2,
                        name: tle_data.name,
                        epoch,
                        source: "celestrak",
                    })
                }
            }
        }
    } else if let Some(name_query) = norad_name {
        // Fetch from Celestrak by satellite name
        let tle_data = fetch_tle_by_name(name_query)?;
        let epoch = extract_tle_epoch(&tle_data.line1)?;
        Ok(FetchedTLE {
            line1: tle_data.line1,
            line2: tle_data.line2,
            name: tle_data.name,
            epoch,
            source: "celestrak",
        })
    } else {
        Err("Must provide one of: tle path/URL, norad_id, or norad_name".into())
    }
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
