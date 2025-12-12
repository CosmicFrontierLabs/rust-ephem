//! OMM (Orbital Mean-Elements Message) parsing and fetching utilities
//!
//! Provides utilities for:
//! - Parsing CCSDS OMM format
//! - Fetching OMM data from Space-Track.org and Celestrak
//! - Converting OMM to SGP4-compatible TLE format

use crate::utils::config::{
    CACHE_DIR, DEFAULT_EPOCH_TOLERANCE_DAYS, OMM_CACHE_TTL, SPACETRACK_API_BASE,
    SPACETRACK_USERNAME_ENV,
};
use chrono::{DateTime, Utc};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// OMM data structure containing mean orbital elements
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OMMData {
    pub norad_cat_id: u32,
    pub object_name: Option<String>,
    pub epoch: DateTime<Utc>,
    pub mean_motion: f64, // revolutions per day
    pub eccentricity: f64,
    pub inclination: f64,       // degrees
    pub ra_of_asc_node: f64,    // degrees
    pub arg_of_pericenter: f64, // degrees
    pub mean_anomaly: f64,      // degrees
    pub ephemeris_type: u32,
    pub classification_type: Option<String>,
    pub norad_cat_id_str: String,
    pub element_set_no: u32,
    pub rev_at_epoch: u64,
    pub bstar: f64,
    pub mean_motion_dot: f64,
    pub mean_motion_ddot: f64,
    pub semimajor_axis: Option<f64>, // km
    pub period: Option<f64>,         // minutes
    pub apoapsis: Option<f64>,       // km
    pub periapsis: Option<f64>,      // km
}

/// Result of parsing OMM data
#[derive(Debug, Clone)]
pub struct FetchedOMM {
    pub data: OMMData,
}

/// Parse OMM data from a string in CCSDS OMM format (key-value pairs)
#[allow(dead_code)]
pub fn parse_omm_string(content: &str) -> Result<OMMData, Box<dyn Error>> {
    let lines = content.lines();

    let mut norad_cat_id = None;
    let mut object_name = None;
    let mut epoch = None;
    let mut mean_motion = None;
    let mut eccentricity = None;
    let mut inclination = None;
    let mut ra_of_asc_node = None;
    let mut arg_of_pericenter = None;
    let mut mean_anomaly = None;
    let mut ephemeris_type = None;
    let mut classification_type = None;
    let mut norad_cat_id_str = None;
    let mut element_set_no = None;
    let mut rev_at_epoch: Option<u64> = None;
    let mut bstar = None;
    let mut mean_motion_dot = None;
    let mut mean_motion_ddot = None;
    let mut semimajor_axis = None;
    let mut period = None;
    let mut apoapsis = None;
    let mut periapsis = None;

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("COMMENT") {
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();

            match key {
                "NORAD_CAT_ID" => {
                    norad_cat_id = Some(value.parse()?);
                    norad_cat_id_str = Some(value.to_string());
                }
                "OBJECT_NAME" => {
                    object_name = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "EPOCH" => {
                    // Parse ISO 8601 datetime
                    epoch = Some(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc));
                }
                "MEAN_MOTION" => mean_motion = Some(value.parse()?),
                "ECCENTRICITY" => eccentricity = Some(value.parse()?),
                "INCLINATION" => inclination = Some(value.parse()?),
                "RA_OF_ASC_NODE" => ra_of_asc_node = Some(value.parse()?),
                "ARG_OF_PERICENTER" => arg_of_pericenter = Some(value.parse()?),
                "MEAN_ANOMALY" => mean_anomaly = Some(value.parse()?),
                "EPHEMERIS_TYPE" => ephemeris_type = Some(value.parse()?),
                "CLASSIFICATION_TYPE" => {
                    classification_type = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "ELEMENT_SET_NO" => element_set_no = Some(value.parse()?),
                "REV_AT_EPOCH" => rev_at_epoch = Some(value.parse()?),
                "BSTAR" => bstar = Some(value.parse()?),
                "MEAN_MOTION_DOT" => mean_motion_dot = Some(value.parse()?),
                "MEAN_MOTION_DDOT" => mean_motion_ddot = Some(value.parse()?),
                "SEMIMAJOR_AXIS" => semimajor_axis = Some(value.parse()?),
                "PERIOD" => period = Some(value.parse()?),
                "APOAPSIS" => apoapsis = Some(value.parse()?),
                "PERIAPSIS" => periapsis = Some(value.parse()?),
                _ => {} // Ignore unknown fields
            }
        }
    }

    Ok(OMMData {
        norad_cat_id: norad_cat_id.ok_or("Missing NORAD_CAT_ID")?,
        object_name,
        epoch: epoch.ok_or("Missing EPOCH")?,
        mean_motion: mean_motion.ok_or("Missing MEAN_MOTION")?,
        eccentricity: eccentricity.ok_or("Missing ECCENTRICITY")?,
        inclination: inclination.ok_or("Missing INCLINATION")?,
        ra_of_asc_node: ra_of_asc_node.ok_or("Missing RA_OF_ASC_NODE")?,
        arg_of_pericenter: arg_of_pericenter.ok_or("Missing ARG_OF_PERICENTER")?,
        mean_anomaly: mean_anomaly.ok_or("Missing MEAN_ANOMALY")?,
        ephemeris_type: ephemeris_type.ok_or("Missing EPHEMERIS_TYPE")?,
        classification_type,
        norad_cat_id_str: norad_cat_id_str.ok_or("Missing NORAD_CAT_ID")?,
        element_set_no: element_set_no.ok_or("Missing ELEMENT_SET_NO")?,
        rev_at_epoch: rev_at_epoch.ok_or("Missing REV_AT_EPOCH")?,
        bstar: bstar.ok_or("Missing BSTAR")?,
        mean_motion_dot: mean_motion_dot.ok_or("Missing MEAN_MOTION_DOT")?,
        mean_motion_ddot: mean_motion_ddot.ok_or("Missing MEAN_MOTION_DDOT")?,
        semimajor_axis,
        period,
        apoapsis,
        periapsis,
    })
}

/// Parse OMM data from Celestrak/Space-Track JSON format
pub fn parse_omm_json(content: &str) -> Result<OMMData, Box<dyn Error>> {
    use serde_json::Value;

    let content = content.trim();
    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG: About to parse JSON, length: {}", content.len());
        eprintln!(
            "DEBUG: First 100 chars: {:?}",
            &content[..content.len().min(100)]
        );
    }
    let json: Value = serde_json::from_str(content).map_err(|e| {
        let err_msg = format!(
            "JSON parse error at position {}: {} (content length: {})",
            e.column(),
            e,
            content.len()
        );
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: {}", err_msg);
        err_msg
    })?;

    #[cfg(debug_assertions)]
    eprintln!("DEBUG: JSON parsed successfully!");

    // Celestrak returns an array with one object
    let obj = json
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|obj| obj.as_object())
        .ok_or_else(|| {
            format!(
                "Invalid JSON format: expected array with object, got: {:?}",
                json
            )
        })?;

    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Extracted object from array");

    // Helper function to get f64 values (handles both numbers and strings)
    let get_f64 = |key: &str| -> Result<f64, Box<dyn Error>> {
        match obj.get(key) {
            Some(value) => {
                // Try as number first, then as string
                if let Some(num) = value.as_f64() {
                    Ok(num)
                } else if let Some(s) = value.as_str() {
                    s.parse::<f64>().map_err(|e| {
                        format!(
                            "Field {} cannot be parsed as f64: {} (value: {:?})",
                            key, e, s
                        )
                        .into()
                    })
                } else {
                    Err(format!("Field {} is not a number or string: {:?}", key, value).into())
                }
            }
            None => Err(format!("Missing {} field", key).into()),
        }
    };

    // Helper function to get u32 values (handles both numbers and strings)
    let get_u32 = |key: &str| -> Result<u32, Box<dyn Error>> {
        match obj.get(key) {
            Some(value) => {
                // Try as number first, then as string
                if let Some(num) = value.as_u64() {
                    Ok(num as u32)
                } else if let Some(s) = value.as_str() {
                    s.parse::<u32>().map_err(|e| {
                        format!(
                            "Field {} cannot be parsed as u32: {} (value: {:?})",
                            key, e, s
                        )
                        .into()
                    })
                } else {
                    Err(format!("Field {} is not a number or string: {:?}", key, value).into())
                }
            }
            None => Err(format!(
                "Missing {} field. Available keys: {:?}",
                key,
                obj.keys().collect::<Vec<_>>()
            )
            .into()),
        }
    };

    // Helper function to get u64 values (handles both numbers and strings)
    let get_u64 = |key: &str| -> Result<u64, Box<dyn Error>> {
        match obj.get(key) {
            Some(value) => {
                if let Some(num) = value.as_u64() {
                    Ok(num)
                } else if let Some(s) = value.as_str() {
                    s.parse::<u64>().map_err(|e| {
                        format!(
                            "Field {} cannot be parsed as u64: {} (value: {:?})",
                            key, e, s
                        )
                        .into()
                    })
                } else {
                    Err(format!("Field {} is not a number or string: {:?}", key, value).into())
                }
            }
            None => Err(format!("Missing {} field", key).into()),
        }
    };

    // Helper function to get optional f64 values (handles both numbers and strings)
    let get_opt_f64 = |key: &str| -> Option<f64> {
        obj.get(key).and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
        })
    };

    // Helper function to get string values
    let get_string = |key: &str| -> Result<String, Box<dyn Error>> {
        obj.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("Missing or invalid {} field", key).into())
    };

    // Helper function to get optional string values
    let get_opt_string = |key: &str| -> Option<String> {
        obj.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    };

    let norad_cat_id = get_u32("NORAD_CAT_ID")?;
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Got NORAD_CAT_ID: {}", norad_cat_id);
    let epoch_raw = get_string("EPOCH")?;
    let epoch_str = epoch_raw.trim();
    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG: Got EPOCH raw: {:?}", epoch_raw);
        eprintln!("DEBUG: Got EPOCH trimmed: {:?}", epoch_str);
    }

    // Parse epoch - handle both formats: with and without timezone.
    // Some sources omit timezone (e.g. "2025-12-12T03:41:57.165504"). Treat those as UTC.
    let epoch = if epoch_str.ends_with('Z')
        || epoch_str.contains('+')
        || (epoch_str.matches('-').count() > 2)
    {
        // Has explicit timezone (has more than 2 dashes, meaning a timezone offset like -05:00)
        DateTime::parse_from_rfc3339(epoch_str)
            .map_err(|e| {
                format!(
                    "Failed to parse EPOCH (timezone) as RFC3339: {:?} (bytes: {:02X?}): {}",
                    epoch_str,
                    epoch_str.as_bytes(),
                    e
                )
            })?
            .with_timezone(&Utc)
    } else {
        // No timezone, assume UTC by appending 'Z' and parsing as RFC3339.
        let epoch_rfc3339 = format!("{}Z", epoch_str);
        #[cfg(debug_assertions)]
        eprintln!("DEBUG: EPOCH assumed-UTC RFC3339: {:?}", epoch_rfc3339);
        DateTime::parse_from_rfc3339(&epoch_rfc3339)
            .map_err(|e| {
                format!(
                    "Failed to parse EPOCH (assumed UTC) as RFC3339: {:?} (bytes: {:02X?}): {}",
                    epoch_rfc3339,
                    epoch_rfc3339.as_bytes(),
                    e
                )
            })?
            .with_timezone(&Utc)
    };

    Ok(OMMData {
        norad_cat_id,
        object_name: get_opt_string("OBJECT_NAME"),
        epoch,
        mean_motion: get_f64("MEAN_MOTION")?,
        eccentricity: get_f64("ECCENTRICITY")?,
        inclination: get_f64("INCLINATION")?,
        ra_of_asc_node: get_f64("RA_OF_ASC_NODE")?,
        arg_of_pericenter: get_f64("ARG_OF_PERICENTER")?,
        mean_anomaly: get_f64("MEAN_ANOMALY")?,
        ephemeris_type: get_u32("EPHEMERIS_TYPE")?,
        classification_type: get_opt_string("CLASSIFICATION_TYPE"),
        norad_cat_id_str: norad_cat_id.to_string(),
        element_set_no: get_u32("ELEMENT_SET_NO")?,
        rev_at_epoch: get_u64("REV_AT_EPOCH")?,
        bstar: get_f64("BSTAR")?,
        mean_motion_dot: get_f64("MEAN_MOTION_DOT")?,
        mean_motion_ddot: get_f64("MEAN_MOTION_DDOT")?,
        semimajor_axis: get_opt_f64("SEMIMAJOR_AXIS"),
        period: get_opt_f64("PERIOD"),
        apoapsis: get_opt_f64("APOAPSIS"),
        periapsis: get_opt_f64("PERIAPSIS"),
    })
}

/// Convert OMM data to SGP4 Elements for direct propagation
pub fn omm_to_elements(omm: &OMMData) -> Result<sgp4::Elements, Box<dyn Error>> {
    use serde_json::json;

    // sgp4 deserializes OMM EPOCH into chrono::NaiveDateTime (no timezone).
    // If we include a timezone suffix ("Z" or "+00:00"), chrono will error with "trailing input".
    let epoch_omm = omm
        .epoch
        .naive_utc()
        .format("%Y-%m-%dT%H:%M:%S%.f")
        .to_string();

    let elements_json = json!({
        "OBJECT_NAME": omm.object_name,
        "OBJECT_ID": omm.norad_cat_id_str,
        "EPOCH": epoch_omm,
        "MEAN_MOTION": omm.mean_motion,
        "ECCENTRICITY": omm.eccentricity,
        "INCLINATION": omm.inclination,
        "RA_OF_ASC_NODE": omm.ra_of_asc_node,
        "ARG_OF_PERICENTER": omm.arg_of_pericenter,
        "MEAN_ANOMALY": omm.mean_anomaly,
        "EPHEMERIS_TYPE": omm.ephemeris_type,
        "CLASSIFICATION_TYPE": omm.classification_type,
        "NORAD_CAT_ID": omm.norad_cat_id,
        "ELEMENT_SET_NO": omm.element_set_no,
        "REV_AT_EPOCH": omm.rev_at_epoch,
        "BSTAR": omm.bstar,
        "MEAN_MOTION_DOT": omm.mean_motion_dot,
        "MEAN_MOTION_DDOT": omm.mean_motion_ddot
    });

    let elements: sgp4::Elements = serde_json::from_value(elements_json)?;
    Ok(elements)
}

/// Get cache path for OMM data by NORAD ID and source
fn get_omm_cache_path(norad_id: u32, source: &str) -> PathBuf {
    let mut path = CACHE_DIR.clone();
    path.push("omm_cache");
    path.push(format!("{}_{}.json", norad_id, source));
    path
}

/// Try to read OMM data from cache if it's fresh
fn try_read_fresh_omm_cache(path: &Path, ttl: Duration) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if let Ok(modified) = meta.modified() {
        if let Ok(age) = SystemTime::now().duration_since(modified) {
            if age <= ttl {
                if let Ok(content) = fs::read_to_string(path) {
                    // Only print debug info in debug builds
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "OMM loaded from cache: {} (age: {}s)",
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

/// Save OMM content to cache
fn save_omm_to_cache(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        if let Err(_e) = fs::create_dir_all(parent) {
            // Log error but don't fail - caching is optional
            #[cfg(debug_assertions)]
            eprintln!("Warning: Failed to create OMM cache directory: {}", _e);
            return;
        }
    }
    if let Err(_e) = fs::File::create(path).and_then(|mut f| f.write_all(content.as_bytes())) {
        // Log error but don't fail - caching is optional
        #[cfg(debug_assertions)]
        eprintln!("Warning: Failed to write OMM to cache: {}", _e);
    }
}

/// Fetch OMM data from Celestrak by NORAD ID with caching
pub fn fetch_omm_from_celestrak(norad_id: u32) -> Result<FetchedOMM, Box<dyn Error>> {
    let cache_path = get_omm_cache_path(norad_id, "celestrak");
    let ttl = Duration::from_secs(OMM_CACHE_TTL);

    // Try to use cached version
    if let Some(content) = try_read_fresh_omm_cache(&cache_path, ttl) {
        return parse_omm_json(&content).map(|data| FetchedOMM { data });
    }

    // Download fresh OMM data
    let url = format!(
        "https://celestrak.org/NORAD/elements/gp.php?CATNR={}&FORMAT=JSON",
        norad_id
    );

    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| format!("Failed to fetch OMM data from Celestrak: {}", e))?;

    if response.status() != 200 {
        return Err(format!("Celestrak query failed with status: {}", response.status()).into());
    }

    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read Celestrak response body: {}", e))?;

    #[cfg(debug_assertions)]
    eprintln!("Celestrak response body length: {}", body.len());

    // Always log the actual content for debugging
    eprintln!("DEBUG: Celestrak body: {:?}", &body[..body.len().min(200)]);
    eprintln!(
        "DEBUG: Celestrak body end: {:?}",
        &body[body.len().saturating_sub(50)..]
    );

    if body.trim().is_empty() || body.contains("No GP data found") {
        return Err(format!("No OMM data found for NORAD ID {} on Celestrak", norad_id).into());
    }

    // Cache the successful response
    save_omm_to_cache(&cache_path, &body);

    let omm_data = parse_omm_json(&body)?;
    Ok(FetchedOMM { data: omm_data })
}

/// Fetch OMM data from Space-Track.org by NORAD ID with caching
pub fn fetch_omm_from_spacetrack(
    norad_id: u32,
    target_epoch: Option<&DateTime<Utc>>,
    credentials: Option<crate::utils::tle_utils::SpaceTrackCredentials>,
) -> Result<FetchedOMM, Box<dyn Error>> {
    // Create cache key that includes epoch information
    let cache_key = if let Some(epoch) = target_epoch {
        format!("spacetrack_{}_epoch_{}", norad_id, epoch.format("%Y%m%d"))
    } else {
        format!("spacetrack_{}_latest", norad_id)
    };
    let cache_path = get_omm_cache_path(norad_id, &cache_key);
    let ttl = Duration::from_secs(OMM_CACHE_TTL);

    // Try to use cached version
    if let Some(content) = try_read_fresh_omm_cache(&cache_path, ttl) {
        return parse_omm_json(&content).map(|data| FetchedOMM { data });
    }

    // Download fresh OMM data
    let creds = credentials
        .map(Ok)
        .unwrap_or_else(|| crate::utils::tle_utils::SpaceTrackCredentials::from_env())?;

    let agent = crate::utils::tle_utils::create_spacetrack_agent(&creds)?;

    let query_url = if let Some(epoch) = target_epoch {
        // Query for OMM data near the target epoch in JSON format
        let start_epoch = *epoch - chrono::Duration::days(DEFAULT_EPOCH_TOLERANCE_DAYS as i64);
        let end_epoch = *epoch + chrono::Duration::days(DEFAULT_EPOCH_TOLERANCE_DAYS as i64);

        let start_str = start_epoch.format("%Y-%m-%d").to_string();
        let end_str = end_epoch.format("%Y-%m-%d").to_string();

        format!(
            "{}/basicspacedata/query/class/gp/NORAD_CAT_ID/{}/EPOCH/{}--{}/orderby/EPOCH%20desc/limit/1/format/json",
            SPACETRACK_API_BASE, norad_id, start_str, end_str
        )
    } else {
        // Query for latest OMM data in JSON format
        format!(
            "{}/basicspacedata/query/class/gp/NORAD_CAT_ID/{}/orderby/EPOCH%20desc/limit/1/format/json",
            SPACETRACK_API_BASE, norad_id
        )
    };

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
            "No OMM data found for NORAD ID {} on Space-Track.org",
            norad_id
        )
        .into());
    }

    // Cache the successful response
    save_omm_to_cache(&cache_path, &body);

    let omm_data = parse_omm_json(&body)?;
    Ok(FetchedOMM { data: omm_data })
}

/// Unified OMM fetching function that tries multiple sources
pub fn fetch_omm_unified(
    norad_id: Option<u32>,
    norad_name: Option<&str>,
    target_epoch: Option<&DateTime<Utc>>,
    spacetrack_credentials: Option<crate::utils::tle_utils::SpaceTrackCredentials>,
    enforce_source: Option<&str>,
) -> Result<FetchedOMM, Box<dyn Error>> {
    // Determine which source to use
    match enforce_source {
        Some("spacetrack") => {
            let nid = norad_id.ok_or("norad_id required when enforce_source='spacetrack'")?;
            return fetch_omm_from_spacetrack(nid, target_epoch, spacetrack_credentials);
        }
        Some("celestrak") => {
            let nid = norad_id.ok_or("norad_id required when enforce_source='celestrak'")?;
            return fetch_omm_from_celestrak(nid);
        }
        Some(other) => {
            return Err(format!(
                "Unknown enforce_source: {}. Valid options: spacetrack, celestrak",
                other
            )
            .into());
        }
        None => {} // Try both sources
    }

    // Try Space-Track first if credentials are available, then fall back to Celestrak
    if let Some(nid) = norad_id {
        if spacetrack_credentials.is_some() || std::env::var(SPACETRACK_USERNAME_ENV).is_ok() {
            match fetch_omm_from_spacetrack(nid, target_epoch, spacetrack_credentials.clone()) {
                Ok(result) => return Ok(result),
                Err(spacetrack_error) => {
                    // Fall back to Celestrak if Space-Track fails
                    match fetch_omm_from_celestrak(nid) {
                        Ok(result) => return Ok(result),
                        Err(celestrak_error) => {
                            return Err(format!("Both Space-Track.org and Celestrak OMM fetch failed. Space-Track error: {}. Celestrak error: {}",
                                spacetrack_error, celestrak_error).into());
                        }
                    }
                }
            }
        } else {
            // No Space-Track credentials, try Celestrak directly
            return fetch_omm_from_celestrak(nid);
        }
    }

    // If we have a name but no ID, we can't fetch OMM data
    if norad_name.is_some() {
        return Err("OMM fetching by name not supported. Use norad_id parameter.".into());
    }

    Err("Must provide norad_id parameter for OMM fetching".into())
}
