//! OMM (Orbital Mean-Elements Message) parsing and fetching utilities
//!
//! Provides utilities for:
//! - Parsing CCSDS OMM format
//! - Fetching OMM data from Space-Track.org and Celestrak
//! - Converting OMM to SGP4-compatible TLE format

use crate::utils::config::{
    DEFAULT_EPOCH_TOLERANCE_DAYS, SPACETRACK_API_BASE, SPACETRACK_USERNAME_ENV,
};
use chrono::{DateTime, Utc};
use std::error::Error;

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
    pub rev_at_epoch: f64,
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

/// Parse OMM data from a string in CCSDS OMM format
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
    let mut rev_at_epoch = None;
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

/// Convert OMM data to SGP4 Elements for direct propagation
pub fn omm_to_elements(omm: &OMMData) -> Result<sgp4::Elements, Box<dyn Error>> {
    use serde_json::json;

    let elements_json = json!({
        "OBJECT_NAME": omm.object_name,
        "OBJECT_ID": omm.norad_cat_id_str,
        "EPOCH": omm.epoch.to_rfc3339(),
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

/// Fetch OMM data from Celestrak by NORAD ID
/// Note: Celestrak does not provide OMM format, only TLE. This function returns an error.
pub fn fetch_omm_from_celestrak(_norad_id: u32) -> Result<FetchedOMM, Box<dyn Error>> {
    Err("Celestrak does not provide OMM data format. Use Space-Track.org for OMM data.".into())
}

/// Fetch OMM data from Space-Track.org by NORAD ID
pub fn fetch_omm_from_spacetrack(
    norad_id: u32,
    target_epoch: Option<&DateTime<Utc>>,
    credentials: Option<crate::utils::tle_utils::SpaceTrackCredentials>,
) -> Result<FetchedOMM, Box<dyn Error>> {
    let creds = credentials
        .map(Ok)
        .unwrap_or_else(|| crate::utils::tle_utils::SpaceTrackCredentials::from_env())?;

    let agent = crate::utils::tle_utils::create_spacetrack_agent(&creds)?;

    let query_url = if let Some(epoch) = target_epoch {
        // Query for OMM data near the target epoch
        let start_epoch = *epoch - chrono::Duration::days(DEFAULT_EPOCH_TOLERANCE_DAYS as i64);
        let end_epoch = *epoch + chrono::Duration::days(DEFAULT_EPOCH_TOLERANCE_DAYS as i64);

        let start_str = start_epoch.format("%Y-%m-%d").to_string();
        let end_str = end_epoch.format("%Y-%m-%d").to_string();

        format!(
            "{}/basicspacedata/query/class/omm/NORAD_CAT_ID/{}/EPOCH/{}--{}/orderby/EPOCH%20desc/format/omm",
            SPACETRACK_API_BASE, norad_id, start_str, end_str
        )
    } else {
        // Query for latest OMM data
        format!(
            "{}/basicspacedata/query/class/omm/NORAD_CAT_ID/{}/orderby/EPOCH%20desc/limit/1/format/omm",
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

    let omm_data = parse_omm_string(&body)?;
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

    // Try Space-Track first if credentials are available or norad_id is provided
    if let Some(nid) = norad_id {
        if spacetrack_credentials.is_some() || std::env::var(SPACETRACK_USERNAME_ENV).is_ok() {
            match fetch_omm_from_spacetrack(nid, target_epoch, spacetrack_credentials.clone()) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    return Err(format!("Space-Track.org OMM fetch failed: {}. OMM data requires Space-Track.org access.", e).into());
                }
            }
        } else {
            return Err("OMM data requires Space-Track.org credentials. Set SPACETRACK_USERNAME and SPACETRACK_PASSWORD environment variables or provide spacetrack_username/spacetrack_password parameters.".into());
        }
    }

    // If we have a name but no ID, we can't fetch OMM data
    if norad_name.is_some() {
        return Err("OMM fetching by name not supported. Use norad_id parameter.".into());
    }

    Err("Must provide norad_id parameter for OMM fetching".into())
}
