/// JPL Horizons API integration for querying body ephemerides
///
/// This module provides functions to query NASA's JPL Horizons system
/// for solar system body positions and velocities when SPICE kernels
/// are not available or do not contain the required data.
use chrono::{DateTime, Datelike, TimeZone, Utc};
use ndarray::Array2;

/// Query JPL Horizons for body ephemeris data by NAIF ID in GCRS frame
///
/// Returns positions and velocities in GCRS (Geocentric Celestial Reference System)
/// frame, which is directly usable for observer-relative calculations.
/// The returned array has shape (N, 6) containing [x, y, z, vx, vy, vz] in km and km/s.
///
/// # Arguments
/// * `times` - Vector of timestamps for which to calculate positions
/// * `body_id` - NAIF ID of the target body (numeric)
///
/// # Returns
/// `Ok(Array2<f64>)` with shape (N, 6) containing GCRS [x, y, z, vx, vy, vz]
/// or `Err(String)` if the query fails
///
/// # Note
/// This function queries JPL Horizons API directly via HTTP to retrieve GCRS
/// coordinates, which avoids frame conversion issues that can arise from
/// converting heliocentric coordinates to observer-relative.
pub fn query_horizons_body(times: &[DateTime<Utc>], body_id: i32) -> Result<Array2<f64>, String> {
    if times.is_empty() {
        return Err("No times provided for Horizons query".to_string());
    }

    let start_time = times[0];
    let end_time = times[times.len() - 1];

    // Query Horizons API for GCRS vectors (observer = Earth = 399)
    // Format timestamps for Horizons: YYYY-MM-DD (date only to avoid URL encoding issues with spaces)
    let start_str = format!(
        "{}-{:02}-{:02}",
        start_time.year(),
        start_time.month(),
        start_time.day()
    );

    // Add one day to end_time to ensure we get data even when start/end are the same day
    // Horizons needs different start/end dates to return ephemeris data
    let end_plus_day = end_time + chrono::Duration::days(1);
    let end_str = format!(
        "{}-{:02}-{:02}",
        end_plus_day.year(),
        end_plus_day.month(),
        end_plus_day.day()
    );

    // Step size in minutes - use 1 minute steps for good coverage
    let step_minutes = 1;

    // Build Horizons API URL with ICRF/equatorial output (VECTORS format)
    // CENTER='@399' = geocentric
    // REF_PLANE='FRAME' = use ICRF reference frame (equatorial, not ecliptic)
    // VEC_TABLE='2' = position and velocity
    let url = format!(
        "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{}'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&VEC_TABLE='2'&CENTER='@399'&REF_PLANE='FRAME'&START_TIME='{}'&STOP_TIME='{}'&STEP_SIZE='{}m'&OUT_UNITS='KM-S'&CSV_FORMAT='YES'",
        body_id, start_str, end_str, step_minutes
    );

    // Query the API
    let response = ureq::get(&url)
        .call()
        .map_err(|e| format!("Horizons API request failed: {}", e))?;

    // Read response body as string
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| format!("Failed to read Horizons response: {}", e))?;

    // Debug: check if response contains error
    if body.contains("ERROR") || body.contains("error") {
        eprintln!("Horizons error response for body {}:", body_id);
        eprintln!("{}", body);
        return Err("Horizons returned error".to_string());
    }

    // Parse the response
    parse_horizons_csv_response(&body, times)
}

/// Parse Horizons CSV response and interpolate to requested times
fn parse_horizons_csv_response(
    response: &str,
    times: &[DateTime<Utc>],
) -> Result<Array2<f64>, String> {
    let mut data_lines = Vec::new();

    // Skip header lines until we find the data section
    let mut in_data_section = false;
    for line in response.lines() {
        if line.contains("$$SOE") {
            in_data_section = true;
            continue;
        }
        if line.contains("$$EOE") {
            break;
        }
        if in_data_section && !line.trim().is_empty() {
            data_lines.push(line);
        }
    }

    if data_lines.is_empty() {
        return Err("No ephemeris data found in Horizons response".to_string());
    }

    // Parse CSV lines into position/velocity data
    let mut horizons_times = Vec::new();
    let mut horizons_data = Vec::new();

    for line in data_lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            continue; // Skip malformed lines - need JDTDB, CalDate, X, Y, Z, VX, VY, VZ
        }

        // Parse time (first column is JDTDB)
        let time_str = parts[0].trim();
        // Horizons time format: Julian Date
        if let Ok(time) = parse_horizons_datetime(time_str) {
            horizons_times.push(time);

            // Parse X, Y, Z, VX, VY, VZ (columns 3-8, skipping JDTDB and CalDate)
            let mut pv = Vec::new();
            for part in parts.iter().skip(2).take(6) {
                if let Ok(val) = part.trim().parse::<f64>() {
                    pv.push(val);
                } else {
                    return Err(format!("Failed to parse number: '{}'", part.trim()));
                }
            }
            if pv.len() == 6 {
                horizons_data.push(pv);
            }
        }
    }

    if horizons_data.is_empty() {
        return Err("Could not parse any ephemeris data from Horizons".to_string());
    }

    // Interpolate to requested times
    interpolate_horizons_data(&horizons_times, &horizons_data, times)
}

/// Parse Horizons Julian Date to DateTime<Utc>
fn parse_horizons_datetime(s: &str) -> Result<DateTime<Utc>, String> {
    // Parse Julian Date (e.g., "2461018.500000000")
    let jd: f64 = s
        .parse()
        .map_err(|_| format!("Invalid Julian Date: {}", s))?;

    // Convert Julian Date to Unix timestamp
    // JD 2440587.5 = Unix epoch (1970-01-01 00:00:00 UTC)
    let unix_seconds = (jd - 2440587.5) * 86400.0;

    // Create DateTime from timestamp
    let secs = unix_seconds.floor() as i64;
    let nsecs = ((unix_seconds - unix_seconds.floor()) * 1e9) as u32;

    Utc.timestamp_opt(secs, nsecs)
        .single()
        .ok_or_else(|| format!("Invalid timestamp from JD {}", jd))
}

/// Interpolate Horizons data to requested times using nearest-neighbor
fn interpolate_horizons_data(
    horizons_times: &[DateTime<Utc>],
    horizons_data: &[Vec<f64>],
    requested_times: &[DateTime<Utc>],
) -> Result<Array2<f64>, String> {
    let mut result = Array2::<f64>::zeros((requested_times.len(), 6));

    for (i, &req_time) in requested_times.iter().enumerate() {
        // Find nearest horizons time
        let nearest_idx = horizons_times
            .iter()
            .enumerate()
            .min_by_key(|(_, &t)| (t - req_time).num_seconds().abs())
            .map(|(idx, _)| idx)
            .ok_or("No horizons data available")?;

        // Copy data
        for j in 0..6 {
            result[[i, j]] = horizons_data[nearest_idx][j];
        }
    }

    Ok(result)
}

/// Query JPL Horizons for body ephemeris data by name (for comets and other named objects)
///
/// This function allows querying comets and other non-standard objects by name,
/// such as "Halley", "67P", "NEOWISE", etc.
///
/// # Arguments
/// * `_times` - Vector of timestamps for which to calculate positions
/// * `body_name` - Name of the comet or object (e.g., "Halley", "C/2020 F3", "67P")
///
/// # Returns
/// `Ok(Array2<f64>)` with shape (N, 6) containing heliocentric [x, y, z, vx, vy, vz]
/// or `Err(String)` if the query fails or the body name is not recognized by Horizons
///
/// # Note
/// This uses Horizons' name-based object search. For well-known comets, use common
/// names like "Halley" or "Hale-Bopp". For SOHO discoveries, use the full designation
/// like "C/2020 F3 (NEOWISE)".
///
/// Currently, this function requires a numeric NAIF ID to query. For pure name-based
/// queries, please use the JPL Horizons web interface to find the NAIF ID first, then
/// use the ID with query_horizons_body().
pub fn query_horizons_body_by_name(
    _times: &[DateTime<Utc>],
    body_name: &str,
) -> Result<Array2<f64>, String> {
    Err(format!(
        "Comet/object name-based query '{}' is not yet fully supported. \
         Please look up the NAIF ID at https://ssd.jpl.nasa.gov/horizons/ \
         and use query_horizons_body() with the numeric ID instead.",
        body_name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    #[ignore] // Ignore by default since it requires network access
    fn test_query_horizons_mars() {
        let times = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        ];

        let result = query_horizons_body(&times, 499); // Mars NAIF ID
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.shape(), &[2, 6]);

        // Check that we got reasonable values (Mars is roughly at AU scale)
        let pos_mag = (data[[0, 0]].powi(2) + data[[0, 1]].powi(2) + data[[0, 2]].powi(2)).sqrt();
        assert!(pos_mag > 1e8 && pos_mag < 5e8); // Roughly 1-5 AU in km
    }
}
