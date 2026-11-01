/// JPL Horizons API integration for querying body ephemerides
///
/// This module provides functions to query NASA's JPL Horizons system
/// for solar system body positions and velocities when SPICE kernels
/// are not available or do not contain the required data.
use chrono::{DateTime, Datelike, TimeZone, Utc};
use hifitime::Epoch;
use ndarray::Array2;
use std::cmp::Ordering;

use crate::utils::time_utils::chrono_to_epoch;

/// Translate NAIF asteroid IDs into Horizons designation searches.
///
/// Users continue to pass numeric NAIF/SPK IDs. The original and extended
/// asteroid ranges are documented by NAIF; the designation prefix prevents
/// Horizons from treating an SPK ID as a permanent asteroid number.
/// https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/naif_ids.html
fn horizons_command(body_id: i32) -> Result<String, String> {
    match body_id {
        // These Galileo-era IDs are exceptions to the ordinary asteroid scheme.
        // DES=2431010/2431011 would select unrelated asteroids 431010/431011.
        2_431_010 | 2_431_011 => {
            let name = if body_id == 2_431_010 {
                "Ida"
            } else {
                "Dactyl"
            };
            Err(format!(
                "Horizons fallback does not support legacy NAIF ID {} ({}); \
                 use a SPICE kernel containing this target",
                body_id, name
            ))
        }
        2_000_001..=3_999_999 | 20_000_001..=99_999_999 => Ok(format!("DES%3D{}%3B", body_id)),
        _ => Ok(body_id.to_string()),
    }
}

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

    let command = horizons_command(body_id)?;
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
        command, start_str, end_str, step_minutes
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

/// Parse Horizons JDTDB to DateTime<Utc>
fn parse_horizons_datetime(s: &str) -> Result<DateTime<Utc>, String> {
    // Parse JDTDB (e.g., "2461018.500000000")
    let jd_tdb: f64 = s
        .parse()
        .map_err(|_| format!("Invalid Julian Date: {}", s))?;

    // Horizons times are in TDB; convert to UTC via hifitime.
    let epoch = Epoch::from_jde_tdb(jd_tdb);
    let unix_seconds = epoch.to_unix_seconds();
    if !unix_seconds.is_finite() {
        return Err(format!("Invalid Unix seconds from JD {}", jd_tdb));
    }

    let mut secs = unix_seconds.floor() as i64;
    let mut nsecs = ((unix_seconds - secs as f64) * 1e9).round() as i64;
    if nsecs == 1_000_000_000 {
        secs += 1;
        nsecs = 0;
    }
    if nsecs < 0 {
        secs -= 1;
        nsecs += 1_000_000_000;
    }

    Utc.timestamp_opt(secs, nsecs as u32)
        .single()
        .ok_or_else(|| format!("Invalid timestamp from JD {}", jd_tdb))
}

/// Interpolate Horizons data to requested times using linear interpolation
fn interpolate_horizons_data(
    horizons_times: &[DateTime<Utc>],
    horizons_data: &[Vec<f64>],
    requested_times: &[DateTime<Utc>],
) -> Result<Array2<f64>, String> {
    let mut result = Array2::<f64>::zeros((requested_times.len(), 6));

    let first_time = horizons_times.first().ok_or("No horizons data available")?;
    let base_epoch = chrono_to_epoch(first_time);
    let horizons_seconds: Vec<f64> = horizons_times
        .iter()
        .map(|t| (chrono_to_epoch(t) - base_epoch).to_seconds())
        .collect();

    for (i, &req_time) in requested_times.iter().enumerate() {
        let req_seconds = (chrono_to_epoch(&req_time) - base_epoch).to_seconds();

        let idx = match horizons_seconds
            .binary_search_by(|t| t.partial_cmp(&req_seconds).unwrap_or(Ordering::Equal))
        {
            Ok(exact_idx) => {
                for j in 0..6 {
                    result[[i, j]] = horizons_data[exact_idx][j];
                }
                continue;
            }
            Err(insert_idx) => insert_idx,
        };

        if idx == 0 {
            for j in 0..6 {
                result[[i, j]] = horizons_data[0][j];
            }
            continue;
        }
        if idx >= horizons_seconds.len() {
            let last_idx = horizons_seconds.len() - 1;
            for j in 0..6 {
                result[[i, j]] = horizons_data[last_idx][j];
            }
            continue;
        }

        let left_idx = idx - 1;
        let right_idx = idx;
        let t_left = horizons_seconds[left_idx];
        let t_right = horizons_seconds[right_idx];
        let denom = t_right - t_left;
        let weight = if denom == 0.0 {
            0.0
        } else {
            (req_seconds - t_left) / denom
        };

        for j in 0..6 {
            let left_val = horizons_data[left_idx][j];
            let right_val = horizons_data[right_idx][j];
            result[[i, j]] = left_val + weight * (right_val - left_val);
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
    use crate::utils::time_utils::{chrono_to_epoch, get_tai_utc_offset};
    use chrono::Duration;
    use chrono::TimeZone;

    #[test]
    fn test_naif_asteroid_ids_use_horizons_designation_syntax() {
        assert_eq!(horizons_command(20000624).unwrap(), "DES%3D20000624%3B");
        assert_eq!(horizons_command(2000624).unwrap(), "DES%3D2000624%3B");
        assert_eq!(horizons_command(20136108).unwrap(), "DES%3D20136108%3B");
        assert_eq!(horizons_command(3000001).unwrap(), "DES%3D3000001%3B");
        assert_eq!(horizons_command(50000001).unwrap(), "DES%3D50000001%3B");
        for id in [624, 499, 301, 10, -82, 920065803, 120065803] {
            assert_eq!(horizons_command(id).unwrap(), id.to_string());
        }
    }

    #[test]
    fn test_legacy_ida_dactyl_ids_fail_without_querying_unrelated_asteroids() {
        let times: Vec<_> = (0..3)
            .map(|minute| Utc.with_ymd_and_hms(2029, 3, 1, 0, minute, 0).unwrap())
            .collect();
        for (legacy_id, name, unrelated_id) in [
            (2_431_010, "Ida", 20_431_010),
            (2_431_011, "Dactyl", 20_431_011),
        ] {
            let error = horizons_command(legacy_id).unwrap_err();
            assert!(error.contains(&format!("legacy NAIF ID {} ({})", legacy_id, name)));
            assert!(error.contains("does not support"));
            // The query must propagate the identity error before any HTTP request.
            assert_eq!(query_horizons_body(&times, legacy_id).unwrap_err(), error);
            assert_eq!(
                horizons_command(unrelated_id).unwrap(),
                format!("DES%3D{}%3B", unrelated_id)
            );
        }
    }

    #[test]
    fn test_recorded_small_body_responses_preserve_identity_and_vectors() {
        let responses = [
            (
                include_str!("../../tests/fixtures/horizons_moon.txt"),
                "Kiviuq",
            ),
            (
                include_str!("../../tests/fixtures/horizons_asteroid.txt"),
                "Hektor",
            ),
            (
                include_str!("../../tests/fixtures/horizons_dwarf_planet.txt"),
                "Haumea",
            ),
        ];
        let mut first_positions = Vec::new();
        for (response, name) in responses {
            assert!(response
                .lines()
                .any(|line| line.starts_with("Target body name:") && line.contains(name)));
            let line = response
                .split("$$SOE")
                .nth(1)
                .unwrap()
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap();
            let columns: Vec<_> = line.split(',').collect();
            let time = parse_horizons_datetime(columns[0].trim()).unwrap();
            let vectors = parse_horizons_csv_response(response, &[time]).unwrap();
            assert_eq!(vectors.shape(), &[1, 6]);
            for k in 0..6 {
                let expected: f64 = columns[k + 2].trim().parse().unwrap();
                assert!((vectors[[0, k]] - expected).abs() < 1e-9);
            }
            first_positions.push(vectors[[0, 0]]);
        }
        assert!((first_positions[0] - first_positions[1]).abs() > 1e6);
    }

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

    #[test]
    fn test_parse_horizons_datetime_tdb_roundtrip() {
        let dt = Utc.with_ymd_and_hms(2024, 6, 1, 12, 34, 56).unwrap();
        let jd_tdb = chrono_to_epoch(&dt).to_jde_tdb_days();
        let parsed = parse_horizons_datetime(&format!("{:.12}", jd_tdb))
            .expect("parse_horizons_datetime failed");
        let diff_ns = (parsed - dt).num_nanoseconds().unwrap().abs();
        assert!(diff_ns < 1_000_000);
    }

    #[test]
    fn test_parse_horizons_datetime_offsets_from_naive_jd() {
        let dt = Utc.with_ymd_and_hms(2024, 6, 1, 12, 34, 56).unwrap();
        let jd_tdb = chrono_to_epoch(&dt).to_jde_tdb_days();

        let parsed = parse_horizons_datetime(&format!("{:.12}", jd_tdb))
            .expect("parse_horizons_datetime failed");

        // Naive conversion treats JD(TDB) as if it were UTC.
        let unix_seconds_naive = (jd_tdb - 2440587.5) * 86400.0;
        let mut secs = unix_seconds_naive.floor() as i64;
        let mut nsecs = ((unix_seconds_naive - secs as f64) * 1e9).round() as i64;
        if nsecs == 1_000_000_000 {
            secs += 1;
            nsecs = 0;
        }
        if nsecs < 0 {
            secs -= 1;
            nsecs += 1_000_000_000;
        }
        let naive = Utc
            .timestamp_opt(secs, nsecs as u32)
            .single()
            .expect("naive JD conversion failed");

        let diff_ns = (parsed - naive).num_nanoseconds().unwrap();
        let diff_sec = diff_ns as f64 / 1e9;

        let tai_utc = get_tai_utc_offset(&parsed).expect("TAI-UTC offset missing");
        let expected = tai_utc + 32.184;
        assert!((diff_sec + expected).abs() < 0.05);
    }

    #[test]
    fn test_interpolate_horizons_data_linear() {
        let t0 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t1 = t0 + Duration::seconds(10);
        let horizons_times = vec![t0, t1];
        let horizons_data = vec![
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            vec![10.0, 20.0, 30.0, 4.0, 6.0, 8.0],
        ];
        let requested_times = vec![t0 + Duration::seconds(5)];
        let result =
            interpolate_horizons_data(&horizons_times, &horizons_data, &requested_times).unwrap();

        assert!((result[[0, 0]] - 5.0).abs() < 1e-9);
        assert!((result[[0, 1]] - 10.0).abs() < 1e-9);
        assert!((result[[0, 2]] - 15.0).abs() < 1e-9);
        assert!((result[[0, 3]] - 2.5).abs() < 1e-9);
        assert!((result[[0, 4]] - 4.0).abs() < 1e-9);
        assert!((result[[0, 5]] - 5.5).abs() < 1e-9);
    }
}
