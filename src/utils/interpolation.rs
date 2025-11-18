//! Interpolation utilities for ephemeris data
//!
//! Provides Hermite interpolation for state vectors (position and velocity)

use chrono::{DateTime, Utc};
use ndarray::Array2;

/// Hermite interpolation for state vectors (position and velocity)
///
/// Uses cubic Hermite interpolation which is appropriate for position-velocity data.
/// This method ensures C1 continuity (continuous first derivative).
///
/// # Arguments
/// * `query_times` - Times at which to interpolate
/// * `data_times` - Known data times (must be sorted)
/// * `data_states` - Known state vectors (N x 6: position xyz, velocity xyz) in km and km/s
///
/// # Returns
/// Interpolated state vectors (M x 6) where M = query_times.len()
///
/// # Panics
/// Panics if data_times and data_states have different lengths or if data has fewer than 2 points
pub fn hermite_interpolate(
    query_times: &[DateTime<Utc>],
    data_times: &[DateTime<Utc>],
    data_states: &Array2<f64>,
) -> Array2<f64> {
    assert_eq!(data_times.len(), data_states.nrows());
    assert!(data_times.len() >= 2, "Need at least 2 data points");

    let mut result = Array2::<f64>::zeros((query_times.len(), 6));

    // Convert times to seconds since first epoch for easier computation
    let t0 = data_times[0];
    let data_t_secs: Vec<f64> = data_times
        .iter()
        .map(|t| (*t - t0).num_milliseconds() as f64 / 1000.0)
        .collect();

    for (out_idx, query_time) in query_times.iter().enumerate() {
        let query_t_sec = (*query_time - t0).num_milliseconds() as f64 / 1000.0;

        // Find the interval containing query_time
        let idx = find_interval(&data_t_secs, query_t_sec);

        // Get the two bracketing points
        let t0_val = data_t_secs[idx];
        let t1_val = data_t_secs[idx + 1];
        let dt = t1_val - t0_val;

        // Normalized parameter [0, 1]
        let t = (query_t_sec - t0_val) / dt;

        // Hermite basis functions
        let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2); // position at t0
        let h10 = t * (1.0 - t).powi(2); // velocity at t0
        let h01 = t.powi(2) * (3.0 - 2.0 * t); // position at t1
        let h11 = t.powi(2) * (t - 1.0); // velocity at t1

        // Interpolate position (columns 0-2) and velocity (columns 3-5) separately
        for i in 0..3 {
            // Position interpolation
            let p0 = data_states[[idx, i]];
            let p1 = data_states[[idx + 1, i]];
            let v0 = data_states[[idx, i + 3]];
            let v1 = data_states[[idx + 1, i + 3]];

            result[[out_idx, i]] = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;

            // Velocity interpolation (derivative of position interpolation)
            let dh00_dt = 6.0 * t * (t - 1.0) / dt;
            let dh10_dt = (1.0 - 4.0 * t + 3.0 * t.powi(2)) / dt;
            let dh01_dt = 6.0 * t * (1.0 - t) / dt;
            let dh11_dt = (3.0 * t.powi(2) - 2.0 * t) / dt;

            result[[out_idx, i + 3]] =
                dh00_dt * p0 + dh10_dt * dt * v0 + dh01_dt * p1 + dh11_dt * dt * v1;
        }
    }

    result
}

/// Find the interval index for a query time
///
/// Returns the index i such that data_times[i] <= query_time < data_times[i+1]
/// Clamps to valid range if query_time is outside the data range.
///
/// # Arguments
/// * `data_times` - Sorted array of time values (in seconds)
/// * `query_time` - Query time value (in seconds)
///
/// # Returns
/// Index i where the query time falls between data_times[i] and data_times[i+1]
fn find_interval(data_times: &[f64], query_time: f64) -> usize {
    // Handle edge cases
    if query_time <= data_times[0] {
        return 0;
    }
    if query_time >= data_times[data_times.len() - 1] {
        return data_times.len() - 2;
    }

    // Binary search for the interval
    let mut left = 0;
    let mut right = data_times.len() - 1;

    while right - left > 1 {
        let mid = (left + right) / 2;
        if query_time < data_times[mid] {
            right = mid;
        } else {
            left = mid;
        }
    }

    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_hermite_interpolation_linear() {
        // Test with simple linear motion
        let t0 = Utc::now();
        let data_times = vec![t0, t0 + Duration::seconds(10), t0 + Duration::seconds(20)];

        // Position increases linearly at 1 km/s in x direction
        let mut data_states = Array2::<f64>::zeros((3, 6));
        data_states[[0, 0]] = 0.0; // x at t0
        data_states[[1, 0]] = 10.0; // x at t0+10s
        data_states[[2, 0]] = 20.0; // x at t0+20s
        data_states[[0, 3]] = 1.0; // vx at t0
        data_states[[1, 3]] = 1.0; // vx at t0+10s
        data_states[[2, 3]] = 1.0; // vx at t0+20s

        // Query at t0 + 5s
        let query_times = vec![t0 + Duration::seconds(5)];
        let result = hermite_interpolate(&query_times, &data_times, &data_states);

        // Should be approximately at x=5.0, vx=1.0
        assert!((result[[0, 0]] - 5.0).abs() < 0.1);
        assert!((result[[0, 3]] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_find_interval() {
        let times = vec![0.0, 10.0, 20.0, 30.0];

        assert_eq!(find_interval(&times, 5.0), 0);
        assert_eq!(find_interval(&times, 15.0), 1);
        assert_eq!(find_interval(&times, 25.0), 2);

        // Edge cases
        assert_eq!(find_interval(&times, -5.0), 0); // before range
        assert_eq!(find_interval(&times, 35.0), 2); // after range
    }
}
