//! Interpolation utilities for ephemeris data
//!
//! Provides Hermite interpolation for state vectors (position and velocity)

use crate::utils::time_utils::chrono_to_epoch;
use chrono::{DateTime, Utc};
use ndarray::Array2;

/// Difference between two DateTimes in seconds
#[inline]
fn diff_seconds(dt1: &DateTime<Utc>, dt2: &DateTime<Utc>) -> f64 {
    (chrono_to_epoch(dt1) - chrono_to_epoch(dt2)).to_seconds()
}

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
    let t0 = &data_times[0];
    let data_t_secs: Vec<f64> = data_times.iter().map(|t| diff_seconds(t, t0)).collect();

    for (out_idx, query_time) in query_times.iter().enumerate() {
        let query_t_sec = diff_seconds(query_time, t0);

        // Find the interval containing query_time
        let idx = find_interval(&data_t_secs, query_t_sec);

        // Get the two bracketing points
        let t0_val = data_t_secs[idx];
        let t1_val = data_t_secs[idx + 1];
        let dt = t1_val - t0_val;

        // Normalized parameter [0, 1]
        let t = (query_t_sec - t0_val) / dt;

        let before: [f64; 6] = std::array::from_fn(|k| data_states[[idx, k]]);
        let after: [f64; 6] = std::array::from_fn(|k| data_states[[idx + 1, k]]);
        for (k, value) in hermite_state(t, dt, &before, &after)
            .into_iter()
            .enumerate()
        {
            result[[out_idx, k]] = value;
        }
    }

    result
}

/// The cubic Hermite state at normalised parameter `t` in [0, 1] of an
/// interval `dt` seconds long, between the states `before` and `after`
/// (position xyz, velocity xyz).
#[inline]
fn hermite_state(t: f64, dt: f64, before: &[f64; 6], after: &[f64; 6]) -> [f64; 6] {
    // Hermite basis functions
    let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2); // position at t0
    let h10 = t * (1.0 - t).powi(2); // velocity at t0
    let h01 = t.powi(2) * (3.0 - 2.0 * t); // position at t1
    let h11 = t.powi(2) * (t - 1.0); // velocity at t1

    // Derivatives of the basis functions, for the velocity
    let dh00_dt = 6.0 * t * (t - 1.0) / dt;
    let dh10_dt = (1.0 - 4.0 * t + 3.0 * t.powi(2)) / dt;
    let dh01_dt = 6.0 * t * (1.0 - t) / dt;
    let dh11_dt = (3.0 * t.powi(2) - 2.0 * t) / dt;

    let mut state = [0.0; 6];
    for i in 0..3 {
        let (p0, p1, v0, v1) = (before[i], after[i], before[i + 3], after[i + 3]);
        state[i] = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;
        state[i + 3] = dh00_dt * p0 + dh10_dt * dt * v0 + dh01_dt * p1 + dh11_dt * dt * v1;
    }
    state
}

/// `hermite_interpolate` of one time strictly inside one bracketing pair of
/// samples, without the per-call tables: the same arithmetic, so the same
/// result to the last bit.
pub fn hermite_between(
    query_time: &DateTime<Utc>,
    time_before: &DateTime<Utc>,
    time_after: &DateTime<Utc>,
    before: &[f64; 6],
    after: &[f64; 6],
) -> [f64; 6] {
    // `hermite_interpolate` measures from the first sample: the interval
    // starts at 0.0 and has this length.
    let dt = diff_seconds(time_after, time_before);
    let t = diff_seconds(query_time, time_before) / dt;
    hermite_state(t, dt, before, after)
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
    fn hermite_between_is_hermite_interpolate_to_the_last_bit() {
        let start = Utc::now();
        let times = [start, start + Duration::microseconds(60_000_123)];
        let mut states = Array2::<f64>::zeros((2, 6));
        let rows = [
            [1.47e8, -2.3e7, 6.1e6, -29.7, 0.31, 7.5],
            [
                1.47e8 - 1700.0,
                -2.3e7 + 20.0,
                6.1e6 + 450.0,
                -29.6,
                0.33,
                7.4,
            ],
        ];
        for (i, row) in rows.iter().enumerate() {
            for (k, value) in row.iter().enumerate() {
                states[[i, k]] = *value;
            }
        }
        for micros in [1, 17, 12_345_678, 30_000_000, 59_999_999, 60_000_122] {
            let query = start + Duration::microseconds(micros);
            let expected = hermite_interpolate(&[query], &times, &states);
            let actual = hermite_between(&query, &times[0], &times[1], &rows[0], &rows[1]);
            for k in 0..6 {
                assert_eq!(actual[k].to_bits(), expected[[0, k]].to_bits());
            }
        }
    }

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
