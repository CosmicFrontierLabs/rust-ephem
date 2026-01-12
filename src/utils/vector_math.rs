/// Vector math utilities for constraint calculations
///
/// This module provides helper functions for vector operations used in
/// astronomical constraint calculations, including coordinate conversions,
/// vector normalization, and angular separation calculations.
/// Convert RA/Dec coordinates to a unit vector
///
/// # Arguments
/// * `ra_deg` - Right ascension in degrees
/// * `dec_deg` - Declination in degrees
///
/// # Returns
/// Unit vector [x, y, z] in ICRS/J2000 frame
#[allow(dead_code)]
pub fn radec_to_unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra_rad = ra_deg.to_radians();
    let dec_rad = dec_deg.to_radians();
    let cos_dec = dec_rad.cos();
    [
        cos_dec * ra_rad.cos(),
        cos_dec * ra_rad.sin(),
        dec_rad.sin(),
    ]
}

/// Normalize a 3D vector to unit length
///
/// # Arguments
/// * `v` - Input vector [x, y, z]
///
/// # Returns
/// Normalized unit vector, or [0, 0, 0] if input magnitude is zero
pub fn normalize_vector(v: &[f64; 3]) -> [f64; 3] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag, v[2] / mag]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Calculate the dot product of two 3D vectors
///
/// # Arguments
/// * `a` - First vector [x, y, z]
/// * `b` - Second vector [x, y, z]
///
/// # Returns
/// Scalar dot product a·b
pub fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Calculate the magnitude (length) of a 3D vector
///
/// # Arguments
/// * `v` - Input vector [x, y, z]
///
/// # Returns
/// Magnitude (length) of the vector
pub fn vector_magnitude(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Calculate angular separation between a target direction and a body
///
/// Computes the angular separation between a target direction (specified as a unit vector)
/// and the direction from an observer to a celestial body.
///
/// # Arguments
/// * `target_vec` - Unit vector pointing to the target (ICRS/J2000)
/// * `body_position` - Position of the body in km (GCRS)
/// * `observer_position` - Position of the observer in km (GCRS)
///
/// # Returns
/// Angular separation in degrees
#[allow(dead_code)]
pub fn calculate_angular_separation(
    target_vec: &[f64; 3],
    body_position: &[f64; 3],
    observer_position: &[f64; 3],
) -> f64 {
    let body_rel = [
        body_position[0] - observer_position[0],
        body_position[1] - observer_position[1],
        body_position[2] - observer_position[2],
    ];
    let body_unit = normalize_vector(&body_rel);
    let cos_angle = dot_product(target_vec, &body_unit);
    cos_angle.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Calculate cosine of angular separation between a target and a body (optimized)
///
/// This is an optimized alternative to `calculate_angular_separation()` that avoids
/// the expensive `acos()` call. Returns the cosine of the angle instead of the angle
/// itself, which is suitable for threshold comparisons.
///
/// Uses the mathematical property: angle < threshold ⟺ cos(angle) > cos(threshold)
/// (since cosine is decreasing on [0, π])
///
/// # Arguments
/// * `target_vec` - Unit vector pointing to the target (ICRS/J2000)
/// * `body_position` - Position of the body in km (GCRS)
/// * `observer_position` - Position of the observer in km (GCRS)
///
/// # Returns
/// Cosine of the angular separation (in range [-1, 1])
pub fn calculate_cosine_separation(
    target_vec: &[f64; 3],
    body_position: &[f64; 3],
    observer_position: &[f64; 3],
) -> f64 {
    let body_rel = [
        body_position[0] - observer_position[0],
        body_position[1] - observer_position[1],
        body_position[2] - observer_position[2],
    ];
    let body_unit = normalize_vector(&body_rel);
    dot_product(target_vec, &body_unit)
}

// ============================================================================
// Vectorized batch operations for performance
// ============================================================================

use ndarray::Array2;

/// Convert multiple RA/Dec coordinates to unit vectors (vectorized)
///
/// # Arguments
/// * `ras_deg` - Array of right ascensions in degrees
/// * `decs_deg` - Array of declinations in degrees
///
/// # Returns
/// Array2 with shape (N, 3) containing unit vectors [x, y, z] in ICRS/J2000 frame
///
/// # Performance
/// This vectorized implementation is significantly faster than calling
/// radec_to_unit_vector() in a loop for large numbers of targets.
pub fn radec_to_unit_vectors_batch(ras_deg: &[f64], decs_deg: &[f64]) -> Array2<f64> {
    assert_eq!(
        ras_deg.len(),
        decs_deg.len(),
        "RA and Dec arrays must have same length"
    );

    let n = ras_deg.len();
    let mut result = Array2::<f64>::zeros((n, 3));

    for (i, (&ra, &dec)) in ras_deg.iter().zip(decs_deg.iter()).enumerate() {
        let ra_rad = ra.to_radians();
        let dec_rad = dec.to_radians();
        let cos_dec = dec_rad.cos();

        result[[i, 0]] = cos_dec * ra_rad.cos();
        result[[i, 1]] = cos_dec * ra_rad.sin();
        result[[i, 2]] = dec_rad.sin();
    }

    result
}

/// Calculate angular separations between target vectors and body positions (vectorized batch)
///
/// This function computes angular separations for multiple targets and times simultaneously,
/// providing significant performance improvements over individual calculations.
///
/// # Arguments
/// * `target_vectors` - Array of target unit vectors (N_targets x 3)
/// * `body_positions` - Array of body positions relative to observer (N_times x 3)
///
/// # Returns
/// Array2<f64> with shape (N_targets, N_times) containing angular separations in degrees
///
/// # Performance
/// Vectorized implementation avoids loops in Python and provides maximum performance.
pub fn calculate_angular_separations_batch(
    target_vectors: &Array2<f64>,
    body_positions: &Array2<f64>,
) -> Array2<f64> {
    let n_targets = target_vectors.nrows();
    let n_times = body_positions.nrows();
    let mut result = Array2::<f64>::zeros((n_targets, n_times));

    for i in 0..n_targets {
        let target_vec = [
            target_vectors[[i, 0]],
            target_vectors[[i, 1]],
            target_vectors[[i, 2]],
        ];

        for j in 0..n_times {
            let body_pos = [
                body_positions[[j, 0]],
                body_positions[[j, 1]],
                body_positions[[j, 2]],
            ];

            // Calculate relative position vector from observer to body
            let body_rel = body_pos;
            let body_unit = normalize_vector(&body_rel);

            // Calculate angular separation
            let cos_angle = dot_product(&target_vec, &body_unit);
            let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

            result[[i, j]] = angle_deg;
        }
    }

    result
}
