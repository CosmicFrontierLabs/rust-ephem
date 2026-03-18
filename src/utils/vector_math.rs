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

/// Convert multiple unit vectors to RA/Dec coordinates (vectorized)
///
/// # Arguments
/// * `unit_vectors` - Array2 with shape (N, 3) containing [x, y, z] rows
///
/// # Returns
/// Tuple of `(ras_deg, decs_deg)` vectors, each length N
pub fn unit_vectors_to_radec_batch(unit_vectors: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        unit_vectors.ncols(),
        3,
        "unit_vectors must have shape (N, 3)"
    );

    let n = unit_vectors.nrows();
    if let Some(slice) = unit_vectors.as_slice_memory_order() {
        let mut ras = Vec::with_capacity(n);
        let mut decs = Vec::with_capacity(n);
        for xyz in slice.chunks_exact(3) {
            let x = xyz[0];
            let y = xyz[1];
            let z = xyz[2].clamp(-1.0, 1.0);

            let mut ra_deg = y.atan2(x).to_degrees();
            if ra_deg < 0.0 {
                ra_deg += 360.0;
            }
            let dec_deg = z.asin().to_degrees();

            ras.push(ra_deg);
            decs.push(dec_deg);
        }

        (ras, decs)
    } else {
        let mut ras = Vec::with_capacity(n);
        let mut decs = Vec::with_capacity(n);
        for i in 0..n {
            let x = unit_vectors[[i, 0]];
            let y = unit_vectors[[i, 1]];
            let z = unit_vectors[[i, 2]].clamp(-1.0, 1.0);

            let mut ra_deg = y.atan2(x).to_degrees();
            if ra_deg < 0.0 {
                ra_deg += 360.0;
            }
            let dec_deg = z.asin().to_degrees();

            ras.push(ra_deg);
            decs.push(dec_deg);
        }

        (ras, decs)
    }
}

/// Build a 3x3 rotation matrix from intrinsic Z-Y-X Euler angles in degrees.
///
/// Rotation order is yaw (Z), pitch (Y), roll (X):
/// `R = Rz(yaw) * Ry(pitch) * Rx(roll)`.
#[allow(dead_code)]
pub fn euler_zyx_rotation_matrix(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [[f64; 3]; 3] {
    let roll = roll_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let yaw = yaw_deg.to_radians();

    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// Rotate an RA/Dec direction using a precomputed 3x3 rotation matrix.
///
/// # Arguments
/// * `ra_deg` - Input right ascension in degrees
/// * `dec_deg` - Input declination in degrees
/// * `rotation_matrix` - 3x3 rotation matrix
///
/// # Returns
/// Rotated `(ra_deg, dec_deg)` in degrees.
#[allow(dead_code)]
pub fn rotate_radec_with_matrix(
    ra_deg: f64,
    dec_deg: f64,
    rotation_matrix: &[[f64; 3]; 3],
) -> (f64, f64) {
    let v = radec_to_unit_vector(ra_deg, dec_deg);

    let x =
        rotation_matrix[0][0] * v[0] + rotation_matrix[0][1] * v[1] + rotation_matrix[0][2] * v[2];
    let y =
        rotation_matrix[1][0] * v[0] + rotation_matrix[1][1] * v[1] + rotation_matrix[1][2] * v[2];
    let z =
        rotation_matrix[2][0] * v[0] + rotation_matrix[2][1] * v[1] + rotation_matrix[2][2] * v[2];

    let dec_rot = z.clamp(-1.0, 1.0).asin().to_degrees();
    let mut ra_rot = y.atan2(x).to_degrees();
    if ra_rot < 0.0 {
        ra_rot += 360.0;
    }

    (ra_rot, dec_rot)
}
