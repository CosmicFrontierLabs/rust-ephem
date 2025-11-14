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
