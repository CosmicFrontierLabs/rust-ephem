/// Transpose a 3x3 matrix
#[inline]
pub fn transpose_matrix(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Compute the polar motion rotation matrix
///
/// This matrix transforms from the Celestial Intermediate Pole (CIP) frame
/// to the ITRS frame by applying polar motion corrections.
///
/// # Arguments
/// * `xp` - X-component of polar motion in radians
/// * `yp` - Y-component of polar motion in radians
///
/// # Returns
/// A 3x3 rotation matrix for polar motion correction
///
/// # Reference
/// IERS Conventions (2010), Section 5.4.1
#[inline]
pub fn polar_motion_matrix(xp: f64, yp: f64) -> [[f64; 3]; 3] {
    // For small angles (xp, yp typically < 1 arcsec = 4.85e-6 rad),
    // we can use the small angle approximation:
    // cos(x) ≈ 1, sin(x) ≈ x
    //
    // The polar motion matrix is:
    // W = Ry(-xp) * Rx(-yp)
    //
    // Where Ry and Rx are rotations about Y and X axes respectively.
    // For small angles, this simplifies to:
    [[1.0, 0.0, xp], [0.0, 1.0, -yp], [-xp, yp, 1.0]]
}
