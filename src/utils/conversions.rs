/// Generic coordinate frame conversion functions.
///
/// This module provides reusable conversion functions for transforming between
/// different coordinate frames (TEME, ITRS, GCRS). These functions are used by
/// both TLEEphemeris and SPICEEphemeris to avoid code duplication.
use chrono::{DateTime, Utc};
use erfa::{
    earth::earth_rotation_angle_00, prenut::pn_matrix_06a, vectors_and_matrices::mat_mul_pvec,
};
use ndarray::Array2;

use crate::utils::config::*;
use crate::utils::eop_provider::get_polar_motion_rad;
use crate::utils::math_utils::{polar_motion_matrix, transpose_matrix};
use crate::utils::time_utils::{datetime_to_jd, datetime_to_jd_ut1, get_tt_offset_days};

/// Supported coordinate frames for conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Frame {
    TEME,
    GCRS,
    ITRS,
}

/// Represents a rotation transformation between two frames.
enum Rotation {
    /// 3x3 matrix rotation (for precession-nutation)
    Matrix3x3 { matrix: [[f64; 3]; 3] },
    /// 2D rotation about Z-axis (for GMST/ERA)
    RotationZ {
        cos_angle: f64,
        sin_angle: f64,
        /// Whether to apply Earth rotation velocity correction (for ITRS conversions)
        earth_rotation: bool,
    },
    /// Composed rotation: ERA + polar motion (for GCRS <-> ITRS with polar motion correction)
    EraWithPolarMotion {
        cos_era: f64,
        sin_era: f64,
        polar_motion: [[f64; 3]; 3],
    },
}

impl Rotation {
    /// Apply rotation to position and velocity vectors.
    /// `inverse`: if true, apply the inverse rotation (transpose for orthogonal matrices).
    fn apply(&self, pos: [f64; 3], vel: [f64; 3], inverse: bool) -> ([f64; 3], [f64; 3]) {
        match self {
            Rotation::Matrix3x3 { matrix } => {
                let mat = if inverse {
                    transpose_matrix(*matrix)
                } else {
                    *matrix
                };
                let new_pos = mat_mul_pvec(mat, pos);
                let new_vel = mat_mul_pvec(mat, vel);
                (new_pos, new_vel)
            }
            Rotation::RotationZ {
                cos_angle,
                sin_angle,
                earth_rotation,
            } => {
                let (c, s) = (*cos_angle, *sin_angle);
                let (x, y, z) = (pos[0], pos[1], pos[2]);
                let (vx, vy, vz) = (vel[0], vel[1], vel[2]);

                let (new_x, new_y, new_vx, new_vy) = if inverse {
                    // Inverse rotation: transpose the rotation matrix
                    let nx = c * x - s * y;
                    let ny = s * x + c * y;
                    let nvx = if *earth_rotation {
                        c * vx - s * vy - OMEGA_EARTH * ny
                    } else {
                        c * vx - s * vy
                    };
                    let nvy = if *earth_rotation {
                        s * vx + c * vy + OMEGA_EARTH * nx
                    } else {
                        s * vx + c * vy
                    };
                    (nx, ny, nvx, nvy)
                } else {
                    // Forward rotation
                    let nx = c * x + s * y;
                    let ny = -s * x + c * y;
                    let nvx = if *earth_rotation {
                        c * vx + s * vy + OMEGA_EARTH * ny
                    } else {
                        c * vx + s * vy
                    };
                    let nvy = if *earth_rotation {
                        -s * vx + c * vy - OMEGA_EARTH * nx
                    } else {
                        -s * vx + c * vy
                    };
                    (nx, ny, nvx, nvy)
                };

                ([new_x, new_y, z], [new_vx, new_vy, vz])
            }
            Rotation::EraWithPolarMotion {
                cos_era,
                sin_era,
                polar_motion,
            } => {
                // For GCRS -> ITRS: First apply ERA rotation, then polar motion
                // For ITRS -> GCRS: First apply inverse polar motion, then inverse ERA
                if inverse {
                    // ITRS -> GCRS: W^T * R_z(-ERA)
                    // Step 1: Apply inverse polar motion (transpose)
                    let pm_t = transpose_matrix(*polar_motion);
                    let pos1 = mat_mul_pvec(pm_t, pos);
                    let vel1 = mat_mul_pvec(pm_t, vel);

                    // Step 2: Apply inverse ERA (with Earth rotation velocity)
                    let (c, s) = (*cos_era, *sin_era);
                    let (x, y, z) = (pos1[0], pos1[1], pos1[2]);
                    let (vx, vy, vz) = (vel1[0], vel1[1], vel1[2]);

                    let nx = c * x - s * y;
                    let ny = s * x + c * y;
                    let nvx = c * vx - s * vy - OMEGA_EARTH * ny;
                    let nvy = s * vx + c * vy + OMEGA_EARTH * nx;

                    ([nx, ny, z], [nvx, nvy, vz])
                } else {
                    // GCRS -> ITRS: R_z(ERA) * W
                    // Step 1: Apply ERA rotation (with Earth rotation velocity)
                    let (c, s) = (*cos_era, *sin_era);
                    let (x, y, z) = (pos[0], pos[1], pos[2]);
                    let (vx, vy, vz) = (vel[0], vel[1], vel[2]);

                    let x1 = c * x + s * y;
                    let y1 = -s * x + c * y;
                    let vx1 = c * vx + s * vy + OMEGA_EARTH * y1;
                    let vy1 = -s * vx + c * vy - OMEGA_EARTH * x1;

                    // Step 2: Apply polar motion
                    let pos1 = [x1, y1, z];
                    let vel1 = [vx1, vy1, vz];
                    let new_pos = mat_mul_pvec(*polar_motion, pos1);
                    let new_vel = mat_mul_pvec(*polar_motion, vel1);

                    (new_pos, new_vel)
                }
            }
        }
    }
}

/// Get the rotation transformation for a specific frame conversion at a given time.
fn get_rotation(from: Frame, to: Frame, dt: &DateTime<Utc>, polar_motion: bool) -> Rotation {
    match (from, to) {
        // Precession-nutation transformation (TEME <-> GCRS)
        (Frame::TEME, Frame::GCRS) | (Frame::GCRS, Frame::TEME) => {
            let (jd_utc1, jd_utc2) = datetime_to_jd(dt);
            let jd_tt1 = jd_utc1;
            let jd_tt2 = jd_utc2 + get_tt_offset_days(dt);
            let matrix = pn_matrix_06a(jd_tt1, jd_tt2);
            Rotation::Matrix3x3 { matrix }
        }
        // GMST rotation (TEME <-> ITRS)
        (Frame::TEME, Frame::ITRS) | (Frame::ITRS, Frame::TEME) => {
            // Use UT1 time scale for Earth rotation
            let (jd_ut1_1, jd_ut1_2) = datetime_to_jd_ut1(dt);
            let jd_ut1 = jd_ut1_1 + jd_ut1_2;
            let t_ut1 = (jd_ut1 - JD_J2000) / DAYS_PER_CENTURY;
            let t_ut1_sq = t_ut1 * t_ut1;
            let t_ut1_cb = t_ut1_sq * t_ut1;
            let gmst_sec = GMST_COEFF_0
                + GMST_COEFF_1 * t_ut1
                + GMST_COEFF_2 * t_ut1_sq
                + GMST_COEFF_3 * t_ut1_cb;
            let gmst_rad = (gmst_sec % SECS_PER_DAY) * PI_OVER_43200;

            if polar_motion {
                // Apply polar motion correction for TEME↔ITRS transformation
                let (xp, yp) = get_polar_motion_rad(dt);
                let pm_matrix = polar_motion_matrix(xp, yp);

                Rotation::EraWithPolarMotion {
                    cos_era: gmst_rad.cos(),
                    sin_era: gmst_rad.sin(),
                    polar_motion: pm_matrix,
                }
            } else {
                // Simple rotation without polar motion
                Rotation::RotationZ {
                    cos_angle: gmst_rad.cos(),
                    sin_angle: gmst_rad.sin(),
                    earth_rotation: true,
                }
            }
        }
        // ERA rotation (GCRS <-> ITRS) - with optional polar motion correction
        (Frame::GCRS, Frame::ITRS) | (Frame::ITRS, Frame::GCRS) => {
            // Use UT1 time scale for Earth rotation angle
            let (jd_ut1_1, jd_ut1_2) = datetime_to_jd_ut1(dt);
            let era = earth_rotation_angle_00(jd_ut1_1, jd_ut1_2);

            if polar_motion {
                // Get polar motion parameters (xp, yp in radians)
                // If EOP data is not available, this will return (0.0, 0.0) as fallback
                let (xp, yp) = get_polar_motion_rad(dt);
                let pm_matrix = polar_motion_matrix(xp, yp);

                Rotation::EraWithPolarMotion {
                    cos_era: era.cos(),
                    sin_era: era.sin(),
                    polar_motion: pm_matrix,
                }
            } else {
                // Simple rotation without polar motion correction
                Rotation::RotationZ {
                    cos_angle: era.cos(),
                    sin_angle: era.sin(),
                    earth_rotation: true,
                }
            }
        }
        _ => unreachable!("Invalid frame combination"),
    }
}

/// Generic frame conversion function.
///
/// Converts `data` (Nx6 array of [x,y,z,vx,vy,vz]) from `input_frame` to `output_frame`
/// for the timestamps `times`.
///
/// Supports all conversions between TEME, GCRS, and ITRS frames.
/// Uses generic rotation mathematics that automatically handles forward and inverse transformations.
///
/// # Arguments
/// * `data` - Nx6 array of position and velocity [x,y,z,vx,vy,vz]
/// * `times` - Array of timestamps
/// * `input_frame` - Input coordinate frame
/// * `output_frame` - Output coordinate frame
/// * `polar_motion` - Whether to apply polar motion correction (default: false for backward compatibility)
pub fn convert_frames(
    data: &Array2<f64>,
    times: &[DateTime<Utc>],
    input_frame: Frame,
    output_frame: Frame,
    polar_motion: bool,
) -> Array2<f64> {
    // Fast path: same-frame -> return a copy
    if input_frame == output_frame {
        return data.to_owned();
    }

    let n = times.len();
    let mut out = Array2::<f64>::zeros((n, 6));

    // Determine if we need the inverse transformation
    // TEME->GCRS uses transpose (inverse) of pn_matrix
    // GCRS->TEME uses forward pn_matrix
    // For Z-rotations: TEME->ITRS and GCRS->ITRS are forward, inverses are ITRS->TEME and ITRS->GCRS
    let needs_inverse = matches!(
        (input_frame, output_frame),
        (Frame::TEME, Frame::GCRS) | (Frame::ITRS, Frame::TEME) | (Frame::ITRS, Frame::GCRS)
    );

    for (i, dt) in times.iter().enumerate() {
        // Get the base rotation (always defined in the "forward" direction)
        let rotation = get_rotation(input_frame, output_frame, dt, polar_motion);

        let in_row = data.row(i);
        let pos = [in_row[0], in_row[1], in_row[2]];
        let vel = [in_row[3], in_row[4], in_row[5]];

        let (new_pos, new_vel) = rotation.apply(pos, vel, needs_inverse);

        let mut out_row = out.row_mut(i);
        out_row[0] = new_pos[0];
        out_row[1] = new_pos[1];
        out_row[2] = new_pos[2];
        out_row[3] = new_vel[0];
        out_row[4] = new_vel[1];
        out_row[5] = new_vel[2];
    }

    out
}
