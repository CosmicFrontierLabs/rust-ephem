/// Generic coordinate frame conversion functions.
///
/// This module provides reusable conversion functions for transforming between
/// different coordinate frames (TEME, ITRS, GCRS), shared by ephemeris classes.
use chrono::{DateTime, Utc};
use ndarray::Array2;
use sofars::{
    consts::D2PI,
    erst::{gmst06, gst06},
    pnp::{c2t06a, pnm06a},
    vm::{anp, rxp, rxr},
};
use std::f64::consts::PI;

use crate::utils::config::*;
use crate::utils::eop_provider::get_polar_motion_rad;
use crate::utils::math_utils::{polar_motion_matrix, transpose_matrix};
use crate::utils::time_utils::{datetime_to_jd_tt, datetime_to_jd_ut1};

fn norm_angle_pm(angle: f64) -> f64 {
    // Normalize to [-pi, pi) to preserve small signed offsets across 2pi wrap.
    let mut w = anp(angle);
    if w >= PI {
        w -= D2PI;
    }
    w
}

fn teme_gcrs_matrix(dt: &DateTime<Utc>) -> [[f64; 3]; 3] {
    let (jd_tt1, jd_tt2) = datetime_to_jd_tt(dt);
    let bpn = pnm06a(jd_tt1, jd_tt2);
    let (jd_ut1_1, jd_ut1_2) = datetime_to_jd_ut1(dt);
    let gast = gst06(jd_ut1_1, jd_ut1_2, jd_tt1, jd_tt2, &bpn);
    let gmst = gmst06(jd_ut1_1, jd_ut1_2, jd_tt1, jd_tt2);
    let eqeq = norm_angle_pm(gast - gmst);
    // TEME uses the mean equinox; rotate from true equinox via equation of equinoxes.
    let (sin_eq, cos_eq) = eqeq.sin_cos();
    let eqeq_rot = [
        [cos_eq, sin_eq, 0.0],
        [-sin_eq, cos_eq, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut result = [[0.0; 3]; 3];
    rxr(&eqeq_rot, &bpn, &mut result);
    result
}

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
    /// Time-dependent GCRS -> ITRS matrix and its derivative per second.
    MatrixWithRate {
        matrix: [[f64; 3]; 3],
        rate: [[f64; 3]; 3],
    },
    /// 2D rotation about Z-axis (for GMST/ERA)
    RotationZ {
        cos_angle: f64,
        sin_angle: f64,
        /// Whether to apply Earth rotation velocity correction (for ITRS conversions)
        earth_rotation: bool,
    },
    /// Composed rotation: sidereal rotation + polar motion (for TEME <-> ITRS).
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
            Rotation::MatrixWithRate { matrix, rate } => {
                let mut new_pos = [0.0; 3];
                let mut new_vel = [0.0; 3];
                let mut frame_velocity = [0.0; 3];
                if inverse {
                    // r_c = M^T r_t; v_c = M^T (v_t - Mdot r_c).
                    let transpose = transpose_matrix(*matrix);
                    rxp(&transpose, &pos, &mut new_pos);
                    rxp(rate, &new_pos, &mut frame_velocity);
                    let corrected = std::array::from_fn(|i| vel[i] - frame_velocity[i]);
                    rxp(&transpose, &corrected, &mut new_vel);
                } else {
                    // r_t = M r_c; v_t = M v_c + Mdot r_c.
                    rxp(matrix, &pos, &mut new_pos);
                    rxp(matrix, &vel, &mut new_vel);
                    rxp(rate, &pos, &mut frame_velocity);
                    for i in 0..3 {
                        new_vel[i] += frame_velocity[i];
                    }
                }
                (new_pos, new_vel)
            }
            Rotation::Matrix3x3 { matrix } => {
                let mat = if inverse {
                    transpose_matrix(*matrix)
                } else {
                    *matrix
                };
                let mut new_pos = [0.0; 3];
                rxp(&mat, &pos, &mut new_pos);
                let mut new_vel = [0.0; 3];
                rxp(&mat, &vel, &mut new_vel);
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
                // For TEME -> ITRS: sidereal rotation, then polar motion.
                // The inverse reverses this order.
                if inverse {
                    // ITRS -> TEME: R_z(-GMST) * W^T
                    // Step 1: Apply inverse polar motion (transpose)
                    let pm_t = transpose_matrix(*polar_motion);
                    let mut pos1 = [0.0; 3];
                    rxp(&pm_t, &pos, &mut pos1);
                    let mut vel1 = [0.0; 3];
                    rxp(&pm_t, &vel, &mut vel1);

                    // Step 2: Apply inverse sidereal rotation and spin velocity.
                    let (c, s) = (*cos_era, *sin_era);
                    let (x, y, z) = (pos1[0], pos1[1], pos1[2]);
                    let (vx, vy, vz) = (vel1[0], vel1[1], vel1[2]);

                    let nx = c * x - s * y;
                    let ny = s * x + c * y;
                    let nvx = c * vx - s * vy - OMEGA_EARTH * ny;
                    let nvy = s * vx + c * vy + OMEGA_EARTH * nx;

                    ([nx, ny, z], [nvx, nvy, vz])
                } else {
                    // TEME -> ITRS: W * R_z(GMST)
                    // Step 1: Apply sidereal rotation and spin velocity.
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
                    let mut new_pos = [0.0; 3];
                    rxp(polar_motion, &pos1, &mut new_pos);
                    let mut new_vel = [0.0; 3];
                    rxp(polar_motion, &vel1, &mut new_vel);

                    (new_pos, new_vel)
                }
            }
        }
    }
}

/// IAU 2006/2000A celestial-to-terrestrial rotation: W * R3(ERA) * Q.
/// Differentiate over one second, including precession/nutation and Earth spin.
/// EOP values are held fixed locally (no polar-motion rates or LOD correction).
fn gcrs_itrs_rotation(tt: (f64, f64), ut1: (f64, f64), xp: f64, yp: f64) -> Rotation {
    // Put offsets into fractional days, avoiding cancellation when adding small
    // time steps to a large Julian/MJD day number. No UTC/leap-second differencing.
    let tt = (tt.0 + tt.1.floor(), tt.1 - tt.1.floor());
    let ut1 = (ut1.0 + ut1.1.floor(), ut1.1 - ut1.1.floor());
    let matrix_at = |seconds: f64| {
        let days = seconds / SECONDS_PER_DAY;
        c2t06a(tt.0, tt.1 + days, ut1.0, ut1.1 + days, xp, yp)
    };
    let matrix = matrix_at(0.0);
    let before = matrix_at(-0.5);
    let after = matrix_at(0.5);
    let rate = std::array::from_fn(|i| std::array::from_fn(|j| after[i][j] - before[i][j]));
    Rotation::MatrixWithRate { matrix, rate }
}

/// Get the rotation transformation for a specific frame conversion at a given time.
fn get_rotation(from: Frame, to: Frame, dt: &DateTime<Utc>, polar_motion: bool) -> Rotation {
    match (from, to) {
        // Precession-nutation transformation (TEME <-> GCRS)
        (Frame::TEME, Frame::GCRS) | (Frame::GCRS, Frame::TEME) => {
            let matrix = teme_gcrs_matrix(dt);
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
        // Full celestial-to-terrestrial rotation; only polar motion is optional.
        (Frame::GCRS, Frame::ITRS) | (Frame::ITRS, Frame::GCRS) => {
            let (xp, yp) = if polar_motion {
                get_polar_motion_rad(dt)
            } else {
                (0.0, 0.0)
            };
            gcrs_itrs_rotation(datetime_to_jd_tt(dt), datetime_to_jd_ut1(dt), xp, yp)
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
    // TEME->ITRS and GCRS->ITRS are forward; their inverses start in ITRS.
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_gcrs_itrs_reference_and_round_trip() {
        // Independent ERFA c2t06a reference, TT = UT1 = MJD 53736,
        // xp = 2.55060238e-7, yp = 1.860359247e-6 radians. Reference velocities
        // use a fourth-order, 2-second finite difference (not our stencil).
        let pos = [7000.0, -1200.0, 1800.0];
        let vel = [1.2, 6.8, -2.4];
        let expected_pos = [-2447.286866729401, -6666.06286530538, 1803.9935886056828];
        let expected_vel = [5.984149542054486, -2.2341190029645612, -2.3990368739985186];
        // Equivalent date splits, including a negative second part.
        for date in [(2400000.5, 53736.0), (2453736.5, 0.0), (2453737.5, -1.0)] {
            let rotation = gcrs_itrs_rotation(date, date, 2.55060238e-7, 1.860359247e-6);
            let (itrs_pos, itrs_vel) = rotation.apply(pos, vel, false);
            let (restored_pos, restored_vel) = rotation.apply(itrs_pos, itrs_vel, true);
            for i in 0..3 {
                assert!((itrs_pos[i] - expected_pos[i]).abs() < 1e-9);
                assert!((itrs_vel[i] - expected_vel[i]).abs() < 1e-9);
                assert!((restored_pos[i] - pos[i]).abs() < 1e-9);
                assert!((restored_vel[i] - vel[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_teme_to_gcrs_includes_eqeq() {
        let dt = Utc.with_ymd_and_hms(2025, 10, 14, 0, 0, 0).unwrap();
        let matrix = teme_gcrs_matrix(&dt);
        let (jd_tt1, jd_tt2) = datetime_to_jd_tt(&dt);
        let bpn = pnm06a(jd_tt1, jd_tt2);

        let mut max_diff = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                max_diff = max_diff.max((matrix[i][j] - bpn[i][j]).abs());
            }
        }
        assert!(max_diff > 1e-7, "eqeq too small for reliable test");

        let expected_matrix = transpose_matrix(matrix);

        let input = Array2::from_shape_vec((1, 6), vec![7000.0, 1000.0, -2000.0, 1.0, -2.0, 0.5])
            .expect("input array");
        let output = convert_frames(&input, &[dt], Frame::TEME, Frame::GCRS, false);

        let mut expected_pos = [0.0; 3];
        rxp(
            &expected_matrix,
            &[7000.0, 1000.0, -2000.0],
            &mut expected_pos,
        );
        let mut expected_vel = [0.0; 3];
        rxp(&expected_matrix, &[1.0, -2.0, 0.5], &mut expected_vel);

        for i in 0..3 {
            assert!((output[[0, i]] - expected_pos[i]).abs() < 1e-9);
            assert!((output[[0, i + 3]] - expected_vel[i]).abs() < 1e-9);
        }
    }
}
