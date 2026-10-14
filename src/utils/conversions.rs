/// Generic coordinate frame conversion functions.
///
/// This module provides reusable conversion functions for transforming between
/// different coordinate frames (TEME, ITRS, GCRS), shared by ephemeris classes.
use chrono::{DateTime, Utc};
use ndarray::Array2;
use sofars::{
    erst::{era00, gmst82},
    pnp::{c2i06a, c2tcio, pom00, sp00},
    vm::rxp,
};

use crate::utils::config::SECONDS_PER_DAY;
use crate::utils::eop_provider::get_polar_motion_rad;
use crate::utils::math_utils::transpose_matrix;
use crate::utils::time_utils::{datetime_to_jd_tt, datetime_to_jd_ut1};

/// Supported coordinate frames for conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Frame {
    TEME,
    GCRS,
    ITRS,
}

/// A forward frame rotation and its time derivative per second.
struct Rotation {
    matrix: [[f64; 3]; 3],
    rate: [[f64; 3]; 3],
}

impl Rotation {
    /// Transform position and velocity, including motion of the reference frame.
    fn apply(&self, pos: [f64; 3], vel: [f64; 3], inverse: bool) -> ([f64; 3], [f64; 3]) {
        let mut new_pos = [0.0; 3];
        let mut new_vel = [0.0; 3];
        let mut frame_velocity = [0.0; 3];
        if inverse {
            // r_in = M^T r_out; v_in = M^T (v_out - Mdot r_in).
            let transpose = transpose_matrix(self.matrix);
            rxp(&transpose, &pos, &mut new_pos);
            rxp(&self.rate, &new_pos, &mut frame_velocity);
            let corrected = std::array::from_fn(|i| vel[i] - frame_velocity[i]);
            rxp(&transpose, &corrected, &mut new_vel);
        } else {
            rxp(&self.matrix, &pos, &mut new_pos);
            rxp(&self.matrix, &vel, &mut new_vel);
            rxp(&self.rate, &pos, &mut frame_velocity);
            for i in 0..3 {
                new_vel[i] += frame_velocity[i];
            }
        }
        (new_pos, new_vel)
    }
}

/// Differentiate over one second of TT/UT1, with EOP held fixed locally.
fn rotation_with_rate(mut matrix_at: impl FnMut(f64) -> [[f64; 3]; 3]) -> Rotation {
    let matrix = matrix_at(0.0);
    let before = matrix_at(-0.5);
    let after = matrix_at(0.5);
    let rate = std::array::from_fn(|i| std::array::from_fn(|j| after[i][j] - before[i][j]));
    Rotation { matrix, rate }
}

fn split_day(date: (f64, f64)) -> (f64, f64) {
    (date.0 + date.1.floor(), date.1 - date.1.floor())
}

/// Vallado TEME -> ITRS: IAU 1982 mean sidereal time, then polar motion.
/// TEME has no TIO locator s'; use an orthogonal polar-motion matrix.
fn teme_itrs_rotation(ut1: (f64, f64), xp: f64, yp: f64) -> Rotation {
    let ut1 = split_day(ut1);
    let polar = pom00(xp, yp, 0.0);
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    rotation_with_rate(|seconds| {
        let gmst = gmst82(ut1.0, ut1.1 + seconds / SECONDS_PER_DAY);
        c2tcio(&identity, gmst, &polar)
    })
}

const CELESTIAL_STEP_SECONDS: f64 = 300.0;
type CelestialNode = (f64, i64, [[f64; 3]; 3]);

/// Batch-local, four-node cubic interpolation of the slowly varying Q matrix.
/// The fixed five-minute TT grid is independent of input order and cadence.
/// Only Q is cached: UT1, Earth spin and polar motion remain sample-specific.
#[derive(Default)]
struct CelestialMatrixCache {
    nodes: [Option<CelestialNode>; 4],
}

impl CelestialMatrixCache {
    fn at(&mut self, tt: (f64, f64)) -> [[f64; 3]; 3] {
        let index = (tt.1 * SECONDS_PER_DAY / CELESTIAL_STEP_SECONDS).floor() as i64;
        let x = (tt.1 * SECONDS_PER_DAY - index as f64 * CELESTIAL_STEP_SECONDS)
            / CELESTIAL_STEP_SECONDS;
        let weights = [
            -x * (x - 1.0) * (x - 2.0) / 6.0,
            (x + 1.0) * (x - 1.0) * (x - 2.0) / 2.0,
            -(x + 1.0) * x * (x - 2.0) / 2.0,
            (x + 1.0) * x * (x - 1.0) / 6.0,
        ];
        let matrices: [_; 4] = std::array::from_fn(|k| {
            let node = index + k as i64 - 1;
            let slot = &mut self.nodes[node.rem_euclid(4) as usize];
            if !matches!(slot, Some((day, index, _)) if *day == tt.0 && *index == node) {
                *slot = Some((
                    tt.0,
                    node,
                    c2i06a(tt.0, node as f64 * CELESTIAL_STEP_SECONDS / SECONDS_PER_DAY),
                ));
            }
            slot.as_ref().unwrap().2
        });
        // Interpolate differences so constant terms (especially the diagonal)
        // do not acquire rounding noise from summing the Lagrange weights.
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                matrices[1][i][j]
                    + (0..4)
                        .map(|k| weights[k] * (matrices[k][i][j] - matrices[1][i][j]))
                        .sum::<f64>()
            })
        })
    }
}

/// Rotate GCRS to ITRS or TEME using the same celestial matrix.
///
/// ITRS = W(xp, yp, s') * R3(ERA) * Q * GCRS.
/// TEME = R3(-GMST82) * W(xp, yp, 0)^T * ITRS, so polar motion cancels:
/// TEME = R3(s') * R3(ERA - GMST82) * Q * GCRS.
/// This retains the Vallado TEME convention instead of substituting GMST06.
fn gcrs_rotation(
    tt: (f64, f64),
    ut1: (f64, f64),
    xp: f64,
    yp: f64,
    to: Frame,
    mut cache: Option<&mut CelestialMatrixCache>,
) -> Rotation {
    // Add small offsets to fractional days rather than large Julian dates.
    let tt = split_day(tt);
    let ut1 = split_day(ut1);
    rotation_with_rate(|seconds| {
        let days = seconds / SECONDS_PER_DAY;
        let q = match cache.as_mut() {
            Some(cache) => cache.at((tt.0, tt.1 + days)),
            None => c2i06a(tt.0, tt.1 + days),
        };
        let era = era00(ut1.0, ut1.1 + days);
        let sp = sp00(tt.0, tt.1 + days);
        let (angle, polar) = match to {
            Frame::ITRS => (era, pom00(xp, yp, sp)),
            Frame::TEME => (era - gmst82(ut1.0, ut1.1 + days), pom00(0.0, 0.0, sp)),
            Frame::GCRS => unreachable!("GCRS is already the input frame"),
        };
        c2tcio(&q, angle, &polar)
    })
}

/// Get the forward rotation for either direction of a frame pair.
fn get_rotation(
    from: Frame,
    to: Frame,
    dt: &DateTime<Utc>,
    polar_motion: bool,
    cache: Option<&mut CelestialMatrixCache>,
) -> Rotation {
    let ut1 = datetime_to_jd_ut1(dt);
    let (xp, yp) = if polar_motion && (from == Frame::ITRS || to == Frame::ITRS) {
        get_polar_motion_rad(dt)
    } else {
        (0.0, 0.0)
    };
    match (from, to) {
        (Frame::TEME, Frame::GCRS) | (Frame::GCRS, Frame::TEME) => {
            gcrs_rotation(datetime_to_jd_tt(dt), ut1, 0.0, 0.0, Frame::TEME, cache)
        }
        (Frame::TEME, Frame::ITRS) | (Frame::ITRS, Frame::TEME) => teme_itrs_rotation(ut1, xp, yp),
        (Frame::GCRS, Frame::ITRS) | (Frame::ITRS, Frame::GCRS) => {
            gcrs_rotation(datetime_to_jd_tt(dt), ut1, xp, yp, Frame::ITRS, cache)
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
    // TEME->GCRS uses the inverse of the GCRS->TEME rotation.
    // GCRS->TEME uses that rotation directly.
    // TEME->ITRS and GCRS->ITRS are forward; their inverses start in ITRS.
    let needs_inverse = matches!(
        (input_frame, output_frame),
        (Frame::TEME, Frame::GCRS) | (Frame::ITRS, Frame::TEME) | (Frame::ITRS, Frame::GCRS)
    );

    // Sparse or tiny batches do not amortize the four interpolation nodes.
    // Keep their exact evaluation path; no global cache, locks or EOP state.
    let mut celestial_cache = (matches!(
        (input_frame, output_frame),
        (Frame::GCRS, Frame::ITRS)
            | (Frame::ITRS, Frame::GCRS)
            | (Frame::GCRS, Frame::TEME)
            | (Frame::TEME, Frame::GCRS)
    ) && n >= 4
        && times.windows(2).all(|pair| {
            (pair[1] - pair[0]).num_milliseconds().abs() <= (CELESTIAL_STEP_SECONDS * 1000.0) as i64
        }))
    .then(CelestialMatrixCache::default);

    for (i, dt) in times.iter().enumerate() {
        // Get the base rotation (always defined in the "forward" direction)
        let rotation = get_rotation(
            input_frame,
            output_frame,
            dt,
            polar_motion,
            celestial_cache.as_mut(),
        );

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

    #[test]
    fn test_cached_rotation_matches_exact_across_boundaries() {
        let mut cache = CelestialMatrixCache::default();
        // Subsecond queries, grid crossings, day crossings and out-of-order
        // epochs exercise eviction and the derivative's two stencil points.
        for day in [
            53736.0, 88069.0, 33282.0, 62502.0, 60310.0, 60401.0, 60492.0, 60583.0,
        ] {
            for seconds in [
                0.0, 0.25, 75.0, 150.0, 225.0, 299.75, 300.0, 300.25, 86400.0, 86399.75, 0.0,
            ] {
                let tt = (2400000.5, day + seconds / SECONDS_PER_DAY);
                let ut1 = (2400000.5, day + (seconds - 65.0) / SECONDS_PER_DAY);
                for (xp, yp) in [(0.0, 0.0), (2.55060238e-7, 1.860359247e-6)] {
                    let exact = gcrs_rotation(tt, ut1, xp, yp, Frame::ITRS, None);
                    let cached = gcrs_rotation(tt, ut1, xp, yp, Frame::ITRS, Some(&mut cache));
                    // Beyond GEO, in both directions, moving and stationary.
                    for vel in [[1.2, 6.8, -2.4], [0.0; 3]] {
                        let pos = [70000.0, -12000.0, 18000.0];
                        for inverse in [false, true] {
                            let (ep, ev) = exact.apply(pos, vel, inverse);
                            let (cp, cv) = cached.apply(pos, vel, inverse);
                            let (rp, rv) = cached.apply(cp, cv, !inverse);
                            for i in 0..3 {
                                assert!(
                                    (cp[i] - ep[i]).abs() < 1e-6,
                                    "position: {}",
                                    cp[i] - ep[i]
                                );
                                assert!(
                                    (cv[i] - ev[i]).abs() < 1e-9,
                                    "velocity: {}",
                                    cv[i] - ev[i]
                                );
                                assert!((rp[i] - pos[i]).abs() < 1e-9);
                                assert!((rv[i] - vel[i]).abs() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
    }

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
            let rotation =
                gcrs_rotation(date, date, 2.55060238e-7, 1.860359247e-6, Frame::ITRS, None);
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
    fn test_teme_independent_erfa_references() {
        // ERFA: Vallado gmst82 + pom00(xp, yp, 0), followed by inverse c2t06a
        // for GCRS. TT = MJD 53736; UT1 = TT - 65 s. Reference velocities use
        // a fourth-order 2-second stencil with small offsets on split dates.
        let tt = (2453736.5, 0.0);
        let ut1 = (2453735.5, 1.0 - 65.0 / SECONDS_PER_DAY);
        let xp = 2.55060238e-7;
        let yp = 1.860359247e-6;
        let pos = [7000.0, -1200.0, 1800.0];
        let vel = [1.2, 6.8, -2.4];
        let references = [
            (
                teme_itrs_rotation(ut1, xp, yp),
                false,
                [-2424.734576966214, -6675.380491901762, 1799.9881998507262],
                [5.99178300005794, -2.21237804717858, -2.400005644087713],
            ),
            (
                gcrs_rotation(tt, ut1, 0.0, 0.0, Frame::TEME, None),
                true,
                [6999.4249191088065, -1209.3186244943124, 1795.9953413664189],
                [1.207732380373841, 6.798287265377525, -2.400973217918344],
            ),
        ];
        for (rotation, inverse, expected_pos, expected_vel) in references {
            let (actual_pos, actual_vel) = rotation.apply(pos, vel, inverse);
            let (restored_pos, restored_vel) = rotation.apply(actual_pos, actual_vel, !inverse);
            for i in 0..3 {
                assert!((actual_pos[i] - expected_pos[i]).abs() < 1e-6);
                assert!((actual_vel[i] - expected_vel[i]).abs() < 1e-9);
                assert!((restored_pos[i] - pos[i]).abs() < 1e-9);
                assert!((restored_vel[i] - vel[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_teme_routes_agree_in_both_directions() {
        let mut cache = CelestialMatrixCache::default();
        for day in [53736.0, 88069.0, 33282.0, 62502.0, 60310.0] {
            for seconds in [0.0, 0.25, 299.75, 300.25, 86400.0, 86399.75, 0.0] {
                let tt = (2400000.5, day + seconds / SECONDS_PER_DAY);
                let ut1 = (2400000.5, day + (seconds - 65.0) / SECONDS_PER_DAY);
                let celestial = gcrs_rotation(tt, ut1, 0.0, 0.0, Frame::TEME, Some(&mut cache));
                let exact = gcrs_rotation(tt, ut1, 0.0, 0.0, Frame::TEME, None);
                for (xp, yp) in [(0.0, 0.0), (2.55060238e-7, 1.860359247e-6)] {
                    let terrestrial = gcrs_rotation(tt, ut1, xp, yp, Frame::ITRS, None);
                    let direct = teme_itrs_rotation(ut1, xp, yp);
                    for vel in [[1.2, 6.8, -2.4], [0.0; 3]] {
                        let pos = [70000.0, -12000.0, 18000.0];
                        for inverse in [false, true] {
                            let (direct_pos, direct_vel) = direct.apply(pos, vel, inverse);
                            let (via_pos, via_vel) = if inverse {
                                let (p, v) = terrestrial.apply(pos, vel, true);
                                celestial.apply(p, v, false)
                            } else {
                                let (p, v) = celestial.apply(pos, vel, true);
                                terrestrial.apply(p, v, false)
                            };
                            let (cp, cv) = celestial.apply(pos, vel, inverse);
                            let (ep, ev) = exact.apply(pos, vel, inverse);
                            let (rp, rv) = celestial.apply(cp, cv, !inverse);
                            for i in 0..3 {
                                assert!((via_pos[i] - direct_pos[i]).abs() < 1e-6);
                                assert!((via_vel[i] - direct_vel[i]).abs() < 1e-9);
                                assert!((cp[i] - ep[i]).abs() < 1e-6);
                                assert!((cv[i] - ev[i]).abs() < 1e-9);
                                assert!((rp[i] - pos[i]).abs() < 1e-9);
                                assert!((rv[i] - vel[i]).abs() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
    }
}
