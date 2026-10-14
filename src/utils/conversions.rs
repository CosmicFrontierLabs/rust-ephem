/// Generic coordinate frame conversion functions.
///
/// This module provides reusable conversion functions for transforming between
/// different coordinate frames (TEME, ITRS, GCRS), shared by ephemeris classes.
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::{exceptions::PyValueError, PyResult};
use sofars::{
    consts::{DAS2R, DJ00, DJC, DS2R},
    erst::{era00, gmst82},
    pnp::{c2i06a, pom00, sp00},
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

/// Compose rotations and their derivatives using the product rule.
fn compose(left: &Rotation, right: &Rotation) -> Rotation {
    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| left.matrix[i][k] * right.matrix[k][j]).sum())
    });
    let rate = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..3)
                .map(|k| {
                    left.rate[i][k] * right.matrix[k][j] + left.matrix[i][k] * right.rate[k][j]
                })
                .sum()
        })
    });
    Rotation { matrix, rate }
}

fn rotation_z(angle: f64, rate: f64) -> Rotation {
    let (s, c) = angle.sin_cos();
    Rotation {
        matrix: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        rate: [
            [-s * rate, c * rate, 0.0],
            [-c * rate, -s * rate, 0.0],
            [0.0; 3],
        ],
    }
}

fn split_day(date: (f64, f64)) -> (f64, f64) {
    (date.0 + date.1.floor(), date.1 - date.1.floor())
}

/// Analytic derivative of SOFA's IAU 1982 GMST polynomial, radians/second.
fn gmst82_rate(ut1: (f64, f64)) -> f64 {
    let t = ((ut1.0 - DJ00) + ut1.1) / DJC;
    DS2R * (1.0
        + (8640184.812866 + 2.0 * 0.093104 * t - 3.0 * 6.2e-6 * t * t) / (DJC * SECONDS_PER_DAY))
}

/// Vallado TEME -> ITRS, with EOP held fixed during the local derivative.
fn teme_itrs_rotation(ut1: (f64, f64), xp: f64, yp: f64) -> Rotation {
    let ut1 = split_day(ut1);
    compose(
        &Rotation {
            matrix: pom00(xp, yp, 0.0),
            rate: [[0.0; 3]; 3],
        },
        &rotation_z(gmst82(ut1.0, ut1.1), gmst82_rate(ut1)),
    )
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Keep interpolation on SO(3), differentiating the same orthonormalization.
/// This makes transpose a valid inverse even for planetary-distance states.
fn orthonormalize(rotation: Rotation) -> Rotation {
    fn normalize(v: [f64; 3], dv: [f64; 3]) -> ([f64; 3], [f64; 3]) {
        let length = dot(v, v).sqrt();
        let n = v.map(|x| x / length);
        let radial_rate = dot(n, dv);
        (
            n,
            std::array::from_fn(|i| (dv[i] - n[i] * radial_rate) / length),
        )
    }
    let (a, da) = normalize(rotation.matrix[0], rotation.rate[0]);
    let projection = dot(a, rotation.matrix[1]);
    let projection_rate = dot(da, rotation.matrix[1]) + dot(a, rotation.rate[1]);
    let (b, db) = normalize(
        std::array::from_fn(|i| rotation.matrix[1][i] - projection * a[i]),
        std::array::from_fn(|i| rotation.rate[1][i] - projection_rate * a[i] - projection * da[i]),
    );
    let dc1 = cross(da, b);
    let dc2 = cross(a, db);
    Rotation {
        matrix: [a, b, cross(a, b)],
        rate: [da, db, std::array::from_fn(|i| dc1[i] + dc2[i])],
    }
}

#[cfg(test)]
thread_local! {
    static CELESTIAL_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn celestial_matrix(tt: (f64, f64)) -> [[f64; 3]; 3] {
    #[cfg(test)]
    CELESTIAL_EVALUATIONS.with(|count| count.set(count.get() + 1));
    c2i06a(tt.0, tt.1)
}

/// Two full series evaluations suffice for an isolated sample: midpoint Q and
/// its centered derivative. Q curvature over half a second is below roundoff
/// at the tested epochs; Earth spin is never averaged or differenced here.
fn isolated_celestial_rotation(tt: (f64, f64)) -> Rotation {
    let before = celestial_matrix((tt.0, tt.1 - 0.5 / SECONDS_PER_DAY));
    let after = celestial_matrix((tt.0, tt.1 + 0.5 / SECONDS_PER_DAY));
    orthonormalize(Rotation {
        matrix: std::array::from_fn(|i| {
            std::array::from_fn(|j| (after[i][j] + before[i][j]) * 0.5)
        }),
        rate: std::array::from_fn(|i| std::array::from_fn(|j| after[i][j] - before[i][j])),
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
    fn at(&mut self, tt: (f64, f64)) -> Rotation {
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
                    celestial_matrix((
                        tt.0,
                        node as f64 * CELESTIAL_STEP_SECONDS / SECONDS_PER_DAY,
                    )),
                ));
            }
            slot.as_ref().unwrap().2
        });
        // Differentiate the cubic itself, using the same four nodes once.
        // Subtract the center matrix to suppress cancellation of constant terms.
        let derivative_weights = [
            (-3.0 * x * x + 6.0 * x - 2.0) / 6.0,
            (3.0 * x * x - 4.0 * x - 1.0) / 2.0,
            (-3.0 * x * x + 2.0 * x + 2.0) / 2.0,
            (3.0 * x * x - 1.0) / 6.0,
        ];
        let matrix = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                matrices[1][i][j]
                    + (0..4)
                        .map(|k| weights[k] * (matrices[k][i][j] - matrices[1][i][j]))
                        .sum::<f64>()
            })
        });
        let rate = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..4)
                    .map(|k| derivative_weights[k] * (matrices[k][i][j] - matrices[1][i][j]))
                    .sum::<f64>()
                    / CELESTIAL_STEP_SECONDS
            })
        });
        orthonormalize(Rotation { matrix, rate })
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
    cache: Option<&mut CelestialMatrixCache>,
) -> Rotation {
    let tt = split_day(tt);
    let ut1 = split_day(ut1);
    let q = match cache {
        Some(cache) => cache.at(tt),
        None => isolated_celestial_rotation(tt),
    };
    // W(xp,yp,s') = W(xp,yp,0) R3(s'). Combine the two axial rotations
    // before applying Q; retain both Qdot and the small s' rate.
    let era = era00(ut1.0, ut1.1);
    let era_rate = std::f64::consts::TAU * 1.002_737_811_911_354_6 / SECONDS_PER_DAY;
    let sp = sp00(tt.0, tt.1);
    let sp_rate = -47e-6 * DAS2R / (DJC * SECONDS_PER_DAY);
    let (angle, rate, polar) = match to {
        Frame::ITRS => (era + sp, era_rate + sp_rate, pom00(xp, yp, 0.0)),
        Frame::TEME => (
            era - gmst82(ut1.0, ut1.1) + sp,
            era_rate - gmst82_rate(ut1) + sp_rate,
            pom00(0.0, 0.0, 0.0),
        ),
        Frame::GCRS => unreachable!("GCRS is already the input frame"),
    };
    compose(
        &Rotation {
            matrix: polar,
            rate: [[0.0; 3]; 3],
        },
        &compose(&rotation_z(angle, rate), &q),
    )
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
) -> PyResult<Array2<f64>> {
    if data.nrows() != times.len() || data.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "Frame conversion requires shape ({}, 6), got ({}, {})",
            times.len(),
            data.nrows(),
            data.ncols()
        )));
    }
    // Fast path: same-frame -> return a copy
    if input_frame == output_frame {
        return Ok(data.to_owned());
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

    let mut celestial_cache = CelestialMatrixCache::default();

    for (i, dt) in times.iter().enumerate() {
        // Only nearby samples amortize interpolation. A data hole affects its
        // isolated samples, never the dense stretches elsewhere in the batch.
        let nearby = |other: &DateTime<Utc>| {
            (*other - *dt).num_milliseconds().abs() <= (CELESTIAL_STEP_SECONDS * 1000.0) as i64
        };
        let use_cache = times.get(i + 1).is_some_and(nearby)
            || i.checked_sub(1)
                .and_then(|j| times.get(j))
                .is_some_and(nearby);
        // Get the base rotation (always defined in the "forward" direction)
        let rotation = get_rotation(
            input_frame,
            output_frame,
            dt,
            polar_motion,
            use_cache.then_some(&mut celestial_cache),
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

    Ok(out)
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
    #[test]
    fn test_shape_errors_are_value_errors_even_for_identity() {
        pyo3::Python::initialize();
        let times = ["2024-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap()];
        for shape in [(0, 6), (2, 6), (1, 3), (1, 7)] {
            for to in [Frame::GCRS, Frame::ITRS] {
                let err = convert_frames(&Array2::zeros(shape), &times, Frame::GCRS, to, false)
                    .unwrap_err();
                pyo3::Python::attach(|py| assert!(err.is_instance_of::<PyValueError>(py)));
            }
        }
        assert_eq!(
            convert_frames(&Array2::zeros((0, 6)), &[], Frame::GCRS, Frame::ITRS, false)
                .unwrap()
                .shape(),
            &[0, 6]
        );
    }

    #[test]
    fn test_planetary_round_trips_and_rotation_derivatives() {
        let mut cache = CelestialMatrixCache::default();
        for day in [33282.0, 53736.0, 60310.0, 88069.0] {
            for seconds in [0.0, 0.25, 75.0, 299.75, 300.25, 86399.75] {
                let tt = (2400000.5 + day, seconds / SECONDS_PER_DAY);
                for cached in [false, true] {
                    for to in [Frame::ITRS, Frame::TEME] {
                        let r = gcrs_rotation(tt, tt, 1e-6, 2e-6, to, cached.then_some(&mut cache));
                        // A rotation's derivative must be tangent to SO(3).
                        for i in 0..3 {
                            for j in 0..3 {
                                let identity = if i == j { 1.0 } else { 0.0 };
                                assert!((dot(r.matrix[i], r.matrix[j]) - identity).abs() < 1e-15);
                                assert!(
                                    (dot(r.rate[i], r.matrix[j]) + dot(r.matrix[i], r.rate[j]))
                                        .abs()
                                        < 1e-19
                                );
                            }
                        }
                        for distance in [384400.0, 149597870.7, 4.5e9] {
                            let p = [distance, -distance * 0.3, distance * 0.2];
                            let v = [1.2, -6.8, 2.4];
                            for inverse in [false, true] {
                                let (p1, v1) = r.apply(p, v, inverse);
                                let (p2, v2) = r.apply(p1, v1, !inverse);
                                for i in 0..3 {
                                    assert!((p2[i] - p[i]).abs() < 2e-15 * distance);
                                    assert!((v2[i] - v[i]).abs() < 2e-15 * distance * 1e-4 + 1e-12);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_sparse_gap_and_reordered_batches_agree() {
        let begin = "2024-01-01T23:59:37.250Z".parse::<DateTime<Utc>>().unwrap();
        let offsets = [0, 60, 120, 180, 7200, 7260, 7320, 7380, 20000];
        let times: Vec<_> = offsets.map(|s| begin + chrono::Duration::seconds(s)).into();
        let data = Array2::from_shape_fn((times.len(), 6), |(_, j)| {
            [70000.0, -12000.0, 18000.0, 1.2, 6.8, -2.4][j]
        });
        for (from, to) in [(Frame::GCRS, Frame::ITRS), (Frame::TEME, Frame::GCRS)] {
            let whole = convert_frames(&data, &times, from, to, false).unwrap();
            for (i, time) in times.iter().enumerate() {
                let single = convert_frames(
                    &data.slice(ndarray::s![i..i + 1, ..]).to_owned(),
                    &[*time],
                    from,
                    to,
                    false,
                )
                .unwrap();
                for j in 0..6 {
                    let tolerance = if j < 3 { 1e-6 } else { 1e-9 };
                    assert!((whole[[i, j]] - single[[0, j]]).abs() < tolerance);
                }
            }
            let reversed = convert_frames(
                &data,
                &times.iter().copied().rev().collect::<Vec<_>>(),
                from,
                to,
                false,
            )
            .unwrap();
            for i in 0..times.len() {
                for j in 0..6 {
                    let tolerance = if j < 3 { 1e-6 } else { 1e-9 };
                    assert!((whole[[i, j]] - reversed[[times.len() - i - 1, j]]).abs() < tolerance);
                }
            }
        }
    }
    #[test]
    fn test_series_work_is_bounded_for_isolated_and_gapped_queries() {
        let begin = "2024-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let dense: Vec<_> = (0..120)
            .map(|i| begin + chrono::Duration::seconds(60 * i))
            .collect();
        let gapped: Vec<_> = dense
            .iter()
            .enumerate()
            .map(|(i, t)| *t + chrono::Duration::seconds(if i >= 60 { 7200 } else { 0 }))
            .collect();
        for times in [
            vec![begin],
            vec![begin, begin + chrono::Duration::seconds(3600)],
            dense,
            gapped,
        ] {
            CELESTIAL_EVALUATIONS.with(|count| count.set(0));
            convert_frames(
                &Array2::zeros((times.len(), 6)),
                &times,
                Frame::GCRS,
                Frame::ITRS,
                false,
            )
            .unwrap();
            let evaluations = CELESTIAL_EVALUATIONS.with(|count| count.get());
            if times.len() <= 2 {
                assert_eq!(evaluations, 2 * times.len());
            } else {
                // One isolated hole must not cause 3 full evaluations for all 120 samples.
                assert!(evaluations < 40, "{evaluations} full series evaluations");
            }
        }
        let mut cache = CelestialMatrixCache::default();
        CELESTIAL_EVALUATIONS.with(|count| count.set(0));
        cache.at((2453736.5, 300.25 / SECONDS_PER_DAY));
        cache.at((2453736.5, 300.49 / SECONDS_PER_DAY));
        assert_eq!(CELESTIAL_EVALUATIONS.with(|count| count.get()), 4);
    }
}
