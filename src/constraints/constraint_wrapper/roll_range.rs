/// Vectorized roll-sweep helpers for `roll_range`.
///
/// The key optimization over a naive per-roll loop: instead of building N evaluators
/// and calling `in_constraint_batch` N times (once per roll), `roll_sweep_vec` walks
/// the constraint JSON tree and calls `in_constraint_batch` **once per leaf constraint**
/// with all N pre-rotated target directions.  This reduces the call count from
/// `O(N × leaves)` to `O(leaves)`.
use crate::ephemeris::ephemeris_common::EphemerisBase;
use pyo3::PyResult;

use super::json_parser::parse_constraint_json;

// ---------------------------------------------------------------------------
// Low-level vector helpers (avoid pulling in a heavy dependency for 3-element ops)
// ---------------------------------------------------------------------------

#[inline]
fn rsv_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn rsv_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn rsv_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn rsv_radec_to_unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let (sd, cd) = dec.sin_cos();
    let (sr, cr) = ra.sin_cos();
    [cd * cr, cd * sr, sd]
}

// ---------------------------------------------------------------------------
// Sun-direction helper
// ---------------------------------------------------------------------------

/// Return the normalized sun direction relative to the observer at `time_idx`.
/// Falls back to `[1, 0, 0]` if the positions are degenerate.
pub(super) fn get_sun_unit_at(ephem: &dyn EphemerisBase, time_idx: usize) -> PyResult<[f64; 3]> {
    let sun = ephem.get_sun_positions()?;
    let obs = ephem.get_gcrs_positions()?;
    let v = [
        sun[[time_idx, 0]] - obs[[time_idx, 0]],
        sun[[time_idx, 1]] - obs[[time_idx, 1]],
        sun[[time_idx, 2]] - obs[[time_idx, 2]],
    ];
    let n = rsv_norm(v);
    if n < 1.0e-12 {
        return Ok([1.0, 0.0, 0.0]);
    }
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

// ---------------------------------------------------------------------------
// Boresight rotation
// ---------------------------------------------------------------------------

/// Rotate `target` through a boresight-offset geometry.
///
/// * `eff_roll_ccw_deg` — combined roll (instrument base + spacecraft), CCW-positive.
/// * `z_ref` — reference axis for roll = 0: `[0, 0, 1]` (north) or the sun
///   direction.
///
/// Mirrors `BoresightOffsetEvaluator::rotated_target_for_time_with_params` exactly so
/// that `roll_range` produces numerically identical results to the full evaluator path.
pub(super) fn boresight_rotate(
    target: [f64; 3],
    z_ref: [f64; 3],
    eff_roll_ccw_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
) -> PyResult<[f64; 3]> {
    const NEAR_ZERO: f64 = 1.0e-12;
    let (sr, cr) = eff_roll_ccw_deg.to_radians().sin_cos();
    let (sp, cp) = pitch_deg.to_radians().sin_cos();
    let (sy, cy) = yaw_deg.to_radians().sin_cos();
    let local_x = cp * cy;
    let local_y = cp * sy;
    let local_z = -sp;

    let x_axis = target;

    // Project z_ref onto the plane normal to x_axis to form the roll=0 Z basis.
    let zref_dot_x = rsv_dot(x_axis, z_ref);
    let mut z_axis = [
        z_ref[0] - zref_dot_x * x_axis[0],
        z_ref[1] - zref_dot_x * x_axis[1],
        z_ref[2] - zref_dot_x * x_axis[2],
    ];
    if rsv_norm(z_axis) <= NEAR_ZERO {
        // z_ref is near-parallel to boresight; pick a stable fallback axis.
        let reference = if x_axis[2].abs() < 0.9 {
            [0.0_f64, 0.0, 1.0]
        } else {
            [0.0_f64, 1.0, 0.0]
        };
        let dot_ref_x = rsv_dot(x_axis, reference);
        z_axis = [
            reference[0] - dot_ref_x * x_axis[0],
            reference[1] - dot_ref_x * x_axis[1],
            reference[2] - dot_ref_x * x_axis[2],
        ];
    }
    let z_n = rsv_norm(z_axis);
    if z_n <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boresight: unable to construct roll frame",
        ));
    }
    let z_axis = [z_axis[0] / z_n, z_axis[1] / z_n, z_axis[2] / z_n];

    let y_raw = rsv_cross(z_axis, x_axis);
    let y_n = rsv_norm(y_raw);
    if y_n <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boresight: unable to construct +Y axis",
        ));
    }
    let y_axis = [y_raw[0] / y_n, y_raw[1] / y_n, y_raw[2] / y_n];
    let z_axis = rsv_cross(x_axis, y_axis); // recompute for strict orthonormality

    // Rotate the Y/Z frame axes by the effective roll.
    let y_roll = [
        y_axis[0] * cr + z_axis[0] * sr,
        y_axis[1] * cr + z_axis[1] * sr,
        y_axis[2] * cr + z_axis[2] * sr,
    ];
    let z_roll = [
        -y_axis[0] * sr + z_axis[0] * cr,
        -y_axis[1] * sr + z_axis[1] * cr,
        -y_axis[2] * sr + z_axis[2] * cr,
    ];

    Ok([
        local_x * x_axis[0] + local_y * y_roll[0] + local_z * z_roll[0],
        local_x * x_axis[1] + local_y * y_roll[1] + local_z * z_roll[1],
        local_x * x_axis[2] + local_y * y_roll[2] + local_z * z_roll[2],
    ])
}

// ---------------------------------------------------------------------------
// Vectorized roll sweep
// ---------------------------------------------------------------------------

/// Compute the violated status for every roll sample in a single tree-walk pass.
///
/// Returns `Vec<bool>` of length `rolls.len()` where `result[i] = true` means the
/// constraint is **violated** at spacecraft roll `rolls[i]`.
///
/// `target_ras[i]` / `target_decs[i]` are the target directions **already transformed
/// by ancestor boresight nodes** above the current node in the tree.
pub(super) fn roll_sweep_vec(
    config: &serde_json::Value,
    target_ras: &[f64],
    target_decs: &[f64],
    rolls: &[f64],
    ephem: &dyn EphemerisBase,
    time_idx: usize,
    sun_unit: &[f64; 3],
) -> PyResult<Vec<bool>> {
    let n = rolls.len();
    debug_assert_eq!(target_ras.len(), n);
    debug_assert_eq!(target_decs.len(), n);

    match config.get("type").and_then(|v| v.as_str()) {
        Some("boresight_offset") => {
            let base_roll = config
                .get("roll_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let clockwise = config
                .get("roll_clockwise")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let pitch_deg = config
                .get("pitch_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let yaw_deg = config
                .get("yaw_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let is_sun_ref = config
                .get("roll_reference")
                .and_then(|v| v.as_str())
                .unwrap_or("north")
                == "sun";
            let inner = config.get("constraint").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("boresight_offset is missing 'constraint'")
            })?;

            let z_ref: [f64; 3] = if is_sun_ref {
                *sun_unit
            } else {
                [0.0, 0.0, 1.0] // celestial north
            };
            let base_ccw = if clockwise { -base_roll } else { base_roll };

            // Pre-compute the rotated target direction for every roll sample.
            let mut new_ras = Vec::with_capacity(n);
            let mut new_decs = Vec::with_capacity(n);
            for i in 0..n {
                let target = rsv_radec_to_unit(target_ras[i], target_decs[i]);
                let eval_ccw = if clockwise { -rolls[i] } else { rolls[i] };
                let eff_roll = base_ccw + eval_ccw;
                let rotated = boresight_rotate(target, z_ref, eff_roll, pitch_deg, yaw_deg)?;
                let dec = rotated[2].clamp(-1.0, 1.0).asin().to_degrees();
                let mut ra = rotated[1].atan2(rotated[0]).to_degrees();
                if ra < 0.0 {
                    ra += 360.0;
                }
                new_ras.push(ra);
                new_decs.push(dec);
            }

            roll_sweep_vec(inner, &new_ras, &new_decs, rolls, ephem, time_idx, sun_unit)
        }

        // OR: violated if ANY child is violated.
        Some("or") => {
            let children = config
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("'or' node is missing 'constraints'")
                })?;
            if children.is_empty() {
                return Ok(vec![false; n]);
            }
            let mut result = roll_sweep_vec(
                &children[0],
                target_ras,
                target_decs,
                rolls,
                ephem,
                time_idx,
                sun_unit,
            )?;
            for child in &children[1..] {
                let child_result = roll_sweep_vec(
                    child,
                    target_ras,
                    target_decs,
                    rolls,
                    ephem,
                    time_idx,
                    sun_unit,
                )?;
                for (r, c) in result.iter_mut().zip(child_result) {
                    *r = *r || c;
                }
            }
            Ok(result)
        }

        // AND: violated only if ALL children are violated.
        Some("and") => {
            let children = config
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("'and' node is missing 'constraints'")
                })?;
            if children.is_empty() {
                return Ok(vec![true; n]); // vacuously all violated
            }
            let mut result = roll_sweep_vec(
                &children[0],
                target_ras,
                target_decs,
                rolls,
                ephem,
                time_idx,
                sun_unit,
            )?;
            for child in &children[1..] {
                let child_result = roll_sweep_vec(
                    child,
                    target_ras,
                    target_decs,
                    rolls,
                    ephem,
                    time_idx,
                    sun_unit,
                )?;
                for (r, c) in result.iter_mut().zip(child_result) {
                    *r = *r && c;
                }
            }
            Ok(result)
        }

        Some("not") => {
            let inner = config.get("constraint").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("'not' node is missing 'constraint'")
            })?;
            let result = roll_sweep_vec(
                inner,
                target_ras,
                target_decs,
                rolls,
                ephem,
                time_idx,
                sun_unit,
            )?;
            Ok(result.into_iter().map(|v| !v).collect())
        }

        // AT_LEAST: violated if at least `min_violated` children are violated.
        Some("at_least") => {
            let children = config
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "at_least node is missing 'constraints'",
                    )
                })?;
            if children.is_empty() {
                return Ok(vec![false; n]);
            }
            let min_violated = config
                .get("min_violated")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;

            // collect child results
            let mut child_results: Vec<Vec<bool>> = Vec::with_capacity(children.len());
            for child in children {
                child_results.push(roll_sweep_vec(
                    child,
                    target_ras,
                    target_decs,
                    rolls,
                    ephem,
                    time_idx,
                    sun_unit,
                )?);
            }

            let mut result = vec![false; n];
            for i in 0..n {
                let mut count = 0usize;
                for cr in &child_results {
                    if cr[i] {
                        count += 1;
                        if count >= min_violated {
                            break;
                        }
                    }
                }
                result[i] = count >= min_violated;
            }
            Ok(result)
        }

        // XOR: violated if exactly one child is violated.
        Some("xor") => {
            let children = config
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("xor node is missing 'constraints'")
                })?;
            if children.is_empty() {
                return Ok(vec![false; n]);
            }

            let mut child_results: Vec<Vec<bool>> = Vec::with_capacity(children.len());
            for child in children {
                child_results.push(roll_sweep_vec(
                    child,
                    target_ras,
                    target_decs,
                    rolls,
                    ephem,
                    time_idx,
                    sun_unit,
                )?);
            }

            let mut result = vec![false; n];
            for i in 0..n {
                let count = child_results.iter().filter(|cr| cr[i]).count();
                result[i] = count == 1;
            }
            Ok(result)
        }

        _ => {
            // Leaf constraint (sun, moon, eclipse, …) or unsupported compound (xor,
            // at_least).  Build ONE evaluator, call in_constraint_batch once with all N
            // pre-rotated targets at time_idx.  arr[[i, 0]] = violated status for roll i.
            let evaluator = parse_constraint_json(config)?;
            let arr =
                evaluator.in_constraint_batch(ephem, target_ras, target_decs, Some(&[time_idx]))?;
            Ok((0..n).map(|i| arr[[i, 0]]).collect())
        }
    }
}

/// Resolve sun unit vector at `time_idx`, then run the vectorized roll sweep.
pub(super) fn run_roll_sweep(
    config: &serde_json::Value,
    target_ras: &[f64],
    target_decs: &[f64],
    rolls: &[f64],
    ephem: &dyn EphemerisBase,
    time_idx: usize,
) -> PyResult<Vec<bool>> {
    let sun_unit = get_sun_unit_at(ephem, time_idx)?;
    roll_sweep_vec(
        config,
        target_ras,
        target_decs,
        rolls,
        ephem,
        time_idx,
        &sun_unit,
    )
}
