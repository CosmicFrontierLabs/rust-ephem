/// Solar roll constraint implementation.
///
/// Violated when the spacecraft's actual roll deviates from the solar-optimal roll
/// (the roll that maximises solar illumination of the +Y body panel) by more than
/// `tolerance_deg` degrees.  The optimal roll is computed using the north-referenced
/// body frame convention shared with `boresight_rotate` in roll_range.rs.
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::ephemeris::ephemeris_common::EphemerisBase;
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

pub fn default_panel_normal() -> [f64; 3] {
    [0.0, 1.0, 0.0]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarRollConfig {
    pub tolerance_deg: f64,
    #[serde(default)]
    pub roll_deg: Option<f64>,
    #[serde(default = "default_panel_normal")]
    pub panel_normal: [f64; 3],
}

impl ConstraintConfig for SolarRollConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(SolarRollEvaluator {
            tolerance_deg: self.tolerance_deg,
            roll_deg: self.roll_deg,
            panel_normal: self.panel_normal,
        })
    }
}

struct SolarRollEvaluator {
    tolerance_deg: f64,
    roll_deg: Option<f64>,
    panel_normal: [f64; 3],
}

const NEAR_ZERO: f64 = 1.0e-12;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm3(v);
    if n <= NEAR_ZERO {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

#[inline]
fn radec_to_unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let (sd, cd) = dec.sin_cos();
    let (sr, cr) = ra.sin_cos();
    [cd * cr, cd * sr, sd]
}

/// Compute the north-referenced spacecraft roll (degrees, CCW positive) that maximises
/// solar illumination of the panel with the given body-frame normal.
///
/// North-referenced body frame (same convention as `boresight_rotate` in roll_range.rs):
///   x = boresight (target direction)
///   z = celestial north projected onto the plane perpendicular to x
///   y = z × x  (recomputed for orthonormality)
///
/// At roll θ the body Y-axis is:  y_ref·cos(θ) + z_ref·sin(θ)
/// For panel normal n = [nx, ny, nz] (body frame, x = boresight), the panel
/// illumination is maximised at:
///   θ_opt = atan2(sun_z, sun_y) - atan2(nz, ny)
/// The +Y default (ny=1, nz=0) recovers the original formula.
pub fn solar_optimal_roll_deg(
    target: &[f64; 3],
    sun_unit: &[f64; 3],
    panel_normal: [f64; 3],
) -> f64 {
    let x_axis = *target;
    let z_ref = [0.0_f64, 0.0, 1.0];

    let zref_dot_x = dot3(x_axis, z_ref);
    let z_raw = [
        z_ref[0] - zref_dot_x * x_axis[0],
        z_ref[1] - zref_dot_x * x_axis[1],
        z_ref[2] - zref_dot_x * x_axis[2],
    ];

    let z_axis = normalize3(z_raw).unwrap_or_else(|| {
        // Boresight near the celestial pole — pick a stable orthogonal fallback.
        let fallback: [f64; 3] = if x_axis[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let d = dot3(x_axis, fallback);
        let raw = [
            fallback[0] - d * x_axis[0],
            fallback[1] - d * x_axis[1],
            fallback[2] - d * x_axis[2],
        ];
        normalize3(raw).unwrap_or([0.0, 1.0, 0.0])
    });

    let y_raw = cross3(z_axis, x_axis);
    let y_axis = normalize3(y_raw).unwrap_or([0.0, 1.0, 0.0]);
    let z_axis = cross3(x_axis, y_axis); // recomputed for strict orthonormality

    let sun_y = dot3(*sun_unit, y_axis);
    let sun_z = dot3(*sun_unit, z_axis);

    // Shift by the angle of the panel normal in the body Y-Z plane.
    let panel_angle = f64::atan2(panel_normal[2], panel_normal[1]);
    (f64::atan2(sun_z, sun_y) - panel_angle).to_degrees()
}

/// Shortest angular arc between two roll angles (result in [0, 180]).
#[inline]
pub fn circular_diff_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    180.0 - (d - 180.0).abs()
}

impl SolarRollEvaluator {
    fn name_str(&self) -> String {
        format!("SolarRollConstraint(tolerance={:.1}°)", self.tolerance_deg)
    }
}

impl ConstraintEvaluator for SolarRollEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times().expect("Ephemeris must have times");
        let times_filtered: Vec<_> = if let Some(idx) = time_indices {
            idx.iter().map(|&i| times[i]).collect()
        } else {
            times.to_vec()
        };

        let Some(roll) = self.roll_deg else {
            // No roll provided — constraint is not applicable; report always satisfied.
            return Ok(ConstraintResult::new(
                vec![],
                true,
                self.name_str(),
                times_filtered,
            ));
        };

        let target = radec_to_unit(target_ra, target_dec);

        // Pre-compute sun unit vectors with correct ephemeris index mapping.
        let sun = ephemeris.get_sun_positions()?;
        let obs = ephemeris.get_gcrs_positions()?;
        let sun_units: Vec<[f64; 3]> = (0..times_filtered.len())
            .map(|i| {
                let idx = time_indices.map_or(i, |indices| indices[i]);
                let v = [
                    sun[[idx, 0]] - obs[[idx, 0]],
                    sun[[idx, 1]] - obs[[idx, 1]],
                    sun[[idx, 2]] - obs[[idx, 2]],
                ];
                let n = norm3(v);
                if n < NEAR_ZERO {
                    [1.0, 0.0, 0.0]
                } else {
                    [v[0] / n, v[1] / n, v[2] / n]
                }
            })
            .collect();

        let violations = track_violations(
            &times_filtered,
            |i| {
                let opt = solar_optimal_roll_deg(&target, &sun_units[i], self.panel_normal);
                let diff = circular_diff_deg(roll, opt);
                let violated = diff > self.tolerance_deg;
                (
                    violated,
                    if violated {
                        diff - self.tolerance_deg
                    } else {
                        0.0
                    },
                )
            },
            |i, violated| {
                if !violated {
                    return "".to_string();
                }
                let opt = solar_optimal_roll_deg(&target, &sun_units[i], self.panel_normal);
                let diff = circular_diff_deg(roll, opt);
                format!(
                    "Roll {:.1}° deviates {:.1}° from solar-optimal {:.1}° (tolerance {:.1}°)",
                    roll, diff, opt, self.tolerance_deg
                )
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name_str(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        let (times_filtered,) = extract_time_data!(ephemeris, time_indices);
        let n_targets = target_ras.len();
        let n_times = times_filtered.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        let Some(roll) = self.roll_deg else {
            // No roll — always satisfied (not violated).
            return Ok(result);
        };

        let sun = ephemeris.get_sun_positions()?;
        let obs = ephemeris.get_gcrs_positions()?;

        for j in 0..n_targets {
            let target = radec_to_unit(target_ras[j], target_decs[j]);
            for i in 0..n_times {
                let source_i = time_indices.map_or(i, |indices| indices[i]);
                let v = [
                    sun[[source_i, 0]] - obs[[source_i, 0]],
                    sun[[source_i, 1]] - obs[[source_i, 1]],
                    sun[[source_i, 2]] - obs[[source_i, 2]],
                ];
                let n = norm3(v);
                let sun_unit = if n < NEAR_ZERO {
                    [1.0, 0.0, 0.0]
                } else {
                    [v[0] / n, v[1] / n, v[2] / n]
                };
                let opt = solar_optimal_roll_deg(&target, &sun_unit, self.panel_normal);
                result[[j, i]] = circular_diff_deg(roll, opt) > self.tolerance_deg;
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        _ephemeris: &dyn EphemerisBase,
        _target_unit_vectors: &Array2<f64>,
        _time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        Ok(None)
    }

    fn name(&self) -> String {
        self.name_str()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Boresight along GCRS +X gives a clean reference frame:
    //   x_axis = [1, 0, 0]
    //   z_ref  = [0, 0, 1]  → z_raw = [0, 0, 1] (z_ref fully perpendicular to x)
    //   z_axis = [0, 0, 1]
    //   y_raw  = z × x = [0, 1, 0]
    //   y_axis = [0, 1, 0]
    // So sun_y = sun·ŷ and sun_z = sun·ẑ are just the sun's Y and Z GCRS components.
    const X_BORESIGHT: [f64; 3] = [1.0, 0.0, 0.0];

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-8
    }

    // --- solar_optimal_roll_deg ---

    #[test]
    fn test_optimal_roll_default_panel_sun_along_y() {
        // Sun on +Y, panel normal +Y → sun is already on the panel → optimal roll = 0°
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq(opt, 0.0), "expected 0°, got {opt}");
    }

    #[test]
    fn test_optimal_roll_default_panel_sun_along_z() {
        // Sun on +Z, panel normal +Y → need 90° roll to face the sun
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq(opt, 90.0), "expected 90°, got {opt}");
    }

    #[test]
    fn test_optimal_roll_default_panel_sun_along_neg_y() {
        // Sun on -Y, panel normal +Y → need ±180° roll
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq(opt.abs(), 180.0), "expected ±180°, got {opt}");
    }

    #[test]
    fn test_optimal_roll_panel_z_sun_along_z() {
        // Sun on +Z, panel normal +Z → no roll needed, optimal = 0°
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        assert!(approx_eq(opt, 0.0), "expected 0°, got {opt}");
    }

    #[test]
    fn test_optimal_roll_panel_z_sun_along_y() {
        // Sun on +Y, panel normal +Z → optimal is -90° (panel must roll -90° to face +Y)
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(approx_eq(opt, -90.0), "expected -90°, got {opt}");
    }

    #[test]
    fn test_optimal_roll_panel_45deg_sun_along_z() {
        // Panel tilted 45° from +Y toward +Z, sun on +Z → need 45° roll
        let c = 45.0_f64.to_radians().cos();
        let s = 45.0_f64.to_radians().sin();
        let opt = solar_optimal_roll_deg(&X_BORESIGHT, &[0.0, 0.0, 1.0], [0.0, c, s]);
        assert!(approx_eq(opt, 45.0), "expected 45°, got {opt}");
    }

    #[test]
    fn test_panel_normal_shift_is_90deg() {
        // Rotating the panel 90° in the Y-Z plane shifts the optimal roll by exactly -90°
        // for any sun direction in the Y-Z plane.
        let sun = [0.0, 1.0_f64 / 2.0_f64.sqrt(), 1.0_f64 / 2.0_f64.sqrt()];
        let opt_y = solar_optimal_roll_deg(&X_BORESIGHT, &sun, [0.0, 1.0, 0.0]);
        let opt_z = solar_optimal_roll_deg(&X_BORESIGHT, &sun, [0.0, 0.0, 1.0]);
        // +Z panel needs 90° less roll than +Y panel to face the same sun direction.
        assert!(
            approx_eq(opt_y - opt_z, 90.0),
            "expected 90° shift, got {opt_y} - {opt_z} = {}",
            opt_y - opt_z
        );
    }

    #[test]
    fn test_panel_x_component_ignored() {
        // The X component of the panel normal (along boresight) does not affect the result.
        let sun = [0.0, 0.0, 1.0];
        let opt_no_x = solar_optimal_roll_deg(&X_BORESIGHT, &sun, [0.0, 1.0, 0.0]);
        let opt_with_x = solar_optimal_roll_deg(&X_BORESIGHT, &sun, [5.0, 1.0, 0.0]);
        assert!(
            approx_eq(opt_no_x, opt_with_x),
            "X component changed result: {opt_no_x} vs {opt_with_x}"
        );
    }

    // --- circular_diff_deg ---

    #[test]
    fn test_circular_diff_identical() {
        assert!(approx_eq(circular_diff_deg(45.0, 45.0), 0.0));
    }

    #[test]
    fn test_circular_diff_antipodal() {
        assert!(approx_eq(circular_diff_deg(0.0, 180.0), 180.0));
        assert!(approx_eq(circular_diff_deg(180.0, 0.0), 180.0));
    }

    #[test]
    fn test_circular_diff_wraparound() {
        // 350° and 10° differ by 20° across the 0/360 boundary.
        assert!(approx_eq(circular_diff_deg(350.0, 10.0), 20.0));
        assert!(approx_eq(circular_diff_deg(10.0, 350.0), 20.0));
    }

    #[test]
    fn test_circular_diff_symmetric() {
        let a = 270.0_f64;
        let b = 90.0_f64;
        assert!(approx_eq(circular_diff_deg(a, b), circular_diff_deg(b, a)));
    }
}
