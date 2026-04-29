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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarRollConfig {
    pub tolerance_deg: f64,
    #[serde(default)]
    pub roll_deg: Option<f64>,
}

impl ConstraintConfig for SolarRollConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(SolarRollEvaluator {
            tolerance_deg: self.tolerance_deg,
            roll_deg: self.roll_deg,
        })
    }
}

struct SolarRollEvaluator {
    tolerance_deg: f64,
    roll_deg: Option<f64>,
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
/// solar illumination of the +Y body panel.
///
/// North-referenced body frame (same convention as `boresight_rotate` in roll_range.rs):
///   x = boresight (target direction)
///   z = celestial north projected onto the plane perpendicular to x
///   y = z × x  (recomputed for orthonormality)
///
/// At roll θ the +Y body axis is:  y·cos(θ) + z·sin(θ)
/// Illumination of +Y panel:       s_y(θ) = sun_y·cos(θ) + sun_z·sin(θ)
/// Optimal roll (d/dθ = 0):        θ_opt = atan2(sun_z, sun_y)
pub fn solar_optimal_roll_deg(target: &[f64; 3], sun_unit: &[f64; 3]) -> f64 {
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

    f64::atan2(sun_z, sun_y).to_degrees()
}

/// Shortest angular arc between two roll angles (result in [0, 180]).
#[inline]
pub fn circular_diff_deg(a: f64, b: f64) -> f64 {
    ((a - b).rem_euclid(360.0) - 180.0).abs()
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
                let opt = solar_optimal_roll_deg(&target, &sun_units[i]);
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
                let opt = solar_optimal_roll_deg(&target, &sun_units[i]);
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
                let opt = solar_optimal_roll_deg(&target, &sun_unit);
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
