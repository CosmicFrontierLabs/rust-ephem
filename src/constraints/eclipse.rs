/// Eclipse constraint implementation
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult, ConstraintViolation};
use crate::utils::config::{EARTH_RADIUS_KM, SUN_RADIUS_KM};
use crate::utils::vector_math::vector_magnitude;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for eclipse constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EclipseConfig {
    /// Umbra only (true) or include penumbra (false)
    pub umbra_only: bool,
}

impl ConstraintConfig for EclipseConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(EclipseEvaluator {
            umbra_only: self.umbra_only,
        })
    }
}

/// Evaluator for eclipse constraint
struct EclipseEvaluator {
    umbra_only: bool,
}

impl EclipseEvaluator {
    fn shadow_geometry(
        obs_pos: [f64; 3],
        sun_pos: [f64; 3],
    ) -> Option<(f64, f64, f64)> {
        let sun_dist = vector_magnitude(&sun_pos);
        if sun_dist <= 0.0 {
            return None;
        }

        let sun_unit = [
            sun_pos[0] / sun_dist,
            sun_pos[1] / sun_dist,
            sun_pos[2] / sun_dist,
        ];

        // Observer must be behind Earth relative to Sun direction.
        let dot = obs_pos[0] * sun_unit[0] + obs_pos[1] * sun_unit[1] + obs_pos[2] * sun_unit[2];
        if dot >= 0.0 {
            return None;
        }

        // Distance along the shadow axis (behind Earth).
        let s = -dot;

        // Perpendicular distance from shadow axis.
        let perp = [
            obs_pos[0] - sun_unit[0] * dot,
            obs_pos[1] - sun_unit[1] * dot,
            obs_pos[2] - sun_unit[2] * dot,
        ];
        let dist_to_axis = vector_magnitude(&perp);

        // Umbra and penumbra cone lengths.
        let l_umbra = EARTH_RADIUS_KM * sun_dist / (SUN_RADIUS_KM - EARTH_RADIUS_KM);
        let l_penumbra = EARTH_RADIUS_KM * sun_dist / (SUN_RADIUS_KM + EARTH_RADIUS_KM);

        // Umbra radius decreases linearly to zero at L_umbra.
        let umbra_radius = if s <= l_umbra {
            EARTH_RADIUS_KM * (1.0 - s / l_umbra)
        } else {
            0.0
        };

        // Penumbra radius increases linearly with distance.
        let penumbra_radius = EARTH_RADIUS_KM * (1.0 + s / l_penumbra);

        Some((dist_to_axis, umbra_radius, penumbra_radius))
    }

    fn shadow_status(&self, obs_pos: [f64; 3], sun_pos: [f64; 3]) -> (bool, f64) {
        if let Some((dist_to_axis, umbra_radius, penumbra_radius)) =
            Self::shadow_geometry(obs_pos, sun_pos)
        {
            let in_umbra = umbra_radius > 0.0 && dist_to_axis < umbra_radius;
            if in_umbra {
                let severity = 1.0 - dist_to_axis / umbra_radius;
                return (true, severity);
            }

            if !self.umbra_only && dist_to_axis < penumbra_radius {
                let denom = (penumbra_radius - umbra_radius).max(1e-9);
                let penumbra_depth = (penumbra_radius - dist_to_axis) / denom;
                return (true, 0.5 * penumbra_depth);
            }
        }

        (false, 0.0)
    }

    /// Compute eclipse mask for all times (returns true where eclipse occurs)
    fn compute_eclipse_mask(
        &self,
        times: &[DateTime<Utc>],
        sun_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> Vec<bool> {
        let mut result = vec![false; times.len()];

        for i in 0..times.len() {
            let obs_pos = [
                observer_positions[[i, 0]],
                observer_positions[[i, 1]],
                observer_positions[[i, 2]],
            ];

            let sun_pos = [
                sun_positions[[i, 0]],
                sun_positions[[i, 1]],
                sun_positions[[i, 2]],
            ];

            let (in_shadow, _severity) = self.shadow_status(obs_pos, sun_pos);
            result[i] = in_shadow;
        }

        result
    }
}

impl ConstraintEvaluator for EclipseEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        _target_ra: f64,
        _target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Extract and filter ephemeris data
        let (times_filtered, sun_filtered, obs_filtered) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        for (i, _time) in times_filtered.iter().enumerate() {
            let obs_pos = [
                obs_filtered[[i, 0]],
                obs_filtered[[i, 1]],
                obs_filtered[[i, 2]],
            ];

            let sun_pos = [
                sun_filtered[[i, 0]],
                sun_filtered[[i, 1]],
                sun_filtered[[i, 2]],
            ];

            let (in_shadow, severity) = self.shadow_status(obs_pos, sun_pos);

            if in_shadow {
                match current_violation {
                    Some((start_idx, max_sev)) => {
                        current_violation = Some((start_idx, max_sev.max(severity)));
                    }
                    None => {
                        current_violation = Some((i, severity));
                    }
                }
            } else if let Some((start_idx, max_severity)) = current_violation {
                violations.push(ConstraintViolation {
                    start_time_internal: times_filtered[start_idx],
                    end_time_internal: times_filtered[i - 1],
                    max_severity,
                    description: if self.umbra_only {
                        "Observer in umbra".to_string()
                    } else {
                        "Observer in shadow".to_string()
                    },
                });
                current_violation = None;
            }
        }

        // Close any open violation at the end
        if let Some((start_idx, max_severity)) = current_violation {
            violations.push(ConstraintViolation {
                start_time_internal: times_filtered[start_idx],
                end_time_internal: times_filtered[times_filtered.len() - 1],
                max_severity,
                description: if self.umbra_only {
                    "Observer in umbra".to_string()
                } else {
                    "Observer in shadow".to_string()
                },
            });
        }

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered.to_vec(),
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        // Extract and filter ephemeris data
        let (times_filtered, sun_filtered, obs_filtered) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();

        // Eclipse is target-independent - compute once for all times
        let time_results = self.compute_eclipse_mask(&times_filtered, &sun_filtered, &obs_filtered);

        // Broadcast to all targets (same result for each RA/Dec)
        let result = Array2::from_shape_fn((n_targets, n_times), |(_, j)| time_results[j]);

        Ok(result)
    }

    fn name(&self) -> String {
        format!(
            "Eclipse({})",
            if self.umbra_only {
                "umbra"
            } else {
                "umbra+penumbra"
            }
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::EclipseEvaluator;
    use crate::utils::config::AU_TO_KM;

    #[test]
    fn test_penumbra_wider_than_umbra() {
        let sun_pos = [AU_TO_KM, 0.0, 0.0];
        let obs_pos = [-7000.0, 0.0, 0.0];
        let (_dist_to_axis, umbra_radius, penumbra_radius) =
            EclipseEvaluator::shadow_geometry(obs_pos, sun_pos).expect("shadow geometry");
        assert!(umbra_radius > 0.0, "umbra radius should be positive");
        assert!(
            penumbra_radius > umbra_radius,
            "penumbra radius should exceed umbra radius"
        );
    }

    #[test]
    fn test_penumbra_includes_outside_umbra() {
        let sun_pos = [AU_TO_KM, 0.0, 0.0];
        let s = 7000.0;
        let on_axis = [-s, 0.0, 0.0];
        let (_dist_to_axis, umbra_radius, penumbra_radius) =
            EclipseEvaluator::shadow_geometry(on_axis, sun_pos).expect("shadow geometry");
        let d = 0.5 * (umbra_radius + penumbra_radius);
        let obs_pos = [-s, d, 0.0];

        let umbra_only = EclipseEvaluator { umbra_only: true };
        let with_penumbra = EclipseEvaluator { umbra_only: false };

        let (in_umbra_only, _) = umbra_only.shadow_status(obs_pos, sun_pos);
        let (in_penumbra, _) = with_penumbra.shadow_status(obs_pos, sun_pos);

        assert!(!in_umbra_only, "point should be outside umbra");
        assert!(in_penumbra, "point should be inside penumbra");
    }
}
