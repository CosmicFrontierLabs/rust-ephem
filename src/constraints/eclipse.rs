/// Eclipse constraint implementation
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult, ConstraintViolation};
use crate::utils::vector_math::{normalize_vector, vector_magnitude};
use chrono::{DateTime, Utc};
use ndarray::Array2;
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
    /// Compute eclipse mask for all times (returns true where eclipse occurs)
    fn compute_eclipse_mask(
        &self,
        times: &[DateTime<Utc>],
        sun_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> Vec<bool> {
        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;
        // Sun radius in km (mean)
        const SUN_RADIUS: f64 = 696000.0;

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

            // Vector from observer to Sun
            let obs_to_sun = [
                sun_pos[0] - obs_pos[0],
                sun_pos[1] - obs_pos[1],
                sun_pos[2] - obs_pos[2],
            ];

            let sun_dist = vector_magnitude(&obs_to_sun);
            let sun_unit = normalize_vector(&obs_to_sun);

            // Find closest point on observer-to-Sun line to Earth center
            let t =
                -(obs_pos[0] * sun_unit[0] + obs_pos[1] * sun_unit[1] + obs_pos[2] * sun_unit[2]);

            // If closest point is behind observer or beyond Sun, not in shadow
            if t < 0.0 || t > sun_dist {
                continue;
            }

            // Closest point on line to Earth center
            let closest_point = [
                obs_pos[0] + t * sun_unit[0],
                obs_pos[1] + t * sun_unit[1],
                obs_pos[2] + t * sun_unit[2],
            ];

            // Distance from Earth center to closest point
            let dist_to_earth = vector_magnitude(&closest_point);

            // Calculate umbra and penumbra radii at observer distance
            let umbra_radius = EARTH_RADIUS - (EARTH_RADIUS - SUN_RADIUS) * t / sun_dist;
            let penumbra_radius = EARTH_RADIUS + (SUN_RADIUS - EARTH_RADIUS) * t / sun_dist;

            let in_shadow = if dist_to_earth < umbra_radius {
                // In umbra
                true
            } else if !self.umbra_only && dist_to_earth < penumbra_radius {
                // In penumbra
                true
            } else {
                false
            };

            result[i] = in_shadow;
        }

        result
    }
}

impl ConstraintEvaluator for EclipseEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        _target_ra: f64,
        _target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;
        // Sun radius in km (mean)
        const SUN_RADIUS: f64 = 696000.0;

        for (i, _time) in times.iter().enumerate() {
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

            // Vector from observer to Sun
            let obs_to_sun = [
                sun_pos[0] - obs_pos[0],
                sun_pos[1] - obs_pos[1],
                sun_pos[2] - obs_pos[2],
            ];

            let sun_dist = vector_magnitude(&obs_to_sun);
            let sun_unit = normalize_vector(&obs_to_sun);

            // Find closest point on observer-to-Sun line to Earth center
            let t =
                -(obs_pos[0] * sun_unit[0] + obs_pos[1] * sun_unit[1] + obs_pos[2] * sun_unit[2]);

            // If closest point is behind observer or beyond Sun, not in shadow
            if t < 0.0 || t > sun_dist {
                // Close any open violation
                if let Some((start_idx, max_severity)) = current_violation {
                    violations.push(ConstraintViolation {
                        start_time: times[start_idx].to_rfc3339(),
                        end_time: times[i - 1].to_rfc3339(),
                        max_severity,
                        description: if self.umbra_only {
                            "Observer in umbra".to_string()
                        } else {
                            "Observer in shadow".to_string()
                        },
                    });
                    current_violation = None;
                }
                continue;
            }

            // Closest point on line to Earth center
            let closest_point = [
                obs_pos[0] + t * sun_unit[0],
                obs_pos[1] + t * sun_unit[1],
                obs_pos[2] + t * sun_unit[2],
            ];

            // Distance from Earth center to closest point
            let dist_to_earth = vector_magnitude(&closest_point);

            // Calculate umbra and penumbra radii at observer distance
            let umbra_radius = EARTH_RADIUS - (EARTH_RADIUS - SUN_RADIUS) * t / sun_dist;
            let penumbra_radius = EARTH_RADIUS + (SUN_RADIUS - EARTH_RADIUS) * t / sun_dist;

            let (in_shadow, severity) = if dist_to_earth < umbra_radius {
                // In umbra
                (true, 1.0 - dist_to_earth / umbra_radius)
            } else if !self.umbra_only && dist_to_earth < penumbra_radius {
                // In penumbra
                let penumbra_depth =
                    (penumbra_radius - dist_to_earth) / (penumbra_radius - umbra_radius);
                (true, 0.5 * penumbra_depth)
            } else {
                (false, 0.0)
            };

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
                    start_time: times[start_idx].to_rfc3339(),
                    end_time: times[i - 1].to_rfc3339(),
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
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity,
                description: if self.umbra_only {
                    "Observer in umbra".to_string()
                } else {
                    "Observer in shadow".to_string()
                },
            });
        }

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(violations, all_satisfied, self.name(), times.to_vec())
    }

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();
        let n_times = times.len();

        // Eclipse is target-independent - compute once for all times
        let time_results = self.compute_eclipse_mask(times, sun_positions, observer_positions);

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
