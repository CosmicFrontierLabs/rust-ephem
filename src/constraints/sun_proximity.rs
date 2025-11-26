/// Sun proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::{
    calculate_angular_separation, radec_to_unit_vector, radec_to_unit_vectors_batch,
};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Sun proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunProximityConfig {
    /// Minimum allowed angular separation from Sun in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from Sun in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for SunProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(SunProximityEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }
}

/// Evaluator for Sun proximity constraint
struct SunProximityEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl_proximity_evaluator_helpers!(SunProximityEvaluator, "Sun", "Sun");

impl ConstraintEvaluator for SunProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Cache target vector computation outside the loop
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            times,
            |i| {
                let sun_pos = [
                    sun_positions[[i, 0]],
                    sun_positions[[i, 1]],
                    sun_positions[[i, 2]],
                ];
                let obs_pos = [
                    observer_positions[[i, 0]],
                    observer_positions[[i, 1]],
                    observer_positions[[i, 2]],
                ];
                let angle_deg = calculate_angular_separation(&target_vec, &sun_pos, &obs_pos);

                let is_violated = angle_deg < self.min_angle_deg
                    || self.max_angle_deg.is_some_and(|max| angle_deg > max);

                let severity = if angle_deg < self.min_angle_deg {
                    (self.min_angle_deg - angle_deg) / self.min_angle_deg
                } else if let Some(max) = self.max_angle_deg {
                    (angle_deg - max) / max
                } else {
                    0.0
                };

                (is_violated, severity)
            },
            |_, is_final| {
                if is_final {
                    self.final_violation_description()
                } else {
                    // Get angle for description (recompute at violation end)
                    "Target violates Sun proximity constraint".to_string()
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(violations, all_satisfied, self.name(), times.to_vec())
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        // Validate inputs
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();
        let n_times = times.len();

        // Convert all target RA/Dec to unit vectors at once
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        // Initialize result array: false = not violated (constraint satisfied)
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // For each time, check all targets
        for t in 0..n_times {
            let sun_pos = [
                sun_positions[[t, 0]],
                sun_positions[[t, 1]],
                sun_positions[[t, 2]],
            ];
            let obs_pos = [
                observer_positions[[t, 0]],
                observer_positions[[t, 1]],
                observer_positions[[t, 2]],
            ];

            // Compute relative sun position from observer
            let sun_rel = [
                sun_pos[0] - obs_pos[0],
                sun_pos[1] - obs_pos[1],
                sun_pos[2] - obs_pos[2],
            ];
            let sun_dist =
                (sun_rel[0] * sun_rel[0] + sun_rel[1] * sun_rel[1] + sun_rel[2] * sun_rel[2])
                    .sqrt();
            let sun_unit = [
                sun_rel[0] / sun_dist,
                sun_rel[1] / sun_dist,
                sun_rel[2] / sun_dist,
            ];

            // Check all targets at this time
            for target_idx in 0..n_targets {
                let target_vec = [
                    target_vectors[[target_idx, 0]],
                    target_vectors[[target_idx, 1]],
                    target_vectors[[target_idx, 2]],
                ];

                // Calculate angle between target and sun
                let dot = target_vec[0] * sun_unit[0]
                    + target_vec[1] * sun_unit[1]
                    + target_vec[2] * sun_unit[2];
                let dot_clamped = dot.clamp(-1.0, 1.0);
                let angle_rad = dot_clamped.acos();
                let angle_deg = angle_rad.to_degrees();

                // Check if violated
                let is_violated = angle_deg < self.min_angle_deg
                    || self.max_angle_deg.is_some_and(|max| angle_deg > max);

                result[[target_idx, t]] = is_violated;
            }
        }

        Ok(result)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
