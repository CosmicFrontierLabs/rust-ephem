/// Moon proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Moon proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonProximityConfig {
    /// Minimum allowed angular separation from Moon in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from Moon in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for MoonProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonProximityEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }
}

/// Evaluator for Moon proximity constraint
struct MoonProximityEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl_proximity_evaluator!(MoonProximityEvaluator, "Moon", "Moon", moon_positions);

impl MoonProximityEvaluator {
    fn default_final_violation_description(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to Moon (min: {:.1}°) or too far (max: {:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to Moon (min allowed: {:.1}°)",
                self.min_angle_deg
            ),
        }
    }

    fn default_intermediate_violation_description(&self) -> String {
        "Target violates Moon proximity constraint".to_string()
    }

    fn format_name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!("MoonProximity(min={}°, max={}°)", self.min_angle_deg, max),
            None => format!("MoonProximity(min={}°)", self.min_angle_deg),
        }
    }
}

impl ConstraintEvaluator for MoonProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        self.evaluate_common(
            times,
            (target_ra, target_dec),
            moon_positions,
            observer_positions,
            || self.default_final_violation_description(),
            || self.default_intermediate_violation_description(),
        )
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        _sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
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
            let moon_pos = [
                moon_positions[[t, 0]],
                moon_positions[[t, 1]],
                moon_positions[[t, 2]],
            ];
            let obs_pos = [
                observer_positions[[t, 0]],
                observer_positions[[t, 1]],
                observer_positions[[t, 2]],
            ];

            // Compute relative moon position from observer
            let moon_rel = [
                moon_pos[0] - obs_pos[0],
                moon_pos[1] - obs_pos[1],
                moon_pos[2] - obs_pos[2],
            ];
            let moon_dist =
                (moon_rel[0] * moon_rel[0] + moon_rel[1] * moon_rel[1] + moon_rel[2] * moon_rel[2])
                    .sqrt();
            let moon_unit = [
                moon_rel[0] / moon_dist,
                moon_rel[1] / moon_dist,
                moon_rel[2] / moon_dist,
            ];

            // Check all targets at this time
            for target_idx in 0..n_targets {
                let target_vec = [
                    target_vectors[[target_idx, 0]],
                    target_vectors[[target_idx, 1]],
                    target_vectors[[target_idx, 2]],
                ];

                // Calculate angle between target and moon
                let dot = target_vec[0] * moon_unit[0]
                    + target_vec[1] * moon_unit[1]
                    + target_vec[2] * moon_unit[2];
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
