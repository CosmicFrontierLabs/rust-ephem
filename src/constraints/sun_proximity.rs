/// Sun proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::{calculate_angular_separation, radec_to_unit_vector};
use chrono::{DateTime, Utc};
use ndarray::Array2;
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

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
