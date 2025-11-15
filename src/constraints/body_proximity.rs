/// Generic solar system body proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::{calculate_angular_separation, radec_to_unit_vector};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for generic solar system body proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyProximityConfig {
    /// Body identifier (NAIF ID or name, e.g., "Jupiter", "499")
    pub body: String,
    /// Minimum allowed angular separation in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for BodyProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(BodyProximityEvaluator {
            body: self.body.clone(),
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }
}

/// Evaluator for generic body proximity - requires body positions computed externally
pub struct BodyProximityEvaluator {
    pub body: String,
    pub min_angle_deg: f64,
    pub max_angle_deg: Option<f64>,
}

impl ConstraintEvaluator for BodyProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Body positions are passed via sun_positions slot
        let body_positions = sun_positions;
        // Cache target vector computation outside the loop
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            times,
            |i| {
                let body_pos = [
                    body_positions[[i, 0]],
                    body_positions[[i, 1]],
                    body_positions[[i, 2]],
                ];
                let obs_pos = [
                    observer_positions[[i, 0]],
                    observer_positions[[i, 1]],
                    observer_positions[[i, 2]],
                ];
                let angle_deg = calculate_angular_separation(&target_vec, &body_pos, &obs_pos);

                let is_violation = if let Some(max_angle) = self.max_angle_deg {
                    angle_deg < self.min_angle_deg || angle_deg > max_angle
                } else {
                    angle_deg < self.min_angle_deg
                };

                let severity = if angle_deg < self.min_angle_deg {
                    (self.min_angle_deg - angle_deg) / self.min_angle_deg
                } else if let Some(max_angle) = self.max_angle_deg {
                    (angle_deg - max_angle) / max_angle.max(1e-9)
                } else {
                    0.0
                };

                (is_violation, severity)
            },
            |_, is_final| {
                if is_final {
                    match self.max_angle_deg {
                        Some(max) => format!(
                            "Target too close to {} (min: {:.1}°, max: {:.1}°)",
                            self.body, self.min_angle_deg, max
                        ),
                        None => format!(
                            "Target too close to {} (min allowed: {:.1}°)",
                            self.body, self.min_angle_deg
                        ),
                    }
                } else {
                    format!("Target violates {} proximity constraint", self.body)
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(violations, all_satisfied, self.name(), times.to_vec())
    }

    fn name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "BodyProximity(body='{}', min={}°, max={}°)",
                self.body, self.min_angle_deg, max
            ),
            None => format!(
                "BodyProximity(body='{}', min={}°)",
                self.body, self.min_angle_deg
            ),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
