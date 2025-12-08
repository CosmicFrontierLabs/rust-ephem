/// Generic solar system body proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

impl_proximity_evaluator!(BodyProximityEvaluator, "Body", "body", sun_positions);

impl BodyProximityEvaluator {
    fn final_violation_description(&self) -> String {
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
    }

    fn intermediate_violation_description(&self) -> String {
        format!("Target violates {} proximity constraint", self.body)
    }
}

impl ConstraintEvaluator for BodyProximityEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<ConstraintResult> {
        let (times_slice, sun_positions_slice, observer_positions_slice) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);

        let result = self.evaluate_common(
            &times_slice,
            (target_ra, target_dec),
            &sun_positions_slice,
            &observer_positions_slice,
            || self.final_violation_description(),
            || self.intermediate_violation_description(),
        );
        Ok(result)
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        use crate::utils::vector_math::radec_to_unit_vectors_batch;

        let times = ephemeris.get_times()?;
        let (sun_positions_slice, observer_positions_slice, n_times) =
            if let Some(indices) = time_indices {
                let sun_filtered = ephemeris
                    .get_sun_positions()?
                    .select(ndarray::Axis(0), indices);
                let obs_filtered = ephemeris
                    .get_gcrs_positions()?
                    .select(ndarray::Axis(0), indices);
                (sun_filtered, obs_filtered, indices.len())
            } else {
                let sun_positions = ephemeris.get_sun_positions()?;
                let observer_positions = ephemeris.get_gcrs_positions()?;
                (
                    sun_positions.clone(),
                    observer_positions.clone(),
                    times.len(),
                )
            };
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();

        // Convert all target RA/Dec to unit vectors at once
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        // Initialize result array (false = not violated)
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // Pre-compute threshold (constant for all times)
        let threshold = self.min_angle_deg.to_radians().cos();

        let max_threshold = self.max_angle_deg.map(|max| max.to_radians().cos());

        // For each target
        for (i, target_row) in target_vectors.axis_iter(ndarray::Axis(0)).enumerate() {
            let target_unit = [target_row[0], target_row[1], target_row[2]];

            // Check all times for this target
            for j in 0..n_times {
                let body_pos = [
                    sun_positions_slice[[j, 0]],
                    sun_positions_slice[[j, 1]],
                    sun_positions_slice[[j, 2]],
                ];
                let obs_pos = [
                    observer_positions_slice[[j, 0]],
                    observer_positions_slice[[j, 1]],
                    observer_positions_slice[[j, 2]],
                ];

                // Compute relative body position from observer
                let body_rel = [
                    body_pos[0] - obs_pos[0],
                    body_pos[1] - obs_pos[1],
                    body_pos[2] - obs_pos[2],
                ];
                let body_dist = (body_rel[0] * body_rel[0]
                    + body_rel[1] * body_rel[1]
                    + body_rel[2] * body_rel[2])
                    .sqrt();

                if body_dist == 0.0 {
                    continue;
                }

                let body_unit = [
                    body_rel[0] / body_dist,
                    body_rel[1] / body_dist,
                    body_rel[2] / body_dist,
                ];

                // Dot product
                let cos_angle = target_unit[0] * body_unit[0]
                    + target_unit[1] * body_unit[1]
                    + target_unit[2] * body_unit[2];

                // Check constraints
                let too_close = cos_angle > threshold;
                let too_far = if let Some(max_thresh) = max_threshold {
                    cos_angle < max_thresh
                } else {
                    false
                };

                result[[i, j]] = too_close || too_far;
            }
        }

        Ok(result)
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
