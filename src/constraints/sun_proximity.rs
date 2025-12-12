/// Sun proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

impl_proximity_evaluator!(SunProximityEvaluator, "Sun", "Sun", sun_positions);

impl SunProximityEvaluator {
    #[allow(dead_code)]
    fn default_final_violation_description(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to Sun (min: {:.1}°) or too far (max: {:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to Sun (min allowed: {:.1}°)",
                self.min_angle_deg
            ),
        }
    }

    #[allow(dead_code)]
    fn default_intermediate_violation_description(&self) -> String {
        "Target violates Sun proximity constraint".to_string()
    }

    fn format_name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!("SunProximity(min={}°, max={}°)", self.min_angle_deg, max),
            None => format!("SunProximity(min={}°)", self.min_angle_deg),
        }
    }
}

impl ConstraintEvaluator for SunProximityEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let (times_filtered, sun_filtered, obs_filtered) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);

        Ok(self.evaluate_common(
            &times_filtered,
            (target_ra, target_dec),
            &sun_filtered,
            &obs_filtered,
            || self.default_final_violation_description(),
            || self.default_intermediate_violation_description(),
        ))
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
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
        // Validate inputs
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();

        // Convert all target RA/Dec to unit vectors at once
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        // Initialize result array: false = not violated (constraint satisfied)
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // For each time, check all targets
        for t in 0..n_times {
            let sun_pos = [
                sun_filtered[[t, 0]],
                sun_filtered[[t, 1]],
                sun_filtered[[t, 2]],
            ];
            let obs_pos = [
                obs_filtered[[t, 0]],
                obs_filtered[[t, 1]],
                obs_filtered[[t, 2]],
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

    /// Optimized diagonal evaluation for moving bodies - O(N) instead of O(N²)
    ///
    /// For moving bodies, we only need target_i at time_i (diagonal of M×N matrix).
    /// This avoids allocating and computing the full N×N matrix.
    fn in_constraint_batch_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        let n = target_ras.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Get ephemeris data
        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        // Ensure we have enough time steps
        if sun_positions.nrows() < n || observer_positions.nrows() < n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Not enough ephemeris time steps for diagonal evaluation",
            ));
        }

        // Convert all target RA/Dec to unit vectors
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        // Evaluate only diagonal elements: target_i at time_i
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
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

            // Get target vector at this index
            let target_vec = [
                target_vectors[[i, 0]],
                target_vectors[[i, 1]],
                target_vectors[[i, 2]],
            ];

            // Calculate angle between target and sun
            let dot = target_vec[0] * sun_unit[0]
                + target_vec[1] * sun_unit[1]
                + target_vec[2] * sun_unit[2];
            let dot_clamped = dot.clamp(-1.0, 1.0);
            let angle_deg = dot_clamped.acos().to_degrees();

            // Check if violated
            let is_violated = angle_deg < self.min_angle_deg
                || self.max_angle_deg.is_some_and(|max| angle_deg > max);

            result.push(is_violated);
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
