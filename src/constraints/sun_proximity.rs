/// Sun proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::{dot_product, radec_to_unit_vectors_batch};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use rayon::prelude::*;
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

fn geometric_sun_direction(sun: &Array2<f64>, observer: &Array2<f64>, t: usize) -> [f64; 3] {
    let relative = std::array::from_fn(|j| sun[[t, j]] - observer[[t, j]]);
    let distance = dot_product(&relative, &relative).sqrt();
    relative.map(|v| v / distance)
}

impl SunProximityEvaluator {
    fn mask_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        if target_unit_vectors.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }

        let (times_filtered, sun_filtered, obs_filtered) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);

        let n_targets = target_unit_vectors.nrows();
        let n_times = times_filtered.len();

        let min_cos_threshold = self.min_angle_deg.to_radians().cos();
        let max_cos_threshold = self.max_angle_deg.map(|max| max.to_radians().cos());

        // On an apparent view the catalogue cosine becomes the apparent one.
        if let Some(corrections) = super::core::target_corrections_at(ephemeris, time_indices) {
            let bodies: Vec<[f64; 3]> = (0..n_times)
                .map(|t| {
                    crate::utils::vector_math::normalize_vector(&std::array::from_fn(|j| {
                        sun_filtered[[t, j]] - obs_filtered[[t, j]]
                    }))
                })
                .collect();
            return super::core::apparent_mask(target_unit_vectors, &bodies, &corrections, |_| {
                (min_cos_threshold, max_cos_threshold)
            });
        }

        let direction = |t| geometric_sun_direction(&sun_filtered, &obs_filtered, t);
        if let Some(mask) = super::gpu::try_mask(target_unit_vectors, n_times, |t| {
            (direction(t), min_cos_threshold, max_cos_threshold)
        })? {
            return Ok(mask);
        }
        let mut result = Array2::from_elem((n_targets, n_times), false);
        for t in 0..n_times {
            let sun_unit = direction(t);
            for target_idx in 0..n_targets {
                let cos_angle = target_unit_vectors[[target_idx, 0]] * sun_unit[0]
                    + target_unit_vectors[[target_idx, 1]] * sun_unit[1]
                    + target_unit_vectors[[target_idx, 2]] * sun_unit[2];

                let too_close = cos_angle > min_cos_threshold;
                let too_far = max_cos_threshold.is_some_and(|max_thresh| cos_angle < max_thresh);
                result[[target_idx, t]] = too_close || too_far;
            }
        }

        Ok(result)
    }

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
    fn fused_leaf_grid(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<super::core::LeafValueGrid>> {
        if ephemeris.target_corrections().is_some() {
            return Ok(None);
        }
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let sun = ephemeris.get_sun_positions_at_indices(time_indices)?;
        let observer = ephemeris.get_gcrs_positions_at_indices(time_indices)?;
        let bodies: Vec<_> = (0..sun.nrows())
            .map(|t| geometric_sun_direction(&sun, &observer, t))
            .collect();
        if !bodies.iter().flatten().all(|v| v.is_finite()) {
            // Standalone masks and angles have distinct zero-distance conventions.
            return Ok(None);
        }
        let targets = radec_to_unit_vectors_batch(target_ras, target_decs);
        let upper = self.min_angle_deg.to_radians().cos();
        let lower = self
            .max_angle_deg
            .map(|v| v.to_radians().cos())
            .unwrap_or(f64::NEG_INFINITY);
        let shape = (targets.nrows(), bodies.len());
        let mut grid = super::core::LeafValueGrid {
            key: "sun_angle_deg",
            values: Array2::zeros(shape),
            violated: Array2::from_elem(shape, false),
        };
        if grid.values.is_empty() {
            return Ok(Some(grid));
        }
        if super::gpu::mask_eligible(shape.0, shape.1, super::gpu::ANGULAR_MIN_PAIRS)
            && targets.iter().all(|v| v.is_finite())
            && super::gpu::angle_values(&targets, &bodies, upper, lower, &mut grid)?
        {
            return Ok(Some(grid));
        }
        let row = |(i, (values, mask)): (usize, (&mut [f64], &mut [bool]))| {
            let target = std::array::from_fn(|j| targets[[i, j]]);
            for ((angle, flag), body) in values.iter_mut().zip(mask).zip(&bodies) {
                let dot = dot_product(&target, body);
                *angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
                *flag = dot > upper || dot < lower;
            }
        };
        let values = grid.values.as_slice_mut().unwrap();
        let mask = grid.violated.as_slice_mut().unwrap();
        if values.len() >= super::core::PARALLEL_MIN_GRID_CELLS {
            values
                .par_chunks_mut(shape.1)
                .zip(mask.par_chunks_mut(shape.1))
                .enumerate()
                .for_each(row);
        } else {
            values
                .chunks_mut(shape.1)
                .zip(mask.chunks_mut(shape.1))
                .enumerate()
                .for_each(row);
        }
        Ok(Some(grid))
    }

    fn evaluate_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<super::diagonal::DiagonalEvaluation> {
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        let body =
            ephemeris.data().sun_gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No sun positions available")
            })?;
        let observer = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No GCRS positions available")
        })?;
        Ok(super::diagonal::proximity(
            &self.name(),
            "sun_angle_deg",
            target_ras,
            target_decs,
            &indices,
            body,
            observer,
            self.min_angle_deg,
            self.max_angle_deg,
        ))
    }

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
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let vectors = radec_to_unit_vectors_batch(target_ras, target_decs);
        self.mask_vectors(ephemeris, &vectors, time_indices)
    }

    fn applies_target_corrections(&self) -> bool {
        true
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        self.mask_vectors(ephemeris, target_unit_vectors, time_indices)
            .map(Some)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (_times_filtered, sun_filtered, obs_filtered) =
            extract_standard_ephemeris_data!(ephemeris, time_indices);
        let angle_deg = super::core::compute_angle_deg_batch_on(
            ephemeris,
            time_indices,
            target_ras,
            target_decs,
            &sun_filtered,
            &obs_filtered,
        );
        let mut values = HashMap::new();
        values.insert("sun_angle_deg".to_string(), angle_deg);
        Ok(values)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn ignores_explicit_roll(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
