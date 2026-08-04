use crate::constraints::core::{track_violations, ConstraintEvaluator, ConstraintResult};
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::FileEphemeris;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::ParquetEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use pyo3::prelude::*;
use std::collections::HashMap;

use super::PyConstraint;
use crate::constraints::constraint_wrapper::field_of_regard::DEFAULT_N_ROLL_SAMPLES;
use crate::constraints::constraint_wrapper::json_parser::parse_constraint_json;

impl PyConstraint {
    pub(super) fn resolve_time_indices(
        &self,
        bound: &Bound<'_, PyAny>,
        times: Option<&Bound<'_, PyAny>>,
        indices: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Vec<usize>>> {
        if let Some(times_arg) = times {
            if indices.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both 'times' and 'indices' parameters",
                ));
            }
            Ok(Some(self.parse_times_to_indices(bound, times_arg)?))
        } else if let Some(indices_arg) = indices {
            Ok(Some(self.parse_indices(indices_arg)?))
        } else {
            Ok(None)
        }
    }

    pub(super) fn build_roll_groups(
        rolls: &[f64],
    ) -> std::collections::BTreeMap<String, Vec<usize>> {
        let mut roll_map: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, roll) in rolls.iter().enumerate() {
            let key = format!("{}", roll);
            roll_map.entry(key).or_default().push(idx);
        }
        roll_map
    }

    pub(super) fn extract_group_targets(
        target_ras: &[f64],
        target_decs: &[f64],
        indices: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        (
            indices.iter().map(|&i| target_ras[i]).collect(),
            indices.iter().map(|&i| target_decs[i]).collect(),
        )
    }

    pub(super) fn reassemble_grouped_batch_results(
        n_targets: usize,
        n_times: usize,
        all_groups: Vec<(Vec<usize>, ndarray::Array2<bool>)>,
    ) -> PyResult<ndarray::Array2<bool>> {
        let mut final_results: Vec<Vec<bool>> = vec![vec![false; n_times]; n_targets];

        for (group_indices, group_array) in all_groups {
            for (row_in_group, &orig_idx) in group_indices.iter().enumerate() {
                for col in 0..n_times {
                    final_results[orig_idx][col] = group_array[[row_in_group, col]];
                }
            }
        }

        ndarray::Array2::from_shape_vec(
            (n_targets, n_times),
            final_results.into_iter().flatten().collect(),
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to create array: {}", e))
        })
    }

    pub(super) fn with_ephemeris<T, F>(&self, bound: &Bound<'_, PyAny>, mut f: F) -> PyResult<T>
    where
        F: FnMut(&dyn EphemerisBase) -> PyResult<T>,
    {
        if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }
        if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }
        if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }
        if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }
        if let Ok(ephem) = bound.extract::<PyRef<FileEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }
        if let Ok(ephem) = bound.extract::<PyRef<ParquetEphemeris>>() {
            return f(&*ephem as &dyn EphemerisBase);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, OEMEphemeris, FileEphemeris, or ParquetEphemeris",
        ))
    }

    pub(super) fn inject_solar_roll(config: &mut serde_json::Value, roll_deg: f64) {
        let Some(obj) = config.as_object_mut() else {
            return;
        };

        if obj.get("type").and_then(|v| v.as_str()) == Some("solar_roll") {
            obj.insert("roll_deg".to_string(), serde_json::json!(roll_deg));
        }

        if let Some(inner) = obj.get_mut("constraint") {
            Self::inject_solar_roll(inner, roll_deg);
        }

        if let Some(children) = obj.get_mut("constraints").and_then(|v| v.as_array_mut()) {
            for child in children {
                Self::inject_solar_roll(child, roll_deg);
            }
        }
    }

    pub(super) fn in_constraint_batch_with_roll_sweep(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
    ) -> PyResult<ndarray::Array2<bool>> {
        evaluator.in_constraint_batch_constrained_at_every_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            n_roll_samples,
        )
    }

    /// Slice out a single target's row from each named-value matrix returned by
    /// `compute_named_values` (shape M x N, targets x times).
    pub(super) fn extract_target_values(
        values_map: &HashMap<String, ndarray::Array2<f64>>,
        target_index: usize,
    ) -> HashMap<String, Vec<f64>> {
        values_map
            .iter()
            .map(|(key, arr)| {
                let row: Vec<f64> = (0..arr.ncols()).map(|i| arr[[target_index, i]]).collect();
                (key.clone(), row)
            })
            .collect()
    }

    /// Slice out a single target's row from each named-boolean matrix returned by
    /// `compute_named_booleans` (shape M x N, targets x times).
    pub(super) fn extract_target_booleans(
        booleans_map: &HashMap<String, ndarray::Array2<bool>>,
        target_index: usize,
    ) -> HashMap<String, Vec<bool>> {
        booleans_map
            .iter()
            .map(|(key, arr)| {
                let row: Vec<bool> = (0..arr.ncols()).map(|i| arr[[target_index, i]]).collect();
                (key.clone(), row)
            })
            .collect()
    }

    pub(super) fn with_effective_evaluator<T, F>(
        &self,
        target_roll: Option<f64>,
        f: F,
    ) -> PyResult<T>
    where
        F: FnOnce(&dyn ConstraintEvaluator) -> PyResult<T>,
    {
        let Some(target_roll_deg) = target_roll else {
            return f(&*self.evaluator);
        };

        if !target_roll_deg.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_roll must be a finite number",
            ));
        }

        let mut config: serde_json::Value = serde_json::from_str(&self.config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Self::inject_solar_roll(&mut config, target_roll_deg);

        let constraint_type = config
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        let is_boresight_offset = constraint_type == "boresight_offset";
        let is_bright_star = constraint_type == "bright_star";
        let is_body_polygon = constraint_type == "body" && config.get("fov_polygon").is_some();
        let is_solar_roll = constraint_type == "solar_roll";

        // Bright star or body proximity with a polygon FoV: inject target_roll as roll_deg
        // so the evaluator rotates the polygon to the requested angle.  Both constraint types
        // handle roll internally, so we bypass the BoresightOffset wrapper.
        if is_bright_star || is_body_polygon {
            if config.get("fov_polygon").is_some() {
                if let Some(obj) = config.as_object_mut() {
                    obj.insert("roll_deg".to_string(), serde_json::json!(target_roll_deg));
                }
            }
            let evaluator = parse_constraint_json(&config)?;
            return f(&*evaluator);
        }

        // SolarRoll: inject the spacecraft roll so the evaluator can compare to the
        // solar-optimal roll.  Handled internally — bypass the BoresightOffset wrapper.
        if is_solar_roll {
            let evaluator = parse_constraint_json(&config)?;
            return f(&*evaluator);
        }

        if is_boresight_offset {
            let base_roll_deg = config
                .get("roll_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let base_clockwise = config
                .get("roll_clockwise")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Match RustConstraintMixin semantics: add evaluation-time roll in the
            // configured command convention.
            let signed_target_roll = if base_clockwise {
                -target_roll_deg
            } else {
                target_roll_deg
            };

            if let Some(obj) = config.as_object_mut() {
                obj.insert(
                    "roll_deg".to_string(),
                    serde_json::json!(base_roll_deg + signed_target_roll),
                );
            }
        } else {
            config = serde_json::json!({
                "type": "boresight_offset",
                "constraint": config,
                "roll_deg": target_roll_deg,
                "roll_clockwise": false,
                "roll_reference": "north",
                "pitch_deg": 0.0,
                "yaw_deg": 0.0
            });
        }

        let evaluator = parse_constraint_json(&config)?;
        f(&*evaluator)
    }

    /// Internal helper to evaluate against any Ephemeris implementing EphemerisBase
    #[allow(deprecated)]
    pub(super) fn eval_with_ephemeris(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<ConstraintResult> {
        // PERFORMANCE OPTIMIZATION: Use fast batch path internally
        // Instead of the slow evaluate() that tracks violations step-by-step,
        // use in_constraint_batch() which is 1700x faster, then construct violations from the result

        // Call the fast batch evaluation for single target
        let violation_array = self.in_constraint_batch_with_roll_sweep(
            evaluator,
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices.as_deref(),
            DEFAULT_N_ROLL_SAMPLES,
        )?;

        // Get the times we evaluated
        let all_times = ephemeris.get_times()?;
        let times: Vec<_> = if let Some(ref indices) = time_indices {
            indices.iter().map(|&i| all_times[i]).collect()
        } else {
            all_times.to_vec()
        };

        // Extract the boolean array for our single target (first row)
        // Note: in_constraint_batch now consistently returns true when VIOLATED (matches track_violations)
        let violated: Vec<bool> = (0..violation_array.ncols())
            .map(|i| violation_array[[0, i]])
            .collect();

        // Track violations using the same helper function
        let violations = track_violations(
            &times,
            |i| (violated[i], if violated[i] { 1.0 } else { 0.0 }),
            |_i, _is_open| evaluator.name(),
        );

        let all_satisfied = violations.is_empty();

        // Compute named continuous values once (not swept across roll angles).
        let values_map = evaluator.compute_named_values(
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices.as_deref(),
        )?;
        let values = Self::extract_target_values(&values_map, 0);

        // Compute per-leaf-constraint violation masks once, used to attribute
        // VisibilityWindow.start_cause/end_cause.
        let booleans_map = evaluator.compute_named_booleans(
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices.as_deref(),
        )?;
        let component_violated = Self::extract_target_booleans(&booleans_map, 0);

        // Map cause tags to their constraint_values key(s). Only key *names* matter
        // here (not the actual data), so this evaluates at a single dummy target/time
        // rather than paying for a full recompute just to introspect key structure.
        let cause_value_keys =
            evaluator.compute_cause_value_keys(ephemeris, &[0.0], &[0.0], Some(&[0]))?;

        Ok(
            ConstraintResult::new(violations, all_satisfied, evaluator.name(), times)
                .with_constraint_values(values)
                .with_component_violated(component_violated)
                .with_cause_value_keys(cause_value_keys),
        )
    }

    pub(super) fn eval_batch_with_ephemeris(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<Vec<ConstraintResult>> {
        let violation_array = self.in_constraint_batch_with_roll_sweep(
            evaluator,
            ephemeris,
            target_ras,
            target_decs,
            time_indices.as_deref(),
            DEFAULT_N_ROLL_SAMPLES,
        )?;

        let all_times = ephemeris.get_times()?;
        let times: Vec<_> = if let Some(ref indices) = time_indices {
            indices.iter().map(|&i| all_times[i]).collect()
        } else {
            all_times.to_vec()
        };

        // Compute named continuous values once for all targets (not swept across roll angles).
        let values_map = evaluator.compute_named_values(
            ephemeris,
            target_ras,
            target_decs,
            time_indices.as_deref(),
        )?;

        // Compute per-leaf-constraint violation masks once for all targets, used to
        // attribute VisibilityWindow.start_cause/end_cause.
        let booleans_map = evaluator.compute_named_booleans(
            ephemeris,
            target_ras,
            target_decs,
            time_indices.as_deref(),
        )?;

        // Map cause tags to their constraint_values key(s), shared across all targets
        // (the mapping only depends on the evaluator's structure, not the target
        // positions). Only key *names* matter, so this uses a dummy target/time
        // rather than paying for a full recompute just to introspect key structure.
        let cause_value_keys =
            evaluator.compute_cause_value_keys(ephemeris, &[0.0], &[0.0], Some(&[0]))?;

        let mut results = Vec::with_capacity(target_ras.len());
        for target_index in 0..target_ras.len() {
            let violated: Vec<bool> = (0..violation_array.ncols())
                .map(|i| violation_array[[target_index, i]])
                .collect();

            let violations = track_violations(
                &times,
                |i| (violated[i], if violated[i] { 1.0 } else { 0.0 }),
                |_i, _is_open| evaluator.name(),
            );

            let all_satisfied = violations.is_empty();
            let values = Self::extract_target_values(&values_map, target_index);
            let component_violated = Self::extract_target_booleans(&booleans_map, target_index);
            results.push(
                ConstraintResult::new(violations, all_satisfied, evaluator.name(), times.clone())
                    .with_constraint_values(values)
                    .with_component_violated(component_violated)
                    .with_cause_value_keys(cause_value_keys.clone()),
            );
        }

        Ok(results)
    }

    /// Vectorized evaluation for moving bodies - evaluates all targets at their corresponding times
    ///
    /// For N targets at N times, this calls in_constraint_batch once with all N targets
    /// Uses the efficient diagonal batch evaluation for moving bodies.
    /// Each target_i is evaluated only at time_i, which is O(N) instead of O(N²).
    pub(super) fn eval_moving_body_batch_diagonal(
        &self,
        py: Python,
        ephemeris: &Py<PyAny>,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        let n = target_ras.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let bound = ephemeris.bind(py);

        // Use the efficient diagonal batch evaluation
        if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<FileEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<ParquetEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, OEMEphemeris, FileEphemeris, or ParquetEphemeris",
            ))
        }
    }
}
