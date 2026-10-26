use crate::constraints::core::{
    sweep_rolls_with_attribution, track_violations, ConstraintEvaluator, ConstraintResult,
    EvaluationRoll,
};
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

    pub(super) fn validate_target_roll(roll: f64) -> PyResult<()> {
        if !roll.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_roll must be a finite number",
            ));
        }
        Ok(())
    }

    pub(super) fn build_roll_groups(
        rolls: &[f64],
    ) -> PyResult<std::collections::BTreeMap<u64, Vec<usize>>> {
        let mut groups = std::collections::BTreeMap::new();
        for (idx, &roll) in rolls.iter().enumerate() {
            Self::validate_target_roll(roll)?;
            groups
                .entry(roll.to_bits())
                .or_insert_with(Vec::new)
                .push(idx);
        }
        Ok(groups)
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

    /// Reject a zero roll-sample count before it reaches a sweep loop, where it
    /// would divide by zero computing the roll step and silently produce a
    /// zero-iteration sweep.
    pub(super) fn validate_n_roll_samples(n_roll_samples: usize) -> PyResult<()> {
        if n_roll_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_roll_samples must be greater than 0",
            ));
        }
        Ok(())
    }

    /// Slice out a single target's row from each named matrix keyed by name (shape
    /// M x N, targets x times) — the per-target values from `compute_named_values`
    /// and the per-leaf cause masks from the sweep are both shaped this way.
    pub(super) fn extract_target_row<T: Copy>(
        matrices: &HashMap<String, ndarray::Array2<T>>,
        target_index: usize,
    ) -> HashMap<String, Vec<T>> {
        matrices
            .iter()
            .map(|(key, arr)| {
                let row: Vec<T> = (0..arr.ncols()).map(|i| arr[[target_index, i]]).collect();
                (key.clone(), row)
            })
            .collect()
    }

    /// Cause tag → `constraint_values` key(s) for this evaluator's tree.
    ///
    /// Only key *names* are wanted, and those depend on the evaluator's structure
    /// rather than on the target position or time, so this probes a single dummy
    /// target/time instead of paying for a full recompute just to introspect key
    /// structure.
    pub(super) fn cause_value_keys_for(
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        evaluator.compute_cause_value_keys(ephemeris, &[0.0], &[0.0], Some(&[0]))
    }

    /// Internal helper to evaluate against any Ephemeris implementing EphemerisBase
    #[allow(deprecated)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_with_ephemeris(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<Vec<usize>>,
        n_roll_samples: usize,
        target_roll: Option<f64>,
    ) -> PyResult<ConstraintResult> {
        // PERFORMANCE OPTIMIZATION: Use fast batch path internally
        // Instead of the slow evaluate() that tracks violations step-by-step,
        // use in_constraint_batch() which is 1700x faster, then construct violations from the result

        // One coordinated free-roll sweep yields both the violation array and the
        // per-leaf cause masks, taken at the same witness roll per timestamp, so the
        // reported cause always describes an orientation consistent with the
        // visibility it explains.  A no-op single evaluation when the tree isn't
        // roll-dependent.
        let swept = if let Some(roll) = target_roll {
            evaluator.in_constraint_batch_with_causes(
                ephemeris,
                &[target_ra],
                &[target_dec],
                time_indices.as_deref(),
                Some(EvaluationRoll::Fixed(roll)),
            )?
        } else {
            sweep_rolls_with_attribution(
                evaluator,
                ephemeris,
                &[target_ra],
                &[target_dec],
                time_indices.as_deref(),
                n_roll_samples,
            )?
        };
        let violation_array = swept.violated;

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
            |_i, _is_open| evaluator.name_at_roll(target_roll),
        );

        let all_satisfied = violations.is_empty();

        // Compute named continuous values once (not swept across roll angles).
        let values_map = evaluator.compute_named_values_at_roll(
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices.as_deref(),
            target_roll,
        )?;
        let values = Self::extract_target_row(&values_map, 0);

        // Per-leaf violation masks used to attribute
        // VisibilityWindow.start_cause/end_cause, from the same sweep that produced
        // `violation_array` above.
        let component_violated = Self::extract_target_row(&swept.named, 0);
        let cause_value_keys = Self::cause_value_keys_for(evaluator, ephemeris)?;

        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            evaluator.name_at_roll(target_roll),
            times,
        )
        .with_constraint_values(values)
        .with_component_violated(component_violated)
        .with_cause_value_keys(cause_value_keys))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_batch_with_ephemeris(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<Vec<usize>>,
        n_roll_samples: usize,
        target_roll: Option<f64>,
    ) -> PyResult<Vec<ConstraintResult>> {
        // See `eval_with_ephemeris`: one coordinated sweep produces both the
        // violation array and the cause masks consistent with it.
        let swept = if let Some(roll) = target_roll {
            evaluator.in_constraint_batch_with_causes(
                ephemeris,
                target_ras,
                target_decs,
                time_indices.as_deref(),
                Some(EvaluationRoll::Fixed(roll)),
            )?
        } else {
            sweep_rolls_with_attribution(
                evaluator,
                ephemeris,
                target_ras,
                target_decs,
                time_indices.as_deref(),
                n_roll_samples,
            )?
        };
        let violation_array = swept.violated;

        let all_times = ephemeris.get_times()?;
        let times: Vec<_> = if let Some(ref indices) = time_indices {
            indices.iter().map(|&i| all_times[i]).collect()
        } else {
            all_times.to_vec()
        };

        // Compute named continuous values once for all targets (not swept across roll angles).
        let values_map = evaluator.compute_named_values_at_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices.as_deref(),
            target_roll,
        )?;

        // Shared across all targets: the mapping depends only on the evaluator's
        // structure, not on the target positions.
        let cause_value_keys = Self::cause_value_keys_for(evaluator, ephemeris)?;

        let mut results = Vec::with_capacity(target_ras.len());
        for target_index in 0..target_ras.len() {
            let violated: Vec<bool> = (0..violation_array.ncols())
                .map(|i| violation_array[[target_index, i]])
                .collect();

            let violations = track_violations(
                &times,
                |i| (violated[i], if violated[i] { 1.0 } else { 0.0 }),
                |_i, _is_open| evaluator.name_at_roll(target_roll),
            );

            let all_satisfied = violations.is_empty();
            let values = Self::extract_target_row(&values_map, target_index);
            let component_violated = Self::extract_target_row(&swept.named, target_index);
            results.push(
                ConstraintResult::new(
                    violations,
                    all_satisfied,
                    evaluator.name_at_roll(target_roll),
                    times.clone(),
                )
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
