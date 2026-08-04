// Logical combinator evaluators
use crate::constraints::core::{
    track_violations, value_key_prefix, ConstraintEvaluator, ConstraintResult, ConstraintViolation,
};
use crate::utils::vector_math::unit_vectors_to_radec_batch;
use ndarray::Array2;
use pyo3::PyResult;
use std::collections::HashMap;

/// Merge named-value maps from a set of child evaluators, namespacing each child's
/// keys under a short type tag (e.g. `sun.sun_angle_deg`) so composite constraints
/// don't lose per-child detail or silently collide on identical key names. On tag
/// collision (two children of the same type), the 2nd+ occurrence gets a numeric
/// suffix (`sun_2`, `sun_3`, ...).
fn merge_children_named_values(
    children: &[Box<dyn ConstraintEvaluator>],
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_ras: &[f64],
    target_decs: &[f64],
    time_indices: Option<&[usize]>,
) -> PyResult<HashMap<String, Array2<f64>>> {
    let mut merged = HashMap::new();
    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for child in children {
        let child_values =
            child.compute_named_values(ephemeris, target_ras, target_decs, time_indices)?;
        if child_values.is_empty() {
            continue;
        }

        let base_tag = value_key_prefix(&child.name());
        let count = tag_counts.entry(base_tag.clone()).or_insert(0);
        *count += 1;
        let prefix = if *count == 1 {
            base_tag
        } else {
            format!("{base_tag}_{count}")
        };

        for (key, arr) in child_values {
            merged.insert(format!("{prefix}.{key}"), arr);
        }
    }

    Ok(merged)
}

/// Insert `(key, value)` into `merged`, renaming on collision by appending `_2`,
/// `_3`, ... until a free slot is found. Checking against `merged` itself (rather
/// than a separate per-original-key counter) guarantees no entry is ever silently
/// discarded via `HashMap::insert`, even when a nested child's own merge already
/// produced a suffixed key (e.g. `sun_2`) that a sibling's rename would otherwise
/// collide with.
fn insert_no_collision<V>(merged: &mut HashMap<String, V>, key: String, value: V) {
    let mut final_key = key.clone();
    let mut suffix = 2;
    while merged.contains_key(&final_key) {
        final_key = format!("{key}_{suffix}");
        suffix += 1;
    }
    merged.insert(final_key, value);
}

/// Drain a child's own returned key/value pairs in key-sorted order. `HashMap`
/// iteration order is otherwise unspecified (randomized per-instance), which would
/// make `insert_no_collision`'s renaming choice for same-named entries within a
/// single child's map nondeterministic across calls. Sorting first makes the
/// resulting key strings reproducible, which `compute_cause_value_keys` relies on
/// to stay in lockstep with this merge's actual output.
fn sorted_entries<V>(map: HashMap<String, V>) -> Vec<(String, V)> {
    let mut entries: Vec<(String, V)> = map.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Merge per-leaf violation masks from a set of child evaluators, flattening each
/// child's own tag(s) (already namespaced by `compute_named_booleans` — e.g. `sun`
/// for a leaf, or `moon`/`sun_2` for an already-merged nested combinator) into one
/// map. On tag collision between sibling children, the 2nd+ occurrence gets a
/// numeric suffix (`sun_2`, `sun_3`, ...); see `insert_no_collision`.
fn merge_children_named_booleans(
    children: &[Box<dyn ConstraintEvaluator>],
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_ras: &[f64],
    target_decs: &[f64],
    time_indices: Option<&[usize]>,
) -> PyResult<HashMap<String, Array2<bool>>> {
    let mut merged = HashMap::new();

    for child in children {
        let child_booleans =
            child.compute_named_booleans(ephemeris, target_ras, target_decs, time_indices)?;
        for (key, arr) in sorted_entries(child_booleans) {
            insert_no_collision(&mut merged, key, arr);
        }
    }

    Ok(merged)
}

/// Diagonal variant of `merge_children_named_booleans` for moving-body evaluation,
/// merging each child's `compute_named_booleans_diagonal` (target_i paired with
/// time_i) instead of the full M×N batch.
fn merge_children_named_booleans_diagonal(
    children: &[Box<dyn ConstraintEvaluator>],
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_ras: &[f64],
    target_decs: &[f64],
) -> PyResult<HashMap<String, Vec<bool>>> {
    let mut merged = HashMap::new();

    for child in children {
        let child_booleans =
            child.compute_named_booleans_diagonal(ephemeris, target_ras, target_decs)?;
        for (key, arr) in sorted_entries(child_booleans) {
            insert_no_collision(&mut merged, key, arr);
        }
    }

    Ok(merged)
}

/// Merge per-leaf cause-tag → `constraint_values`-key mapping from a set of child
/// evaluators, mirroring `merge_children_named_booleans`' cause-tag identity/
/// collision-renaming and `merge_children_named_values`' per-child value-key
/// prefixing *in lockstep*, so a caller can look up which `constraint_values`
/// key(s) a given cause tag corresponds to without guessing from string prefixes
/// (which don't match — cause tags are flat/nesting-stable, value keys are
/// hierarchical/path-based; see `ConstraintResult.cause_value_keys`).
fn merge_children_cause_value_keys(
    children: &[Box<dyn ConstraintEvaluator>],
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_ras: &[f64],
    target_decs: &[f64],
    time_indices: Option<&[usize]>,
) -> PyResult<HashMap<String, Vec<String>>> {
    let mut merged: HashMap<String, Vec<String>> = HashMap::new();
    let mut value_tag_counts: HashMap<String, usize> = HashMap::new();

    for child in children {
        let child_map =
            child.compute_cause_value_keys(ephemeris, target_ras, target_decs, time_indices)?;

        // A child with no named values anywhere in its subtree (e.g. a bare SAA
        // leaf) doesn't consume a numbering slot here, matching
        // merge_children_named_values' `if child_values.is_empty() { continue; }`
        // skip — but its cause tag(s) must still surface, just with an empty
        // value-key list, so `cause_value_keys` never silently drops a tag.
        let has_any_values = child_map.values().any(|v| !v.is_empty());
        let value_prefix = if has_any_values {
            let base_tag = value_key_prefix(&child.name());
            let count = value_tag_counts.entry(base_tag.clone()).or_insert(0);
            *count += 1;
            Some(if *count == 1 {
                base_tag
            } else {
                format!("{base_tag}_{count}")
            })
        } else {
            None
        };

        for (cause_tag, value_key_names) in sorted_entries(child_map) {
            let prefixed_value_keys: Vec<String> = match &value_prefix {
                Some(prefix) => value_key_names
                    .into_iter()
                    .map(|k| format!("{prefix}.{k}"))
                    .collect(),
                None => value_key_names,
            };
            // Mirrors merge_children_named_booleans' cause-tag collision renaming.
            insert_no_collision(&mut merged, cause_tag, prefixed_value_keys);
        }
    }

    Ok(merged)
}

fn validate_unit_vector_shape(target_unit_vectors: &Array2<f64>) -> PyResult<()> {
    if target_unit_vectors.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_unit_vectors must have shape (N, 3)",
        ));
    }
    Ok(())
}

fn eval_constraint_batch_from_unit_vectors(
    constraint: &dyn ConstraintEvaluator,
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_unit_vectors: &Array2<f64>,
    time_indices: Option<&[usize]>,
) -> PyResult<Array2<bool>> {
    validate_unit_vector_shape(target_unit_vectors)?;

    if let Some(result) =
        constraint.in_constraint_batch_unit_vectors(ephemeris, target_unit_vectors, time_indices)?
    {
        return Ok(result);
    }

    let (target_ras, target_decs) = unit_vectors_to_radec_batch(target_unit_vectors);
    constraint.in_constraint_batch(ephemeris, &target_ras, &target_decs, time_indices)
}

fn eval_constraints_batch_from_unit_vectors(
    constraints: &[Box<dyn ConstraintEvaluator>],
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_unit_vectors: &Array2<f64>,
    time_indices: Option<&[usize]>,
) -> PyResult<Vec<Array2<bool>>> {
    validate_unit_vector_shape(target_unit_vectors)?;

    let mut fallback_radec: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut results = Vec::with_capacity(constraints.len());

    for constraint in constraints {
        if let Some(result) = constraint.in_constraint_batch_unit_vectors(
            ephemeris,
            target_unit_vectors,
            time_indices,
        )? {
            results.push(result);
        } else {
            let (target_ras, target_decs) = fallback_radec
                .get_or_insert_with(|| unit_vectors_to_radec_batch(target_unit_vectors));
            results.push(constraint.in_constraint_batch(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
            )?);
        }
    }

    Ok(results)
}

pub(super) struct AndEvaluator {
    pub(super) constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for AndEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times()?;

        // Build the actual indices we'll iterate over
        let indices: Vec<usize> = if let Some(idx) = time_indices {
            idx.to_vec()
        } else {
            (0..times.len()).collect()
        };

        let times_filtered: Vec<_> = indices.iter().map(|&i| times[i]).collect();

        let violations = track_violations(
            &times_filtered,
            |i| {
                let mut all_violated = true;
                let mut min_severity = f64::MAX;

                // Use the ORIGINAL index, not the loop index
                let original_idx = indices[i];

                // Check each constraint at this time
                for constraint in &self.constraints {
                    let result = constraint.evaluate(
                        ephemeris,
                        target_ra,
                        target_dec,
                        Some(&[original_idx]),
                    );
                    if let Ok(ref res) = result {
                        if res.violations.is_empty() {
                            all_violated = false;
                        } else {
                            for violation in &res.violations {
                                min_severity = min_severity.min(violation.max_severity);
                            }
                        }
                    } else {
                        all_violated = false;
                    }
                }

                (
                    all_violated,
                    if min_severity == f64::MAX {
                        1.0
                    } else {
                        min_severity
                    },
                )
            },
            |i, _is_open| {
                let mut descriptions = Vec::new();
                let original_idx = indices[i];

                // Get descriptions from all violated constraints at this time
                for constraint in &self.constraints {
                    let result = constraint.evaluate(
                        ephemeris,
                        target_ra,
                        target_dec,
                        Some(&[original_idx]),
                    );
                    if let Ok(ref res) = result {
                        for violation in &res.violations {
                            descriptions.push(violation.description.clone());
                        }
                    }
                }

                if descriptions.is_empty() {
                    "AND violation".to_string()
                } else {
                    format!("AND violation: {}", descriptions.join("; "))
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let times = ephemeris.get_times()?;
        // Use filtered time count if time_indices provided, otherwise full times
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices))
            .collect();
        let results = results?;

        let n_targets = target_ras.len();

        // Preserve vacuous truth: AND over zero constraints is true everywhere.
        if results.is_empty() {
            return Ok(Array2::from_elem((n_targets, n_times), true));
        }

        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] && sub_result[[i, j]];
                }
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let results = eval_constraints_batch_from_unit_vectors(
            &self.constraints,
            ephemeris,
            target_unit_vectors,
            time_indices,
        )?;

        let n_times = if let Some(first) = results.first() {
            first.ncols()
        } else {
            time_indices
                .map(|idx| idx.len())
                .unwrap_or(ephemeris.get_times()?.len())
        };

        let n_targets = target_unit_vectors.nrows();

        if results.is_empty() {
            return Ok(Some(Array2::from_elem((n_targets, n_times), true)));
        }

        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] && sub_result[[i, j]];
                }
            }
        }

        Ok(Some(result))
    }

    /// Optimized diagonal evaluation for AND - uses O(N) diagonal from each sub-constraint
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

        // Get diagonal results from each sub-constraint
        let sub_results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch_diagonal(ephemeris, target_ras, target_decs))
            .collect();
        let sub_results = sub_results?;

        // AND logic: violated if ALL sub-constraints are violated at each time
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let all_violated = sub_results.iter().all(|r| r[i]);
            result.push(all_violated);
        }

        Ok(result)
    }

    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let times = ephemeris.get_times()?;
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )
            })
            .collect();
        let results = results?;
        let n_targets = target_ras.len();
        if results.is_empty() {
            return Ok(Array2::from_elem((n_targets, n_times), true));
        }
        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] && sub_result[[i, j]];
                }
            }
        }
        Ok(result)
    }

    fn is_roll_dependent(&self) -> bool {
        self.constraints.iter().any(|c| c.is_roll_dependent())
    }

    /// Hoist roll-independent children: `AND_step (V_indep ∧ V_dep_at_step)
    /// = V_indep ∧ AND_step (V_dep_at_step)` since V_indep doesn't depend on the roll.
    fn in_constraint_batch_constrained_at_every_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
    ) -> PyResult<Array2<bool>> {
        if !self.is_roll_dependent() {
            return self.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices);
        }

        let n_times = match time_indices {
            Some(idx) => idx.len(),
            None => ephemeris.get_times()?.len(),
        };
        let n_targets = target_ras.len();

        let (indep_children, dep_children): (Vec<_>, Vec<_>) = self
            .constraints
            .iter()
            .partition(|c| !c.is_roll_dependent());

        // V_indep_and — AND over all roll-independent children, computed once.
        let mut v_indep: Option<Array2<bool>> = None;
        for c in &indep_children {
            let r = c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices)?;
            match v_indep {
                None => v_indep = Some(r),
                Some(ref mut a) => a.zip_mut_with(&r, |x, &y| *x &= y),
            }
        }

        if dep_children.is_empty() {
            return Ok(
                v_indep.unwrap_or_else(|| Array2::<bool>::from_elem((n_targets, n_times), true))
            );
        }

        if dep_children.len() == 1 && indep_children.is_empty() {
            return dep_children[0].in_constraint_batch_constrained_at_every_roll(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                n_roll_samples,
            );
        }

        // AND_step (AND over dep_children of in_constraint_batch_at_roll).
        let roll_step = 360.0 / n_roll_samples as f64;
        let mut acc: Option<Array2<bool>> = None;
        for step in 0..n_roll_samples {
            if let Some(ref a) = acc {
                if a.iter().all(|&b| !b) {
                    break;
                }
            }
            let roll_deg = step as f64 * roll_step;
            let mut step_and: Option<Array2<bool>> = None;
            for c in &dep_children {
                let r = c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )?;
                match step_and {
                    None => step_and = Some(r),
                    Some(ref mut a) => a.zip_mut_with(&r, |x, &y| *x &= y),
                }
            }
            let step_and =
                step_and.unwrap_or_else(|| Array2::<bool>::from_elem((n_targets, n_times), true));
            match acc {
                None => acc = Some(step_and),
                Some(ref mut a) => a.zip_mut_with(&step_and, |x, &y| *x &= y),
            }
        }

        let mut result =
            acc.unwrap_or_else(|| Array2::<bool>::from_elem((n_targets, n_times), false));
        if let Some(v) = v_indep {
            result.zip_mut_with(&v, |x, &y| *x &= y);
        }
        Ok(result)
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        if self.constraints.is_empty() {
            return Ok(vec![true; n_targets]);
        }
        let mut result = self.constraints[0].field_of_regard_violated_at_roll(
            ephemeris,
            target_unit_vectors,
            time_index,
            roll_deg,
        )?;
        for constraint in &self.constraints[1..] {
            let sub = constraint.field_of_regard_violated_at_roll(
                ephemeris,
                target_unit_vectors,
                time_index,
                roll_deg,
            )?;
            for i in 0..n_targets {
                result[i] = result[i] && sub[i];
            }
        }
        Ok(result)
    }

    /// AND decomposes trivially through the universal quantifier:
    /// `∀θ. ⋀_c c(θ) violated  =  ⋀_c (∀θ. c(θ) violated)`,
    /// i.e. the AND combinator's FoR-violation is the AND of each child's FoR-violation.
    /// Each child therefore gets to use its own optimised `field_of_regard_violated_batch`
    /// (e.g. bright_star's cached gnomonic projection), avoiding the default coupled sweep
    /// that would re-call every child once per roll step.
    fn field_of_regard_violated_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        n_roll_samples: usize,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        if self.constraints.is_empty() {
            return Ok(vec![true; n_targets]);
        }

        let mut result = self.constraints[0].field_of_regard_violated_batch(
            ephemeris,
            target_unit_vectors,
            time_index,
            n_roll_samples,
        )?;
        for c in &self.constraints[1..] {
            if result.iter().all(|&v| !v) {
                return Ok(result); // every target already accessible — AND can't flip it back
            }
            let sub = c.field_of_regard_violated_batch(
                ephemeris,
                target_unit_vectors,
                time_index,
                n_roll_samples,
            )?;
            for i in 0..n_targets {
                result[i] = result[i] && sub[i];
            }
        }
        Ok(result)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        merge_children_named_values(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        merge_children_named_booleans(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        merge_children_named_booleans_diagonal(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
        )
    }

    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        merge_children_cause_value_keys(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn name(&self) -> String {
        format!(
            "AND({})",
            self.constraints
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) struct OrEvaluator {
    pub(super) constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for OrEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times()?;

        // Build the actual indices we'll iterate over
        let indices: Vec<usize> = if let Some(idx) = time_indices {
            idx.to_vec()
        } else {
            (0..times.len()).collect()
        };

        let times_filtered: Vec<_> = indices.iter().map(|&i| times[i]).collect();

        let violations = track_violations(
            &times_filtered,
            |i| {
                let mut any_violated = false;
                let mut max_severity = 0.0f64;

                // Use the ORIGINAL index, not the loop index
                let original_idx = indices[i];

                // OR logic: violated if ANY sub-constraint is violated
                // (if any constraint blocks observation, target is not visible)
                for constraint in &self.constraints {
                    let result = constraint.evaluate(
                        ephemeris,
                        target_ra,
                        target_dec,
                        Some(&[original_idx]),
                    );
                    if let Ok(ref res) = result {
                        if !res.violations.is_empty() {
                            any_violated = true;
                            for violation in &res.violations {
                                max_severity = max_severity.max(violation.max_severity);
                            }
                        }
                    }
                }

                (any_violated, max_severity)
            },
            |i, _is_open| {
                let mut descriptions = Vec::new();
                let original_idx = indices[i];

                // Get descriptions from all violated constraints at this time
                for constraint in &self.constraints {
                    let result = constraint.evaluate(
                        ephemeris,
                        target_ra,
                        target_dec,
                        Some(&[original_idx]),
                    );
                    if let Ok(ref res) = result {
                        for violation in &res.violations {
                            descriptions.push(violation.description.clone());
                        }
                    }
                }

                if descriptions.is_empty() {
                    "OR violation".to_string()
                } else {
                    format!("OR violation: {}", descriptions.join("; "))
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let times = ephemeris.get_times()?;
        // Use filtered time count if time_indices provided, otherwise full times
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices))
            .collect();
        let results = results?;

        let n_targets = target_ras.len();

        // Preserve identity: OR over zero constraints is false everywhere.
        if results.is_empty() {
            return Ok(Array2::from_elem((n_targets, n_times), false));
        }

        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] || sub_result[[i, j]];
                }
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let results = eval_constraints_batch_from_unit_vectors(
            &self.constraints,
            ephemeris,
            target_unit_vectors,
            time_indices,
        )?;

        let n_times = if let Some(first) = results.first() {
            first.ncols()
        } else {
            time_indices
                .map(|idx| idx.len())
                .unwrap_or(ephemeris.get_times()?.len())
        };

        let n_targets = target_unit_vectors.nrows();

        if results.is_empty() {
            return Ok(Some(Array2::from_elem((n_targets, n_times), false)));
        }

        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] || sub_result[[i, j]];
                }
            }
        }

        Ok(Some(result))
    }

    /// Optimized diagonal evaluation for OR - uses O(N) diagonal from each sub-constraint
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

        // Get diagonal results from each sub-constraint
        let sub_results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch_diagonal(ephemeris, target_ras, target_decs))
            .collect();
        let sub_results = sub_results?;

        // OR logic: violated if ANY sub-constraint is violated at each time
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let any_violated = sub_results.iter().any(|r| r[i]);
            result.push(any_violated);
        }

        Ok(result)
    }

    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let times = ephemeris.get_times()?;
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )
            })
            .collect();
        let results = results?;
        let n_targets = target_ras.len();
        if results.is_empty() {
            return Ok(Array2::from_elem((n_targets, n_times), false));
        }
        let mut result = results[0].clone();
        for sub_result in &results[1..] {
            for i in 0..n_targets {
                for j in 0..n_times {
                    result[[i, j]] = result[[i, j]] || sub_result[[i, j]];
                }
            }
        }
        Ok(result)
    }

    fn is_roll_dependent(&self) -> bool {
        self.constraints.iter().any(|c| c.is_roll_dependent())
    }

    /// Hoist roll-independent children: `AND_step (V_indep ∨ V_dep_at_step)
    /// = V_indep ∨ AND_step (V_dep_at_step)` since V_indep doesn't depend on the roll.
    /// Avoids re-evaluating roll-independent siblings 360 times in the outer sweep.
    fn in_constraint_batch_constrained_at_every_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
    ) -> PyResult<Array2<bool>> {
        if !self.is_roll_dependent() {
            return self.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices);
        }

        let n_times = match time_indices {
            Some(idx) => idx.len(),
            None => ephemeris.get_times()?.len(),
        };
        let n_targets = target_ras.len();

        let (indep_children, dep_children): (Vec<_>, Vec<_>) = self
            .constraints
            .iter()
            .partition(|c| !c.is_roll_dependent());

        // V_indep_or — OR of all roll-independent children, computed once.
        let mut v_indep = Array2::<bool>::from_elem((n_targets, n_times), false);
        for c in &indep_children {
            let r = c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices)?;
            v_indep.zip_mut_with(&r, |x, &y| *x |= y);
        }

        if dep_children.is_empty() {
            return Ok(v_indep);
        }

        // For roll-dependent children, recurse into their swept method when possible —
        // BUT only when there's a single dep child, since we need per-step values to OR
        // across siblings.  With multiple dep children we must loop step-by-step.
        if dep_children.len() == 1 && indep_children.is_empty() {
            return dep_children[0].in_constraint_batch_constrained_at_every_roll(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                n_roll_samples,
            );
        }

        // AND_step (OR over dep_children of in_constraint_batch_at_roll).
        let roll_step = 360.0 / n_roll_samples as f64;
        let mut acc: Option<Array2<bool>> = None;
        for step in 0..n_roll_samples {
            if let Some(ref a) = acc {
                if a.iter().all(|&b| !b) {
                    break;
                }
            }
            let roll_deg = step as f64 * roll_step;
            let mut step_or = Array2::<bool>::from_elem((n_targets, n_times), false);
            for c in &dep_children {
                let r = c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )?;
                step_or.zip_mut_with(&r, |x, &y| *x |= y);
            }
            match acc {
                None => acc = Some(step_or),
                Some(ref mut a) => a.zip_mut_with(&step_or, |x, &y| *x &= y),
            }
        }

        let mut result =
            acc.unwrap_or_else(|| Array2::<bool>::from_elem((n_targets, n_times), false));
        result.zip_mut_with(&v_indep, |x, &y| *x |= y);
        Ok(result)
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        if self.constraints.is_empty() {
            return Ok(vec![false; n_targets]);
        }
        let mut result = self.constraints[0].field_of_regard_violated_at_roll(
            ephemeris,
            target_unit_vectors,
            time_index,
            roll_deg,
        )?;
        for constraint in &self.constraints[1..] {
            let sub = constraint.field_of_regard_violated_at_roll(
                ephemeris,
                target_unit_vectors,
                time_index,
                roll_deg,
            )?;
            for i in 0..n_targets {
                result[i] = result[i] || sub[i];
            }
        }
        Ok(result)
    }

    /// Mirror of `in_constraint_batch_constrained_at_every_roll` for the FoR path.
    ///
    /// Hoist roll-independent children out of the sweep — they evaluate to a constant
    /// `V_indep_or` per target, so a target with `V_indep_or[i] = true` is FoR-violated
    /// regardless of θ and is removed from the dep sweep entirely.  When there is exactly
    /// one roll-dependent child left, delegate to its own optimised
    /// `field_of_regard_violated_batch` (this is the `OR(sun_prox, bright_star)` shape that
    /// users typically write).  Multiple dep children stay coupled and sweep together.
    fn field_of_regard_violated_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        n_roll_samples: usize,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        if self.constraints.is_empty() {
            return Ok(vec![false; n_targets]);
        }

        let (indep_children, dep_children): (Vec<_>, Vec<_>) = self
            .constraints
            .iter()
            .partition(|c| !c.is_roll_dependent());

        // V_indep_or — OR of all roll-independent children, evaluated once each.
        let mut v_indep = vec![false; n_targets];
        for c in &indep_children {
            let sub = c.field_of_regard_violated_batch(
                ephemeris,
                target_unit_vectors,
                time_index,
                n_roll_samples,
            )?;
            for i in 0..n_targets {
                v_indep[i] = v_indep[i] || sub[i];
            }
        }

        if dep_children.is_empty() {
            return Ok(v_indep);
        }

        // Singleton dep-child: result[i] = V_indep_or[i] || dep_for_violated[i].
        if dep_children.len() == 1 {
            let dep_for = dep_children[0].field_of_regard_violated_batch(
                ephemeris,
                target_unit_vectors,
                time_index,
                n_roll_samples,
            )?;
            return Ok((0..n_targets).map(|i| v_indep[i] || dep_for[i]).collect());
        }

        // Multiple dep children: must couple at each θ.
        // accessible[i] = ∃θ: every dep child not violated at θ; targets with v_indep[i]
        // are excluded from this — they're already FoR-violated.
        let roll_step_deg = 360.0 / n_roll_samples as f64;
        let mut accessible = vec![false; n_targets];
        for step in 0..n_roll_samples {
            // Early-exit when every non-pre-violated target has found a clear roll.
            if (0..n_targets).all(|i| v_indep[i] || accessible[i]) {
                break;
            }
            let roll_deg = step as f64 * roll_step_deg;
            let mut step_or_violated = vec![false; n_targets];
            for c in &dep_children {
                let r = c.field_of_regard_violated_at_roll(
                    ephemeris,
                    target_unit_vectors,
                    time_index,
                    roll_deg,
                )?;
                for i in 0..n_targets {
                    step_or_violated[i] = step_or_violated[i] || r[i];
                }
            }
            for i in 0..n_targets {
                if !accessible[i] && !v_indep[i] && !step_or_violated[i] {
                    accessible[i] = true;
                }
            }
        }

        Ok((0..n_targets)
            .map(|i| v_indep[i] || !accessible[i])
            .collect())
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        merge_children_named_values(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        merge_children_named_booleans(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        merge_children_named_booleans_diagonal(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
        )
    }

    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        merge_children_cause_value_keys(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn name(&self) -> String {
        format!(
            "OR({})",
            self.constraints
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) struct NotEvaluator {
    pub(super) constraint: Box<dyn ConstraintEvaluator>,
}

impl ConstraintEvaluator for NotEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times()?;

        // Build the filtered timeline – consistent with And/Or/AtLeast evaluators.
        let indices: Vec<usize> = if let Some(idx) = time_indices {
            idx.to_vec()
        } else {
            (0..times.len()).collect()
        };
        let times_filtered: Vec<_> = indices.iter().map(|&i| times[i]).collect();

        let result = self
            .constraint
            .evaluate(ephemeris, target_ra, target_dec, time_indices)?;

        // Invert violations over times_filtered only, not the full timeline.
        let mut inverted_violations = Vec::new();

        if times_filtered.is_empty() {
            // Nothing to invert.
        } else if result.violations.is_empty() {
            // Inner constraint was satisfied everywhere in the subset → NOT is violated everywhere.
            inverted_violations.push(ConstraintViolation {
                start_time_internal: times_filtered[0],
                end_time_internal: times_filtered[times_filtered.len() - 1],
                max_severity: 1.0,
                description: format!(
                    "NOT({}): inner constraint was satisfied",
                    self.constraint.name()
                ),
            });
        } else {
            // Find gaps between violations within times_filtered (gaps become new violations).
            let mut last_end = times_filtered[0];

            for violation in &result.violations {
                if last_end < violation.start_time_internal {
                    inverted_violations.push(ConstraintViolation {
                        start_time_internal: last_end,
                        end_time_internal: violation.start_time_internal,
                        max_severity: 0.5,
                        description: format!(
                            "NOT({}): inner constraint was satisfied",
                            self.constraint.name()
                        ),
                    });
                }
                last_end = violation.end_time_internal;
            }

            // Check for a gap after the last violation.
            let final_time = times_filtered[times_filtered.len() - 1];
            if last_end < final_time {
                inverted_violations.push(ConstraintViolation {
                    start_time_internal: last_end,
                    end_time_internal: final_time,
                    max_severity: 0.5,
                    description: format!(
                        "NOT({}): inner constraint was satisfied",
                        self.constraint.name()
                    ),
                });
            }
        }

        let all_satisfied = inverted_violations.is_empty();
        Ok(ConstraintResult::new(
            inverted_violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        let times = ephemeris.get_times()?;
        // Evaluate sub-constraint in batch
        let sub_result = self.constraint.in_constraint_batch(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )?;

        let n_targets = target_ras.len();
        // Use filtered time count if time_indices provided, otherwise full times
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // NOT logic: invert all values
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = !sub_result[[i, j]];
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let sub_result = eval_constraint_batch_from_unit_vectors(
            self.constraint.as_ref(),
            ephemeris,
            target_unit_vectors,
            time_indices,
        )?;

        let n_targets = sub_result.nrows();
        let n_times = sub_result.ncols();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // NOT logic: invert all values
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = !sub_result[[i, j]];
            }
        }

        Ok(Some(result))
    }

    /// Optimized diagonal evaluation for NOT - uses O(N) diagonal from sub-constraint
    fn in_constraint_batch_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        let sub_result =
            self.constraint
                .in_constraint_batch_diagonal(ephemeris, target_ras, target_decs)?;

        // NOT logic: invert all values
        Ok(sub_result.into_iter().map(|v| !v).collect())
    }

    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> pyo3::PyResult<Array2<bool>> {
        let times = ephemeris.get_times()?;
        let sub_result = self.constraint.in_constraint_batch_at_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            roll_deg,
        )?;
        let n_targets = target_ras.len();
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let mut result = Array2::from_elem((n_targets, n_times), false);
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = !sub_result[[i, j]];
            }
        }
        Ok(result)
    }

    fn is_roll_dependent(&self) -> bool {
        self.constraint.is_roll_dependent()
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let sub = self.constraint.field_of_regard_violated_at_roll(
            ephemeris,
            target_unit_vectors,
            time_index,
            roll_deg,
        )?;
        // NOTE: Do not negate here: inner result is already aggregated over roll freedom.
        Ok(sub)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        // Single child, no collision risk - pass its values straight through unprefixed.
        self.constraint
            .compute_named_values(ephemeris, target_ras, target_decs, time_indices)
    }

    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        // Values are uninverted (the child's own violation mask, unchanged: cause
        // attribution reports whether the underlying leaf condition, e.g. "too close
        // to sun", itself changed, not the NOT-negated combined value), but keys are
        // prefixed with "not." so a wrapped leaf's cause tag is never mistaken for its
        // non-negated counterpart - matching how `merge_children_named_values` already
        // tags a NOT child when nested under AND/OR/XOR/AT_LEAST.
        let child_booleans = self.constraint.compute_named_booleans(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )?;
        Ok(child_booleans
            .into_iter()
            .map(|(key, arr)| (format!("not.{key}"), arr))
            .collect())
    }

    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        // See `compute_named_booleans`: values are uninverted, keys prefixed with "not.".
        let child_booleans =
            self.constraint
                .compute_named_booleans_diagonal(ephemeris, target_ras, target_decs)?;
        Ok(child_booleans
            .into_iter()
            .map(|(key, arr)| (format!("not.{key}"), arr))
            .collect())
    }

    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        // Cause tags get "not." (matching compute_named_booleans); value keys pass
        // through unprefixed (matching compute_named_values' passthrough — "single
        // child, no collision risk").
        let child_map = self.constraint.compute_cause_value_keys(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )?;
        Ok(child_map
            .into_iter()
            .map(|(cause_tag, value_keys)| (format!("not.{cause_tag}"), value_keys))
            .collect())
    }

    fn name(&self) -> String {
        format!("NOT({})", self.constraint.name())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) struct XorEvaluator {
    pub(super) constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for XorEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times()?;

        // Build the filtered timeline – consistent with And/Or/Not/AtLeast evaluators.
        let indices: Vec<usize> = if let Some(idx) = time_indices {
            idx.to_vec()
        } else {
            (0..times.len()).collect()
        };
        let times_filtered: Vec<_> = indices.iter().map(|&i| times[i]).collect();

        // Evaluate all constraints
        let results: Vec<_> = self
            .constraints
            .iter()
            .map(|c| c.evaluate(ephemeris, target_ra, target_dec, time_indices))
            .collect::<PyResult<Vec<_>>>()?;

        // Violate when EXACTLY ONE sub-constraint is violated
        let mut merged_violations = Vec::new();
        let mut current_violation: Option<(usize, f64, Vec<String>)> = None;

        for (i, time) in times_filtered.iter().enumerate() {
            let mut active: Vec<&ConstraintViolation> = Vec::new();

            for result in &results {
                for violation in &result.violations {
                    if violation.start_time_internal <= *time
                        && *time <= violation.end_time_internal
                    {
                        active.push(violation);
                        break;
                    }
                }
            }

            if active.len() == 1 {
                let violation = active[0];
                match &mut current_violation {
                    Some((_, sev, descs)) => {
                        *sev = sev.max(violation.max_severity);
                        if !descs.iter().any(|d| d == &violation.description) {
                            descs.push(violation.description.clone());
                        }
                    }
                    None => {
                        current_violation = Some((
                            i,
                            violation.max_severity,
                            vec![violation.description.clone()],
                        ));
                    }
                }
            } else if let Some((start_idx, severity, descs)) = current_violation.take() {
                merged_violations.push(ConstraintViolation {
                    start_time_internal: times_filtered[start_idx],
                    end_time_internal: times_filtered[i - 1],
                    max_severity: severity,
                    description: format!("XOR violation: {}", descs.join("; ")),
                });
            }
        }

        if let Some((start_idx, severity, descs)) = current_violation {
            merged_violations.push(ConstraintViolation {
                start_time_internal: times_filtered[start_idx],
                end_time_internal: times_filtered[times_filtered.len() - 1],
                max_severity: severity,
                description: format!("XOR violation: {}", descs.join("; ")),
            });
        }

        let all_satisfied = merged_violations.is_empty();
        Ok(ConstraintResult::new(
            merged_violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let times = ephemeris.get_times()?;
        // Use filtered time count if time_indices provided, otherwise full times
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices))
            .collect();
        let results = results?;

        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // XOR logic: violated when EXACTLY ONE sub-constraint is violated
        for i in 0..n_targets {
            for j in 0..n_times {
                let violation_count = results.iter().filter(|r| r[[i, j]]).count();
                result[[i, j]] = violation_count == 1;
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let results = eval_constraints_batch_from_unit_vectors(
            &self.constraints,
            ephemeris,
            target_unit_vectors,
            time_indices,
        )?;

        let n_times = if let Some(first) = results.first() {
            first.ncols()
        } else {
            time_indices
                .map(|idx| idx.len())
                .unwrap_or(ephemeris.get_times()?.len())
        };

        let n_targets = target_unit_vectors.nrows();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // XOR logic: violated when EXACTLY ONE sub-constraint is violated
        for i in 0..n_targets {
            for j in 0..n_times {
                let violation_count = results.iter().filter(|r| r[[i, j]]).count();
                result[[i, j]] = violation_count == 1;
            }
        }

        Ok(Some(result))
    }

    /// Optimized diagonal evaluation for XOR - uses O(N) diagonal from each sub-constraint
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

        // Get diagonal results from each sub-constraint
        let sub_results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch_diagonal(ephemeris, target_ras, target_decs))
            .collect();
        let sub_results = sub_results?;

        // XOR logic: violated when EXACTLY ONE sub-constraint is violated at each time
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let violation_count = sub_results.iter().filter(|r| r[i]).count();
            result.push(violation_count == 1);
        }

        Ok(result)
    }

    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let times = ephemeris.get_times()?;
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )
            })
            .collect();
        let results = results?;
        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);
        for i in 0..n_targets {
            for j in 0..n_times {
                let count = results.iter().filter(|r| r[[i, j]]).count();
                result[[i, j]] = count == 1;
            }
        }
        Ok(result)
    }

    fn is_roll_dependent(&self) -> bool {
        self.constraints.iter().any(|c| c.is_roll_dependent())
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        let sub_results: PyResult<Vec<_>> = self
            .constraints
            .iter()
            .map(|c| {
                c.field_of_regard_violated_at_roll(
                    ephemeris,
                    target_unit_vectors,
                    time_index,
                    roll_deg,
                )
            })
            .collect();
        let sub_results = sub_results?;
        let mut result = Vec::with_capacity(n_targets);
        for i in 0..n_targets {
            let count = sub_results.iter().filter(|r| r[i]).count();
            result.push(count == 1);
        }
        Ok(result)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        merge_children_named_values(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        merge_children_named_booleans(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        merge_children_named_booleans_diagonal(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
        )
    }

    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        merge_children_cause_value_keys(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn name(&self) -> String {
        format!(
            "XOR({})",
            self.constraints
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) struct AtLeastEvaluator {
    pub(super) constraints: Vec<Box<dyn ConstraintEvaluator>>,
    pub(super) min_violated: usize,
}

impl ConstraintEvaluator for AtLeastEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times()?;

        let indices: Vec<usize> = if let Some(idx) = time_indices {
            idx.to_vec()
        } else {
            (0..times.len()).collect()
        };

        let times_filtered: Vec<_> = indices.iter().map(|&i| times[i]).collect();
        let n_times = times_filtered.len();

        let mut violated_descriptions: Vec<Vec<String>> = vec![Vec::new(); n_times];
        let mut is_violated: Vec<bool> = vec![false; n_times];
        let mut severity: Vec<f64> = vec![0.0; n_times];

        // Evaluate each sub-constraint once per selected time index and cache
        // the results for both violation tracking and descriptions.
        for (i, &original_idx) in indices.iter().enumerate() {
            let mut violation_count = 0usize;
            let mut max_severity = 0.0f64;

            for constraint in &self.constraints {
                let result =
                    constraint.evaluate(ephemeris, target_ra, target_dec, Some(&[original_idx]));
                if let Ok(ref res) = result {
                    if !res.violations.is_empty() {
                        violation_count += 1;
                        for violation in &res.violations {
                            max_severity = max_severity.max(violation.max_severity);
                            violated_descriptions[i].push(violation.description.clone());
                        }
                    }
                }
            }

            is_violated[i] = violation_count >= self.min_violated;
            severity[i] = max_severity;
        }

        let violations = track_violations(
            &times_filtered,
            |i| (is_violated[i], severity[i]),
            |i, _is_open| {
                let descriptions = &violated_descriptions[i];

                if descriptions.is_empty() {
                    format!("AT_LEAST(k={}) violation", self.min_violated)
                } else {
                    format!(
                        "AT_LEAST(k={}) violation: {}",
                        self.min_violated,
                        descriptions.join("; ")
                    )
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let times = ephemeris.get_times()?;
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());

        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices))
            .collect();
        let results = results?;

        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        for i in 0..n_targets {
            for j in 0..n_times {
                let mut violation_count = 0usize;
                for sub_result in &results {
                    if sub_result[[i, j]] {
                        violation_count += 1;
                        if violation_count >= self.min_violated {
                            break;
                        }
                    }
                }
                result[[i, j]] = violation_count >= self.min_violated;
            }
        }

        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let results = eval_constraints_batch_from_unit_vectors(
            &self.constraints,
            ephemeris,
            target_unit_vectors,
            time_indices,
        )?;

        let n_times = if let Some(first) = results.first() {
            first.ncols()
        } else {
            time_indices
                .map(|idx| idx.len())
                .unwrap_or(ephemeris.get_times()?.len())
        };

        let n_targets = target_unit_vectors.nrows();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        for i in 0..n_targets {
            for j in 0..n_times {
                let mut violation_count = 0usize;
                for sub_result in &results {
                    if sub_result[[i, j]] {
                        violation_count += 1;
                        if violation_count >= self.min_violated {
                            break;
                        }
                    }
                }
                result[[i, j]] = violation_count >= self.min_violated;
            }
        }

        Ok(Some(result))
    }

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

        let sub_results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| c.in_constraint_batch_diagonal(ephemeris, target_ras, target_decs))
            .collect();
        let sub_results = sub_results?;

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let violation_count = sub_results.iter().filter(|r| r[i]).count();
            result.push(violation_count >= self.min_violated);
        }

        Ok(result)
    }

    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let times = ephemeris.get_times()?;
        let n_times = time_indices.map(|idx| idx.len()).unwrap_or(times.len());
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch_at_roll(
                    ephemeris,
                    target_ras,
                    target_decs,
                    time_indices,
                    roll_deg,
                )
            })
            .collect();
        let results = results?;
        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);
        for i in 0..n_targets {
            for j in 0..n_times {
                let mut count = 0usize;
                for sub_result in &results {
                    if sub_result[[i, j]] {
                        count += 1;
                        if count >= self.min_violated {
                            break;
                        }
                    }
                }
                result[[i, j]] = count >= self.min_violated;
            }
        }
        Ok(result)
    }

    fn is_roll_dependent(&self) -> bool {
        self.constraints.iter().any(|c| c.is_roll_dependent())
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();
        let sub_results: PyResult<Vec<_>> = self
            .constraints
            .iter()
            .map(|c| {
                c.field_of_regard_violated_at_roll(
                    ephemeris,
                    target_unit_vectors,
                    time_index,
                    roll_deg,
                )
            })
            .collect();
        let sub_results = sub_results?;
        let mut result = Vec::with_capacity(n_targets);
        for i in 0..n_targets {
            let count = sub_results.iter().filter(|r| r[i]).count();
            result.push(count >= self.min_violated);
        }
        Ok(result)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        merge_children_named_values(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        merge_children_named_booleans(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        merge_children_named_booleans_diagonal(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
        )
    }

    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        merge_children_cause_value_keys(
            &self.constraints,
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
        )
    }

    fn name(&self) -> String {
        format!(
            "AT_LEAST(k={}, {})",
            self.min_violated,
            self.constraints
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
