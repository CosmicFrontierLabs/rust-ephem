// Logical combinator evaluators
use crate::constraints::core::{
    track_violations, ConstraintEvaluator, ConstraintResult, ConstraintViolation,
};
use crate::utils::vector_math::unit_vectors_to_radec_batch;
use ndarray::Array2;
use pyo3::PyResult;

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
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // AND logic: violated only if ALL sub-constraints are violated
        for i in 0..n_targets {
            for j in 0..n_times {
                let all_violated = results.iter().all(|r| r[[i, j]]);
                result[[i, j]] = all_violated;
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

        // AND logic: violated only if ALL sub-constraints are violated
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = results.iter().all(|r| r[[i, j]]);
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
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // OR logic: violated if ANY sub-constraint is violated
        for i in 0..n_targets {
            for j in 0..n_times {
                let any_violated = results.iter().any(|r| r[[i, j]]);
                result[[i, j]] = any_violated;
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

        // OR logic: violated if ANY sub-constraint is violated
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = results.iter().any(|r| r[[i, j]]);
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
