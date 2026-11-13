use crate::constraints::core::{
    sweep_rolls_with_attribution, track_violations, ConstraintEvaluator, ConstraintResult,
    EvaluationRoll,
};
use crate::constraints::diagonal::DiagonalEvaluation;
use crate::constraints::elementwise::evaluate_elements;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::sampled_ephemeris::{locate_times, SampledEphemeris};
use crate::ephemeris::FileEphemeris;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::ParquetEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use crate::utils::apparent::ObserverAberration;
use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use super::PyConstraint;

/// Target-vector components corrected per block of times (about 64 MB).
const APPARENT_BLOCK_VALUES: usize = 8_000_000;
/// Fewest target x time cells in a block worth spreading over the cores.
const APPARENT_PARALLEL_MIN_CELLS: usize = 20_000;

impl PyConstraint {
    pub(super) fn parse_datetime_input(times: &Bound<PyAny>) -> PyResult<Vec<DateTime<Utc>>> {
        crate::utils::time_utils::python_datetimes_to_utc(times)
    }

    /// Keep exact-grid queries on the original ephemeris. Off-grid queries use
    /// a small interpolated view at the requested times, in the supplied order.
    pub(super) fn with_evaluation_ephemeris<T, F>(
        &self,
        bound: &Bound<'_, PyAny>,
        times: Option<&Bound<'_, PyAny>>,
        indices: Option<&Bound<'_, PyAny>>,
        mut f: F,
    ) -> PyResult<T>
    where
        F: FnMut(&dyn EphemerisBase, Option<Vec<usize>>) -> PyResult<T>,
    {
        if times.is_some() && indices.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot specify both 'times' and 'indices' parameters",
            ));
        }
        let requested = times.map(Self::parse_datetime_input).transpose()?;
        let indices = indices.map(|arg| self.parse_indices(arg)).transpose()?;
        self.with_ephemeris(bound, |source| {
            let Some(ref times) = requested else {
                // Validate once before any leaf, cached sweep, or empty-target
                // shortcut can use user-supplied indices for unchecked indexing.
                if let Some(ref selected) = indices {
                    let n_times = source
                        .data()
                        .times
                        .as_ref()
                        .ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "No ephemeris timestamps available",
                            )
                        })?
                        .len();
                    if let Some(index) = selected.iter().find(|&&index| index >= n_times) {
                        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                            "index {index} out of bounds for ephemeris with {n_times} timestamps"
                        )));
                    }
                }
                return f(source, indices.clone());
            };
            let source_times = source.data().times.as_deref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No ephemeris timestamps available")
            })?;
            let locations = locate_times(source_times, times)?;
            if let Ok(exact) = locations.iter().copied().collect::<Result<Vec<_>, _>>() {
                return f(source, Some(exact));
            }
            let sampled = SampledEphemeris::new(source, times, &locations)?;
            f(&sampled, None)
        })
    }

    /// Resolve per-element times to ONE evaluation ephemeris plus, for each
    /// element, the row of that ephemeris holding its time.
    ///
    /// `times = None` pairs element `i` with ephemeris timestamp `i`. Otherwise
    /// the requested times (any order, duplicates allowed) are deduplicated; if
    /// every unique time is an ephemeris sample the original ephemeris is used
    /// directly, else a `SampledEphemeris` is interpolated at the unique times
    /// (exact samples copied, off-grid ones Hermite-interpolated from their
    /// bracketing samples). Each element is evaluated at its own time — never
    /// paired with an ephemeris row by position.
    pub(super) fn with_elementwise_ephemeris<T, F>(
        &self,
        bound: &Bound<'_, PyAny>,
        n_elements: usize,
        times: Option<&Bound<'_, PyAny>>,
        mut f: F,
    ) -> PyResult<T>
    where
        F: FnMut(&dyn EphemerisBase, &[usize], Vec<DateTime<Utc>>) -> PyResult<T>,
    {
        let requested = times.map(Self::parse_datetime_input).transpose()?;
        if let Some(ref requested) = requested {
            if requested.len() != n_elements {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "times must have the same length as target_ras and target_decs \
                     (got {} times for {} targets)",
                    requested.len(),
                    n_elements
                )));
            }
        }
        self.with_ephemeris(bound, |source| {
            let source_times = source.data().times.as_deref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No ephemeris timestamps available")
            })?;
            let Some(ref requested) = requested else {
                if n_elements != source_times.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "times=None pairs element i with ephemeris timestamp i, so \
                         target_ras/target_decs must have one entry per ephemeris \
                         timestamp (got {} targets, ephemeris has {} timestamps)",
                        n_elements,
                        source_times.len()
                    )));
                }
                let indices: Vec<usize> = (0..n_elements).collect();
                return f(source, &indices, source_times.to_vec());
            };
            let mut unique = requested.clone();
            unique.sort_unstable();
            unique.dedup();
            let locations = locate_times(source_times, &unique)?;
            let element_rows: Vec<usize> = requested
                .iter()
                .map(|time| unique.binary_search(time).unwrap_or_else(|i| i))
                .collect();
            if let Ok(exact) = locations.iter().copied().collect::<Result<Vec<_>, _>>() {
                let indices: Vec<usize> = element_rows.iter().map(|&u| exact[u]).collect();
                return f(source, &indices, requested.clone());
            }
            let sampled = SampledEphemeris::new(source, &unique, &locations)?;
            f(&sampled, &element_rows, requested.clone())
        })
    }

    /// Shared body of `in_constraint_elementwise` / `evaluate_elementwise`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn elementwise(
        &self,
        py: Python,
        ephemeris: &Py<PyAny>,
        target_ras: &[f64],
        target_decs: &[f64],
        times: Option<&Bound<PyAny>>,
        target_rolls: Option<&[Option<f64>]>,
        n_roll_samples: usize,
        with_values: bool,
        apparent: bool,
    ) -> PyResult<(Vec<DateTime<Utc>>, DiagonalEvaluation)> {
        Self::validate_n_roll_samples(n_roll_samples)?;
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        if let Some(rolls) = target_rolls {
            if rolls.len() != target_ras.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "target_rolls must have the same length as target_ras and target_decs",
                ));
            }
            rolls
                .iter()
                .flatten()
                .try_for_each(|&roll| Self::validate_target_roll(roll))?;
        }
        let bound = ephemeris.bind(py);
        self.with_elementwise_ephemeris(
            bound,
            target_ras.len(),
            times,
            |ephem, indices, element_times| {
                if apparent {
                    // One apparent row per distinct time; each target is aberrated
                    // for the observer state of its own time.
                    let mut rows = indices.to_vec();
                    rows.sort_unstable();
                    rows.dedup();
                    let (view, aberrations) = SampledEphemeris::apparent(ephem, &rows)?;
                    let local: Vec<usize> = indices
                        .iter()
                        .map(|row| rows.binary_search(row).unwrap_or_else(|i| i))
                        .collect();
                    let (ras, decs): (Vec<f64>, Vec<f64>) = local
                        .iter()
                        .enumerate()
                        .map(|(i, &row)| {
                            aberrations[row].apply_radec_deg(target_ras[i], target_decs[i])
                        })
                        .unzip();
                    let evaluation = evaluate_elements(
                        &*self.evaluator,
                        &view,
                        &ras,
                        &decs,
                        &local,
                        target_rolls,
                        n_roll_samples,
                        with_values,
                    )?;
                    return Ok((element_times, evaluation));
                }
                let evaluation = evaluate_elements(
                    &*self.evaluator,
                    ephem,
                    target_ras,
                    target_decs,
                    indices,
                    target_rolls,
                    n_roll_samples,
                    with_values,
                )?;
                Ok((element_times, evaluation))
            },
        )
    }

    /// Target x time violation grid with apparent geometry: the bodies of each
    /// time are apparent (`SampledEphemeris::apparent`) and every target is
    /// aberrated for that time's observer state, so each column is evaluated
    /// with its own target directions. `kernel` evaluates one column from
    /// RA/Dec; with `unit_vectors` the leaf's unit-vector kernel is tried first,
    /// which avoids the trigonometry (roll-independent trees only).
    pub(super) fn apparent_batch<K>(
        &self,
        ephemeris: &dyn EphemerisBase,
        time_indices: Option<&[usize]>,
        target_ras: &[f64],
        target_decs: &[f64],
        unit_vectors: bool,
        kernel: K,
    ) -> PyResult<ndarray::Array2<bool>>
    where
        K: Fn(&dyn EphemerisBase, &[f64], &[f64], &[usize]) -> PyResult<ndarray::Array2<bool>>,
    {
        let rows: Vec<usize> = match time_indices {
            Some(indices) => indices.to_vec(),
            None => (0..ephemeris.get_times()?.len()).collect(),
        };
        let (view, aberrations) = SampledEphemeris::apparent(ephemeris, &rows)?;
        let catalogue =
            crate::utils::vector_math::radec_to_unit_vectors_batch(target_ras, target_decs);
        let catalogue = catalogue
            .as_slice()
            .expect("unit vectors are built in standard layout");
        let n_targets = target_ras.len();
        let mut result = ndarray::Array2::from_elem((n_targets, rows.len()), false);
        let (mut ras, mut decs) = (vec![0.0; n_targets], vec![0.0; n_targets]);
        // Correcting every target for every time is the one per-cell cost of
        // apparent evaluation. It is plain arithmetic, so a block of times is
        // corrected across the cores before its columns are evaluated in turn
        // (the evaluators themselves are not shared between threads).
        let block = (APPARENT_BLOCK_VALUES / (3 * n_targets).max(1)).clamp(1, 4096);
        let mut proper =
            vec![ndarray::Array2::<f64>::zeros((n_targets, 3)); block.min(aberrations.len())];
        for (start, corrections) in aberrations.chunks(block).enumerate() {
            let start = start * block;
            let correct = |(out, aberration): (&mut ndarray::Array2<f64>, &ObserverAberration)| {
                let out = out.as_slice_mut().expect("standard layout");
                for (out, p) in out
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(catalogue.as_chunks::<3>().0)
                {
                    *out = aberration.apply(*p);
                }
            };
            if corrections.len() * n_targets >= APPARENT_PARALLEL_MIN_CELLS {
                proper
                    .par_iter_mut()
                    .zip(corrections.par_iter())
                    .for_each(correct);
            } else {
                proper.iter_mut().zip(corrections).for_each(correct);
            }
            for (offset, vectors) in proper.iter().take(corrections.len()).enumerate() {
                let selected = [start + offset];
                let fast = if unit_vectors {
                    self.evaluator.in_constraint_batch_unit_vectors(
                        &view,
                        vectors,
                        Some(&selected),
                    )?
                } else {
                    None
                };
                let mask = match fast {
                    Some(mask) => mask,
                    None => {
                        for (i, p) in vectors.rows().into_iter().enumerate() {
                            ras[i] = p[1].atan2(p[0]).to_degrees().rem_euclid(360.0);
                            decs[i] = p[2].atan2(p[0].hypot(p[1])).to_degrees();
                        }
                        kernel(&view, &ras, &decs, &selected)?
                    }
                };
                result.column_mut(start + offset).assign(&mask.column(0));
            }
        }
        Ok(result)
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
}
