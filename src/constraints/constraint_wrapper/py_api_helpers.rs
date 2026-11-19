use crate::constraints::core::{
    sweep_rolls_with_attribution, track_violations, ConstraintEvaluator, ConstraintResult,
    EvaluationRoll,
};
use crate::constraints::diagonal::DiagonalEvaluation;
use crate::constraints::elementwise::evaluate_elements;
use crate::constraints::grid::GridEvaluation;
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

/// Paired elements above which their per-element set-up is shared across the cores.
const ELEMENTWISE_PARALLEL_MIN_ROWS: usize = 8192;

/// Cells of an apparent grid evaluated as paired elements in one call.
const APPARENT_PAIRED_BLOCK_CELLS: usize = 1 << 20;

/// Read a 1-D sequence of numbers as `f64`s. A NumPy array is copied in one
/// step; anything else goes through `numpy.asarray`, so lists, tuples and
/// arrays of other dtypes are accepted alike.
pub(super) fn extract_f64s(values: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    use numpy::{PyArray1, PyArrayMethods};

    let invalid = |reason: String| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be a 1-D sequence of numbers: {reason}"
        ))
    };
    let array = match values.cast::<PyArray1<f64>>() {
        Ok(array) => array.clone(),
        Err(_) => {
            let converted = values
                .py()
                .import("numpy")?
                .call_method1("asarray", (values, "float64"))
                .map_err(|err| invalid(err.to_string()))?;
            converted
                .cast_into::<PyArray1<f64>>()
                .map_err(|_| invalid("it is not one-dimensional".to_string()))?
        }
    };
    let values = array.readonly().as_array().to_vec();
    Ok(values)
}

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
        let mut requested = times.map(Self::parse_datetime_input).transpose()?;
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
            let Some(requested) = requested.take() else {
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
            // Elements usually share few distinct times, all on the grid:
            // each is then found directly, with nothing to sort.
            let find = |time: &DateTime<Utc>| source_times.binary_search(time).ok();
            let on_grid: Option<Vec<usize>> = if requested.len() >= ELEMENTWISE_PARALLEL_MIN_ROWS {
                requested.par_iter().map(find).collect()
            } else {
                requested.iter().map(find).collect()
            };
            if let Some(indices) = on_grid {
                return f(source, &indices, requested);
            }
            let mut unique = requested.clone();
            unique.par_sort_unstable();
            unique.dedup();
            let locations = locate_times(source_times, &unique)?;
            let row = |time: &DateTime<Utc>| unique.binary_search(time).unwrap_or_else(|i| i);
            let element_rows: Vec<usize> = if requested.len() >= ELEMENTWISE_PARALLEL_MIN_ROWS {
                requested.par_iter().map(row).collect()
            } else {
                requested.iter().map(row).collect()
            };
            let sampled = SampledEphemeris::new(source, &unique, &locations)?;
            f(&sampled, &element_rows, requested)
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
                    let n_source = ephem.get_times()?.len();
                    if let Some(row) = indices.iter().find(|&&row| row >= n_source) {
                        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                            "index {row} out of bounds for ephemeris with {n_source} timestamps"
                        )));
                    }
                    // The distinct rows in order, and each one's place among them.
                    let mut place = vec![usize::MAX; n_source];
                    for &row in indices {
                        place[row] = 0;
                    }
                    let rows: Vec<usize> = (0..n_source).filter(|&row| place[row] == 0).collect();
                    for (position, &row) in rows.iter().enumerate() {
                        place[row] = position;
                    }
                    let (view, aberrations) = SampledEphemeris::apparent(ephem, &rows)?;
                    let local: Vec<usize> = indices.iter().map(|&row| place[row]).collect();
                    let correct = |i: usize| {
                        aberrations[local[i]].apply_radec_deg(target_ras[i], target_decs[i])
                    };
                    let (ras, decs): (Vec<f64>, Vec<f64>) =
                        if local.len() >= ELEMENTWISE_PARALLEL_MIN_ROWS {
                            (0..local.len()).into_par_iter().map(correct).unzip()
                        } else {
                            (0..local.len()).map(correct).unzip()
                        };
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

    /// The apparent target x time violation grid of a tree with an expensive
    /// member, evaluated as paired elements: every cell in one pass across the
    /// cores, each tree member seeing only the cells the cheaper ones left
    /// undecided. `apparent_batch` hands such a tree one time column at a
    /// time, which keeps it on one core and repeats its setup per column.
    ///
    /// `roll` is the explicit spacecraft roll of the grid, if any. `None` when
    /// this does not apply (a cheap tree, a free roll on a roll-dependent tree,
    /// or no paired kernel), and `apparent_batch` is to be used.
    pub(super) fn apparent_mask_paired(
        &self,
        ephemeris: &dyn EphemerisBase,
        time_indices: Option<&[usize]>,
        target_ras: &[f64],
        target_decs: &[f64],
        roll: Option<f64>,
    ) -> PyResult<Option<ndarray::Array2<bool>>> {
        let evaluator = &*self.evaluator;
        if !evaluator.is_expensive() || (roll.is_none() && evaluator.is_roll_dependent()) {
            return Ok(None);
        }
        let rows: Vec<usize> = match time_indices {
            Some(indices) => indices.to_vec(),
            None => (0..ephemeris.get_times()?.len()).collect(),
        };
        let (view, aberrations) = SampledEphemeris::apparent(ephemeris, &rows)?;
        let (n_targets, n_times) = (target_ras.len(), rows.len());
        let mut result = ndarray::Array2::from_elem((n_targets, n_times), false);
        if n_targets == 0 || n_times == 0 {
            return Ok(Some(result));
        }
        // Whole targets per call, so the paired lists stay bounded.
        let targets_per_block = (APPARENT_PAIRED_BLOCK_CELLS / n_times).max(1);
        let local: Vec<usize> = (0..targets_per_block.min(n_targets))
            .flat_map(|_| 0..n_times)
            .collect();
        for (block, (ras, decs)) in target_ras
            .chunks(targets_per_block)
            .zip(target_decs.chunks(targets_per_block))
            .enumerate()
        {
            let cells = ras.len() * n_times;
            let (cell_ras, cell_decs): (Vec<f64>, Vec<f64>) = (0..cells)
                .into_par_iter()
                .map(|c| {
                    aberrations[c % n_times].apply_radec_deg(ras[c / n_times], decs[c / n_times])
                })
                .unzip();
            let rolls = roll.map(|roll| vec![Some(roll); cells]);
            let Some(mask) = evaluator.mask_diagonal(
                &view,
                &cell_ras,
                &cell_decs,
                Some(&local[..cells]),
                rolls.as_deref(),
            )?
            else {
                return Ok(None);
            };
            let first = block * targets_per_block;
            for (c, violated) in mask.into_iter().enumerate() {
                result[[first + c / n_times, c % n_times]] = violated;
            }
        }
        Ok(Some(result))
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
        // Trees whose kernels correct the targets themselves keep the grid in
        // one call, vectorised across times.
        if unit_vectors && self.evaluator.applies_target_corrections() {
            let view = view.with_target_corrections(aberrations);
            return match self
                .evaluator
                .in_constraint_batch_unit_vectors(&view, &catalogue, None)?
            {
                Some(mask) => Ok(mask),
                None => self
                    .evaluator
                    .in_constraint_batch(&view, target_ras, target_decs, None),
            };
        }
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

    /// The grid with apparent geometry: the bodies of each time are apparent
    /// (`SampledEphemeris::apparent`) and every target is corrected for that
    /// time's observer state.
    ///
    /// A tree whose kernels correct targets themselves is evaluated in one
    /// grid call, as `apparent_batch` does for the mask alone. Any other tree
    /// needs each target corrected per time before it is handed over, which is
    /// done along the shorter side of the grid: per time column for a wide
    /// grid, per target for a long one.
    pub(super) fn grid_apparent(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
        target_roll: Option<f64>,
    ) -> PyResult<GridEvaluation> {
        let evaluator = &*self.evaluator;
        let rows: Vec<usize> = match time_indices {
            Some(indices) => indices.to_vec(),
            None => (0..ephemeris.get_times()?.len()).collect(),
        };
        let (view, aberrations) = SampledEphemeris::apparent(ephemeris, &rows)?;
        let (n_targets, n_times) = (target_ras.len(), rows.len());
        if evaluator.applies_target_corrections() && !evaluator.is_roll_dependent() {
            let view = view.with_target_corrections(aberrations);
            return Self::grid_geometric(
                evaluator,
                &view,
                target_ras,
                target_decs,
                None,
                n_roll_samples,
                target_roll,
            );
        }
        if n_targets == 0 || n_times == 0 {
            // Nothing to correct: the kernels report the empty grid's schema.
            return Self::grid_geometric(
                evaluator,
                &view,
                target_ras,
                target_decs,
                None,
                n_roll_samples,
                target_roll,
            );
        }

        let times: std::sync::Arc<[DateTime<Utc>]> = view.get_times()?.into();
        let mut violated = ndarray::Array2::from_elem((n_targets, n_times), false);
        let mut named: HashMap<String, ndarray::Array2<bool>> = HashMap::new();
        let mut values: HashMap<String, ndarray::Array2<f64>> = HashMap::new();
        if n_targets > n_times {
            // Wide: one time at a time, every target corrected for it.
            for (column, aberration) in aberrations.iter().enumerate() {
                let correct = |(&ra, &dec): (&f64, &f64)| aberration.apply_radec_deg(ra, dec);
                let (ras, decs): (Vec<f64>, Vec<f64>) = if n_targets >= APPARENT_PARALLEL_MIN_CELLS
                {
                    target_ras
                        .par_iter()
                        .zip(target_decs.par_iter())
                        .map(correct)
                        .unzip()
                } else {
                    target_ras.iter().zip(target_decs).map(correct).unzip()
                };
                let part = Self::grid_geometric(
                    evaluator,
                    &view,
                    &ras,
                    &decs,
                    Some(&[column]),
                    n_roll_samples,
                    target_roll,
                )?;
                violated.column_mut(column).assign(&part.violated.column(0));
                for (key, array) in part.named {
                    named
                        .entry(key)
                        .or_insert_with(|| ndarray::Array2::from_elem((n_targets, n_times), false))
                        .column_mut(column)
                        .assign(&array.column(0));
                }
                for (key, array) in part.values {
                    values
                        .entry(key)
                        .or_insert_with(|| {
                            ndarray::Array2::from_elem((n_targets, n_times), f64::NAN)
                        })
                        .column_mut(column)
                        .assign(&array.column(0));
                }
            }
        } else {
            // Long: one target at a time, corrected for every time and
            // evaluated as paired elements.
            let local: Vec<usize> = (0..n_times).collect();
            let rolls = target_roll.map(|roll| vec![Some(roll); n_times]);
            for (row, (&ra, &dec)) in target_ras.iter().zip(target_decs).enumerate() {
                let (ras, decs): (Vec<f64>, Vec<f64>) = aberrations
                    .iter()
                    .map(|aberration| aberration.apply_radec_deg(ra, dec))
                    .unzip();
                let evaluation = evaluate_elements(
                    evaluator,
                    &view,
                    &ras,
                    &decs,
                    &local,
                    rolls.as_deref(),
                    n_roll_samples,
                    true,
                )?;
                for (cell, flag) in violated.row_mut(row).iter_mut().zip(evaluation.violated) {
                    *cell = flag;
                }
                for (key, flags) in evaluation.named {
                    let array = named
                        .entry(key)
                        .or_insert_with(|| ndarray::Array2::from_elem((n_targets, n_times), false));
                    for (cell, flag) in array.row_mut(row).iter_mut().zip(flags) {
                        *cell = flag;
                    }
                }
                for (key, numbers) in evaluation.values {
                    let array = values.entry(key).or_insert_with(|| {
                        ndarray::Array2::from_elem((n_targets, n_times), f64::NAN)
                    });
                    for (cell, number) in array.row_mut(row).iter_mut().zip(numbers) {
                        *cell = number;
                    }
                }
            }
        }
        Ok(GridEvaluation {
            name: evaluator.name_at_roll(target_roll),
            times,
            violated,
            named,
            values,
            cause_value_keys: std::sync::Arc::new(Self::cause_value_keys_for(evaluator, &view)?),
        })
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
            .map(|(key, arr)| (key.clone(), arr.row(target_index).to_vec()))
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

    /// The grid of the tree: the combined mask, the per-leaf masks and the
    /// named values of every target at the selected times, with the roll
    /// semantics of `evaluate_batch` (`target_roll = None` sweeps the rolls of a
    /// roll-dependent tree).
    pub(super) fn grid_geometric(
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
        target_roll: Option<f64>,
    ) -> PyResult<GridEvaluation> {
        // See `eval_with_ephemeris`: one coordinated sweep produces both the
        // violation array and the cause masks consistent with it.
        let swept = if let Some(roll) = target_roll {
            evaluator.in_constraint_batch_with_causes(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                Some(EvaluationRoll::Fixed(roll)),
            )?
        } else {
            sweep_rolls_with_attribution(
                evaluator,
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                n_roll_samples,
            )?
        };
        Ok(GridEvaluation {
            name: evaluator.name_at_roll(target_roll),
            // Times, the cause-key mapping and the name are the same for every
            // target: built once and shared.
            times: ephemeris.get_times_at_indices(time_indices)?.into(),
            violated: swept.violated,
            named: swept.named,
            // Named continuous values, once for all targets (not swept across
            // roll angles).
            values: evaluator.compute_named_values_at_roll(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                target_roll,
            )?,
            // The mapping depends only on the evaluator's structure, not on
            // the target positions.
            cause_value_keys: std::sync::Arc::new(Self::cause_value_keys_for(
                evaluator, ephemeris,
            )?),
        })
    }

    /// The grid, geometric or apparent.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn grid(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        n_roll_samples: usize,
        target_roll: Option<f64>,
        apparent: bool,
    ) -> PyResult<GridEvaluation> {
        if apparent {
            self.grid_apparent(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                n_roll_samples,
                target_roll,
            )
        } else {
            Self::grid_geometric(
                &*self.evaluator,
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                n_roll_samples,
                target_roll,
            )
        }
    }
}
