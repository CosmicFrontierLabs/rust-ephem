//! Grid (target x time) constraint evaluation with array results: the
//! violation mask, the named values and the per-leaf masks of every target at
//! every time, kept as (n_targets, n_times) arrays instead of being split into
//! one result object per target.
use super::core::{track_violations, ConstraintResult};
use crate::utils::time_utils::utc_to_python_datetime;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// What one evaluation of a constraint tree over a grid produces.
pub struct GridEvaluation {
    /// The constraint's name at the roll evaluated. For a grid assembled from
    /// targets at different rolls it is the name of the definition, and
    /// `target_names` holds each target's own.
    pub name: String,
    /// Per target, the name at the roll that target was evaluated at; `None`
    /// when every target shares `name`.
    pub target_names: Option<Vec<Arc<str>>>,
    /// The evaluated times, one per column.
    pub times: Arc<[DateTime<Utc>]>,
    /// (targets x times), `true` where the constraint is violated.
    pub violated: Array2<bool>,
    /// Per-leaf violation masks, keyed by cause tag.
    pub named: HashMap<String, Array2<bool>>,
    /// Named continuous values.
    pub values: HashMap<String, Array2<f64>>,
    /// Cause tag -> the value key(s) it corresponds to.
    pub cause_value_keys: Arc<HashMap<String, Vec<String>>>,
}

impl GridEvaluation {
    /// The result of one target: its violation windows over the evaluated
    /// times, with its row of every value and per-leaf mask.
    pub fn target(&self, index: usize) -> ConstraintResult {
        let name = self.target_name(index);
        let violated = self.violated.row(index);
        let violations = track_violations(
            &self.times,
            |i| (violated[i], if violated[i] { 1.0 } else { 0.0 }),
            |_i, _is_open| name.to_string(),
        );
        let all_satisfied = violations.is_empty();
        let row = |array: &Array2<f64>| array.row(index).to_vec();
        ConstraintResult::new(
            violations,
            all_satisfied,
            name.to_string(),
            self.times.clone(),
        )
        .with_constraint_values(
            self.values
                .iter()
                .map(|(key, array)| (key.clone(), row(array)))
                .collect(),
        )
        .with_component_violated(
            self.named
                .iter()
                .map(|(key, array)| (key.clone(), array.row(index).to_vec()))
                .collect(),
        )
        .with_cause_value_keys(self.cause_value_keys.clone())
    }

    /// The constraint's name at the roll target `index` was evaluated at.
    pub fn target_name(&self, index: usize) -> &str {
        match &self.target_names {
            Some(names) => &names[index],
            None => &self.name,
        }
    }

    /// One result per target, in target order.
    pub fn into_results(self) -> Vec<ConstraintResult> {
        (0..self.violated.nrows())
            .map(|index| self.target(index))
            .collect()
    }

    /// Assemble the grid of `n_targets` targets from grids of disjoint subsets
    /// of them: `parts` pairs the target indices of a subset with its grid.
    /// Every part covers the same times and reports the same keys.
    ///
    /// Each part is named for the roll it was evaluated at, and its targets
    /// keep that name. The grid as a whole takes it too when the parts agree,
    /// and `definition_name` (the constraint as defined) when they do not.
    pub fn from_target_groups(
        definition_name: String,
        n_targets: usize,
        mut parts: Vec<(Vec<usize>, GridEvaluation)>,
    ) -> Option<GridEvaluation> {
        // The usual case: every target evaluated together, in order.
        if let [(targets, _)] = parts.as_slice() {
            if targets.iter().copied().eq(0..n_targets) {
                return parts.pop().map(|(_, grid)| grid);
            }
        }
        let (_, first) = parts.first()?;
        let n_times = first.violated.ncols();
        let shared = parts.iter().all(|(_, part)| part.name == first.name);
        let target_names = (!shared).then(|| {
            let mut names: Vec<Arc<str>> = vec![Arc::from(""); n_targets];
            for (targets, part) in &parts {
                let name: Arc<str> = Arc::from(part.name.as_str());
                for &target in targets {
                    names[target] = name.clone();
                }
            }
            names
        });
        let mut merged = GridEvaluation {
            name: if shared {
                first.name.clone()
            } else {
                definition_name
            },
            target_names,
            times: first.times.clone(),
            violated: Array2::from_elem((n_targets, n_times), false),
            named: first
                .named
                .keys()
                .map(|key| (key.clone(), Array2::from_elem((n_targets, n_times), false)))
                .collect(),
            values: first
                .values
                .keys()
                .map(|key| {
                    (
                        key.clone(),
                        Array2::from_elem((n_targets, n_times), f64::NAN),
                    )
                })
                .collect(),
            cause_value_keys: first.cause_value_keys.clone(),
        };
        for (targets, part) in &parts {
            for (local, &target) in targets.iter().enumerate() {
                merged
                    .violated
                    .row_mut(target)
                    .assign(&part.violated.row(local));
                for (key, array) in &part.named {
                    if let Some(out) = merged.named.get_mut(key) {
                        out.row_mut(target).assign(&array.row(local));
                    }
                }
                for (key, array) in &part.values {
                    if let Some(out) = merged.values.get_mut(key) {
                        out.row_mut(target).assign(&array.row(local));
                    }
                }
            }
        }
        Some(merged)
    }
}

/// Result of `Constraint.evaluate_grid`: every target at every time, as
/// `(n_targets, n_times)` arrays.
#[pyclass(name = "GridResult")]
pub struct GridResult {
    grid: GridEvaluation,
    ras: Vec<f64>,
    decs: Vec<f64>,
    constraint_array_cache: OnceLock<Py<PyAny>>,
    visibility_flags_cache: OnceLock<Py<PyAny>>,
    constraint_values_cache: OnceLock<Py<PyAny>>,
    component_violated_cache: OnceLock<Py<PyAny>>,
    timestamp_cache: OnceLock<Py<PyAny>>,
}

impl GridResult {
    pub fn new(grid: GridEvaluation, ras: Vec<f64>, decs: Vec<f64>) -> Self {
        Self {
            grid,
            ras,
            decs,
            constraint_array_cache: OnceLock::new(),
            visibility_flags_cache: OnceLock::new(),
            constraint_values_cache: OnceLock::new(),
            component_violated_cache: OnceLock::new(),
            timestamp_cache: OnceLock::new(),
        }
    }
}

/// Mark a freshly built NumPy array read-only: it is about to be cached and
/// handed to every later caller.
pub(crate) fn freeze<T, D>(array: &Bound<'_, numpy::PyArray<T, D>>) {
    use numpy::PyUntypedArrayMethods;
    // SAFETY: `array` was created by the caller and has not been shared yet,
    // so nothing else can be reading or writing its flags.
    unsafe {
        (*array.as_array_ptr()).flags &= !numpy::npyffi::NPY_ARRAY_WRITEABLE;
    }
}

/// A read-only NumPy array of `array`.
fn read_only<T: numpy::Element>(py: Python<'_>, array: Array2<T>) -> PyResult<Py<PyAny>> {
    let array = array.into_pyarray(py);
    freeze(&array);
    Ok(array.into_any().unbind())
}

/// Build a value on first use and hand out the same object afterwards.
fn cached(
    py: Python,
    cache: &OnceLock<Py<PyAny>>,
    build: impl FnOnce() -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    if let Some(value) = cache.get() {
        return Ok(value.clone_ref(py));
    }
    let value = build()?;
    Ok(cache.get_or_init(|| value).clone_ref(py))
}

/// A fresh dict over `arrays`, so that adding or removing keys on the caller's
/// side leaves the cached arrays' dict alone.
fn dict_copy(py: Python, dict: Py<PyAny>) -> PyResult<Py<PyAny>> {
    Ok(dict.bind(py).call_method0("copy")?.unbind())
}

#[pymethods]
impl GridResult {
    fn __repr__(&self) -> String {
        let (n_targets, n_times) = self.grid.violated.dim();
        format!(
            "GridResult(constraint='{}', n_targets={n_targets}, n_times={n_times}, n_violated={})",
            self.grid.name,
            self.grid.violated.iter().filter(|&&v| v).count()
        )
    }

    /// `(n_targets, n_times)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.grid.violated.dim()
    }

    /// Constraint name/description, at the roll the targets were evaluated
    /// at. When they were evaluated at different rolls it is the name of the
    /// constraint as defined; `target(i).constraint_name` is each target's own.
    #[getter]
    fn constraint_name(&self) -> &str {
        &self.grid.name
    }

    /// Map from a `component_violated` cause tag to the `constraint_values`
    /// key(s) it corresponds to (see `ConstraintResult.cause_value_keys`).
    #[getter]
    fn cause_value_keys(&self) -> HashMap<String, Vec<String>> {
        (*self.grid.cause_value_keys).clone()
    }

    /// numpy bool array (n_targets, n_times): True where the constraint is
    /// VIOLATED. Equal to `in_constraint_batch`. Read-only.
    #[getter]
    fn constraint_array(&self, py: Python) -> PyResult<Py<PyAny>> {
        cached(py, &self.constraint_array_cache, || {
            read_only(py, self.grid.violated.clone())
        })
    }

    /// numpy bool array (n_targets, n_times): True where the target is visible
    /// (the inverse of `constraint_array`). Read-only.
    #[getter]
    fn visibility_flags(&self, py: Python) -> PyResult<Py<PyAny>> {
        cached(py, &self.visibility_flags_cache, || {
            read_only(py, self.grid.violated.mapv(|violated| !violated))
        })
    }

    /// dict[str, numpy float array (n_targets, n_times)] of the named
    /// continuous values, with the keys `evaluate_batch` reports. The arrays
    /// are read-only and shared between accesses.
    #[getter]
    fn constraint_values(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = cached(py, &self.constraint_values_cache, || {
            let dict = pyo3::types::PyDict::new(py);
            for (key, values) in &self.grid.values {
                dict.set_item(key, read_only(py, values.clone())?)?;
            }
            Ok(dict.into_any().unbind())
        })?;
        dict_copy(py, dict)
    }

    /// dict[str, numpy bool array (n_targets, n_times)]: per-leaf violation
    /// masks keyed by cause tag (e.g. `sun`, `moon`). The arrays are read-only
    /// and shared between accesses.
    #[getter]
    fn component_violated(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = cached(py, &self.component_violated_cache, || {
            let dict = pyo3::types::PyDict::new(py);
            for (key, values) in &self.grid.named {
                dict.set_item(key, read_only(py, values.clone())?)?;
            }
            Ok(dict.into_any().unbind())
        })?;
        dict_copy(py, dict)
    }

    /// numpy float array (n_targets,) of target right ascensions in degrees.
    /// A new copy on each access.
    #[getter]
    fn ras(&self, py: Python) -> Py<PyAny> {
        self.ras.clone().into_pyarray(py).into_any().unbind()
    }

    /// numpy float array (n_targets,) of target declinations in degrees.
    /// A new copy on each access.
    #[getter]
    fn decs(&self, py: Python) -> Py<PyAny> {
        self.decs.clone().into_pyarray(py).into_any().unbind()
    }

    /// numpy object array (n_times,) of the UTC datetimes evaluated. Built on
    /// first access and shared by later ones, so read-only.
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Py<PyAny>> {
        cached(py, &self.timestamp_cache, || {
            let list = pyo3::types::PyList::empty(py);
            for time in self.grid.times.iter() {
                list.append(utc_to_python_datetime(py, time)?)?;
            }
            let numpy = pyo3::types::PyModule::import(py, "numpy")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("dtype", "object")?;
            let array = numpy.call_method("array", (list,), Some(&kwargs))?;
            let read_only = pyo3::types::PyDict::new(py);
            read_only.set_item("write", false)?;
            array.call_method("setflags", (), Some(&read_only))?;
            Ok(array.unbind())
        })
    }

    /// The `ConstraintResult` of one target: its violation and visibility
    /// windows over the evaluated times, built on demand from row `index`.
    fn target(&self, index: isize) -> PyResult<ConstraintResult> {
        let n_targets = self.grid.violated.nrows() as isize;
        let resolved = if index < 0 { index + n_targets } else { index };
        if !(0..n_targets).contains(&resolved) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "target index {index} out of range for {n_targets} targets"
            )));
        }
        Ok(self.grid.target(resolved as usize))
    }
}
