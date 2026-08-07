/// Constraint system for calculating when astronomical constraints are satisfied
///
/// This module provides a generic constraint API for evaluating constraints on
/// astronomical observations, such as:
/// - Sun proximity constraints
/// - Moon proximity constraints
/// - Eclipse constraints
/// - Logical combinations of constraints (AND, OR, NOT)
///
/// Constraints operate on ephemeris data and target coordinates to produce
/// time-based violation windows.
use crate::utils::time_utils::{python_datetime_to_utc, utc_to_python_datetime};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

/// Short, stable tag for a leaf/combinator constraint's `values`/`component_violated`
/// key prefix, derived from the text before the first `(` in its `name()`. Falls back
/// to a sanitized version of the head for anything unmapped, so new constraint types
/// don't panic or silently collide — they just get a slightly uglier (but unique-ish)
/// prefix.
pub(crate) fn value_key_prefix(name: &str) -> String {
    let head = name.split('(').next().unwrap_or(name).trim();
    let tag = match head {
        "SunProximity" => "sun",
        "MoonProximity" => "moon",
        "BodyProximity" => "body",
        "Eclipse" => "eclipse",
        "AirmassConstraint" => "airmass",
        "AltAzConstraint" => "alt_az",
        "SAAConstraint" => "saa",
        "DaytimeConstraint" => "daytime",
        "MoonPhaseConstraint" => "moon_phase",
        "OrbitPoleConstraint" => "orbit_pole",
        "OrbitRamConstraint" => "orbit_ram",
        "EarthLimb" => "earth_limb",
        "BrightStar" => "bright_star",
        "SolarRollConstraint" => "solar_roll",
        "AND" => "and",
        "OR" => "or",
        "NOT" => "not",
        "XOR" => "xor",
        "AT_LEAST" => "at_least",
        "BoresightOffset" => "boresight_offset",
        other => {
            return other
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        }
    };
    tag.to_string()
}

/// Names of leaf constraints (keyed the same way as `constraint_values`) whose own
/// violation state differs between `idx_before` and `idx_after`, sorted for
/// determinism. Used to attribute a `VisibilityWindow`'s `start_cause`/`end_cause`.
/// Returns `None` if no leaf's state actually changed between the two samples.
fn flipped_causes(
    component_violated: &HashMap<String, Vec<bool>>,
    idx_before: usize,
    idx_after: usize,
) -> Option<Vec<String>> {
    let mut causes: Vec<String> = component_violated
        .iter()
        .filter(|(_, arr)| arr.get(idx_before) != arr.get(idx_after))
        .map(|(key, _)| key.clone())
        .collect();
    causes.sort();
    if causes.is_empty() {
        None
    } else {
        Some(causes)
    }
}

/// Result of constraint evaluation
///
/// Contains information about when and where a constraint is violated.
#[pyclass(name = "ConstraintViolation", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ConstraintViolation {
    /// Start time of the violation window (internal storage)
    pub start_time_internal: DateTime<Utc>,
    /// End time of the violation window (internal storage)
    pub end_time_internal: DateTime<Utc>,
    /// Maximum severity of violation in this window (0.0 = just violated, 1.0+ = severe)
    #[pyo3(get)]
    pub max_severity: f64,
    /// Human-readable description of the violation
    #[pyo3(get)]
    pub description: String,
}

#[pymethods]
impl ConstraintViolation {
    #[getter]
    fn start_time(&self, py: Python) -> PyResult<Py<PyAny>> {
        utc_to_python_datetime(py, &self.start_time_internal)
    }

    #[getter]
    fn end_time(&self, py: Python) -> PyResult<Py<PyAny>> {
        utc_to_python_datetime(py, &self.end_time_internal)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConstraintViolation(start='{}', end='{}', max_severity={:.3}, description='{}')",
            self.start_time_internal.to_rfc3339(),
            self.end_time_internal.to_rfc3339(),
            self.max_severity,
            self.description
        )
    }
}

/// Visibility window indicating when target is not constrained
#[pyclass(name = "VisibilityWindow")]
pub struct VisibilityWindow {
    /// Start time of the visibility window
    #[pyo3(get)]
    pub start_time: Py<PyAny>, // Python datetime object
    /// End time of the visibility window
    #[pyo3(get)]
    pub end_time: Py<PyAny>, // Python datetime object
    /// Namespaced tag(s) of the leaf constraint(s) whose own pass/fail state changed
    /// between the sample immediately before this window started and this window's
    /// first sample (same tags as `ConstraintResult.constraint_values` keys' prefixes,
    /// e.g. "sun", "moon"). This reports *that* a leaf's own state flipped, not the
    /// direction of the flip relative to the overall (possibly negated) result — e.g.
    /// under a `NOT` wrapper, a `"not.sun"` entry means the underlying `sun` leaf's own
    /// state changed, which is the opposite transition to the `NOT` result's own
    /// violated/satisfied transition. `None` if the window starts at the first
    /// evaluated sample (no prior sample to compare against).
    ///
    /// Under a free-roll evaluation (`target_roll=None` on a roll-dependent tree)
    /// each leaf's state is taken at that sample's *witness roll* — the orientation
    /// the sweep selected, see `sweep_rolls_with_attribution`. Both samples either
    /// side of a boundary therefore describe orientations the spacecraft could
    /// actually hold, but they need not be the *same* orientation: a window can
    /// open because the spacecraft would have to roll elsewhere, not because any
    /// leaf changed at a fixed attitude.
    #[pyo3(get)]
    pub start_cause: Option<Vec<String>>,
    /// Namespaced tag(s) of the leaf constraint(s) whose own pass/fail state changed
    /// between this window's last sample and the sample immediately after this window
    /// ended. See `start_cause` for why no flip direction is implied. `None` if the
    /// window ends at the last evaluated sample.
    #[pyo3(get)]
    pub end_cause: Option<Vec<String>>,
}

#[pymethods]
impl VisibilityWindow {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let start_str = self.start_time.bind(py).str()?.to_string();
        let end_str = self.end_time.bind(py).str()?.to_string();
        let duration = self.duration_seconds(py)?;
        Ok(format!(
            "VisibilityWindow(start_time={}, end_time={}, duration_seconds={}, start_cause={:?}, end_cause={:?})",
            start_str, end_str, duration, self.start_cause, self.end_cause
        ))
    }
    #[getter]
    fn duration_seconds(&self, py: Python) -> PyResult<f64> {
        let start_dt = python_datetime_to_utc(self.start_time.bind(py))?;
        let end_dt = python_datetime_to_utc(self.end_time.bind(py))?;
        let duration = end_dt.signed_duration_since(start_dt);
        Ok(duration.num_seconds() as f64)
    }
}

/// Result of constraint evaluation containing all violations
#[pyclass(name = "ConstraintResult")]
pub struct ConstraintResult {
    /// List of time windows where the constraint was violated
    #[pyo3(get)]
    pub violations: Vec<ConstraintViolation>,
    /// Whether the constraint was satisfied for the entire time range
    #[pyo3(get)]
    pub all_satisfied: bool,
    /// Constraint name/description
    #[pyo3(get)]
    pub constraint_name: String,
    /// Named continuous values computed during evaluation (e.g. `sun_angle_deg`),
    /// one array per key, aligned with `times`/`timestamp`.
    #[pyo3(get)]
    pub constraint_values: HashMap<String, Vec<f64>>,
    /// Per-leaf-constraint violation mask (namespaced the same way as
    /// `constraint_values`' keys), aligned with `times`. Used to attribute
    /// `VisibilityWindow.start_cause`/`end_cause`; not directly exposed to Python.
    component_violated: HashMap<String, Vec<bool>>,
    /// Map from a `start_cause`/`end_cause` tag to the `constraint_values` key(s)
    /// it corresponds to. The two namespaces use different prefixing conventions
    /// (cause tags are flat and nesting-stable; value keys are hierarchical and
    /// path-based) so are not derivable from one another by string matching — use
    /// this instead of guessing.
    #[pyo3(get)]
    pub cause_value_keys: HashMap<String, Vec<String>>,
    /// Evaluation times as Rust DateTime<Utc>, not directly exposed to Python
    pub times: Vec<DateTime<Utc>>,
    /// Step size in seconds between timestamps (for O(1) index lookup)
    step_seconds: i64,
    /// Cached Python timestamp array (not directly exposed, use getter)
    timestamp_cache: OnceLock<Py<PyAny>>,
    /// Cached constraint vector (Rust-side, used by both constraint_array and visibility)
    constraint_vec_cache: OnceLock<Vec<bool>>,
    /// Cached constraint array (Python-side, not directly exposed, use getter)
    constraint_array_cache: OnceLock<Py<PyAny>>,
}

impl ConstraintResult {
    /// Create a new ConstraintResult with initialized caches
    pub fn new(
        violations: Vec<ConstraintViolation>,
        all_satisfied: bool,
        constraint_name: String,
        times: Vec<DateTime<Utc>>,
    ) -> Self {
        // Compute step size from first two timestamps (0 if fewer than 2 times)
        let step_seconds = if times.len() >= 2 {
            (times[1] - times[0]).num_seconds()
        } else {
            0
        };
        Self {
            violations,
            all_satisfied,
            constraint_name,
            constraint_values: HashMap::new(),
            component_violated: HashMap::new(),
            cause_value_keys: HashMap::new(),
            times,
            step_seconds,
            timestamp_cache: OnceLock::new(),
            constraint_vec_cache: OnceLock::new(),
            constraint_array_cache: OnceLock::new(),
        }
    }

    /// Attach named continuous values computed during evaluation.
    pub fn with_constraint_values(mut self, constraint_values: HashMap<String, Vec<f64>>) -> Self {
        self.constraint_values = constraint_values;
        self
    }

    /// Attach per-leaf-constraint violation masks computed during evaluation.
    pub fn with_component_violated(
        mut self,
        component_violated: HashMap<String, Vec<bool>>,
    ) -> Self {
        self.component_violated = component_violated;
        self
    }

    /// Attach the cause-tag → constraint_values-key mapping computed during evaluation.
    pub fn with_cause_value_keys(mut self, cause_value_keys: HashMap<String, Vec<String>>) -> Self {
        self.cause_value_keys = cause_value_keys;
        self
    }
}

#[pymethods]
impl ConstraintResult {
    fn __repr__(&self) -> String {
        format!(
            "ConstraintResult(constraint='{}', violations={}, all_satisfied={})",
            self.constraint_name,
            self.violations.len(),
            self.all_satisfied
        )
    }

    /// Get the total duration of violations in seconds
    fn total_violation_duration(&self) -> PyResult<f64> {
        let mut total_seconds = 0.0;
        for violation in &self.violations {
            let start = violation.start_time_internal;
            let end = violation.end_time_internal;
            total_seconds += (end - start).num_seconds() as f64;
        }
        Ok(total_seconds)
    }

    /// Internal: get cached constraint vector, computing if necessary
    ///
    /// NOTE: This returns a *violation mask* where True means the constraint
    /// is violated (target NOT visible) at that timestamp. The public
    /// `constraint_array` property therefore exposes violation semantics
    /// (True == violated) to Python; visibility windows are computed by
    /// inverting this mask.
    fn _get_constraint_vec(&self) -> &Vec<bool> {
        self.constraint_vec_cache.get_or_init(|| {
            if self.times.is_empty() {
                return Vec::new();
            }

            // Pre-allocate result vector: default false == not violated
            let mut violated = vec![false; self.times.len()];

            // Early return if no violations (all false)
            if self.violations.is_empty() {
                return violated;
            }

            // Mark violated times - violations are already sorted by time
            for (i, t) in self.times.iter().enumerate() {
                // Binary search could be used here, but violation count is typically small
                for v in &self.violations {
                    if t < &v.start_time_internal {
                        break; // Violations are sorted, no need to check further
                    }
                    if &v.start_time_internal <= t && t <= &v.end_time_internal {
                        violated[i] = true;
                        break;
                    }
                }
            }
            violated
        })
    }

    /// Property: array of booleans for each timestamp where True means constraint violated
    #[getter]
    fn constraint_array(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Use cached Python value if available
        if let Some(cached) = self.constraint_array_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        // Get cached Rust vector (computes if needed), convert to Python list
        // Return a Python list of bools (True == violated) so indexing yields
        // native Python bool values. Tests historically expect identity
        // comparisons ("is True"), so returning Python bools is safer.
        let arr = self._get_constraint_vec();
        let py_list = pyo3::types::PyList::empty(py);
        for b in arr {
            py_list.append(pyo3::types::PyBool::new(py, *b))?;
        }
        let py_obj: Py<PyAny> = py_list.into();

        // Cache the Python result (ignore if already set by another thread)
        let _ = self.constraint_array_cache.set(py_obj.clone_ref(py));

        Ok(py_obj)
    }

    /// Property: array of Python datetime objects for each evaluation time (as numpy array)
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Use cached value if available
        if let Some(cached) = self.timestamp_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        // Import numpy
        let np = pyo3::types::PyModule::import(py, "numpy")
            .map_err(|_| pyo3::exceptions::PyImportError::new_err("numpy is required"))?;

        // Build list of Python datetime objects
        let py_list = pyo3::types::PyList::empty(py);
        for dt in &self.times {
            let py_dt = utc_to_python_datetime(py, dt)?;
            py_list.append(py_dt)?;
        }

        // Convert to numpy array with dtype=object
        let np_array = np.getattr("array")?.call1((py_list,))?;
        let py_obj: Py<PyAny> = np_array.into();

        // Cache the result (ignore if already set by another thread)
        let _ = self.timestamp_cache.set(py_obj.clone_ref(py));

        Ok(py_obj)
    }

    /// Check if the target is in-constraint at a given time.
    /// Accepts a Python datetime object (naive datetimes are treated as UTC).
    fn in_constraint(&self, _py: Python, time: &Bound<PyAny>) -> PyResult<bool> {
        let dt = python_datetime_to_utc(time)?;

        // O(1) index calculation instead of O(n) linear search
        if self.times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "no evaluated timestamps",
            ));
        }

        let begin = self.times[0];
        let offset_seconds = (dt - begin).num_seconds();

        // Check if time is before begin
        if offset_seconds < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ));
        }

        // Calculate index directly (O(1) instead of O(n))
        let idx = if self.step_seconds > 0 {
            (offset_seconds / self.step_seconds) as usize
        } else {
            0
        };

        // Verify index is in bounds and matches exactly
        if idx >= self.times.len() || self.times[idx] != dt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ));
        }

        // Check if this time falls within any violation window
        for v in &self.violations {
            if v.start_time_internal <= dt && dt <= v.end_time_internal {
                // Time is in a violation window, so in-constraint (violated)
                return Ok(true);
            }
        }
        // No violations found for this time, so not in-constraint
        Ok(false)
    }

    /// Property: array of visibility windows when target is not constrained
    #[getter]
    fn visibility(&self, py: Python) -> PyResult<Vec<VisibilityWindow>> {
        if self.times.is_empty() {
            return Ok(Vec::new());
        }

        let mut windows = Vec::new();
        let mut current_window_start: Option<usize> = None;

        // Get cached violation mask for each time (True == violated)
        let violated_vec = self._get_constraint_vec();

        for (i, &is_violated) in violated_vec.iter().enumerate() {
            let is_satisfied = !is_violated;
            if is_satisfied {
                // Constraint is satisfied (target is visible)
                if current_window_start.is_none() {
                    current_window_start = Some(i);
                }
            } else {
                // Constraint is violated (target not visible)
                if let Some(start_idx) = current_window_start {
                    // Only add window if it's non-zero length
                    if i - 1 != start_idx {
                        let start_cause = if start_idx == 0 {
                            None
                        } else {
                            flipped_causes(&self.component_violated, start_idx - 1, start_idx)
                        };
                        let end_cause = flipped_causes(&self.component_violated, i - 1, i);
                        windows.push(VisibilityWindow {
                            start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                            end_time: utc_to_python_datetime(py, &self.times[i - 1])?,
                            start_cause,
                            end_cause,
                        });
                    }
                    current_window_start = None;
                }
            }
        }

        // Close any open visibility window at the end
        if let Some(start_idx) = current_window_start {
            let start_cause = if start_idx == 0 {
                None
            } else {
                flipped_causes(&self.component_violated, start_idx - 1, start_idx)
            };
            windows.push(VisibilityWindow {
                start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                end_time: utc_to_python_datetime(py, &self.times[self.times.len() - 1])?,
                start_cause,
                end_cause: None,
            });
        }

        Ok(windows)
    }
}

/// Result of constraint evaluation for a moving body
///
/// Extends ConstraintResult with RA/Dec arrays for the moving body's position
/// at each evaluation time.
#[pyclass(name = "MovingBodyResult")]
pub struct MovingBodyResult {
    /// List of time windows where the constraint was violated
    #[pyo3(get)]
    pub violations: Vec<ConstraintViolation>,
    /// Whether the constraint was satisfied for the entire time range
    #[pyo3(get)]
    pub all_satisfied: bool,
    /// Constraint name/description
    #[pyo3(get)]
    pub constraint_name: String,
    /// Right ascensions in degrees for each timestamp
    #[pyo3(get)]
    pub ras: Vec<f64>,
    /// Declinations in degrees for each timestamp
    #[pyo3(get)]
    pub decs: Vec<f64>,
    /// Named continuous values computed during evaluation (e.g. `sun_angle_deg`),
    /// one array per key, aligned with `times`/`timestamp`.
    #[pyo3(get)]
    pub constraint_values: HashMap<String, Vec<f64>>,
    /// Per-leaf-constraint violation mask (namespaced the same way as
    /// `constraint_values`' keys), aligned with `times`. Used to attribute
    /// `VisibilityWindow.start_cause`/`end_cause`; not directly exposed to Python.
    component_violated: HashMap<String, Vec<bool>>,
    /// Map from a `start_cause`/`end_cause` tag to the `constraint_values` key(s)
    /// it corresponds to. See `ConstraintResult.cause_value_keys`.
    #[pyo3(get)]
    pub cause_value_keys: HashMap<String, Vec<String>>,
    /// Evaluation times as Rust DateTime<Utc>, not directly exposed to Python
    pub times: Vec<DateTime<Utc>>,
    /// Step size in seconds between timestamps (for O(1) index lookup)
    step_seconds: i64,
    /// Boolean array indicating constraint violation at each time (True = violated)
    constraint_vec: Vec<bool>,
}

impl MovingBodyResult {
    /// Create a new MovingBodyResult
    pub fn new(
        violations: Vec<ConstraintViolation>,
        all_satisfied: bool,
        constraint_name: String,
        times: Vec<DateTime<Utc>>,
        ras: Vec<f64>,
        decs: Vec<f64>,
        constraint_vec: Vec<bool>,
    ) -> Self {
        // Compute step size from first two timestamps (0 if fewer than 2 times)
        let step_seconds = if times.len() >= 2 {
            (times[1] - times[0]).num_seconds()
        } else {
            0
        };
        Self {
            violations,
            all_satisfied,
            constraint_name,
            ras,
            decs,
            constraint_values: HashMap::new(),
            component_violated: HashMap::new(),
            cause_value_keys: HashMap::new(),
            times,
            step_seconds,
            constraint_vec,
        }
    }

    /// Attach named continuous values computed during evaluation.
    pub fn with_constraint_values(mut self, constraint_values: HashMap<String, Vec<f64>>) -> Self {
        self.constraint_values = constraint_values;
        self
    }

    /// Attach per-leaf-constraint violation masks computed during evaluation.
    pub fn with_component_violated(
        mut self,
        component_violated: HashMap<String, Vec<bool>>,
    ) -> Self {
        self.component_violated = component_violated;
        self
    }

    /// Attach the cause-tag → constraint_values-key mapping computed during evaluation.
    pub fn with_cause_value_keys(mut self, cause_value_keys: HashMap<String, Vec<String>>) -> Self {
        self.cause_value_keys = cause_value_keys;
        self
    }
}

#[pymethods]
impl MovingBodyResult {
    fn __repr__(&self) -> String {
        format!(
            "MovingBodyResult(constraint='{}', violations={}, all_satisfied={}, n_times={})",
            self.constraint_name,
            self.violations.len(),
            self.all_satisfied,
            self.times.len()
        )
    }

    /// Get the total duration of violations in seconds
    fn total_violation_duration(&self) -> PyResult<f64> {
        let mut total_seconds = 0.0;
        for violation in &self.violations {
            let start = violation.start_time_internal;
            let end = violation.end_time_internal;
            total_seconds += (end - start).num_seconds() as f64;
        }
        Ok(total_seconds)
    }

    /// Property: array of booleans for each timestamp where True means constraint violated
    #[getter]
    fn constraint_array(&self, py: Python) -> PyResult<Py<PyAny>> {
        let py_list = pyo3::types::PyList::empty(py);
        for b in &self.constraint_vec {
            py_list.append(pyo3::types::PyBool::new(py, *b))?;
        }
        Ok(py_list.into())
    }

    /// Property: array of Python datetime objects for each evaluation time (as numpy array)
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Py<PyAny>> {
        let np = pyo3::types::PyModule::import(py, "numpy")
            .map_err(|_| pyo3::exceptions::PyImportError::new_err("numpy is required"))?;

        let py_list = pyo3::types::PyList::empty(py);
        for dt in &self.times {
            let py_dt = utc_to_python_datetime(py, dt)?;
            py_list.append(py_dt)?;
        }

        let np_array = np.getattr("array")?.call1((py_list,))?;
        Ok(np_array.into())
    }

    /// Check if the target is in-constraint at a given time.
    fn in_constraint(&self, _py: Python, time: &Bound<PyAny>) -> PyResult<bool> {
        let dt = python_datetime_to_utc(time)?;

        // O(1) index calculation instead of O(n) linear search
        if self.times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "no evaluated timestamps",
            ));
        }

        let begin = self.times[0];
        let offset_seconds = (dt - begin).num_seconds();

        // Check if time is before begin
        if offset_seconds < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ));
        }

        // Calculate index directly (O(1) instead of O(n))
        let idx = if self.step_seconds > 0 {
            (offset_seconds / self.step_seconds) as usize
        } else {
            0
        };

        // Verify index is in bounds and matches exactly
        if idx >= self.times.len() || self.times[idx] != dt {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ));
        }

        Ok(self.constraint_vec[idx])
    }

    /// Property: array of visibility windows when target is not constrained
    #[getter]
    fn visibility(&self, py: Python) -> PyResult<Vec<VisibilityWindow>> {
        if self.times.is_empty() {
            return Ok(Vec::new());
        }

        let mut windows = Vec::new();
        let mut current_window_start: Option<usize> = None;

        for (i, &is_violated) in self.constraint_vec.iter().enumerate() {
            let is_satisfied = !is_violated;
            if is_satisfied {
                if current_window_start.is_none() {
                    current_window_start = Some(i);
                }
            } else if let Some(start_idx) = current_window_start {
                if i - 1 != start_idx {
                    let start_cause = if start_idx == 0 {
                        None
                    } else {
                        flipped_causes(&self.component_violated, start_idx - 1, start_idx)
                    };
                    let end_cause = flipped_causes(&self.component_violated, i - 1, i);
                    windows.push(VisibilityWindow {
                        start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                        end_time: utc_to_python_datetime(py, &self.times[i - 1])?,
                        start_cause,
                        end_cause,
                    });
                }
                current_window_start = None;
            }
        }

        if let Some(start_idx) = current_window_start {
            let start_cause = if start_idx == 0 {
                None
            } else {
                flipped_causes(&self.component_violated, start_idx - 1, start_idx)
            };
            windows.push(VisibilityWindow {
                start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                end_time: utc_to_python_datetime(py, &self.times[self.times.len() - 1])?,
                start_cause,
                end_cause: None,
            });
        }

        Ok(windows)
    }
}

/// Configuration for constraint evaluation
///
/// This is the base trait that all constraint configurations must implement.
pub trait ConstraintConfig: fmt::Debug + Send + Sync {
    /// Create a constraint evaluator from this configuration
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator>;
}

/// Trait for evaluating constraints
///
/// Implementations of this trait perform the actual constraint checking logic.
pub trait ConstraintEvaluator: Send + Sync {
    /// Evaluate the constraint with full ephemeris access
    ///
    /// # Arguments
    /// * `ephemeris` - Ephemeris object providing all positional data
    /// * `target_ra` - Right ascension of target in degrees (ICRS/J2000)
    /// * `target_dec` - Declination of target in degrees (ICRS/J2000)
    /// * `time_indices` - Optional subset of time indices to evaluate
    ///
    /// # Returns
    /// Result containing violation windows
    #[allow(dead_code)]
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult>;

    /// Compute named continuous values (e.g. `sun_angle_deg`) for each target/time,
    /// mirroring the geometry already computed by `in_constraint_batch` before it gets
    /// thresholded into a boolean. Called once per `evaluate()`/`evaluate_batch()` call
    /// (never inside a roll-sweep loop), so this can afford to recompute the underlying
    /// scalar rather than sharing state with the boolean path.
    ///
    /// # Returns
    /// Map from value name to an (M x N) array (targets x times), matching the shape of
    /// `in_constraint_batch`. Default: no named values (constraint doesn't expose one).
    fn compute_named_values(
        &self,
        _ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        _target_ras: &[f64],
        _target_decs: &[f64],
        _time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        Ok(HashMap::new())
    }

    /// Diagonal variant of `compute_named_values` for moving-body evaluation: target_i
    /// paired with time_i only. Default falls back to the full batch and extracts the
    /// diagonal, mirroring `in_constraint_batch_diagonal`'s default.
    fn compute_named_values_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<f64>>> {
        let n = target_ras.len();
        let time_indices: Vec<usize> = (0..n).collect();
        let full =
            self.compute_named_values(ephemeris, target_ras, target_decs, Some(&time_indices))?;
        Ok(full
            .into_iter()
            .map(|(key, arr)| (key, (0..n).map(|i| arr[[i, i]]).collect()))
            .collect())
    }

    /// Compute a per-leaf-constraint violation mask (namespaced under a short tag, e.g.
    /// `sun`), used to attribute `VisibilityWindow.start_cause`/`end_cause` when this
    /// constraint is combined with others via `&`, `|`, `~`, `^`, `.at_least()`. Called
    /// once per `evaluate()`/`evaluate_batch()` call, never inside a roll-sweep loop.
    ///
    /// Default (leaf constraints): a single-entry map from this constraint's own tag to
    /// its own `in_constraint_batch` result. Combinators override this to merge their
    /// children's maps instead (see `combinators.rs`).
    ///
    /// # Returns
    /// Map from tag to an (M x N) boolean array (targets x times), matching the shape of
    /// `in_constraint_batch`.
    fn compute_named_booleans(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        let arr = self.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices)?;
        Ok(HashMap::from([(value_key_prefix(&self.name()), arr)]))
    }

    /// Per-leaf violation masks at one fixed candidate spacecraft roll, with exactly
    /// the same tag identity and collision renaming as `compute_named_booleans`.
    ///
    /// This is the per-step primitive of the coordinated free-roll cause attribution
    /// in `sweep_rolls_with_attribution`: that sweep needs, at each roll angle, both
    /// the combined outcome (`in_constraint_batch_at_roll`) and every leaf's own
    /// state *at that same angle*, so it can record which leaves were violated at
    /// the roll it eventually selects as the witness. Tags must stay in lockstep
    /// with `compute_named_booleans` / `compute_cause_value_keys` — a caller maps a
    /// cause tag to its `constraint_values` key(s) through the latter.
    ///
    /// Default (leaf constraints): a single-entry map from this constraint's own tag
    /// to its own `in_constraint_batch_at_roll` result, mirroring
    /// `compute_named_booleans`' default.
    fn compute_named_booleans_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> PyResult<HashMap<String, Array2<bool>>> {
        let arr = self.in_constraint_batch_at_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            roll_deg,
        )?;
        Ok(HashMap::from([(value_key_prefix(&self.name()), arr)]))
    }

    /// Diagonal variant of `compute_named_booleans` for moving-body evaluation: target_i
    /// paired with time_i only. Default (leaf constraints): a single-entry map from this
    /// constraint's own tag to its own `in_constraint_batch_diagonal` result, mirroring
    /// `compute_named_booleans`'s default — this reuses whatever O(N) diagonal
    /// implementation the leaf already has (e.g. SAA) instead of building the full M×N
    /// matrix and slicing it, which would silently regress to O(N²). Combinators override
    /// this to merge their children's diagonal maps instead (see `combinators.rs`).
    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        let arr = self.in_constraint_batch_diagonal(ephemeris, target_ras, target_decs)?;
        Ok(HashMap::from([(value_key_prefix(&self.name()), arr)]))
    }

    /// Map this constraint's own `compute_named_booleans` cause tag(s) to the
    /// `compute_named_values` key(s) they correspond to, so a caller can look up
    /// "which `constraint_values` key(s) does this cause tag describe" without
    /// guessing from string prefixes — the two namespaces use different prefixing
    /// conventions (cause tags are flat and stable regardless of nesting depth;
    /// value keys are hierarchical and encode the full nesting path) and are not
    /// generally derivable from one another by string matching. Called once per
    /// `evaluate()`/`evaluate_batch()` call, alongside `compute_named_values`/
    /// `compute_named_booleans`, never inside a roll-sweep loop.
    ///
    /// Default (leaf constraints): a single-entry map from this constraint's own
    /// cause tag to the key name(s) `compute_named_values` returns (empty if this
    /// constraint exposes no named values, e.g. SAA). Combinators override this to
    /// merge their children's maps in lockstep with `compute_named_booleans`' cause-
    /// tag renaming and `compute_named_values`' per-child key prefixing (see
    /// `combinators.rs`).
    ///
    /// # Returns
    /// Map from cause tag to the list of `constraint_values` key(s) it corresponds to.
    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        let own_tag = value_key_prefix(&self.name());
        let value_keys: Vec<String> = self
            .compute_named_values(ephemeris, target_ras, target_decs, time_indices)?
            .into_keys()
            .collect();
        Ok(HashMap::from([(own_tag, value_keys)]))
    }

    /// Check if targets are in-constraint for multiple RA/Dec positions (vectorized)
    ///
    /// # Arguments
    /// * `ephemeris` - Ephemeris object providing all positional data
    /// * `target_ras` - Array of right ascensions in degrees (length M)
    /// * `target_decs` - Array of declinations in degrees (length M)
    /// * `time_indices` - Optional subset of time indices to evaluate
    ///
    /// # Returns
    /// 2D boolean array (M x N) where True indicates constraint violation
    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>>;

    /// Optional vectorized batch evaluation using precomputed unit vectors.
    ///
    /// # Arguments
    /// * `ephemeris` - Ephemeris object providing all positional data
    /// * `target_unit_vectors` - Array of shape (M, 3) containing target unit vectors
    /// * `time_indices` - Optional subset of time indices to evaluate
    ///
    /// # Returns
    /// `Ok(Some(result))` when a specialized implementation is available,
    /// otherwise `Ok(None)` to request fallback to RA/Dec batch evaluation.
    fn in_constraint_batch_unit_vectors(
        &self,
        _ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        _target_unit_vectors: &Array2<f64>,
        _time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        Ok(None)
    }

    /// Evaluate constraint for moving body (diagonal evaluation)
    ///
    /// For moving bodies, we need to evaluate target_i at time_i only (diagonal).
    /// This is much more efficient than computing the full M×N matrix and extracting diagonal.
    ///
    /// # Arguments
    /// * `ephemeris` - Ephemeris object providing all positional data
    /// * `target_ras` - Array of right ascensions in degrees (length N)
    /// * `target_decs` - Array of declinations in degrees (length N)
    ///
    /// # Returns
    /// 1D boolean array (N) where result[i] = in_constraint(target_i, time_i)
    ///
    /// Default implementation falls back to N×N batch with diagonal extraction.
    /// Implementations can override for O(N) performance.
    fn in_constraint_batch_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        // Default: compute full N×N and extract diagonal
        let n = target_ras.len();
        let time_indices: Vec<usize> = (0..n).collect();
        let full =
            self.in_constraint_batch(ephemeris, target_ras, target_decs, Some(&time_indices))?;
        Ok((0..n).map(|i| full[[i, i]]).collect())
    }

    /// Returns `true` if this constraint's FoR result changes depending on the spacecraft
    /// roll angle.  Roll-independent constraints return `false` (the default); only
    /// `BoresightOffsetEvaluator` with a free roll angle (`roll_deg = None`) and a
    /// non-zero pitch/yaw offset returns `true`.
    ///
    /// Combinators delegate to their children: they are roll-dependent if any child is.
    fn is_roll_dependent(&self) -> bool {
        false
    }

    /// Evaluate violated-per-direction for a *single* roll angle, at one timestamp.
    ///
    /// Returns one bool per target direction: `true` = violated at this roll,
    /// `false` = accessible at this roll.
    ///
    /// `roll_deg` is the candidate roll angle (degrees).  Roll-independent constraints
    /// ignore it; `BoresightOffsetEvaluator` injects it when its own roll is free (`None`).
    ///
    /// Default: delegates to `in_constraint_batch_unit_vectors` / `in_constraint_batch`
    /// ignoring `roll_deg`, correct for constraints that do not depend on roll.
    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        _roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();

        if let Some(result) = self.in_constraint_batch_unit_vectors(
            ephemeris,
            target_unit_vectors,
            Some(&[time_index]),
        )? {
            return Ok((0..n_targets).map(|i| result[[i, 0]]).collect());
        }

        // Fallback: convert unit vectors to RA/Dec inline and use the scalar batch path.
        let mut ras = Vec::with_capacity(n_targets);
        let mut decs = Vec::with_capacity(n_targets);
        for i in 0..n_targets {
            let x = target_unit_vectors[[i, 0]];
            let y = target_unit_vectors[[i, 1]];
            let z = target_unit_vectors[[i, 2]];
            let ra = y.atan2(x).to_degrees().rem_euclid(360.0);
            let dec = z.clamp(-1.0, 1.0).asin().to_degrees();
            ras.push(ra);
            decs.push(dec);
        }

        let result = self.in_constraint_batch(ephemeris, &ras, &decs, Some(&[time_index]))?;
        Ok((0..n_targets).map(|i| result[[i, 0]]).collect())
    }

    /// For field-of-regard evaluation: determine which sky directions are inaccessible,
    /// sweeping over roll angles where applicable.
    ///
    /// Returns one bool per target direction:
    ///   - `true`  → violated / inaccessible (blocked for every applicable roll)
    ///   - `false` → accessible for at least one valid roll
    ///
    /// `time_index` is the single ephemeris timestamp to evaluate at.
    /// `n_roll_samples` uniform roll angles are swept over [0°, 360°) for evaluators
    /// that depend on spacecraft roll; constraints without roll dependence ignore it.
    ///
    /// Default: sweeps `n_roll_samples` roll angles and calls `field_of_regard_violated_at_roll`
    /// at each step, marking a direction accessible if *any* roll satisfies it.
    /// Leaf implementations that carry their own roll state (e.g. `BoresightOffsetEvaluator`
    /// with fixed roll) override `field_of_regard_violated_at_roll` instead, and the sweep
    /// here automatically propagates the correct roll to them.
    fn field_of_regard_violated_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        n_roll_samples: usize,
    ) -> PyResult<Vec<bool>> {
        // For roll-independent constraints the result is identical at every roll
        // angle, so a single evaluation at roll=0° is sufficient.
        if !self.is_roll_dependent() {
            return self.field_of_regard_violated_at_roll(
                ephemeris,
                target_unit_vectors,
                time_index,
                0.0,
            );
        }

        // Roll-dependent: sweep n_roll_samples angles over [0°, 360°).
        let n_targets = target_unit_vectors.nrows();
        let roll_step_deg = 360.0 / n_roll_samples as f64;
        let mut accessible = vec![false; n_targets];

        for step in 0..n_roll_samples {
            if accessible.iter().all(|&a| a) {
                break;
            }
            let roll_deg = step as f64 * roll_step_deg;
            let violated = self.field_of_regard_violated_at_roll(
                ephemeris,
                target_unit_vectors,
                time_index,
                roll_deg,
            )?;
            for i in 0..n_targets {
                if !violated[i] {
                    accessible[i] = true;
                }
            }
        }

        Ok(accessible.iter().map(|&a| !a).collect())
    }

    /// Evaluate constraint for the full target set at a single fixed roll angle.
    ///
    /// This is the hot path for the coordinated roll sweep in `target_rolls=None` mode.
    /// Roll-independent constraints ignore `roll_deg` and delegate to `in_constraint_batch`.
    /// Roll-dependent leaves (SolarRoll, BodyProximity polygon, BrightStar polygon) override
    /// this to use `roll_deg` directly, eliminating JSON round-trips across 72 sweep steps.
    ///
    /// Returns (M × N) boolean violation array, same semantics as `in_constraint_batch`.
    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        _roll_deg: f64,
    ) -> PyResult<Array2<bool>> {
        self.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices)
    }

    /// Sweep `n_roll_samples` roll angles and return targets that are violated at *every* roll
    /// (i.e., no clear roll exists).  Used by `in_constraint_batch(target_rolls=None)`.
    ///
    /// Default: roll-independent evaluators bypass the sweep with a single
    /// `in_constraint_batch` call; roll-dependent ones loop calling `in_constraint_batch_at_roll`
    /// and AND-accumulate.  Combinators override this to hoist roll-independent children out
    /// of the loop, and BrightStar overrides it to reuse its cached gnomonic projections.
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
        let roll_step = 360.0 / n_roll_samples as f64;
        let mut acc: Option<Array2<bool>> = None;
        for step in 0..n_roll_samples {
            if let Some(ref a) = acc {
                if a.iter().all(|&b| !b) {
                    break;
                }
            }
            let roll_deg = step as f64 * roll_step;
            let step_result = self.in_constraint_batch_at_roll(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
                roll_deg,
            )?;
            match acc {
                None => acc = Some(step_result),
                Some(ref mut a) => a.zip_mut_with(&step_result, |x, &y| *x &= y),
            }
        }
        Ok(acc.unwrap_or_else(|| Array2::from_elem((target_ras.len(), 0), false)))
    }

    /// Get constraint name
    fn name(&self) -> String;

    /// Downcast support for special handling
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Combined result and per-leaf cause attribution from one free-roll sweep.
pub struct RollSweepAttribution {
    /// (M x N) targets x times, `true` where the constraint is violated at *every*
    /// swept roll — identical semantics to
    /// `ConstraintEvaluator::in_constraint_batch_constrained_at_every_roll`.
    pub violated: Array2<bool>,
    /// Per-leaf violation masks taken at each cell's witness roll (see
    /// `sweep_rolls_with_attribution`), keyed by the same cause tags as
    /// `ConstraintEvaluator::compute_named_booleans`.
    pub named: HashMap<String, Array2<bool>>,
}

/// Sweep `n_roll_samples` spacecraft rolls, returning both the combined free-roll
/// violation mask and per-leaf cause masks that are consistent with it.
///
/// Cause attribution has to answer "which leaf changed?" about a result that was
/// itself produced by a search over rolls, and the naive decomposition — ask each
/// leaf separately whether *it alone* is violated at every roll — is not equivalent
/// for `OR`, `XOR` or `AT_LEAST`. Two leaves can block complementary roll ranges so
/// that together they leave no clear roll while neither blocks every roll on its
/// own; the combined window then opens or closes with both per-leaf answers
/// unchanged, and the boundary gets no cause at all.
///
/// So instead of decomposing, this picks a **witness roll** per (target, time) cell
/// and reports every leaf's state at that one shared, physically realisable
/// orientation:
///
/// * the first swept roll at which the whole tree is satisfied, if any exists — the
///   leaf states then describe an orientation the spacecraft could actually fly; or
/// * failing that (the cell is violated at every roll), the roll that leaves the
///   fewest leaf tags violated, i.e. the closest the tree came to opening up. Ties
///   go to the lowest roll angle, so the choice is deterministic.
///
/// Both outputs come from the same pass: `violated` is the AND of the per-roll
/// combined masks, exactly as `in_constraint_batch_constrained_at_every_roll`
/// computes it, so callers needing both do not sweep twice.
///
/// For a roll-independent tree this degenerates to a single non-swept evaluation.
pub fn sweep_rolls_with_attribution(
    evaluator: &dyn ConstraintEvaluator,
    ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    target_ras: &[f64],
    target_decs: &[f64],
    time_indices: Option<&[usize]>,
    n_roll_samples: usize,
) -> PyResult<RollSweepAttribution> {
    if !evaluator.is_roll_dependent() {
        return Ok(RollSweepAttribution {
            violated: evaluator.in_constraint_batch(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
            )?,
            named: evaluator.compute_named_booleans(
                ephemeris,
                target_ras,
                target_decs,
                time_indices,
            )?,
        });
    }

    let roll_step = 360.0 / n_roll_samples as f64;
    let mut violated: Option<Array2<bool>> = None;
    let mut named: HashMap<String, Array2<bool>> = HashMap::new();
    // Per cell: how many leaf tags were violated at the best roll seen so far, and
    // whether that best roll actually cleared the whole tree (in which case no later
    // roll can improve on it and the cell is settled).
    let mut best_violated_tags: Array2<usize> = Array2::from_elem((0, 0), 0);
    let mut settled: Array2<bool> = Array2::from_elem((0, 0), false);

    for step in 0..n_roll_samples {
        if settled.iter().all(|&s| s) && step > 0 {
            break;
        }

        let roll_deg = step as f64 * roll_step;
        let combined_step = evaluator.in_constraint_batch_at_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            roll_deg,
        )?;
        let named_step = evaluator.compute_named_booleans_at_roll(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            roll_deg,
        )?;

        let (n_targets, n_times) = combined_step.dim();
        if step == 0 {
            best_violated_tags = Array2::from_elem((n_targets, n_times), usize::MAX);
            settled = Array2::from_elem((n_targets, n_times), false);
            named = named_step
                .keys()
                .map(|key| (key.clone(), Array2::from_elem((n_targets, n_times), false)))
                .collect();
        }

        match violated {
            None => violated = Some(combined_step.clone()),
            Some(ref mut acc) => acc.zip_mut_with(&combined_step, |x, &y| *x &= y),
        }

        for row in 0..n_targets {
            for col in 0..n_times {
                if settled[[row, col]] {
                    continue;
                }
                let satisfied = !combined_step[[row, col]];
                let violated_tags = named_step.values().filter(|arr| arr[[row, col]]).count();
                if !satisfied && violated_tags >= best_violated_tags[[row, col]] {
                    continue;
                }
                for (key, arr) in named.iter_mut() {
                    // `named_step` is produced by the same evaluator tree as the map
                    // `named` was seeded from, so its key set is identical every step;
                    // the lookup is defensive rather than load-bearing.
                    if let Some(step_arr) = named_step.get(key) {
                        arr[[row, col]] = step_arr[[row, col]];
                    }
                }
                best_violated_tags[[row, col]] = violated_tags;
                settled[[row, col]] = satisfied;
            }
        }
    }

    Ok(RollSweepAttribution {
        violated: violated.unwrap_or_else(|| Array2::from_elem((target_ras.len(), 0), false)),
        named,
    })
}

/// Macro to generate common methods for proximity evaluators
/// This is exported so constraint modules can use it
macro_rules! impl_proximity_evaluator {
    ($evaluator:ty, $body_name:expr, $friendly_name:expr, $positions:ident) => {
        impl $evaluator {
            #[allow(dead_code)]
            fn evaluate_common(
                &self,
                times: &[DateTime<Utc>],
                target_ra_dec: (f64, f64),
                $positions: &Array2<f64>,
                observer_positions: &Array2<f64>,
                final_desc_fn: impl Fn() -> String,
                intermediate_desc_fn: impl Fn() -> String,
            ) -> ConstraintResult {
                // Cache target vector computation outside the loop
                let target_vec = crate::utils::vector_math::radec_to_unit_vector(
                    target_ra_dec.0,
                    target_ra_dec.1,
                );

                // Pre-compute cosine thresholds (avoids acos() in inner loop)
                // For angle comparison: angle < threshold ⟺ cos(angle) > cos(threshold)
                let min_cos_threshold = self.min_angle_deg.to_radians().cos();
                let max_cos_threshold = self.max_angle_deg.map(|max| max.to_radians().cos());

                let violations = track_violations(
                    times,
                    |i| {
                        let body_pos = [$positions[[i, 0]], $positions[[i, 1]], $positions[[i, 2]]];
                        let obs_pos = [
                            observer_positions[[i, 0]],
                            observer_positions[[i, 1]],
                            observer_positions[[i, 2]],
                        ];

                        // Calculate cosine of angle (avoids acos call)
                        let cos_angle = crate::utils::vector_math::calculate_cosine_separation(
                            &target_vec,
                            &body_pos,
                            &obs_pos,
                        );

                        // Check constraints using cosine comparison
                        // too_close: angle < min_angle ⟺ cos(angle) > cos(min_angle)
                        let too_close = cos_angle > min_cos_threshold;
                        let too_far =
                            max_cos_threshold.is_some_and(|max_thresh| cos_angle < max_thresh);
                        let is_violated = too_close || too_far;

                        // Compute severity using the angle (required for violation windows)
                        // Only compute acos when there's actually a violation to report
                        let severity = if is_violated {
                            let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
                            if angle_deg < self.min_angle_deg {
                                (self.min_angle_deg - angle_deg) / self.min_angle_deg
                            } else if let Some(max) = self.max_angle_deg {
                                (angle_deg - max) / max
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };

                        (is_violated, severity)
                    },
                    |_, is_final| {
                        if is_final {
                            final_desc_fn()
                        } else {
                            intermediate_desc_fn()
                        }
                    },
                );

                let all_satisfied = violations.is_empty();
                ConstraintResult::new(violations, all_satisfied, self.name(), times.to_vec())
            }
        }
    };
}

/// Compute the angular separation (degrees) between each target direction and a
/// celestial body's apparent position (as seen from the observer), for every
/// target/time combination. Mirrors the cosine-based geometry already computed
/// inline by each proximity constraint's `in_constraint_batch`, minus the
/// thresholding — used to populate `compute_named_values` (e.g. `sun_angle_deg`,
/// `moon_angle_deg`, `body_angle_deg`) without duplicating that math.
///
/// # Returns
/// (M x N) array (targets x times) of angles in degrees.
pub(crate) fn compute_angle_deg_batch(
    target_ras: &[f64],
    target_decs: &[f64],
    body_positions: &Array2<f64>,
    observer_positions: &Array2<f64>,
) -> Array2<f64> {
    let n_targets = target_ras.len();
    let n_times = body_positions.nrows();
    let target_vectors =
        crate::utils::vector_math::radec_to_unit_vectors_batch(target_ras, target_decs);

    let mut result = Array2::<f64>::zeros((n_targets, n_times));

    for t in 0..n_times {
        let body_pos = [
            body_positions[[t, 0]],
            body_positions[[t, 1]],
            body_positions[[t, 2]],
        ];
        let obs_pos = [
            observer_positions[[t, 0]],
            observer_positions[[t, 1]],
            observer_positions[[t, 2]],
        ];

        let body_rel = [
            body_pos[0] - obs_pos[0],
            body_pos[1] - obs_pos[1],
            body_pos[2] - obs_pos[2],
        ];
        let body_dist =
            (body_rel[0] * body_rel[0] + body_rel[1] * body_rel[1] + body_rel[2] * body_rel[2])
                .sqrt();
        let body_unit = if body_dist > 0.0 {
            [
                body_rel[0] / body_dist,
                body_rel[1] / body_dist,
                body_rel[2] / body_dist,
            ]
        } else {
            [1.0, 0.0, 0.0]
        };

        for target_idx in 0..n_targets {
            let target_vec = [
                target_vectors[[target_idx, 0]],
                target_vectors[[target_idx, 1]],
                target_vectors[[target_idx, 2]],
            ];

            let cos_angle = target_vec[0] * body_unit[0]
                + target_vec[1] * body_unit[1]
                + target_vec[2] * body_unit[2];

            result[[target_idx, t]] = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
        }
    }

    result
}

// Helper function for tracking violation windows
pub(crate) fn track_violations<F>(
    times: &[DateTime<Utc>],
    mut is_violated: F,
    mut get_description: impl FnMut(usize, bool) -> String,
) -> Vec<ConstraintViolation>
where
    F: FnMut(usize) -> (bool, f64),
{
    // Pre-allocate with reasonable capacity estimate
    let mut violations = Vec::with_capacity(4);
    let mut current_violation: Option<(usize, f64)> = None;

    for i in 0..times.len() {
        let (violated, severity) = is_violated(i);

        if violated {
            match current_violation {
                Some((start_idx, max_sev)) => {
                    current_violation = Some((start_idx, max_sev.max(severity)));
                }
                None => {
                    current_violation = Some((i, severity));
                }
            }
        } else if let Some((start_idx, max_severity)) = current_violation {
            violations.push(ConstraintViolation {
                start_time_internal: times[start_idx],
                end_time_internal: times[i - 1],
                max_severity,
                description: get_description(start_idx, false),
            });
            current_violation = None;
        }
    }

    // Close any open violation at the end
    if let Some((start_idx, max_severity)) = current_violation {
        violations.push(ConstraintViolation {
            start_time_internal: times[start_idx],
            end_time_internal: times[times.len() - 1],
            max_severity,
            description: get_description(start_idx, true),
        });
    }

    violations
}

/// Macro to extract and filter ephemeris data with celestial body positions
/// Usage: extract_body_ephemeris_data!(ephemeris, time_indices, get_body_positions)
/// Returns: (times_filtered, body_positions_filtered, observer_positions_filtered)
macro_rules! extract_body_ephemeris_data {
    ($ephemeris:expr, $time_indices:expr, $body_getter:ident) => {{
        let times = $ephemeris.get_times()?;
        let body_positions = $ephemeris.$body_getter()?;
        let observer_positions = $ephemeris.get_gcrs_positions()?;

        if let Some(indices) = $time_indices {
            let filtered_times: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
            let body_filtered = body_positions.select(ndarray::Axis(0), indices);
            let obs_filtered = observer_positions.select(ndarray::Axis(0), indices);
            (filtered_times, body_filtered, obs_filtered)
        } else {
            // body_positions and observer_positions are already owned (from .to_owned() in getters)
            // so no need to clone again
            (times.to_vec(), body_positions, observer_positions)
        }
    }};
}

/// Macro to extract and filter common ephemeris data (times, sun_positions, observer_positions)
/// Usage: extract_standard_ephemeris_data!(ephemeris, time_indices)
/// Returns: (times_filtered, sun_positions_filtered, observer_positions_filtered)
macro_rules! extract_standard_ephemeris_data {
    ($ephemeris:expr, $time_indices:expr) => {{
        extract_body_ephemeris_data!($ephemeris, $time_indices, get_sun_positions)
    }};
}

/// Returns: (times_filtered, observer_positions_filtered)
macro_rules! extract_observer_ephemeris_data {
    ($ephemeris:expr, $time_indices:expr) => {{
        let times = $ephemeris.get_times()?;
        let observer_positions = $ephemeris.get_gcrs_positions()?;

        if let Some(indices) = $time_indices {
            let filtered_times: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
            let obs_filtered = observer_positions.select(ndarray::Axis(0), indices);
            (filtered_times, obs_filtered)
        } else {
            // observer_positions is already owned (from .to_owned() in getter)
            (times.to_vec(), observer_positions)
        }
    }};
}

/// Macro to extract and filter time data
/// Usage: extract_time_data!(ephemeris, time_indices)
/// Returns: (times_filtered,)
macro_rules! extract_time_data {
    ($ephemeris:expr, $time_indices:expr) => {{
        let times = $ephemeris.get_times()?;

        let times_filtered = if let Some(indices) = $time_indices {
            indices.iter().map(|&i| times[i]).collect()
        } else {
            times.to_vec()
        };

        (times_filtered,)
    }};
}

/// Returns: (times_filtered, lats_filtered, lons_filtered)
/// Usage: extract_latlon_data!(ephemeris, time_indices)
macro_rules! extract_latlon_data {
    ($ephemeris:expr, $time_indices:expr) => {{
        let (lats_vec, lons_vec) = {
            use numpy::{PyArray1, PyArrayMethods};
            use pyo3::Python;

            Python::attach(|py| -> pyo3::PyResult<(Vec<f64>, Vec<f64>)> {
                let lat_opt = $ephemeris.get_latitude_deg(py)?;
                let lon_opt = $ephemeris.get_longitude_deg(py)?;

                let lat_array = lat_opt.ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Latitude data not available")
                })?;
                let lon_array = lon_opt.ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Longitude data not available")
                })?;

                let lat_bound = lat_array.cast_bound::<PyArray1<f64>>(py)?;
                let lon_bound = lon_array.cast_bound::<PyArray1<f64>>(py)?;

                let lats = lat_bound.readonly().as_slice()?.to_vec();
                let lons = lon_bound.readonly().as_slice()?.to_vec();

                Ok((lats, lons))
            })?
        };
        let times = $ephemeris.get_times()?;

        let (times_slice, lats_slice, lons_slice) = if let Some(indices) = $time_indices {
            let filtered_times: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
            let filtered_lats: Vec<f64> = indices.iter().map(|&i| lats_vec[i]).collect();
            let filtered_lons: Vec<f64> = indices.iter().map(|&i| lons_vec[i]).collect();
            (filtered_times, filtered_lats, filtered_lons)
        } else {
            (times.to_vec(), lats_vec, lons_vec)
        };

        (times_slice, lats_slice, lons_slice)
    }};
}

#[cfg(test)]
mod tests {
    use super::{flipped_causes, value_key_prefix};
    use std::collections::HashMap;

    #[test]
    fn test_flipped_causes_detects_single_flip() {
        let mut component_violated = HashMap::new();
        component_violated.insert("sun".to_string(), vec![true, true, false]);
        component_violated.insert("moon".to_string(), vec![false, false, false]);

        assert_eq!(
            flipped_causes(&component_violated, 1, 2),
            Some(vec!["sun".to_string()])
        );
    }

    #[test]
    fn test_flipped_causes_lists_multiple_flips_sorted() {
        let mut component_violated = HashMap::new();
        component_violated.insert("sun".to_string(), vec![true, false]);
        component_violated.insert("moon".to_string(), vec![false, true]);

        assert_eq!(
            flipped_causes(&component_violated, 0, 1),
            Some(vec!["moon".to_string(), "sun".to_string()])
        );
    }

    #[test]
    fn test_flipped_causes_none_when_nothing_changed() {
        let mut component_violated = HashMap::new();
        component_violated.insert("sun".to_string(), vec![true, true]);
        component_violated.insert("moon".to_string(), vec![false, false]);

        assert_eq!(flipped_causes(&component_violated, 0, 1), None);
    }

    #[test]
    fn test_value_key_prefix_known_and_unknown_tags() {
        assert_eq!(value_key_prefix("SunProximity(min=45°)"), "sun");
        assert_eq!(value_key_prefix("AND(SunProximity(min=45°))"), "and");
        assert_eq!(
            value_key_prefix("SomeNewConstraint(foo=1)"),
            "somenewconstraint"
        );
    }
}
