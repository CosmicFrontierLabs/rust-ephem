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
use std::fmt;
use std::sync::OnceLock;

/// Result of constraint evaluation
///
/// Contains information about when and where a constraint is violated.
#[pyclass(name = "ConstraintViolation")]
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
}

#[pymethods]
impl VisibilityWindow {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let start_str = self.start_time.bind(py).str()?.to_string();
        let end_str = self.end_time.bind(py).str()?.to_string();
        let duration = self.duration_seconds(py)?;
        Ok(format!(
            "VisibilityWindow(start_time={}, end_time={}, duration_seconds={})",
            start_str, end_str, duration
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
            times,
            step_seconds,
            timestamp_cache: OnceLock::new(),
            constraint_vec_cache: OnceLock::new(),
            constraint_array_cache: OnceLock::new(),
        }
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
                        windows.push(VisibilityWindow {
                            start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                            end_time: utc_to_python_datetime(py, &self.times[i - 1])?,
                        });
                    }
                    current_window_start = None;
                }
            }
        }

        // Close any open visibility window at the end
        if let Some(start_idx) = current_window_start {
            windows.push(VisibilityWindow {
                start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                end_time: utc_to_python_datetime(py, &self.times[self.times.len() - 1])?,
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
            times,
            step_seconds,
            constraint_vec,
        }
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
                    windows.push(VisibilityWindow {
                        start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                        end_time: utc_to_python_datetime(py, &self.times[i - 1])?,
                    });
                }
                current_window_start = None;
            }
        }

        if let Some(start_idx) = current_window_start {
            windows.push(VisibilityWindow {
                start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                end_time: utc_to_python_datetime(py, &self.times[self.times.len() - 1])?,
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

    /// Get constraint name
    fn name(&self) -> String;

    /// Downcast support for special handling
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn std::any::Any;
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

                let lat_bound = lat_array.downcast_bound::<PyArray1<f64>>(py)?;
                let lon_bound = lon_array.downcast_bound::<PyArray1<f64>>(py)?;

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
