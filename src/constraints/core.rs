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
    /// Start time of the violation window
    #[pyo3(get)]
    pub start_time: String, // ISO 8601 format
    /// End time of the violation window
    #[pyo3(get)]
    pub end_time: String, // ISO 8601 format
    /// Maximum severity of violation in this window (0.0 = just violated, 1.0+ = severe)
    #[pyo3(get)]
    pub max_severity: f64,
    /// Human-readable description of the violation
    #[pyo3(get)]
    pub description: String,
}

#[pymethods]
impl ConstraintViolation {
    fn __repr__(&self) -> String {
        format!(
            "ConstraintViolation(start='{}', end='{}', max_severity={:.3}, description='{}')",
            self.start_time, self.end_time, self.max_severity, self.description
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
    /// Cached Python timestamp array (not directly exposed, use getter)
    timestamp_cache: OnceLock<Py<PyAny>>,
    /// Cached constraint array (not directly exposed, use getter)
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
        Self {
            violations,
            all_satisfied,
            constraint_name,
            times,
            timestamp_cache: OnceLock::new(),
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
            let start = DateTime::parse_from_rfc3339(&violation.start_time)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid start time: {e}"))
                })?
                .with_timezone(&Utc);
            let end = DateTime::parse_from_rfc3339(&violation.end_time)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid end time: {e}"))
                })?
                .with_timezone(&Utc);
            total_seconds += (end - start).num_seconds() as f64;
        }
        Ok(total_seconds)
    }

    /// Internal: compute boolean array indicating if constraint is violated at each time
    ///
    /// NOTE: This returns a *violation mask* where True means the constraint
    /// is violated (target NOT visible) at that timestamp. The public
    /// `constraint_array` property therefore exposes violation semantics
    /// (True == violated) to Python; visibility windows are computed by
    /// inverting this mask.
    fn _compute_constraint_vec(&self) -> Vec<bool> {
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
            let t_str = t.to_rfc3339();
            // Binary search could be used here, but violation count is typically small
            // and the string comparison overhead dominates
            for v in &self.violations {
                if t_str < v.start_time {
                    break; // Violations are sorted, no need to check further
                }
                if v.start_time <= t_str && t_str <= v.end_time {
                    violated[i] = true;
                    break;
                }
            }
        }
        violated
    }

    /// Property: array of booleans for each timestamp where True means constraint violated
    #[getter]
    fn constraint_array(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Use cached value if available
        if let Some(cached) = self.constraint_array_cache.get() {
            return Ok(cached.clone_ref(py));
        }

        // Compute and cache
        // Return a Python list of bools (True == violated) so indexing yields
        // native Python bool values. Tests historically expect identity
        // comparisons ("is True"), so returning Python bools is safer.
        let arr = self._compute_constraint_vec();
        let py_list = pyo3::types::PyList::empty(py);
        for b in arr {
            py_list.append(pyo3::types::PyBool::new(py, b))?;
        }
        let py_obj: Py<PyAny> = py_list.into();

        // Cache the result (ignore if already set by another thread)
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

        // Find matching time in our array
        if let Some(_idx) = self.times.iter().position(|t| t == &dt) {
            // Check if this time falls within any violation window
            let t_str = dt.to_rfc3339();
            for v in &self.violations {
                if v.start_time <= t_str && t_str <= v.end_time {
                    // Time is in a violation window, so in-constraint (violated)
                    return Ok(true);
                }
            }
            // No violations found for this time, so not in-constraint
            Ok(false)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ))
        }
    }

    /// Property: array of visibility windows when target is not constrained
    #[getter]
    fn visibility(&self, py: Python) -> PyResult<Vec<VisibilityWindow>> {
        if self.times.is_empty() {
            return Ok(Vec::new());
        }

        let mut windows = Vec::new();
        let mut current_window_start: Option<usize> = None;

        // Get violation mask for each time (True == violated)
        let violated_vec = self._compute_constraint_vec();

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
                    windows.push(VisibilityWindow {
                        start_time: utc_to_python_datetime(py, &self.times[start_idx])?,
                        end_time: utc_to_python_datetime(py, &self.times[i - 1])?,
                    });
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
    /// Evaluate the constraint over a time range
    ///
    /// # Arguments
    /// * `times` - Vector of timestamps to evaluate
    /// * `target_ra` - Right ascension of target in degrees (ICRS/J2000)
    /// * `target_dec` - Declination of target in degrees (ICRS/J2000)
    /// * `sun_positions` - Sun positions in GCRS (N x 3 array, km)
    /// * `moon_positions` - Moon positions in GCRS (N x 3 array, km)
    /// * `observer_positions` - Observer positions in GCRS (N x 3 array, km)
    ///
    /// # Returns
    /// Result containing violation windows
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult;

    /// Get constraint name
    fn name(&self) -> String;

    /// Downcast support for special handling
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Macro to generate common methods for proximity evaluators
/// This is exported so constraint modules can use it
#[macro_export]
macro_rules! impl_proximity_evaluator_helpers {
    ($evaluator:ty, $body_name:expr, $friendly_name:expr) => {
        impl $evaluator {
            fn final_violation_description(&self) -> String {
                match self.max_angle_deg {
                    Some(max) => format!(
                        "Target too close to {} (min: {:.1}°) or too far (max: {:.1}°)",
                        $friendly_name, self.min_angle_deg, max
                    ),
                    None => format!(
                        "Target too close to {} (min allowed: {:.1}°)",
                        $friendly_name, self.min_angle_deg
                    ),
                }
            }

            fn format_name(&self) -> String {
                match self.max_angle_deg {
                    Some(max) => format!(
                        "{}Proximity(min={}°, max={}°)",
                        $body_name, self.min_angle_deg, max
                    ),
                    None => format!("{}Proximity(min={}°)", $body_name, self.min_angle_deg),
                }
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
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[i - 1].to_rfc3339(),
                max_severity,
                description: get_description(start_idx, false),
            });
            current_violation = None;
        }
    }

    // Close any open violation at the end
    if let Some((start_idx, max_severity)) = current_violation {
        violations.push(ConstraintViolation {
            start_time: times[start_idx].to_rfc3339(),
            end_time: times[times.len() - 1].to_rfc3339(),
            max_severity,
            description: get_description(start_idx, true),
        });
    }

    violations
}
