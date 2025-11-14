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
use crate::time_utils::python_datetime_to_utc;
use chrono::{DateTime, Datelike, Timelike, Utc};
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// Result of constraint evaluation containing all violations
#[pyclass(name = "ConstraintResult")]
#[derive(Clone, Debug)]
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

    /// Internal: compute boolean array indicating if constraint is satisfied at each time
    fn _compute_constraint_vec(&self) -> Vec<bool> {
        if self.times.is_empty() {
            return Vec::new();
        }
        let mut ok = vec![true; self.times.len()];
        for (i, t) in self.times.iter().enumerate() {
            let t_str = t.to_rfc3339();
            for v in &self.violations {
                if v.start_time <= t_str && t_str <= v.end_time {
                    ok[i] = false;
                    break;
                }
            }
        }
        ok
    }

    /// Property: array of booleans for each timestamp where True means constraint satisfied
    #[getter]
    fn constraint_array(&self, py: Python) -> PyResult<Py<PyAny>> {
        let arr = self._compute_constraint_vec();
        let np = pyo3::types::PyModule::import(py, "numpy")
            .map_err(|_| pyo3::exceptions::PyImportError::new_err("numpy is required"))?;
        let py_arr = np.getattr("array")?.call1((arr,))?;
        Ok(py_arr.into())
    }

    /// Property: array of Python datetime objects for each evaluation time
    #[getter]
    fn timestamp(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let datetime_mod = py.import("datetime")?;
        let timezone_class = datetime_mod.getattr("timezone")?;
        let timezone_utc = timezone_class.getattr("utc")?;

        let mut result = Vec::with_capacity(self.times.len());
        for dt in &self.times {
            let py_dt = datetime_mod.getattr("datetime")?.call1((
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                dt.second(),
                dt.timestamp_subsec_micros(),
                timezone_utc.clone(),
            ))?;
            result.push(py_dt.into());
        }
        Ok(result)
    }

    /// Check if the target is in-constraint at a given time.
    /// Accepts a Python datetime object (naive datetimes are treated as UTC).
    fn in_constraint(&self, _py: Python, time: &Bound<PyAny>) -> PyResult<bool> {
        let dt = python_datetime_to_utc(time)?;

        // Find matching time in our array
        if let Some(idx) = self.times.iter().position(|t| t == &dt) {
            let ok_vec = self._compute_constraint_vec();
            Ok(ok_vec[idx])
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "time not found in evaluated timestamps",
            ))
        }
    }
}

/// Configuration for constraint evaluation
///
/// This is the base trait that all constraint configurations must implement.
/// It allows serialization to/from JSON and Python dictionaries.
pub trait ConstraintConfig: fmt::Debug + Send + Sync {
    /// Create a constraint evaluator from this configuration
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator>;

    /// Get a human-readable name for this constraint
    #[allow(dead_code)]
    fn name(&self) -> String;

    /// Serialize to JSON
    #[allow(dead_code)]
    fn to_json(&self) -> String;
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

/// Configuration for Sun proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SunProximityConfig {
    /// Minimum allowed angular separation from Sun in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from Sun in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for SunProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(SunProximityEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }

    fn name(&self) -> String {
        match self.max_angle {
            Some(max) => format!("SunProximity(min={}°, max={}°)", self.min_angle, max),
            None => format!("SunProximity(min={}°)", self.min_angle),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Evaluator for Sun proximity constraint
struct SunProximityEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl ConstraintEvaluator for SunProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            times,
            |i| {
                let sun_pos = [
                    sun_positions[[i, 0]],
                    sun_positions[[i, 1]],
                    sun_positions[[i, 2]],
                ];
                let obs_pos = [
                    observer_positions[[i, 0]],
                    observer_positions[[i, 1]],
                    observer_positions[[i, 2]],
                ];
                let angle_deg = calculate_angular_separation(&target_vec, &sun_pos, &obs_pos);

                let is_violated = angle_deg < self.min_angle_deg
                    || self.max_angle_deg.is_some_and(|max| angle_deg > max);

                let severity = if angle_deg < self.min_angle_deg {
                    (self.min_angle_deg - angle_deg) / self.min_angle_deg
                } else if let Some(max) = self.max_angle_deg {
                    (angle_deg - max) / max
                } else {
                    0.0
                };

                (is_violated, severity)
            },
            |_, is_final| {
                if is_final {
                    self.final_violation_description()
                } else {
                    // Get angle for description (recompute at violation end)
                    "Target violates Sun proximity constraint".to_string()
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult {
            violations,
            all_satisfied,
            constraint_name: self.name(),
            times: times.to_vec(),
        }
    }

    fn name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!("SunProximity(min={}°, max={}°)", self.min_angle_deg, max),
            None => format!("SunProximity(min={}°)", self.min_angle_deg),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl SunProximityEvaluator {
    fn final_violation_description(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to Sun (min: {:.1}°) or too far (max: {:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to Sun (min allowed: {:.1}°)",
                self.min_angle_deg
            ),
        }
    }
}

/// Configuration for Moon proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonProximityConfig {
    /// Minimum allowed angular separation from Moon in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from Moon in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for MoonProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonProximityEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }

    fn name(&self) -> String {
        match self.max_angle {
            Some(max) => format!("MoonProximity(min={}°, max={}°)", self.min_angle, max),
            None => format!("MoonProximity(min={}°)", self.min_angle),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Evaluator for Moon proximity constraint
struct MoonProximityEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl ConstraintEvaluator for MoonProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            times,
            |i| {
                let moon_pos = [
                    moon_positions[[i, 0]],
                    moon_positions[[i, 1]],
                    moon_positions[[i, 2]],
                ];
                let obs_pos = [
                    observer_positions[[i, 0]],
                    observer_positions[[i, 1]],
                    observer_positions[[i, 2]],
                ];
                let angle_deg = calculate_angular_separation(&target_vec, &moon_pos, &obs_pos);

                let is_violated = angle_deg < self.min_angle_deg
                    || self.max_angle_deg.is_some_and(|max| angle_deg > max);

                let severity = if angle_deg < self.min_angle_deg {
                    (self.min_angle_deg - angle_deg) / self.min_angle_deg
                } else if let Some(max) = self.max_angle_deg {
                    (angle_deg - max) / max
                } else {
                    0.0
                };

                (is_violated, severity)
            },
            |_, is_final| {
                if is_final {
                    self.final_violation_description()
                } else {
                    "Target violates Moon proximity constraint".to_string()
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult {
            violations,
            all_satisfied,
            constraint_name: self.name(),
            times: times.to_vec(),
        }
    }

    fn name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!("MoonProximity(min={}°, max={}°)", self.min_angle_deg, max),
            None => format!("MoonProximity(min={}°)", self.min_angle_deg),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MoonProximityEvaluator {
    fn final_violation_description(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to Moon (min: {:.1}°) or too far (max: {:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to Moon (min allowed: {:.1}°)",
                self.min_angle_deg
            ),
        }
    }
}
/// Configuration for Earth limb avoidance constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthLimbConfig {
    /// Additional margin beyond the Earth's apparent angular radius (degrees)
    pub min_angle: f64,
    /// Maximum allowed angular separation from Earth's limb in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for EarthLimbConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(EarthLimbEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }

    fn name(&self) -> String {
        match self.max_angle {
            Some(max) => format!("EarthLimb(min={}°, max={}°)", self.min_angle, max),
            None => format!("EarthLimb(min={}°)", self.min_angle),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Evaluator for Earth limb avoidance
struct EarthLimbEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl ConstraintEvaluator for EarthLimbEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;

        // Convert target RA/Dec to unit vector
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        for (i, _time) in times.iter().enumerate() {
            // Vector from observer to Earth center is -observer position
            let obs_pos = [
                observer_positions[[i, 0]],
                observer_positions[[i, 1]],
                observer_positions[[i, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            let ratio = (EARTH_RADIUS / r).clamp(-1.0, 1.0);
            let earth_ang_radius_deg = ratio.asin().to_degrees();
            let threshold_deg = earth_ang_radius_deg + self.min_angle_deg;

            let center_unit = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
            let cos_angle = dot_product(&target_vec, &center_unit);
            let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

            let is_violation = if let Some(max_angle) = self.max_angle_deg {
                angle_deg < threshold_deg || angle_deg > max_angle
            } else {
                angle_deg < threshold_deg
            };

            if is_violation {
                let severity = if angle_deg < threshold_deg {
                    (threshold_deg - angle_deg) / threshold_deg.max(1e-9)
                } else if let Some(max_angle) = self.max_angle_deg {
                    // For max angle violations, severity increases as angle exceeds max
                    (angle_deg - max_angle) / max_angle.max(1e-9)
                } else {
                    0.0 // This shouldn't happen since is_violation would be false
                };
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
                    description: match self.max_angle_deg {
                        Some(max) => format!(
                            "Target within Earth limb + margin (min: {threshold_deg:.1}°, max: {max:.1}°)"
                        ),
                        None => format!(
                            "Target within Earth limb + margin (min allowed: {threshold_deg:.1}°)"
                        ),
                    },
                });
                current_violation = None;
            }
        }

        if let Some((start_idx, max_severity)) = current_violation {
            // Compute threshold at final time for description consistency
            let obs_pos = [
                observer_positions[[times.len() - 1, 0]],
                observer_positions[[times.len() - 1, 1]],
                observer_positions[[times.len() - 1, 2]],
            ];
            let r = vector_magnitude(&obs_pos);
            let ratio = (EARTH_RADIUS / r).clamp(-1.0, 1.0);
            let earth_ang_radius_deg = ratio.asin().to_degrees();
            let threshold_deg = earth_ang_radius_deg + self.min_angle_deg;

            violations.push(ConstraintViolation {
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity,
                description: match self.max_angle_deg {
                    Some(max) => format!(
                        "Target within Earth limb + margin (min: {threshold_deg:.1}°, max: {max:.1}°)"
                    ),
                    None => format!(
                        "Target within Earth limb + margin (min allowed: {threshold_deg:.1}°)"
                    ),
                },
            });
        }

        let all_satisfied = violations.is_empty();
        ConstraintResult {
            violations,
            all_satisfied,
            constraint_name: self.name(),
            times: times.to_vec(),
        }
    }

    fn name(&self) -> String {
        format!("EarthLimb(min={}°)", self.min_angle_deg)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Configuration for eclipse constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EclipseConfig {
    /// Umbra only (true) or include penumbra (false)
    pub umbra_only: bool,
}

impl ConstraintConfig for EclipseConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(EclipseEvaluator {
            umbra_only: self.umbra_only,
        })
    }

    fn name(&self) -> String {
        format!(
            "Eclipse({})",
            if self.umbra_only {
                "umbra"
            } else {
                "umbra+penumbra"
            }
        )
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Evaluator for eclipse constraint
struct EclipseEvaluator {
    umbra_only: bool,
}

impl ConstraintEvaluator for EclipseEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        _target_ra: f64,
        _target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;
        // Sun radius in km (mean)
        const SUN_RADIUS: f64 = 696000.0;

        for (i, _time) in times.iter().enumerate() {
            let obs_pos = [
                observer_positions[[i, 0]],
                observer_positions[[i, 1]],
                observer_positions[[i, 2]],
            ];

            let sun_pos = [
                sun_positions[[i, 0]],
                sun_positions[[i, 1]],
                sun_positions[[i, 2]],
            ];

            // Vector from observer to Sun
            let obs_to_sun = [
                sun_pos[0] - obs_pos[0],
                sun_pos[1] - obs_pos[1],
                sun_pos[2] - obs_pos[2],
            ];

            let sun_dist = vector_magnitude(&obs_to_sun);
            let sun_unit = normalize_vector(&obs_to_sun);

            // Find closest point on observer-to-Sun line to Earth center
            let t =
                -(obs_pos[0] * sun_unit[0] + obs_pos[1] * sun_unit[1] + obs_pos[2] * sun_unit[2]);

            // If closest point is behind observer or beyond Sun, not in shadow
            if t < 0.0 || t > sun_dist {
                // Close any open violation
                if let Some((start_idx, max_severity)) = current_violation {
                    violations.push(ConstraintViolation {
                        start_time: times[start_idx].to_rfc3339(),
                        end_time: times[i - 1].to_rfc3339(),
                        max_severity,
                        description: if self.umbra_only {
                            "Observer in umbra".to_string()
                        } else {
                            "Observer in shadow".to_string()
                        },
                    });
                    current_violation = None;
                }
                continue;
            }

            // Closest point on line to Earth center
            let closest_point = [
                obs_pos[0] + t * sun_unit[0],
                obs_pos[1] + t * sun_unit[1],
                obs_pos[2] + t * sun_unit[2],
            ];

            // Distance from Earth center to closest point
            let dist_to_earth = vector_magnitude(&closest_point);

            // Calculate umbra and penumbra radii at observer distance
            let umbra_radius = EARTH_RADIUS - (EARTH_RADIUS - SUN_RADIUS) * t / sun_dist;
            let penumbra_radius = EARTH_RADIUS + (SUN_RADIUS - EARTH_RADIUS) * t / sun_dist;

            let (in_shadow, severity) = if dist_to_earth < umbra_radius {
                // In umbra
                (true, 1.0 - dist_to_earth / umbra_radius)
            } else if !self.umbra_only && dist_to_earth < penumbra_radius {
                // In penumbra
                let penumbra_depth =
                    (penumbra_radius - dist_to_earth) / (penumbra_radius - umbra_radius);
                (true, 0.5 * penumbra_depth)
            } else {
                (false, 0.0)
            };

            if in_shadow {
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
                    description: if self.umbra_only {
                        "Observer in umbra".to_string()
                    } else {
                        "Observer in shadow".to_string()
                    },
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
                description: if self.umbra_only {
                    "Observer in umbra".to_string()
                } else {
                    "Observer in shadow".to_string()
                },
            });
        }

        let all_satisfied = violations.is_empty();
        ConstraintResult {
            violations,
            all_satisfied,
            constraint_name: self.name(),
            times: times.to_vec(),
        }
    }

    fn name(&self) -> String {
        format!(
            "Eclipse({})",
            if self.umbra_only {
                "umbra"
            } else {
                "umbra+penumbra"
            }
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Configuration for generic solar system body proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyProximityConfig {
    /// Body identifier (NAIF ID or name, e.g., "Jupiter", "499")
    pub body: String,
    /// Minimum allowed angular separation in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for BodyProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(BodyProximityEvaluator {
            body: self.body.clone(),
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }

    fn name(&self) -> String {
        match self.max_angle {
            Some(max) => format!(
                "BodyProximity(body='{}', min={}°, max={}°)",
                self.body, self.min_angle, max
            ),
            None => format!(
                "BodyProximity(body='{}', min={}°)",
                self.body, self.min_angle
            ),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Evaluator for generic body proximity - requires body positions computed externally
pub struct BodyProximityEvaluator {
    pub body: String,
    pub min_angle_deg: f64,
    pub max_angle_deg: Option<f64>,
}

impl ConstraintEvaluator for BodyProximityEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Body positions are passed via sun_positions slot
        let body_positions = sun_positions;
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            times,
            |i| {
                let body_pos = [
                    body_positions[[i, 0]],
                    body_positions[[i, 1]],
                    body_positions[[i, 2]],
                ];
                let obs_pos = [
                    observer_positions[[i, 0]],
                    observer_positions[[i, 1]],
                    observer_positions[[i, 2]],
                ];
                let angle_deg = calculate_angular_separation(&target_vec, &body_pos, &obs_pos);

                let is_violation = if let Some(max_angle) = self.max_angle_deg {
                    angle_deg < self.min_angle_deg || angle_deg > max_angle
                } else {
                    angle_deg < self.min_angle_deg
                };

                let severity = if angle_deg < self.min_angle_deg {
                    (self.min_angle_deg - angle_deg) / self.min_angle_deg
                } else if let Some(max_angle) = self.max_angle_deg {
                    (angle_deg - max_angle) / max_angle.max(1e-9)
                } else {
                    0.0
                };

                (is_violation, severity)
            },
            |_, is_final| {
                if is_final {
                    match self.max_angle_deg {
                        Some(max) => format!(
                            "Target too close to {} (min: {:.1}°, max: {:.1}°)",
                            self.body, self.min_angle_deg, max
                        ),
                        None => format!(
                            "Target too close to {} (min allowed: {:.1}°)",
                            self.body, self.min_angle_deg
                        ),
                    }
                } else {
                    format!("Target violates {} proximity constraint", self.body)
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult {
            violations,
            all_satisfied,
            constraint_name: self.name(),
            times: times.to_vec(),
        }
    }

    fn name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "BodyProximity(body='{}', min={}°, max={}°)",
                self.body, self.min_angle_deg, max
            ),
            None => format!(
                "BodyProximity(body='{}', min={}°)",
                self.body, self.min_angle_deg
            ),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Logical AND combinator
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndConfig {
    pub constraints: Vec<serde_json::Value>,
}

/// Logical OR combinator
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrConfig {
    pub constraints: Vec<serde_json::Value>,
}

/// Logical NOT combinator
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotConfig {
    pub constraint: Box<serde_json::Value>,
}

// Helper function for tracking violation windows
fn track_violations<F>(
    times: &[DateTime<Utc>],
    mut is_violated: F,
    mut get_description: impl FnMut(usize, bool) -> String,
) -> Vec<ConstraintViolation>
where
    F: FnMut(usize) -> (bool, f64),
{
    let mut violations = Vec::new();
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

// Helper functions for vector math
fn radec_to_unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra_rad = ra_deg.to_radians();
    let dec_rad = dec_deg.to_radians();
    let cos_dec = dec_rad.cos();
    [
        cos_dec * ra_rad.cos(),
        cos_dec * ra_rad.sin(),
        dec_rad.sin(),
    ]
}

fn normalize_vector(v: &[f64; 3]) -> [f64; 3] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag, v[2] / mag]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vector_magnitude(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// Helper function for calculating angular separation between target and body
fn calculate_angular_separation(
    target_vec: &[f64; 3],
    body_position: &[f64; 3],
    observer_position: &[f64; 3],
) -> f64 {
    let body_rel = [
        body_position[0] - observer_position[0],
        body_position[1] - observer_position[1],
        body_position[2] - observer_position[2],
    ];
    let body_unit = normalize_vector(&body_rel);
    let cos_angle = dot_product(target_vec, &body_unit);
    cos_angle.clamp(-1.0, 1.0).acos().to_degrees()
}
