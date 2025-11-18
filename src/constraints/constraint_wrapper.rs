/// Python wrapper for constraint system
///
/// This module provides the Python API for constraint evaluation,
/// including JSON-based configuration and convenient factory methods.
use crate::constraints::body_proximity::{BodyProximityConfig, BodyProximityEvaluator};
use crate::constraints::core::*;
use crate::constraints::earth_limb::EarthLimbConfig;
use crate::constraints::eclipse::EclipseConfig;
use crate::constraints::moon_proximity::MoonProximityConfig;
use crate::constraints::sun_proximity::SunProximityConfig;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

/// Python-facing constraint evaluator
///
/// This wraps the Rust constraint system and provides a convenient Python API.
#[pyclass(name = "Constraint")]
pub struct PyConstraint {
    evaluator: Box<dyn ConstraintEvaluator>,
    config_json: String,
}

impl PyConstraint {
    /// Internal helper to evaluate against any Ephemeris implementing EphemerisBase
    #[allow(deprecated)] // Python::with_gil is appropriate in this non-pyo3 context
    fn eval_with_ephemeris<E: EphemerisBase>(
        &self,
        ephemeris: &E,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<ConstraintResult> {
        let mut times = ephemeris.get_times()?;
        let mut sun_positions = ephemeris.get_sun_positions()?;
        let mut moon_positions = ephemeris.get_moon_positions()?;
        let mut observer_positions = ephemeris.get_gcrs_positions()?;

        // Filter by time indices if provided
        if let Some(ref indices) = time_indices {
            // Validate indices
            let n_times = times.len();
            for &idx in indices {
                if idx >= n_times {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Index {idx} out of range for ephemeris with {n_times} timestamps"
                    )));
                }
            }

            // Filter times and positions
            times = indices.iter().map(|&i| times[i]).collect();
            sun_positions =
                Array2::from_shape_fn((indices.len(), 3), |(i, j)| sun_positions[[indices[i], j]]);
            moon_positions =
                Array2::from_shape_fn((indices.len(), 3), |(i, j)| moon_positions[[indices[i], j]]);
            observer_positions = Array2::from_shape_fn((indices.len(), 3), |(i, j)| {
                observer_positions[[indices[i], j]]
            });
        }

        // Special handling for body proximity - need to compute body positions
        if let Some(body_eval) = self
            .evaluator
            .as_any()
            .downcast_ref::<BodyProximityEvaluator>()
        {
            // Get body positions via get_body_pv
            use pyo3::Python;
            let result = Python::with_gil(|py| {
                let body_pv = ephemeris.get_body_pv(py, &body_eval.body)?;
                let body_pv_ref = body_pv.bind(py).borrow();
                let mut body_positions = body_pv_ref.position.clone();

                // Filter body positions if time_indices was provided
                if let Some(ref indices) = time_indices {
                    body_positions = Array2::from_shape_fn((indices.len(), 3), |(i, j)| {
                        body_positions[[indices[i], j]]
                    });
                }

                // Pass body positions via sun_positions slot (a bit hacky but avoids API changes)
                Ok::<_, PyErr>(self.evaluator.evaluate(
                    &times,
                    target_ra,
                    target_dec,
                    &body_positions,
                    &moon_positions,
                    &observer_positions,
                ))
            })?;
            return Ok(result);
        }

        Ok(self.evaluator.evaluate(
            &times,
            target_ra,
            target_dec,
            &sun_positions,
            &moon_positions,
            &observer_positions,
        ))
    }
}

#[pymethods]
impl PyConstraint {
    /// Create a Sun proximity constraint
    ///
    /// Args:
    ///     min_angle (float): Minimum allowed angular separation from Sun in degrees
    ///     max_angle (float, optional): Maximum allowed angular separation from Sun in degrees
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(min_angle, max_angle=None))]
    #[staticmethod]
    fn sun_proximity(min_angle: f64, max_angle: Option<f64>) -> PyResult<Self> {
        if !(0.0..=180.0).contains(&min_angle) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_angle must be between 0 and 180 degrees",
            ));
        }

        if let Some(max) = max_angle {
            if !(0.0..=180.0).contains(&max) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be between 0 and 180 degrees",
                ));
            }
            if max <= min_angle {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be greater than min_angle",
                ));
            }
        }

        let config = SunProximityConfig {
            min_angle,
            max_angle,
        };
        let mut json_obj = serde_json::json!({
            "type": "sun",
            "min_angle": min_angle
        });
        if let Some(max) = max_angle {
            json_obj["max_angle"] = serde_json::json!(max);
        }
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create a Moon proximity constraint
    ///
    /// Args:
    ///     min_angle (float): Minimum allowed angular separation from Moon in degrees
    ///     max_angle (float, optional): Maximum allowed angular separation from Moon in degrees
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(min_angle, max_angle=None))]
    #[staticmethod]
    fn moon_proximity(min_angle: f64, max_angle: Option<f64>) -> PyResult<Self> {
        if !(0.0..=180.0).contains(&min_angle) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_angle must be between 0 and 180 degrees",
            ));
        }

        if let Some(max) = max_angle {
            if !(0.0..=180.0).contains(&max) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be between 0 and 180 degrees",
                ));
            }
            if max <= min_angle {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be greater than min_angle",
                ));
            }
        }

        let config = MoonProximityConfig {
            min_angle,
            max_angle,
        };
        let mut json_obj = serde_json::json!({
            "type": "moon",
            "min_angle": min_angle
        });
        if let Some(max) = max_angle {
            json_obj["max_angle"] = serde_json::json!(max);
        }
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create an eclipse constraint
    ///
    /// Args:
    ///     umbra_only (bool): If True, only umbra counts as eclipse. If False, penumbra also counts.
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[staticmethod]
    #[pyo3(signature = (umbra_only=true))]
    fn eclipse(umbra_only: bool) -> PyResult<Self> {
        let config = EclipseConfig { umbra_only };
        let config_json = serde_json::json!({
            "type": "eclipse",
            "umbra_only": umbra_only
        })
        .to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create an Earth limb avoidance constraint
    ///
    /// Args:
    ///     min_angle (float): Additional margin beyond Earth's apparent angular radius (degrees)
    ///     max_angle (float, optional): Maximum allowed angular separation from Earth limb (degrees)
    ///     include_refraction (bool, optional): Include atmospheric refraction correction for ground observers (default: False)
    ///     horizon_dip (bool, optional): Include geometric horizon dip correction for ground observers (default: False)
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(min_angle, max_angle=None, include_refraction=false, horizon_dip=false))]
    #[staticmethod]
    fn earth_limb(
        min_angle: f64,
        max_angle: Option<f64>,
        include_refraction: bool,
        horizon_dip: bool,
    ) -> PyResult<Self> {
        if !(0.0..=180.0).contains(&min_angle) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_angle must be between 0 and 180 degrees",
            ));
        }

        if let Some(max) = max_angle {
            if !(0.0..=180.0).contains(&max) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be between 0 and 180 degrees",
                ));
            }
            if max <= min_angle {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be greater than min_angle",
                ));
            }
        }

        let config = EarthLimbConfig {
            min_angle,
            max_angle,
            include_refraction,
            horizon_dip,
        };
        let mut json_obj = serde_json::json!({
            "type": "earth_limb",
            "min_angle": min_angle,
            "include_refraction": include_refraction
        });
        if let Some(max) = max_angle {
            json_obj["max_angle"] = serde_json::json!(max);
        }
        json_obj["horizon_dip"] = serde_json::json!(horizon_dip);
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create a generic solar system body avoidance constraint
    ///
    /// Args:
    ///     body (str): Body identifier - NAIF ID or name (e.g., "Jupiter", "499", "Mars")
    ///     min_angle (float): Minimum allowed angular separation in degrees
    ///     max_angle (float, optional): Maximum allowed angular separation in degrees
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Note:
    ///     Supported bodies depend on the ephemeris type and loaded kernels.
    ///     Common bodies: Sun (10), Moon (301), planets (199, 299, 399, 499, 599, 699, 799, 899)
    #[pyo3(signature=(body, min_angle, max_angle=None))]
    #[staticmethod]
    fn body_proximity(body: String, min_angle: f64, max_angle: Option<f64>) -> PyResult<Self> {
        if !(0.0..=180.0).contains(&min_angle) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_angle must be between 0 and 180 degrees",
            ));
        }

        if let Some(max) = max_angle {
            if !(0.0..=180.0).contains(&max) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be between 0 and 180 degrees",
                ));
            }
            if max <= min_angle {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_angle must be greater than min_angle",
                ));
            }
        }

        let config = BodyProximityConfig {
            body: body.clone(),
            min_angle,
            max_angle,
        };
        let mut json_obj = serde_json::json!({
            "type": "body",
            "body": body,
            "min_angle": min_angle
        });
        if let Some(max) = max_angle {
            json_obj["max_angle"] = serde_json::json!(max);
        }
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create a constraint from JSON configuration
    ///
    /// Args:
    ///     json_str (str): JSON string containing constraint configuration
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Example JSON formats:
    ///     {"type": "sun", "min_angle": 45.0}
    ///     {"type": "moon", "min_angle": 10.0}
    ///     {"type": "eclipse", "umbra_only": true}
    ///     {"type": "and", "constraints": [...]}
    ///     {"type": "or", "constraints": [...]}
    ///     {"type": "xor", "constraints": [...]}  // exactly one violated -> violation
    ///     {"type": "not", "constraint": {...}}
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {e}")))?;

        let evaluator = parse_constraint_json(&value)?;

        Ok(PyConstraint {
            evaluator,
            config_json: json_str.to_string(),
        })
    }

    /// Combine constraints with logical AND
    ///
    /// Args:
    ///     *constraints: Variable number of Constraint objects
    ///
    /// Returns:
    ///     Constraint: A new constraint that is satisfied only if all input constraints are satisfied
    #[staticmethod]
    #[pyo3(name = "and_", signature = (*constraints))]
    fn and(constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        if constraints.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one constraint required for AND",
            ));
        }

        let configs: Vec<serde_json::Value> = constraints
            .iter()
            .map(|c| serde_json::from_str(&c.config_json).unwrap())
            .collect();

        let config_json = serde_json::json!({
            "type": "and",
            "constraints": configs
        })
        .to_string();

        let evaluator = parse_constraint_json(&serde_json::from_str(&config_json).unwrap())?;

        Ok(PyConstraint {
            evaluator,
            config_json,
        })
    }

    /// Combine constraints with logical OR
    ///
    /// Args:
    ///     *constraints: Variable number of Constraint objects
    ///
    /// Returns:
    ///     Constraint: A new constraint that is satisfied if any input constraint is satisfied
    #[staticmethod]
    #[pyo3(name = "or_", signature = (*constraints))]
    fn or(constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        if constraints.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one constraint required for OR",
            ));
        }

        let configs: Vec<serde_json::Value> = constraints
            .iter()
            .map(|c| serde_json::from_str(&c.config_json).unwrap())
            .collect();

        let config_json = serde_json::json!({
            "type": "or",
            "constraints": configs
        })
        .to_string();

        let evaluator = parse_constraint_json(&serde_json::from_str(&config_json).unwrap())?;

        Ok(PyConstraint {
            evaluator,
            config_json,
        })
    }

    /// Combine constraints with logical XOR
    ///
    /// Args:
    ///     *constraints: Variable number of Constraint objects (minimum 2)
    ///
    /// Returns:
    ///     Constraint: A new constraint that is violated when EXACTLY ONE input constraint is violated
    #[staticmethod]
    #[pyo3(name = "xor_", signature = (*constraints))]
    fn xor(constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        if constraints.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least two constraints required for XOR",
            ));
        }

        let configs: Vec<serde_json::Value> = constraints
            .iter()
            .map(|c| serde_json::from_str(&c.config_json).unwrap())
            .collect();

        let config_json = serde_json::json!({
            "type": "xor",
            "constraints": configs
        })
        .to_string();

        let evaluator = parse_constraint_json(&serde_json::from_str(&config_json).unwrap())?;

        Ok(PyConstraint {
            evaluator,
            config_json,
        })
    }

    /// Negate a constraint with logical NOT
    ///
    /// Args:
    ///     constraint (Constraint): Constraint to negate
    ///
    /// Returns:
    ///     Constraint: A new constraint that is satisfied when the input is violated
    #[staticmethod]
    #[pyo3(name = "not_")]
    fn not(constraint: PyRef<PyConstraint>) -> PyResult<Self> {
        let config: serde_json::Value = serde_json::from_str(&constraint.config_json).unwrap();

        let config_json = serde_json::json!({
            "type": "not",
            "constraint": config
        })
        .to_string();

        let evaluator = parse_constraint_json(&serde_json::from_str(&config_json).unwrap())?;

        Ok(PyConstraint {
            evaluator,
            config_json,
        })
    }

    /// Evaluate constraint against any supported ephemeris type
    ///
    /// Args:
    ///     ephemeris: One of `TLEEphemeris`, `SPICEEphemeris`, or `GroundEphemeris`
    ///     target_ra (float): Target right ascension in degrees (ICRS/J2000)
    ///     target_dec (float): Target declination in degrees (ICRS/J2000)
    ///     times (datetime or list[datetime], optional): Specific time(s) to evaluate.
    ///         Can be a single datetime or list of datetimes. If provided, only these
    ///         times will be evaluated (must exist in the ephemeris).
    ///     indices (int or list[int], optional): Specific time index/indices to evaluate.
    ///         Can be a single index or list of indices into the ephemeris timestamp array.
    ///
    /// Returns:
    ///     ConstraintResult: Result containing violation windows
    ///
    /// Note:
    ///     Only one of `times` or `indices` should be provided. If neither is provided,
    ///     all ephemeris times are evaluated.
    #[pyo3(signature = (ephemeris, target_ra, target_dec, times=None, indices=None))]
    fn evaluate(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        target_ra: f64,
        target_dec: f64,
        times: Option<&Bound<PyAny>>,
        indices: Option<&Bound<PyAny>>,
    ) -> PyResult<ConstraintResult> {
        // Parse time filtering options
        let bound = ephemeris.bind(py);
        let time_indices = if let Some(times_arg) = times {
            if indices.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both 'times' and 'indices' parameters",
                ));
            }
            Some(self.parse_times_to_indices(bound, times_arg)?)
        } else if let Some(indices_arg) = indices {
            Some(self.parse_indices(indices_arg)?)
        } else {
            None
        };

        if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            return self.eval_with_ephemeris(&*ephem, target_ra, target_dec, time_indices);
        }
        if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            return self.eval_with_ephemeris(&*ephem, target_ra, target_dec, time_indices);
        }
        if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            return self.eval_with_ephemeris(&*ephem, target_ra, target_dec, time_indices);
        }
        if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            return self.eval_with_ephemeris(&*ephem, target_ra, target_dec, time_indices);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
        ))
    }

    /// Helper to parse times parameter and convert to indices
    fn parse_times_to_indices(
        &self,
        ephemeris: &Bound<PyAny>,
        times_arg: &Bound<PyAny>,
    ) -> PyResult<Vec<usize>> {
        use std::collections::HashMap;

        // Get ephemeris times - need to clone to avoid lifetime issues
        let ephem_times: Vec<DateTime<Utc>> =
            if let Ok(ephem) = ephemeris.extract::<PyRef<TLEEphemeris>>() {
                ephem.data().times.as_ref().cloned()
            } else if let Ok(ephem) = ephemeris.extract::<PyRef<SPICEEphemeris>>() {
                ephem.data().times.as_ref().cloned()
            } else if let Ok(ephem) = ephemeris.extract::<PyRef<GroundEphemeris>>() {
                ephem.data().times.as_ref().cloned()
            } else if let Ok(ephem) = ephemeris.extract::<PyRef<OEMEphemeris>>() {
                ephem.data().times.as_ref().cloned()
            } else {
                None
            }
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No times in ephemeris"))?;

        // Parse input times (single datetime or list)
        let input_times: Vec<DateTime<Utc>> = if times_arg.is_instance_of::<pyo3::types::PyList>() {
            let list = times_arg.downcast::<pyo3::types::PyList>()?;
            list.iter()
                .map(|item| {
                    let year: i32 = item.getattr("year")?.extract()?;
                    let month: u32 = item.getattr("month")?.extract()?;
                    let day: u32 = item.getattr("day")?.extract()?;
                    let hour: u32 = item.getattr("hour")?.extract()?;
                    let minute: u32 = item.getattr("minute")?.extract()?;
                    let second: u32 = item.getattr("second")?.extract()?;
                    let microsecond: u32 = item.getattr("microsecond")?.extract()?;

                    chrono::NaiveDate::from_ymd_opt(year, month, day)
                        .and_then(|d| d.and_hms_micro_opt(hour, minute, second, microsecond))
                        .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid datetime"))
                })
                .collect::<PyResult<_>>()?
        } else {
            // Single datetime
            let year: i32 = times_arg.getattr("year")?.extract()?;
            let month: u32 = times_arg.getattr("month")?.extract()?;
            let day: u32 = times_arg.getattr("day")?.extract()?;
            let hour: u32 = times_arg.getattr("hour")?.extract()?;
            let minute: u32 = times_arg.getattr("minute")?.extract()?;
            let second: u32 = times_arg.getattr("second")?.extract()?;
            let microsecond: u32 = times_arg.getattr("microsecond")?.extract()?;

            let dt = chrono::NaiveDate::from_ymd_opt(year, month, day)
                .and_then(|d| d.and_hms_micro_opt(hour, minute, second, microsecond))
                .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid datetime"))?;

            vec![dt]
        };

        // Build HashMap for O(1) lookup when multiple times are requested
        let mut indices = Vec::with_capacity(input_times.len());

        if input_times.len() > 3 {
            // Use HashMap for multiple lookups
            let time_map: HashMap<DateTime<Utc>, usize> = ephem_times
                .iter()
                .enumerate()
                .map(|(i, t)| (*t, i))
                .collect();

            for dt in input_times {
                if let Some(&idx) = time_map.get(&dt) {
                    indices.push(idx);
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Time {} not found in ephemeris timestamps",
                        dt.to_rfc3339()
                    )));
                }
            }
        } else {
            // Use linear search for small number of lookups
            for dt in input_times {
                if let Some(idx) = ephem_times.iter().position(|t| t == &dt) {
                    indices.push(idx);
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Time {} not found in ephemeris timestamps",
                        dt.to_rfc3339()
                    )));
                }
            }
        }

        Ok(indices)
    }

    /// Helper to parse indices parameter
    fn parse_indices(&self, indices_arg: &Bound<PyAny>) -> PyResult<Vec<usize>> {
        if indices_arg.is_instance_of::<pyo3::types::PyList>() {
            let list = indices_arg.downcast::<pyo3::types::PyList>()?;
            list.iter()
                .map(|item| item.extract::<usize>())
                .collect::<PyResult<_>>()
        } else {
            // Single index
            let idx: usize = indices_arg.extract()?;
            Ok(vec![idx])
        }
    }

    /// Check if the target violates the constraint at a given time
    ///
    /// Args:
    ///     time (datetime): The time to check (must exist in ephemeris)
    ///     ephemeris: One of `TLEEphemeris`, `SPICEEphemeris`, or `GroundEphemeris`
    ///     target_ra (float): Target right ascension in degrees (ICRS/J2000)
    ///     target_dec (float): Target declination in degrees (ICRS/J2000)
    ///
    /// Returns:
    ///     bool: True if constraint is violated at the given time, False otherwise
    fn in_constraint(
        &self,
        py: Python,
        time: &Bound<PyAny>,
        ephemeris: Py<PyAny>,
        target_ra: f64,
        target_dec: f64,
    ) -> PyResult<bool> {
        // Evaluate constraint for just this single time
        let result = self.evaluate(py, ephemeris, target_ra, target_dec, Some(time), None)?;

        // Check if there are any violations
        // If violations exist, the constraint is violated (in-constraint)
        Ok(!result.all_satisfied)
    }

    /// Get constraint configuration as JSON string
    fn to_json(&self) -> String {
        self.config_json.clone()
    }

    /// Get constraint configuration as Python dictionary
    fn to_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let json_value: serde_json::Value = serde_json::from_str(&self.config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {e}")))?;
        json_to_pyobject(py, &json_value)
    }

    fn __repr__(&self) -> String {
        format!("Constraint({})", self.evaluator.name())
    }
}

// Helper function to parse constraint JSON into evaluator
fn parse_constraint_json(value: &serde_json::Value) -> PyResult<Box<dyn ConstraintEvaluator>> {
    let constraint_type = value.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing or invalid 'type' field in JSON")
    })?;

    match constraint_type {
        "sun" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = SunProximityConfig {
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "moon" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = MoonProximityConfig {
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "eclipse" => {
            let umbra_only = value
                .get("umbra_only")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let config = EclipseConfig { umbra_only };
            Ok(config.to_evaluator())
        }
        "earth_limb" => {
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let include_refraction = value
                .get("include_refraction")
                .and_then(|v| v.as_bool())
                .unwrap_or(false); // Default to false if not specified
            let horizon_dip = value
                .get("horizon_dip")
                .and_then(|v| v.as_bool())
                .unwrap_or(false); // Default to false if not specified
            let config = EarthLimbConfig {
                min_angle,
                max_angle,
                include_refraction,
                horizon_dip,
            };
            Ok(config.to_evaluator())
        }
        "body" => {
            let body = value
                .get("body")
                .and_then(|v| v.as_str())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'body' field"))?
                .to_string();
            let min_angle = value
                .get("min_angle")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_angle' field")
                })?;
            let max_angle = value.get("max_angle").and_then(|v| v.as_f64());
            let config = BodyProximityConfig {
                body,
                min_angle,
                max_angle,
            };
            Ok(config.to_evaluator())
        }
        "and" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for AND")
                })?;
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(AndEvaluator {
                constraints: evaluators?,
            }))
        }
        "or" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for OR")
                })?;
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(OrEvaluator {
                constraints: evaluators?,
            }))
        }
        "xor" => {
            let constraints = value
                .get("constraints")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'constraints' array for XOR")
                })?;
            if constraints.len() < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "XOR requires at least two sub-constraints",
                ));
            }
            let evaluators: Result<Vec<_>, _> =
                constraints.iter().map(parse_constraint_json).collect();
            Ok(Box::new(XorEvaluator {
                constraints: evaluators?,
            }))
        }
        "not" => {
            let constraint = value.get("constraint").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'constraint' field for NOT")
            })?;
            let evaluator = parse_constraint_json(constraint)?;
            Ok(Box::new(NotEvaluator {
                constraint: evaluator,
            }))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown constraint type: {constraint_type}"
        ))),
    }
}

// Logical combinator evaluators
struct AndEvaluator {
    constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for AndEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Evaluate all constraints
        let results: Vec<_> = self
            .constraints
            .iter()
            .map(|c| {
                c.evaluate(
                    times,
                    target_ra,
                    target_dec,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();

        // Merge violations - a violation exists if ALL constraints are violated at that time
        let mut merged_violations = Vec::new();
        let mut current_violation: Option<(usize, f64, Vec<String>)> = None;

        for (i, time) in times.iter().enumerate() {
            let time_str = time.to_rfc3339();
            let mut all_violated = true;
            let mut min_severity = f64::MAX;
            let mut descriptions = Vec::new();

            // Check if all constraints are violated at this time
            for result in &results {
                let mut this_violated = false;
                for violation in &result.violations {
                    if violation.start_time <= time_str && time_str <= violation.end_time {
                        this_violated = true;
                        min_severity = min_severity.min(violation.max_severity);
                        descriptions.push(&violation.description);
                        break;
                    }
                }
                if !this_violated {
                    all_violated = false;
                    break;
                }
            }

            if all_violated {
                match &mut current_violation {
                    Some((_, sev, descs)) => {
                        *sev = sev.max(min_severity);
                        for desc in descriptions {
                            // Only store string references, clone at the end
                            if !descs.iter().any(|d| d == desc) {
                                descs.push(desc.to_string());
                            }
                        }
                    }
                    None => {
                        current_violation = Some((
                            i,
                            min_severity,
                            descriptions.iter().map(|s| s.to_string()).collect(),
                        ));
                    }
                }
            } else if let Some((start_idx, severity, descs)) = current_violation.take() {
                merged_violations.push(ConstraintViolation {
                    start_time: times[start_idx].to_rfc3339(),
                    end_time: times[i - 1].to_rfc3339(),
                    max_severity: severity,
                    description: format!("AND violation: {}", descs.join("; ")),
                });
            }
        }

        // Close any open violation
        if let Some((start_idx, severity, descs)) = current_violation {
            merged_violations.push(ConstraintViolation {
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity: severity,
                description: format!("AND violation: {}", descs.join("; ")),
            });
        }

        let all_satisfied = merged_violations.is_empty();
        ConstraintResult::new(
            merged_violations,
            all_satisfied,
            self.name(),
            times.to_vec(),
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

struct OrEvaluator {
    constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for OrEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Evaluate all constraints
        let results: Vec<_> = self
            .constraints
            .iter()
            .map(|c| {
                c.evaluate(
                    times,
                    target_ra,
                    target_dec,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();

        // For OR, we violate when ANY constraint is violated
        let mut merged_violations = Vec::new();
        let mut current_violation: Option<(usize, f64, Vec<String>)> = None;

        for (i, time) in times.iter().enumerate() {
            let time_str = time.to_rfc3339();
            let mut any_violated = false;
            let mut max_severity: f64 = 0.0;
            let mut descriptions = Vec::new();

            // Check if any constraint is violated at this time
            for result in &results {
                for violation in &result.violations {
                    if violation.start_time <= time_str && time_str <= violation.end_time {
                        any_violated = true;
                        max_severity = max_severity.max(violation.max_severity);
                        // Avoid cloning by using reference
                        descriptions.push(&violation.description);
                        break;
                    }
                }
            }

            if any_violated {
                match &mut current_violation {
                    Some((_, sev, descs)) => {
                        *sev = sev.max(max_severity);
                        for desc in descriptions {
                            // Only store string references, clone at the end
                            if !descs.iter().any(|d| d == desc) {
                                descs.push(desc.to_string());
                            }
                        }
                    }
                    None => {
                        current_violation = Some((
                            i,
                            max_severity,
                            descriptions.iter().map(|s| s.to_string()).collect(),
                        ));
                    }
                }
            } else if let Some((start_idx, severity, descs)) = current_violation.take() {
                merged_violations.push(ConstraintViolation {
                    start_time: times[start_idx].to_rfc3339(),
                    end_time: times[i - 1].to_rfc3339(),
                    max_severity: severity,
                    description: format!("OR violation: {}", descs.join("; ")),
                });
            }
        }

        // Close any open violation
        if let Some((start_idx, severity, descs)) = current_violation {
            merged_violations.push(ConstraintViolation {
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity: severity,
                description: format!("OR violation: {}", descs.join("; ")),
            });
        }

        let all_satisfied = merged_violations.is_empty();
        ConstraintResult::new(
            merged_violations,
            all_satisfied,
            self.name(),
            times.to_vec(),
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

struct NotEvaluator {
    constraint: Box<dyn ConstraintEvaluator>,
}

impl ConstraintEvaluator for NotEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let result = self.constraint.evaluate(
            times,
            target_ra,
            target_dec,
            sun_positions,
            moon_positions,
            observer_positions,
        );

        // Invert violations - find time periods NOT in violation
        let mut inverted_violations = Vec::new();

        if result.violations.is_empty() {
            // Everything was satisfied, so NOT means everything is violated
            if !times.is_empty() {
                inverted_violations.push(ConstraintViolation {
                    start_time: times[0].to_rfc3339(),
                    end_time: times[times.len() - 1].to_rfc3339(),
                    max_severity: 1.0,
                    description: format!(
                        "NOT({}): inner constraint was satisfied",
                        self.constraint.name()
                    ),
                });
            }
        } else {
            // Find gaps between violations (these become new violations)
            let mut last_end = times[0].to_rfc3339();

            for violation in &result.violations {
                if last_end < violation.start_time {
                    inverted_violations.push(ConstraintViolation {
                        start_time: last_end,
                        end_time: violation.start_time.clone(),
                        max_severity: 0.5,
                        description: format!(
                            "NOT({}): inner constraint was satisfied",
                            self.constraint.name()
                        ),
                    });
                }
                last_end = violation.end_time.clone();
            }

            // Check for gap after last violation
            let final_time = times[times.len() - 1].to_rfc3339();
            if last_end < final_time {
                inverted_violations.push(ConstraintViolation {
                    start_time: last_end,
                    end_time: final_time,
                    max_severity: 0.5,
                    description: format!(
                        "NOT({}): inner constraint was satisfied",
                        self.constraint.name()
                    ),
                });
            }
        }

        let all_satisfied = inverted_violations.is_empty();
        ConstraintResult::new(
            inverted_violations,
            all_satisfied,
            self.name(),
            times.to_vec(),
        )
    }

    fn name(&self) -> String {
        format!("NOT({})", self.constraint.name())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct XorEvaluator {
    constraints: Vec<Box<dyn ConstraintEvaluator>>,
}

impl ConstraintEvaluator for XorEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Evaluate all constraints
        let results: Vec<_> = self
            .constraints
            .iter()
            .map(|c| {
                c.evaluate(
                    times,
                    target_ra,
                    target_dec,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();

        // Violate when EXACTLY ONE sub-constraint is violated
        let mut merged_violations = Vec::new();
        let mut current_violation: Option<(usize, f64, Vec<String>)> = None;

        for (i, time) in times.iter().enumerate() {
            let time_str = time.to_rfc3339();
            let mut active: Vec<&ConstraintViolation> = Vec::new();

            for result in &results {
                for violation in &result.violations {
                    if violation.start_time <= time_str && time_str <= violation.end_time {
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
                    start_time: times[start_idx].to_rfc3339(),
                    end_time: times[i - 1].to_rfc3339(),
                    max_severity: severity,
                    description: format!("XOR violation: {}", descs.join("; ")),
                });
            }
        }

        if let Some((start_idx, severity, descs)) = current_violation {
            merged_violations.push(ConstraintViolation {
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity: severity,
                description: format!("XOR violation: {}", descs.join("; ")),
            });
        }

        let all_satisfied = merged_violations.is_empty();
        ConstraintResult::new(
            merged_violations,
            all_satisfied,
            self.name(),
            times.to_vec(),
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

// Helper to convert serde_json::Value to Py<PyAny>
fn json_to_pyobject(py: Python, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            let py_bool = PyBool::new(py, *b);
            Ok(py_bool.as_any().clone().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let py_int = PyInt::new(py, i);
                Ok(py_int.as_any().clone().unbind())
            } else if let Some(f) = n.as_f64() {
                let py_float = PyFloat::new(py, f);
                Ok(py_float.as_any().clone().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => {
            let py_str = PyString::new(py, s);
            Ok(py_str.as_any().clone().unbind())
        }
        serde_json::Value::Array(arr) => {
            let py_list: Vec<Py<PyAny>> = arr
                .iter()
                .map(|v| json_to_pyobject(py, v))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, py_list)?.as_any().clone().unbind())
        }
        serde_json::Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(py_dict.as_any().clone().unbind())
        }
    }
}
