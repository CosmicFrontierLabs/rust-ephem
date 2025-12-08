/// Python wrapper for constraint system
///
/// This module provides the Python API for constraint evaluation,
/// including JSON-based configuration and convenient factory methods.
use crate::constraints::airmass::AirmassConfig;
use crate::constraints::alt_az::AltAzConfig;
use crate::constraints::body_proximity::{BodyProximityConfig, BodyProximityEvaluator};
use crate::constraints::core::*;
use crate::constraints::daytime::{DaytimeConfig, TwilightType};
use crate::constraints::earth_limb::EarthLimbConfig;
use crate::constraints::eclipse::EclipseConfig;
use crate::constraints::moon_phase::MoonPhaseConfig;
use crate::constraints::moon_proximity::MoonProximityConfig;
use crate::constraints::saa::SAAConfig;
use crate::constraints::sun_proximity::SunProximityConfig;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods};
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

    /// Create a Daytime constraint
    ///
    /// This constraint prevents observations during daytime hours.
    ///
    /// Args:
    ///     twilight (str, optional): Twilight definition - "civil", "nautical", "astronomical", or "none" (default: "civil")
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Twilight definitions:
    ///     - "civil": Civil twilight (-6° below horizon)
    ///     - "nautical": Nautical twilight (-12° below horizon)
    ///     - "astronomical": Astronomical twilight (-18° below horizon)
    ///     - "none": Strict daytime only (Sun above horizon)
    #[pyo3(signature=(twilight="civil"))]
    #[staticmethod]
    fn daytime(twilight: &str) -> PyResult<Self> {
        let twilight_type = match twilight.to_lowercase().as_str() {
            "civil" => TwilightType::Civil,
            "nautical" => TwilightType::Nautical,
            "astronomical" => TwilightType::Astronomical,
            "none" => TwilightType::None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "twilight must be one of: 'civil', 'nautical', 'astronomical', 'none'",
                ));
            }
        };

        let config = DaytimeConfig {
            twilight: twilight_type,
        };

        let twilight_str = match config.twilight {
            TwilightType::Civil => "civil",
            TwilightType::Nautical => "nautical",
            TwilightType::Astronomical => "astronomical",
            TwilightType::None => "none",
        };

        let config_json = serde_json::json!({
            "type": "daytime",
            "twilight": twilight_str
        })
        .to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create an Airmass constraint
    ///
    /// Args:
    ///     max_airmass (float): Maximum allowed airmass (lower = better observing conditions)
    ///     min_airmass (float, optional): Minimum allowed airmass (for excluding very high targets)
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Airmass represents the optical path length through the atmosphere:
    /// - Airmass = 1 at zenith (best conditions)
    /// - Airmass = 2 at 30° altitude
    /// - Airmass = 3 at ~19° altitude
    /// - Higher airmass = worse observing conditions
    #[pyo3(signature=(max_airmass, min_airmass=None))]
    #[staticmethod]
    fn airmass(max_airmass: f64, min_airmass: Option<f64>) -> PyResult<Self> {
        if max_airmass <= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_airmass must be greater than 1.0",
            ));
        }

        if let Some(min) = min_airmass {
            if min <= 1.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_airmass must be greater than 1.0",
                ));
            }
            if min >= max_airmass {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_airmass must be less than max_airmass",
                ));
            }
        }

        let config = AirmassConfig {
            max_airmass,
            min_airmass,
        };

        let mut json_obj = serde_json::json!({
            "type": "airmass",
            "max_airmass": max_airmass
        });
        if let Some(min) = min_airmass {
            json_obj["min_airmass"] = serde_json::json!(min);
        }
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create a Moon phase constraint
    ///
    /// Args:
    ///     max_illumination (float): Maximum allowed Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
    ///     min_illumination (float, optional): Minimum allowed Moon illumination fraction
    ///     min_distance (float, optional): Minimum allowed Moon distance in degrees from target
    ///     max_distance (float, optional): Maximum allowed Moon distance in degrees from target
    ///     enforce_when_below_horizon (bool, optional): Whether to enforce constraint when Moon is below horizon (default: false)
    ///     moon_visibility (str, optional): Moon visibility requirement - "full" or "partial" (default: "full")
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Moon illumination affects observing conditions:
    /// - 0.0: New moon (dark, best for deep observations)
    /// - 0.5: Quarter moon
    /// - 1.0: Full moon (bright, worst for deep observations)
    ///
    /// Moon visibility options:
    /// - "full": Only enforce when Moon is fully above horizon
    /// - "partial": Enforce when any part of Moon is visible above horizon
    #[pyo3(signature=(max_illumination, min_illumination=None, min_distance=None, max_distance=None, enforce_when_below_horizon=false, moon_visibility="full"))]
    #[staticmethod]
    fn moon_phase(
        max_illumination: f64,
        min_illumination: Option<f64>,
        min_distance: Option<f64>,
        max_distance: Option<f64>,
        enforce_when_below_horizon: bool,
        moon_visibility: &str,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&max_illumination) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_illumination must be between 0.0 and 1.0",
            ));
        }

        if let Some(min) = min_illumination {
            if !(0.0..=1.0).contains(&min) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_illumination must be between 0.0 and 1.0",
                ));
            }
            if min >= max_illumination {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_illumination must be less than max_illumination",
                ));
            }
        }

        if let Some(min_dist) = min_distance {
            if min_dist < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_distance must be non-negative",
                ));
            }
        }

        if let Some(max_dist) = max_distance {
            if max_dist < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_distance must be non-negative",
                ));
            }
            if let Some(min_dist) = min_distance {
                if min_dist >= max_dist {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "min_distance must be less than max_distance",
                    ));
                }
            }
        }

        if moon_visibility != "full" && moon_visibility != "partial" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "moon_visibility must be 'full' or 'partial'",
            ));
        }

        let config = MoonPhaseConfig {
            max_illumination,
            min_illumination,
            min_distance,
            max_distance,
            enforce_when_below_horizon,
            moon_visibility: moon_visibility.to_string(),
        };

        let mut json_obj = serde_json::json!({
            "type": "moon_phase",
            "max_illumination": max_illumination,
            "enforce_when_below_horizon": enforce_when_below_horizon,
            "moon_visibility": moon_visibility
        });
        if let Some(min) = min_illumination {
            json_obj["min_illumination"] = serde_json::json!(min);
        }
        if let Some(min_dist) = min_distance {
            json_obj["min_distance"] = serde_json::json!(min_dist);
        }
        if let Some(max_dist) = max_distance {
            json_obj["max_distance"] = serde_json::json!(max_dist);
        }
        let config_json = json_obj.to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create a South Atlantic Anomaly constraint
    ///
    /// The South Atlantic Anomaly is a region of reduced magnetic field strength
    /// that increases radiation exposure for satellites.
    ///
    /// Args:
    ///     polygon (list of tuples): List of (longitude, latitude) pairs defining the SAA region boundary
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(polygon))]
    #[staticmethod]
    fn saa(polygon: Vec<(f64, f64)>) -> PyResult<Self> {
        if polygon.len() < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Polygon must have at least 3 vertices",
            ));
        }

        let config = SAAConfig {
            polygon: polygon.clone(),
        };
        let config_json = serde_json::json!({
            "type": "saa",
            "polygon": polygon
        })
        .to_string();

        Ok(PyConstraint {
            evaluator: config.to_evaluator(),
            config_json,
        })
    }

    /// Create an Altitude/Azimuth constraint
    ///
    /// Args:
    ///     min_altitude (float): Minimum allowed altitude in degrees (0 = horizon, 90 = zenith)
    ///     max_altitude (float, optional): Maximum allowed altitude in degrees
    ///     min_azimuth (float, optional): Minimum allowed azimuth in degrees (0 = North, 90 = East)
    ///     max_azimuth (float, optional): Maximum allowed azimuth in degrees
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Altitude and azimuth define the target's position in the sky:
    /// - Altitude: Angular distance from horizon (0° = horizon, 90° = zenith)
    /// - Azimuth: Angular distance from North, measured eastward (0° = North, 90° = East, etc.)
    ///
    /// For azimuth ranges that cross North (e.g., 330° to 30°), specify min_azimuth > max_azimuth.
    #[pyo3(signature=(min_altitude, max_altitude=None, min_azimuth=None, max_azimuth=None))]
    #[staticmethod]
    fn alt_az(
        min_altitude: f64,
        max_altitude: Option<f64>,
        min_azimuth: Option<f64>,
        max_azimuth: Option<f64>,
    ) -> PyResult<Self> {
        if !(0.0..=90.0).contains(&min_altitude) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_altitude must be between 0 and 90 degrees",
            ));
        }

        if let Some(max_alt) = max_altitude {
            if !(0.0..=90.0).contains(&max_alt) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_altitude must be between 0 and 90 degrees",
                ));
            }
            if max_alt <= min_altitude {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_altitude must be greater than min_altitude",
                ));
            }
        }

        if let Some(min_az) = min_azimuth {
            if !(0.0..=360.0).contains(&min_az) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_azimuth must be between 0 and 360 degrees",
                ));
            }
        }

        if let Some(max_az) = max_azimuth {
            if !(0.0..=360.0).contains(&max_az) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_azimuth must be between 0 and 360 degrees",
                ));
            }
        }

        let config = AltAzConfig {
            min_altitude,
            max_altitude,
            min_azimuth,
            max_azimuth,
        };

        let mut json_obj = serde_json::json!({
            "type": "alt_az",
            "min_altitude": min_altitude
        });
        if let Some(max_alt) = max_altitude {
            json_obj["max_altitude"] = serde_json::json!(max_alt);
        }
        if let Some(min_az) = min_azimuth {
            json_obj["min_azimuth"] = serde_json::json!(min_az);
        }
        if let Some(max_az) = max_azimuth {
            json_obj["max_azimuth"] = serde_json::json!(max_az);
        }
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

    /// Check if targets are in-constraint for multiple RA/Dec positions (vectorized)
    ///
    /// This method efficiently evaluates the constraint for many target positions
    /// at once, returning a 2D boolean array where rows correspond to targets
    /// and columns correspond to times.
    ///
    /// Args:
    ///     ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
    ///     target_ras (array-like): Array of right ascensions in degrees (ICRS/J2000)
    ///     target_decs (array-like): Array of declinations in degrees (ICRS/J2000)
    ///     times (datetime or list[datetime], optional): Specific times to evaluate
    ///     indices (int or list[int], optional): Specific time index/indices to evaluate
    ///
    /// Returns:
    ///     numpy.ndarray: 2D boolean array of shape (n_targets, n_times) where True
    ///                    indicates the constraint is violated for that target at that time
    ///
    /// Example:
    ///     >>> ras = [10.0, 20.0, 30.0]  # Three targets
    ///     >>> decs = [45.0, -10.0, 60.0]
    ///     >>> violations = constraint.in_constraint_batch(ephem, ras, decs)
    ///     >>> violations.shape  # (3, n_times)
    ///     >>> violations[0, :]  # Violations for first target across all times
    #[pyo3(signature = (ephemeris, target_ras, target_decs, times=None, indices=None))]
    fn in_constraint_batch(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        target_ras: Vec<f64>,
        target_decs: Vec<f64>,
        times: Option<&Bound<PyAny>>,
        indices: Option<&Bound<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

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

        // Get ephemeris data based on type
        let (mut times_vec, mut sun_positions, mut moon_positions, mut observer_positions) =
            if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
                (
                    ephem.get_times()?,
                    ephem.get_sun_positions()?,
                    ephem.get_moon_positions()?,
                    ephem.get_gcrs_positions()?,
                )
            } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
                (
                    ephem.get_times()?,
                    ephem.get_sun_positions()?,
                    ephem.get_moon_positions()?,
                    ephem.get_gcrs_positions()?,
                )
            } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
                (
                    ephem.get_times()?,
                    ephem.get_sun_positions()?,
                    ephem.get_moon_positions()?,
                    ephem.get_gcrs_positions()?,
                )
            } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
                (
                    ephem.get_times()?,
                    ephem.get_sun_positions()?,
                    ephem.get_moon_positions()?,
                    ephem.get_gcrs_positions()?,
                )
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
                ));
            };

        // Filter by time indices if provided
        if let Some(ref indices) = time_indices {
            // Validate indices
            let n_times = times_vec.len();
            for &idx in indices {
                if idx >= n_times {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Index {idx} out of range for ephemeris with {n_times} timestamps"
                    )));
                }
            }

            // Filter times and positions
            times_vec = indices.iter().map(|&i| times_vec[i]).collect();
            sun_positions =
                Array2::from_shape_fn((indices.len(), 3), |(i, j)| sun_positions[[indices[i], j]]);
            moon_positions =
                Array2::from_shape_fn((indices.len(), 3), |(i, j)| moon_positions[[indices[i], j]]);
            observer_positions = Array2::from_shape_fn((indices.len(), 3), |(i, j)| {
                observer_positions[[indices[i], j]]
            });
        }

        // Call batch evaluation
        let result_array = self.evaluator.in_constraint_batch(
            &times_vec,
            &target_ras,
            &target_decs,
            &sun_positions,
            &moon_positions,
            &observer_positions,
        )?;

        // Convert to numpy array
        use numpy::IntoPyArray;
        Ok(result_array.into_pyarray(py).into())
    }

    /// Evaluate constraint for multiple RA/Dec positions (vectorized)
    ///
    /// **DEPRECATED:** Use `in_constraint_batch()` instead. This method will be removed
    /// in a future version.
    /// Check if target is in-constraint at given time(s)
    ///
    /// This method evaluates the constraint for a single target position at one or more times.
    /// For multiple times, it efficiently uses the batch evaluation internally.
    ///
    /// Args:
    ///     time (datetime or list[datetime] or numpy.ndarray): Time(s) to check (must exist in ephemeris).
    ///           Can be a single datetime, list of datetimes, or numpy array of datetimes.
    ///     ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
    ///     target_ra (float): Target right ascension in degrees (ICRS/J2000)
    ///     target_dec (float): Target declination in degrees (ICRS/J2000)
    ///
    /// Returns:
    ///     bool or list[bool]: True if constraint is violated at the given time(s).
    ///     Returns a single bool for a single time, or a list of bools for multiple times.
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

        // Parse input times (single datetime or iterable of datetimes)
        let input_times: Vec<DateTime<Utc>> =
            if let Ok(iter) = pyo3::types::PyIterator::from_object(times_arg) {
                // Handle any iterable (list, numpy array, etc.)
                iter.map(|item| {
                    let item = item?;
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
    /// Check if the constraint is satisfied for the given times and target.
    ///
    /// This method wraps `in_constraint_batch` for efficiency when evaluating multiple times
    /// for a single target. If a single time is provided, it returns a boolean. If multiple
    /// times are provided, it returns a list of booleans.
    ///
    /// # Arguments
    /// * `time` - A single time or list of times to evaluate
    /// * `ephemeris` - The ephemeris to use for evaluation
    /// * `target_ra` - Right ascension of the target in degrees
    /// * `target_dec` - Declination of the target in degrees
    ///
    /// # Returns
    /// A boolean if a single time is provided, or a list of booleans if multiple times are provided
    #[pyo3(signature = (time, ephemeris, target_ra, target_dec))]
    fn in_constraint(
        &self,
        py: Python,
        time: Py<PyAny>,
        ephemeris: Py<PyAny>,
        target_ra: f64,
        target_dec: f64,
    ) -> PyResult<Py<PyAny>> {
        // Check if time is a single value or a sequence
        let bound_time = time.bind(py);

        // Try to get the length - if it succeeds, it's a sequence
        let len_result = bound_time.len();
        let is_sequence = len_result.is_ok();
        let num_times = len_result.unwrap_or(1);

        // Repeat target_ra and target_dec for each time
        let target_ras = vec![target_ra; num_times];
        let target_decs = vec![target_dec; num_times];

        // Call the batch method with the time parameter as is
        let result_array = self.in_constraint_batch(
            py,
            ephemeris,
            target_ras,
            target_decs,
            Some(bound_time),
            None,
        )?;

        // Extract the results for the single target (first row)
        let array = result_array.downcast_bound::<PyArray2<bool>>(py)?;
        let readonly_array = array.readonly();
        let array_data = readonly_array.as_array();
        let mut results: Vec<bool> = Vec::with_capacity(num_times);
        for i in 0..num_times {
            results.push(array_data[[0, i]]);
        }

        // Return single bool if single time, else list of bools
        if is_sequence {
            Ok(PyList::new(py, &results)?.as_any().clone().unbind())
        } else {
            Ok(PyBool::new(py, results[0]).as_any().clone().unbind())
        }
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
        "daytime" => {
            let twilight = value
                .get("twilight")
                .and_then(|v| v.as_str())
                .unwrap_or("civil");
            let twilight_type = match twilight {
                "civil" => TwilightType::Civil,
                "nautical" => TwilightType::Nautical,
                "astronomical" => TwilightType::Astronomical,
                "none" => TwilightType::None,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown twilight type: {twilight}"
                    )))
                }
            };
            let config = DaytimeConfig {
                twilight: twilight_type,
            };
            Ok(config.to_evaluator())
        }
        "airmass" => {
            let max_airmass = value
                .get("max_airmass")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'max_airmass' field")
                })?;
            let min_airmass = value.get("min_airmass").and_then(|v| v.as_f64());
            let config = AirmassConfig {
                min_airmass,
                max_airmass,
            };
            Ok(config.to_evaluator())
        }
        "moon_phase" => {
            let max_illumination = value
                .get("max_illumination")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'max_illumination' field")
                })?;
            let min_illumination = value.get("min_illumination").and_then(|v| v.as_f64());
            let min_distance = value.get("min_distance").and_then(|v| v.as_f64());
            let max_distance = value.get("max_distance").and_then(|v| v.as_f64());
            let enforce_when_below_horizon = value
                .get("enforce_when_below_horizon")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let moon_visibility = value
                .get("moon_visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("full")
                .to_string();
            let config = MoonPhaseConfig {
                min_illumination,
                max_illumination,
                min_distance,
                max_distance,
                enforce_when_below_horizon,
                moon_visibility,
            };
            Ok(config.to_evaluator())
        }
        "saa" => {
            let polygon = value
                .get("polygon")
                .and_then(|v| v.as_array())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'polygon' field"))?
                .iter()
                .map(|point| {
                    let arr = point.as_array().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Polygon points must be arrays")
                    })?;
                    if arr.len() != 2 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Polygon points must be [lon, lat] pairs",
                        ));
                    }
                    let lon = arr[0].as_f64().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Longitude must be a number")
                    })?;
                    let lat = arr[1].as_f64().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Latitude must be a number")
                    })?;
                    Ok((lon, lat))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let config = SAAConfig { polygon };
            Ok(config.to_evaluator())
        }
        "alt_az" => {
            let min_altitude = value
                .get("min_altitude")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'min_altitude' field")
                })?;
            let max_altitude = value.get("max_altitude").and_then(|v| v.as_f64());
            let min_azimuth = value.get("min_azimuth").and_then(|v| v.as_f64());
            let max_azimuth = value.get("max_azimuth").and_then(|v| v.as_f64());
            let config = AltAzConfig {
                min_altitude,
                max_altitude,
                min_azimuth,
                max_azimuth,
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

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch(
                    times,
                    target_ras,
                    target_decs,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();
        let results = results?;

        let n_targets = target_ras.len();
        let n_times = times.len();
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

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch(
                    times,
                    target_ras,
                    target_decs,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();
        let results = results?;

        let n_targets = target_ras.len();
        let n_times = times.len();
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

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        // Evaluate sub-constraint in batch
        let sub_result = self.constraint.in_constraint_batch(
            times,
            target_ras,
            target_decs,
            sun_positions,
            moon_positions,
            observer_positions,
        )?;

        let n_targets = target_ras.len();
        let n_times = times.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        // NOT logic: invert all values
        for i in 0..n_targets {
            for j in 0..n_times {
                result[[i, j]] = !sub_result[[i, j]];
            }
        }

        Ok(result)
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

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        sun_positions: &Array2<f64>,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        // Evaluate all sub-constraints in batch
        let results: Result<Vec<_>, _> = self
            .constraints
            .iter()
            .map(|c| {
                c.in_constraint_batch(
                    times,
                    target_ras,
                    target_decs,
                    sun_positions,
                    moon_positions,
                    observer_positions,
                )
            })
            .collect();
        let results = results?;

        let n_targets = target_ras.len();
        let n_times = times.len();
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
