/// Python wrapper for constraint system
///
/// This module provides the Python API for constraint evaluation,
/// including JSON-based configuration and convenient factory methods.
use crate::constraints::airmass::AirmassConfig;
use crate::constraints::alt_az::AltAzConfig;
use crate::constraints::body_proximity::BodyProximityConfig;
use crate::constraints::core::*;
use crate::constraints::daytime::{DaytimeConfig, TwilightType};
use crate::constraints::earth_limb::EarthLimbConfig;
use crate::constraints::eclipse::EclipseConfig;
use crate::constraints::moon_phase::MoonPhaseConfig;
use crate::constraints::moon_proximity::MoonProximityConfig;
use crate::constraints::orbit_pole::OrbitPoleConfig;
use crate::constraints::orbit_ram::OrbitRamConfig;
use crate::constraints::saa::SAAConfig;
use crate::constraints::sun_proximity::SunProximityConfig;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::ephemeris::GroundEphemeris;
use crate::ephemeris::OEMEphemeris;
use crate::ephemeris::SPICEEphemeris;
use crate::ephemeris::TLEEphemeris;
use chrono::{DateTime, Utc};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList};
use std::f64::consts::PI;

use super::json_parser::parse_constraint_json;
use super::json_to_py::json_to_pyobject;

/// Python-facing constraint evaluator
///
/// This wraps the Rust constraint system and provides a convenient Python API.
#[pyclass(name = "Constraint")]
pub struct PyConstraint {
    evaluator: Box<dyn ConstraintEvaluator>,
    config_json: String,
}

impl PyConstraint {
    fn with_effective_evaluator<T, F>(&self, target_roll: Option<f64>, f: F) -> PyResult<T>
    where
        F: FnOnce(&dyn ConstraintEvaluator) -> PyResult<T>,
    {
        let Some(target_roll_deg) = target_roll else {
            return f(&*self.evaluator);
        };

        if !target_roll_deg.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_roll must be a finite number",
            ));
        }

        let mut config: serde_json::Value = serde_json::from_str(&self.config_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let is_boresight_offset = config
            .get("type")
            .and_then(|v| v.as_str())
            .map(|t| t == "boresight_offset")
            .unwrap_or(false);

        if is_boresight_offset {
            let base_roll_deg = config
                .get("roll_deg")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let base_clockwise = config
                .get("roll_clockwise")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Match RustConstraintMixin semantics: add evaluation-time roll in the
            // configured command convention.
            let signed_target_roll = if base_clockwise {
                -target_roll_deg
            } else {
                target_roll_deg
            };

            if let Some(obj) = config.as_object_mut() {
                obj.insert(
                    "roll_deg".to_string(),
                    serde_json::json!(base_roll_deg + signed_target_roll),
                );
            }
        } else {
            config = serde_json::json!({
                "type": "boresight_offset",
                "constraint": config,
                "roll_deg": target_roll_deg,
                "roll_clockwise": false,
                "roll_reference": "sun",
                "pitch_deg": 0.0,
                "yaw_deg": 0.0
            });
        }

        let evaluator = parse_constraint_json(&config)?;
        f(&*evaluator)
    }

    /// Internal helper to evaluate against any Ephemeris implementing EphemerisBase
    #[allow(deprecated)]
    fn eval_with_ephemeris<E: EphemerisBase>(
        &self,
        evaluator: &dyn ConstraintEvaluator,
        ephemeris: &E,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<Vec<usize>>,
    ) -> PyResult<ConstraintResult> {
        // PERFORMANCE OPTIMIZATION: Use fast batch path internally
        // Instead of the slow evaluate() that tracks violations step-by-step,
        // use in_constraint_batch() which is 1700x faster, then construct violations from the result

        // Call the fast batch evaluation for single target
        let violation_array = evaluator.in_constraint_batch(
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices.as_deref(),
        )?;

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
        let violations = crate::constraints::core::track_violations(
            &times,
            |i| (violated[i], if violated[i] { 1.0 } else { 0.0 }),
            |_i, _is_open| evaluator.name(),
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            evaluator.name(),
            times,
        ))
    }

    /// Vectorized evaluation for moving bodies - evaluates all targets at their corresponding times
    ///
    /// For N targets at N times, this calls in_constraint_batch once with all N targets
    /// Uses the efficient diagonal batch evaluation for moving bodies.
    /// Each target_i is evaluated only at time_i, which is O(N) instead of O(N²).
    fn eval_moving_body_batch_diagonal(
        &self,
        py: Python,
        ephemeris: &Py<PyAny>,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        let n = target_ras.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let bound = ephemeris.bind(py);

        // Use the efficient diagonal batch evaluation
        if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            self.evaluator.in_constraint_batch_diagonal(
                &*ephem as &dyn EphemerisBase,
                target_ras,
                target_decs,
            )
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
            ))
        }
    }
}

fn fibonacci_sphere_radec(n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ras = Vec::with_capacity(n_points);
    let mut decs = Vec::with_capacity(n_points);

    if n_points == 0 {
        return (ras, decs);
    }

    // Golden angle in radians
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let n = n_points as f64;

    for i in 0..n_points {
        let k = i as f64 + 0.5;
        let z = 1.0 - (2.0 * k) / n;
        let radius_xy = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;

        let x = radius_xy * phi.cos();
        let y = radius_xy * phi.sin();

        let mut ra_deg = y.atan2(x).to_degrees();
        if ra_deg < 0.0 {
            ra_deg += 360.0;
        }
        let dec_deg = z.clamp(-1.0, 1.0).asin().to_degrees();

        ras.push(ra_deg);
        decs.push(dec_deg);
    }

    (ras, decs)
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

    /// Create an Orbit RAM direction constraint
    ///
    /// Ensures target maintains minimum angular separation from the spacecraft's
    /// velocity vector (RAM direction). Useful for instruments that need to sample
    /// the atmosphere or for thermal management.
    ///
    /// Args:
    ///     min_angle (float): Minimum allowed angular separation from RAM direction in degrees
    ///     max_angle (float, optional): Maximum allowed angular separation from RAM direction in degrees
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(min_angle, max_angle=None))]
    #[staticmethod]
    fn orbit_ram(min_angle: f64, max_angle: Option<f64>) -> PyResult<Self> {
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

        let config = OrbitRamConfig {
            min_angle,
            max_angle,
        };
        let mut json_obj = serde_json::json!({
            "type": "orbit_ram",
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

    /// Create an Orbit pole direction constraint
    ///
    /// Ensures target maintains minimum angular separation from both the north and south
    /// orbital poles (directions perpendicular to the orbital plane). Useful for maintaining
    /// specific orientations relative to the spacecraft's orbit.
    ///
    /// Args:
    ///     min_angle (float): Minimum allowed angular separation from both orbital poles in degrees
    ///     max_angle (float, optional): Maximum allowed angular separation from both orbital poles in degrees
    ///     earth_limb_pole (bool, optional): If True, pole avoidance angle is earth_radius_deg + min_angle - 90.
    ///                                       Used for NASA's Neil Gehrels Swift Observatory.
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    #[pyo3(signature=(min_angle, max_angle=None, earth_limb_pole=false))]
    #[staticmethod]
    fn orbit_pole(
        min_angle: f64,
        max_angle: Option<f64>,
        earth_limb_pole: Option<bool>,
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

        let earth_limb_pole = earth_limb_pole.unwrap_or(false);

        let config = OrbitPoleConfig {
            min_angle,
            max_angle,
            earth_limb_pole,
        };
        let mut json_obj = serde_json::json!({
            "type": "orbit_pole",
            "min_angle": min_angle,
            "earth_limb_pole": earth_limb_pole
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

    /// Create an Altitude/Azimuth constraint
    ///
    /// Args:
    ///     min_altitude (float, optional): Minimum allowed altitude in degrees (0 = horizon, 90 = zenith)
    ///     max_altitude (float, optional): Maximum allowed altitude in degrees
    ///     min_azimuth (float, optional): Minimum allowed azimuth in degrees (0 = North, 90 = East)
    ///     max_azimuth (float, optional): Maximum allowed azimuth in degrees
    ///     polygon (list of tuples, optional): List of (altitude, azimuth) pairs defining allowed region
    ///
    /// Returns:
    ///     Constraint: A new constraint object
    ///
    /// Altitude and azimuth define the target's position in the sky:
    /// - Altitude: Angular distance from horizon (0° = horizon, 90° = zenith)
    /// - Azimuth: Angular distance from North, measured eastward (0° = North, 90° = East, etc.)
    ///
    /// For azimuth ranges that cross North (e.g., 330° to 30°), specify min_azimuth > max_azimuth.
    /// If polygon is provided, the target must be inside this polygon to satisfy the constraint.
    #[pyo3(signature=(min_altitude=None, max_altitude=None, min_azimuth=None, max_azimuth=None, polygon=None))]
    #[staticmethod]
    fn alt_az(
        min_altitude: Option<f64>,
        max_altitude: Option<f64>,
        min_azimuth: Option<f64>,
        max_azimuth: Option<f64>,
        polygon: Option<Vec<(f64, f64)>>,
    ) -> PyResult<Self> {
        if let Some(min_alt) = min_altitude {
            if !(0.0..=90.0).contains(&min_alt) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "min_altitude must be between 0 and 90 degrees",
                ));
            }
        }

        if let Some(max_alt) = max_altitude {
            if !(0.0..=90.0).contains(&max_alt) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_altitude must be between 0 and 90 degrees",
                ));
            }
        }

        if let (Some(min_alt), Some(max_alt)) = (min_altitude, max_altitude) {
            if max_alt <= min_alt {
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

        // Validate polygon if provided
        if let Some(ref poly) = polygon {
            if poly.len() < 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "polygon must have at least 3 vertices",
                ));
            }
        }

        let config = AltAzConfig {
            min_altitude,
            max_altitude,
            min_azimuth,
            max_azimuth,
            polygon: polygon.clone(),
        };

        let mut json_obj = serde_json::json!({
            "type": "alt_az"
        });
        if let Some(min_alt) = min_altitude {
            json_obj["min_altitude"] = serde_json::json!(min_alt);
        }
        if let Some(max_alt) = max_altitude {
            json_obj["max_altitude"] = serde_json::json!(max_alt);
        }
        if let Some(min_az) = min_azimuth {
            json_obj["min_azimuth"] = serde_json::json!(min_az);
        }
        if let Some(max_az) = max_azimuth {
            json_obj["max_azimuth"] = serde_json::json!(max_az);
        }
        if let Some(poly) = polygon {
            json_obj["polygon"] = serde_json::json!(poly);
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
    ///     {"type": "boresight_offset", "constraint": {...}, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 1.5}
    ///     {"type": "and", "constraints": [...]}
    ///     {"type": "or", "constraints": [...]}
    ///     {"type": "xor", "constraints": [...]}  // exactly one violated -> violation
    ///     {"type": "at_least", "min_violated": 2, "constraints": [...]}  // k-of-n violated -> violation
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

    /// Combine this constraint with others using logical AND (instance method)
    ///
    /// Args:
    ///     *constraints: Variable number of Constraint objects to combine with self
    ///
    /// Returns:
    ///     Constraint: A new constraint that is violated only if ALL input constraints are violated
    ///
    /// Example:
    ///     >>> combined = sun_constraint.combine_and(moon_constraint, saa_constraint)
    #[pyo3(signature = (*constraints))]
    fn combine_and(&self, constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        // Start with self's config
        let mut configs: Vec<serde_json::Value> =
            vec![serde_json::from_str(&self.config_json).unwrap()];

        // Add all other constraints
        configs.extend(
            constraints
                .iter()
                .map(|c| serde_json::from_str(&c.config_json).unwrap()),
        );

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

    /// Combine this constraint with others using logical OR (instance method)
    ///
    /// Args:
    ///     *constraints: Variable number of Constraint objects to combine with self
    ///
    /// Returns:
    ///     Constraint: A new constraint that is violated if ANY input constraint is violated
    ///
    /// Example:
    ///     >>> combined = sun_constraint.combine_or(moon_constraint, saa_constraint)
    #[pyo3(signature = (*constraints))]
    fn combine_or(&self, constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        // Start with self's config
        let mut configs: Vec<serde_json::Value> =
            vec![serde_json::from_str(&self.config_json).unwrap()];

        // Add all other constraints
        configs.extend(
            constraints
                .iter()
                .map(|c| serde_json::from_str(&c.config_json).unwrap()),
        );

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

    /// Combine constraints with threshold logic (k-of-n)
    ///
    /// Args:
    ///     min_violated: Minimum number of sub-constraints that must be violated
    ///                   for this combined constraint to be violated (k)
    ///     constraints: List of sub-constraints (n)
    ///
    /// Returns:
    ///     Constraint: A new constraint violated when at least `min_violated` sub-constraints
    ///         are violated
    #[staticmethod]
    fn at_least(min_violated: usize, constraints: Vec<PyRef<PyConstraint>>) -> PyResult<Self> {
        if constraints.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one constraint required for at_least",
            ));
        }
        if min_violated == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_violated must be at least 1",
            ));
        }
        if min_violated > constraints.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "min_violated ({min_violated}) cannot exceed number of constraints ({})",
                constraints.len()
            )));
        }

        let configs: Vec<serde_json::Value> = constraints
            .iter()
            .map(|c| serde_json::from_str(&c.config_json).unwrap())
            .collect();

        let config_json = serde_json::json!({
            "type": "at_least",
            "min_violated": min_violated,
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

    /// Apply a fixed boresight offset to a constraint using Euler angles
    ///
    /// This wraps an existing constraint and evaluates it at a rotated target direction,
    /// enabling shared-axis telescope planning where secondary instruments are offset
    /// from the primary boresight.
    ///
    /// Args:
    ///     constraint (Constraint): Inner constraint to evaluate at offset direction
    ///     roll_deg (float): Fixed boresight roll offset about +X in degrees (default 0.0)
    ///     roll_clockwise (bool): If True, positive fixed boresight roll is clockwise
    ///         when looking along +X. If False, positive is counterclockwise.
    ///     roll_reference (str): Roll-zero reference axis: "sun" (default) or "north".
    ///     pitch_deg (float): Pitch angle about +Y in degrees
    ///     yaw_deg (float): Yaw angle about +Z in degrees
    ///
    /// Returns:
    ///     Constraint: A new boresight-offset wrapped constraint
    #[staticmethod]
    #[pyo3(signature = (constraint, roll_deg=0.0, roll_clockwise=false, roll_reference="sun", pitch_deg=0.0, yaw_deg=0.0))]
    fn boresight_offset(
        constraint: PyRef<PyConstraint>,
        roll_deg: f64,
        roll_clockwise: bool,
        roll_reference: &str,
        pitch_deg: f64,
        yaw_deg: f64,
    ) -> PyResult<Self> {
        if !roll_deg.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roll_deg must be a finite number",
            ));
        }
        if !pitch_deg.is_finite() || !yaw_deg.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pitch_deg and yaw_deg must be finite numbers",
            ));
        }

        let roll_reference_normalized = roll_reference.to_ascii_lowercase();
        if roll_reference_normalized != "sun" && roll_reference_normalized != "north" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roll_reference must be either 'sun' or 'north'",
            ));
        }

        let config: serde_json::Value = serde_json::from_str(&constraint.config_json).unwrap();

        let config_json = serde_json::json!({
            "type": "boresight_offset",
            "constraint": config,
            "roll_deg": roll_deg,
            "roll_clockwise": roll_clockwise,
            "roll_reference": roll_reference_normalized,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ephemeris, target_ra, target_dec, times=None, indices=None, target_roll=None))]
    fn evaluate(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        target_ra: f64,
        target_dec: f64,
        times: Option<&Bound<PyAny>>,
        indices: Option<&Bound<PyAny>>,
        target_roll: Option<f64>,
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

        self.with_effective_evaluator(target_roll, |evaluator| {
            if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
                return self.eval_with_ephemeris(
                    evaluator,
                    &*ephem,
                    target_ra,
                    target_dec,
                    time_indices.clone(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
                return self.eval_with_ephemeris(
                    evaluator,
                    &*ephem,
                    target_ra,
                    target_dec,
                    time_indices.clone(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
                return self.eval_with_ephemeris(
                    evaluator,
                    &*ephem,
                    target_ra,
                    target_dec,
                    time_indices.clone(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
                return self.eval_with_ephemeris(
                    evaluator,
                    &*ephem,
                    target_ra,
                    target_dec,
                    time_indices.clone(),
                );
            }

            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
            ))
        })
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
    ///                    indicates the constraint is VIOLATED (target not allowed) at that time
    ///
    /// Example:
    ///     >>> ras = [10.0, 20.0, 30.0]  # Three targets
    ///     >>> decs = [45.0, -10.0, 60.0]
    ///     >>> violations = constraint.in_constraint_batch(ephem, ras, decs)
    ///     >>> violations.shape  # (3, n_times)
    ///     >>> violations[0, :]  # Violations for first target across all times
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ephemeris, target_ras, target_decs, times=None, indices=None, target_roll=None))]
    fn in_constraint_batch(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        target_ras: Vec<f64>,
        target_decs: Vec<f64>,
        times: Option<&Bound<PyAny>>,
        indices: Option<&Bound<PyAny>>,
        target_roll: Option<f64>,
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

        // Call batch evaluation with ephemeris based on type
        let result_array = self.with_effective_evaluator(target_roll, |evaluator| {
            if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
                return evaluator.in_constraint_batch(
                    &*ephem as &dyn EphemerisBase,
                    &target_ras,
                    &target_decs,
                    time_indices.as_deref(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
                return evaluator.in_constraint_batch(
                    &*ephem as &dyn EphemerisBase,
                    &target_ras,
                    &target_decs,
                    time_indices.as_deref(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
                return evaluator.in_constraint_batch(
                    &*ephem as &dyn EphemerisBase,
                    &target_ras,
                    &target_decs,
                    time_indices.as_deref(),
                );
            }
            if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
                return evaluator.in_constraint_batch(
                    &*ephem as &dyn EphemerisBase,
                    &target_ras,
                    &target_decs,
                    time_indices.as_deref(),
                );
            }

            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
            ))
        })?;

        // Convert to numpy array
        use numpy::IntoPyArray;
        Ok(result_array.into_pyarray(py).into())
    }

    /// Compute instantaneous field of regard for this constraint in steradians.
    ///
    /// Field of regard is the visible solid angle at a single timestamp, where
    /// visibility is defined by `constraint == False` (not violated).
    ///
    /// Args:
    ///     ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
    ///     time (datetime, optional): Specific timestamp to evaluate (must exist in ephemeris)
    ///     index (int, optional): Specific time index to evaluate
    ///     n_points (int, optional): Number of sky samples (Fibonacci sphere). Default 20000.
    ///
    /// Returns:
    ///     float: Instantaneous field of regard in steradians (range [0, 4π])
    ///
    /// Notes:
    ///     - Exactly one of `time` or `index` must be provided.
    ///     - Higher `n_points` improves accuracy at higher computational cost.
    #[pyo3(signature = (ephemeris, time=None, index=None, n_points=20000))]
    fn instantaneous_field_of_regard(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        time: Option<&Bound<PyAny>>,
        index: Option<usize>,
        n_points: usize,
    ) -> PyResult<f64> {
        if n_points == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_points must be greater than 0",
            ));
        }

        let has_time = time.is_some();
        let has_index = index.is_some();
        if has_time == has_index {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Specify exactly one of 'time' or 'index'",
            ));
        }

        let bound = ephemeris.bind(py);
        let eval_index = if let Some(t) = time {
            let parsed = self.parse_times_to_indices(bound, t)?;
            if parsed.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "'time' must be a single datetime for instantaneous evaluation",
                ));
            }
            parsed[0]
        } else {
            index.expect("index checked above")
        };

        if has_index {
            let n_times = if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
                ephem.get_times()?.len()
            } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
                ephem.get_times()?.len()
            } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
                ephem.get_times()?.len()
            } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
                ephem.get_times()?.len()
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
                ));
            };

            if eval_index >= n_times {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "index {} out of range for ephemeris with {} timestamps",
                    eval_index, n_times
                )));
            }
        }

        let (target_ras, target_decs) = fibonacci_sphere_radec(n_points);

        let violated = if let Ok(ephem) = bound.extract::<PyRef<TLEEphemeris>>() {
            self.evaluator.in_constraint_batch(
                &*ephem as &dyn EphemerisBase,
                &target_ras,
                &target_decs,
                Some(&[eval_index]),
            )?
        } else if let Ok(ephem) = bound.extract::<PyRef<SPICEEphemeris>>() {
            self.evaluator.in_constraint_batch(
                &*ephem as &dyn EphemerisBase,
                &target_ras,
                &target_decs,
                Some(&[eval_index]),
            )?
        } else if let Ok(ephem) = bound.extract::<PyRef<GroundEphemeris>>() {
            self.evaluator.in_constraint_batch(
                &*ephem as &dyn EphemerisBase,
                &target_ras,
                &target_decs,
                Some(&[eval_index]),
            )?
        } else if let Ok(ephem) = bound.extract::<PyRef<OEMEphemeris>>() {
            self.evaluator.in_constraint_batch(
                &*ephem as &dyn EphemerisBase,
                &target_ras,
                &target_decs,
                Some(&[eval_index]),
            )?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported ephemeris type. Expected TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris",
            ));
        };

        let visible_count = (0..n_points).filter(|&i| !violated[[i, 0]]).count();
        let visible_fraction = visible_count as f64 / n_points as f64;

        Ok(4.0 * PI * visible_fraction)
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
    #[pyo3(signature = (time, ephemeris, target_ra, target_dec, target_roll=None))]
    fn in_constraint(
        &self,
        py: Python,
        time: Py<PyAny>,
        ephemeris: Py<PyAny>,
        target_ra: f64,
        target_dec: f64,
        target_roll: Option<f64>,
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
            target_roll,
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

    /// Evaluate constraint for a moving body (varying RA/Dec over time)
    ///
    /// This method evaluates the constraint for a body whose position changes over time,
    /// such as a comet, asteroid, or planet. It returns detailed results including
    /// per-timestamp violation status, visibility windows, and the body's coordinates.
    ///
    /// There are two ways to specify the body's position:
    /// 1. Explicit coordinates: Provide `target_ras`, `target_decs`, and optionally `times`
    /// 2. Body lookup: Provide `body` name/ID and optionally `use_horizons` to query positions
    ///
    /// Args:
    ///     ephemeris: One of TLEEphemeris, SPICEEphemeris, GroundEphemeris, or OEMEphemeris
    ///     target_ras (list[float], optional): Array of right ascensions in degrees (ICRS/J2000)
    ///     target_decs (list[float], optional): Array of declinations in degrees (ICRS/J2000)
    ///     times (datetime or list[datetime], optional): Specific times to evaluate (must match ras/decs length)
    ///     body (str, optional): Body identifier (NAIF ID or name like "Jupiter", "90004910")
    ///     use_horizons (bool): If True, query JPL Horizons for body positions (default: False)
    ///     spice_kernel (str, optional): SPICE kernel specification for body lookup
    ///
    /// Returns:
    ///     MovingBodyResult: Result object containing:
    ///         - timestamps: list of datetime objects
    ///         - ras: list of right ascensions in degrees
    ///         - decs: list of declinations in degrees
    ///         - constraint_array: list of bools (True = violated)
    ///         - visibility_flags: list of bools (True = visible, inverse of constraint_array)
    ///         - visibility: list of visibility window dicts with start_time, end_time, duration_seconds
    ///         - all_satisfied: bool indicating if constraint was never violated
    ///         - constraint_name: string name of the constraint
    ///
    /// Example:
    ///     >>> # Using body name (queries SPICE or Horizons for positions)
    ///     >>> result = constraint.evaluate_moving_body(ephem, body="Jupiter")
    ///     >>> # Using explicit coordinates for a comet
    ///     >>> result = constraint.evaluate_moving_body(ephem, target_ras=ras, target_decs=decs)
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ephemeris, target_ras=None, target_decs=None, times=None, body=None, use_horizons=false, spice_kernel=None))]
    fn evaluate_moving_body(
        &self,
        py: Python,
        ephemeris: Py<PyAny>,
        target_ras: Option<Vec<f64>>,
        target_decs: Option<Vec<f64>>,
        times: Option<&Bound<PyAny>>,
        body: Option<&str>,
        use_horizons: bool,
        spice_kernel: Option<&str>,
    ) -> PyResult<MovingBodyResult> {
        use crate::constraints::core::MovingBodyResult;

        let bound = ephemeris.bind(py);

        // Determine ras, decs, and timestamps based on input mode
        let (ras, decs, timestamps): (Vec<f64>, Vec<f64>, Vec<DateTime<Utc>>) =
            if let Some(body_id) = body {
                // Body lookup mode: get positions from ephemeris.get_body()
                // Build kwargs dict with use_horizons and optional spice_kernel
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("use_horizons", use_horizons)?;
                if let Some(ks) = spice_kernel {
                    kwargs.set_item("spice_kernel", ks)?;
                }
                let skycoord = bound.call_method("get_body", (body_id,), Some(&kwargs))?;

                // Extract RA/Dec from SkyCoord
                let ra_attr = skycoord.getattr("ra")?;
                let dec_attr = skycoord.getattr("dec")?;
                let ra_deg = ra_attr.getattr("deg")?;
                let dec_deg = dec_attr.getattr("deg")?;

                // Convert to Vec<f64>
                let ras: Vec<f64> = ra_deg.extract()?;
                let decs: Vec<f64> = dec_deg.extract()?;

                // Get timestamps from ephemeris
                let ts_attr = bound.getattr("timestamp")?;
                let ts_list: Vec<DateTime<Utc>> = if let Ok(iter) =
                    pyo3::types::PyIterator::from_object(&ts_attr)
                {
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
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err("Invalid datetime")
                            })
                    })
                    .collect::<PyResult<_>>()?
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Could not iterate ephemeris timestamps",
                    ));
                };

                (ras, decs, ts_list)
            } else {
                // Explicit coordinates mode
                let ras = target_ras.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Either 'body' or 'target_ras'/'target_decs' must be provided",
                    )
                })?;
                let decs = target_decs.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "target_decs must be provided when target_ras is specified",
                    )
                })?;

                if ras.len() != decs.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "target_ras and target_decs must have the same length",
                    ));
                }

                // Get timestamps - either from 'times' parameter or from ephemeris
                let ts_list: Vec<DateTime<Utc>> = if let Some(times_arg) = times {
                    // Parse times parameter
                    if let Ok(iter) = pyo3::types::PyIterator::from_object(times_arg) {
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
                                .and_then(|d| {
                                    d.and_hms_micro_opt(hour, minute, second, microsecond)
                                })
                                .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
                                .ok_or_else(|| {
                                    pyo3::exceptions::PyValueError::new_err("Invalid datetime")
                                })
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
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err("Invalid datetime")
                            })?;

                        vec![dt]
                    }
                } else {
                    // Use ephemeris timestamps
                    let ts_attr = bound.getattr("timestamp")?;
                    if let Ok(iter) = pyo3::types::PyIterator::from_object(&ts_attr) {
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
                                .and_then(|d| {
                                    d.and_hms_micro_opt(hour, minute, second, microsecond)
                                })
                                .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
                                .ok_or_else(|| {
                                    pyo3::exceptions::PyValueError::new_err("Invalid datetime")
                                })
                        })
                        .collect::<PyResult<_>>()?
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Could not iterate ephemeris timestamps",
                        ));
                    }
                };

                if ts_list.len() != ras.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "timestamps length must match target_ras/target_decs length",
                    ));
                }

                (ras, decs, ts_list)
            };

        // Evaluate constraint at each timestamp with corresponding RA/Dec
        // VECTORIZED: Use batch evaluation with diagonal extraction for speed
        let constraint_vec = self.eval_moving_body_batch_diagonal(py, &ephemeris, &ras, &decs)?;

        // Build violation windows from constraint_vec
        let violations = track_violations(
            &timestamps,
            |i| (constraint_vec[i], if constraint_vec[i] { 1.0 } else { 0.0 }),
            |_i, _is_open| self.evaluator.name(),
        );

        let all_satisfied = !constraint_vec.iter().any(|&v| v);

        Ok(MovingBodyResult::new(
            violations,
            all_satisfied,
            self.evaluator.name(),
            timestamps,
            ras,
            decs,
            constraint_vec,
        ))
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
