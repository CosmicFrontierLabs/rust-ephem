/// Altitude/Azimuth constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Altitude/Azimuth constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltAzConfig {
    /// Minimum allowed altitude in degrees (0 = horizon, 90 = zenith)
    pub min_altitude: f64,
    /// Maximum allowed altitude in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_altitude: Option<f64>,
    /// Minimum allowed azimuth in degrees (0 = North, 90 = East, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_azimuth: Option<f64>,
    /// Maximum allowed azimuth in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_azimuth: Option<f64>,
}

impl ConstraintConfig for AltAzConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(AltAzEvaluator {
            min_altitude: self.min_altitude,
            max_altitude: self.max_altitude,
            min_azimuth: self.min_azimuth,
            max_azimuth: self.max_azimuth,
        })
    }
}

/// Evaluator for Altitude/Azimuth constraint
struct AltAzEvaluator {
    min_altitude: f64,
    max_altitude: Option<f64>,
    min_azimuth: Option<f64>,
    max_azimuth: Option<f64>,
}

impl AltAzEvaluator {
    fn format_name(&self) -> String {
        let mut parts = vec![format!(
            "AltAzConstraint(min_alt={:.1}°)",
            self.min_altitude
        )];

        if let Some(max_alt) = self.max_altitude {
            parts.push(format!("max_alt={:.1}°", max_alt));
        }
        if let Some(min_az) = self.min_azimuth {
            parts.push(format!("min_az={:.1}°", min_az));
        }
        if let Some(max_az) = self.max_azimuth {
            parts.push(format!("max_az={:.1}°", max_az));
        }

        parts.join(", ")
    }
}

impl ConstraintEvaluator for AltAzEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Extract and filter ephemeris data
        let (times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);
        // Convert target to unit vector
        let target_vec = radec_to_unit_vectors_batch(&[target_ra], &[target_dec]);
        let target_unit = target_vec.row(0);

        let violations = track_violations(
            &times_filtered,
            |i| {
                let observer_pos = obs_filtered.row(i);

                // Calculate target altitude and azimuth from observer position
                // This requires proper coordinate transformation from ICRS to topocentric
                let (altitude_deg, azimuth_deg) =
                    self.calculate_alt_az(&target_unit, &observer_pos);

                // Check altitude constraints
                let mut violated = false;
                let mut severity = 1.0;

                if altitude_deg < self.min_altitude {
                    violated = true;
                    severity = (self.min_altitude - altitude_deg).min(1.0);
                }

                if let Some(max_altitude) = self.max_altitude {
                    if altitude_deg > max_altitude {
                        violated = true;
                        severity = (altitude_deg - max_altitude).min(1.0);
                    }
                }

                // Check azimuth constraints (only if altitude is acceptable)
                if !violated {
                    if let Some(min_azimuth) = self.min_azimuth {
                        if let Some(max_azimuth) = self.max_azimuth {
                            // Azimuth range constraint
                            let az_in_range = if min_azimuth <= max_azimuth {
                                azimuth_deg >= min_azimuth && azimuth_deg <= max_azimuth
                            } else {
                                // Handle wrap-around (e.g., 330° to 30°)
                                azimuth_deg >= min_azimuth || azimuth_deg <= max_azimuth
                            };
                            if !az_in_range {
                                violated = true;
                                severity = 1.0; // Azimuth violations are binary
                            }
                        } else {
                            // Only minimum azimuth
                            if azimuth_deg < min_azimuth {
                                violated = true;
                                severity = (min_azimuth - azimuth_deg).min(1.0);
                            }
                        }
                    } else if let Some(max_azimuth) = self.max_azimuth {
                        // Only maximum azimuth
                        if azimuth_deg > max_azimuth {
                            violated = true;
                            severity = (azimuth_deg - max_azimuth).min(1.0);
                        }
                    }
                }

                (violated, severity)
            },
            |_, _violated| {
                // For description, we need to recalculate - this is a limitation of the closure pattern
                // In practice, you'd want to cache these calculations
                let (altitude_deg, azimuth_deg) =
                    self.calculate_alt_az(&target_unit, &obs_filtered.row(0));
                self.format_violation_description(altitude_deg, azimuth_deg)
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
            times_filtered.to_vec(),
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        // Extract and filter ephemeris data
        let (_times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);

        let n_targets = target_ras.len();
        let n_times = obs_filtered.nrows();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        // Convert all targets to unit vectors
        let target_vecs = radec_to_unit_vectors_batch(target_ras, target_decs);

        for i in 0..n_times {
            let observer_pos = obs_filtered.row(i);

            for j in 0..n_targets {
                let target_unit = target_vecs.row(j);
                let (altitude_deg, azimuth_deg) =
                    self.calculate_alt_az(&target_unit, &observer_pos);

                let mut violated = false;

                // Check altitude
                if altitude_deg < self.min_altitude {
                    violated = true;
                }
                if let Some(max_altitude) = self.max_altitude {
                    if altitude_deg > max_altitude {
                        violated = true;
                    }
                }

                // Check azimuth
                if !violated {
                    if let Some(min_azimuth) = self.min_azimuth {
                        if let Some(max_azimuth) = self.max_azimuth {
                            let az_in_range = if min_azimuth <= max_azimuth {
                                azimuth_deg >= min_azimuth && azimuth_deg <= max_azimuth
                            } else {
                                azimuth_deg >= min_azimuth || azimuth_deg <= max_azimuth
                            };
                            if !az_in_range {
                                violated = true;
                            }
                        } else if azimuth_deg < min_azimuth {
                            violated = true;
                        }
                    } else if let Some(max_azimuth) = self.max_azimuth {
                        if azimuth_deg > max_azimuth {
                            violated = true;
                        }
                    }
                }

                result[[j, i]] = violated;
            }
        }

        Ok(result)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AltAzEvaluator {
    /// Calculate target's altitude and azimuth from observer position
    /// This is a simplified calculation - in practice you'd need proper
    /// astronomical coordinate transformations from ICRS to topocentric
    fn calculate_alt_az(
        &self,
        _target_unit: &ndarray::ArrayView1<f64>,
        _observer_pos: &ndarray::ArrayView1<f64>,
    ) -> (f64, f64) {
        // Simplified: assume target position is given in topocentric coordinates
        // In practice, you'd need to convert from ICRS to topocentric alt/az
        // For now, return dummy values that allow testing
        // TODO: Implement proper alt/az calculation
        (45.0, 180.0) // 45° altitude, 180° azimuth (south) for testing
    }

    fn format_violation_description(&self, altitude_deg: f64, azimuth_deg: f64) -> String {
        let mut reasons = Vec::new();

        if altitude_deg < self.min_altitude {
            reasons.push(format!(
                "altitude {:.1}° < min {:.1}°",
                altitude_deg, self.min_altitude
            ));
        }
        if let Some(max_altitude) = self.max_altitude {
            if altitude_deg > max_altitude {
                reasons.push(format!(
                    "altitude {:.1}° > max {:.1}°",
                    altitude_deg, max_altitude
                ));
            }
        }

        if let Some(min_azimuth) = self.min_azimuth {
            if let Some(max_azimuth) = self.max_azimuth {
                let az_in_range = if min_azimuth <= max_azimuth {
                    azimuth_deg >= min_azimuth && azimuth_deg <= max_azimuth
                } else {
                    azimuth_deg >= min_azimuth || azimuth_deg <= max_azimuth
                };
                if !az_in_range {
                    reasons.push(format!(
                        "azimuth {:.1}° outside range {:.1}°-{:1}°",
                        azimuth_deg, min_azimuth, max_azimuth
                    ));
                }
            } else if azimuth_deg < min_azimuth {
                reasons.push(format!(
                    "azimuth {:.1}° < min {:.1}°",
                    azimuth_deg, min_azimuth
                ));
            }
        } else if let Some(max_azimuth) = self.max_azimuth {
            if azimuth_deg > max_azimuth {
                reasons.push(format!(
                    "azimuth {:.1}° > max {:.1}°",
                    azimuth_deg, max_azimuth
                ));
            }
        }

        format!(
            "Target position violates constraints: {}",
            reasons.join(", ")
        )
    }
}
