/// Altitude/Azimuth constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::polygon;
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
    /// Polygon defining an allowed region in Alt/Az space as (altitude, azimuth) pairs in degrees
    /// If provided, the target must be inside this polygon (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<(f64, f64)>>,
}

impl ConstraintConfig for AltAzConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(AltAzEvaluator {
            min_altitude: self.min_altitude,
            max_altitude: self.max_altitude,
            min_azimuth: self.min_azimuth,
            max_azimuth: self.max_azimuth,
            polygon: self.polygon.clone(),
        })
    }
}

/// Evaluator for Altitude/Azimuth constraint
struct AltAzEvaluator {
    min_altitude: f64,
    max_altitude: Option<f64>,
    min_azimuth: Option<f64>,
    max_azimuth: Option<f64>,
    polygon: Option<Vec<(f64, f64)>>,
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
        if self.polygon.is_some() {
            parts.push("polygon".to_string());
        }

        parts.join(", ")
    }

    /// Check if a point (altitude, azimuth) is inside the polygon using the winding number algorithm
    fn point_in_polygon(&self, altitude: f64, azimuth: f64) -> bool {
        let polygon = match &self.polygon {
            Some(p) => p,
            None => return true, // If no polygon defined, point is always inside
        };

        polygon::point_in_polygon(polygon, altitude, azimuth)
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
        let (times_filtered, _obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);

        // Compute alt/az for this target at selected times
        let altaz = ephemeris.radec_to_altaz(target_ra, target_dec, time_indices);

        let violations = track_violations(
            &times_filtered,
            |i| {
                let altitude_deg = altaz[[i, 0]];
                let azimuth_deg = altaz[[i, 1]];

                // Check polygon constraint first if defined
                if self.polygon.is_some() && !self.point_in_polygon(altitude_deg, azimuth_deg) {
                    return (true, 1.0);
                }

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
                // Use the first timestamp alt/az for the description
                let altitude_deg = altaz[[0, 0]];
                let azimuth_deg = altaz[[0, 1]];
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
        let n_targets = target_ras.len();
        let altaz_list: Vec<_> = target_ras
            .iter()
            .zip(target_decs.iter())
            .map(|(&ra, &dec)| ephemeris.radec_to_altaz(ra, dec, time_indices))
            .collect();

        let n_times = altaz_list.first().map(|a| a.nrows()).unwrap_or(0);

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            for (j, altaz) in altaz_list.iter().enumerate() {
                let altitude_deg = altaz[[i, 0]];
                let azimuth_deg = altaz[[i, 1]];

                let mut violated = false;

                // Check polygon constraint first if defined
                if self.polygon.is_some() && !self.point_in_polygon(altitude_deg, azimuth_deg) {
                    violated = true;
                }

                if !violated {
                    // Check altitude
                    if altitude_deg < self.min_altitude {
                        violated = true;
                    }
                    if let Some(max_altitude) = self.max_altitude {
                        if altitude_deg > max_altitude {
                            violated = true;
                        }
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
    /// Format a description of the violation based on altitude and azimuth
    fn format_violation_description(&self, altitude_deg: f64, azimuth_deg: f64) -> String {
        let mut reasons = Vec::new();

        // Check polygon violation first
        if self.polygon.is_some() && !self.point_in_polygon(altitude_deg, azimuth_deg) {
            reasons.push(format!(
                "outside allowed Alt/Az polygon (alt: {:.1}°, az: {:.1}°)",
                altitude_deg, azimuth_deg
            ));
        }

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
