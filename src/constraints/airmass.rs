/// Airmass constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use chrono::{DateTime, Timelike, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Airmass constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirmassConfig {
    /// Maximum allowed airmass (lower values = better observing conditions)
    pub max_airmass: f64,
    /// Minimum allowed airmass (optional, for excluding very high targets)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_airmass: Option<f64>,
}

impl ConstraintConfig for AirmassConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(AirmassEvaluator {
            max_airmass: self.max_airmass,
            min_airmass: self.min_airmass,
        })
    }
}

/// Evaluator for Airmass constraint
struct AirmassEvaluator {
    max_airmass: f64,
    min_airmass: Option<f64>,
}

impl AirmassEvaluator {
    fn format_name(&self) -> String {
        match self.min_airmass {
            Some(min) => format!(
                "AirmassConstraint(min={:.2}, max={:.2})",
                min, self.max_airmass
            ),
            None => format!("AirmassConstraint(max={:.2})", self.max_airmass),
        }
    }

    /// Calculate airmass from altitude angle
    /// Airmass = 1 / sin(altitude) for altitude > 0
    /// For low altitudes, use more accurate approximations
    fn altitude_to_airmass(altitude_deg: f64) -> f64 {
        let alt_rad = altitude_deg.to_radians();

        if alt_rad <= 0.0 {
            // Target below horizon - infinite airmass
            f64::INFINITY
        } else if alt_rad < 0.174533 {
            // 10 degrees
            // Use Rozenberg approximation for low elevations
            1.0 / (alt_rad.sin() + 0.025 * (alt_rad + 0.15 * alt_rad.powi(2)).exp().ln())
        } else {
            // Simple secant approximation: airmass ≈ 1 / sin(altitude)
            1.0 / alt_rad.sin()
        }
    }
}

impl ConstraintEvaluator for AirmassEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Convert target to unit vector
        let target_vec = radec_to_unit_vectors_batch(&[target_ra], &[target_dec]);
        let target_unit = target_vec.row(0);

        let violations = track_violations(
            times,
            |i| {
                let observer_pos = observer_positions.row(i);

                // Calculate target altitude from observer position
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times[i]);
                let airmass = Self::altitude_to_airmass(altitude_deg);

                let mut violated = false;
                let mut severity = 1.0;

                if airmass > self.max_airmass {
                    violated = true;
                    severity = (airmass - self.max_airmass).min(1.0);
                }

                if let Some(min_airmass) = self.min_airmass {
                    if airmass < min_airmass {
                        violated = true;
                        severity = (min_airmass - airmass).min(1.0);
                    }
                }

                (violated, severity)
            },
            |i, _violated| {
                let observer_pos = observer_positions.row(i);
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times[i]);
                let airmass = Self::altitude_to_airmass(altitude_deg);

                if airmass > self.max_airmass {
                    format!(
                        "Airmass {:.2} > max {:.2} (altitude: {:.1}°)",
                        airmass, self.max_airmass, altitude_deg
                    )
                } else if let Some(min_airmass) = self.min_airmass {
                    if airmass < min_airmass {
                        format!(
                            "Airmass {:.2} < min {:.2} (altitude: {:.1}°)",
                            airmass, min_airmass, altitude_deg
                        )
                    } else {
                        "Airmass constraint satisfied".to_string()
                    }
                } else {
                    format!("Airmass: {:.2} (altitude: {:.1}°)", airmass, altitude_deg)
                }
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
            times.to_vec(),
        )
    }

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        let n_targets = target_ras.len();
        let n_times = times.len();

        // Convert all targets to unit vectors
        let target_vecs = radec_to_unit_vectors_batch(target_ras, target_decs);

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let observer_pos = observer_positions.row(i);

            for j in 0..n_targets {
                let target_unit = target_vecs.row(j);
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times[i]);
                let airmass = Self::altitude_to_airmass(altitude_deg);

                let mut violated = false;
                if airmass > self.max_airmass {
                    violated = true;
                }
                if let Some(min_airmass) = self.min_airmass {
                    if airmass < min_airmass {
                        violated = true;
                    }
                }

                result[[j, i]] = !violated;
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

impl AirmassEvaluator {
    /// Placeholder for target altitude calculation
    /// TODO: Implement proper alt/az calculation from ICRS to topocentric coordinates
    #[allow(dead_code)]
    fn calculate_target_altitude_placeholder(&self) -> f64 {
        45.0 // 45 degrees altitude for testing
    }

    /// Calculate target's altitude angle from observer position
    /// This is a simplified calculation - in practice you'd need proper
    /// astronomical coordinate transformations from ICRS to topocentric
    #[allow(dead_code)]
    fn calculate_target_altitude(
        &self,
        _target_unit: &ndarray::ArrayView1<f64>,
        _observer_pos: &ndarray::ArrayView1<f64>,
        _time: &DateTime<Utc>,
    ) -> f64 {
        // TODO: Implement proper ICRS to topocentric alt/az transformation
        // For now, return a dummy value that varies with time for testing
        // In practice, this would involve:
        // 1. Converting ICRS coordinates to topocentric coordinates
        // 2. Computing altitude angle from the local horizon

        // Simple time-based variation for testing (45° ± 15°)
        let hour_of_day = (_time.hour() as f64 + _time.minute() as f64 / 60.0) / 24.0;
        45.0 + 15.0 * (hour_of_day * 2.0 * std::f64::consts::PI).sin()
    }
}
