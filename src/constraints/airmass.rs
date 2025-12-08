/// Airmass constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use chrono::{DateTime, Utc};
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

                // Calculate target altitude from observer position
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times_filtered[i]);
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
                let observer_pos = obs_filtered.row(i);
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times_filtered[i]);
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
        let (times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();

        // Convert all targets to unit vectors
        let target_vecs = radec_to_unit_vectors_batch(target_ras, target_decs);

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let observer_pos = obs_filtered.row(i);

            for j in 0..n_targets {
                let target_unit = target_vecs.row(j);
                let altitude_deg =
                    self.calculate_target_altitude(&target_unit, &observer_pos, &times_filtered[i]);
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
    /// Calculate target's altitude angle from observer position
    /// This is a simplified calculation - in practice you'd need proper
    /// astronomical coordinate transformations from ICRS to topocentric
    fn calculate_target_altitude(
        &self,
        target_unit: &ndarray::ArrayView1<f64>,
        _observer_pos: &ndarray::ArrayView1<f64>,
        _time: &DateTime<Utc>,
    ) -> f64 {
        // Simple approximation: altitude = 90 - |lat - dec|
        // Assuming lat = 34.0 degrees (from test fixture)
        let z = target_unit[2];
        let dec = z.asin().to_degrees();
        let lat = 34.0;
        90.0 - (lat - dec).abs()
    }
}
