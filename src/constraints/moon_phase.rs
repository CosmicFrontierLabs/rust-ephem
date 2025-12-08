/// Moon phase constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Moon phase constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonPhaseConfig {
    /// Maximum allowed Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
    pub max_illumination: f64,
    /// Minimum allowed Moon illumination fraction (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_illumination: Option<f64>,
}

impl ConstraintConfig for MoonPhaseConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonPhaseEvaluator {
            max_illumination: self.max_illumination,
            min_illumination: self.min_illumination,
        })
    }
}

/// Evaluator for Moon phase constraint
struct MoonPhaseEvaluator {
    max_illumination: f64,
    min_illumination: Option<f64>,
}

impl MoonPhaseEvaluator {
    fn format_name(&self) -> String {
        match self.min_illumination {
            Some(min) => format!(
                "MoonPhaseConstraint(min={:.2}, max={:.2})",
                min, self.max_illumination
            ),
            None => format!("MoonPhaseConstraint(max={:.2})", self.max_illumination),
        }
    }

    /// Calculate Moon illumination fraction
    /// This is a simplified calculation - in practice you'd use proper lunar phase algorithms
    fn calculate_moon_illumination(&self, time: &DateTime<Utc>) -> f64 {
        // Simplified: use a basic approximation based on time
        // In practice, you'd calculate the phase angle between Sun, Earth, and Moon
        // For now, return a dummy value that cycles roughly monthly
        let days_since_epoch = time.timestamp() as f64 / 86400.0;
        let phase = (days_since_epoch / 29.53) % 1.0; // Lunar cycle ~29.53 days

        // Simple approximation: illumination = |sin(2π * phase)|
        // This gives 0 at new moon, 1 at full moon
        (2.0 * std::f64::consts::PI * phase).sin().abs()
    }
}

impl ConstraintEvaluator for MoonPhaseEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        _target_ra: f64,
        _target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        _observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let violations = track_violations(
            times,
            |i| {
                let time = &times[i];
                let illumination = self.calculate_moon_illumination(time);

                let mut violated = false;
                let mut severity = 1.0;

                if illumination > self.max_illumination {
                    violated = true;
                    severity = (illumination - self.max_illumination).min(1.0);
                }

                if let Some(min_illumination) = self.min_illumination {
                    if illumination < min_illumination {
                        violated = true;
                        severity = (min_illumination - illumination).min(1.0);
                    }
                }

                (violated, severity)
            },
            |i, _violated| {
                let time = &times[i];
                let illumination = self.calculate_moon_illumination(time);
                let phase_name = self.get_moon_phase_name(illumination);
                if illumination > self.max_illumination {
                    format!(
                        "Moon too bright ({:.1}%, {}) - exceeds max {:.1}%",
                        illumination * 100.0,
                        phase_name,
                        self.max_illumination * 100.0
                    )
                } else {
                    format!(
                        "Moon too dim ({:.1}%, {}) - below min {:.1}%",
                        illumination * 100.0,
                        phase_name,
                        self.min_illumination.unwrap() * 100.0
                    )
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
        _target_decs: &[f64],
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        _observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        let n_targets = target_ras.len();
        let n_times = times.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let illumination = self.calculate_moon_illumination(&times[i]);

            let mut violated = false;
            if illumination > self.max_illumination {
                violated = true;
            }
            if let Some(min_illumination) = self.min_illumination {
                if illumination < min_illumination {
                    violated = true;
                }
            }

            // Same violation status for all targets at this time
            for j in 0..n_targets {
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

impl MoonPhaseEvaluator {
    /// Get descriptive name for moon phase based on illumination
    fn get_moon_phase_name(&self, illumination: f64) -> &'static str {
        if illumination < 0.02 {
            "New Moon"
        } else if illumination < 0.48 {
            "Waxing Crescent"
        } else if illumination < 0.52 {
            "First Quarter"
        } else if illumination < 0.98 {
            "Waxing Gibbous"
        } else if illumination <= 1.02 {
            "Full Moon"
        } else if illumination < 1.48 {
            "Waning Gibbous"
        } else if illumination < 1.52 {
            "Last Quarter"
        } else {
            "Waning Crescent"
        }
    }
}
