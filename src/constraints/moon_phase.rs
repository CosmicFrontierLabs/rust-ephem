/// Moon phase constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Timelike, Utc};
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
    /// Minimum allowed Moon distance in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_distance: Option<f64>,
    /// Maximum allowed Moon distance in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f64>,
    /// Whether to enforce constraint when Moon is below horizon (default: false)
    #[serde(default)]
    pub enforce_when_below_horizon: bool,
    /// Moon visibility requirement: "full" (only when fully above horizon) or "partial" (when any part visible)
    #[serde(default = "default_moon_visibility")]
    pub moon_visibility: String,
}

fn default_moon_visibility() -> String {
    "full".to_string()
}

impl ConstraintConfig for MoonPhaseConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonPhaseEvaluator {
            max_illumination: self.max_illumination,
            min_illumination: self.min_illumination,
            min_distance: self.min_distance,
            max_distance: self.max_distance,
            enforce_when_below_horizon: self.enforce_when_below_horizon,
            moon_visibility: self.moon_visibility.clone(),
        })
    }
}

/// Evaluator for Moon phase constraint
struct MoonPhaseEvaluator {
    max_illumination: f64,
    min_illumination: Option<f64>,
    min_distance: Option<f64>,
    max_distance: Option<f64>,
    enforce_when_below_horizon: bool,
    moon_visibility: String,
}

impl MoonPhaseEvaluator {
    fn format_name(&self) -> String {
        let mut parts = Vec::new();

        match (self.min_illumination, self.max_illumination) {
            (Some(min), max) => parts.push(format!("illum={:.2}-{:.2}", min, max)),
            (None, max) => parts.push(format!("illum≤{:.2}", max)),
        }

        if let Some(min_dist) = self.min_distance {
            if let Some(max_dist) = self.max_distance {
                parts.push(format!("dist={:.1}°-{:.1}°", min_dist, max_dist));
            } else {
                parts.push(format!("dist≥{:.1}°", min_dist));
            }
        } else if let Some(max_dist) = self.max_distance {
            parts.push(format!("dist≤{:.1}°", max_dist));
        }

        if !self.enforce_when_below_horizon {
            parts.push("no-enforce-below-horizon".to_string());
        }

        if self.moon_visibility != "full" {
            parts.push(format!("visibility={}", self.moon_visibility));
        }

        format!("MoonPhaseConstraint({})", parts.join(", "))
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

    /// Calculate Moon distance in degrees from target
    /// This is a placeholder - in practice you'd calculate actual angular separation
    fn calculate_moon_distance(
        &self,
        _time: &DateTime<Utc>,
        _target_ra: f64,
        _target_dec: f64,
    ) -> f64 {
        // Placeholder: return a fixed distance for now
        // In practice, calculate angular separation between Moon and target
        45.0 // degrees
    }

    /// Calculate Moon altitude in degrees above horizon
    /// This is a placeholder - in practice you'd use astronomical calculations
    fn calculate_moon_altitude(&self, _time: &DateTime<Utc>) -> f64 {
        // Placeholder: return varying altitude based on time
        // In practice, calculate actual Moon altitude for the observer
        let hour_of_day = (_time.hour() as f64 + _time.minute() as f64 / 60.0) / 24.0;
        (hour_of_day * 2.0 * std::f64::consts::PI).sin() * 45.0 + 45.0 // -45° to +135°
    }

    /// Check if Moon is sufficiently above horizon based on visibility setting
    fn is_moon_visible(&self, altitude: f64) -> bool {
        match self.moon_visibility.as_str() {
            "full" => altitude >= 0.0,     // Moon center above horizon
            "partial" => altitude >= -0.5, // Allow some portion of Moon to be visible
            _ => altitude >= 0.0,
        }
    }
}

impl ConstraintEvaluator for MoonPhaseEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        _observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let violations = track_violations(
            times,
            |i| {
                let time = &times[i];
                let illumination = self.calculate_moon_illumination(time);
                let moon_altitude = self.calculate_moon_altitude(time);
                let moon_distance = self.calculate_moon_distance(time, target_ra, target_dec);

                // Check if we should enforce constraint based on Moon visibility
                let moon_visible = self.is_moon_visible(moon_altitude);
                if !self.enforce_when_below_horizon && !moon_visible {
                    // Moon is below horizon and we don't enforce in this case
                    return (false, 0.0);
                }

                let mut violated = false;
                let mut severity = 1.0;

                // Check illumination constraints
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

                // Check distance constraints
                if let Some(min_distance) = self.min_distance {
                    if moon_distance < min_distance {
                        violated = true;
                        severity = (min_distance - moon_distance).min(1.0);
                    }
                }
                if let Some(max_distance) = self.max_distance {
                    if moon_distance > max_distance {
                        violated = true;
                        severity = (moon_distance - max_distance).min(1.0);
                    }
                }

                (violated, severity)
            },
            |i, violated| {
                if !violated {
                    return "".to_string();
                }

                let time = &times[i];
                let illumination = self.calculate_moon_illumination(time);
                let moon_altitude = self.calculate_moon_altitude(time);
                let moon_distance = self.calculate_moon_distance(time, target_ra, target_dec);
                let phase_name = self.get_moon_phase_name(illumination);

                let mut reasons = Vec::new();

                if illumination > self.max_illumination {
                    reasons.push(format!(
                        "Moon too bright ({:.1}%, {}) - exceeds max {:.1}%",
                        illumination * 100.0,
                        phase_name,
                        self.max_illumination * 100.0
                    ));
                }
                if let Some(min_illumination) = self.min_illumination {
                    if illumination < min_illumination {
                        reasons.push(format!(
                            "Moon too dim ({:.1}%, {}) - below min {:.1}%",
                            illumination * 100.0,
                            phase_name,
                            min_illumination * 100.0
                        ));
                    }
                }

                if let Some(min_distance) = self.min_distance {
                    if moon_distance < min_distance {
                        reasons.push(format!(
                            "Moon too close ({:.1}°) - below min {:.1}°",
                            moon_distance, min_distance
                        ));
                    }
                }
                if let Some(max_distance) = self.max_distance {
                    if moon_distance > max_distance {
                        reasons.push(format!(
                            "Moon too far ({:.1}°) - exceeds max {:.1}°",
                            moon_distance, max_distance
                        ));
                    }
                }

                if reasons.is_empty() {
                    format!(
                        "Moon altitude: {:.1}°, distance: {:.1}°",
                        moon_altitude, moon_distance
                    )
                } else {
                    reasons.join("; ")
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
        _observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        let n_targets = target_ras.len();
        let n_times = times.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let time = &times[i];
            let illumination = self.calculate_moon_illumination(time);
            let moon_altitude = self.calculate_moon_altitude(time);

            // Check if we should enforce constraint based on Moon visibility
            let moon_visible = self.is_moon_visible(moon_altitude);
            if !self.enforce_when_below_horizon && !moon_visible {
                // Moon is below horizon and we don't enforce in this case
                // All targets are considered satisfied
                for j in 0..n_targets {
                    result[[j, i]] = true;
                }
                continue;
            }

            let mut violated = false;

            // Check illumination constraints
            if illumination > self.max_illumination {
                violated = true;
            }
            if let Some(min_illumination) = self.min_illumination {
                if illumination < min_illumination {
                    violated = true;
                }
            }

            // Check distance constraints for each target
            for j in 0..n_targets {
                let moon_distance =
                    self.calculate_moon_distance(time, target_ras[j], target_decs[j]);

                let mut target_violated = violated;

                if let Some(min_distance) = self.min_distance {
                    if moon_distance < min_distance {
                        target_violated = true;
                    }
                }
                if let Some(max_distance) = self.max_distance {
                    if moon_distance > max_distance {
                        target_violated = true;
                    }
                }

                result[[j, i]] = !target_violated;
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
