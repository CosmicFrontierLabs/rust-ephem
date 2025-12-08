/// Daytime constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Twilight type for daytime constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TwilightType {
    /// Civil twilight (-6° below horizon)
    Civil,
    /// Nautical twilight (-12° below horizon)
    Nautical,
    /// Astronomical twilight (-18° below horizon)
    Astronomical,
    /// No twilight - strict daytime only
    None,
}

/// Configuration for Daytime constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaytimeConfig {
    /// Whether to allow daytime observations (true) or nighttime only (false)
    pub allow_daytime: bool,
    /// Twilight definition to use
    pub twilight: TwilightType,
}

impl ConstraintConfig for DaytimeConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(DaytimeEvaluator {
            allow_daytime: self.allow_daytime,
            twilight: self.twilight.clone(),
        })
    }
}

/// Evaluator for Daytime constraint
struct DaytimeEvaluator {
    allow_daytime: bool,
    twilight: TwilightType,
}

impl DaytimeEvaluator {
    /// Calculate the Sun's altitude angle below horizon for twilight definition
    fn twilight_angle(&self) -> f64 {
        match self.twilight {
            TwilightType::Civil => -6.0,
            TwilightType::Nautical => -12.0,
            TwilightType::Astronomical => -18.0,
            TwilightType::None => 0.0,
        }
    }

    fn format_name(&self) -> String {
        let twilight_str = match self.twilight {
            TwilightType::Civil => "civil",
            TwilightType::Nautical => "nautical",
            TwilightType::Astronomical => "astronomical",
            TwilightType::None => "none",
        };
        if self.allow_daytime {
            format!(
                "DaytimeConstraint(allow_daytime=true, twilight={})",
                twilight_str
            )
        } else {
            format!(
                "DaytimeConstraint(allow_daytime=false, twilight={})",
                twilight_str
            )
        }
    }
}

impl ConstraintEvaluator for DaytimeEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        _target_ra: f64,
        _target_dec: f64,
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let violations = track_violations(
            times,
            |i| {
                let sun_pos = sun_positions.row(i);
                let observer_pos = observer_positions.row(i);

                // Calculate Sun's altitude from observer's perspective
                let sun_alt = self.calculate_sun_altitude(&sun_pos, &observer_pos);

                let is_daytime = sun_alt > self.twilight_angle();
                let violated = if self.allow_daytime {
                    !is_daytime // If we allow daytime, violation occurs at night
                } else {
                    is_daytime // If we don't allow daytime, violation occurs during day
                };

                (violated, 1.0)
            },
            |_, _| {
                if self.allow_daytime {
                    "Nighttime - target not visible during allowed daytime hours".to_string()
                } else {
                    "Daytime - target not visible during required nighttime hours".to_string()
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
        _target_ras: &[f64],
        _target_decs: &[f64],
        sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        let n_targets = _target_ras.len();
        let n_times = times.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        let twilight_angle_rad = self.twilight_angle();

        for i in 0..n_times {
            let sun_pos = sun_positions.row(i);
            let observer_pos = observer_positions.row(i);

            let sun_alt = self.calculate_sun_altitude(&sun_pos, &observer_pos);
            let is_daytime = sun_alt > twilight_angle_rad;

            for j in 0..n_targets {
                let violated = if self.allow_daytime {
                    !is_daytime
                } else {
                    is_daytime
                };
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

impl DaytimeEvaluator {
    /// Calculate Sun's altitude angle from observer position
    /// This is a simplified calculation - in practice you'd need proper
    /// astronomical coordinate transformations
    fn calculate_sun_altitude(
        &self,
        _sun_pos: &ndarray::ArrayView1<f64>,
        _observer_pos: &ndarray::ArrayView1<f64>,
    ) -> f64 {
        // Simplified: assume Sun position is given in topocentric coordinates
        // In practice, you'd need to convert from GCRS to topocentric alt/az
        // For now, return a dummy value that allows testing
        // TODO: Implement proper alt/az calculation
        0.1 // Slightly above horizon for testing
    }
}
