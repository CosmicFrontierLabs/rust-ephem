/// Airmass constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::celestial::calculate_airmass_batch_fast;
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
}

impl ConstraintEvaluator for AirmassEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Get airmass using fast Kasten formula (50-100x faster than SOFA)
        // Vectorized call handles the single target via slice
        let airmass_array =
            calculate_airmass_batch_fast(&[target_ra], &[target_dec], ephemeris, time_indices);
        let airmass_values = airmass_array.row(0).to_owned();

        // Extract and filter ephemeris data for times
        let (times_filtered, _) = extract_observer_ephemeris_data!(ephemeris, time_indices);

        let violations = track_violations(
            &times_filtered,
            |i| {
                let airmass = airmass_values[i];

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
                let airmass = airmass_values[i];

                if airmass > self.max_airmass {
                    format!(
                        "Airmass {:.2} > max {:.2} (too close to horizon)",
                        airmass, self.max_airmass
                    )
                } else if let Some(min_airmass) = self.min_airmass {
                    if airmass < min_airmass {
                        format!(
                            "Airmass {:.2} < min {:.2} (too high in sky)",
                            airmass, min_airmass
                        )
                    } else {
                        "Airmass constraint satisfied".to_string()
                    }
                } else {
                    format!("Airmass: {:.2}", airmass)
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
        // Get airmass for ALL targets at ALL times in one vectorized call
        let airmass_values =
            calculate_airmass_batch_fast(target_ras, target_decs, ephemeris, time_indices);

        // Vectorized constraint evaluation - single pass with mapv, no nested loops
        let result = airmass_values.mapv(|airmass| {
            let mut violated = airmass > self.max_airmass;
            if let Some(min_airmass) = self.min_airmass {
                violated = violated || airmass < min_airmass;
            }
            violated
        });

        Ok(result)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
