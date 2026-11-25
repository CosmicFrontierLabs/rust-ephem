/// Airmass constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use super::gpu;
use crate::utils::celestial::{
    airmass_directions, calculate_airmass_batch_fast, calculate_airmass_kasten,
};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// For finite limits >= 1 the Kasten curve crosses the limit once above
/// the horizon, or jumps across it at the horizon. Bisect in sin(altitude),
/// the quantity the angular kernel compares. Ties are resolved in f64.
fn sin_altitude_limit(airmass: f64) -> f64 {
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..54 {
        let mid: f64 = (low + high) * 0.5;
        if calculate_airmass_kasten(mid.asin().to_degrees()) > airmass {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) * 0.5
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
        let n_times = time_indices.map_or_else(
            || ephemeris.data().gcrs.as_ref().map_or(0, |p| p.nrows()),
            |idx| idx.len(),
        );
        let valid_limit = |v: f64| v.is_finite() && v >= 1.0;
        if gpu::mask_eligible(target_ras.len(), n_times, gpu::TRIG_MIN_PAIRS)
            && valid_limit(self.max_airmass)
            && self.min_airmass.is_none_or(valid_limit)
            // For unwrapped longitudes, lon - RA can lose precision before
            // the CPU's cosine. Preserve that behavior for those inputs.
            && target_ras.iter().all(|ra| ra.abs() <= 360.0)
        {
            let (targets, observers) =
                airmass_directions(target_ras, target_decs, ephemeris, time_indices);
            let violated = |i: usize, t: usize| {
                let value = targets[i].airmass(observers[t]);
                value > self.max_airmass || self.min_airmass.is_some_and(|min| value < min)
            };
            let upper = self.min_airmass.map_or(f64::INFINITY, sin_altitude_limit);
            let lower = sin_altitude_limit(self.max_airmass);
            let mut vectors = Array2::zeros((targets.len(), 3));
            for (mut row, target) in vectors.rows_mut().into_iter().zip(&targets) {
                for (component, value) in row.iter_mut().zip(target.unit()) {
                    *component = value;
                }
            }
            if let Some(mask) = gpu::dispatch_angle_mask(
                &vectors,
                observers.len(),
                |t| (observers[t].unit(), upper, Some(lower)),
                gpu::Angle::Direction,
                Some(&violated),
            )? {
                return Ok(mask);
            }
        }
        Ok(
            calculate_airmass_batch_fast(target_ras, target_decs, ephemeris, time_indices).mapv(
                |value| value > self.max_airmass || self.min_airmass.is_some_and(|min| value < min),
            ),
        )
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let airmass_values =
            calculate_airmass_batch_fast(target_ras, target_decs, ephemeris, time_indices);
        let mut values = HashMap::new();
        values.insert("airmass".to_string(), airmass_values);
        Ok(values)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn ignores_explicit_roll(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
