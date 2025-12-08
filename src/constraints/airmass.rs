/// Airmass constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::celestial::radec_to_altaz;
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

    /// Calculate airmass from altitude angle and observer height
    /// Airmass = 1 / sin(altitude) for altitude > 0, corrected for height above sea level
    /// For low altitudes, use more accurate approximations
    fn altitude_to_airmass(altitude_deg: f64, height_km: f64) -> f64 {
        let alt_rad = altitude_deg.to_radians();
        let scale_height_km = 8.5; // Atmospheric scale height in km

        let airmass_sea_level = if alt_rad <= 0.0 {
            // Target below horizon - infinite airmass
            f64::INFINITY
        } else if alt_rad < 0.174533 {
            // 10 degrees
            // Use Rozenberg approximation for low elevations
            1.0 / (alt_rad.sin() + 0.025 * (alt_rad + 0.15 * alt_rad.powi(2)).exp().ln())
        } else {
            // Simple secant approximation: airmass ≈ 1 / sin(altitude)
            1.0 / alt_rad.sin()
        };

        // Correct for observer height above sea level
        // At higher altitudes, there's less atmosphere above, so airmass is reduced
        if airmass_sea_level.is_finite() {
            airmass_sea_level * (-height_km / scale_height_km).exp()
        } else {
            airmass_sea_level
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
        // Get alt/az using the proper calculation
        let altaz = radec_to_altaz(target_ra, target_dec, ephemeris, time_indices);

        // Extract and filter ephemeris data for times
        let (times_filtered, _) = extract_observer_ephemeris_data!(ephemeris, time_indices);

        // Get observer heights (ensure caches are computed)
        ephemeris.compute_latlon_caches()?;
        let heights_km =
            ephemeris.data().height_km_cache.get().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Height data not available")
            })?;

        // Filter heights by time indices
        let heights_filtered: Vec<f64> = if let Some(indices) = time_indices {
            indices.iter().map(|&i| heights_km[i]).collect()
        } else {
            heights_km.to_vec()
        };

        let violations = track_violations(
            &times_filtered,
            |i| {
                let altitude_deg = altaz[[i, 0]];
                let height_km = heights_filtered[i];
                let airmass = Self::altitude_to_airmass(altitude_deg, height_km);

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
                let altitude_deg = altaz[[i, 0]];
                let height_km = heights_filtered[i];
                let airmass = Self::altitude_to_airmass(altitude_deg, height_km);

                if airmass > self.max_airmass {
                    format!(
                        "Airmass {:.2} > max {:.2} (altitude: {:.1}°, height: {:.1} km)",
                        airmass, self.max_airmass, altitude_deg, height_km
                    )
                } else if let Some(min_airmass) = self.min_airmass {
                    if airmass < min_airmass {
                        format!(
                            "Airmass {:.2} < min {:.2} (altitude: {:.1}°, height: {:.1} km)",
                            airmass, min_airmass, altitude_deg, height_km
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
        let (times_filtered, _) = extract_observer_ephemeris_data!(ephemeris, time_indices);

        // Get observer heights (ensure caches are computed)
        ephemeris.compute_latlon_caches()?;
        let heights_km =
            ephemeris.data().height_km_cache.get().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Height data not available")
            })?;

        // Filter heights by time indices
        let heights_filtered: Vec<f64> = if let Some(indices) = time_indices {
            indices.iter().map(|&i| heights_km[i]).collect()
        } else {
            heights_km.to_vec()
        };

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for j in 0..n_targets {
            // Get alt/az for this target
            let altaz = radec_to_altaz(target_ras[j], target_decs[j], ephemeris, time_indices);

            for i in 0..n_times {
                let altitude_deg = altaz[[i, 0]];
                let height_km = heights_filtered[i];
                let airmass = Self::altitude_to_airmass(altitude_deg, height_km);

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
