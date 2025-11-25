/// Earth limb avoidance constraint implementation
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult, ConstraintViolation};
use crate::utils::vector_math::{
    dot_product, normalize_vector, radec_to_unit_vector, radec_to_unit_vectors_batch,
    vector_magnitude,
};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for Earth limb avoidance constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarthLimbConfig {
    /// Additional margin beyond the Earth's apparent angular radius (degrees)
    pub min_angle: f64,
    /// Maximum allowed angular separation from Earth's limb in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
    /// Include atmospheric refraction correction for ground observers (default: true)
    /// Adds ~0.57° to horizon for standard atmosphere
    #[serde(default = "default_refraction")]
    pub include_refraction: bool,
    /// Include geometric horizon dip correction for ground observers (default: true)
    #[serde(default = "default_horizon_dip")]
    pub horizon_dip: bool,
}

fn default_refraction() -> bool {
    false
}

fn default_horizon_dip() -> bool {
    false
}

impl ConstraintConfig for EarthLimbConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(EarthLimbEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
            include_refraction: self.include_refraction,
            horizon_dip: self.horizon_dip,
        })
    }
}

/// Evaluator for Earth limb avoidance
struct EarthLimbEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
    include_refraction: bool,
    horizon_dip: bool,
}

impl ConstraintEvaluator for EarthLimbEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        target_ra: f64,
        target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;

        // Convert target RA/Dec to unit vector
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        for (i, _time) in times.iter().enumerate() {
            // Vector from observer to Earth center is -observer position
            let obs_pos = [
                observer_positions[[i, 0]],
                observer_positions[[i, 1]],
                observer_positions[[i, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            let ratio = (EARTH_RADIUS / r).clamp(-1.0, 1.0);
            let earth_ang_radius_deg = ratio.asin().to_degrees();

            // For ground observers (r close to EARTH_RADIUS), add horizon dip correction
            // Horizon dip angle = arccos(R/r), which makes objects visible slightly beyond 90°
            // For spacecraft (r >> R), this correction is negligible
            let horizon_dip_correction = if self.horizon_dip && (r - EARTH_RADIUS).abs() < 100.0 {
                // True ground observer or very low altitude (<100 km above surface)
                let dip_angle_deg = (EARTH_RADIUS / r).clamp(-1.0, 1.0).acos().to_degrees();
                let refraction = if self.include_refraction { 0.57 } else { 0.0 };
                dip_angle_deg + refraction
            } else {
                // Spacecraft or high altitude - no correction needed
                0.0
            };

            let threshold_deg = earth_ang_radius_deg + self.min_angle_deg + horizon_dip_correction;

            let center_unit = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
            let cos_angle = dot_product(&target_vec, &center_unit);
            let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

            let is_violation = if let Some(max_angle) = self.max_angle_deg {
                angle_deg < threshold_deg || angle_deg > max_angle
            } else {
                angle_deg < threshold_deg
            };

            if is_violation {
                let severity = if angle_deg < threshold_deg {
                    (threshold_deg - angle_deg) / threshold_deg.max(1e-9)
                } else if let Some(max_angle) = self.max_angle_deg {
                    // For max angle violations, severity increases as angle exceeds max
                    (angle_deg - max_angle) / max_angle.max(1e-9)
                } else {
                    0.0 // This shouldn't happen since is_violation would be false
                };
                match current_violation {
                    Some((start_idx, max_sev)) => {
                        current_violation = Some((start_idx, max_sev.max(severity)));
                    }
                    None => {
                        current_violation = Some((i, severity));
                    }
                }
            } else if let Some((start_idx, max_severity)) = current_violation {
                violations.push(ConstraintViolation {
                    start_time: times[start_idx].to_rfc3339(),
                    end_time: times[i - 1].to_rfc3339(),
                    max_severity,
                    description: match self.max_angle_deg {
                        Some(max) => format!(
                            "Target within Earth limb + margin (min: {threshold_deg:.1}°, max: {max:.1}°)"
                        ),
                        None => format!(
                            "Target within Earth limb + margin (min allowed: {threshold_deg:.1}°)"
                        ),
                    },
                });
                current_violation = None;
            }
        }

        if let Some((start_idx, max_severity)) = current_violation {
            // Compute threshold at final time for description consistency
            let obs_pos = [
                observer_positions[[times.len() - 1, 0]],
                observer_positions[[times.len() - 1, 1]],
                observer_positions[[times.len() - 1, 2]],
            ];
            let r = vector_magnitude(&obs_pos);
            let ratio = (EARTH_RADIUS / r).clamp(-1.0, 1.0);
            let earth_ang_radius_deg = ratio.asin().to_degrees();

            let horizon_dip_correction = if (r - EARTH_RADIUS).abs() < 100.0 {
                if self.horizon_dip {
                    let dip_angle_deg = (EARTH_RADIUS / r).clamp(-1.0, 1.0).acos().to_degrees();
                    let refraction = if self.include_refraction { 0.57 } else { 0.0 };
                    dip_angle_deg + refraction
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let threshold_deg = earth_ang_radius_deg + self.min_angle_deg + horizon_dip_correction;

            violations.push(ConstraintViolation {
                start_time: times[start_idx].to_rfc3339(),
                end_time: times[times.len() - 1].to_rfc3339(),
                max_severity,
                description: match self.max_angle_deg {
                    Some(max) => format!(
                        "Target within Earth limb + margin (min: {threshold_deg:.1}°, max: {max:.1}°)"
                    ),
                    None => format!(
                        "Target within Earth limb + margin (min allowed: {threshold_deg:.1}°)"
                    ),
                },
            });
        }

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(violations, all_satisfied, self.name(), times.to_vec())
    }

    fn name(&self) -> String {
        format!("EarthLimb(min={}°)", self.min_angle_deg)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
    fn evaluate_batch(
        &self,
        times: &[DateTime<Utc>],
        target_ras: &[f64],
        target_decs: &[f64],
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> pyo3::PyResult<Array2<bool>> {
        // Earth radius in km
        const EARTH_RADIUS: f64 = 6378.137;

        // Validate inputs
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        // Convert all target RA/Dec to unit vectors at once
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        let n_targets = target_ras.len();
        let n_times = times.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), true);

        // Pre-compute thresholds and center unit vectors for each time point
        // These only depend on time, not on target, so compute once and reuse
        let mut thresholds = vec![0.0; n_times];
        let mut center_units = vec![[0.0; 3]; n_times];

        for t in 0..n_times {
            let obs_pos = [
                observer_positions[[t, 0]],
                observer_positions[[t, 1]],
                observer_positions[[t, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            let ratio = (EARTH_RADIUS / r).clamp(-1.0, 1.0);
            let earth_ang_radius_deg = ratio.asin().to_degrees();

            let horizon_dip_correction = if self.horizon_dip && (r - EARTH_RADIUS).abs() < 100.0 {
                let dip_angle_deg = (EARTH_RADIUS / r).clamp(-1.0, 1.0).acos().to_degrees();
                let refraction = if self.include_refraction { 0.57 } else { 0.0 };
                dip_angle_deg + refraction
            } else {
                0.0
            };

            thresholds[t] = earth_ang_radius_deg + self.min_angle_deg + horizon_dip_correction;

            // Pre-compute Earth center direction unit vector
            center_units[t] = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
        }

        // Vectorized evaluation for each target
        for i in 0..n_targets {
            let target_vec = [
                target_vectors[[i, 0]],
                target_vectors[[i, 1]],
                target_vectors[[i, 2]],
            ];

            for t in 0..n_times {
                // Use pre-computed center unit vector
                let cos_angle = dot_product(&target_vec, &center_units[t]);
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

                // Check constraint - INVERTED: false = satisfied, true = violated
                let is_violated = if let Some(max_angle) = self.max_angle_deg {
                    angle_deg < thresholds[t] || angle_deg > max_angle
                } else {
                    angle_deg < thresholds[t]
                };

                result[[i, t]] = is_violated;
            }
        }

        Ok(result)
    }
}
