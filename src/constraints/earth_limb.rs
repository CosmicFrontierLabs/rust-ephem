/// Earth limb avoidance constraint implementation
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult, ConstraintViolation};
use crate::utils::config::EARTH_RADIUS_KM;
use crate::utils::vector_math::{
    dot_product, normalize_vector, radec_to_unit_vector, radec_to_unit_vectors_batch,
    vector_magnitude,
};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

impl EarthLimbEvaluator {
    /// Angle (degrees) from the Earth's centre to its limb as seen from an
    /// observer `distance_km` from the centre: the apparent angular radius,
    /// plus horizon dip / refraction for ground observers when enabled. Every
    /// limb-referenced quantity (`min_angle`, `max_angle`, the reported
    /// `earth_limb_angle_deg`) is measured from this limb.
    fn limb_offset_deg(&self, distance_km: f64) -> f64 {
        let ratio = (EARTH_RADIUS_KM / distance_km).clamp(-1.0, 1.0);
        let correction = if self.horizon_dip && (distance_km - EARTH_RADIUS_KM).abs() < 100.0 {
            ratio.acos().to_degrees() + if self.include_refraction { 0.57 } else { 0.0 }
        } else {
            0.0
        };
        ratio.asin().to_degrees() + correction
    }

    /// Minimum allowed angle from the Earth's centre.
    fn threshold_deg(&self, distance_km: f64) -> f64 {
        self.limb_offset_deg(distance_km) + self.min_angle_deg
    }

    /// Cosine bounds on the angle from the Earth's centre at `distance_km`:
    /// violated when `cos > min_cos` (too close to the limb) or `cos < max_cos`
    /// (too far from it).
    ///
    /// A minimum beyond 180° from the centre cannot be met by any direction, so
    /// it maps to `-inf` (always violated) rather than being clamped to 180°,
    /// which would wrongly allow the antipodal direction. A maximum at or beyond
    /// 180° can never be exceeded, so it maps to `None` (no upper bound) rather
    /// than `cos = -1`, which dot-product roundoff below -1 could falsely
    /// violate for directions opposite the Earth.
    fn cos_bounds(&self, distance_km: f64) -> (f64, Option<f64>) {
        let offset = self.limb_offset_deg(distance_km);
        let min_total = offset + self.min_angle_deg;
        let min_cos = if min_total > 180.0 {
            f64::NEG_INFINITY
        } else {
            min_total.to_radians().cos()
        };
        (
            min_cos,
            self.max_angle_deg
                .map(|max| offset + max)
                .filter(|&max_total| max_total < 180.0)
                .map(|max_total| max_total.to_radians().cos()),
        )
    }
}

impl ConstraintEvaluator for EarthLimbEvaluator {
    fn evaluate_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<super::diagonal::DiagonalEvaluation> {
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        let observer = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No GCRS positions available")
        })?;
        let mut violated = Vec::with_capacity(target_ras.len());
        let mut angles = Vec::with_capacity(target_ras.len());
        for (i, &t) in indices.iter().enumerate() {
            let position = [observer[[t, 0]], observer[[t, 1]], observer[[t, 2]]];
            let distance = vector_magnitude(&position);
            let (min_cos, max_cos) = self.cos_bounds(distance);
            let center = normalize_vector(&[-position[0], -position[1], -position[2]]);
            let target = radec_to_unit_vector(target_ras[i], target_decs[i]);
            let cosine = dot_product(&target, &center);
            violated.push(cosine > min_cos || max_cos.is_some_and(|max| cosine < max));
            angles
                .push(cosine.clamp(-1.0, 1.0).acos().to_degrees() - self.limb_offset_deg(distance));
        }
        Ok(super::diagonal::DiagonalEvaluation::leaf(
            &self.name(),
            violated,
            HashMap::from([("earth_limb_angle_deg".to_string(), angles)]),
        ))
    }

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
        let mut violations = Vec::new();
        let mut current_violation: Option<(usize, f64)> = None;

        // Convert target RA/Dec to unit vector
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        for (i, _time) in times_filtered.iter().enumerate() {
            // Vector from observer to Earth center is -observer position
            let obs_pos = [
                obs_filtered[[i, 0]],
                obs_filtered[[i, 1]],
                obs_filtered[[i, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            let threshold_deg = self.threshold_deg(r);
            let (cos_min, cos_max) = self.cos_bounds(r);

            let center_unit = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
            let cos_angle = dot_product(&target_vec, &center_unit);
            let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

            let is_min_violation = cos_angle > cos_min;
            let is_max_violation = cos_max.is_some_and(|cos_max| cos_angle < cos_max);
            let is_violation = is_min_violation || is_max_violation;

            if is_violation {
                let severity = if is_min_violation {
                    (threshold_deg - angle_deg) / threshold_deg.max(1e-9)
                } else if let Some(max_angle) = self.max_angle_deg {
                    // For max angle violations, severity increases as the angle
                    // from the limb exceeds max
                    let limb_angle = angle_deg - self.limb_offset_deg(r);
                    (limb_angle - max_angle) / max_angle.max(1e-9)
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
                    start_time_internal: times_filtered[start_idx],
                    end_time_internal: times_filtered[i - 1],
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
                obs_filtered[[times_filtered.len() - 1, 0]],
                obs_filtered[[times_filtered.len() - 1, 1]],
                obs_filtered[[times_filtered.len() - 1, 2]],
            ];
            let r = vector_magnitude(&obs_pos);
            let threshold_deg = self.threshold_deg(r);

            violations.push(ConstraintViolation {
                start_time_internal: times_filtered[start_idx],
                end_time_internal: times_filtered[times_filtered.len() - 1],
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
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered.to_vec(),
        ))
    }

    fn name(&self) -> String {
        format!("EarthLimb(min={}°)", self.min_angle_deg)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);

        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);
        let n_targets = target_ras.len();
        let n_times = times_filtered.len();

        let mut center_units = vec![[0.0; 3]; n_times];
        let mut limb_offsets = vec![0.0; n_times];
        for t in 0..n_times {
            let obs_pos = [
                obs_filtered[[t, 0]],
                obs_filtered[[t, 1]],
                obs_filtered[[t, 2]],
            ];
            center_units[t] = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
            limb_offsets[t] = self.limb_offset_deg(vector_magnitude(&obs_pos));
        }

        let mut earth_limb_angle_deg = Array2::<f64>::zeros((n_targets, n_times));
        for i in 0..n_targets {
            let target_vec = [
                target_vectors[[i, 0]],
                target_vectors[[i, 1]],
                target_vectors[[i, 2]],
            ];
            for t in 0..n_times {
                let cos_angle = dot_product(&target_vec, &center_units[t]);
                earth_limb_angle_deg[[i, t]] =
                    cos_angle.clamp(-1.0, 1.0).acos().to_degrees() - limb_offsets[t];
            }
        }

        let mut values = HashMap::new();
        values.insert("earth_limb_angle_deg".to_string(), earth_limb_angle_deg);
        Ok(values)
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        // An apparent view asks the kernel to correct the targets: one path does that.
        if ephemeris.target_corrections().is_some() {
            let vectors =
                crate::utils::vector_math::radec_to_unit_vectors_batch(target_ras, target_decs);
            if let Some(result) =
                self.in_constraint_batch_unit_vectors(ephemeris, &vectors, time_indices)?
            {
                return Ok(result);
            }
        }
        // Extract and filter ephemeris data
        let (times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);
        // Earth radius in km

        // Validate inputs
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        // Convert all target RA/Dec to unit vectors at once
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();
        // Initialize to false (not violated) for consistency with default implementation
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        // Pre-compute cosine thresholds and center unit vectors for each time point
        // These only depend on time, not on target, so compute once and reuse
        // Using cosine trick: angle < threshold_deg ⟺ cos(angle) > cos(threshold_deg)
        let mut cos_thresholds = vec![0.0; n_times];
        let mut cos_max_thresholds = vec![None; n_times];
        let mut center_units = vec![[0.0; 3]; n_times];

        for t in 0..n_times {
            let obs_pos = [
                obs_filtered[[t, 0]],
                obs_filtered[[t, 1]],
                obs_filtered[[t, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            // Pre-compute cosine bounds (avoids acos() in inner loop)
            (cos_thresholds[t], cos_max_thresholds[t]) = self.cos_bounds(r);

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

                // Check constraint using cosine trick (avoids expensive acos/asin in inner loop)
                // angle < threshold_deg ⟺ cos(angle) > cos(threshold_deg)
                // angle > max_angle ⟺ cos(angle) < cos(max_angle)
                let is_violated = if let Some(cos_max) = cos_max_thresholds[t] {
                    // Too close to limb OR too far from limb
                    cos_angle > cos_thresholds[t] || cos_angle < cos_max
                } else {
                    // Only check if too close to limb
                    cos_angle > cos_thresholds[t]
                };

                result[[i, t]] = is_violated;
            }
        }

        Ok(result)
    }

    fn applies_target_corrections(&self) -> bool {
        true
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        // Extract and filter ephemeris data
        let (times_filtered, obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);
        // Earth radius in km

        if target_unit_vectors.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }

        let n_targets = target_unit_vectors.nrows();
        let n_times = times_filtered.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        // Pre-compute cosine thresholds and center unit vectors for each time point.
        let mut cos_thresholds = vec![0.0; n_times];
        let mut cos_max_thresholds = vec![None; n_times];
        let mut center_units = vec![[0.0; 3]; n_times];

        for t in 0..n_times {
            let obs_pos = [
                obs_filtered[[t, 0]],
                obs_filtered[[t, 1]],
                obs_filtered[[t, 2]],
            ];

            let r = vector_magnitude(&obs_pos);
            (cos_thresholds[t], cos_max_thresholds[t]) = self.cos_bounds(r);
            center_units[t] = normalize_vector(&[-obs_pos[0], -obs_pos[1], -obs_pos[2]]);
        }

        // On an apparent view the catalogue cosine becomes the apparent one.
        if let Some(corrections) = super::core::target_corrections_at(ephemeris, time_indices) {
            return Ok(Some(super::core::apparent_cosine_grid(
                target_unit_vectors,
                &center_units,
                &corrections,
                |t, cos_angle| {
                    cos_angle > cos_thresholds[t]
                        || cos_max_thresholds[t].is_some_and(|cos_max| cos_angle < cos_max)
                },
            )));
        }

        for i in 0..n_targets {
            let target_vec = [
                target_unit_vectors[[i, 0]],
                target_unit_vectors[[i, 1]],
                target_unit_vectors[[i, 2]],
            ];

            for t in 0..n_times {
                let cos_angle = dot_product(&target_vec, &center_units[t]);
                let is_violated = if let Some(cos_max) = cos_max_thresholds[t] {
                    cos_angle > cos_thresholds[t] || cos_angle < cos_max
                } else {
                    cos_angle > cos_thresholds[t]
                };
                result[[i, t]] = is_violated;
            }
        }

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::test_support::TestEphemeris;

    // TestEphemeris puts the observer at (-7000, 0, 0) km, so the Earth's
    // centre is at RA 0°, Dec 0° and its apparent radius is asin(R / 7000).
    fn earth_radius_deg() -> f64 {
        (EARTH_RADIUS_KM / 7000.0).asin().to_degrees()
    }

    fn evaluator(min: f64, max: Option<f64>) -> Box<dyn ConstraintEvaluator> {
        EarthLimbConfig {
            min_angle: min,
            max_angle: max,
            include_refraction: false,
            horizon_dip: false,
        }
        .to_evaluator()
    }

    #[test]
    fn earth_limb_angle_is_measured_from_the_limb() {
        let eph = TestEphemeris::new(3);
        let ras = [70.0, 120.0];
        let decs = [0.0, 0.0];
        let c = evaluator(5.0, None);
        let values =
            &c.compute_named_values(&eph, &ras, &decs, None).unwrap()["earth_limb_angle_deg"];
        assert!((values[[0, 0]] - (70.0 - earth_radius_deg())).abs() < 1e-9);
        assert!((values[[1, 2]] - (120.0 - earth_radius_deg())).abs() < 1e-9);

        // The paired path reports the same value.
        let diag = c
            .evaluate_diagonal(&eph, &[70.0; 3], &[0.0; 3], None)
            .unwrap();
        assert!((diag.values["earth_limb_angle_deg"][1] - values[[0, 1]]).abs() < 1e-12);
    }

    #[test]
    fn violation_matches_limb_angle_against_min_and_max() {
        let eph = TestEphemeris::new(1);
        let c = evaluator(5.0, Some(30.0));
        // Limb angles: ~2.3° (too close), ~14.3° (ok), ~34.3° (too far).
        let ras = [68.0, 80.0, 100.0];
        let decs = [0.0; 3];
        let mask = c.in_constraint_batch(&eph, &ras, &decs, None).unwrap();
        let values =
            &c.compute_named_values(&eph, &ras, &decs, None).unwrap()["earth_limb_angle_deg"];
        for i in 0..3 {
            let limb = values[[i, 0]];
            assert_eq!(mask[[i, 0]], !(5.0..=30.0).contains(&limb), "target {i}");
            assert_eq!(
                mask[[i, 0]],
                !c.evaluate(&eph, ras[i], decs[i], None)
                    .unwrap()
                    .all_satisfied
            );
        }
        assert_eq!(
            mask.column(0).to_vec(),
            vec![true, false, true],
            "max_angle must be measured from the limb, not the centre"
        );
        let unit = c
            .in_constraint_batch_unit_vectors(&eph, &radec_to_unit_vectors_batch(&ras, &decs), None)
            .unwrap()
            .unwrap();
        assert_eq!(unit, mask);
    }

    #[test]
    fn impossible_limb_minimum_is_always_violated() {
        // Earth radius ~65.7° from 7000 km; min 120° from the limb is ~185.7°
        // from the centre, so even the antipode (RA 180°, Dec 0°) is violated.
        let eph = TestEphemeris::new(2);
        let c = evaluator(120.0, None);
        let ras = [180.0, 178.0];
        let decs = [0.0, 0.0];
        let batch = c.in_constraint_batch(&eph, &ras, &decs, None).unwrap();
        assert!(batch.iter().all(|&v| v), "batch: {batch:?}");
        let unit = c
            .in_constraint_batch_unit_vectors(&eph, &radec_to_unit_vectors_batch(&ras, &decs), None)
            .unwrap()
            .unwrap();
        assert_eq!(unit, batch);
        for (&ra, &dec) in ras.iter().zip(&decs) {
            assert!(!c.evaluate(&eph, ra, dec, None).unwrap().all_satisfied);
        }
        let diag = c
            .evaluate_diagonal(&eph, &[180.0; 2], &[0.0; 2], None)
            .unwrap();
        assert_eq!(diag.violated, vec![true; 2]);
    }

    #[test]
    fn unattainable_limb_maximum_is_disabled() {
        let c = EarthLimbEvaluator {
            min_angle_deg: 10.0,
            max_angle_deg: Some(120.0),
            include_refraction: false,
            horizon_dip: false,
        };
        // offset ~65.7° + 120° > 180°: no upper bound, so roundoff in the dot
        // product can never trip it.
        assert_eq!(c.cos_bounds(7000.0).1, None);
        let c = EarthLimbEvaluator {
            max_angle_deg: Some(100.0),
            ..c
        };
        assert!(c.cos_bounds(7000.0).1.is_some());
    }

    #[test]
    fn unattainable_limb_maximum_allows_earth_opposite_directions() {
        // Earth radius ~65.7° from 7000 km; max 120° from the limb is ~185.7°
        // from the centre, so directions near the antipode must be allowed.
        let eph = TestEphemeris::new(2);
        let c = evaluator(10.0, Some(120.0));
        let ras = [180.0, 178.0];
        let decs = [0.0, 0.0];
        let batch = c.in_constraint_batch(&eph, &ras, &decs, None).unwrap();
        assert!(batch.iter().all(|&v| !v), "batch: {batch:?}");
        let unit = c
            .in_constraint_batch_unit_vectors(&eph, &radec_to_unit_vectors_batch(&ras, &decs), None)
            .unwrap()
            .unwrap();
        assert_eq!(unit, batch);
        for (&ra, &dec) in ras.iter().zip(&decs) {
            assert!(c.evaluate(&eph, ra, dec, None).unwrap().all_satisfied);
        }
        let diag = c
            .evaluate_diagonal(&eph, &[180.0; 2], &[0.0; 2], None)
            .unwrap();
        assert_eq!(diag.violated, vec![false; 2]);
    }
}
