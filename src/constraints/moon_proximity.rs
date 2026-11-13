/// Moon proximity constraint implementation
use super::core::{
    compute_angle_deg_batch, track_violations, ConstraintConfig, ConstraintEvaluator,
    ConstraintResult,
};
use crate::utils::config::MOON_RADIUS_KM;
use crate::utils::vector_math::{dot_product, radec_to_unit_vector, radec_to_unit_vectors_batch};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Moon proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonProximityConfig {
    /// Minimum allowed angular separation from Moon in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from Moon in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
    /// Measure separations from the Moon's limb instead of its centre. The
    /// Moon's apparent angular radius (from the observer's distance to it at
    /// each time step) is added to `min_angle`/`max_angle`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub from_limb: bool,
}

impl ConstraintConfig for MoonProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonProximityEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
            from_limb: self.from_limb,
        })
    }
}

/// Evaluator for Moon proximity constraint
struct MoonProximityEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
    from_limb: bool,
}

/// Observer→Moon unit vector and distance (km) at row `t`.
fn moon_direction(moon: &Array2<f64>, observer: &Array2<f64>, t: usize) -> ([f64; 3], f64) {
    let rel = [
        moon[[t, 0]] - observer[[t, 0]],
        moon[[t, 1]] - observer[[t, 1]],
        moon[[t, 2]] - observer[[t, 2]],
    ];
    let dist = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
    if dist > 0.0 {
        ([rel[0] / dist, rel[1] / dist, rel[2] / dist], dist)
    } else {
        ([1.0, 0.0, 0.0], dist)
    }
}

impl MoonProximityEvaluator {
    /// Apparent angular radius of the Moon (degrees) at `dist_km`, or 0 when
    /// separations are measured from the Moon's centre.
    fn limb_offset_deg(&self, dist_km: f64) -> f64 {
        if !self.from_limb {
            return 0.0;
        }
        if dist_km <= MOON_RADIUS_KM {
            return 90.0;
        }
        (MOON_RADIUS_KM / dist_km).asin().to_degrees()
    }

    /// Cosine thresholds on the centre separation for a given limb offset.
    /// `angle < min` ⟺ `cos(angle) > min_cos`; `angle > max` ⟺ `cos(angle) < max_cos`.
    ///
    /// A minimum beyond 180° from the centre cannot be met by any direction, so
    /// it maps to `-inf` (always violated) rather than being clamped to 180°,
    /// which would wrongly allow the antipodal direction. A maximum beyond 180°
    /// can never be exceeded, so clamping it is correct.
    fn cos_thresholds(&self, offset_deg: f64) -> (f64, Option<f64>) {
        let min_total = self.min_angle_deg + offset_deg;
        let min_cos = if min_total > 180.0 {
            f64::NEG_INFINITY
        } else {
            min_total.to_radians().cos()
        };
        let max_cos = self
            .max_angle_deg
            .map(|max| (max + offset_deg).min(180.0).to_radians().cos());
        (min_cos, max_cos)
    }

    fn is_violated(cos_angle: f64, (min_cos, max_cos): (f64, Option<f64>)) -> bool {
        cos_angle > min_cos || max_cos.is_some_and(|max| cos_angle < max)
    }

    fn default_final_violation_description(&self) -> String {
        let reference = if self.from_limb { "Moon limb" } else { "Moon" };
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to {reference} (min: {:.1}°) or too far (max: {:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to {reference} (min allowed: {:.1}°)",
                self.min_angle_deg
            ),
        }
    }

    fn default_intermediate_violation_description(&self) -> String {
        "Target violates Moon proximity constraint".to_string()
    }

    fn format_name(&self) -> String {
        let limb = if self.from_limb { ", from_limb" } else { "" };
        match self.max_angle_deg {
            Some(max) => format!(
                "MoonProximity(min={}°, max={}°{limb})",
                self.min_angle_deg, max
            ),
            None => format!("MoonProximity(min={}°{limb})", self.min_angle_deg),
        }
    }

    /// Shared (targets x times) kernel over precomputed target unit vectors.
    fn violation_mask(
        &self,
        target_vectors: &Array2<f64>,
        moon: &Array2<f64>,
        observer: &Array2<f64>,
    ) -> Array2<bool> {
        let n_targets = target_vectors.nrows();
        let n_times = moon.nrows();
        let mut result = Array2::from_elem((n_targets, n_times), false);
        for t in 0..n_times {
            let (unit, dist) = moon_direction(moon, observer, t);
            let thresholds = self.cos_thresholds(self.limb_offset_deg(dist));
            for i in 0..n_targets {
                let cos_angle = target_vectors[[i, 0]] * unit[0]
                    + target_vectors[[i, 1]] * unit[1]
                    + target_vectors[[i, 2]] * unit[2];
                result[[i, t]] = Self::is_violated(cos_angle, thresholds);
            }
        }
        result
    }
}

impl ConstraintEvaluator for MoonProximityEvaluator {
    fn evaluate_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<super::diagonal::DiagonalEvaluation> {
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        let body = ephemeris.data().moon_gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No moon positions available")
        })?;
        let observer = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No GCRS positions available")
        })?;
        if !self.from_limb {
            return Ok(super::diagonal::proximity(
                &self.name(),
                "moon_angle_deg",
                target_ras,
                target_decs,
                &indices,
                body,
                observer,
                self.min_angle_deg,
                self.max_angle_deg,
            ));
        }
        let n = indices.len();
        let mut violated = Vec::with_capacity(n);
        let mut centre = Vec::with_capacity(n);
        let mut limb = Vec::with_capacity(n);
        for (i, &t) in indices.iter().enumerate() {
            let target = radec_to_unit_vector(target_ras[i], target_decs[i]);
            let (unit, dist) = moon_direction(body, observer, t);
            let offset = self.limb_offset_deg(dist);
            let cos_angle = dot_product(&target, &unit);
            violated.push(Self::is_violated(cos_angle, self.cos_thresholds(offset)));
            let angle = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
            centre.push(angle);
            limb.push(angle - offset);
        }
        Ok(super::diagonal::DiagonalEvaluation::leaf(
            &self.name(),
            violated,
            HashMap::from([
                ("moon_angle_deg".to_string(), centre),
                ("moon_limb_angle_deg".to_string(), limb),
            ]),
        ))
    }

    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let (times, moon, observer) =
            extract_body_ephemeris_data!(ephemeris, time_indices, get_moon_positions_at_indices);
        let target = radec_to_unit_vector(target_ra, target_dec);

        let violations = track_violations(
            &times,
            |t| {
                let (unit, dist) = moon_direction(&moon, &observer, t);
                let offset = self.limb_offset_deg(dist);
                let cos_angle = dot_product(&target, &unit);
                if !Self::is_violated(cos_angle, self.cos_thresholds(offset)) {
                    return (false, 0.0);
                }
                // Severity is expressed relative to the configured angles, measured
                // from the same reference (centre or limb) as the thresholds.
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees() - offset;
                let severity = if angle_deg < self.min_angle_deg {
                    if self.min_angle_deg > 0.0 {
                        (self.min_angle_deg - angle_deg) / self.min_angle_deg
                    } else {
                        0.0
                    }
                } else if let Some(max) = self.max_angle_deg {
                    (angle_deg - max) / max
                } else {
                    0.0
                };
                (true, severity)
            },
            |_, is_final| {
                if is_final {
                    self.default_final_violation_description()
                } else {
                    self.default_intermediate_violation_description()
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times.to_vec(),
        ))
    }

    /// Vectorized batch evaluation - MUCH faster than calling evaluate() in a loop
    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        let (_times, moon, observer) =
            extract_body_ephemeris_data!(ephemeris, time_indices, get_moon_positions_at_indices);
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);
        Ok(self.violation_mask(&target_vectors, &moon, &observer))
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        if target_unit_vectors.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }
        let (_times, moon, observer) =
            extract_body_ephemeris_data!(ephemeris, time_indices, get_moon_positions_at_indices);
        Ok(Some(self.violation_mask(
            target_unit_vectors,
            &moon,
            &observer,
        )))
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (_times_slice, moon_positions_slice, observer_positions_slice) =
            extract_body_ephemeris_data!(ephemeris, time_indices, get_moon_positions_at_indices);
        let angle_deg = compute_angle_deg_batch(
            target_ras,
            target_decs,
            &moon_positions_slice,
            &observer_positions_slice,
        );
        let mut values = HashMap::new();
        if self.from_limb {
            let mut limb_angle_deg = angle_deg.clone();
            for t in 0..moon_positions_slice.nrows() {
                let (_, dist) = moon_direction(&moon_positions_slice, &observer_positions_slice, t);
                let offset = self.limb_offset_deg(dist);
                limb_angle_deg.column_mut(t).mapv_inplace(|a| a - offset);
            }
            values.insert("moon_limb_angle_deg".to_string(), limb_angle_deg);
        }
        values.insert("moon_angle_deg".to_string(), angle_deg);
        Ok(values)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::test_support::TestEphemeris;

    /// Ephemeris where the Moon sits along +Y from the observer at a distance
    /// giving it an apparent angular radius of exactly 10°.
    fn close_moon_ephemeris() -> TestEphemeris {
        let mut eph = TestEphemeris::new(3);
        let dist = MOON_RADIUS_KM / 10f64.to_radians().sin();
        let moon = eph.data.moon_gcrs.as_mut().unwrap();
        for t in 0..3 {
            moon[[t, 0]] = -7000.0;
            moon[[t, 1]] = dist;
            moon[[t, 2]] = 0.0;
        }
        eph
    }

    fn evaluator(min: f64, max: Option<f64>, from_limb: bool) -> Box<dyn ConstraintEvaluator> {
        MoonProximityConfig {
            min_angle: min,
            max_angle: max,
            from_limb,
        }
        .to_evaluator()
    }

    #[test]
    fn limb_mode_adds_apparent_radius_to_min_angle() {
        let eph = close_moon_ephemeris();
        // Target 12° from Moon centre = 2° from its limb.
        let (ra, dec) = ([102.0], [0.0]);

        let centre = evaluator(5.0, None, false);
        let limb = evaluator(5.0, None, true);
        assert!(!centre.in_constraint_batch(&eph, &ra, &dec, None).unwrap()[[0, 0]]);
        assert!(limb.in_constraint_batch(&eph, &ra, &dec, None).unwrap()[[0, 0]]);
        assert!(
            !limb
                .evaluate(&eph, ra[0], dec[0], None)
                .unwrap()
                .all_satisfied
        );
        assert!(
            centre
                .evaluate(&eph, ra[0], dec[0], None)
                .unwrap()
                .all_satisfied
        );

        // 16° from centre = 6° from the limb: clear of a 5° limb avoidance.
        assert!(
            !limb
                .in_constraint_batch(&eph, &[106.0], &dec, None)
                .unwrap()[[0, 0]]
        );
    }

    #[test]
    fn limb_mode_applies_to_max_angle() {
        let eph = close_moon_ephemeris();
        // 25° from centre = 15° from limb; max 20° from limb is satisfied,
        // but max 20° from centre is not.
        let (ra, dec) = ([115.0], [0.0]);
        assert!(
            !evaluator(0.0, Some(20.0), true)
                .in_constraint_batch(&eph, &ra, &dec, None)
                .unwrap()[[0, 0]]
        );
        assert!(
            evaluator(0.0, Some(20.0), false)
                .in_constraint_batch(&eph, &ra, &dec, None)
                .unwrap()[[0, 0]]
        );
    }

    #[test]
    fn limb_mode_reports_limb_angle_values_and_matches_diagonal() {
        let eph = close_moon_ephemeris();
        let limb = evaluator(5.0, None, true);
        let values = limb
            .compute_named_values(&eph, &[102.0], &[0.0], None)
            .unwrap();
        assert!((values["moon_angle_deg"][[0, 0]] - 12.0).abs() < 1e-9);
        assert!((values["moon_limb_angle_deg"][[0, 0]] - 2.0).abs() < 1e-9);

        let ras = [102.0, 106.0, 102.0];
        let decs = [0.0, 0.0, 0.0];
        let diag = limb.evaluate_diagonal(&eph, &ras, &decs, None).unwrap();
        assert_eq!(diag.violated, vec![true, false, true]);
        assert!((diag.values["moon_limb_angle_deg"][1] - 6.0).abs() < 1e-9);

        let centre_values = evaluator(5.0, None, false)
            .compute_named_values(&eph, &[102.0], &[0.0], None)
            .unwrap();
        assert!(!centre_values.contains_key("moon_limb_angle_deg"));
    }

    #[test]
    fn impossible_limb_minimum_is_always_violated() {
        // Moon radius 10°, min 175° from the limb = 185° from the centre: no
        // direction qualifies, including the exact antipode (RA 270°, Dec 0°).
        let eph = close_moon_ephemeris();
        let c = evaluator(175.0, None, true);
        let ras = [270.0, 268.0, 90.0];
        let decs = [0.0; 3];

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
            .evaluate_diagonal(&eph, &[270.0; 3], &[0.0; 3], None)
            .unwrap();
        assert_eq!(diag.violated, vec![true; 3]);

        // Centre mode with min_angle=180 is still satisfiable at the antipode.
        assert!(
            !evaluator(180.0, None, false)
                .in_constraint_batch(&eph, &[270.0], &[0.0], None)
                .unwrap()[[0, 0]]
        );
    }

    #[test]
    fn from_limb_round_trips_through_serde_and_name() {
        let cfg: MoonProximityConfig =
            serde_json::from_str(r#"{"min_angle": 5.0, "from_limb": true}"#).unwrap();
        assert!(cfg.from_limb);
        assert_eq!(
            cfg.to_evaluator().name(),
            "MoonProximity(min=5°, from_limb)"
        );
        let legacy: MoonProximityConfig = serde_json::from_str(r#"{"min_angle": 5.0}"#).unwrap();
        assert!(!legacy.from_limb);
        assert!(!serde_json::to_string(&legacy)
            .unwrap()
            .contains("from_limb"));
    }
}
