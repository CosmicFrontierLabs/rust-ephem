/// Generic solar system body proximity constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::constraints::fov_polygon;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for generic solar system body proximity constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyProximityConfig {
    /// Body identifier (NAIF ID or name, e.g., "Jupiter", "499")
    pub body: String,
    /// Minimum allowed angular separation in degrees (circle mode; mutually exclusive with fov_polygon)
    #[serde(default)]
    pub min_angle: Option<f64>,
    /// Maximum allowed angular separation in degrees (circle mode only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
    /// Polygon FoV vertices in instrument frame (u_deg, v_deg), mutually exclusive with min_angle.
    /// At roll=0, +u points east and +v points north on the sky.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov_polygon: Option<Vec<[f64; 2]>>,
    /// Roll angle in degrees (position angle of instrument +v from north, east of north).
    /// None means sweep all rolls (polygon mode only).
    #[serde(default)]
    pub roll_deg: Option<f64>,
}

impl ConstraintConfig for BodyProximityConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(BodyProximityEvaluator {
            body: self.body.clone(),
            // Kept for macro-generated evaluate_common (only used in circle mode)
            min_angle_deg: self.min_angle.unwrap_or(0.0),
            max_angle_deg: self.max_angle,
            fov_polygon: self.fov_polygon.clone(),
            roll_rad: self.roll_deg.map(|r| r.to_radians()),
        })
    }
}

/// Evaluator for generic body proximity - requires body positions computed externally
pub struct BodyProximityEvaluator {
    pub body: String,
    /// Used by the macro-generated evaluate_common (circle mode only)
    pub min_angle_deg: f64,
    pub max_angle_deg: Option<f64>,
    /// When set, the body must not fall inside this polygon FoV
    pub fov_polygon: Option<Vec<[f64; 2]>>,
    /// Fixed roll in radians; None means sweep all rolls (polygon mode only)
    pub roll_rad: Option<f64>,
}

impl_proximity_evaluator!(BodyProximityEvaluator, "Body", "body", sun_positions);

impl BodyProximityEvaluator {
    /// Return body positions in GCRS (km) for the correct body.
    ///
    /// Earth centre is the GCRS origin so its position is all zeros.
    fn body_positions(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
    ) -> pyo3::PyResult<Array2<f64>> {
        match self.body.to_lowercase().as_str() {
            "sun" => ephemeris.get_sun_positions(),
            "moon" => ephemeris.get_moon_positions(),
            "earth" => {
                // Earth centre = GCRS origin: return zero array with the same shape as observer.
                let obs = ephemeris.get_gcrs_positions()?;
                Ok(Array2::zeros(obs.raw_dim()))
            }
            _ => ephemeris.get_any_body_gcrs_positions(&self.body),
        }
    }

    #[allow(dead_code)]
    fn final_violation_description(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "Target too close to {} (min: {:.1}°, max: {:.1}°)",
                self.body, self.min_angle_deg, max
            ),
            None => format!(
                "Target too close to {} (min allowed: {:.1}°)",
                self.body, self.min_angle_deg
            ),
        }
    }

    #[allow(dead_code)]
    fn intermediate_violation_description(&self) -> String {
        format!("Target violates {} proximity constraint", self.body)
    }

    fn format_name(&self) -> String {
        if let Some(ref vertices) = self.fov_polygon {
            match self.roll_rad {
                Some(r) => format!(
                    "BodyProximity(body='{}', fov_polygon={} vertices, roll={:.1}°)",
                    self.body,
                    vertices.len(),
                    r.to_degrees()
                ),
                None => format!(
                    "BodyProximity(body='{}', fov_polygon={} vertices, any_roll)",
                    self.body,
                    vertices.len()
                ),
            }
        } else {
            match self.max_angle_deg {
                Some(max) => format!(
                    "BodyProximity(body='{}', min={:.1}°, max={:.1}°)",
                    self.body, self.min_angle_deg, max
                ),
                None => format!(
                    "BodyProximity(body='{}', min={:.1}°)",
                    self.body, self.min_angle_deg
                ),
            }
        }
    }

    /// Compute the body's RA/Dec (radians) from its GCRS position relative to the observer.
    fn body_radec_at(
        body_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
        i: usize,
    ) -> Option<(f64, f64)> {
        let body_rel = [
            body_positions[[i, 0]] - observer_positions[[i, 0]],
            body_positions[[i, 1]] - observer_positions[[i, 1]],
            body_positions[[i, 2]] - observer_positions[[i, 2]],
        ];
        let dist =
            (body_rel[0] * body_rel[0] + body_rel[1] * body_rel[1] + body_rel[2] * body_rel[2])
                .sqrt();
        if dist == 0.0 {
            return None;
        }
        let unit = [body_rel[0] / dist, body_rel[1] / dist, body_rel[2] / dist];
        Some(fov_polygon::unit_to_radec(&unit))
    }
}

impl ConstraintEvaluator for BodyProximityEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<ConstraintResult> {
        let all_times = ephemeris.get_times()?;
        let body_all = self.body_positions(ephemeris)?;
        let obs_all = ephemeris.get_gcrs_positions()?;
        let (times_slice, body_positions_slice, observer_positions_slice) =
            if let Some(indices) = time_indices {
                (
                    indices.iter().map(|&i| all_times[i]).collect::<Vec<_>>(),
                    body_all.select(ndarray::Axis(0), indices),
                    obs_all.select(ndarray::Axis(0), indices),
                )
            } else {
                (all_times.to_vec(), body_all, obs_all)
            };

        if let Some(ref vertices) = self.fov_polygon {
            let target_ra_rad = target_ra.to_radians();
            let target_dec_rad = target_dec.to_radians();
            let roll_rad = self.roll_rad;
            let name = self.format_name();
            let name_clone = name.clone();

            let violations = track_violations(
                &times_slice,
                |i| {
                    let violated =
                        Self::body_radec_at(&body_positions_slice, &observer_positions_slice, i)
                            .map(|(body_ra, body_dec)| {
                                fov_polygon::point_violates_polygon(
                                    target_ra_rad,
                                    target_dec_rad,
                                    body_ra,
                                    body_dec,
                                    vertices,
                                    roll_rad,
                                )
                            })
                            .unwrap_or(false);
                    (violated, if violated { 1.0 } else { 0.0 })
                },
                |_, _| name_clone.clone(),
            );
            let all_satisfied = violations.is_empty();
            return Ok(ConstraintResult::new(
                violations,
                all_satisfied,
                name,
                times_slice,
            ));
        }

        // Circle mode
        let result = self.evaluate_common(
            &times_slice,
            (target_ra, target_dec),
            &body_positions_slice,
            &observer_positions_slice,
            || self.final_violation_description(),
            || self.intermediate_violation_description(),
        );
        Ok(result)
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Array2<bool>> {
        use crate::utils::vector_math::radec_to_unit_vectors_batch;

        let times = ephemeris.get_times()?;
        let (body_positions_slice, observer_positions_slice, n_times) =
            if let Some(indices) = time_indices {
                let body_filtered = self
                    .body_positions(ephemeris)?
                    .select(ndarray::Axis(0), indices);
                let obs_filtered = ephemeris
                    .get_gcrs_positions()?
                    .select(ndarray::Axis(0), indices);
                (body_filtered, obs_filtered, indices.len())
            } else {
                let body_positions = self.body_positions(ephemeris)?;
                let observer_positions = ephemeris.get_gcrs_positions()?;
                (body_positions, observer_positions, times.len())
            };
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        if let Some(ref vertices) = self.fov_polygon {
            let roll_rad = self.roll_rad;
            for (i, (&ra, &dec)) in target_ras.iter().zip(target_decs.iter()).enumerate() {
                let target_ra_rad = ra.to_radians();
                let target_dec_rad = dec.to_radians();
                for j in 0..n_times {
                    if let Some((body_ra, body_dec)) =
                        Self::body_radec_at(&body_positions_slice, &observer_positions_slice, j)
                    {
                        result[[i, j]] = fov_polygon::point_violates_polygon(
                            target_ra_rad,
                            target_dec_rad,
                            body_ra,
                            body_dec,
                            vertices,
                            roll_rad,
                        );
                    }
                }
            }
            return Ok(result);
        }

        // Circle mode
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);
        let threshold = self.min_angle_deg.to_radians().cos();
        let max_threshold = self.max_angle_deg.map(|max| max.to_radians().cos());

        for (i, target_row) in target_vectors.axis_iter(ndarray::Axis(0)).enumerate() {
            let target_unit = [target_row[0], target_row[1], target_row[2]];

            for j in 0..n_times {
                let body_pos = [
                    body_positions_slice[[j, 0]],
                    body_positions_slice[[j, 1]],
                    body_positions_slice[[j, 2]],
                ];
                let obs_pos = [
                    observer_positions_slice[[j, 0]],
                    observer_positions_slice[[j, 1]],
                    observer_positions_slice[[j, 2]],
                ];
                let body_rel = [
                    body_pos[0] - obs_pos[0],
                    body_pos[1] - obs_pos[1],
                    body_pos[2] - obs_pos[2],
                ];
                let body_dist = (body_rel[0] * body_rel[0]
                    + body_rel[1] * body_rel[1]
                    + body_rel[2] * body_rel[2])
                    .sqrt();
                if body_dist == 0.0 {
                    continue;
                }
                let body_unit = [
                    body_rel[0] / body_dist,
                    body_rel[1] / body_dist,
                    body_rel[2] / body_dist,
                ];
                let cos_angle = target_unit[0] * body_unit[0]
                    + target_unit[1] * body_unit[1]
                    + target_unit[2] * body_unit[2];
                let too_close = cos_angle > threshold;
                let too_far = if let Some(max_thresh) = max_threshold {
                    cos_angle < max_thresh
                } else {
                    false
                };
                result[[i, j]] = too_close || too_far;
            }
        }
        Ok(result)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<Option<Array2<bool>>> {
        let times = ephemeris.get_times()?;
        let (body_positions_slice, observer_positions_slice, n_times) =
            if let Some(indices) = time_indices {
                let body_filtered = self
                    .body_positions(ephemeris)?
                    .select(ndarray::Axis(0), indices);
                let obs_filtered = ephemeris
                    .get_gcrs_positions()?
                    .select(ndarray::Axis(0), indices);
                (body_filtered, obs_filtered, indices.len())
            } else {
                let body_positions = self.body_positions(ephemeris)?;
                let observer_positions = ephemeris.get_gcrs_positions()?;
                (body_positions, observer_positions, times.len())
            };

        if target_unit_vectors.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }

        let n_targets = target_unit_vectors.nrows();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        if let Some(ref vertices) = self.fov_polygon {
            let roll_rad = self.roll_rad;
            for i in 0..n_targets {
                let ux = target_unit_vectors[[i, 0]];
                let uy = target_unit_vectors[[i, 1]];
                let uz = target_unit_vectors[[i, 2]];
                let (target_ra_rad, target_dec_rad) = fov_polygon::unit_to_radec(&[ux, uy, uz]);
                for j in 0..n_times {
                    if let Some((body_ra, body_dec)) =
                        Self::body_radec_at(&body_positions_slice, &observer_positions_slice, j)
                    {
                        result[[i, j]] = fov_polygon::point_violates_polygon(
                            target_ra_rad,
                            target_dec_rad,
                            body_ra,
                            body_dec,
                            vertices,
                            roll_rad,
                        );
                    }
                }
            }
            return Ok(Some(result));
        }

        // Circle mode
        let threshold = self.min_angle_deg.to_radians().cos();
        let max_threshold = self.max_angle_deg.map(|max| max.to_radians().cos());

        for i in 0..n_targets {
            let target_unit = [
                target_unit_vectors[[i, 0]],
                target_unit_vectors[[i, 1]],
                target_unit_vectors[[i, 2]],
            ];
            for j in 0..n_times {
                let body_pos = [
                    body_positions_slice[[j, 0]],
                    body_positions_slice[[j, 1]],
                    body_positions_slice[[j, 2]],
                ];
                let obs_pos = [
                    observer_positions_slice[[j, 0]],
                    observer_positions_slice[[j, 1]],
                    observer_positions_slice[[j, 2]],
                ];
                let body_rel = [
                    body_pos[0] - obs_pos[0],
                    body_pos[1] - obs_pos[1],
                    body_pos[2] - obs_pos[2],
                ];
                let body_dist = (body_rel[0] * body_rel[0]
                    + body_rel[1] * body_rel[1]
                    + body_rel[2] * body_rel[2])
                    .sqrt();
                if body_dist == 0.0 {
                    continue;
                }
                let body_unit = [
                    body_rel[0] / body_dist,
                    body_rel[1] / body_dist,
                    body_rel[2] / body_dist,
                ];
                let cos_angle = target_unit[0] * body_unit[0]
                    + target_unit[1] * body_unit[1]
                    + target_unit[2] * body_unit[2];
                let too_close = cos_angle > threshold;
                let too_far = if let Some(max_thresh) = max_threshold {
                    cos_angle < max_thresh
                } else {
                    false
                };
                result[[i, j]] = too_close || too_far;
            }
        }
        Ok(Some(result))
    }

    /// When this evaluator uses a free-roll polygon (no fixed roll), the polygon
    /// orientation changes with roll, so the outer field-of-regard sweep must pass
    /// the same roll to every constraint at each step.  Returning `true` here opts
    /// this evaluator into that sweep.
    fn is_roll_dependent(&self) -> bool {
        self.fov_polygon.is_some() && self.roll_rad.is_none()
    }

    /// For the free-roll polygon case: check whether the body falls inside the polygon
    /// at the specific `roll_deg` coming from the outer sweep.  This ensures the body
    /// and solar-roll constraints are evaluated at the same roll angle simultaneously.
    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> pyo3::PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();

        // Not free-roll polygon mode — fall back to the default (uses in_constraint_batch_unit_vectors).
        let vertices = match &self.fov_polygon {
            Some(v) if self.roll_rad.is_none() => v,
            _ => {
                if let Some(result) = self.in_constraint_batch_unit_vectors(
                    ephemeris,
                    target_unit_vectors,
                    Some(&[time_index]),
                )? {
                    return Ok((0..n_targets).map(|i| result[[i, 0]]).collect());
                }
                return Ok(vec![false; n_targets]);
            }
        };

        let body_positions = self.body_positions(ephemeris)?;
        let obs_positions = ephemeris.get_gcrs_positions()?;

        let body_radec = Self::body_radec_at(&body_positions, &obs_positions, time_index);

        let (sin_roll, cos_roll) = roll_deg.to_radians().sin_cos();
        let mut result = Vec::with_capacity(n_targets);
        for i in 0..n_targets {
            let ux = target_unit_vectors[[i, 0]];
            let uy = target_unit_vectors[[i, 1]];
            let uz = target_unit_vectors[[i, 2]];
            let (target_ra_rad, target_dec_rad) = fov_polygon::unit_to_radec(&[ux, uy, uz]);
            let violated = match body_radec {
                Some((body_ra, body_dec)) => fov_polygon::point_in_polygon_at_roll(
                    target_ra_rad,
                    target_dec_rad,
                    body_ra,
                    body_dec,
                    vertices,
                    sin_roll,
                    cos_roll,
                ),
                None => false,
            };
            result.push(violated);
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
