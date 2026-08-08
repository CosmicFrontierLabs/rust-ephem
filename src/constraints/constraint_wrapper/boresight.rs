use crate::constraints::core::{
    track_violations, BatchWithCauses, ConstraintEvaluator, ConstraintResult,
};
use crate::utils::vector_math::unit_vectors_to_radec_batch;
use ndarray::Array2;
use pyo3::PyResult;
use std::collections::HashMap;

/// Threshold below which a vector norm or angle (in degrees) is treated as zero.
const NEAR_ZERO: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
pub(super) enum RollReference {
    Sun,
    North,
}

pub(super) struct BoresightOffsetEvaluator {
    pub(super) constraint: Box<dyn ConstraintEvaluator>,
    /// Fixed instrument mounting roll about the boresight +X axis, or `None` for a
    /// free roll mounted at 0°. See `roll_free`.
    pub(super) roll_deg: Option<f64>,
    /// Spacecraft roll is free (to be swept), with `roll_deg` retained as the
    /// instrument's mounting angle that every swept candidate is added to.
    pub(super) roll_free: bool,
    pub(super) pitch_deg: f64,
    pub(super) yaw_deg: f64,
    pub(super) roll_clockwise: bool,
    pub(super) roll_reference: RollReference,
}

#[derive(Clone, Copy)]
struct RotationParams {
    sr: f64,
    cr: f64,
    local_x: f64,
    local_y: f64,
    local_z: f64,
}

impl BoresightOffsetEvaluator {
    fn rotation_params(&self) -> RotationParams {
        self.rotation_params_with_roll(self.roll_deg.unwrap_or(0.0))
    }

    /// Build rotation params for a specific roll angle (degrees), using the
    /// configured clockwise convention but overriding the stored roll_deg.
    fn rotation_params_with_roll(&self, roll_deg: f64) -> RotationParams {
        let signed_roll = if self.roll_clockwise {
            -roll_deg
        } else {
            roll_deg
        };
        self.rotation_params_from_signed_roll(signed_roll)
    }

    /// Build rotation params from a roll angle already expressed in the
    /// physical (`roll_clockwise = false`) sign convention, bypassing this
    /// node's own clockwise flip. See `rotation_params_at_candidate_roll`.
    fn rotation_params_from_signed_roll(&self, signed_roll: f64) -> RotationParams {
        let (sr, cr) = signed_roll.to_radians().sin_cos();

        // Apply yaw then pitch in the rolled local frame (same sign convention
        // as existing Euler usage in this codebase).
        let (sp, cp) = self.pitch_deg.to_radians().sin_cos();
        let (sy, cy) = self.yaw_deg.to_radians().sin_cos();

        RotationParams {
            sr,
            cr,
            local_x: cp * cy,
            local_y: cp * sy,
            local_z: -sp,
        }
    }

    /// True when this node's own boresight offset leaves the spacecraft roll free,
    /// i.e. there is a pitch/yaw offset for a roll to act on and the roll is not
    /// pinned. Does not consider the inner constraint; see `is_roll_dependent` for
    /// the whole-subtree question.
    fn own_roll_is_free(&self) -> bool {
        (self.roll_free || self.roll_deg.is_none())
            && !(self.pitch_deg.abs() <= NEAR_ZERO && self.yaw_deg.abs() <= NEAR_ZERO)
    }

    /// Rotation params for a swept candidate spacecraft roll.
    ///
    /// The candidate is a coordinated *spacecraft*-frame roll, shared unchanged by
    /// every boresight node in a tree — it is expressed in one fixed physical sign
    /// convention (the same one `roll_clockwise = false` uses) regardless of any
    /// individual node's own convention. Only this node's own mounting angle
    /// (`roll_deg`) is converted through its own `roll_clockwise` sign before the
    /// candidate is added. Applying the candidate's sign flip per-node instead
    /// (as a naive `rotation_params_with_roll(roll_deg + candidate)` would) shifts
    /// the relative orientation between a CW- and a CCW-convention node by twice
    /// the candidate roll, which defeats the "coordinated spacecraft roll" this
    /// sweep exists to model — a tree holding several offsets must keep their
    /// relative mounting angles fixed as the spacecraft rolls.
    fn rotation_params_at_candidate_roll(&self, candidate_roll_deg: f64) -> RotationParams {
        let own_signed_roll = if self.roll_clockwise {
            -self.roll_deg.unwrap_or(0.0)
        } else {
            self.roll_deg.unwrap_or(0.0)
        };
        self.rotation_params_from_signed_roll(own_signed_roll + candidate_roll_deg)
    }

    /// Shared body of `in_constraint_batch` / `in_constraint_batch_at_roll`.
    ///
    /// `params` fixes the boresight rotation to apply to every target; `inner_roll`
    /// is the candidate roll to forward to a roll-dependent inner constraint, or
    /// `None` to evaluate the inner constraint at its own configured roll. When it
    /// is `Some`, the `in_constraint_batch_unit_vectors` fast path is bypassed:
    /// that entry point has no roll-carrying variant, so taking it would drop the
    /// candidate roll on the floor.
    fn in_constraint_batch_with(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        params: RotationParams,
        inner_roll: Option<f64>,
    ) -> PyResult<Array2<bool>> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        if matches!(self.roll_reference, RollReference::North) {
            let n_targets = target_ras.len();
            let target_units: Vec<[f64; 3]> = target_ras
                .iter()
                .zip(target_decs.iter())
                .map(|(&ra, &dec)| crate::utils::vector_math::radec_to_unit_vector(ra, dec))
                .collect();

            let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated = self.rotated_target_for_time_with_params(
                    target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            if inner_roll.is_none() {
                if let Some(result) = self.constraint.in_constraint_batch_unit_vectors(
                    ephemeris,
                    &rotated_units,
                    time_indices,
                )? {
                    return Ok(result);
                }
            }

            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);

            return match inner_roll {
                Some(roll) => self.constraint.in_constraint_batch_at_roll(
                    ephemeris,
                    &rotated_ras,
                    &rotated_decs,
                    time_indices,
                    roll,
                ),
                None => self.constraint.in_constraint_batch(
                    ephemeris,
                    &rotated_ras,
                    &rotated_decs,
                    time_indices,
                ),
            };
        }

        let all_times = ephemeris.get_times()?;
        let indices: Vec<usize> = if let Some(subset) = time_indices {
            subset.to_vec()
        } else {
            (0..all_times.len()).collect()
        };

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        let n_targets = target_ras.len();
        let n_times = indices.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        let target_units: Vec<[f64; 3]> = target_ras
            .iter()
            .zip(target_decs.iter())
            .map(|(&ra, &dec)| crate::utils::vector_math::radec_to_unit_vector(ra, dec))
            .collect();
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for (col, &time_idx) in indices.iter().enumerate() {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated =
                    self.rotated_target_for_time_with_params(target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            let unit_vector_result = if inner_roll.is_none() {
                self.constraint.in_constraint_batch_unit_vectors(
                    ephemeris,
                    &rotated_units,
                    Some(&[time_idx]),
                )?
            } else {
                None
            };

            let one_col = match unit_vector_result {
                Some(r) => r,
                None => {
                    let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
                    match inner_roll {
                        Some(roll) => self.constraint.in_constraint_batch_at_roll(
                            ephemeris,
                            &rotated_ras,
                            &rotated_decs,
                            Some(&[time_idx]),
                            roll,
                        )?,
                        None => self.constraint.in_constraint_batch(
                            ephemeris,
                            &rotated_ras,
                            &rotated_decs,
                            Some(&[time_idx]),
                        )?,
                    }
                }
            };

            for row in 0..n_targets {
                result[[row, col]] = one_col[[row, 0]];
            }
        }

        Ok(result)
    }

    fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn norm(v: &[f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn normalize_or_none(v: &[f64; 3]) -> Option<[f64; 3]> {
        let n = Self::norm(v);
        if n <= 0.0 {
            None
        } else {
            Some([v[0] / n, v[1] / n, v[2] / n])
        }
    }

    fn choose_perpendicular_reference(x: &[f64; 3]) -> [f64; 3] {
        // Pick the inertial axis least parallel to boresight for numerical stability.
        if x[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    fn unit_vector_to_radec(v: &[f64; 3]) -> (f64, f64) {
        let dec_deg = v[2].clamp(-1.0, 1.0).asin().to_degrees();
        let mut ra_deg = v[1].atan2(v[0]).to_degrees();
        if ra_deg < 0.0 {
            ra_deg += 360.0;
        }
        (ra_deg, dec_deg)
    }

    fn rotated_target_for_time_with_params(
        &self,
        target_unit: &[f64; 3],
        sun_rel: &[f64; 3],
        params: RotationParams,
    ) -> PyResult<[f64; 3]> {
        let x_axis = *target_unit;

        // Roll=0 frame basis:
        // - Sun reference: +Z is Sun direction projected into plane normal to +X.
        // - North reference: +Z is celestial north projected into plane normal to +X.
        let z_ref = match self.roll_reference {
            RollReference::Sun => *sun_rel,
            RollReference::North => [0.0, 0.0, 1.0],
        };
        let zref_dot_x = x_axis[0] * z_ref[0] + x_axis[1] * z_ref[1] + x_axis[2] * z_ref[2];
        let mut z_axis = [
            z_ref[0] - zref_dot_x * x_axis[0],
            z_ref[1] - zref_dot_x * x_axis[1],
            z_ref[2] - zref_dot_x * x_axis[2],
        ];

        if Self::norm(&z_axis) <= NEAR_ZERO {
            let reference = Self::choose_perpendicular_reference(&x_axis);
            let dot_ref_x =
                x_axis[0] * reference[0] + x_axis[1] * reference[1] + x_axis[2] * reference[2];
            z_axis = [
                reference[0] - dot_ref_x * x_axis[0],
                reference[1] - dot_ref_x * x_axis[1],
                reference[2] - dot_ref_x * x_axis[2],
            ];
        }

        let z_axis = Self::normalize_or_none(&z_axis).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Unable to construct boresight frame for roll calculation",
            )
        })?;
        let mut y_axis = Self::cross(&z_axis, &x_axis);
        y_axis = Self::normalize_or_none(&y_axis).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Unable to construct boresight +Y axis for roll calculation",
            )
        })?;

        // Recompute Z from X and Y to enforce an orthonormal right-handed frame.
        let z_axis = Self::cross(&x_axis, &y_axis);

        // Rotate local +Y/+Z around +X by roll.
        let y_roll = [
            y_axis[0] * params.cr + z_axis[0] * params.sr,
            y_axis[1] * params.cr + z_axis[1] * params.sr,
            y_axis[2] * params.cr + z_axis[2] * params.sr,
        ];
        let z_roll = [
            -y_axis[0] * params.sr + z_axis[0] * params.cr,
            -y_axis[1] * params.sr + z_axis[1] * params.cr,
            -y_axis[2] * params.sr + z_axis[2] * params.cr,
        ];

        Ok([
            params.local_x * x_axis[0] + params.local_y * y_roll[0] + params.local_z * z_roll[0],
            params.local_x * x_axis[1] + params.local_y * y_roll[1] + params.local_z * z_roll[1],
            params.local_x * x_axis[2] + params.local_y * y_roll[2] + params.local_z * z_roll[2],
        ])
    }
}

impl ConstraintEvaluator for BoresightOffsetEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let params = self.rotation_params();

        if matches!(self.roll_reference, RollReference::North) {
            let target_unit =
                crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);
            let rotated =
                self.rotated_target_for_time_with_params(&target_unit, &[0.0, 0.0, 0.0], params)?;
            let (rotated_ra, rotated_dec) = Self::unit_vector_to_radec(&rotated);
            let inner =
                self.constraint
                    .evaluate(ephemeris, rotated_ra, rotated_dec, time_indices)?;

            return Ok(ConstraintResult::new(
                inner.violations,
                inner.all_satisfied,
                self.name(),
                inner.times,
            ));
        }

        let all_times = ephemeris.get_times()?;
        let indices: Vec<usize> = if let Some(subset) = time_indices {
            subset.to_vec()
        } else {
            (0..all_times.len()).collect()
        };
        let times_filtered: Vec<_> = indices.iter().map(|&idx| all_times[idx]).collect();

        let target_unit = crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);
        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        // Preserve wrapped-constraint metadata by evaluating one timestamp at a time
        // after boresight rotation and carrying forward the inner severity/description.
        let mut per_time_eval = Vec::with_capacity(indices.len());
        for &time_idx in &indices {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            let rotated =
                self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
            let (rotated_ra, rotated_dec) = Self::unit_vector_to_radec(&rotated);
            let inner =
                self.constraint
                    .evaluate(ephemeris, rotated_ra, rotated_dec, Some(&[time_idx]))?;

            if inner.violations.is_empty() {
                per_time_eval.push((false, 0.0f64, String::new()));
            } else {
                let severity = inner
                    .violations
                    .iter()
                    .map(|v| v.max_severity)
                    .fold(0.0f64, f64::max);
                let description = inner
                    .violations
                    .iter()
                    .map(|v| v.description.as_str())
                    .collect::<Vec<_>>()
                    .join("; ");
                per_time_eval.push((true, severity, description));
            }
        }

        let violations = track_violations(
            &times_filtered,
            |i| (per_time_eval[i].0, per_time_eval[i].1),
            |i, _is_open| {
                if per_time_eval[i].2.is_empty() {
                    self.name()
                } else {
                    per_time_eval[i].2.clone()
                }
            },
        );
        let all_satisfied = violations.is_empty();

        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times_filtered,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        self.in_constraint_batch_with(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            self.rotation_params(),
            None,
        )
    }

    /// Evaluate at a single candidate spacecraft roll from the free-roll sweep.
    ///
    /// Without this override the default trait implementation would fall through to
    /// `in_constraint_batch`, which ignores `roll_deg` — so a whole
    /// `in_constraint_batch_constrained_at_every_roll` sweep would AND together
    /// `n_roll_samples` copies of the *same* roll-0 answer and silently report the
    /// fixed-roll result as if it were the free-roll one.
    ///
    /// Two independent sources of roll dependence are handled separately: this
    /// node's own pitch/yaw offset (swept only when its `roll_deg` is `None`, i.e.
    /// the roll was left free rather than pinned to a fixed instrument mounting),
    /// and a roll-dependent inner constraint, which receives the same candidate
    /// roll so a coordinated sweep stays coordinated through the offset wrapper.
    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> PyResult<Array2<bool>> {
        let params = if self.own_roll_is_free() {
            self.rotation_params_at_candidate_roll(roll_deg)
        } else {
            self.rotation_params()
        };
        let inner_roll = if self.constraint.is_roll_dependent() {
            Some(roll_deg)
        } else {
            None
        };
        self.in_constraint_batch_with(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            params,
            inner_roll,
        )
    }

    fn in_constraint_batch_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<Vec<bool>> {
        let params = self.rotation_params();

        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n = target_ras.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        if matches!(self.roll_reference, RollReference::North) {
            let mut rotated_ras = Vec::with_capacity(n);
            let mut rotated_decs = Vec::with_capacity(n);
            for i in 0..n {
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ras[i], target_decs[i]);
                let rotated = self.rotated_target_for_time_with_params(
                    &target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                let (ra, dec) = Self::unit_vector_to_radec(&rotated);
                rotated_ras.push(ra);
                rotated_decs.push(dec);
            }

            return self.constraint.in_constraint_batch_diagonal(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
            );
        }

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        if sun_positions.nrows() < n || observer_positions.nrows() < n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Ephemeris does not have enough samples for diagonal boresight evaluation",
            ));
        }

        let mut rotated_ras = Vec::with_capacity(n);
        let mut rotated_decs = Vec::with_capacity(n);
        for i in 0..n {
            let target_unit =
                crate::utils::vector_math::radec_to_unit_vector(target_ras[i], target_decs[i]);
            let sun_rel = [
                sun_positions[[i, 0]] - observer_positions[[i, 0]],
                sun_positions[[i, 1]] - observer_positions[[i, 1]],
                sun_positions[[i, 2]] - observer_positions[[i, 2]],
            ];
            let rotated =
                self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
            let (ra, dec) = Self::unit_vector_to_radec(&rotated);
            rotated_ras.push(ra);
            rotated_decs.push(dec);
        }

        self.constraint
            .in_constraint_batch_diagonal(ephemeris, &rotated_ras, &rotated_decs)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        let params = self.rotation_params();

        if target_unit_vectors.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }

        let n_targets = target_unit_vectors.nrows();

        if matches!(self.roll_reference, RollReference::North) {
            let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
            for i in 0..n_targets {
                let target_unit = [
                    target_unit_vectors[[i, 0]],
                    target_unit_vectors[[i, 1]],
                    target_unit_vectors[[i, 2]],
                ];
                let rotated = self.rotated_target_for_time_with_params(
                    &target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            if let Some(result) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                time_indices,
            )? {
                return Ok(Some(result));
            }

            let (target_ras, target_decs) = unit_vectors_to_radec_batch(&rotated_units);
            return self
                .constraint
                .in_constraint_batch(ephemeris, &target_ras, &target_decs, time_indices)
                .map(Some);
        }

        let all_times = ephemeris.get_times()?;
        let indices: Vec<usize> = if let Some(subset) = time_indices {
            subset.to_vec()
        } else {
            (0..all_times.len()).collect()
        };

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        if indices.len() == 1 {
            let time_idx = indices[0];
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
            for i in 0..n_targets {
                let target_unit = [
                    target_unit_vectors[[i, 0]],
                    target_unit_vectors[[i, 1]],
                    target_unit_vectors[[i, 2]],
                ];
                let rotated =
                    self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            if let Some(result) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                Some(&[time_idx]),
            )? {
                return Ok(Some(result));
            }

            let (target_ras, target_decs) = unit_vectors_to_radec_batch(&rotated_units);
            return self
                .constraint
                .in_constraint_batch(ephemeris, &target_ras, &target_decs, Some(&[time_idx]))
                .map(Some);
        }

        let n_times = indices.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for (col, &time_idx) in indices.iter().enumerate() {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            for i in 0..n_targets {
                let target_unit = [
                    target_unit_vectors[[i, 0]],
                    target_unit_vectors[[i, 1]],
                    target_unit_vectors[[i, 2]],
                ];
                let rotated =
                    self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            let one_col = if let Some(r) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                Some(&[time_idx]),
            )? {
                r
            } else {
                let (target_ras, target_decs) = unit_vectors_to_radec_batch(&rotated_units);
                self.constraint.in_constraint_batch(
                    ephemeris,
                    &target_ras,
                    &target_decs,
                    Some(&[time_idx]),
                )?
            };

            for row in 0..n_targets {
                result[[row, col]] = one_col[[row, 0]];
            }
        }

        Ok(Some(result))
    }

    /// Roll-dependent if this node's own offset leaves roll free, or if the inner
    /// constraint is itself roll-dependent (e.g. a `SolarRoll` or a polygon FoV
    /// nested under a zero-offset wrapper). Mirrors the Python-side
    /// `RustConstraintMixin._is_roll_dependent`, which likewise recurses through a
    /// boresight node into its child; without the recursion a sweep would stop at
    /// this wrapper and the inner constraint would never see a candidate roll.
    fn is_roll_dependent(&self) -> bool {
        self.own_roll_is_free() || self.constraint.is_roll_dependent()
    }

    /// Efficient standalone sweep for free-roll FoR: reuses a single allocation across
    /// all roll steps.  The default trait sweep via `field_of_regard_violated_at_roll`
    /// would allocate a new buffer per step; this override avoids that cost.
    fn field_of_regard_violated_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        n_roll_samples: usize,
    ) -> pyo3::PyResult<Vec<bool>> {
        if !self.own_roll_is_free() {
            // Fixed roll or no offset – rolling this node's frame changes nothing,
            // so a single evaluation suffices. Deliberately `own_roll_is_free`
            // rather than `is_roll_dependent`: the latter also reports true for a
            // roll-dependent *inner* constraint, which this loop cannot sweep
            // anyway (`field_of_regard_violated_at_roll` evaluates the inner
            // constraint at its own configured roll), so looping on it would burn
            // `n_roll_samples` identical evaluations for the same answer.
            return self.field_of_regard_violated_at_roll(
                ephemeris,
                target_unit_vectors,
                time_index,
                0.0,
            );
        }

        let n_targets = target_unit_vectors.nrows();
        let roll_step_deg = 360.0 / n_roll_samples as f64;

        // Compute sun_rel once outside the loop.
        let sun_rel: [f64; 3] = match self.roll_reference {
            RollReference::Sun => {
                let sun_positions = ephemeris.get_sun_positions()?;
                let observer_positions = ephemeris.get_gcrs_positions()?;
                [
                    sun_positions[[time_index, 0]] - observer_positions[[time_index, 0]],
                    sun_positions[[time_index, 1]] - observer_positions[[time_index, 1]],
                    sun_positions[[time_index, 2]] - observer_positions[[time_index, 2]],
                ]
            }
            RollReference::North => [0.0, 0.0, 0.0],
        };

        // Reuse a single allocation for the rotated unit vectors across all roll steps.
        let mut accessible = vec![false; n_targets];
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for step in 0..n_roll_samples {
            if accessible.iter().all(|&a| a) {
                break;
            }

            let params = self.rotation_params_at_candidate_roll(step as f64 * roll_step_deg);

            for i in 0..n_targets {
                let target_unit = [
                    target_unit_vectors[[i, 0]],
                    target_unit_vectors[[i, 1]],
                    target_unit_vectors[[i, 2]],
                ];
                let rotated =
                    self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            let violated_col = if let Some(r) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                Some(&[time_index]),
            )? {
                r
            } else {
                let (ras, decs) = unit_vectors_to_radec_batch(&rotated_units);
                self.constraint
                    .in_constraint_batch(ephemeris, &ras, &decs, Some(&[time_index]))?
            };

            for i in 0..n_targets {
                if !violated_col[[i, 0]] {
                    accessible[i] = true;
                }
            }
        }

        Ok(accessible.iter().map(|&a| !a).collect())
    }

    fn field_of_regard_violated_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_index: usize,
        roll_deg: f64,
    ) -> pyo3::PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();

        // When roll is pinned or there is no pitch/yaw offset, roll either does not
        // affect the result or is already fixed – ignore the candidate roll_deg and
        // evaluate with the configured state (which already encodes the fixed roll).
        if !self.own_roll_is_free() {
            if let Some(result) = self.in_constraint_batch_unit_vectors(
                ephemeris,
                target_unit_vectors,
                Some(&[time_index]),
            )? {
                return Ok((0..n_targets).map(|i| result[[i, 0]]).collect());
            }
            let (ras, decs) = unit_vectors_to_radec_batch(target_unit_vectors);
            let result = self.in_constraint_batch(ephemeris, &ras, &decs, Some(&[time_index]))?;
            return Ok((0..n_targets).map(|i| result[[i, 0]]).collect());
        }

        // Free roll: evaluate at the specific roll_deg provided by the sweep.
        let sun_rel: [f64; 3] = match self.roll_reference {
            RollReference::Sun => {
                let sun_positions = ephemeris.get_sun_positions()?;
                let observer_positions = ephemeris.get_gcrs_positions()?;
                [
                    sun_positions[[time_index, 0]] - observer_positions[[time_index, 0]],
                    sun_positions[[time_index, 1]] - observer_positions[[time_index, 1]],
                    sun_positions[[time_index, 2]] - observer_positions[[time_index, 2]],
                ]
            }
            RollReference::North => [0.0, 0.0, 0.0],
        };

        let params = self.rotation_params_at_candidate_roll(roll_deg);
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for i in 0..n_targets {
            let target_unit = [
                target_unit_vectors[[i, 0]],
                target_unit_vectors[[i, 1]],
                target_unit_vectors[[i, 2]],
            ];
            let rotated =
                self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
            rotated_units[[i, 0]] = rotated[0];
            rotated_units[[i, 1]] = rotated[1];
            rotated_units[[i, 2]] = rotated[2];
        }

        let violated_col = if let Some(r) = self.constraint.in_constraint_batch_unit_vectors(
            ephemeris,
            &rotated_units,
            Some(&[time_index]),
        )? {
            r
        } else {
            let (ras, decs) = unit_vectors_to_radec_batch(&rotated_units);
            self.constraint
                .in_constraint_batch(ephemeris, &ras, &decs, Some(&[time_index]))?
        };

        Ok((0..n_targets).map(|i| violated_col[[i, 0]]).collect())
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let params = self.rotation_params();
        let n_targets = target_ras.len();
        let target_units: Vec<[f64; 3]> = target_ras
            .iter()
            .zip(target_decs.iter())
            .map(|(&ra, &dec)| crate::utils::vector_math::radec_to_unit_vector(ra, dec))
            .collect();

        // Single child, evaluated at the rotated direction — pass through unprefixed.
        if matches!(self.roll_reference, RollReference::North) {
            // Rotation is time-invariant for the north reference: rotate once per
            // target and delegate directly, letting the child broadcast over times.
            let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated = self.rotated_target_for_time_with_params(
                    target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }
            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
            return self.constraint.compute_named_values(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                time_indices,
            );
        }

        // Sun reference: the rotation depends on the Sun direction at each timestamp,
        // so evaluate one time column at a time (same cost pattern already used by
        // in_constraint_batch for this reference mode).
        let all_times = ephemeris.get_times()?;
        let indices: Vec<usize> = if let Some(subset) = time_indices {
            subset.to_vec()
        } else {
            (0..all_times.len()).collect()
        };

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        let mut merged: HashMap<String, Array2<f64>> = HashMap::new();
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for (col, &time_idx) in indices.iter().enumerate() {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated =
                    self.rotated_target_for_time_with_params(target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
            let column_values = self.constraint.compute_named_values(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                Some(&[time_idx]),
            )?;

            for (key, arr) in column_values {
                let entry = merged
                    .entry(key)
                    .or_insert_with(|| Array2::<f64>::zeros((n_targets, indices.len())));
                for row in 0..n_targets {
                    entry[[row, col]] = arr[[row, 0]];
                }
            }
        }

        Ok(merged)
    }

    /// Rotate the targets once, then take both the combined mask and the wrapped
    /// subtree's leaf masks from a single evaluation at the rotated directions.
    ///
    /// The child's cause tags pass through unprefixed (mirroring `compute_named_values`
    /// above). Without this override the trait default would collapse a wrapped
    /// subtree's leaf attribution into one `boresight_offset` tag, hiding which inner
    /// leaf (e.g. `sun`, `moon`) is actually responsible and making the wrapped subtree
    /// count as a single leaf for witness selection.
    ///
    /// `roll_deg` is picked apart the same way `in_constraint_batch_at_roll` does, into
    /// the two independent sources of roll dependence: this node's own free pitch/yaw
    /// offset (via `rotation_params_at_candidate_roll`) and a roll-dependent inner
    /// constraint, which receives the same candidate roll so a coordinated sweep stays
    /// coordinated through the offset wrapper.
    ///
    /// Unlike `in_constraint_batch`, this always reaches the child through RA/Dec rather
    /// than trying `in_constraint_batch_unit_vectors` first: the child has to produce its
    /// leaf masks from the RA/Dec entry point anyway, and routing both outputs through it
    /// keeps them from disagreeing at a threshold while saving a whole second traversal.
    fn in_constraint_batch_with_causes(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: Option<f64>,
    ) -> PyResult<BatchWithCauses> {
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let (params, inner_roll) = match roll_deg {
            Some(roll) => (
                if self.own_roll_is_free() {
                    self.rotation_params_at_candidate_roll(roll)
                } else {
                    self.rotation_params()
                },
                if self.constraint.is_roll_dependent() {
                    Some(roll)
                } else {
                    None
                },
            ),
            None => (self.rotation_params(), None),
        };

        let n_targets = target_ras.len();
        let target_units: Vec<[f64; 3]> = target_ras
            .iter()
            .zip(target_decs.iter())
            .map(|(&ra, &dec)| crate::utils::vector_math::radec_to_unit_vector(ra, dec))
            .collect();

        if matches!(self.roll_reference, RollReference::North) {
            let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated = self.rotated_target_for_time_with_params(
                    target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }
            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
            return self.constraint.in_constraint_batch_with_causes(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                time_indices,
                inner_roll,
            );
        }

        // Sun-referenced roll: the rotation depends on the Sun direction and so changes
        // with time, meaning targets must be re-rotated (and the child re-evaluated) one
        // time column at a time — the same cost pattern `in_constraint_batch` uses here.
        let all_times = ephemeris.get_times()?;
        let indices: Vec<usize> = if let Some(subset) = time_indices {
            subset.to_vec()
        } else {
            (0..all_times.len()).collect()
        };

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        let n_times = indices.len();
        let mut violated = Array2::<bool>::from_elem((n_targets, n_times), false);
        let mut named: HashMap<String, Array2<bool>> = HashMap::new();
        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));

        for (col, &time_idx) in indices.iter().enumerate() {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            for (i, target_unit) in target_units.iter().enumerate() {
                let rotated =
                    self.rotated_target_for_time_with_params(target_unit, &sun_rel, params)?;
                rotated_units[[i, 0]] = rotated[0];
                rotated_units[[i, 1]] = rotated[1];
                rotated_units[[i, 2]] = rotated[2];
            }

            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
            let column = self.constraint.in_constraint_batch_with_causes(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                Some(&[time_idx]),
                inner_roll,
            )?;

            for row in 0..n_targets {
                violated[[row, col]] = column.violated[[row, 0]];
            }
            for (key, arr) in column.named {
                let entry = named
                    .entry(key)
                    .or_insert_with(|| Array2::<bool>::from_elem((n_targets, n_times), false));
                for row in 0..n_targets {
                    entry[[row, col]] = arr[[row, 0]];
                }
            }
        }

        Ok(BatchWithCauses { violated, named })
    }

    /// Diagonal variant of the cause attribution for moving-body evaluation:
    /// target_i paired with time_i. Mirrors `in_constraint_batch_diagonal`'s
    /// rotation logic above, delegating to the child's own diagonal attribution.
    fn compute_named_booleans_diagonal(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
    ) -> PyResult<HashMap<String, Vec<bool>>> {
        let params = self.rotation_params();

        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        let n = target_ras.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        if matches!(self.roll_reference, RollReference::North) {
            let mut rotated_ras = Vec::with_capacity(n);
            let mut rotated_decs = Vec::with_capacity(n);
            for i in 0..n {
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ras[i], target_decs[i]);
                let rotated = self.rotated_target_for_time_with_params(
                    &target_unit,
                    &[0.0, 0.0, 0.0],
                    params,
                )?;
                let (ra, dec) = Self::unit_vector_to_radec(&rotated);
                rotated_ras.push(ra);
                rotated_decs.push(dec);
            }

            return self.constraint.compute_named_booleans_diagonal(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
            );
        }

        let sun_positions = ephemeris.get_sun_positions()?;
        let observer_positions = ephemeris.get_gcrs_positions()?;

        if sun_positions.nrows() < n || observer_positions.nrows() < n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Ephemeris does not have enough samples for diagonal boresight evaluation",
            ));
        }

        let mut rotated_ras = Vec::with_capacity(n);
        let mut rotated_decs = Vec::with_capacity(n);
        for i in 0..n {
            let target_unit =
                crate::utils::vector_math::radec_to_unit_vector(target_ras[i], target_decs[i]);
            let sun_rel = [
                sun_positions[[i, 0]] - observer_positions[[i, 0]],
                sun_positions[[i, 1]] - observer_positions[[i, 1]],
                sun_positions[[i, 2]] - observer_positions[[i, 2]],
            ];
            let rotated =
                self.rotated_target_for_time_with_params(&target_unit, &sun_rel, params)?;
            let (ra, dec) = Self::unit_vector_to_radec(&rotated);
            rotated_ras.push(ra);
            rotated_decs.push(dec);
        }

        self.constraint
            .compute_named_booleans_diagonal(ephemeris, &rotated_ras, &rotated_decs)
    }

    /// Maps this wrapper's (pass-through) cause tag(s) to their `constraint_values`
    /// key(s), mirroring `in_constraint_batch_with_causes`' unprefixed passthrough so a
    /// wrapped leaf's cause tag (e.g. `sun`) — not `boresight_offset` — is what
    /// callers see. The rotated position only needs to be *some* valid boresight
    /// direction: the value-key *names* a child returns from
    /// `compute_named_values` don't vary with target position, only the numeric
    /// values do, so a single representative rotation (rather than per-time-column
    /// rotation) is sufficient here.
    fn compute_cause_value_keys(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Vec<String>>> {
        let params = self.rotation_params();
        let n_targets = target_ras.len();
        let target_units: Vec<[f64; 3]> = target_ras
            .iter()
            .zip(target_decs.iter())
            .map(|(&ra, &dec)| crate::utils::vector_math::radec_to_unit_vector(ra, dec))
            .collect();

        let sun_rel: [f64; 3] = match self.roll_reference {
            RollReference::North => [0.0, 0.0, 0.0],
            RollReference::Sun => {
                let all_times = ephemeris.get_times()?;
                if all_times.is_empty() {
                    [0.0, 0.0, 0.0]
                } else {
                    let time_idx = time_indices
                        .and_then(|idx| idx.first().copied())
                        .unwrap_or(0)
                        .min(all_times.len() - 1);
                    let sun_positions = ephemeris.get_sun_positions()?;
                    let observer_positions = ephemeris.get_gcrs_positions()?;
                    [
                        sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                        sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                        sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
                    ]
                }
            }
        };

        let mut rotated_units = Array2::<f64>::zeros((n_targets, 3));
        for (i, target_unit) in target_units.iter().enumerate() {
            let rotated =
                self.rotated_target_for_time_with_params(target_unit, &sun_rel, params)?;
            rotated_units[[i, 0]] = rotated[0];
            rotated_units[[i, 1]] = rotated[1];
            rotated_units[[i, 2]] = rotated[2];
        }
        let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);

        self.constraint.compute_cause_value_keys(
            ephemeris,
            &rotated_ras,
            &rotated_decs,
            time_indices,
        )
    }

    fn name(&self) -> String {
        format!(
            "BoresightOffset({}, roll={:.3}°, roll_clockwise={}, roll_reference={}, pitch={:.3}°, yaw={:.3}°)",
            self.constraint.name(),
            self.roll_deg.unwrap_or(0.0),
            self.roll_clockwise,
            match self.roll_reference {
                RollReference::Sun => "sun",
                RollReference::North => "north",
            },
            self.pitch_deg,
            self.yaw_deg
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::core::ConstraintResult;

    struct DummyLeaf;

    impl ConstraintEvaluator for DummyLeaf {
        fn evaluate(
            &self,
            _ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
            _target_ra: f64,
            _target_dec: f64,
            _time_indices: Option<&[usize]>,
        ) -> PyResult<ConstraintResult> {
            unimplemented!("not exercised by these rotation-math tests")
        }

        fn in_constraint_batch(
            &self,
            _ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
            _target_ras: &[f64],
            _target_decs: &[f64],
            _time_indices: Option<&[usize]>,
        ) -> PyResult<Array2<bool>> {
            unimplemented!("not exercised by these rotation-math tests")
        }

        fn name(&self) -> String {
            "dummy".to_string()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn evaluator(roll_deg: f64, roll_clockwise: bool) -> BoresightOffsetEvaluator {
        BoresightOffsetEvaluator {
            constraint: Box::new(DummyLeaf),
            roll_deg: Some(roll_deg),
            roll_free: true,
            pitch_deg: 5.0,
            yaw_deg: 3.0,
            roll_clockwise,
            roll_reference: RollReference::North,
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn assert_params_eq(a: RotationParams, b: RotationParams, context: &str) {
        assert!(
            approx_eq(a.sr, b.sr),
            "{context}: sr differs ({} vs {})",
            a.sr,
            b.sr
        );
        assert!(
            approx_eq(a.cr, b.cr),
            "{context}: cr differs ({} vs {})",
            a.cr,
            b.cr
        );
    }

    /// A coordinated spacecraft-roll candidate must be applied in one fixed
    /// physical convention: `roll_deg=90, roll_clockwise=false` and
    /// `roll_deg=-90, roll_clockwise=true` describe the *same* physical
    /// mounting, so `rotation_params_at_candidate_roll` must return identical
    /// results for both, at every candidate roll. Before the fix, the candidate
    /// was re-signed through each node's own convention, so the CW node's result
    /// diverged from the CCW node's by twice the candidate roll.
    #[test]
    fn candidate_roll_preserves_relative_geometry_across_conventions() {
        let ccw_node = evaluator(90.0, false);
        let cw_node = evaluator(-90.0, true);

        for candidate in [0.0, 30.0, 137.5, 200.0, 359.0] {
            let ccw_params = ccw_node.rotation_params_at_candidate_roll(candidate);
            let cw_params = cw_node.rotation_params_at_candidate_roll(candidate);
            assert_params_eq(ccw_params, cw_params, &format!("candidate={candidate}"));
        }
    }

    /// Sanity check pinning down the actual physical convention: with no
    /// mounting offset (`roll_deg=0`), the candidate roll alone must produce
    /// the same signed roll a plain `roll_clockwise=false` node would apply
    /// directly, regardless of `self`'s own `roll_clockwise`.
    #[test]
    fn candidate_roll_alone_matches_ccw_convention() {
        let zero_mount_ccw = evaluator(0.0, false);
        let zero_mount_cw = evaluator(0.0, true);

        for candidate in [0.0, 45.0, 90.0, 271.0] {
            let expected = zero_mount_ccw.rotation_params_with_roll(candidate);
            let ccw_result = zero_mount_ccw.rotation_params_at_candidate_roll(candidate);
            let cw_result = zero_mount_cw.rotation_params_at_candidate_roll(candidate);
            assert_params_eq(expected, ccw_result, &format!("ccw candidate={candidate}"));
            assert_params_eq(expected, cw_result, &format!("cw candidate={candidate}"));
        }
    }
}
