use crate::constraints::core::{track_violations, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::unit_vectors_to_radec_batch;
use ndarray::Array2;
use pyo3::PyResult;

/// Threshold below which a vector norm or angle (in degrees) is treated as zero.
const NEAR_ZERO: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
pub(super) enum RollReference {
    Sun,
    North,
}

pub(super) struct BoresightOffsetEvaluator {
    pub(super) constraint: Box<dyn ConstraintEvaluator>,
    pub(super) roll_deg: Option<f64>,
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
        let params = self.rotation_params();

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

            if let Some(result) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                time_indices,
            )? {
                return Ok(result);
            }

            let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);

            return self.constraint.in_constraint_batch(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                time_indices,
            );
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

            let one_col = if let Some(r) = self.constraint.in_constraint_batch_unit_vectors(
                ephemeris,
                &rotated_units,
                Some(&[time_idx]),
            )? {
                r
            } else {
                let (rotated_ras, rotated_decs) = unit_vectors_to_radec_batch(&rotated_units);
                self.constraint.in_constraint_batch(
                    ephemeris,
                    &rotated_ras,
                    &rotated_decs,
                    Some(&[time_idx]),
                )?
            };

            for row in 0..n_targets {
                result[[row, col]] = one_col[[row, 0]];
            }
        }

        Ok(result)
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

    fn is_roll_dependent(&self) -> bool {
        self.roll_deg.is_none()
            && !(self.pitch_deg.abs() <= NEAR_ZERO && self.yaw_deg.abs() <= NEAR_ZERO)
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
        if !self.is_roll_dependent() {
            // Fixed roll or no offset – single evaluation suffices.
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

            let params = self.rotation_params_with_roll(step as f64 * roll_step_deg);

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

        // When roll is fixed (Some) or there is no pitch/yaw offset, roll either does
        // not affect the result or is already pinned – ignore the candidate roll_deg and
        // evaluate with the configured state (which already encodes the fixed roll).
        if self.roll_deg.is_some()
            || (self.pitch_deg.abs() <= NEAR_ZERO && self.yaw_deg.abs() <= NEAR_ZERO)
        {
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

        let params = self.rotation_params_with_roll(roll_deg);
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
