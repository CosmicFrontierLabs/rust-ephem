use crate::constraints::core::{track_violations, ConstraintEvaluator, ConstraintResult};
use ndarray::Array2;
use pyo3::PyResult;

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

impl BoresightOffsetEvaluator {
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

    fn rotated_target_for_time(
        &self,
        target_unit: &[f64; 3],
        sun_rel: &[f64; 3],
    ) -> PyResult<(f64, f64)> {
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

        if Self::norm(&z_axis) <= 1.0e-12 {
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

        let mut signed_roll_deg = self.roll_deg.unwrap_or(0.0);
        if self.roll_clockwise {
            signed_roll_deg = -signed_roll_deg;
        }
        let roll = signed_roll_deg.to_radians();
        let (sr, cr) = roll.sin_cos();

        // Rotate local +Y/+Z around +X by roll.
        let y_roll = [
            y_axis[0] * cr + z_axis[0] * sr,
            y_axis[1] * cr + z_axis[1] * sr,
            y_axis[2] * cr + z_axis[2] * sr,
        ];
        let z_roll = [
            -y_axis[0] * sr + z_axis[0] * cr,
            -y_axis[1] * sr + z_axis[1] * cr,
            -y_axis[2] * sr + z_axis[2] * cr,
        ];

        // Apply yaw then pitch in the rolled local frame (same sign convention
        // as existing Euler usage in this codebase).
        let pitch = self.pitch_deg.to_radians();
        let yaw = self.yaw_deg.to_radians();
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();

        let local_x = cp * cy;
        let local_y = cp * sy;
        let local_z = -sp;

        let rotated = [
            local_x * x_axis[0] + local_y * y_roll[0] + local_z * z_roll[0],
            local_x * x_axis[1] + local_y * y_roll[1] + local_z * z_roll[1],
            local_x * x_axis[2] + local_y * y_roll[2] + local_z * z_roll[2],
        ];

        Ok(Self::unit_vector_to_radec(&rotated))
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

            let (rotated_ra, rotated_dec) = self.rotated_target_for_time(&target_unit, &sun_rel)?;
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
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        if self.roll_deg.is_none()
            && (self.pitch_deg.abs() > 1.0e-12 || self.yaw_deg.abs() > 1.0e-12)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roll_deg is required when boresight offsets (pitch_deg or yaw_deg) are non-zero",
            ));
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

        for (col, &time_idx) in indices.iter().enumerate() {
            let sun_rel = [
                sun_positions[[time_idx, 0]] - observer_positions[[time_idx, 0]],
                sun_positions[[time_idx, 1]] - observer_positions[[time_idx, 1]],
                sun_positions[[time_idx, 2]] - observer_positions[[time_idx, 2]],
            ];

            let mut rotated_ras = Vec::with_capacity(n_targets);
            let mut rotated_decs = Vec::with_capacity(n_targets);
            for target_unit in &target_units {
                let (ra, dec) = self.rotated_target_for_time(target_unit, &sun_rel)?;
                rotated_ras.push(ra);
                rotated_decs.push(dec);
            }

            let one_col = self.constraint.in_constraint_batch(
                ephemeris,
                &rotated_ras,
                &rotated_decs,
                Some(&[time_idx]),
            )?;

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
        if target_ras.len() != target_decs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_ras and target_decs must have the same length",
            ));
        }

        if self.roll_deg.is_none()
            && (self.pitch_deg.abs() > 1.0e-12 || self.yaw_deg.abs() > 1.0e-12)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "roll_deg is required when boresight offsets (pitch_deg or yaw_deg) are non-zero",
            ));
        }

        let n = target_ras.len();
        if n == 0 {
            return Ok(Vec::new());
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
            let (ra, dec) = self.rotated_target_for_time(&target_unit, &sun_rel)?;
            rotated_ras.push(ra);
            rotated_decs.push(dec);
        }

        self.constraint
            .in_constraint_batch_diagonal(ephemeris, &rotated_ras, &rotated_decs)
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
