struct BoresightOffsetEvaluator {
    constraint: Box<dyn ConstraintEvaluator>,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
    rotation_matrix: [[f64; 3]; 3],
}

impl BoresightOffsetEvaluator {
    fn rotate_targets(&self, target_ras: &[f64], target_decs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut offset_ras = Vec::with_capacity(target_ras.len());
        let mut offset_decs = Vec::with_capacity(target_decs.len());

        for (&ra, &dec) in target_ras.iter().zip(target_decs.iter()) {
            let (offset_ra, offset_dec) = rotate_radec_with_matrix(ra, dec, &self.rotation_matrix);
            offset_ras.push(offset_ra);
            offset_decs.push(offset_dec);
        }

        (offset_ras, offset_decs)
    }
}

fn euler_zyx_rotation_matrix(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [[f64; 3]; 3] {
    // Intrinsic Z-Y-X Euler sequence (yaw, pitch, roll)
    // Equivalent to extrinsic X-Y-Z applied to fixed frame.
    let roll = roll_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let yaw = yaw_deg.to_radians();

    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn rotate_radec_with_matrix(
    ra_deg: f64,
    dec_deg: f64,
    rotation_matrix: &[[f64; 3]; 3],
) -> (f64, f64) {
    let v = crate::utils::vector_math::radec_to_unit_vector(ra_deg, dec_deg);

    let x =
        rotation_matrix[0][0] * v[0] + rotation_matrix[0][1] * v[1] + rotation_matrix[0][2] * v[2];
    let y =
        rotation_matrix[1][0] * v[0] + rotation_matrix[1][1] * v[1] + rotation_matrix[1][2] * v[2];
    let z =
        rotation_matrix[2][0] * v[0] + rotation_matrix[2][1] * v[1] + rotation_matrix[2][2] * v[2];

    let dec_rot = z.clamp(-1.0, 1.0).asin().to_degrees();
    let mut ra_rot = y.atan2(x).to_degrees();
    if ra_rot < 0.0 {
        ra_rot += 360.0;
    }

    (ra_rot, dec_rot)
}

impl ConstraintEvaluator for BoresightOffsetEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let (offset_ra, offset_dec) =
            rotate_radec_with_matrix(target_ra, target_dec, &self.rotation_matrix);
        self.constraint
            .evaluate(ephemeris, offset_ra, offset_dec, time_indices)
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

        let (offset_ras, offset_decs) = self.rotate_targets(target_ras, target_decs);

        self.constraint
            .in_constraint_batch(ephemeris, &offset_ras, &offset_decs, time_indices)
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

        let (offset_ras, offset_decs) = self.rotate_targets(target_ras, target_decs);

        self.constraint
            .in_constraint_batch_diagonal(ephemeris, &offset_ras, &offset_decs)
    }

    fn name(&self) -> String {
        format!(
            "BoresightOffset({}, roll={:.3}°, pitch={:.3}°, yaw={:.3}°)",
            self.constraint.name(),
            self.roll_deg,
            self.pitch_deg,
            self.yaw_deg
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
