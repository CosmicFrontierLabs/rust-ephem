use crate::constraints::core::{ConstraintEvaluator, ConstraintResult};
use ndarray::Array2;
use pyo3::PyResult;

pub(super) struct BoresightOffsetEvaluator {
    pub(super) constraint: Box<dyn ConstraintEvaluator>,
    pub(super) roll_deg: f64,
    pub(super) pitch_deg: f64,
    pub(super) yaw_deg: f64,
    pub(super) rotation_matrix: [[f64; 3]; 3],
}

impl BoresightOffsetEvaluator {
    fn rotate_targets(&self, target_ras: &[f64], target_decs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut offset_ras = Vec::with_capacity(target_ras.len());
        let mut offset_decs = Vec::with_capacity(target_decs.len());

        for (&ra, &dec) in target_ras.iter().zip(target_decs.iter()) {
            let (offset_ra, offset_dec) =
                crate::utils::vector_math::rotate_radec_with_matrix(ra, dec, &self.rotation_matrix);
            offset_ras.push(offset_ra);
            offset_decs.push(offset_dec);
        }

        (offset_ras, offset_decs)
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
        let (offset_ra, offset_dec) = crate::utils::vector_math::rotate_radec_with_matrix(
            target_ra,
            target_dec,
            &self.rotation_matrix,
        );
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
