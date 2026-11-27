/// Orbit RAM direction constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Orbit RAM constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitRamConfig {
    /// Minimum allowed angular separation from RAM direction in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from RAM direction in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for OrbitRamConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(OrbitRamEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }
}

/// Evaluator for Orbit RAM constraint
struct OrbitRamEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl OrbitRamEvaluator {
    fn mask_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        targets: &Array2<f64>,
        indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        if targets.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_unit_vectors must have shape (N, 3)",
            ));
        }
        let (times,) = extract_time_data!(ephemeris, indices);
        let states = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("GCRS data not available in ephemeris")
        })?;
        if states.ncols() < 6 {
            return Err(pyo3::exceptions::PyValueError::new_err("Velocity data not available in ephemeris - orbit RAM constraint requires position and velocity data"));
        }
        let max = self.max_angle_deg.map(|v| v.to_radians().cos());
        let min = self.min_angle_deg.to_radians().cos();
        let epochs: Vec<_> = (0..times.len())
            .map(|t| {
                let i = indices.map_or(t, |rows| rows[t]);
                let v = std::array::from_fn(|j| states[[i, j + 3]]);
                (crate::utils::vector_math::normalize_vector(&v), min, max)
            })
            .collect();
        super::gpu::angular_grid(targets, &epochs, super::gpu::Angle::Direction)
    }

    fn format_name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "OrbitRamConstraint(min={:.1}°, max={:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!("OrbitRamConstraint(min={:.1}°)", self.min_angle_deg),
        }
    }
}

impl ConstraintEvaluator for OrbitRamEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Get filtered times
        let times = ephemeris.get_times().expect("Ephemeris must have times");
        let times_filtered = if let Some(indices) = time_indices {
            indices.iter().map(|&i| times[i]).collect()
        } else {
            times.to_vec()
        };

        let violations = track_violations(
            &times_filtered,
            |i| {
                //let time = &times_filtered[i];

                // Get spacecraft velocity vector (RAM direction)
                // Check if ephemeris has velocity data (6 columns: pos + vel)
                let gcrs_data = match ephemeris.data().gcrs.as_ref() {
                    Some(data) => data,
                    None => return (false, 0.0), // No data available
                };

                if gcrs_data.ncols() < 6 {
                    return (false, 0.0); // No velocity data available
                }

                let velocity = [
                    gcrs_data[[i, 3]], // vx
                    gcrs_data[[i, 4]], // vy
                    gcrs_data[[i, 5]], // vz
                ];

                // Normalize velocity to get unit vector in RAM direction
                let ram_unit = crate::utils::vector_math::normalize_vector(&velocity);

                // Convert target RA/Dec to unit vector
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);

                // Calculate angular separation
                let cos_angle = crate::utils::vector_math::dot_product(&target_unit, &ram_unit);
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

                // Check constraints
                let mut violated = false;
                let mut severity = 1.0;
                if angle_deg < self.min_angle_deg {
                    violated = true;
                    severity = (self.min_angle_deg - angle_deg).min(1.0);
                }
                if let Some(max_angle) = self.max_angle_deg {
                    if angle_deg > max_angle {
                        violated = true;
                        severity = (angle_deg - max_angle).min(1.0);
                    }
                }

                (violated, severity)
            },
            |i, violated| {
                if !violated {
                    return "".to_string();
                }

                //let time = &times_filtered[i];
                let gcrs_data = ephemeris.data().gcrs.as_ref().unwrap(); // We already checked this exists
                let velocity = [
                    gcrs_data[[i, 3]], // vx
                    gcrs_data[[i, 4]], // vy
                    gcrs_data[[i, 5]], // vz
                ];
                let ram_unit = crate::utils::vector_math::normalize_vector(&velocity);
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);
                let cos_angle = crate::utils::vector_math::dot_product(&target_unit, &ram_unit);
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

                match self.max_angle_deg {
                    Some(max) => format!(
                        "Target angle from RAM direction ({:.1}°) outside allowed range {:.1}°-{:.1}°",
                        angle_deg, self.min_angle_deg, max
                    ),
                    None => format!(
                        "Target too close to RAM direction ({:.1}° < {:.1}° minimum)",
                        angle_deg, self.min_angle_deg
                    ),
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
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
        let targets = radec_to_unit_vectors_batch(target_ras, target_decs);
        self.mask_vectors(ephemeris, &targets, time_indices)
    }

    fn in_constraint_batch_unit_vectors(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        time_indices: Option<&[usize]>,
    ) -> PyResult<Option<Array2<bool>>> {
        self.mask_vectors(ephemeris, target_unit_vectors, time_indices)
            .map(Some)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (times_filtered,) = extract_time_data!(ephemeris, time_indices);

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();
        let mut orbit_ram_angle_deg = Array2::<f64>::zeros((n_targets, n_times));

        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        let gcrs_data = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("GCRS data not available in ephemeris")
        })?;
        if gcrs_data.ncols() < 6 {
            return Ok(HashMap::new());
        }

        let mut ram_directions = Array2::<f64>::zeros((n_times, 3));
        for i in 0..n_times {
            let source_i = if let Some(indices) = time_indices {
                indices[i]
            } else {
                i
            };
            let velocity = [
                gcrs_data[[source_i, 3]],
                gcrs_data[[source_i, 4]],
                gcrs_data[[source_i, 5]],
            ];
            let ram_unit = crate::utils::vector_math::normalize_vector(&velocity);
            ram_directions[[i, 0]] = ram_unit[0];
            ram_directions[[i, 1]] = ram_unit[1];
            ram_directions[[i, 2]] = ram_unit[2];
        }

        for j in 0..n_targets {
            let target_vec = [
                target_vectors[[j, 0]],
                target_vectors[[j, 1]],
                target_vectors[[j, 2]],
            ];
            for i in 0..n_times {
                let ram_vec = [
                    ram_directions[[i, 0]],
                    ram_directions[[i, 1]],
                    ram_directions[[i, 2]],
                ];
                let cos_angle = crate::utils::vector_math::dot_product(&target_vec, &ram_vec);
                orbit_ram_angle_deg[[j, i]] = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
            }
        }

        let mut values = HashMap::new();
        values.insert("orbit_ram_angle_deg".to_string(), orbit_ram_angle_deg);
        Ok(values)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn ignores_explicit_roll(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
