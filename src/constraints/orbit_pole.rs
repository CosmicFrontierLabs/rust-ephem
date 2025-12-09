/// Orbit pole direction constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for Orbit Pole constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitPoleConfig {
    /// Minimum allowed angular separation from orbital pole in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from orbital pole in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
}

impl ConstraintConfig for OrbitPoleConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(OrbitPoleEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
        })
    }
}

/// Evaluator for Orbit Pole constraint
struct OrbitPoleEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
}

impl OrbitPoleEvaluator {
    fn format_name(&self) -> String {
        match self.max_angle_deg {
            Some(max) => format!(
                "OrbitPoleConstraint(min={:.1}°, max={:.1}°)",
                self.min_angle_deg, max
            ),
            None => format!("OrbitPoleConstraint(min={:.1}°)", self.min_angle_deg),
        }
    }

    /// Calculate the orbital pole unit vector (normal to orbital plane)
    fn calculate_orbital_pole(&self, position: &[f64; 3], velocity: &[f64; 3]) -> [f64; 3] {
        // Orbital pole is the cross product of position and velocity vectors
        let pole = [
            position[1] * velocity[2] - position[2] * velocity[1],
            position[2] * velocity[0] - position[0] * velocity[2],
            position[0] * velocity[1] - position[1] * velocity[0],
        ];

        // Normalize to unit vector
        crate::utils::vector_math::normalize_vector(&pole)
    }
}

impl ConstraintEvaluator for OrbitPoleEvaluator {
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

                // Get spacecraft position and velocity
                // Check if ephemeris has velocity data (6 columns: pos + vel)
                let gcrs_data = match ephemeris.data().gcrs.as_ref() {
                    Some(data) => data,
                    None => return (false, 0.0), // No data available
                };

                if gcrs_data.ncols() < 6 {
                    return (false, 0.0); // No velocity data available
                }

                let position = [
                    gcrs_data[[i, 0]], // x
                    gcrs_data[[i, 1]], // y
                    gcrs_data[[i, 2]], // z
                ];

                let velocity = [
                    gcrs_data[[i, 3]], // vx
                    gcrs_data[[i, 4]], // vy
                    gcrs_data[[i, 5]], // vz
                ];

                // Calculate orbital pole unit vector
                let pole_unit = self.calculate_orbital_pole(&position, &velocity);

                // Convert target RA/Dec to unit vector
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);

                // Calculate angular separation
                let cos_angle = crate::utils::vector_math::dot_product(&target_unit, &pole_unit);
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
                let position = [
                    gcrs_data[[i, 0]], // x
                    gcrs_data[[i, 1]], // y
                    gcrs_data[[i, 2]], // z
                ];
                let velocity = [
                    gcrs_data[[i, 3]], // vx
                    gcrs_data[[i, 4]], // vy
                    gcrs_data[[i, 5]], // vz
                ];

                let pole_unit = self.calculate_orbital_pole(&position, &velocity);
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);
                let cos_angle = crate::utils::vector_math::dot_product(&target_unit, &pole_unit);
                let angle_deg = cos_angle.clamp(-1.0, 1.0).acos().to_degrees();

                match self.max_angle_deg {
                    Some(max) => format!(
                        "Target angle from orbital pole ({:.1}°) outside allowed range {:.1}°-{:.1}°",
                        angle_deg, self.min_angle_deg, max
                    ),
                    None => format!(
                        "Target too close to orbital pole ({:.1}° < {:.1}° minimum)",
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
        // Extract and filter time data
        let (times_filtered,) = extract_time_data!(ephemeris, time_indices);

        let n_targets = target_ras.len();
        let n_times = times_filtered.len();
        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        // Convert target coordinates to unit vectors (vectorized)
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        // Get position and velocity data
        let gcrs_data = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("GCRS data not available in ephemeris")
        })?;

        if gcrs_data.ncols() < 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Velocity data not available in ephemeris - orbit pole constraint requires position and velocity data"
            ));
        }

        // Filter data if time indices provided
        let gcrs_filtered = if let Some(indices) = time_indices {
            gcrs_data.select(ndarray::Axis(0), indices)
        } else {
            gcrs_data.clone()
        };

        // Create orbital pole direction vectors for all times
        let mut pole_directions = Array2::<f64>::zeros((n_times, 3));
        for i in 0..n_times {
            let position = [
                gcrs_filtered[[i, 0]], // x
                gcrs_filtered[[i, 1]], // y
                gcrs_filtered[[i, 2]], // z
            ];
            let velocity = [
                gcrs_filtered[[i, 3]], // vx
                gcrs_filtered[[i, 4]], // vy
                gcrs_filtered[[i, 5]], // vz
            ];

            // Calculate orbital pole unit vector
            let pole_unit = self.calculate_orbital_pole(&position, &velocity);
            pole_directions[[i, 0]] = pole_unit[0];
            pole_directions[[i, 1]] = pole_unit[1];
            pole_directions[[i, 2]] = pole_unit[2];
        }

        // Calculate angular separations for all targets and times (vectorized)
        let angles_deg = crate::utils::vector_math::calculate_angular_separations_batch(
            &target_vectors,
            &pole_directions,
        );

        // Check constraints for all targets and times
        for j in 0..n_targets {
            for i in 0..n_times {
                let angle_deg = angles_deg[[j, i]];
                let mut violated = false;

                if angle_deg < self.min_angle_deg {
                    violated = true;
                }
                if let Some(max_angle) = self.max_angle_deg {
                    if angle_deg > max_angle {
                        violated = true;
                    }
                }

                result[[j, i]] = !violated;
            }
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
