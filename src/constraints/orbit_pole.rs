/// Orbit pole direction constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::config::EARTH_RADIUS_KM;
use crate::utils::vector_math::radec_to_unit_vectors_batch;
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Orbit Pole constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitPoleConfig {
    /// Minimum allowed angular separation from both orbital poles in degrees
    pub min_angle: f64,
    /// Maximum allowed angular separation from both orbital poles in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angle: Option<f64>,
    /// If true, the pole avoidance angle is calculated as earth_radius_deg + min_angle - 90
    /// This is used for NASA's Neil Gehrels Swift Observatory where the pole is an emergent
    /// property of the Earth size plus the Earth limb avoidance angle being greater than 90°
    #[serde(default)]
    pub earth_limb_pole: bool,
}

impl ConstraintConfig for OrbitPoleConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(OrbitPoleEvaluator {
            min_angle_deg: self.min_angle,
            max_angle_deg: self.max_angle,
            earth_limb_pole: self.earth_limb_pole,
        })
    }
}

/// Evaluator for Orbit Pole constraint
struct OrbitPoleEvaluator {
    min_angle_deg: f64,
    max_angle_deg: Option<f64>,
    earth_limb_pole: bool,
}

impl OrbitPoleEvaluator {
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
            return Err(pyo3::exceptions::PyValueError::new_err("Velocity data not available in ephemeris - orbit pole constraint requires position and velocity data"));
        }
        let max = self.max_angle_deg.map(|v| v.to_radians().cos());
        let epochs: Vec<_> = (0..times.len())
            .map(|t| {
                let i = indices.map_or(t, |rows| rows[t]);
                let p = std::array::from_fn(|j| states[[i, j]]);
                let v = std::array::from_fn(|j| states[[i, j + 3]]);
                let (pole, _) = self.calculate_orbital_poles(&p, &v);
                let min = if self.earth_limb_pole {
                    let distance = crate::utils::vector_math::vector_magnitude(&p);
                    (EARTH_RADIUS_KM / distance).min(1.0).asin().to_degrees() + self.min_angle_deg
                        - 90.0
                } else {
                    self.min_angle_deg
                };
                (pole, min.to_radians().cos(), max)
            })
            .collect();
        super::gpu::angular_grid(targets, &epochs, super::gpu::Angle::NearestPole)
    }

    fn format_name(&self) -> String {
        let mut parts = vec![];

        if self.earth_limb_pole {
            parts.push("earth_limb_pole".to_string());
        }

        match self.max_angle_deg {
            Some(max) => {
                parts.push(format!("min={:.1}°", self.min_angle_deg));
                parts.push(format!("max={:.1}°", max));
            }
            None => {
                parts.push(format!("min={:.1}°", self.min_angle_deg));
            }
        }

        format!("OrbitPoleConstraint({})", parts.join(", "))
    }

    /// Calculate the orbital pole unit vectors (both north and south poles)
    /// Returns both possible normals to the orbital plane
    fn calculate_orbital_poles(
        &self,
        position: &[f64; 3],
        velocity: &[f64; 3],
    ) -> ([f64; 3], [f64; 3]) {
        // Orbital pole is the cross product of position and velocity vectors
        let pole = [
            position[1] * velocity[2] - position[2] * velocity[1],
            position[2] * velocity[0] - position[0] * velocity[2],
            position[0] * velocity[1] - position[1] * velocity[0],
        ];

        // Normalize to unit vector
        let north_pole = crate::utils::vector_math::normalize_vector(&pole);

        // South pole is the negative of north pole
        let south_pole = [-north_pole[0], -north_pole[1], -north_pole[2]];

        (north_pole, south_pole)
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

                // Calculate orbital pole unit vectors (both north and south)
                let (north_pole, south_pole) = self.calculate_orbital_poles(&position, &velocity);

                // Convert target RA/Dec to unit vector
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);

                // Calculate effective minimum angle
                let effective_min_angle = if self.earth_limb_pole {
                    // Calculate Earth angular radius at this time
                    // Angular radius = arcsin(EARTH_RADIUS_KM / distance_from_earth_center)
                    let distance = (position[0] * position[0]
                        + position[1] * position[1]
                        + position[2] * position[2])
                        .sqrt();
                    let ratio = (EARTH_RADIUS_KM / distance).min(1.0);
                    let earth_radius_deg = ratio.asin().to_degrees();
                    earth_radius_deg + self.min_angle_deg - 90.0
                } else {
                    self.min_angle_deg
                };

                // Calculate angular separation to both poles
                let cos_angle_north =
                    crate::utils::vector_math::dot_product(&target_unit, &north_pole);
                let cos_angle_south =
                    crate::utils::vector_math::dot_product(&target_unit, &south_pole);
                let angle_north_deg = cos_angle_north.clamp(-1.0, 1.0).acos().to_degrees();
                let angle_south_deg = cos_angle_south.clamp(-1.0, 1.0).acos().to_degrees();

                // Use the smaller angle (closer pole) for constraint checking
                let min_angle_to_pole = angle_north_deg.min(angle_south_deg);

                // Check constraints
                let mut violated = false;
                let mut severity = 1.0;
                if min_angle_to_pole < effective_min_angle {
                    violated = true;
                    severity = (effective_min_angle - min_angle_to_pole).min(1.0);
                }
                if let Some(max_angle) = self.max_angle_deg {
                    if min_angle_to_pole > max_angle {
                        violated = true;
                        severity = (min_angle_to_pole - max_angle).min(1.0);
                    }
                }

                (violated, severity)
            },
            |violation_start_idx, _is_open| {
                // Use the start of the violation window for calculations
                let gcrs_data = ephemeris.data().gcrs.as_ref().unwrap(); // We already checked this exists
                let position = [
                    gcrs_data[[violation_start_idx, 0]], // x
                    gcrs_data[[violation_start_idx, 1]], // y
                    gcrs_data[[violation_start_idx, 2]], // z
                ];
                let velocity = [
                    gcrs_data[[violation_start_idx, 3]], // vx
                    gcrs_data[[violation_start_idx, 4]], // vy
                    gcrs_data[[violation_start_idx, 5]], // vz
                ];

                let (north_pole, south_pole) = self.calculate_orbital_poles(&position, &velocity);
                let target_unit =
                    crate::utils::vector_math::radec_to_unit_vector(target_ra, target_dec);
                let cos_angle_north =
                    crate::utils::vector_math::dot_product(&target_unit, &north_pole);
                let cos_angle_south =
                    crate::utils::vector_math::dot_product(&target_unit, &south_pole);
                let angle_north_deg = cos_angle_north.clamp(-1.0, 1.0).acos().to_degrees();
                let angle_south_deg = cos_angle_south.clamp(-1.0, 1.0).acos().to_degrees();
                let angle_deg = angle_north_deg.min(angle_south_deg);

                // Calculate effective minimum angle for description
                let effective_min_angle = if self.earth_limb_pole {
                    let distance = (position[0] * position[0]
                        + position[1] * position[1]
                        + position[2] * position[2])
                        .sqrt();
                    let ratio = (EARTH_RADIUS_KM / distance).min(1.0);
                    let earth_radius_deg = ratio.asin().to_degrees();
                    earth_radius_deg + self.min_angle_deg - 90.0
                } else {
                    self.min_angle_deg
                };

                match self.max_angle_deg {
                    Some(max) => format!(
                        "Target angle from orbital pole ({:.1}°) outside allowed range {:.1}°-{:.1}°",
                        angle_deg, effective_min_angle, max
                    ),
                    None => format!(
                        "Target too close to orbital pole ({:.1}° < {:.1}° minimum)",
                        angle_deg, effective_min_angle
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
        let mut orbit_pole_angle_deg = Array2::<f64>::zeros((n_targets, n_times));

        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        let gcrs_data = ephemeris.data().gcrs.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("GCRS data not available in ephemeris")
        })?;
        if gcrs_data.ncols() < 6 {
            return Ok(HashMap::new());
        }

        let mut north_pole_directions = Array2::<f64>::zeros((n_times, 3));
        let mut south_pole_directions = Array2::<f64>::zeros((n_times, 3));
        for i in 0..n_times {
            let source_i = if let Some(indices) = time_indices {
                indices[i]
            } else {
                i
            };
            let position = [
                gcrs_data[[source_i, 0]],
                gcrs_data[[source_i, 1]],
                gcrs_data[[source_i, 2]],
            ];
            let velocity = [
                gcrs_data[[source_i, 3]],
                gcrs_data[[source_i, 4]],
                gcrs_data[[source_i, 5]],
            ];
            let (north_pole, south_pole) = self.calculate_orbital_poles(&position, &velocity);
            north_pole_directions[[i, 0]] = north_pole[0];
            north_pole_directions[[i, 1]] = north_pole[1];
            north_pole_directions[[i, 2]] = north_pole[2];
            south_pole_directions[[i, 0]] = south_pole[0];
            south_pole_directions[[i, 1]] = south_pole[1];
            south_pole_directions[[i, 2]] = south_pole[2];
        }

        for j in 0..n_targets {
            let target_vec = [
                target_vectors[[j, 0]],
                target_vectors[[j, 1]],
                target_vectors[[j, 2]],
            ];
            for i in 0..n_times {
                let north_pole_vec = [
                    north_pole_directions[[i, 0]],
                    north_pole_directions[[i, 1]],
                    north_pole_directions[[i, 2]],
                ];
                let south_pole_vec = [
                    south_pole_directions[[i, 0]],
                    south_pole_directions[[i, 1]],
                    south_pole_directions[[i, 2]],
                ];
                let cos_angle_north =
                    crate::utils::vector_math::dot_product(&target_vec, &north_pole_vec);
                let cos_angle_south =
                    crate::utils::vector_math::dot_product(&target_vec, &south_pole_vec);
                // Nearer pole = larger cosine.
                let max_cos_angle = cos_angle_north.max(cos_angle_south);
                orbit_pole_angle_deg[[j, i]] = max_cos_angle.clamp(-1.0, 1.0).acos().to_degrees();
            }
        }

        let mut values = HashMap::new();
        values.insert("orbit_pole_angle_deg".to_string(), orbit_pole_angle_deg);
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
