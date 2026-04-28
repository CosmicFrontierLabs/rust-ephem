/// Bright star avoidance constraint
///
/// Violated when any catalog star falls within the telescope field of view.
/// FoV is defined either as a circle (radius around boresight) or a polygon
/// in instrument frame coordinates that rotates with spacecraft roll.
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Number of roll samples when sweeping all rolls (5° resolution)
const N_ROLL_SAMPLES: usize = 72;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrightStarConfig {
    /// Stars to avoid: (ra_deg, dec_deg) pairs
    pub stars: Vec<[f64; 2]>,
    /// Circular FoV radius in degrees (mutually exclusive with fov_polygon)
    pub fov_radius: Option<f64>,
    /// Polygon FoV vertices in instrument frame (u_deg, v_deg), mutually exclusive with fov_radius.
    /// At roll=0, +u points east and +v points north on the sky.
    pub fov_polygon: Option<Vec<[f64; 2]>>,
    /// Roll angle in degrees (position angle of instrument +v from north, east of north).
    /// None means sweep all rolls: violated only if every roll has a star in the FoV.
    pub roll_deg: Option<f64>,
}

impl ConstraintConfig for BrightStarConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        let stars: Vec<StarData> = self
            .stars
            .iter()
            .map(|&[ra, dec]| StarData::new(ra, dec))
            .collect();

        let fov = if let Some(radius) = self.fov_radius {
            FovDefinition::Circle {
                cos_radius: radius.to_radians().cos(),
            }
        } else if let Some(vertices) = &self.fov_polygon {
            FovDefinition::Polygon {
                vertices: vertices.clone(),
                roll_rad: self.roll_deg.map(|r| r.to_radians()),
            }
        } else {
            // Validated at the Python layer; this path should not be reachable
            FovDefinition::Circle { cos_radius: 1.0 }
        };

        Box::new(BrightStarEvaluator { stars, fov })
    }
}

struct StarData {
    ra_rad: f64,
    dec_rad: f64,
    /// Precomputed ICRS unit vector for dot-product checks in circular FoV
    unit: [f64; 3],
}

impl StarData {
    fn new(ra_deg: f64, dec_deg: f64) -> Self {
        let ra_rad = ra_deg.to_radians();
        let dec_rad = dec_deg.to_radians();
        let (sin_dec, cos_dec) = dec_rad.sin_cos();
        let (sin_ra, cos_ra) = ra_rad.sin_cos();
        Self {
            ra_rad,
            dec_rad,
            unit: [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec],
        }
    }
}

enum FovDefinition {
    Circle {
        /// cos(radius): star is inside if dot(target_unit, star_unit) > cos_radius
        cos_radius: f64,
    },
    Polygon {
        /// Vertices in instrument frame (u_deg, v_deg)
        vertices: Vec<[f64; 2]>,
        /// None = sweep all rolls; Some(r) = evaluate at this roll only
        roll_rad: Option<f64>,
    },
}

pub struct BrightStarEvaluator {
    stars: Vec<StarData>,
    fov: FovDefinition,
}

impl BrightStarEvaluator {
    // ── Circle helpers ────────────────────────────────────────────────────────

    fn any_star_in_circle(&self, target_unit: [f64; 3], cos_radius: f64) -> bool {
        for star in &self.stars {
            let cos_sep = target_unit[0] * star.unit[0]
                + target_unit[1] * star.unit[1]
                + target_unit[2] * star.unit[2];
            if cos_sep > cos_radius {
                return true;
            }
        }
        false
    }

    // ── Polygon helpers ───────────────────────────────────────────────────────

    /// Gnomonic (tangent-plane) projection of `star` relative to `target`.
    /// Returns (east_rad, north_rad) or None if the star is on/behind the tangent plane.
    fn gnomonic_project(
        target_ra_rad: f64,
        target_dec_rad: f64,
        star: &StarData,
    ) -> Option<(f64, f64)> {
        let delta_ra = star.ra_rad - target_ra_rad;
        let (sin_tdec, cos_tdec) = (target_dec_rad.sin(), target_dec_rad.cos());
        let (sin_sdec, cos_sdec) = (star.dec_rad.sin(), star.dec_rad.cos());
        let (sin_dra, cos_dra) = delta_ra.sin_cos();

        let cos_c = sin_tdec * sin_sdec + cos_tdec * cos_sdec * cos_dra;
        if cos_c <= 0.0 {
            return None;
        }

        let east = cos_sdec * sin_dra / cos_c;
        let north = (cos_tdec * sin_sdec - sin_tdec * cos_sdec * cos_dra) / cos_c;
        Some((east, north))
    }

    /// Transform tangent-plane (east, north) offsets to instrument (u, v) at the given roll.
    ///
    /// Convention: at roll=0, u=east and v=north.  Roll is position angle of instrument
    /// +v from north, measured east of north.
    ///
    ///   u = east * cos(roll) - north * sin(roll)
    ///   v = east * sin(roll) + north * cos(roll)
    #[inline]
    fn to_instrument(east: f64, north: f64, sin_roll: f64, cos_roll: f64) -> (f64, f64) {
        (
            east * cos_roll - north * sin_roll,
            east * sin_roll + north * cos_roll,
        )
    }

    /// Ray-casting point-in-polygon test for (u, v) against `vertices` (both in degrees).
    fn point_in_polygon(u: f64, v: f64, vertices: &[[f64; 2]]) -> bool {
        let n = vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let ui = vertices[i][0];
            let vi = vertices[i][1];
            let uj = vertices[j][0];
            let vj = vertices[j][1];
            if ((vi > v) != (vj > v)) && (u < (uj - ui) * (v - vi) / (vj - vi) + ui) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// True if any star falls inside `vertices` when the instrument is at `roll_rad`.
    fn any_star_in_polygon_at_roll(
        &self,
        target_ra_rad: f64,
        target_dec_rad: f64,
        vertices: &[[f64; 2]],
        sin_roll: f64,
        cos_roll: f64,
    ) -> bool {
        for star in &self.stars {
            if let Some((east, north)) = Self::gnomonic_project(target_ra_rad, target_dec_rad, star)
            {
                let (u_rad, v_rad) = Self::to_instrument(east, north, sin_roll, cos_roll);
                let u_deg = u_rad.to_degrees();
                let v_deg = v_rad.to_degrees();
                if Self::point_in_polygon(u_deg, v_deg, vertices) {
                    return true;
                }
            }
        }
        false
    }

    // ── Top-level check ───────────────────────────────────────────────────────

    /// Returns true when the constraint is violated for this target.
    fn is_violated(&self, target_ra_deg: f64, target_dec_deg: f64) -> bool {
        let target_ra_rad = target_ra_deg.to_radians();
        let target_dec_rad = target_dec_deg.to_radians();

        match &self.fov {
            FovDefinition::Circle { cos_radius } => {
                let (sin_dec, cos_dec) = target_dec_rad.sin_cos();
                let (sin_ra, cos_ra) = target_ra_rad.sin_cos();
                let target_unit = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
                self.any_star_in_circle(target_unit, *cos_radius)
            }

            FovDefinition::Polygon { vertices, roll_rad } => {
                if let Some(&roll) = roll_rad.as_ref() {
                    let (sin_roll, cos_roll) = roll.sin_cos();
                    self.any_star_in_polygon_at_roll(
                        target_ra_rad,
                        target_dec_rad,
                        vertices,
                        sin_roll,
                        cos_roll,
                    )
                } else {
                    // Sweep N_ROLL_SAMPLES roll angles evenly over [0°, 360°).
                    // Violated only if every roll has at least one star in the FoV
                    // (i.e., no clear roll exists).
                    let roll_step = 2.0 * PI / N_ROLL_SAMPLES as f64;
                    for step in 0..N_ROLL_SAMPLES {
                        let roll = step as f64 * roll_step;
                        let (sin_roll, cos_roll) = roll.sin_cos();
                        if !self.any_star_in_polygon_at_roll(
                            target_ra_rad,
                            target_dec_rad,
                            vertices,
                            sin_roll,
                            cos_roll,
                        ) {
                            return false; // found a clear roll
                        }
                    }
                    true // every roll blocked
                }
            }
        }
    }

    fn format_name(&self) -> String {
        match &self.fov {
            FovDefinition::Circle { cos_radius } => {
                let r = cos_radius.acos().to_degrees();
                format!("BrightStar(fov_radius={r:.3}°, {} stars)", self.stars.len())
            }
            FovDefinition::Polygon { vertices, roll_rad } => match roll_rad {
                Some(r) => format!(
                    "BrightStar(fov_polygon={} vertices, roll={:.1}°, {} stars)",
                    vertices.len(),
                    r.to_degrees(),
                    self.stars.len()
                ),
                None => format!(
                    "BrightStar(fov_polygon={} vertices, any_roll, {} stars)",
                    vertices.len(),
                    self.stars.len()
                ),
            },
        }
    }
}

impl ConstraintEvaluator for BrightStarEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let all_times = ephemeris.get_times()?;
        let times: Vec<DateTime<Utc>> = match time_indices {
            Some(idx) => idx.iter().map(|&i| all_times[i]).collect(),
            None => all_times.to_vec(),
        };

        // The result is time-invariant: compute once.
        let violated = self.is_violated(target_ra, target_dec);

        let violations = super::core::track_violations(
            &times,
            |_i| (violated, if violated { 1.0 } else { 0.0 }),
            |_i, _open| self.format_name(),
        );
        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
            times,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        let all_times = ephemeris.get_times()?;
        let n_times = match time_indices {
            Some(idx) => idx.len(),
            None => all_times.len(),
        };
        let n_targets = target_ras.len();
        let mut result = Array2::from_elem((n_targets, n_times), false);

        for (i, (&ra, &dec)) in target_ras.iter().zip(target_decs.iter()).enumerate() {
            if self.is_violated(ra, dec) {
                for j in 0..n_times {
                    result[[i, j]] = true;
                }
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
