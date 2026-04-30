/// Bright star avoidance constraint
///
/// Violated when any catalog star falls within the telescope field of view.
/// FoV is defined either as a circle (radius around boresight) or a polygon
/// in instrument frame coordinates that rotates with spacecraft roll.
use super::core::{ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use rayon::prelude::*;
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

        // For polygon FoV, precompute a bounding-circle threshold: the maximum angular
        // separation at which a star could possibly fall inside the polygon at any roll.
        // Stars outside this circle are skipped with a cheap dot-product test before
        // the more expensive gnomonic projection.
        let bounding_cos = if let FovDefinition::Polygon { ref vertices, .. } = fov {
            let max_dist_deg = vertices
                .iter()
                .map(|&[u, v]| (u * u + v * v).sqrt())
                .fold(0.0f64, f64::max);
            // Add 1° margin to be safe around polygon edges
            Some((max_dist_deg + 1.0).to_radians().cos())
        } else {
            None
        };

        Box::new(BrightStarEvaluator {
            stars,
            fov,
            bounding_cos,
        })
    }
}

struct StarData {
    ra_rad: f64,
    sin_dec: f64,
    cos_dec: f64,
    /// Precomputed ICRS unit vector for dot-product checks
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
            sin_dec,
            cos_dec,
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
    /// Precomputed cos(max_polygon_radius + 1°) for fast star rejection in polygon mode.
    /// None for circular FoV (unused).
    bounding_cos: Option<f64>,
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
        // Robust ray-casting. Guard against near-horizontal edges (vj == vi)
        // to avoid division by zero and treat points on the polygon boundary as inside.
        let mut inside = false;
        let mut j = n - 1;
        const EPS: f64 = 1.0e-12;
        for i in 0..n {
            let ui = vertices[i][0];
            let vi = vertices[i][1];
            let uj = vertices[j][0];
            let vj = vertices[j][1];

            if (vi > v) != (vj > v) {
                let denom = vj - vi;
                if denom.abs() < EPS {
                    // Nearly horizontal segment: skip the crossing test but
                    // still check for boundary-on-segment cases below.
                } else {
                    let x_intersect = (uj - ui) * (v - vi) / denom + ui;
                    // Point lies exactly on the edge -> consider inside
                    if (u - x_intersect).abs() < EPS {
                        return true;
                    }
                    if u < x_intersect {
                        inside = !inside;
                    }
                }
            } else if (v - vi).abs() < EPS {
                // v equals a vertex v coordinate and the segment is (nearly)
                // horizontal: check if u lies between the segment endpoints.
                let min_u = ui.min(uj) - EPS;
                let max_u = ui.max(uj) + EPS;
                if u >= min_u && u <= max_u {
                    return true;
                }
            }

            j = i;
        }

        inside
    }

    /// Gnomonic (tangent-plane) project all stars that pass the bounding-circle filter
    /// into (east, north) offsets relative to the target.
    ///
    /// Returns a small Vec of (east_rad, north_rad) pairs — typically 0-50 entries.
    /// Projections are roll-independent; the result is reused across all roll samples.
    fn nearby_tangent_offsets(
        &self,
        target_unit: [f64; 3],
        target_ra_rad: f64,
        sin_tdec: f64,
        cos_tdec: f64,
    ) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for star in &self.stars {
            // Bounding-circle fast rejection
            if let Some(cos_thresh) = self.bounding_cos {
                let dot = target_unit[0] * star.unit[0]
                    + target_unit[1] * star.unit[1]
                    + target_unit[2] * star.unit[2];
                if dot < cos_thresh {
                    continue;
                }
            }
            // Gnomonic projection (roll-independent)
            let delta_ra = star.ra_rad - target_ra_rad;
            let (sin_dra, cos_dra) = delta_ra.sin_cos();
            let cos_c = sin_tdec * star.sin_dec + cos_tdec * star.cos_dec * cos_dra;
            if cos_c <= 0.0 {
                continue;
            }
            let east = star.cos_dec * sin_dra / cos_c;
            let north = (cos_tdec * star.sin_dec - sin_tdec * star.cos_dec * cos_dra) / cos_c;
            out.push((east, north));
        }
        out
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
                let (sin_tdec, cos_tdec) = target_dec_rad.sin_cos();
                let (sin_tra, cos_tra) = target_ra_rad.sin_cos();
                let target_unit = [cos_tdec * cos_tra, cos_tdec * sin_tra, sin_tdec];

                // Gnomonic-project nearby stars once; then only the cheap rotation
                // (to_instrument) is repeated per roll sample.
                let nearby =
                    self.nearby_tangent_offsets(target_unit, target_ra_rad, sin_tdec, cos_tdec);

                if let Some(&roll) = roll_rad.as_ref() {
                    let (sin_roll, cos_roll) = roll.sin_cos();
                    nearby.iter().any(|&(east, north)| {
                        let (u, v) = Self::to_instrument(east, north, sin_roll, cos_roll);
                        Self::point_in_polygon(u.to_degrees(), v.to_degrees(), vertices)
                    })
                } else {
                    // Sweep N_ROLL_SAMPLES roll angles evenly over [0°, 360°).
                    // Violated only if every roll has at least one star in the FoV
                    // (i.e., no clear roll exists).
                    let roll_step = 2.0 * PI / N_ROLL_SAMPLES as f64;
                    for step in 0..N_ROLL_SAMPLES {
                        let roll = step as f64 * roll_step;
                        let (sin_roll, cos_roll) = roll.sin_cos();
                        let any_blocked = nearby.iter().any(|&(east, north)| {
                            let (u, v) = Self::to_instrument(east, north, sin_roll, cos_roll);
                            Self::point_in_polygon(u.to_degrees(), v.to_degrees(), vertices)
                        });
                        if !any_blocked {
                            return false; // clear roll found
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

        // Bright-star violation is time-invariant: evaluate once per target.
        // Parallelise across targets since each is fully independent.
        let violated: Vec<bool> = target_ras
            .par_iter()
            .zip(target_decs.par_iter())
            .map(|(&ra, &dec)| self.is_violated(ra, dec))
            .collect();

        let mut result = Array2::from_elem((n_targets, n_times), false);
        for (i, &v) in violated.iter().enumerate() {
            if v {
                for j in 0..n_times {
                    result[[i, j]] = true;
                }
            }
        }
        Ok(result)
    }

    /// Hot path for coordinated roll sweep: evaluate polygon at `roll_deg` without
    /// JSON round-trips.  Circle mode and fixed-roll polygon delegate unchanged.
    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: f64,
    ) -> PyResult<Array2<bool>> {
        let vertices = match &self.fov {
            FovDefinition::Polygon {
                vertices,
                roll_rad: None,
            } => vertices,
            _ => return self.in_constraint_batch(ephemeris, target_ras, target_decs, time_indices),
        };

        let all_times = ephemeris.get_times()?;
        let n_times = match time_indices {
            Some(idx) => idx.len(),
            None => all_times.len(),
        };
        let n_targets = target_ras.len();
        let (sin_roll, cos_roll) = roll_deg.to_radians().sin_cos();

        // Bright-star violation is time-invariant; evaluate once per target, parallelise.
        let violated: Vec<bool> = target_ras
            .par_iter()
            .zip(target_decs.par_iter())
            .map(|(&ra_deg, &dec_deg)| {
                let target_ra_rad = ra_deg.to_radians();
                let target_dec_rad = dec_deg.to_radians();
                let sin_tdec = target_dec_rad.sin();
                let cos_tdec = target_dec_rad.cos();
                let (sin_tra, cos_tra) = target_ra_rad.sin_cos();
                let target_unit = [cos_tdec * cos_tra, cos_tdec * sin_tra, sin_tdec];
                let nearby =
                    self.nearby_tangent_offsets(target_unit, target_ra_rad, sin_tdec, cos_tdec);
                nearby.iter().any(|&(east, north)| {
                    let (u, v) = Self::to_instrument(east, north, sin_roll, cos_roll);
                    Self::point_in_polygon(u.to_degrees(), v.to_degrees(), vertices)
                })
            })
            .collect();

        let mut result = Array2::from_elem((n_targets, n_times), false);
        for (i, &v) in violated.iter().enumerate() {
            if v {
                for j in 0..n_times {
                    result[[i, j]] = true;
                }
            }
        }
        Ok(result)
    }

    /// Free-roll polygon mode: the polygon rotates with roll, so the outer
    /// field-of-regard sweep must test every constraint at the same roll step.
    fn is_roll_dependent(&self) -> bool {
        matches!(&self.fov, FovDefinition::Polygon { roll_rad: None, .. })
    }

    /// Evaluate at a specific roll angle supplied by the outer sweep, rather than
    /// running an independent internal roll sweep.
    fn field_of_regard_violated_at_roll(
        &self,
        _ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_unit_vectors: &Array2<f64>,
        _time_index: usize,
        roll_deg: f64,
    ) -> PyResult<Vec<bool>> {
        let n_targets = target_unit_vectors.nrows();

        let (vertices, stored_roll_rad) = match &self.fov {
            FovDefinition::Polygon { vertices, roll_rad } => (vertices, roll_rad),
            FovDefinition::Circle { cos_radius } => {
                // Circle mode is not roll-dependent; evaluate normally.
                return Ok((0..n_targets)
                    .map(|i| {
                        let target_unit = [
                            target_unit_vectors[[i, 0]],
                            target_unit_vectors[[i, 1]],
                            target_unit_vectors[[i, 2]],
                        ];
                        self.any_star_in_circle(target_unit, *cos_radius)
                    })
                    .collect());
            }
        };

        // Use stored fixed roll if set, otherwise use the roll from the outer sweep.
        let effective_roll_rad = stored_roll_rad.unwrap_or_else(|| roll_deg.to_radians());
        let (sin_roll, cos_roll) = effective_roll_rad.sin_cos();

        Ok((0..n_targets)
            .map(|i| {
                let ux = target_unit_vectors[[i, 0]];
                let uy = target_unit_vectors[[i, 1]];
                let uz = target_unit_vectors[[i, 2]];
                let target_ra_rad = uy.atan2(ux);
                let target_dec_rad = uz.clamp(-1.0, 1.0).asin();
                let sin_tdec = target_dec_rad.sin();
                let cos_tdec = target_dec_rad.cos();
                let nearby =
                    self.nearby_tangent_offsets([ux, uy, uz], target_ra_rad, sin_tdec, cos_tdec);
                nearby.iter().any(|&(east, north)| {
                    let (u, v) = Self::to_instrument(east, north, sin_roll, cos_roll);
                    Self::point_in_polygon(u.to_degrees(), v.to_degrees(), vertices)
                })
            })
            .collect())
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
