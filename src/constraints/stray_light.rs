//! Stray-light constraint: out-of-field light from the Sun, Earth and Moon
//! scattered onto the focal plane, compared with the zodiacal background.
//!
//! The telescope is described by a point-source transmittance (PST) table
//! supplied by the caller: `PST(theta, phi) = E_focal / E_aperture`, the mean
//! focal-plane irradiance produced by a collimated source of unit irradiance
//! arriving from off-axis angle `theta` and azimuth `phi` about the boresight.
//!
//! For each body the model computes
//!
//! 1. its disk-integrated V magnitude at the observer (distance and phase),
//! 2. the brightness-weighted mean of the PST over its resolved disk,
//! 3. their product, the scattered flux in units of a V = 0 source.
//!
//! The sum over the three bodies, times a caller-supplied conversion factor,
//! is the surface brightness (V = 0 sources per square arcsecond) of the
//! uniform sky that would put the same light on the focal plane. The
//! constraint is violated when that exceeds `max_ratio` times the zodiacal
//! surface brightness, or when the model cannot make a prediction.
//!
//! # Pointing frame
//!
//! The table's azimuth is measured about the boresight from a telescope axis
//! `y`: a source in direction `p` has azimuth `atan2(p.x, p.y)` with
//! `x = y x boresight`. The attitude is tied to the Sun. At the nominal
//! attitude the Sun lies in the plane of the boresight and a body axis, so
//! its direction projected perpendicular to the boresight sits at a fixed
//! table azimuth, `sun_azimuth_deg` (0 when the table is measured from the
//! Sun's side, 180 when from the side away from it). Rolling the spacecraft
//! right-handed about the boresight by `roll_offset_deg` adds that angle to
//! the Sun's azimuth.
//!
//! # Disk integral
//!
//! For a body of angular radius `rho` centred on direction `e` the effective
//! transmittance is the brightness-weighted mean of the PST over the disk, by
//! a midpoint rule in (`gamma`, `beta`): `gamma` in (0, `rho`) from the centre
//! and `beta` around it. The weight of a sample is its solid angle times, for
//! the Earth and Moon, the Lambertian `max(n.s, 0)` of the surface point seen
//! in that direction, so the terminator is resolved. The grid is doubled in
//! both directions until two levels agree and at least one sample is sunlit.
//!
//! # Validity
//!
//! A body yields no prediction, and the constraint counts as violated, when
//! its nearest limb is inside the field, when that limb is inside the first
//! cell of the table, or when the quadrature did not converge. A body showing
//! no sunlit surface is valid: its quoted flux is an upper bound on a
//! contribution that is physically negligible.

use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use super::diagonal::DiagonalEvaluation;
use crate::ephemeris::ephemeris_common::EphemerisBase;
use crate::utils::config::{AU_TO_KM, MOON_RADIUS_KM, SUN_RADIUS_KM};
use crate::utils::vector_math::radec_to_unit_vector;
use ndarray::Array2;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type Vec3 = [f64; 3];

/// Mean radius: the Earth is modelled as a Lambert sphere.
const EARTH_MEAN_RADIUS_KM: f64 = 6371.0;
/// Distance at which the Moon's full-phase magnitude is quoted.
const MOON_MEAN_DISTANCE_KM: f64 = 384_400.0;
/// Obliquity of the ecliptic at J2000, degrees.
const OBLIQUITY_J2000_DEG: f64 = 23.439_291_1;

/// Relative agreement of two quadrature levels that counts as converged.
const QUADRATURE_RTOL: f64 = 0.02;
/// Most times the starting grid is doubled.
const MAX_REFINEMENTS: u32 = 4;
/// A disk with no sunlit sample is accepted when the PST varies by less than
/// this (relative) across it: the unresolved crescent's position then matters
/// by less than that.
const UNLIT_SPREAD_TOLERANCE: f64 = 0.1;
/// Fewest cells worth spreading over the cores.
const PARALLEL_MIN_CELLS: usize = 4;

pub const VALUE_RATIO: &str = "stray_light_ratio";
pub const VALUE_SUN: &str = "sun_stray_light_ratio";
pub const VALUE_EARTH: &str = "earth_stray_light_ratio";
pub const VALUE_MOON: &str = "moon_stray_light_ratio";
pub const VALUE_ZODI: &str = "zodi_v_mag_arcsec2";
const VALUE_KEYS: [&str; 5] = [VALUE_RATIO, VALUE_SUN, VALUE_EARTH, VALUE_MOON, VALUE_ZODI];

fn default_true() -> bool {
    true
}

fn default_max_ratio() -> f64 {
    3.0
}

fn default_zodi() -> ZodiModel {
    ZodiModel::Constant
}

/// The flat zodiacal reference commonly used by exposure-time calculators.
fn default_zodi_v_mag() -> f64 {
    22.5
}

fn default_in_field_deg() -> f64 {
    0.3
}

/// Measured whole-disk value; the older fact-sheet constant is 0.40.
fn default_earth_albedo() -> f64 {
    0.24
}

fn default_sun_v_mag() -> f64 {
    -26.74
}

fn default_moon_v_mag() -> f64 {
    -12.73
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// A point-source transmittance table: `log10(PST)` on an off-axis angle x
/// azimuth grid, interpolated bilinearly in the logarithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PstTableConfig {
    /// Off-axis (polar) angles of the rows, degrees, strictly increasing.
    pub off_axis_deg: Vec<f64>,
    /// Azimuths of the columns, degrees, strictly increasing.
    pub azimuth_deg: Vec<f64>,
    /// `log10(PST)`, one row per off-axis angle. `None` marks a cell with no
    /// value; it takes the lowest tabulated value.
    pub log10_pst: Vec<Vec<Option<f64>>>,
    /// The table covers azimuths 0-180 degrees of a response that is mirror
    /// symmetric about its azimuth-zero plane, and other azimuths are folded
    /// onto it. Otherwise the columns span a full turn and wrap around.
    #[serde(default = "default_true")]
    pub azimuth_symmetric: bool,
}

/// The zodiacal surface brightness the scattered light is compared with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ZodiModel {
    /// The same value everywhere (`zodi_v_mag_arcsec2`).
    Constant,
    /// Tabulated against ecliptic position relative to the Sun.
    Table,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrayLightConfig {
    pub pst: PstTableConfig,
    /// Turns the summed PST-weighted flux of the bodies (V = 0 sources) into
    /// the surface brightness of the equivalent uniform sky (V = 0 sources
    /// per square arcsecond).
    pub conversion_factor: f64,
    /// Largest allowed ratio of scattered light to the zodiacal background.
    #[serde(default = "default_max_ratio")]
    pub max_ratio: f64,
    /// Roll about the boresight from the nominal attitude, degrees.
    #[serde(default)]
    pub roll_offset_deg: f64,
    /// Table azimuth of the Sun at the nominal attitude, degrees.
    #[serde(default)]
    pub sun_azimuth_deg: f64,
    #[serde(default = "default_zodi")]
    pub zodi: ZodiModel,
    /// Surface brightness of the constant zodi, V mag per square arcsecond.
    #[serde(default = "default_zodi_v_mag")]
    pub zodi_v_mag_arcsec2: f64,
    /// A body whose nearest limb is closer to the boresight than this is in
    /// the field: direct light, not stray light.
    #[serde(default = "default_in_field_deg")]
    pub in_field_deg: f64,
    /// Geometric albedo of the Earth in V.
    #[serde(default = "default_earth_albedo")]
    pub earth_albedo: f64,
    /// V magnitude of the Sun at 1 au.
    #[serde(default = "default_sun_v_mag")]
    pub sun_v_mag_1au: f64,
    /// V magnitude of the full Moon at 384,400 km.
    #[serde(default = "default_moon_v_mag")]
    pub moon_v_mag_full: f64,
}

impl PstTableConfig {
    pub fn validate(&self) -> Result<(), String> {
        let increasing = |axis: &[f64]| {
            axis.iter().all(|x| x.is_finite()) && axis.windows(2).all(|pair| pair[0] < pair[1])
        };
        if self.off_axis_deg.len() < 2 || !increasing(&self.off_axis_deg) {
            return Err(
                "pst.off_axis_deg must hold at least two finite, strictly increasing angles".into(),
            );
        }
        if self.off_axis_deg[0] < 0.0 || *self.off_axis_deg.last().unwrap() > 180.0 {
            return Err("pst.off_axis_deg must lie within [0, 180] degrees".into());
        }
        if self.azimuth_deg.is_empty() || !increasing(&self.azimuth_deg) {
            return Err(
                "pst.azimuth_deg must hold at least one finite, strictly increasing angle".into(),
            );
        }
        let (first, last) = (self.azimuth_deg[0], *self.azimuth_deg.last().unwrap());
        if self.azimuth_symmetric && (first < 0.0 || last > 180.0) {
            return Err(
                "pst.azimuth_deg must lie within [0, 180] degrees for a symmetric table".into(),
            );
        }
        if !self.azimuth_symmetric && last - first >= 360.0 {
            return Err("pst.azimuth_deg must span less than 360 degrees".into());
        }
        if self.log10_pst.len() != self.off_axis_deg.len()
            || self
                .log10_pst
                .iter()
                .any(|row| row.len() != self.azimuth_deg.len())
        {
            return Err(format!(
                "pst.log10_pst must have shape ({}, {}): one row per off-axis angle, one column \
                 per azimuth",
                self.off_axis_deg.len(),
                self.azimuth_deg.len()
            ));
        }
        if !self
            .log10_pst
            .iter()
            .flatten()
            .any(|cell| cell.is_some_and(f64::is_finite))
        {
            return Err("pst.log10_pst holds no finite value".into());
        }
        Ok(())
    }
}

impl StrayLightConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.pst.validate()?;
        let positive = |value: f64| value.is_finite() && value > 0.0;
        if !positive(self.conversion_factor) {
            return Err("conversion_factor must be positive and finite".into());
        }
        if !positive(self.max_ratio) {
            return Err("max_ratio must be positive and finite".into());
        }
        if !positive(self.earth_albedo) {
            return Err("earth_albedo must be positive and finite".into());
        }
        if !self.in_field_deg.is_finite() || self.in_field_deg < 0.0 {
            return Err("in_field_deg must be non-negative and finite".into());
        }
        for (name, value) in [
            ("roll_offset_deg", self.roll_offset_deg),
            ("sun_azimuth_deg", self.sun_azimuth_deg),
            ("zodi_v_mag_arcsec2", self.zodi_v_mag_arcsec2),
            ("sun_v_mag_1au", self.sun_v_mag_1au),
            ("moon_v_mag_full", self.moon_v_mag_full),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        Ok(())
    }
}

impl ConstraintConfig for StrayLightConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        let roll = (self.roll_offset_deg + self.sun_azimuth_deg).to_radians();
        Box::new(StrayLightEvaluator {
            pst: PstTable::new(&self.pst),
            conversion_factor: self.conversion_factor,
            max_ratio: self.max_ratio,
            frame_roll: roll.sin_cos(),
            zodi: self.zodi,
            zodi_v_mag_arcsec2: self.zodi_v_mag_arcsec2,
            in_field_deg: self.in_field_deg,
            // V of the full Earth from 1 au: V_sun - 2.5 log10(p (R / au)^2).
            earth_v_full_1au: self.sun_v_mag_1au
                - 2.5 * (self.earth_albedo * (EARTH_MEAN_RADIUS_KM / AU_TO_KM).powi(2)).log10(),
            sun_v_mag_1au: self.sun_v_mag_1au,
            moon_v_mag_full: self.moon_v_mag_full,
        })
    }
}

// ─── Small vector helpers ─────────────────────────────────────────────────────

#[inline(always)]
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline(always)]
fn scaled(a: Vec3, k: f64) -> Vec3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline(always)]
fn unit(a: Vec3) -> Vec3 {
    scaled(a, 1.0 / norm(a))
}

/// `a + k b`
#[inline(always)]
fn add_scaled(a: Vec3, b: Vec3, k: f64) -> Vec3 {
    [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]]
}

/// Angle between two vectors, degrees.
fn angle_deg(a: Vec3, b: Vec3) -> f64 {
    (dot(a, b) / (norm(a) * norm(b)))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

// ─── PST table ────────────────────────────────────────────────────────────────

/// A strictly increasing interpolation axis.
struct Axis {
    values: Vec<f64>,
    /// `1 / step` when the values are evenly spaced, which avoids a search.
    inverse_step: Option<f64>,
}

impl Axis {
    fn new(values: Vec<f64>) -> Self {
        let step = (values[values.len() - 1] - values[0]) / (values.len() - 1) as f64;
        let even = values
            .windows(2)
            .all(|pair| ((pair[1] - pair[0]) - step).abs() <= 1e-9 * step.abs());
        Self {
            inverse_step: even.then_some(1.0 / step),
            values,
        }
    }

    fn first(&self) -> f64 {
        self.values[0]
    }

    fn last(&self) -> f64 {
        self.values[self.values.len() - 1]
    }

    /// The cell holding `x` (which lies within the axis) and the fraction of
    /// the way across it.
    #[inline]
    fn bracket(&self, x: f64) -> (usize, f64) {
        let top = self.values.len() - 2;
        let low = match self.inverse_step {
            Some(inverse) => (((x - self.values[0]) * inverse) as usize).min(top),
            None => self
                .values
                .partition_point(|&v| v <= x)
                .saturating_sub(1)
                .min(top),
        };
        let (a, b) = (self.values[low], self.values[low + 1]);
        (low, (x - a) / (b - a))
    }
}

struct PstTable {
    off_axis: Axis,
    azimuth: Axis,
    /// Natural log of the PST, row-major, one row per off-axis angle.
    ln_pst: Vec<f64>,
    columns: usize,
    /// The lowest tabulated PST: used for empty cells and beyond the last row.
    floor: f64,
    symmetric: bool,
    /// Off-axis angle of the second row: a limb nearer the boresight than
    /// this lies in the first cell, where the table is not a prediction.
    first_cell_deg: f64,
}

impl PstTable {
    /// `config` must have passed `PstTableConfig::validate`.
    fn new(config: &PstTableConfig) -> Self {
        let floor_log10 = config
            .log10_pst
            .iter()
            .flatten()
            .filter_map(|cell| cell.filter(|value| value.is_finite()))
            .fold(f64::INFINITY, f64::min);
        let mut azimuth = config.azimuth_deg.clone();
        let mut rows: Vec<Vec<f64>> = config
            .log10_pst
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| cell.filter(|v| v.is_finite()).unwrap_or(floor_log10))
                    .collect()
            })
            .collect();
        // A closing column makes every lookup an interior one: a full-turn
        // table wraps onto its first column, and a single-column table is
        // constant in azimuth.
        if !config.azimuth_symmetric || azimuth.len() == 1 {
            let wrap = if config.azimuth_symmetric { 1.0 } else { 360.0 };
            azimuth.push(azimuth[0] + wrap);
            for row in &mut rows {
                row.push(row[0]);
            }
        }
        let columns = azimuth.len();
        Self {
            first_cell_deg: config.off_axis_deg[1],
            off_axis: Axis::new(config.off_axis_deg.clone()),
            azimuth: Axis::new(azimuth),
            ln_pst: rows
                .into_iter()
                .flatten()
                .map(|log10| log10 * std::f64::consts::LN_10)
                .collect(),
            columns,
            floor: 10f64.powf(floor_log10),
            symmetric: config.azimuth_symmetric,
        }
    }

    /// PST toward off-axis angle `polar_deg`, azimuth `azimuth_deg`.
    #[inline]
    fn lookup(&self, polar_deg: f64, azimuth_deg: f64) -> f64 {
        if polar_deg > self.off_axis.last() {
            return self.floor;
        }
        let polar = polar_deg.max(self.off_axis.first());
        let first = self.azimuth.first();
        let azimuth = if self.symmetric {
            let folded = azimuth_deg.abs() % 360.0;
            let folded = if folded > 180.0 {
                360.0 - folded
            } else {
                folded
            };
            folded.clamp(first, self.azimuth.last())
        } else {
            first + (azimuth_deg - first).rem_euclid(360.0)
        };
        let (i, u) = self.off_axis.bracket(polar);
        let (j, v) = self.azimuth.bracket(azimuth);
        let at = |i: usize, j: usize| self.ln_pst[i * self.columns + j];
        ((1.0 - u) * ((1.0 - v) * at(i, j) + v * at(i, j + 1))
            + u * ((1.0 - v) * at(i + 1, j) + v * at(i + 1, j + 1)))
        .exp()
    }
}

// ─── Body brightness ──────────────────────────────────────────────────────────

/// Phase law of a Lambert sphere seen from a finite distance.
///
/// The distant-observer law assumes a whole hemisphere is visible. From
/// distance `d` only the cap within `acos(R / d)` of the sub-observer point is,
/// which hides the dim (gibbous) or the lit (crescent) edge. `phi(x, alpha)`,
/// `x = R / d`, is the irradiance on a plane facing the sphere's centre,
/// normalised so that `phi(0, alpha)` is the distant-observer law:
///
/// `phi = 3 d^2 / (2 pi) * integral over the visible cap of
///        cos(i) cos(e) cos(p) / r^2 dOmega`
///
/// with `i` the solar incidence (Sun at infinity), `e` the emission angle
/// toward the observer, `p` the angle of the ray to the plane normal and `r`
/// the distance from the surface point to the observer. It is tabulated as
/// `log10` on a regular (`x`, `alpha`) grid, one row of `x` at a time on first
/// use, and interpolated bilinearly.
mod lambert {
    use std::sync::OnceLock;

    const X_STEP: f64 = 0.01;
    const X_COUNT: usize = 99; // x = 0 .. 0.98
    const ALPHA_STEP_DEG: f64 = 2.5;
    const ALPHA_COUNT: usize = 73; // alpha = 0 .. 180
    const FLOOR_LOG10: f64 = -9.0;
    /// Midpoint samples in colatitude (twice as many in longitude).
    const SAMPLES: usize = 400;

    type Row = [f64; ALPHA_COUNT];
    static ROWS: [OnceLock<Row>; X_COUNT] = [const { OnceLock::new() }; X_COUNT];

    fn compute_row(index: usize) -> Row {
        let x = index as f64 * X_STEP;
        let n = SAMPLES;
        let theta_max = if index == 0 {
            std::f64::consts::FRAC_PI_2
        } else {
            x.acos()
        };
        // Per colatitude: sin, cos and the geometric factor times sin(theta).
        let rings: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| {
                let theta = (i as f64 + 0.5) / n as f64 * theta_max;
                let (st, ct) = theta.sin_cos();
                let geometry = if index == 0 {
                    // Distant observer: r -> d, cos e = cos(theta), cos p = 1.
                    ct * st
                } else {
                    // Lengths in units of d: sphere radius x, observer at 1.
                    let r2 = 1.0 - 2.0 * x * ct + x * x;
                    (ct - x) * (1.0 - x * ct) / r2 * st / r2
                };
                (st, ct, geometry)
            })
            .collect();
        // Symmetric about the Sun's plane: integrate longitude 0..pi, double.
        let cos_longitude: Vec<f64> = (0..2 * n)
            .map(|k| ((k as f64 + 0.5) / (2 * n) as f64 * std::f64::consts::PI).cos())
            .collect();
        let weight = (theta_max / n as f64) * (std::f64::consts::PI / (2 * n) as f64) * 2.0 * 3.0
            / (2.0 * std::f64::consts::PI);
        std::array::from_fn(|ia| {
            let (sa, ca) = (ia as f64 * ALPHA_STEP_DEG).to_radians().sin_cos();
            let mut sum = 0.0;
            for &(st, ct, geometry) in &rings {
                let (along, across) = (ct * ca, st * sa);
                let mut ring = 0.0;
                for &cp in &cos_longitude {
                    ring += (across * cp + along).max(0.0);
                }
                sum += ring * geometry;
            }
            (weight * sum).max(1e-300).log10().max(FLOOR_LOG10)
        })
    }

    fn row(index: usize) -> &'static Row {
        ROWS[index].get_or_init(|| compute_row(index))
    }

    /// `phi(x = R / d, alpha)`; `x` beyond the grid is clamped.
    pub fn phase(x: f64, alpha_deg: f64) -> f64 {
        let fx = x.clamp(0.0, (X_COUNT - 1) as f64 * X_STEP) / X_STEP;
        let fa = alpha_deg.clamp(0.0, 180.0) / ALPHA_STEP_DEG;
        let i = (fx as usize).min(X_COUNT - 2);
        let j = (fa as usize).min(ALPHA_COUNT - 2);
        let (u, v) = (fx - i as f64, fa - j as f64);
        let (low, high) = (row(i), row(i + 1));
        10f64.powf(
            (1.0 - u) * ((1.0 - v) * low[j] + v * low[j + 1])
                + u * ((1.0 - v) * high[j] + v * high[j + 1]),
        )
    }
}

// ─── Zodiacal light ───────────────────────────────────────────────────────────

/// Zodiacal surface brightness, V mag per square arcsecond of a solar-colour
/// spectrum, against helio-ecliptic longitude `|lambda - lambda_sun|` (rows,
/// 0-180 degrees in steps of 15) and ecliptic latitude `|beta|` (columns, 0-90
/// degrees in steps of 15). From Leinert et al. (1998), as tabulated in the
/// HST WFC3 Instrument Handbook (Table 9.4). `NaN`: within HST's solar
/// avoidance zone, not tabulated.
const ZODI_STEP_DEG: f64 = 15.0;
const NA: f64 = f64::NAN;
#[rustfmt::skip]
const ZODI_TABLE_V: [[f64; 7]; 13] = [
    [NA,   NA,   NA,   NA,   22.6, 23.0, 23.3], // longitude   0
    [NA,   NA,   NA,   NA,   22.6, 23.1, 23.3], //            15
    [NA,   NA,   NA,   22.3, 22.7, 23.1, 23.3], //            30
    [NA,   NA,   22.1, 22.5, 22.9, 23.1, 23.3], //            45
    [21.3, 21.9, 22.4, 22.7, 23.0, 23.2, 23.3], //            60
    [21.7, 22.2, 22.6, 22.9, 23.1, 23.2, 23.3], //            75
    [22.0, 22.3, 22.7, 23.0, 23.2, 23.3, 23.3], //            90
    [22.2, 22.5, 22.9, 23.1, 23.3, 23.3, 23.3], //           105
    [22.4, 22.6, 22.9, 23.2, 23.3, 23.3, 23.3], //           120
    [22.4, 22.6, 22.9, 23.2, 23.3, 23.4, 23.3], //           135
    [22.4, 22.6, 22.9, 23.1, 23.3, 23.4, 23.3], //           150
    [22.3, 22.5, 22.8, 23.0, 23.2, 23.4, 23.3], //           165
    [22.1, 22.4, 22.7, 23.0, 23.2, 23.4, 23.3], //           180
];

/// J2000 ecliptic longitude and latitude (degrees) of a GCRS direction.
fn ecliptic_lon_lat_deg(v: Vec3) -> (f64, f64) {
    let (se, ce) = OBLIQUITY_J2000_DEG.to_radians().sin_cos();
    let (x, y, z) = (v[0], ce * v[1] + se * v[2], -se * v[1] + ce * v[2]);
    (
        y.atan2(x).to_degrees().rem_euclid(360.0),
        (z / norm(v)).clamp(-1.0, 1.0).asin().to_degrees(),
    )
}

/// Tabulated zodiacal surface brightness toward `boresight` with the Sun in
/// direction `sun`. `NaN` where a surrounding table cell is not tabulated:
/// unknown, not bright.
fn zodi_table_v_mag(boresight: Vec3, sun: Vec3) -> f64 {
    let (longitude, latitude) = ecliptic_lon_lat_deg(boresight);
    let from_sun = (longitude - ecliptic_lon_lat_deg(sun).0).rem_euclid(360.0);
    let fx = ((from_sun + 180.0).rem_euclid(360.0) - 180.0).abs() / ZODI_STEP_DEG;
    let fy = latitude.abs() / ZODI_STEP_DEG;
    let i = (fx as usize).min(ZODI_TABLE_V.len() - 2);
    let j = (fy as usize).min(ZODI_TABLE_V[0].len() - 2);
    let (u, v) = (fx - i as f64, fy - j as f64);
    let t = &ZODI_TABLE_V;
    (1.0 - u) * ((1.0 - v) * t[i][j] + v * t[i][j + 1])
        + u * ((1.0 - v) * t[i + 1][j] + v * t[i + 1][j + 1])
}

// ─── Evaluator ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Body {
    Sun,
    Earth,
    Moon,
}

impl Body {
    fn radius_km(self) -> f64 {
        match self {
            Body::Sun => SUN_RADIUS_KM,
            Body::Earth => EARTH_MEAN_RADIUS_KM,
            Body::Moon => MOON_RADIUS_KM,
        }
    }

    /// Starting grid of the disk quadrature: (radial, around the centre).
    fn grid(self) -> (usize, usize) {
        match self {
            Body::Sun => (2, 4),
            Body::Earth => (16, 32),
            Body::Moon => (4, 8),
        }
    }
}

/// The telescope axes on the sky: boresight, and the directions of the
/// table's azimuth-zero axis and of azimuth 90 degrees.
struct Frame {
    boresight: Vec3,
    y: Vec3,
    x: Vec3,
}

/// One level of the disk quadrature.
struct DiskLevel {
    /// Brightness-weighted mean PST.
    mean: f64,
    /// Fraction of the samples that are sunlit.
    lit_fraction: f64,
    /// `(max - min) / mean` of the PST over the samples.
    spread: f64,
}

/// What one body contributes at one pointing and time.
#[derive(Clone, Copy)]
struct Contribution {
    /// Scattered flux in units of a V = 0 source: `10^(-0.4 V) <PST>`.
    flux: f64,
    valid: bool,
}

/// The model's result for one target at one time.
#[derive(Clone, Copy)]
struct Cell {
    /// Scattered light over zodi; `NaN` when there is no valid prediction.
    ratio: f64,
    /// The same for each of the Sun, Earth and Moon alone.
    bodies: [f64; 3],
    zodi_v_mag_arcsec2: f64,
    violated: bool,
}

impl Cell {
    fn value(&self, key: usize) -> f64 {
        match key {
            0 => self.ratio,
            1..=3 => self.bodies[key - 1],
            _ => self.zodi_v_mag_arcsec2,
        }
    }
}

struct StrayLightEvaluator {
    pst: PstTable,
    conversion_factor: f64,
    max_ratio: f64,
    /// (sin, cos) of the Sun's table azimuth: its nominal value plus the
    /// roll offset.
    frame_roll: (f64, f64),
    zodi: ZodiModel,
    zodi_v_mag_arcsec2: f64,
    in_field_deg: f64,
    earth_v_full_1au: f64,
    sun_v_mag_1au: f64,
    moon_v_mag_full: f64,
}

/// The three (N, 6) state tables the model reads, borrowed from an ephemeris.
struct States<'a> {
    observer: &'a Array2<f64>,
    sun: &'a Array2<f64>,
    moon: &'a Array2<f64>,
}

impl<'a> States<'a> {
    fn of(ephemeris: &'a dyn EphemerisBase) -> PyResult<Self> {
        let missing = |what: &str| {
            pyo3::exceptions::PyValueError::new_err(format!("No {what} positions available"))
        };
        let data = ephemeris.data();
        Ok(Self {
            observer: data.gcrs.as_ref().ok_or_else(|| missing("GCRS"))?,
            sun: data.sun_gcrs.as_ref().ok_or_else(|| missing("Sun"))?,
            moon: data.moon_gcrs.as_ref().ok_or_else(|| missing("Moon"))?,
        })
    }

    /// Vectors (km) from the observer to the Sun, Earth and Moon at one row.
    fn bodies(&self, row: usize) -> [Vec3; 3] {
        let observer: Vec3 = std::array::from_fn(|k| self.observer[[row, k]]);
        [
            std::array::from_fn(|k| self.sun[[row, k]] - observer[k]),
            scaled(observer, -1.0),
            std::array::from_fn(|k| self.moon[[row, k]] - observer[k]),
        ]
    }
}

impl StrayLightEvaluator {
    fn format_name(&self) -> String {
        format!("StrayLight(max_ratio={})", self.max_ratio)
    }

    /// Telescope axes for a boresight and the direction of the Sun.
    fn frame(&self, boresight: Vec3, sun: Vec3) -> Frame {
        let sun = unit(sun);
        let across = add_scaled(sun, boresight, -dot(sun, boresight));
        let length = norm(across);
        let sun_side = if length < 1e-6 {
            // Sun on the boresight: the roll is undefined and the pointing is
            // invalid anyway. Any perpendicular axis will do.
            let pole = if boresight[2].abs() < 0.95 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            unit(add_scaled(pole, boresight, -dot(pole, boresight)))
        } else {
            scaled(across, 1.0 / length)
        };
        let (sin_roll, cos_roll) = self.frame_roll;
        let y = add_scaled(
            scaled(sun_side, cos_roll),
            cross(boresight, sun_side),
            sin_roll,
        );
        Frame {
            boresight,
            x: cross(y, boresight),
            y,
        }
    }

    /// PST toward direction `p` (a unit vector).
    #[inline(always)]
    fn pst_toward(&self, frame: &Frame, p: Vec3) -> f64 {
        let polar = dot(p, frame.boresight).clamp(-1.0, 1.0).acos().to_degrees();
        let azimuth = dot(p, frame.x).atan2(dot(p, frame.y)).to_degrees();
        self.pst.lookup(polar, azimuth)
    }

    /// One quadrature level over the disk of a body centred on `centre` (km
    /// from the observer), on an `n_r` x `n_az` midpoint grid. With `lambert`
    /// each sample is weighted by the illumination of the surface point it
    /// sees; otherwise the disk is uniform.
    #[allow(clippy::too_many_arguments)]
    fn disk_level(
        &self,
        frame: &Frame,
        centre: Vec3,
        radius_km: f64,
        sun: Vec3,
        n_r: usize,
        n_az: usize,
        lambert: bool,
    ) -> DiskLevel {
        let distance = norm(centre);
        let e = scaled(centre, 1.0 / distance);
        let rho = (radius_km / distance).clamp(-1.0, 1.0).asin();
        let reference = if e[2].abs() < 0.95 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let u = unit(cross(reference, e));
        let w = cross(e, u);
        let around: Vec<(f64, f64)> = (0..n_az)
            .map(|k| ((k as f64 + 0.5) / n_az as f64 * std::f64::consts::TAU).sin_cos())
            .collect();
        let hidden = distance * distance - radius_km * radius_km;

        let (mut weighted, mut weight_sum) = (0.0, 0.0);
        let (mut uniform, mut uniform_sum) = (0.0, 0.0);
        let (mut lowest, mut highest, mut total) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
        let mut lit = 0usize;
        for i in 0..n_r {
            let gamma = rho * (i as f64 + 0.5) / n_r as f64;
            let (sin_gamma, cos_gamma) = gamma.sin_cos();
            // Distance to the near intersection of the ray with the sphere.
            let reach = distance * cos_gamma;
            let range = reach - (reach * reach - hidden).max(0.0).sqrt();
            for &(sin_beta, cos_beta) in &around {
                let p = add_scaled(
                    add_scaled(scaled(e, cos_gamma), u, sin_gamma * cos_beta),
                    w,
                    sin_gamma * sin_beta,
                );
                let value = self.pst_toward(frame, p);
                let weight = if lambert {
                    let surface = scaled(p, range);
                    let normal = unit(add_scaled(surface, centre, -1.0));
                    let to_sun = unit(add_scaled(sun, surface, -1.0));
                    sin_gamma * dot(normal, to_sun).max(0.0)
                } else {
                    sin_gamma
                };
                if weight > 0.0 {
                    lit += 1;
                }
                weighted += weight * value;
                weight_sum += weight;
                uniform += sin_gamma * value;
                uniform_sum += sin_gamma;
                lowest = lowest.min(value);
                highest = highest.max(value);
                total += value;
            }
        }
        let samples = (n_r * n_az) as f64;
        DiskLevel {
            // With no sunlit sample the uniform disk keeps the value defined.
            mean: if weight_sum > 0.0 {
                weighted / weight_sum
            } else {
                uniform / uniform_sum
            },
            lit_fraction: lit as f64 / samples,
            spread: (highest - lowest) / (total / samples),
        }
    }

    /// Brightness-weighted mean PST over a body's disk, refining the grid
    /// until two levels agree and a sample is sunlit. Returns the last level
    /// and whether the result is converged.
    fn disk_mean(&self, frame: &Frame, body: Body, centre: Vec3, sun: Vec3) -> (DiskLevel, bool) {
        let (n_r, n_az) = body.grid();
        let lambert = body != Body::Sun;
        let level = |k: u32| {
            self.disk_level(
                frame,
                centre,
                body.radius_km(),
                sun,
                n_r << k,
                n_az << k,
                lambert,
            )
        };
        let mut current = level(0);
        let mut unsettled = true;
        for k in 1..=MAX_REFINEMENTS {
            let finer = level(k);
            let change = (finer.mean - current.mean).abs() / finer.mean;
            current = finer;
            // A NaN change (a zero mean) counts as unsettled.
            unsettled = change.is_nan() || change > QUADRATURE_RTOL || current.lit_fraction == 0.0;
            if !unsettled {
                break;
            }
        }
        let converged = if current.lit_fraction == 0.0 {
            // A crescent thinner than the finest grid: the uniform-disk mean
            // stands only if the PST hardly varies across the disk.
            current.spread <= UNLIT_SPREAD_TOLERANCE
        } else {
            !unsettled
        };
        (current, converged)
    }

    /// Disk-integrated V magnitude of a body at `distance_km`, `phase_deg`.
    fn v_mag(&self, body: Body, distance_km: f64, phase_deg: f64) -> f64 {
        match body {
            // Self-luminous: no phase term.
            Body::Sun => self.sun_v_mag_1au + 5.0 * (distance_km / AU_TO_KM).log10(),
            // Lambert sphere at 1 au from the Sun, exact at finite distance.
            Body::Earth => {
                self.earth_v_full_1au + 5.0 * (distance_km / AU_TO_KM).log10()
                    - 2.5 * lambert::phase(EARTH_MEAN_RADIUS_KM / distance_km, phase_deg).log10()
            }
            // Allen's empirical phase law, which reproduces the Moon's strong,
            // non-Lambertian darkening away from full.
            Body::Moon => {
                let a = phase_deg.abs();
                self.moon_v_mag_full
                    + 5.0 * (distance_km / MOON_MEAN_DISTANCE_KM).log10()
                    + 0.026 * a
                    + 4e-9 * a.powi(4)
            }
        }
    }

    fn contribution(&self, frame: &Frame, body: Body, centre: Vec3, sun: Vec3) -> Contribution {
        const INVALID: Contribution = Contribution {
            flux: f64::NAN,
            valid: false,
        };
        let radius = body.radius_km();
        let distance = norm(centre);
        // The observer is inside the body, or its position is not a number.
        if distance.is_nan() || distance <= radius {
            return INVALID;
        }
        let separation = angle_deg(centre, frame.boresight);
        let limb = separation - (radius / distance).asin().to_degrees();
        if limb.is_nan() || limb < self.in_field_deg {
            // Direct light: a pointing violation, not a stray-light level.
            return INVALID;
        }
        // Sun-body-observer angle: 0 full, 180 new.
        let phase = if body == Body::Sun {
            0.0
        } else {
            angle_deg(add_scaled(sun, centre, -1.0), scaled(centre, -1.0))
        };
        let (disk, converged) = self.disk_mean(frame, body, centre, sun);
        // No sunlit surface is visible when the day cap (90 degrees from the
        // subsolar point) and the visible cap (acos(R / d) from the
        // sub-observer point) are disjoint.
        let dark = body != Body::Sun
            && disk.lit_fraction == 0.0
            && phase >= 90.0 + (radius / distance).clamp(0.0, 1.0).acos().to_degrees();
        let valid = dark || (limb >= self.pst.first_cell_deg && converged);
        Contribution {
            flux: 10f64.powf(-0.4 * self.v_mag(body, distance, phase)) * disk.mean,
            valid,
        }
    }

    /// The model for one boresight and the bodies of one time.
    fn cell(&self, boresight: Vec3, bodies: &[Vec3; 3]) -> Cell {
        let sun = bodies[0];
        let frame = self.frame(boresight, sun);
        let parts = [Body::Sun, Body::Earth, Body::Moon]
            .map(|body| self.contribution(&frame, body, bodies[body as usize], sun));
        let zodi_v_mag_arcsec2 = match self.zodi {
            ZodiModel::Constant => self.zodi_v_mag_arcsec2,
            ZodiModel::Table => zodi_table_v_mag(boresight, sun),
        };
        // NaN zodi (not tabulated) makes every ratio NaN: unknown.
        let per_zodi = self.conversion_factor / 10f64.powf(-0.4 * zodi_v_mag_arcsec2);
        let ratio = if parts.iter().all(|part| part.valid) {
            parts.iter().map(|part| part.flux).sum::<f64>() * per_zodi
        } else {
            f64::NAN
        };
        Cell {
            ratio,
            bodies: parts.map(|part| {
                if part.valid {
                    part.flux * per_zodi
                } else {
                    f64::NAN
                }
            }),
            zodi_v_mag_arcsec2,
            // No valid prediction (NaN) counts as violated.
            violated: ratio.is_nan() || ratio > self.max_ratio,
        }
    }

    /// The model on the target x time grid, row-major (targets, then times).
    fn grid(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<(Vec<Cell>, usize)> {
        let states = States::of(ephemeris)?;
        let rows: Vec<usize> = match time_indices {
            Some(indices) => indices.to_vec(),
            None => (0..states.observer.nrows()).collect(),
        };
        let bodies: Vec<[Vec3; 3]> = rows.iter().map(|&row| states.bodies(row)).collect();
        let targets: Vec<Vec3> = target_ras
            .iter()
            .zip(target_decs)
            .map(|(&ra, &dec)| radec_to_unit_vector(ra, dec))
            .collect();
        let n_times = rows.len();
        let one = |index: usize| self.cell(targets[index / n_times], &bodies[index % n_times]);
        let count = targets.len() * n_times;
        let cells = if count >= PARALLEL_MIN_CELLS {
            (0..count).into_par_iter().map(one).collect()
        } else {
            (0..count).map(one).collect()
        };
        Ok((cells, n_times))
    }

    /// The model for target `i` at ephemeris row `indices[i]`.
    fn paired(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        indices: &[usize],
    ) -> PyResult<Vec<Cell>> {
        let states = States::of(ephemeris)?;
        let one = |i: usize| {
            self.cell(
                radec_to_unit_vector(target_ras[i], target_decs[i]),
                &states.bodies(indices[i]),
            )
        };
        Ok(if indices.len() >= PARALLEL_MIN_CELLS {
            (0..indices.len()).into_par_iter().map(one).collect()
        } else {
            (0..indices.len()).map(one).collect()
        })
    }
}

impl ConstraintEvaluator for StrayLightEvaluator {
    fn evaluate_diagonal(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<DiagonalEvaluation> {
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        // One pass yields both the mask and the values.
        let cells = self.paired(ephemeris, target_ras, target_decs, &indices)?;
        let values = VALUE_KEYS
            .iter()
            .enumerate()
            .map(|(key, name)| {
                (
                    name.to_string(),
                    cells.iter().map(|cell| cell.value(key)).collect(),
                )
            })
            .collect();
        Ok(DiagonalEvaluation::leaf(
            &self.name(),
            cells.iter().map(|cell| cell.violated).collect(),
            values,
        ))
    }

    fn evaluate(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times_at_indices(time_indices)?;
        let (cells, _) = self.grid(ephemeris, &[target_ra], &[target_dec], time_indices)?;
        let violations = track_violations(
            &times,
            |i| {
                let cell = &cells[i];
                let severity = if cell.ratio.is_finite() {
                    (cell.ratio / self.max_ratio - 1.0).max(0.0)
                } else {
                    1.0
                };
                (cell.violated, severity)
            },
            |i, _is_open| {
                let ratio = cells[i].ratio;
                if ratio.is_finite() {
                    format!(
                        "Scattered light is {ratio:.2} x the zodiacal background (max: {})",
                        self.max_ratio
                    )
                } else {
                    "No valid stray-light prediction (body in the field, limb inside the first \
                     PST cell, or disk integral not converged)"
                        .to_string()
                }
            },
        );
        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.name(),
            times,
        ))
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        let (cells, n_times) = self.grid(ephemeris, target_ras, target_decs, time_indices)?;
        Ok(Array2::from_shape_fn(
            (target_ras.len(), n_times),
            |(i, j)| cells[i * n_times + j].violated,
        ))
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (cells, n_times) = self.grid(ephemeris, target_ras, target_decs, time_indices)?;
        Ok(VALUE_KEYS
            .iter()
            .enumerate()
            .map(|(key, name)| {
                (
                    name.to_string(),
                    Array2::from_shape_fn((target_ras.len(), n_times), |(i, j)| {
                        cells[i * n_times + j].value(key)
                    }),
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::test_support::TestEphemeris;

    /// A smooth synthetic table: steep near the axis, with an azimuth term.
    fn table(symmetric: bool) -> PstTableConfig {
        let off_axis_deg: Vec<f64> = (0..=60).map(|i| 2.0 * i as f64).collect();
        let step = if symmetric { 15.0 } else { 30.0 };
        let count = if symmetric { 13 } else { 12 };
        let azimuth_deg: Vec<f64> = (0..count).map(|j| step * j as f64).collect();
        let log10_pst = off_axis_deg
            .iter()
            .map(|&theta| {
                azimuth_deg
                    .iter()
                    .map(|&phi| Some(-2.0 - 0.05 * theta + 0.4 * phi.to_radians().cos()))
                    .collect()
            })
            .collect();
        PstTableConfig {
            off_axis_deg,
            azimuth_deg,
            log10_pst,
            azimuth_symmetric: symmetric,
        }
    }

    fn config() -> StrayLightConfig {
        StrayLightConfig {
            pst: table(true),
            conversion_factor: 1e-8,
            max_ratio: 3.0,
            roll_offset_deg: 0.0,
            sun_azimuth_deg: 0.0,
            zodi: ZodiModel::Constant,
            zodi_v_mag_arcsec2: 22.5,
            in_field_deg: 0.3,
            earth_albedo: 0.24,
            sun_v_mag_1au: -26.74,
            moon_v_mag_full: -12.73,
        }
    }

    fn evaluator(config: &StrayLightConfig) -> StrayLightEvaluator {
        let roll = (config.roll_offset_deg + config.sun_azimuth_deg).to_radians();
        StrayLightEvaluator {
            pst: PstTable::new(&config.pst),
            conversion_factor: config.conversion_factor,
            max_ratio: config.max_ratio,
            frame_roll: roll.sin_cos(),
            zodi: config.zodi,
            zodi_v_mag_arcsec2: config.zodi_v_mag_arcsec2,
            in_field_deg: config.in_field_deg,
            earth_v_full_1au: config.sun_v_mag_1au
                - 2.5 * (config.earth_albedo * (EARTH_MEAN_RADIUS_KM / AU_TO_KM).powi(2)).log10(),
            sun_v_mag_1au: config.sun_v_mag_1au,
            moon_v_mag_full: config.moon_v_mag_full,
        }
    }

    #[test]
    fn table_lookup_is_bilinear_in_the_logarithm_and_folds_azimuth() {
        let pst = PstTable::new(&table(true));
        let expect = |theta: f64, phi: f64| 10f64.powf(-2.0 - 0.05 * theta + 0.4 * phi.cos());
        // Grid nodes are exact.
        assert!((pst.lookup(10.0, 30.0) / expect(10.0, 30f64.to_radians()) - 1.0).abs() < 1e-12);
        // Midway in off-axis angle the log is linear, so exact too.
        assert!((pst.lookup(11.0, 30.0) / expect(11.0, 30f64.to_radians()) - 1.0).abs() < 1e-12);
        // Folding: -30, 330 and 390 are 30.
        for azimuth in [-30.0, 330.0, 390.0, -330.0] {
            assert_eq!(pst.lookup(25.0, azimuth), pst.lookup(25.0, 30.0));
        }
        assert_eq!(pst.lookup(25.0, 200.0), pst.lookup(25.0, 160.0));
        // Beyond the last row: the lowest tabulated value.
        let floor = 10f64.powf(-2.0 - 0.05 * 120.0 - 0.4);
        assert!((pst.lookup(150.0, 0.0) / floor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_cells_take_the_floor_and_full_turn_tables_wrap() {
        let mut config = table(true);
        config.log10_pst[5][3] = None;
        config.log10_pst[6][3] = Some(f64::NAN);
        let pst = PstTable::new(&config);
        let floor = 10f64.powf(-2.0 - 0.05 * 120.0 - 0.4);
        assert!((pst.lookup(10.0, 45.0) / floor - 1.0).abs() < 1e-12);
        assert!((pst.lookup(12.0, 45.0) / floor - 1.0).abs() < 1e-12);

        let pst = PstTable::new(&table(false));
        // Columns stop at 330: 345 lies between the last column and the first.
        let mean = (pst.lookup(20.0, 330.0).ln() + pst.lookup(20.0, 0.0).ln()) / 2.0;
        assert!((pst.lookup(20.0, 345.0).ln() - mean).abs() < 1e-12);
        assert_eq!(pst.lookup(20.0, -15.0), pst.lookup(20.0, 345.0));
        assert_ne!(pst.lookup(20.0, 90.0), pst.lookup(20.0, 300.0));
    }

    #[test]
    fn validation_rejects_malformed_tables() {
        assert!(table(true).validate().is_ok());
        let mut bad = table(true);
        bad.off_axis_deg[3] = 1.0;
        assert!(bad.validate().unwrap_err().contains("strictly increasing"));
        let mut bad = table(true);
        bad.log10_pst[0].pop();
        assert!(bad.validate().unwrap_err().contains("shape"));
        let mut bad = table(true);
        bad.azimuth_deg[12] = 200.0;
        assert!(bad.validate().unwrap_err().contains("[0, 180]"));
        let mut bad = config();
        bad.conversion_factor = 0.0;
        assert!(bad.validate().unwrap_err().contains("conversion_factor"));
    }

    #[test]
    fn finite_distance_phase_law_reduces_to_lambert() {
        // Distant observer: Phi(alpha) = (sin a + (pi - a) cos a) / pi.
        for alpha in [0.0_f64, 30.0, 90.0, 150.0] {
            let a = alpha.to_radians();
            let expected = (a.sin() + (std::f64::consts::PI - a) * a.cos()) / std::f64::consts::PI;
            assert!((lambert::phase(0.0, alpha) / expected - 1.0).abs() < 1e-4);
        }
        // Close in, a full disk is brighter than the inverse-square law says
        // (the surface is nearer than the centre), while the visible cap hides
        // the lit limb of a crescent.
        assert!(lambert::phase(0.3, 0.0) > 1.1);
        assert!(lambert::phase(0.3, 150.0) < 0.1 * lambert::phase(0.0, 150.0));
    }

    #[test]
    fn the_sun_sits_at_the_configured_azimuth() {
        let boresight = radec_to_unit_vector(40.0, 20.0);
        let sun = radec_to_unit_vector(130.0, -5.0);
        for (roll, zero) in [(0.0, 0.0), (0.0, 180.0), (25.0, 180.0), (-40.0, 30.0)] {
            let mut config = config();
            config.roll_offset_deg = roll;
            config.sun_azimuth_deg = zero;
            let frame = evaluator(&config).frame(boresight, sun);
            // Orthonormal, right-handed with the boresight.
            assert!(dot(frame.x, frame.y).abs() < 1e-12);
            assert!(dot(frame.y, frame.boresight).abs() < 1e-12);
            assert!((norm(frame.x) - 1.0).abs() < 1e-12);
            let azimuth = dot(sun, frame.x).atan2(dot(sun, frame.y)).to_degrees();
            let expected = roll + zero;
            assert!(((azimuth - expected + 180.0).rem_euclid(360.0) - 180.0).abs() < 1e-9);
        }
    }

    #[test]
    fn a_small_disk_approaches_the_point_source_value() {
        let config = config();
        let evaluator = evaluator(&config);
        let boresight = radec_to_unit_vector(0.0, 0.0);
        let sun = scaled(radec_to_unit_vector(90.0, 0.0), AU_TO_KM);
        let frame = evaluator.frame(boresight, sun);
        // The Sun, half a degree across, 90 degrees off axis.
        let (disk, converged) = evaluator.disk_mean(&frame, Body::Sun, sun, sun);
        assert!(converged);
        assert_eq!(disk.lit_fraction, 1.0);
        let centre = evaluator.pst_toward(&frame, unit(sun));
        assert!((disk.mean / centre - 1.0).abs() < 1e-3);
    }

    #[test]
    fn invalid_predictions_are_violations() {
        let mut ephemeris = TestEphemeris::new(1);
        // Observer 60,000 km from the Earth on -X; Sun on +X; Moon on +Y.
        ephemeris.data.gcrs.as_mut().unwrap()[[0, 0]] = -60_000.0;
        ephemeris.data.sun_gcrs.as_mut().unwrap()[[0, 0]] = AU_TO_KM;
        ephemeris.data.moon_gcrs.as_mut().unwrap()[[0, 1]] = 384_400.0;
        let mut config = config();
        config.max_ratio = 1e30;
        let evaluator = config.to_evaluator();
        // (RA, Dec): clear of everything; on the Earth and the Sun (both +X
        // from here); on the Moon.
        let moon_ra = 384_400f64.atan2(60_000.0).to_degrees();
        let ras = [180.0, 0.0, moon_ra];
        let mask = evaluator
            .in_constraint_batch(&ephemeris, &ras, &[45.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(mask.column(0).to_vec(), vec![false, true, true]);
        let values = evaluator
            .compute_named_values(&ephemeris, &ras, &[45.0, 0.0, 0.0], None)
            .unwrap();
        assert!(values[VALUE_RATIO][[0, 0]].is_finite());
        assert!(values[VALUE_RATIO][[1, 0]].is_nan());
        assert!(values[VALUE_MOON][[2, 0]].is_nan());
        // A body that is in the field does not hide the others' values.
        assert!(values[VALUE_SUN][[2, 0]].is_finite());
        // The total is the sum of its parts.
        let sum: f64 = [VALUE_SUN, VALUE_EARTH, VALUE_MOON]
            .iter()
            .map(|key| values[*key][[0, 0]])
            .sum();
        assert!((values[VALUE_RATIO][[0, 0]] / sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn diagonal_matches_the_grid() {
        let mut ephemeris = TestEphemeris::new(6);
        for row in 0..6 {
            let angle = row as f64 * 0.4;
            ephemeris.data.gcrs.as_mut().unwrap()[[row, 0]] = -80_000.0 * angle.cos();
            ephemeris.data.gcrs.as_mut().unwrap()[[row, 1]] = 80_000.0 * angle.sin();
            ephemeris.data.sun_gcrs.as_mut().unwrap()[[row, 0]] = AU_TO_KM;
            ephemeris.data.moon_gcrs.as_mut().unwrap()[[row, 1]] = 384_400.0;
        }
        let mut config = config();
        config.conversion_factor = 3e-9;
        config.max_ratio = 5e3;
        config.zodi = ZodiModel::Table;
        let evaluator = config.to_evaluator();
        let ras = [200.0, 150.0, 250.0, 180.0, 120.0, 90.0];
        let decs = [40.0, -30.0, 10.0, 70.0, -60.0, 20.0];
        let grid = evaluator
            .compute_named_values(&ephemeris, &ras, &decs, None)
            .unwrap();
        let mask = evaluator
            .in_constraint_batch(&ephemeris, &ras, &decs, None)
            .unwrap();
        let diagonal = evaluator
            .evaluate_diagonal(&ephemeris, &ras, &decs, None)
            .unwrap();
        for i in 0..6 {
            assert_eq!(diagonal.violated[i], mask[[i, i]]);
            for key in VALUE_KEYS {
                let (paired, gridded) = (diagonal.values[key][i], grid[key][[i, i]]);
                assert!(paired == gridded || (paired.is_nan() && gridded.is_nan()));
            }
        }
        assert!(diagonal.violated.iter().any(|&v| v) && diagonal.violated.iter().any(|&v| !v));
    }
}
