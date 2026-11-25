//! Stray-light constraint: out-of-field light from the Sun, Earth and Moon
//! scattered onto the focal plane, compared with the zodiacal background.
//!
//! The telescope is described by a point-source transmittance (PST) table
//! supplied by the caller: `PST(theta, phi) = E_focal / E_aperture`, the mean
//! focal-plane irradiance produced by a collimated source of unit irradiance
//! arriving from off-axis angle `theta` and azimuth `phi` about the boresight.
//!
//! For each body the model computes the light it puts on the focal plane, in
//! units of a V = 0 source: the integral over the body's visible, sunlit
//! surface of radiance times PST, `sum of L(p) PST(p) dOmega` over sky
//! directions `p`. It is reported as two factors,
//!
//! 1. the body's V magnitude at the observer (distance and phase),
//! 2. the brightness-weighted mean of the PST over its disk,
//!
//! whose product `10^(-0.4 V) <PST>` is that integral.
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
//! The same attitude in the library's terms: body +Y points to the Sun's
//! side, which is the solar-optimal spacecraft roll of a +Y panel
//! (`SolarRollConstraint`), and the table's azimuth-zero axis is body +Y
//! turned by `sun_azimuth_deg`. A spacecraft roll replaces the nominal roll
//! and the offset: an explicit one (`target_roll`), or the candidate of a
//! free-roll sweep that another rule of the same tree asks for. With no roll
//! and no such rule, this constraint supplies the nominal attitude itself.
//!
//! # Brightness
//!
//! - **Sun**: a uniform disk of the configured magnitude at 1 au, scaled by
//!   the inverse square of its distance.
//! - **Earth**: a Lambert sphere of the configured geometric albedo `p`. A
//!   surface point at solar incidence `i` has radiance
//!   `(3 p / 2 pi) E_sun cos(i)`, with `E_sun` the Sun's irradiance at the
//!   Earth. The Earth's magnitude is that of its radiance summed over the
//!   sky, `sum of L dOmega`. No factor for the inclination of a ray to the
//!   Earth's direction is applied: the PST is defined per source of unit
//!   irradiance on a plane facing that source, wherever on the sky it is.
//!   (From low orbit the Earth spans about 136 degrees, and such a factor
//!   would cut its light by a third or more.)
//! - **Moon**: Allen's empirical phase law for the magnitude, scaled to the
//!   Moon's distances from the observer and from the Sun, spread over the
//!   disk as a Lambert sphere would spread it.
//!
//! # Surface integral
//!
//! The Earth and Moon are integrated over their surface, not over the sky:
//! over colatitude `theta` from the sub-observer point (out to
//! `acos(R / d)`, the edge of the visible cap) and longitude about it, by a
//! midpoint rule. Each longitude is sampled only between the terminator and
//! the edge of the cap, and only the longitudes that hold sunlit surface are
//! sampled, so a thin crescent is resolved as well as a full disk and the
//! integrand has no kink. The Sun is integrated over its disk in angle from
//! its centre and position angle. The grid is doubled in both directions
//! until two levels agree.
//!
//! The part of the Sun or Moon behind the Earth contributes nothing. A body
//! the Earth's limb crosses is integrated over the part above the limb alone,
//! on a grid laid out about the Earth's centre (angle from it, and position
//! angle across the body), so that every sample is in view and a sliver is
//! resolved as well as a whole disk. Only a body wholly behind the Earth is
//! `OCCULTED`.
//!
//! # Validity
//!
//! A body yields no prediction, and the constraint counts as violated, when
//! its nearest limb is inside the field, when that limb is inside the first
//! cell of the table, or when the quadrature did not converge. A body showing
//! no sunlit surface, or wholly hidden by the Earth, is valid and contributes
//! nothing; neither needs the integral.

use super::core::{
    track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult, EvaluationRoll,
};
use super::diagonal::DiagonalEvaluation;
use super::solar_roll::{default_panel_normal, north_referenced_axes, solar_optimal_roll_deg};
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
        Box::new(StrayLightEvaluator {
            pst: PstTable::new(&self.pst),
            conversion_factor: self.conversion_factor,
            max_ratio: self.max_ratio,
            roll_offset_deg: self.roll_offset_deg,
            sun_azimuth_deg: self.sun_azimuth_deg,
            zodi: self.zodi,
            zodi_v_mag_arcsec2: self.zodi_v_mag_arcsec2,
            in_field_deg: self.in_field_deg,
            earth_albedo: self.earth_albedo,
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

/// Ecliptic longitude of `boresight` from the Sun (0-360 degrees) and its
/// ecliptic latitude: the coordinates the zodiacal light is tabulated in.
fn zodi_coordinates_deg(boresight: Vec3, sun: Vec3) -> (f64, f64) {
    let (longitude, latitude) = ecliptic_lon_lat_deg(boresight);
    (
        (longitude - ecliptic_lon_lat_deg(sun).0).rem_euclid(360.0),
        latitude,
    )
}

/// Tabulated zodiacal surface brightness at the coordinates of
/// `zodi_coordinates_deg`. `NaN` where a surrounding table cell is not
/// tabulated: unknown, not bright.
fn zodi_table_v_mag(helio_longitude_deg: f64, latitude_deg: f64) -> f64 {
    let fx = ((helio_longitude_deg + 180.0).rem_euclid(360.0) - 180.0).abs() / ZODI_STEP_DEG;
    let fy = latitude_deg.abs() / ZODI_STEP_DEG;
    let i = (fx as usize).min(ZODI_TABLE_V.len() - 2);
    let j = (fy as usize).min(ZODI_TABLE_V[0].len() - 2);
    let (u, v) = (fx - i as f64, fy - j as f64);
    let t = &ZODI_TABLE_V;
    (1.0 - u) * ((1.0 - v) * t[i][j] + v * t[i][j + 1])
        + u * ((1.0 - v) * t[i + 1][j] + v * t[i + 1][j + 1])
}

// ─── Evaluator ────────────────────────────────────────────────────────────────

/// Half the range of longitudes (radians, about the sub-observer point, zero
/// on the Sun's side) that hold sunlit surface inside the visible cap of a
/// sphere, for phase angle `alpha` and a cap reaching colatitude `theta_max`.
/// `None` when no sunlit surface is visible: the day side (within 90 degrees
/// of the subsolar point) and the cap do not meet,
/// `alpha >= 90 degrees + theta_max`.
fn lit_half_width(cos_alpha: f64, sin_alpha: f64, theta_max: f64) -> Option<f64> {
    if cos_alpha > 0.0 {
        // The sub-observer point itself is sunlit.
        return Some(std::f64::consts::PI);
    }
    // Sunlit at the edge of the cap where
    // cos_alpha cos(theta_max) + sin_alpha sin(theta_max) cos(phi) > 0.
    let edge = -cos_alpha * theta_max.cos();
    let reach = sin_alpha * theta_max.sin();
    (reach > edge).then(|| (edge / reach).clamp(-1.0, 1.0).acos())
}

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

/// One level of the quadrature over a body.
#[derive(Clone, Copy, Default)]
struct DiskLevel {
    /// Sum of weight x PST over the samples in view.
    weighted: f64,
    /// Sum of the weights, hidden samples included: for the Earth and Moon
    /// `cos(i) dOmega` (steradians), for the Sun `dOmega`.
    weight: f64,
    /// Solid angle (steradians) of the sunlit samples in view.
    lit_solid_angle: f64,
}

impl DiskLevel {
    /// Mean PST over the body's light, hidden parts counting as zero.
    fn mean(&self) -> f64 {
        if self.weight > 0.0 {
            self.weighted / self.weight
        } else {
            0.0
        }
    }
}

/// The Earth as an obstacle on the line of sight to another body.
#[derive(Clone, Copy)]
struct Blocker {
    /// Observer to the Earth's centre, km.
    centre: Vec3,
    /// Squared length of the tangent from the observer to the Earth, km^2.
    tangent2: f64,
}

impl Blocker {
    /// Whether the Earth lies on the ray along unit vector `p` before `range`.
    #[inline(always)]
    fn hides(&self, p: Vec3, range: f64) -> bool {
        let along = dot(p, self.centre);
        if along <= 0.0 {
            return false;
        }
        let discriminant = along * along - self.tangent2;
        discriminant > 0.0 && along - discriminant.sqrt() < range
    }
}

/// How much of a body the Earth hides.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hidden {
    None,
    /// Some lines of sight may meet the Earth: each sample is tested.
    Partly,
    All,
}

/// How the quadrature over a body accounts for the Earth in front of it.
#[derive(Clone, Copy)]
enum Occultation {
    /// The Earth hides none of the body.
    None,
    /// The body lies wholly beyond the Earth's limb, which then cuts it along
    /// a circle on the sky: only the part above the limb is sampled.
    AboveLimb(Vec3),
    /// A body reaching nearer than the limb: each sample is tested.
    Sampled(Blocker),
}

/// Why a body has, or has not, a stray-light prediction. The numeric values
/// are what `<body>_status` reports.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// A converged prediction.
    Ok = 0,
    /// No sunlit surface is visible: no contribution. Valid.
    Dark = 1,
    /// The nearest limb is inside the field: direct light.
    InField = 2,
    /// The nearest limb is inside the first off-axis cell of the table.
    FirstCell = 3,
    /// The quadrature did not converge.
    Unconverged = 4,
    /// The Earth hides the whole body: no contribution. Valid.
    Occulted = 5,
}

impl Status {
    fn is_valid(self) -> bool {
        matches!(self, Status::Ok | Status::Dark | Status::Occulted)
    }
}

/// Everything the model works out for one body at one pointing and time.
/// Fields that were not computed are `NaN`.
#[derive(Clone, Copy)]
struct BodyReport {
    /// Scattered flux in units of a V = 0 source: `10^(-0.4 V) <PST>`.
    flux: f64,
    status: Status,
    /// Magnitude of the whole body, hidden or not; `+inf` for an Earth
    /// showing no sunlit surface.
    v_mag: f64,
    /// Sun-body-observer angle, degrees: 0 full, 180 new. `NaN` for the Sun.
    phase_deg: f64,
    distance_km: f64,
    angular_radius_deg: f64,
    /// Boresight to the body's centre, and to its nearest limb.
    separation_deg: f64,
    limb_deg: f64,
    /// Table azimuth of the centre, in (-180, 180].
    azimuth_deg: f64,
    /// PST toward the centre, and its brightness-weighted mean over the
    /// body's light, parts hidden by the Earth counting as zero.
    pst_center: f64,
    pst_mean: f64,
    /// Fraction of the disk's solid angle that is sunlit and in view.
    lit_fraction: f64,
    /// Relative change of the flux between the last two quadrature levels;
    /// `NaN` when no quadrature was needed.
    quadrature_error: f64,
}

/// Names of the per-body values, in the order `BodyReport::values`
/// yields them.
const BODY_VALUES: [&str; 12] = [
    "v_mag",
    "phase_deg",
    "distance_km",
    "angular_radius_deg",
    "separation_deg",
    "limb_deg",
    "azimuth_deg",
    "pst_center",
    "pst_mean",
    "lit_fraction",
    "quadrature_error",
    "status",
];
const BODY_NAMES: [&str; 3] = ["sun", "earth", "moon"];
/// Values describing the pointing itself, ahead of the per-body ones.
const POINTING_VALUES: [&str; 4] = [
    "roll_deg",
    "helio_ecliptic_longitude_deg",
    "ecliptic_latitude_deg",
    "zodi_table_v_mag_arcsec2",
];

impl BodyReport {
    fn values(&self) -> [f64; 12] {
        [
            self.v_mag,
            self.phase_deg,
            self.distance_km,
            self.angular_radius_deg,
            self.separation_deg,
            self.limb_deg,
            self.azimuth_deg,
            self.pst_center,
            self.pst_mean,
            self.lit_fraction,
            self.quadrature_error,
            self.status as u8 as f64,
        ]
    }
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

/// A `Cell` with everything that went into it.
struct Report {
    cell: Cell,
    bodies: [BodyReport; 3],
    /// The spacecraft roll used: the explicit one, or the nominal one.
    roll_deg: f64,
    /// Ecliptic longitude of the boresight from the Sun (0-360), and latitude.
    helio_ecliptic_longitude_deg: f64,
    ecliptic_latitude_deg: f64,
    /// The tabulated zodi at the pointing, whichever zodi the ratio uses.
    zodi_table_v_mag_arcsec2: f64,
}

struct StrayLightEvaluator {
    pst: PstTable,
    conversion_factor: f64,
    max_ratio: f64,
    /// Roll about the boresight from the nominal attitude, degrees.
    roll_offset_deg: f64,
    /// Table azimuth of the Sun at the nominal attitude, degrees.
    sun_azimuth_deg: f64,
    zodi: ZodiModel,
    zodi_v_mag_arcsec2: f64,
    in_field_deg: f64,
    /// Geometric albedo of the Earth.
    earth_albedo: f64,
    sun_v_mag_1au: f64,
    moon_v_mag_full: f64,
}

/// How one evaluation is to be carried out.
#[derive(Clone, Copy)]
struct Mode {
    /// Compute every reported quantity, not only what the mask needs.
    detail: bool,
    /// An explicit spacecraft roll (degrees); `None` for the nominal attitude.
    roll_deg: Option<f64>,
}

impl Mode {
    /// The violation mask at the nominal attitude.
    const MASK: Mode = Mode {
        detail: false,
        roll_deg: None,
    };
    /// Every value at the nominal attitude.
    const VALUES: Mode = Mode {
        detail: true,
        roll_deg: None,
    };

    fn at_roll(self, roll_deg: Option<f64>) -> Mode {
        Mode { roll_deg, ..self }
    }
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

    /// Telescope axes for a boresight, and the spacecraft roll they correspond
    /// to (degrees, 0-360).
    ///
    /// The roll is the library's spacecraft roll: counter-clockwise about the
    /// boresight in the north-referenced body frame (`north_referenced_axes`),
    /// as `SolarRollConstraint` and `BoresightOffsetConstraint` use it. With
    /// `roll = None` it is the Sun-referenced nominal one, body +Y toward the
    /// Sun (the solar-optimal roll of a +Y panel), plus the configured offset.
    /// The table's azimuth-zero axis is body +Y turned by the Sun's nominal
    /// table azimuth.
    fn frame(&self, boresight: Vec3, sun: Vec3, roll_deg: Option<f64>) -> (Frame, f64) {
        let roll_deg = roll_deg.unwrap_or_else(|| {
            solar_optimal_roll_deg(&boresight, &unit(sun), default_panel_normal())
                + self.roll_offset_deg
        });
        let (y_ref, z_ref) = north_referenced_axes(boresight);
        let (sin_roll, cos_roll) = (roll_deg + self.sun_azimuth_deg).to_radians().sin_cos();
        let y = add_scaled(scaled(y_ref, cos_roll), z_ref, sin_roll);
        (
            Frame {
                boresight,
                x: cross(y, boresight),
                y,
            },
            roll_deg.rem_euclid(360.0),
        )
    }

    /// PST toward direction `p` (a unit vector).
    #[inline(always)]
    fn pst_toward(&self, frame: &Frame, p: Vec3) -> f64 {
        let polar = dot(p, frame.boresight).clamp(-1.0, 1.0).acos().to_degrees();
        let azimuth = dot(p, frame.x).atan2(dot(p, frame.y)).to_degrees();
        self.pst.lookup(polar, azimuth)
    }

    /// One quadrature level over the Sun's disk, centred on `centre` (km from
    /// the observer), on an `n_r` x `n_az` midpoint grid in angle from the
    /// centre and position angle. The disk is uniform.
    fn sun_level(
        &self,
        frame: &Frame,
        centre: Vec3,
        n_r: usize,
        n_az: usize,
        blocker: Option<Blocker>,
    ) -> DiskLevel {
        let distance = norm(centre);
        let e = scaled(centre, 1.0 / distance);
        let rho = (SUN_RADIUS_KM / distance).clamp(-1.0, 1.0).asin();
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
        let cell = (rho / n_r as f64) * (std::f64::consts::TAU / n_az as f64);
        let mut level = DiskLevel::default();
        for i in 0..n_r {
            let gamma = rho * (i as f64 + 0.5) / n_r as f64;
            let (sin_gamma, cos_gamma) = gamma.sin_cos();
            let weight = sin_gamma * cell;
            for &(sin_beta, cos_beta) in &around {
                let p = add_scaled(
                    add_scaled(scaled(e, cos_gamma), u, sin_gamma * cos_beta),
                    w,
                    sin_gamma * sin_beta,
                );
                level.weight += weight;
                if blocker.is_some_and(|earth| earth.hides(p, f64::INFINITY)) {
                    continue;
                }
                level.weighted += weight * self.pst_toward(frame, p);
                level.lit_solid_angle += weight;
            }
        }
        level
    }

    /// One quadrature level over the sunlit, visible surface of a sphere
    /// centred on `centre` (km from the observer) and lit from the unit
    /// direction `to_sun`: `n_theta` colatitudes from the sub-observer point
    /// at each of `n_phi` longitudes about it, both by the midpoint rule.
    ///
    /// Longitude zero is the Sun's side. Only the longitudes holding sunlit
    /// surface are sampled, and each only between the terminator and the edge
    /// of the visible cap, so every sample is sunlit. The weight of a sample
    /// is `cos(i) dOmega`: the cosine of the solar incidence times the solid
    /// angle its patch of surface subtends at the observer.
    #[allow(clippy::too_many_arguments)]
    fn cap_level(
        &self,
        frame: &Frame,
        centre: Vec3,
        radius_km: f64,
        to_sun: Vec3,
        n_theta: usize,
        n_phi: usize,
        blocker: Option<Blocker>,
    ) -> DiskLevel {
        let mut level = DiskLevel::default();
        let distance = norm(centre);
        // Unit vector from the body's centre to the observer.
        let toward = scaled(centre, -1.0 / distance);
        let theta_max = (radius_km / distance).clamp(0.0, 1.0).acos();
        // cos and sin of the phase angle, and the Sun's side of the cap.
        let cos_alpha = dot(to_sun, toward);
        let across = add_scaled(to_sun, toward, -cos_alpha);
        let sin_alpha = norm(across);
        let Some(phi_max) = lit_half_width(cos_alpha, sin_alpha, theta_max) else {
            return level;
        };
        let side = if sin_alpha > 1e-12 {
            scaled(across, 1.0 / sin_alpha)
        } else {
            // Full phase: any direction across the line of sight will do.
            let reference = if toward[2].abs() < 0.95 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            unit(cross(reference, toward))
        };
        let third = cross(toward, side);
        let d_phi = 2.0 * phi_max / n_phi as f64;
        let radius2 = radius_km * radius_km;
        for j in 0..n_phi {
            let phi = -phi_max + (j as f64 + 0.5) * d_phi;
            let (sin_phi, cos_phi) = phi.sin_cos();
            // cos(i) = cos_alpha cos(theta) + toward_sun sin(theta).
            let toward_sun = sin_alpha * cos_phi;
            let (low, high) = if cos_alpha > 0.0 {
                if toward_sun < 0.0 {
                    (0.0, theta_max.min((cos_alpha / -toward_sun).atan()))
                } else {
                    (0.0, theta_max)
                }
            } else {
                ((-cos_alpha / toward_sun).atan(), theta_max)
            };
            if high <= low {
                continue;
            }
            let d_theta = (high - low) / n_theta as f64;
            let outward = add_scaled(scaled(side, cos_phi), third, sin_phi);
            for k in 0..n_theta {
                let theta = low + (k as f64 + 0.5) * d_theta;
                let (sin_theta, cos_theta) = theta.sin_cos();
                let normal = add_scaled(scaled(toward, cos_theta), outward, sin_theta);
                let point = add_scaled(centre, normal, radius_km);
                let range = norm(point);
                let cos_emission = (distance * cos_theta - radius_km) / range;
                let solid_angle =
                    cos_emission * radius2 * sin_theta * d_theta * d_phi / (range * range);
                let weight =
                    (cos_alpha * cos_theta + toward_sun * sin_theta).max(0.0) * solid_angle;
                level.weight += weight;
                let p = scaled(point, 1.0 / range);
                if blocker.is_some_and(|earth| earth.hides(p, range)) {
                    continue;
                }
                level.weighted += weight * self.pst_toward(frame, p);
                level.lit_solid_angle += solid_angle;
            }
        }
        level
    }

    /// One quadrature level over the part of the Sun or Moon that shows above
    /// the Earth's limb, for a body lying wholly beyond the limb's distance.
    ///
    /// The grid is laid out about the Earth's centre: `n_psi` angles from it,
    /// between the limb and the far edge of the body, and `n_az` position
    /// angles across the body at each. Every sample is therefore in view, and
    /// a sliver above the limb is sampled as finely as a whole disk: no part
    /// in view can fall between samples. The angles are packed toward the
    /// body's far edge (`psi = far - span s^2`), where its width closes as a
    /// square root.
    ///
    /// The Sun's disk is uniform. A sample on the Moon is weighted by the
    /// cosine of the solar incidence where its line of sight meets the
    /// surface, as in `cap_level`, and counts for nothing on the night side.
    /// `weight` is left at zero: the caller supplies the whole body's.
    #[allow(clippy::too_many_arguments)]
    fn limb_level(
        &self,
        frame: &Frame,
        body: Body,
        centre: Vec3,
        to_sun: Vec3,
        earth: Vec3,
        n_psi: usize,
        n_az: usize,
    ) -> DiskLevel {
        let mut level = DiskLevel::default();
        let (distance, earth_distance) = (norm(centre), norm(earth));
        let radius_km = body.radius_km();
        let toward_earth = scaled(earth, 1.0 / earth_distance);
        let toward_body = scaled(centre, 1.0 / distance);
        let limb = (EARTH_MEAN_RADIUS_KM / earth_distance)
            .clamp(-1.0, 1.0)
            .asin();
        let rho = (radius_km / distance).clamp(-1.0, 1.0).asin();
        let cos_separation = dot(toward_earth, toward_body).clamp(-1.0, 1.0);
        let separation = cos_separation.acos();
        let sin_separation = separation.sin();
        let (near, far) = (limb.max(separation - rho), separation + rho);
        if far <= near || sin_separation < 1e-12 {
            return level;
        }
        let u = unit(add_scaled(toward_body, toward_earth, -cos_separation));
        let w = cross(toward_earth, u);
        let cos_rho = rho.cos();
        let behind2 = distance * distance - radius_km * radius_km;
        for i in 0..n_psi {
            let s = (i as f64 + 0.5) / n_psi as f64;
            let psi = far - (far - near) * s * s;
            let d_psi = 2.0 * (far - near) * s / n_psi as f64;
            let (sin_psi, cos_psi) = psi.sin_cos();
            // Half the body's width in position angle at this distance.
            let half = ((cos_rho - cos_psi * cos_separation) / (sin_psi * sin_separation))
                .clamp(-1.0, 1.0)
                .acos();
            let d_az = 2.0 * half / n_az as f64;
            let solid_angle = sin_psi * d_psi * d_az;
            for j in 0..n_az {
                let (sin_az, cos_az) = (-half + (j as f64 + 0.5) * d_az).sin_cos();
                let p = add_scaled(
                    add_scaled(scaled(toward_earth, cos_psi), u, sin_psi * cos_az),
                    w,
                    sin_psi * sin_az,
                );
                let weight = match body {
                    Body::Sun => solid_angle,
                    _ => {
                        let along = dot(p, centre);
                        let discriminant = along * along - behind2;
                        if discriminant <= 0.0 {
                            continue;
                        }
                        let range = along - discriminant.sqrt();
                        let normal =
                            scaled(add_scaled(scaled(p, range), centre, -1.0), 1.0 / radius_km);
                        let cos_incidence = dot(normal, to_sun);
                        if cos_incidence <= 0.0 {
                            continue;
                        }
                        cos_incidence * solid_angle
                    }
                };
                level.weighted += weight * self.pst_toward(frame, p);
                level.lit_solid_angle += solid_angle;
            }
        }
        level
    }

    /// The quadrature over a body, refined until two levels agree. Returns
    /// the last level, whether it is converged, and the relative change of
    /// the flux between the last two levels.
    ///
    /// `occultation` says how the Earth's shadow on the body is handled. A
    /// partly hidden body that shows no light at one level is refined to the
    /// last: two empty levels are not agreement.
    fn disk_integral(
        &self,
        frame: &Frame,
        body: Body,
        centre: Vec3,
        to_sun: Vec3,
        occultation: Occultation,
    ) -> (DiskLevel, bool, f64) {
        let (n_a, n_b) = body.grid();
        let blocker = match occultation {
            Occultation::Sampled(blocker) => Some(blocker),
            _ => None,
        };
        let whole = |k: u32| match body {
            Body::Sun => self.sun_level(frame, centre, n_a << k, n_b << k, blocker),
            _ => self.cap_level(
                frame,
                centre,
                body.radius_km(),
                to_sun,
                n_a << k,
                n_b << k,
                blocker,
            ),
        };
        let level = |k: u32| match occultation {
            Occultation::AboveLimb(earth) => {
                let mut level =
                    self.limb_level(frame, body, centre, to_sun, earth, n_a << k, n_b << k);
                level.weight = match body {
                    Body::Sun => {
                        let sin_radius = SUN_RADIUS_KM / norm(centre);
                        std::f64::consts::TAU * (1.0 - (1.0 - sin_radius * sin_radius).sqrt())
                    }
                    _ => whole(k).weight,
                };
                level
            }
            _ => whole(k),
        };
        let partly = !matches!(occultation, Occultation::None);
        // What the flux is proportional to: the Earth's own radiance sum is
        // part of its flux; the Sun's and Moon's magnitudes are given.
        let signal = |level: &DiskLevel| match body {
            Body::Earth => level.weighted,
            _ => level.mean(),
        };
        let mut current = level(0);
        let mut change = f64::INFINITY;
        for k in 1..=MAX_REFINEMENTS {
            let finer = level(k);
            let (new, old) = (signal(&finer), signal(&current));
            change = if new == old {
                0.0
            } else {
                (new - old).abs() / new.abs()
            };
            current = finer;
            if change <= QUADRATURE_RTOL && !(partly && new == 0.0) {
                break;
            }
        }
        (current, change <= QUADRATURE_RTOL, change)
    }

    /// How much of a body the Earth hides from the observer.
    fn hidden(body: Body, centre: Vec3, earth: Vec3) -> Hidden {
        if body == Body::Earth {
            return Hidden::None;
        }
        let (distance, earth_distance) = (norm(centre), norm(earth));
        // Inside either body (or a position that is not a number): no geometry.
        let outside = earth_distance > EARTH_MEAN_RADIUS_KM && distance > body.radius_km();
        if !outside {
            return Hidden::None;
        }
        let separation = angle_deg(centre, earth).to_radians();
        let earth_radius = (EARTH_MEAN_RADIUS_KM / earth_distance).asin();
        let radius = (body.radius_km() / distance).asin();
        // A ray that meets the Earth does so within the tangent's length.
        let tangent =
            (earth_distance * earth_distance - EARTH_MEAN_RADIUS_KM * EARTH_MEAN_RADIUS_KM).sqrt();
        if separation >= earth_radius + radius || distance + body.radius_km() <= tangent {
            Hidden::None
        } else if separation + radius <= earth_radius && distance - body.radius_km() >= tangent {
            Hidden::All
        } else {
            Hidden::Partly
        }
    }

    /// `detail` asks for every reported quantity; without it only what the
    /// violation mask needs is computed. `earth` is the vector from the
    /// observer to the Earth's centre.
    fn body_report(
        &self,
        frame: &Frame,
        body: Body,
        centre: Vec3,
        sun: Vec3,
        earth: Vec3,
        detail: bool,
    ) -> BodyReport {
        let radius = body.radius_km();
        let distance = norm(centre);
        let mut report = BodyReport {
            flux: f64::NAN,
            status: Status::InField,
            v_mag: f64::NAN,
            phase_deg: f64::NAN,
            distance_km: distance,
            angular_radius_deg: f64::NAN,
            separation_deg: angle_deg(centre, frame.boresight),
            limb_deg: f64::NAN,
            azimuth_deg: dot(centre, frame.x)
                .atan2(dot(centre, frame.y))
                .to_degrees(),
            pst_center: f64::NAN,
            pst_mean: f64::NAN,
            lit_fraction: f64::NAN,
            quadrature_error: f64::NAN,
        };
        // The observer is inside the body, or its position is not a number.
        if distance.is_nan() || distance <= radius {
            return report;
        }
        let sin_radius = radius / distance;
        report.angular_radius_deg = sin_radius.asin().to_degrees();
        report.limb_deg = report.separation_deg - report.angular_radius_deg;
        if detail {
            report.pst_center = self.pst_toward(frame, scaled(centre, 1.0 / distance));
        }
        // From the body to the Sun: its direction and distance.
        let sunward = add_scaled(sun, centre, -1.0);
        let sun_distance_au = norm(sunward) / AU_TO_KM;
        // Sun-body-observer angle: 0 full, 180 new.
        let mut dark = false;
        match body {
            Body::Sun => {
                report.v_mag = self.sun_v_mag_1au + 5.0 * (distance / AU_TO_KM).log10();
            }
            Body::Earth | Body::Moon => {
                report.phase_deg = angle_deg(sunward, scaled(centre, -1.0));
                let (sin_alpha, cos_alpha) = report.phase_deg.to_radians().sin_cos();
                dark = lit_half_width(cos_alpha, sin_alpha, sin_radius.acos()).is_none();
                if body == Body::Moon {
                    // Allen's empirical phase law, which reproduces the
                    // Moon's strong, non-Lambertian darkening away from full.
                    let a = report.phase_deg;
                    report.v_mag = self.moon_v_mag_full
                        + 5.0 * (distance / MOON_MEAN_DISTANCE_KM).log10()
                        + 5.0 * sun_distance_au.log10()
                        + 0.026 * a
                        + 4e-9 * a.powi(4);
                }
            }
        }
        let nothing = |report: &mut BodyReport, status: Status| {
            report.status = status;
            report.flux = 0.0;
            report.pst_mean = 0.0;
            report.lit_fraction = 0.0;
        };
        // A body wholly behind the Earth sends nothing, wherever it is.
        let hidden = Self::hidden(body, centre, earth);
        if hidden == Hidden::All {
            nothing(&mut report, Status::Occulted);
            return report;
        }
        // Direct light is a pointing violation, not a stray-light level: the
        // integral is only worth its cost when the numbers are asked for.
        let in_field = report.limb_deg < self.in_field_deg;
        if in_field && !detail {
            return report;
        }
        if dark {
            // No sunlit surface is visible: nothing to integrate.
            nothing(&mut report, Status::Dark);
            if body == Body::Earth {
                report.v_mag = f64::INFINITY;
            }
            if in_field {
                report.status = Status::InField;
                report.flux = f64::NAN;
            }
            return report;
        }
        let tangent2 = dot(earth, earth) - EARTH_MEAN_RADIUS_KM * EARTH_MEAN_RADIUS_KM;
        let occultation = if hidden != Hidden::Partly {
            Occultation::None
        } else if (distance - radius) * (distance - radius) >= tangent2 {
            Occultation::AboveLimb(earth)
        } else {
            Occultation::Sampled(Blocker {
                centre: earth,
                tangent2,
            })
        };
        let (disk, converged, error) =
            self.disk_integral(frame, body, centre, unit(sunward), occultation);
        if body == Body::Earth {
            // Radiance (3 p / 2 pi) E_sun cos(i), with E_sun the Sun's
            // irradiance at the Earth, summed over the sky.
            let sun_at_earth = self.sun_v_mag_1au + 5.0 * sun_distance_au.log10();
            let radiance =
                3.0 / std::f64::consts::TAU * self.earth_albedo * 10f64.powf(-0.4 * sun_at_earth);
            report.v_mag = -2.5 * (radiance * disk.weight).log10();
        }
        report.pst_mean = disk.mean();
        let disk_solid_angle =
            std::f64::consts::TAU * (1.0 - (1.0 - sin_radius * sin_radius).sqrt());
        report.lit_fraction = (disk.lit_solid_angle / disk_solid_angle).min(1.0);
        report.quadrature_error = error;
        report.flux = 10f64.powf(-0.4 * report.v_mag) * report.pst_mean;
        report.status = if in_field {
            Status::InField
        } else if hidden == Hidden::Partly && disk.lit_solid_angle == 0.0 {
            match (occultation, body) {
                // The part of the Moon above the limb was sampled throughout
                // and none of it is sunlit: its light is behind the Earth.
                (Occultation::AboveLimb(_), Body::Moon) => Status::Occulted,
                // Part of the body may be in view though no sample saw it.
                _ => Status::Unconverged,
            }
        } else if report.limb_deg < self.pst.first_cell_deg {
            Status::FirstCell
        } else if !converged {
            Status::Unconverged
        } else {
            Status::Ok
        };
        report
    }

    /// The model for one boresight and the bodies of one time.
    fn report(&self, boresight: Vec3, bodies: &[Vec3; 3], mode: Mode) -> Report {
        let detail = mode.detail;
        let sun = bodies[0];
        let (frame, roll_deg) = self.frame(boresight, sun, mode.roll_deg);
        let earth = bodies[Body::Earth as usize];
        let parts = [Body::Sun, Body::Earth, Body::Moon]
            .map(|body| self.body_report(&frame, body, bodies[body as usize], sun, earth, detail));
        let (helio_ecliptic_longitude_deg, ecliptic_latitude_deg) =
            zodi_coordinates_deg(boresight, sun);
        let table_needed = detail || self.zodi == ZodiModel::Table;
        let zodi_table_v_mag_arcsec2 = if table_needed {
            zodi_table_v_mag(helio_ecliptic_longitude_deg, ecliptic_latitude_deg)
        } else {
            f64::NAN
        };
        let zodi_v_mag_arcsec2 = match self.zodi {
            ZodiModel::Constant => self.zodi_v_mag_arcsec2,
            ZodiModel::Table => zodi_table_v_mag_arcsec2,
        };
        // NaN zodi (not tabulated) makes every ratio NaN: unknown.
        let per_zodi = self.conversion_factor / 10f64.powf(-0.4 * zodi_v_mag_arcsec2);
        let ratio = if parts.iter().all(|part| part.status.is_valid()) {
            parts.iter().map(|part| part.flux).sum::<f64>() * per_zodi
        } else {
            f64::NAN
        };
        Report {
            cell: Cell {
                ratio,
                bodies: parts.map(|part| {
                    if part.status.is_valid() {
                        part.flux * per_zodi
                    } else {
                        f64::NAN
                    }
                }),
                zodi_v_mag_arcsec2,
                // No valid prediction (NaN) counts as violated.
                violated: ratio.is_nan() || ratio > self.max_ratio,
            },
            bodies: parts,
            roll_deg,
            helio_ecliptic_longitude_deg,
            ecliptic_latitude_deg,
            zodi_table_v_mag_arcsec2,
        }
    }

    /// Names of the reported values, in the order of `values`.
    fn value_keys() -> Vec<String> {
        let mut keys: Vec<String> = VALUE_KEYS.iter().map(|key| key.to_string()).collect();
        keys.extend(POINTING_VALUES.iter().map(|key| key.to_string()));
        for body in BODY_NAMES {
            keys.extend(BODY_VALUES.iter().map(|key| format!("{body}_{key}")));
        }
        keys
    }

    /// The reported values of a `report` made with `detail`.
    fn values(report: &Report) -> Vec<f64> {
        let cell = &report.cell;
        let mut values = vec![
            cell.ratio,
            cell.bodies[0],
            cell.bodies[1],
            cell.bodies[2],
            cell.zodi_v_mag_arcsec2,
            report.roll_deg,
            report.helio_ecliptic_longitude_deg,
            report.ecliptic_latitude_deg,
            report.zodi_table_v_mag_arcsec2,
        ];
        for body in &report.bodies {
            values.extend(body.values());
        }
        values
    }

    /// The model on the target x time grid, row-major (targets, then times),
    /// keeping `keep(report)` of each cell.
    fn grid<T: Send>(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        mode: Mode,
        keep: impl Fn(&Report) -> T + Sync,
    ) -> PyResult<(Vec<T>, usize)> {
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
        let one = |index: usize| {
            keep(&self.report(targets[index / n_times], &bodies[index % n_times], mode))
        };
        Ok((Self::each(targets.len() * n_times, one), n_times))
    }
}

impl StrayLightEvaluator {
    /// `one(i)` for every `i` below `count`, across the cores.
    fn each<T: Send>(count: usize, one: impl Fn(usize) -> T + Sync) -> Vec<T> {
        if count >= PARALLEL_MIN_CELLS {
            (0..count).into_par_iter().map(&one).collect()
        } else {
            (0..count).map(one).collect()
        }
    }

    /// Paired evaluation of the mask alone, without the work only the values
    /// need.
    fn diagonal_mask(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        rolls: Option<&[Option<f64>]>,
    ) -> PyResult<Vec<bool>> {
        if rolls.is_some_and(|rolls| rolls.len() != target_ras.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_rolls must have the same length as target_ras and target_decs",
            ));
        }
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        let states = States::of(ephemeris)?;
        Ok(Self::each(indices.len(), |i| {
            self.report(
                radec_to_unit_vector(target_ras[i], target_decs[i]),
                &states.bodies(indices[i]),
                Mode::MASK.at_roll(rolls.and_then(|rolls| rolls[i])),
            )
            .cell
            .violated
        }))
    }

    /// The violation mask (targets x times) at spacecraft roll `roll_deg`, or
    /// at the nominal attitude without one.
    fn mask(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: Option<f64>,
    ) -> PyResult<Array2<bool>> {
        let (violated, n_times) = self.grid(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            Mode::MASK.at_roll(roll_deg),
            |report| report.cell.violated,
        )?;
        Ok(
            Array2::from_shape_vec((target_ras.len(), n_times), violated)
                .expect("one cell per target and time"),
        )
    }

    /// Paired evaluation, the mask and the values of each element in one pass:
    /// target `i` at ephemeris row `time_indices[i]`, at spacecraft roll
    /// `rolls[i]` where one is given.
    fn diagonal(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        rolls: Option<&[Option<f64>]>,
    ) -> PyResult<DiagonalEvaluation> {
        let indices =
            super::diagonal::time_indices(ephemeris, target_ras, target_decs, time_indices)?;
        let states = States::of(ephemeris)?;
        let one = |i: usize| {
            let report = self.report(
                radec_to_unit_vector(target_ras[i], target_decs[i]),
                &states.bodies(indices[i]),
                Mode::VALUES.at_roll(rolls.and_then(|rolls| rolls[i])),
            );
            (report.cell.violated, Self::values(&report))
        };
        let cells: Vec<(bool, Vec<f64>)> = Self::each(indices.len(), one);
        let values = Self::value_keys()
            .into_iter()
            .enumerate()
            .map(|(key, name)| (name, cells.iter().map(|(_, values)| values[key]).collect()))
            .collect();
        Ok(DiagonalEvaluation::leaf(
            &self.name(),
            cells.iter().map(|(violated, _)| *violated).collect(),
            values,
        ))
    }
}

impl ConstraintEvaluator for StrayLightEvaluator {
    /// Each cell integrates the response over three resolved disks.
    fn is_expensive(&self) -> bool {
        true
    }

    fn mask_diagonal(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        rolls: Option<&[Option<f64>]>,
    ) -> PyResult<Option<Vec<bool>>> {
        self.diagonal_mask(ephemeris, target_ras, target_decs, time_indices, rolls)
            .map(Some)
    }

    fn evaluate_diagonal(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<DiagonalEvaluation> {
        self.diagonal(ephemeris, target_ras, target_decs, time_indices, None)
    }

    /// An explicit roll is the attitude of its element; the others keep the
    /// nominal one. Every element is independent, so no grouping is needed.
    fn evaluate_diagonal_at_rolls(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        rolls: &[Option<f64>],
    ) -> PyResult<Option<DiagonalEvaluation>> {
        if rolls.len() != target_ras.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target_rolls must have the same length as target_ras and target_decs",
            ));
        }
        self.diagonal(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            Some(rolls),
        )
        .map(Some)
    }

    fn evaluate(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        let times = ephemeris.get_times_at_indices(time_indices)?;
        let (cells, _) = self.grid(
            ephemeris,
            &[target_ra],
            &[target_dec],
            time_indices,
            Mode::MASK,
            |report| report.cell,
        )?;
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
        // No roll: the nominal attitude.
        self.mask(ephemeris, target_ras, target_decs, time_indices, None)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        self.compute_named_values_at_roll(ephemeris, target_ras, target_decs, time_indices, None)
    }

    /// The roll is the attitude, whether it is an explicit spacecraft roll or
    /// the candidate of a sweep that a roll-dependent rule of the same tree
    /// started: every rule of a tree is evaluated at one roll.
    fn in_constraint_batch_at_roll(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        roll_deg: EvaluationRoll,
    ) -> PyResult<Array2<bool>> {
        self.mask(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            Some(roll_deg.degrees()),
        )
    }

    /// Alone, this constraint is not swept: without a roll it supplies its
    /// own attitude, so `is_roll_dependent` stays false. Inside a sweep it is
    /// evaluated at each candidate.
    fn follows_sweep_roll(&self) -> bool {
        true
    }

    fn compute_named_values_at_roll(
        &self,
        ephemeris: &dyn EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
        target_roll: Option<f64>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let (cells, n_times) = self.grid(
            ephemeris,
            target_ras,
            target_decs,
            time_indices,
            Mode::VALUES.at_roll(target_roll),
            Self::values,
        )?;
        Ok(Self::value_keys()
            .into_iter()
            .enumerate()
            .map(|(key, name)| {
                (
                    name,
                    Array2::from_shape_fn((target_ras.len(), n_times), |(i, j)| {
                        cells[i * n_times + j][key]
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
        StrayLightEvaluator {
            pst: PstTable::new(&config.pst),
            conversion_factor: config.conversion_factor,
            max_ratio: config.max_ratio,
            roll_offset_deg: config.roll_offset_deg,
            sun_azimuth_deg: config.sun_azimuth_deg,
            zodi: config.zodi,
            zodi_v_mag_arcsec2: config.zodi_v_mag_arcsec2,
            in_field_deg: config.in_field_deg,
            earth_albedo: config.earth_albedo,
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

    /// The Earth's flux for a constant PST, over (PST x the full-phase flux
    /// of the inverse-square law): the phase law of the model.
    fn earth_phase_law(distance_km: f64, phase_deg: f64) -> f64 {
        let mut config = config();
        config
            .pst
            .log10_pst
            .iter_mut()
            .flatten()
            .for_each(|cell| *cell = Some(-6.0));
        let evaluator = evaluator(&config);
        let boresight = [0.0, 0.0, 1.0];
        let earth = [distance_km, 0.0, 0.0];
        let (sin, cos) = phase_deg.to_radians().sin_cos();
        // Sun-Earth-observer angle `phase_deg`, with the Sun 1 au from the Earth.
        let sun = add_scaled(earth, [-cos, sin, 0.0], AU_TO_KM);
        let (frame, _) = evaluator.frame(boresight, sun, None);
        let report = evaluator.body_report(&frame, Body::Earth, earth, sun, earth, true);
        let full = config.earth_albedo
            * 10f64.powf(-0.4 * config.sun_v_mag_1au)
            * (EARTH_MEAN_RADIUS_KM / distance_km).powi(2);
        report.flux / (1e-6 * full)
    }

    #[test]
    fn earth_flux_is_the_radiance_summed_over_the_sky() {
        // Distant observer: Phi(alpha) = (sin a + (pi - a) cos a) / pi.
        for alpha in [0.0_f64, 30.0, 90.0, 150.0] {
            let a = alpha.to_radians();
            let expected = (a.sin() + (std::f64::consts::PI - a) * a.cos()) / std::f64::consts::PI;
            let model = earth_phase_law(1.0e7, alpha);
            assert!(
                (model / expected - 1.0).abs() < 5e-3,
                "{alpha}: {model} {expected}"
            );
        }
        // Close in at full phase the sum has a closed form: with x = R / d,
        // (3 / 2) d^2 times the integral over the cap of
        // cos(theta) (cos(theta) - x) sin(theta) / (1 - 2 x cos(theta) + x^2)^(3/2).
        for x in [0.2_f64, 0.5, 0.9, 0.93] {
            let n = 200_000;
            let theta_max = x.acos();
            let integral: f64 = (0..n)
                .map(|i| {
                    let (st, ct) = ((i as f64 + 0.5) / n as f64 * theta_max).sin_cos();
                    ct * (ct - x) * st / (1.0 - 2.0 * x * ct + x * x).powf(1.5)
                })
                .sum::<f64>()
                * theta_max
                / n as f64;
            let expected = 3.0 * integral;
            let model = earth_phase_law(EARTH_MEAN_RADIUS_KM / x, 0.0);
            assert!(
                (model / expected - 1.0).abs() < 5e-3,
                "x = {x}: {model} {expected}"
            );
        }
    }

    #[test]
    fn no_sunlit_surface_is_dark_without_an_integral() {
        // From 500 km the cap reaches acos(R / d) = 22 degrees.
        let distance = EARTH_MEAN_RADIUS_KM + 500.0;
        let reach = (EARTH_MEAN_RADIUS_KM / distance).acos().to_degrees();
        assert!(earth_phase_law(distance, 90.0 + reach - 0.5) > 0.0);
        assert_eq!(earth_phase_law(distance, 90.0 + reach + 0.5), 0.0);
        assert!(lit_half_width(-0.5, 0.75f64.sqrt(), 0.1).is_none());
        assert_eq!(lit_half_width(0.1, 0.99, 0.1), Some(std::f64::consts::PI));
    }

    #[test]
    fn the_earth_hides_what_is_behind_it() {
        let earth = [7000.0, 0.0, 0.0];
        let sun_behind = [AU_TO_KM, 0.0, 0.0];
        let sun_clear = [0.0, AU_TO_KM, 0.0];
        assert!(StrayLightEvaluator::hidden(Body::Sun, sun_behind, earth) == Hidden::All);
        assert!(StrayLightEvaluator::hidden(Body::Sun, sun_clear, earth) == Hidden::None);
        // Straddling the limb, 65.5 degrees from the nadir at this height.
        let limb = (EARTH_MEAN_RADIUS_KM / 7000.0).asin();
        let sun_setting = scaled([limb.cos(), limb.sin(), 0.0], AU_TO_KM);
        assert!(StrayLightEvaluator::hidden(Body::Sun, sun_setting, earth) == Hidden::Partly);
        // A Moon nearer than the Earth is in front of it.
        let far_earth = [500_000.0, 0.0, 0.0];
        let moon = [100_000.0, 0.0, 0.0];
        assert!(StrayLightEvaluator::hidden(Body::Moon, moon, far_earth) == Hidden::None);

        let evaluator = evaluator(&config());
        let boresight = [0.0, 0.0, 1.0];
        let (frame, _) = evaluator.frame(boresight, sun_setting, None);
        let half = evaluator.body_report(&frame, Body::Sun, sun_setting, sun_setting, earth, true);
        let whole = evaluator.body_report(
            &frame,
            Body::Sun,
            sun_setting,
            sun_setting,
            scaled(earth, -1.0),
            true,
        );
        assert!(half.status == Status::Ok && whole.status == Status::Ok);
        assert!((half.lit_fraction - 0.5).abs() < 0.02);
        assert!((half.flux / whole.flux - 0.5).abs() < 0.05);
        let gone = evaluator.body_report(&frame, Body::Sun, sun_behind, sun_behind, earth, true);
        assert!(gone.status == Status::Occulted && gone.flux == 0.0);
    }

    #[test]
    fn a_sliver_above_the_limb_is_not_occulted() {
        let earth = [7000.0, 0.0, 0.0];
        let limb = (EARTH_MEAN_RADIUS_KM / 7000.0).asin();
        let rho = (SUN_RADIUS_KM / AU_TO_KM).asin();
        let evaluator = evaluator(&config());
        let boresight = [0.0, 0.0, 1.0];
        let at = |below: f64| {
            // The Sun's centre `below` solar radii under the limb.
            let angle = limb - below * rho;
            let sun = scaled([angle.cos(), angle.sin(), 0.0], AU_TO_KM);
            let (frame, _) = evaluator.frame(boresight, sun, None);
            evaluator.body_report(&frame, Body::Sun, sun, sun, earth, true)
        };
        let whole = at(-1.5);
        assert!(whole.status == Status::Ok && (whole.lit_fraction - 1.0).abs() < 1e-3);
        // The limb is all but straight across half a degree, so the part in
        // view is a segment of the disk.
        for below in [0.5, 0.9, 0.99, 0.999, 0.99999] {
            let segment =
                (f64::acos(below) - below * (1.0 - below * below).sqrt()) / std::f64::consts::PI;
            let sliver = at(below);
            assert!(sliver.status == Status::Ok, "{below}");
            assert!(sliver.flux > 0.0, "{below}");
            assert!(
                (sliver.lit_fraction / segment - 1.0).abs() < 0.03,
                "{below}: {} against {segment}",
                sliver.lit_fraction
            );
            assert!(
                (sliver.flux / whole.flux / segment - 1.0).abs() < 0.05,
                "{below}"
            );
        }
        let gone = at(1.001);
        assert!(gone.status == Status::Occulted && gone.flux == 0.0);
    }

    #[test]
    fn the_sun_sits_at_the_configured_azimuth() {
        let boresight = radec_to_unit_vector(40.0, 20.0);
        let sun = radec_to_unit_vector(130.0, -5.0);
        for (roll, zero) in [(0.0, 0.0), (0.0, 180.0), (25.0, 180.0), (-40.0, 30.0)] {
            let mut config = config();
            config.roll_offset_deg = roll;
            config.sun_azimuth_deg = zero;
            let (frame, _) = evaluator(&config).frame(boresight, sun, None);
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
        let (frame, _) = evaluator.frame(boresight, sun, None);
        // The Sun, half a degree across, 90 degrees off axis.
        let (disk, converged, _) =
            evaluator.disk_integral(&frame, Body::Sun, sun, unit(sun), Occultation::None);
        assert!(converged);
        assert_eq!(disk.lit_solid_angle, disk.weight);
        let centre = evaluator.pst_toward(&frame, unit(sun));
        assert!((disk.mean() / centre - 1.0).abs() < 1e-3);
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
