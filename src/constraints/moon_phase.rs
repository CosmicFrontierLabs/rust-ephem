/// Moon phase constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::{s, Array2};
use once_cell::sync::Lazy;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sofars::astro::atco13;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::utils::conversions::{convert_frames, Frame};
use crate::utils::geo::ecef_to_geodetic_deg;
use crate::utils::time_utils::datetime_to_jd_utc;
use crate::utils::vector_math::{radec_to_unit_vector, radec_to_unit_vectors_batch};
use crate::utils::{eop_provider, ut1_provider};

// Global cache for moon altitudes to avoid recomputing for same ephemeris
// Cache key: (ephemeris_id, time_hash) -> Vec<f64> altitudes
static MOON_ALTITUDE_CACHE: Lazy<Mutex<HashMap<u64, Arc<Vec<f64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn compute_cache_key(
    times: &[DateTime<Utc>],
    moon_positions: &Array2<f64>,
    gcrs_full: &Array2<f64>,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Hash time range and position checksums for cache key
    if !times.is_empty() {
        times[0].timestamp_millis().hash(&mut hasher);
        times[times.len() - 1].timestamp_millis().hash(&mut hasher);
        times.len().hash(&mut hasher);
    }
    // Hash moon and observer positions as checksums
    if moon_positions.nrows() > 0 {
        moon_positions[[0, 0]].to_bits().hash(&mut hasher);
        moon_positions[[0, 1]].to_bits().hash(&mut hasher);
        moon_positions[[0, 2]].to_bits().hash(&mut hasher);
        if moon_positions.nrows() > 1 {
            let last = moon_positions.nrows() - 1;
            moon_positions[[last, 0]].to_bits().hash(&mut hasher);
            moon_positions[[last, 1]].to_bits().hash(&mut hasher);
            moon_positions[[last, 2]].to_bits().hash(&mut hasher);
        }
    }
    if gcrs_full.nrows() > 0 && gcrs_full.ncols() >= 3 {
        gcrs_full[[0, 0]].to_bits().hash(&mut hasher);
        gcrs_full[[0, 1]].to_bits().hash(&mut hasher);
        gcrs_full[[0, 2]].to_bits().hash(&mut hasher);
        if gcrs_full.nrows() > 1 {
            let last = gcrs_full.nrows() - 1;
            gcrs_full[[last, 0]].to_bits().hash(&mut hasher);
            gcrs_full[[last, 1]].to_bits().hash(&mut hasher);
            gcrs_full[[last, 2]].to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Configuration for Moon phase constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonPhaseConfig {
    /// Maximum allowed Moon illumination fraction (0.0 = new moon, 1.0 = full moon)
    pub max_illumination: f64,
    /// Minimum allowed Moon illumination fraction (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_illumination: Option<f64>,
    /// Minimum allowed Moon distance in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_distance: Option<f64>,
    /// Maximum allowed Moon distance in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_distance: Option<f64>,
    /// Whether to enforce constraint when Moon is below horizon (default: false)
    #[serde(default)]
    pub enforce_when_below_horizon: bool,
    /// Moon visibility requirement: "full" (only when fully above horizon) or "partial" (when any part visible)
    #[serde(default = "default_moon_visibility")]
    pub moon_visibility: String,
}

fn default_moon_visibility() -> String {
    "full".to_string()
}

impl ConstraintConfig for MoonPhaseConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(MoonPhaseEvaluator {
            max_illumination: self.max_illumination,
            min_illumination: self.min_illumination,
            min_distance: self.min_distance,
            max_distance: self.max_distance,
            enforce_when_below_horizon: self.enforce_when_below_horizon,
            moon_visibility: self.moon_visibility.clone(),
        })
    }
}

/// Evaluator for Moon phase constraint
struct MoonPhaseEvaluator {
    max_illumination: f64,
    min_illumination: Option<f64>,
    min_distance: Option<f64>,
    max_distance: Option<f64>,
    enforce_when_below_horizon: bool,
    moon_visibility: String,
}

impl MoonPhaseEvaluator {
    fn format_name(&self) -> String {
        let mut parts = Vec::new();

        match (self.min_illumination, self.max_illumination) {
            (Some(min), max) => parts.push(format!("illum={:.2}-{:.2}", min, max)),
            (None, max) => parts.push(format!("illum≤{:.2}", max)),
        }

        if let Some(min_dist) = self.min_distance {
            if let Some(max_dist) = self.max_distance {
                parts.push(format!("dist={:.1}°-{:.1}°", min_dist, max_dist));
            } else {
                parts.push(format!("dist≥{:.1}°", min_dist));
            }
        } else if let Some(max_dist) = self.max_distance {
            parts.push(format!("dist≤{:.1}°", max_dist));
        }

        if !self.enforce_when_below_horizon {
            parts.push("no-enforce-below-horizon".to_string());
        }

        if self.moon_visibility != "full" {
            parts.push(format!("visibility={}", self.moon_visibility));
        }

        format!("MoonPhaseConstraint({})", parts.join(", "))
    }

    fn compute_moon_unit_vectors(
        &self,
        moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<f64>> {
        if moon_positions.nrows() != observer_positions.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Moon and observer position arrays must have the same number of rows.",
            ));
        }

        // Vectorized: moon_rel = moon_positions - observer_positions
        let moon_rel = moon_positions - observer_positions;

        // Vectorized normalization: compute magnitudes for all rows at once
        let magnitudes = moon_rel
            .mapv(|x| x * x)
            .sum_axis(ndarray::Axis(1))
            .mapv(|x| x.sqrt());

        // Normalize each row by dividing by its magnitude
        let mut moon_units = Array2::<f64>::zeros((moon_positions.nrows(), 3));
        for (i, mag) in magnitudes.iter().enumerate() {
            if *mag > 0.0 {
                for j in 0..3 {
                    moon_units[[i, j]] = moon_rel[[i, j]] / mag;
                }
            }
        }

        Ok(moon_units)
    }

    fn compute_moon_altitudes(
        &self,
        times: &[DateTime<Utc>],
        moon_positions: &Array2<f64>,
        gcrs_full: &Array2<f64>,
    ) -> PyResult<Vec<f64>> {
        if moon_positions.nrows() != gcrs_full.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Moon and observer data must have the same number of rows.",
            ));
        }
        if gcrs_full.ncols() < 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GCRS data must contain position and velocity columns.",
            ));
        }

        // Check cache first
        let cache_key = compute_cache_key(times, moon_positions, gcrs_full);
        if let Ok(cache) = MOON_ALTITUDE_CACHE.lock() {
            if let Some(cached) = cache.get(&cache_key) {
                return Ok((**cached).clone());
            }
        }

        let itrs_data = convert_frames(gcrs_full, times, Frame::GCRS, Frame::ITRS, true);
        let itrs_positions = itrs_data.slice(s![.., 0..3]).to_owned();
        let (lats_deg, lons_deg, heights_km) = ecef_to_geodetic_deg(&itrs_positions);

        // Vectorized: compute all relative positions at once
        let gcrs_positions = gcrs_full.slice(s![.., 0..3]);
        let moon_rel = moon_positions - &gcrs_positions;

        // Vectorized: compute distances for all times at once
        let distances = moon_rel
            .mapv(|x| x * x)
            .sum_axis(ndarray::Axis(1))
            .mapv(|x| x.sqrt());

        // Vectorized: compute RA and Dec for all times
        let ra_rad = ndarray::Zip::from(moon_rel.column(1))
            .and(moon_rel.column(0))
            .map_collect(|&y, &x| y.atan2(x));

        let dec_rad = ndarray::Zip::from(moon_rel.column(2))
            .and(&distances)
            .map_collect(|&z, &dist| {
                if dist > 0.0 {
                    (z / dist).asin()
                } else {
                    -std::f64::consts::FRAC_PI_2
                }
            });

        // Precompute per-index scalars to allow parallel altitude computation
        let lat_rad: Vec<f64> = lats_deg.iter().map(|v| v.to_radians()).collect();
        let lon_rad: Vec<f64> = lons_deg.iter().map(|v| v.to_radians()).collect();
        let height_m: Vec<f64> = heights_km.iter().map(|v| v * 1000.0).collect();

        let altitudes: PyResult<Vec<f64>> = (0..times.len())
            .into_par_iter()
            .map(|i| -> PyResult<f64> {
                if distances[i] == 0.0 {
                    return Ok(-90.0);
                }

                let time = &times[i];
                let (utc1, utc2) = datetime_to_jd_utc(time);
                let dut1 = ut1_provider::get_ut1_utc_offset(time);
                let (xp, yp) = eop_provider::get_polar_motion_rad(time);

                let (_aob, zob, _hob, _dob, _rob, _eo) = atco13(
                    ra_rad[i],
                    dec_rad[i],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    utc1,
                    utc2,
                    dut1,
                    lon_rad[i],
                    lat_rad[i],
                    height_m[i],
                    xp,
                    yp,
                    0.0,
                    0.0,
                    0.0,
                    0.55,
                )
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "SOFA atco13 failed for Moon altitude: {e:?}"
                    ))
                })?;

                Ok((std::f64::consts::FRAC_PI_2 - zob).to_degrees())
            })
            .collect();

        // Cache the computed altitudes
        let altitudes_vec = altitudes?;
        let altitudes_arc = Arc::new(altitudes_vec.clone());
        if let Ok(mut cache) = MOON_ALTITUDE_CACHE.lock() {
            cache.insert(cache_key, altitudes_arc);
        }

        Ok(altitudes_vec)
    }

    /// Check if Moon is sufficiently above horizon based on visibility setting
    fn is_moon_visible(&self, altitude: f64) -> bool {
        match self.moon_visibility.as_str() {
            "full" => altitude >= 0.0,     // Moon center above horizon
            "partial" => altitude >= -0.5, // Allow some portion of Moon to be visible
            _ => altitude >= 0.0,
        }
    }
}

impl ConstraintEvaluator for MoonPhaseEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Extract and filter time data
        let (times_filtered,) = extract_time_data!(ephemeris, time_indices);

        let illuminations = ephemeris.moon_illumination(time_indices)?;

        let moon_positions_all = ephemeris.get_moon_positions()?;
        let observer_positions_all = ephemeris.get_gcrs_positions()?;
        let gcrs_full_all =
            ephemeris.data().gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No GCRS data available.")
            })?;

        let moon_positions = if let Some(indices) = time_indices {
            moon_positions_all.select(ndarray::Axis(0), indices)
        } else {
            moon_positions_all
        };
        let observer_positions = if let Some(indices) = time_indices {
            observer_positions_all.select(ndarray::Axis(0), indices)
        } else {
            observer_positions_all
        };
        let gcrs_full = if let Some(indices) = time_indices {
            gcrs_full_all.select(ndarray::Axis(0), indices)
        } else {
            gcrs_full_all.to_owned()
        };

        let moon_altitudes =
            self.compute_moon_altitudes(&times_filtered, &moon_positions, &gcrs_full)?;
        let moon_units = self.compute_moon_unit_vectors(&moon_positions, &observer_positions)?;
        let target_vec = radec_to_unit_vector(target_ra, target_dec);

        // Vectorized: compute all moon distances at once using dot products
        // moon_units is (n_times, 3), target_vec is (3,)
        // We want dot(moon_units[i], target_vec) for all i
        let target_array = ndarray::Array1::from_vec(target_vec.to_vec());
        let cos_angles = moon_units.dot(&target_array);
        let moon_distances: Vec<f64> = cos_angles
            .iter()
            .map(|&cos_angle| cos_angle.clamp(-1.0, 1.0).acos().to_degrees())
            .collect();

        let violations = track_violations(
            &times_filtered,
            |i| {
                let illumination = illuminations[i];
                let moon_altitude = moon_altitudes[i];
                let moon_distance = moon_distances[i];

                // Check if we should enforce constraint based on Moon visibility
                let moon_visible = self.is_moon_visible(moon_altitude);
                if !self.enforce_when_below_horizon && !moon_visible {
                    // Moon is below horizon and we don't enforce in this case
                    return (false, 0.0);
                }

                let mut violated = false;
                let mut severity = 1.0;

                // Check illumination constraints
                if illumination > self.max_illumination {
                    violated = true;
                    severity = (illumination - self.max_illumination).min(1.0);
                }
                if let Some(min_illumination) = self.min_illumination {
                    if illumination < min_illumination {
                        violated = true;
                        severity = (min_illumination - illumination).min(1.0);
                    }
                }

                // Check distance constraints
                if let Some(min_distance) = self.min_distance {
                    if moon_distance < min_distance {
                        violated = true;
                        severity = (min_distance - moon_distance).min(1.0);
                    }
                }
                if let Some(max_distance) = self.max_distance {
                    if moon_distance > max_distance {
                        violated = true;
                        severity = (moon_distance - max_distance).min(1.0);
                    }
                }

                (violated, severity)
            },
            |i, violated| {
                if !violated {
                    return "".to_string();
                }

                let illumination = illuminations[i];
                let moon_altitude = moon_altitudes[i];
                let moon_distance = moon_distances[i];
                let phase_name = self.get_moon_phase_name(illumination);

                let mut reasons = Vec::new();

                if illumination > self.max_illumination {
                    reasons.push(format!(
                        "Moon too bright ({:.1}%, {}) - exceeds max {:.1}%",
                        illumination * 100.0,
                        phase_name,
                        self.max_illumination * 100.0
                    ));
                }
                if let Some(min_illumination) = self.min_illumination {
                    if illumination < min_illumination {
                        reasons.push(format!(
                            "Moon too dim ({:.1}%, {}) - below min {:.1}%",
                            illumination * 100.0,
                            phase_name,
                            min_illumination * 100.0
                        ));
                    }
                }

                if let Some(min_distance) = self.min_distance {
                    if moon_distance < min_distance {
                        reasons.push(format!(
                            "Moon too close ({:.1}°) - below min {:.1}°",
                            moon_distance, min_distance
                        ));
                    }
                }
                if let Some(max_distance) = self.max_distance {
                    if moon_distance > max_distance {
                        reasons.push(format!(
                            "Moon too far ({:.1}°) - exceeds max {:.1}°",
                            moon_distance, max_distance
                        ));
                    }
                }

                if reasons.is_empty() {
                    format!(
                        "Moon altitude: {:.1}°, distance: {:.1}°",
                        moon_altitude, moon_distance
                    )
                } else {
                    reasons.join("; ")
                }
            },
        );

        let all_satisfied = violations.is_empty();
        Ok(ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
            times_filtered.to_vec(),
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

        let illuminations = ephemeris.moon_illumination(time_indices)?;
        let moon_positions_all = ephemeris.get_moon_positions()?;
        let observer_positions_all = ephemeris.get_gcrs_positions()?;
        let gcrs_full_all =
            ephemeris.data().gcrs.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No GCRS data available.")
            })?;

        let moon_positions = if let Some(indices) = time_indices {
            moon_positions_all.select(ndarray::Axis(0), indices)
        } else {
            moon_positions_all
        };
        let observer_positions = if let Some(indices) = time_indices {
            observer_positions_all.select(ndarray::Axis(0), indices)
        } else {
            observer_positions_all
        };
        let gcrs_full = if let Some(indices) = time_indices {
            gcrs_full_all.select(ndarray::Axis(0), indices)
        } else {
            gcrs_full_all.to_owned()
        };

        let moon_altitudes =
            self.compute_moon_altitudes(&times_filtered, &moon_positions, &gcrs_full)?;
        let moon_units = self.compute_moon_unit_vectors(&moon_positions, &observer_positions)?;
        let target_vectors = radec_to_unit_vectors_batch(target_ras, target_decs);

        let min_cos_threshold = self.min_distance.map(|min| min.to_radians().cos());
        let max_cos_threshold = self.max_distance.map(|max| max.to_radians().cos());

        // Vectorized: compute all dot products at once using matrix multiplication
        // moon_units: (n_times, 3), target_vectors: (n_targets, 3)
        // We want: (n_targets, n_times) where result[j, i] = dot(target_vectors[j], moon_units[i])
        // This is target_vectors @ moon_units.T
        let cos_angles = target_vectors.dot(&moon_units.t());

        // Process illumination and visibility for each time index
        for i in 0..n_times {
            let illumination = illuminations[i];
            let moon_altitude = moon_altitudes[i];

            // Check if we should enforce constraint based on Moon visibility
            let moon_visible = self.is_moon_visible(moon_altitude);
            if !self.enforce_when_below_horizon && !moon_visible {
                // Moon is below horizon and we don't enforce in this case
                // All targets are considered satisfied (not violated)
                for j in 0..n_targets {
                    result[[j, i]] = false;
                }
                continue;
            }

            // Check illumination constraints (same for all targets)
            let mut illumination_violated = false;
            if illumination > self.max_illumination {
                illumination_violated = true;
            }
            if let Some(min_illumination) = self.min_illumination {
                if illumination < min_illumination {
                    illumination_violated = true;
                }
            }

            // Vectorized: check distance constraints for all targets at this time
            for j in 0..n_targets {
                let cos_angle = cos_angles[[j, i]].clamp(-1.0, 1.0);

                let mut target_violated = illumination_violated;
                if let Some(min_thresh) = min_cos_threshold {
                    if cos_angle > min_thresh {
                        target_violated = true;
                    }
                }
                if let Some(max_thresh) = max_cos_threshold {
                    if cos_angle < max_thresh {
                        target_violated = true;
                    }
                }

                result[[j, i]] = target_violated;
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

impl MoonPhaseEvaluator {
    /// Get descriptive name for moon phase based on illumination
    #[allow(dead_code)]
    fn get_moon_phase_name(&self, illumination: f64) -> &'static str {
        if illumination < 0.02 {
            "New Moon"
        } else if illumination < 0.48 {
            "Waxing Crescent"
        } else if illumination < 0.52 {
            "First Quarter"
        } else if illumination < 0.98 {
            "Waxing Gibbous"
        } else if illumination <= 1.02 {
            "Full Moon"
        } else if illumination < 1.48 {
            "Waning Gibbous"
        } else if illumination < 1.52 {
            "Last Quarter"
        } else {
            "Waning Crescent"
        }
    }
}
