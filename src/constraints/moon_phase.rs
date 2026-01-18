/// Moon phase constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use chrono::{DateTime, Utc};
use ndarray::{s, Array2};
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use sofars::astro::atco13;

use crate::utils::conversions::{convert_frames, Frame};
use crate::utils::geo::ecef_to_geodetic_deg;
use crate::utils::time_utils::datetime_to_jd_utc;
use crate::utils::vector_math::{
    dot_product, normalize_vector, radec_to_unit_vector, radec_to_unit_vectors_batch,
};
use crate::utils::{eop_provider, ut1_provider};

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
    ) -> PyResult<Vec<[f64; 3]>> {
        if moon_positions.nrows() != observer_positions.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Moon and observer position arrays must have the same number of rows.",
            ));
        }

        let n = moon_positions.nrows();
        let mut moon_units = Vec::with_capacity(n);

        for i in 0..n {
            let moon_row = moon_positions.row(i);
            let obs_row = observer_positions.row(i);
            let moon_rel = [
                moon_row[0] - obs_row[0],
                moon_row[1] - obs_row[1],
                moon_row[2] - obs_row[2],
            ];
            moon_units.push(normalize_vector(&moon_rel));
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

        let itrs_data = convert_frames(gcrs_full, times, Frame::GCRS, Frame::ITRS, true);
        let itrs_positions = itrs_data.slice(s![.., 0..3]).to_owned();
        let (lats_deg, lons_deg, heights_km) = ecef_to_geodetic_deg(&itrs_positions);

        let mut altitudes = Vec::with_capacity(times.len());

        for i in 0..times.len() {
            let obs_row = gcrs_full.row(i);
            let moon_row = moon_positions.row(i);
            let moon_rel = [
                moon_row[0] - obs_row[0],
                moon_row[1] - obs_row[1],
                moon_row[2] - obs_row[2],
            ];
            let dist =
                (moon_rel[0] * moon_rel[0] + moon_rel[1] * moon_rel[1] + moon_rel[2] * moon_rel[2])
                    .sqrt();
            if dist == 0.0 {
                altitudes.push(-90.0);
                continue;
            }

            let ra_rad = moon_rel[1].atan2(moon_rel[0]);
            let dec_rad = (moon_rel[2] / dist).asin();

            let lat_rad = lats_deg[i].to_radians();
            let lon_rad = lons_deg[i].to_radians();
            let height_m = heights_km[i] * 1000.0;
            let time = &times[i];

            let (utc1, utc2) = datetime_to_jd_utc(time);
            let dut1 = ut1_provider::get_ut1_utc_offset(time);
            let (xp, yp) = eop_provider::get_polar_motion_rad(time);

            let (_aob, zob, _hob, _dob, _rob, _eo) = atco13(
                ra_rad, dec_rad, 0.0, 0.0, 0.0, 0.0, utc1, utc2, dut1, lon_rad, lat_rad, height_m,
                xp, yp, 0.0, 0.0, 0.0, 0.55,
            )
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "SOFA atco13 failed for Moon altitude: {e:?}"
                ))
            })?;

            let alt_deg = (std::f64::consts::FRAC_PI_2 - zob).to_degrees();
            altitudes.push(alt_deg);
        }

        Ok(altitudes)
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
        let moon_distances: Vec<f64> = moon_units
            .iter()
            .map(|moon_unit| {
                let cos_angle = dot_product(&target_vec, moon_unit).clamp(-1.0, 1.0);
                cos_angle.acos().to_degrees()
            })
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

        for i in 0..n_times {
            let illumination = illuminations[i];
            let moon_altitude = moon_altitudes[i];

            // Check if we should enforce constraint based on Moon visibility
            let moon_visible = self.is_moon_visible(moon_altitude);
            if !self.enforce_when_below_horizon && !moon_visible {
                // Moon is below horizon and we don't enforce in this case
                // All targets are considered satisfied
                for j in 0..n_targets {
                    result[[j, i]] = false;
                }
                continue;
            }

            let mut violated = false;

            // Check illumination constraints
            if illumination > self.max_illumination {
                violated = true;
            }
            if let Some(min_illumination) = self.min_illumination {
                if illumination < min_illumination {
                    violated = true;
                }
            }

            // Check distance constraints for each target
            for j in 0..n_targets {
                let target_vec = [
                    target_vectors[[j, 0]],
                    target_vectors[[j, 1]],
                    target_vectors[[j, 2]],
                ];
                let cos_angle = dot_product(&target_vec, &moon_units[i]).clamp(-1.0, 1.0);

                let mut target_violated = violated;
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
