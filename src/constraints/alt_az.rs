/// Altitude/Azimuth constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::polygon;
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Altitude/Azimuth constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltAzConfig {
    /// Minimum allowed altitude in degrees (0 = horizon, 90 = zenith, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_altitude: Option<f64>,
    /// Maximum allowed altitude in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_altitude: Option<f64>,
    /// Minimum allowed azimuth in degrees (0 = North, 90 = East, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_azimuth: Option<f64>,
    /// Maximum allowed azimuth in degrees (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_azimuth: Option<f64>,
    /// Polygon defining an allowed region in Alt/Az space as (altitude, azimuth) pairs in degrees
    /// If provided, the target must be inside this polygon (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<(f64, f64)>>,
}

impl ConstraintConfig for AltAzConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(AltAzEvaluator {
            min_altitude: self.min_altitude,
            max_altitude: self.max_altitude,
            min_azimuth: self.min_azimuth,
            max_azimuth: self.max_azimuth,
            polygon: self.polygon.clone(),
        })
    }
}

/// Evaluator for Altitude/Azimuth constraint
struct AltAzEvaluator {
    min_altitude: Option<f64>,
    max_altitude: Option<f64>,
    min_azimuth: Option<f64>,
    max_azimuth: Option<f64>,
    polygon: Option<Vec<(f64, f64)>>,
}

impl AltAzEvaluator {
    fn outside(&self, altitude: f64, azimuth: f64) -> bool {
        self.min_altitude.is_some_and(|v| altitude < v)
            || self.max_altitude.is_some_and(|v| altitude > v)
            || match (self.min_azimuth, self.max_azimuth) {
                (Some(lo), Some(hi)) if lo <= hi => !(azimuth >= lo && azimuth <= hi),
                (Some(lo), Some(hi)) => !(azimuth >= lo || azimuth <= hi),
                (Some(lo), None) => azimuth < lo,
                (None, Some(hi)) => azimuth > hi,
                (None, None) => false,
            }
            || polygon::polygon_violation(self.polygon.as_ref(), altitude, azimuth)
    }

    fn gpu_mask(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        ras: &[f64],
        decs: &[f64],
        indices: Option<&[usize]>,
        n_times: usize,
    ) -> PyResult<Option<Array2<bool>>> {
        use super::gpu;
        use crate::utils::celestial::{altaz_astrometry, observed_altaz};
        use sofars::{
            astro::atoiq,
            vm::{s2c, trxp},
        };
        if !gpu::mask_eligible(ras.len(), n_times, gpu::TRIG_MIN_PAIRS)
            || ras.len() != decs.len()
            || [self.min_altitude, self.max_altitude]
                .into_iter()
                .flatten()
                .any(|v| !(-90.0..=90.0).contains(&v))
            || [self.min_azimuth, self.max_azimuth]
                .into_iter()
                .flatten()
                .any(|v| !(0.0..=360.0).contains(&v))
        {
            return Ok(None);
        }
        let astrom = altaz_astrometry(ephemeris, indices)?;
        // With zero refraction and zero separate diurnal aberration, atioq
        // is a rotation. Only the bounded atciq deflection/aberration moves
        // a target across the rotated geometric planes.
        if !astrom.iter().all(|a| {
            a.diurab == 0.0
                && a.refa == 0.0
                && a.refb == 0.0
                && crate::utils::apparent::can_screen_apparent_direction(a.em, &a.v, &a.eh)
        }) {
            return Ok(None);
        }
        let targets = crate::utils::vector_math::radec_to_unit_vectors_batch(ras, decs);
        let exact = |i: usize, t: usize| {
            let mut state = astrom[t];
            let [alt, az] = observed_altaz(ras[i].to_radians(), decs[i].to_radians(), &mut state);
            self.outside(alt, az)
        };
        let plane = |az: f64, zd: f64, upper: f64, lower: Option<f64>| {
            gpu::dispatch_angle_mask(
                &targets,
                n_times,
                |t| {
                    // Rotate the observed zenith or horizontal edge normal back
                    // through the Earth orientation and bias-precession-nutation.
                    let (ri, di) = atoiq("A", az, zd, &astrom[t]);
                    let mut v = [0.0; 3];
                    trxp(&astrom[t].bpn, &s2c(ri, di), &mut v);
                    (v, upper, lower)
                },
                gpu::Angle::Apparent,
                Some(&exact),
            )
        };
        // A polygon must lie inside its altitude/azimuth bounding box.
        // Screen that box first and apply the winding test to its survivors.
        let (min_alt, max_alt, lo, hi) = if let Some(vertices) = &self.polygon {
            if vertices.is_empty()
                || !vertices
                    .iter()
                    .all(|&(alt, az)| (-90.0..=90.0).contains(&alt) && (0.0..=360.0).contains(&az))
            {
                return Ok(None);
            }
            let mut low_alt = 90.0_f64;
            let mut high_alt = -90.0_f64;
            let mut low_az = 360.0_f64;
            let mut high_az = 0.0_f64;
            for &(alt, az) in vertices {
                low_alt = low_alt.min(alt);
                high_alt = high_alt.max(alt);
                low_az = low_az.min(az);
                high_az = high_az.max(az);
            }
            (Some(low_alt), Some(high_alt), low_az, high_az)
        } else {
            (
                self.min_altitude,
                self.max_altitude,
                self.min_azimuth.unwrap_or(0.0),
                self.max_azimuth.unwrap_or(360.0),
            )
        };
        let mut result = if min_alt.is_some() || max_alt.is_some() {
            let upper = max_alt.map_or(f64::INFINITY, |v| v.to_radians().sin());
            let lower = min_alt.map(|v| v.to_radians().sin());
            let Some(mask) = plane(0.0, 0.0, upper, lower)? else {
                return Ok(None);
            };
            mask
        } else {
            Array2::from_elem((ras.len(), n_times), false)
        };
        if (lo, hi) != (0.0, 360.0) {
            let horizontal =
                |az: f64| plane(az.to_radians(), std::f64::consts::FRAC_PI_2, 0.0, None);
            let Some(mut az) = horizontal(lo - 90.0)? else {
                return Ok(None);
            };
            let Some(end) = horizontal(hi + 90.0)? else {
                return Ok(None);
            };
            if (hi - lo).rem_euclid(360.0) <= 180.0 {
                az |= &end;
            } else {
                az &= &end;
            }
            result |= &az;
        }
        if self.polygon.is_some() {
            for ((i, t), cell) in result.indexed_iter_mut() {
                if !*cell {
                    *cell = exact(i, t);
                }
            }
        }
        Ok(Some(result))
    }

    fn format_name(&self) -> String {
        let mut parts = vec![];

        if let Some(min_alt) = self.min_altitude {
            parts.push(format!("min_alt={:.1}°", min_alt));
        }

        if let Some(max_alt) = self.max_altitude {
            parts.push(format!("max_alt={:.1}°", max_alt));
        }
        if let Some(min_az) = self.min_azimuth {
            parts.push(format!("min_az={:.1}°", min_az));
        }
        if let Some(max_az) = self.max_azimuth {
            parts.push(format!("max_az={:.1}°", max_az));
        }
        if self.polygon.is_some() {
            parts.push("polygon".to_string());
        }

        if parts.is_empty() {
            "AltAzConstraint".to_string()
        } else {
            format!("AltAzConstraint({})", parts.join(", "))
        }
    }
}

impl ConstraintEvaluator for AltAzEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ra: f64,
        target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> PyResult<ConstraintResult> {
        // Extract and filter ephemeris data
        let (times_filtered, _obs_filtered) =
            extract_observer_ephemeris_data!(ephemeris, time_indices);

        // Compute alt/az for this target at selected times
        let altaz = ephemeris.radec_to_altaz(target_ra, target_dec, time_indices)?;

        let violations = track_violations(
            &times_filtered,
            |i| {
                let altitude_deg = altaz[[i, 0]];
                let azimuth_deg = altaz[[i, 1]];

                // Check polygon constraint first if defined
                if polygon::polygon_violation(self.polygon.as_ref(), altitude_deg, azimuth_deg) {
                    return (true, 1.0);
                }

                // Check altitude constraints
                let mut violated = false;
                let mut severity = 1.0;

                if let Some(min_alt) = self.min_altitude {
                    if altitude_deg < min_alt {
                        violated = true;
                        severity = (min_alt - altitude_deg).min(1.0);
                    }
                }

                if let Some(max_altitude) = self.max_altitude {
                    if altitude_deg > max_altitude {
                        violated = true;
                        severity = (altitude_deg - max_altitude).min(1.0);
                    }
                }

                // Check azimuth constraints (only if altitude is acceptable)
                if !violated {
                    if let Some(min_azimuth) = self.min_azimuth {
                        if let Some(max_azimuth) = self.max_azimuth {
                            // Azimuth range constraint
                            let az_in_range = if min_azimuth <= max_azimuth {
                                azimuth_deg >= min_azimuth && azimuth_deg <= max_azimuth
                            } else {
                                // Handle wrap-around (e.g., 330° to 30°)
                                azimuth_deg >= min_azimuth || azimuth_deg <= max_azimuth
                            };
                            if !az_in_range {
                                violated = true;
                                severity = 1.0; // Azimuth violations are binary
                            }
                        } else {
                            // Only minimum azimuth
                            if azimuth_deg < min_azimuth {
                                violated = true;
                                severity = (min_azimuth - azimuth_deg).min(1.0);
                            }
                        }
                    } else if let Some(max_azimuth) = self.max_azimuth {
                        // Only maximum azimuth
                        if azimuth_deg > max_azimuth {
                            violated = true;
                            severity = (azimuth_deg - max_azimuth).min(1.0);
                        }
                    }
                }

                (violated, severity)
            },
            |_, _violated| {
                // Use the first timestamp alt/az for the description
                let altitude_deg = altaz[[0, 0]];
                let azimuth_deg = altaz[[0, 1]];
                self.format_violation_description(altitude_deg, azimuth_deg)
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
        let n_targets = target_ras.len();
        let n_times =
            time_indices.map_or_else(|| ephemeris.get_times().map(|t| t.len()), |i| Ok(i.len()))?;
        if let Some(mask) =
            self.gpu_mask(ephemeris, target_ras, target_decs, time_indices, n_times)?
        {
            return Ok(mask);
        }
        let altaz = crate::utils::celestial::radec_to_altaz_batch(
            target_ras,
            target_decs,
            ephemeris,
            time_indices,
        )?;
        let result = Array2::from_shape_fn((n_targets, n_times), |(i, t)| {
            self.outside(altaz[i][[t, 0]], altaz[i][[t, 1]])
        });
        Ok(result)
    }

    fn compute_named_values(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<HashMap<String, Array2<f64>>> {
        let n_targets = target_ras.len();
        let altaz_list = crate::utils::celestial::radec_to_altaz_batch(
            target_ras,
            target_decs,
            ephemeris,
            time_indices,
        )?;

        let n_times =
            time_indices.map_or_else(|| ephemeris.get_times().map(|t| t.len()), |i| Ok(i.len()))?;

        let mut altitude_deg = Array2::<f64>::zeros((n_targets, n_times));
        let mut azimuth_deg = Array2::<f64>::zeros((n_targets, n_times));
        for (j, altaz) in altaz_list.iter().enumerate() {
            for i in 0..n_times {
                altitude_deg[[j, i]] = altaz[[i, 0]];
                azimuth_deg[[j, i]] = altaz[[i, 1]];
            }
        }

        let mut values = HashMap::new();
        values.insert("altitude_deg".to_string(), altitude_deg);
        values.insert("azimuth_deg".to_string(), azimuth_deg);
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

impl AltAzEvaluator {
    /// Format a description of the violation based on altitude and azimuth
    #[allow(dead_code)]
    fn format_violation_description(&self, altitude_deg: f64, azimuth_deg: f64) -> String {
        let mut reasons = Vec::new();

        // Check polygon violation first
        if polygon::polygon_violation(self.polygon.as_ref(), altitude_deg, azimuth_deg) {
            reasons.push(format!(
                "outside allowed Alt/Az polygon (alt: {:.1}°, az: {:.1}°)",
                altitude_deg, azimuth_deg
            ));
        }

        if let Some(min_alt) = self.min_altitude {
            if altitude_deg < min_alt {
                reasons.push(format!(
                    "altitude {:.1}° < min {:.1}°",
                    altitude_deg, min_alt
                ));
            }
        }
        if let Some(max_altitude) = self.max_altitude {
            if altitude_deg > max_altitude {
                reasons.push(format!(
                    "altitude {:.1}° > max {:.1}°",
                    altitude_deg, max_altitude
                ));
            }
        }

        if let Some(min_azimuth) = self.min_azimuth {
            if let Some(max_azimuth) = self.max_azimuth {
                let az_in_range = if min_azimuth <= max_azimuth {
                    azimuth_deg >= min_azimuth && azimuth_deg <= max_azimuth
                } else {
                    azimuth_deg >= min_azimuth || azimuth_deg <= max_azimuth
                };
                if !az_in_range {
                    reasons.push(format!(
                        "azimuth {:.1}° outside range {:.1}°-{:1}°",
                        azimuth_deg, min_azimuth, max_azimuth
                    ));
                }
            } else if azimuth_deg < min_azimuth {
                reasons.push(format!(
                    "azimuth {:.1}° < min {:.1}°",
                    azimuth_deg, min_azimuth
                ));
            }
        } else if let Some(max_azimuth) = self.max_azimuth {
            if azimuth_deg > max_azimuth {
                reasons.push(format!(
                    "azimuth {:.1}° > max {:.1}°",
                    azimuth_deg, max_azimuth
                ));
            }
        }

        format!(
            "Target position violates constraints: {}",
            reasons.join(", ")
        )
    }
}
