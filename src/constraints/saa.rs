/// South Atlantic Anomaly constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::polygon;
use chrono::{DateTime, Utc};
use ndarray::Array2;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

/// Configuration for South Atlantic Anomaly constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAAConfig {
    /// Polygon defining the SAA region as (longitude, latitude) pairs in degrees
    pub polygon: Vec<(f64, f64)>,
}

impl ConstraintConfig for SAAConfig {
    fn to_evaluator(&self) -> Box<dyn ConstraintEvaluator> {
        Box::new(SAAEvaluator {
            polygon: self.polygon.clone(),
        })
    }
}

/// Evaluator for South Atlantic Anomaly constraint
pub struct SAAEvaluator {
    polygon: Vec<(f64, f64)>,
}

impl SAAEvaluator {
    fn format_name(&self) -> String {
        format!("SAAConstraint(vertices={})", self.polygon.len())
    }

    /// Check if a point is inside the polygon using the winding number algorithm
    /// This is more robust than ray casting for complex polygons
    fn point_in_polygon(&self, lon: f64, lat: f64) -> bool {
        polygon::point_in_polygon(&self.polygon, lon, lat)
    }
}

impl SAAEvaluator {
    /// Evaluate the constraint with pre-computed lat/lon arrays
    pub fn evaluate_with_latlon(
        &self,
        times: &[DateTime<Utc>],
        lats: &[f64],
        lons: &[f64],
    ) -> ConstraintResult {
        let violations = track_violations(
            times,
            |i| {
                let lat = lats[i];
                let lon = lons[i];
                let in_saa = self.point_in_polygon(lon, lat);
                let violated = in_saa;
                let severity = if violated { 1.0 } else { 0.0 };
                (violated, severity)
            },
            |i, _still_violated| {
                // Description should always describe the violation (being in SAA)
                let lat = lats[i];
                let lon = lons[i];
                format!("In SAA region (lat: {:.2}°, lon: {:.2}°)", lat, lon)
            },
        );

        let all_satisfied = violations.is_empty();
        ConstraintResult::new(
            violations,
            all_satisfied,
            self.format_name(),
            times.to_vec(),
        )
    }

    /// Batch evaluation with pre-computed lat/lon arrays
    pub fn in_constraint_batch_with_latlon(
        &self,
        target_ras: &[f64],
        lats: &[f64],
        lons: &[f64],
    ) -> Array2<bool> {
        let n_times = lats.len();
        let n_targets = target_ras.len();

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let lat = lats[i];
            let lon = lons[i];
            let in_saa = self.point_in_polygon(lon, lat);
            let satisfied = !in_saa;

            for target_idx in 0..n_targets {
                result[[target_idx, i]] = satisfied;
            }
        }

        result
    }
}

impl ConstraintEvaluator for SAAEvaluator {
    fn evaluate(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        _target_ra: f64,
        _target_dec: f64,
        time_indices: Option<&[usize]>,
    ) -> pyo3::PyResult<ConstraintResult> {
        // Extract and filter lat/lon data
        let (times_slice, lats_slice, lons_slice) = extract_latlon_data!(ephemeris, time_indices);

        let result = self.evaluate_with_latlon(&times_slice, &lats_slice, &lons_slice);
        Ok(result)
    }

    fn in_constraint_batch(
        &self,
        ephemeris: &dyn crate::ephemeris::ephemeris_common::EphemerisBase,
        target_ras: &[f64],
        _target_decs: &[f64],
        time_indices: Option<&[usize]>,
    ) -> PyResult<Array2<bool>> {
        // Extract and filter lat/lon data (discard times since we don't need them)
        let (_times_slice, lats_slice, lons_slice) = extract_latlon_data!(ephemeris, time_indices);

        let result = self.in_constraint_batch_with_latlon(target_ras, &lats_slice, &lons_slice);
        Ok(result)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
