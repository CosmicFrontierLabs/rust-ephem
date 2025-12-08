/// South Atlantic Anomaly constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
use crate::utils::geo::ecef_to_geodetic_deg;
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
struct SAAEvaluator {
    polygon: Vec<(f64, f64)>,
}

impl SAAEvaluator {
    fn format_name(&self) -> String {
        format!("SAAConstraint(vertices={})", self.polygon.len())
    }

    /// Check if a point is inside the polygon using ray casting algorithm
    /// Returns true if point is inside polygon
    fn point_in_polygon(&self, lon: f64, lat: f64) -> bool {
        let mut inside = false;
        let n = self.polygon.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let (x1, y1) = self.polygon[i];
            let (x2, y2) = self.polygon[j];

            // Check if ray from point intersects edge
            if ((y1 > lat) != (y2 > lat)) && (lon < x1 + (x2 - x1) * (lat - y1) / (y2 - y1)) {
                inside = !inside;
            }
        }

        inside
    }
}

impl ConstraintEvaluator for SAAEvaluator {
    fn evaluate(
        &self,
        times: &[DateTime<Utc>],
        _target_ra: f64,
        _target_dec: f64,
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> ConstraintResult {
        // Convert observer positions to lat/lon
        let (lats, lons, _heights) = ecef_to_geodetic_deg(observer_positions);

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
            |i, violated| {
                let lat = lats[i];
                let lon = lons[i];
                if violated {
                    format!("In SAA region (lat: {:.2}°, lon: {:.2}°)", lat, lon)
                } else {
                    format!("Outside SAA (lat: {:.2}°, lon: {:.2}°)", lat, lon)
                }
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

    fn in_constraint_batch(
        &self,
        times: &[DateTime<Utc>],
        _target_ras: &[f64],
        _target_decs: &[f64],
        _sun_positions: &Array2<f64>,
        _moon_positions: &Array2<f64>,
        observer_positions: &Array2<f64>,
    ) -> PyResult<Array2<bool>> {
        let n_times = times.len();
        let n_targets = 1; // SAA is target-independent, but we need to match the interface

        // Convert observer positions to lat/lon
        let (lats, lons, _heights) = ecef_to_geodetic_deg(observer_positions);

        let mut result = Array2::<bool>::from_elem((n_targets, n_times), false);

        for i in 0..n_times {
            let lat = lats[i];
            let lon = lons[i];
            let in_saa = self.point_in_polygon(lon, lat);
            let satisfied = !in_saa;
            result[[0, i]] = satisfied;
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
