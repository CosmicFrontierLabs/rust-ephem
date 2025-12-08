/// South Atlantic Anomaly constraint implementation
use super::core::{track_violations, ConstraintConfig, ConstraintEvaluator, ConstraintResult};
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
        let mut winding_number = 0.0;
        let n = self.polygon.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let (x1, y1) = self.polygon[i];
            let (x2, y2) = self.polygon[j];

            if y1 <= lat {
                if y2 > lat && (x2 - x1) * (lat - y1) - (lon - x1) * (y2 - y1) > 0.0 {
                    winding_number += 1.0;
                }
            } else if y2 <= lat && (x2 - x1) * (lat - y1) - (lon - x1) * (y2 - y1) < 0.0 {
                winding_number -= 1.0;
            }
        }

        winding_number != 0.0
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
        use numpy::{PyArray1, PyArrayMethods};
        use pyo3::Python;

        // Extract lat/lon data from ephemeris
        let (lats_vec, lons_vec) = Python::attach(|py| -> pyo3::PyResult<(Vec<f64>, Vec<f64>)> {
            let lat_opt = ephemeris.get_latitude_deg(py)?;
            let lon_opt = ephemeris.get_longitude_deg(py)?;

            let lat_array = lat_opt.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Latitude data not available")
            })?;
            let lon_array = lon_opt.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Longitude data not available")
            })?;

            let lat_bound = lat_array.downcast_bound::<PyArray1<f64>>(py)?;
            let lon_bound = lon_array.downcast_bound::<PyArray1<f64>>(py)?;

            let lats = lat_bound.readonly().as_slice()?.to_vec();
            let lons = lon_bound.readonly().as_slice()?.to_vec();

            Ok((lats, lons))
        })?;

        let times = ephemeris.get_times()?;

        let (times_slice, lats_slice, lons_slice) = if let Some(indices) = time_indices {
            let filtered_times: Vec<DateTime<Utc>> = indices.iter().map(|&i| times[i]).collect();
            let filtered_lats: Vec<f64> = indices.iter().map(|&i| lats_vec[i]).collect();
            let filtered_lons: Vec<f64> = indices.iter().map(|&i| lons_vec[i]).collect();
            (filtered_times, filtered_lats, filtered_lons)
        } else {
            (times.to_vec(), lats_vec, lons_vec)
        };

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
        use numpy::{PyArray1, PyArrayMethods};
        use pyo3::Python;

        // Extract lat/lon data from ephemeris
        let (lats_vec, lons_vec) = Python::attach(|py| -> pyo3::PyResult<(Vec<f64>, Vec<f64>)> {
            let lat_opt = ephemeris.get_latitude_deg(py)?;
            let lon_opt = ephemeris.get_longitude_deg(py)?;

            let lat_array = lat_opt.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Latitude data not available")
            })?;
            let lon_array = lon_opt.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Longitude data not available")
            })?;

            let lat_bound = lat_array.downcast_bound::<PyArray1<f64>>(py)?;
            let lon_bound = lon_array.downcast_bound::<PyArray1<f64>>(py)?;

            let lats = lat_bound.readonly().as_slice()?.to_vec();
            let lons = lon_bound.readonly().as_slice()?.to_vec();

            Ok((lats, lons))
        })?;

        let (lats_slice, lons_slice, ras_slice) = if let Some(indices) = time_indices {
            let filtered_lats: Vec<f64> = indices.iter().map(|&i| lats_vec[i]).collect();
            let filtered_lons: Vec<f64> = indices.iter().map(|&i| lons_vec[i]).collect();
            (filtered_lats, filtered_lons, target_ras)
        } else {
            (lats_vec, lons_vec, target_ras)
        };

        let result = self.in_constraint_batch_with_latlon(ras_slice, &lats_slice, &lons_slice);
        Ok(result)
    }

    fn name(&self) -> String {
        self.format_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
