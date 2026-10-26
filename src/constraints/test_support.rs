//! Minimal in-memory ephemeris for evaluation and work-count regression tests.
use crate::ephemeris::ephemeris_common::{EphemerisBase, EphemerisData};
use chrono::{TimeZone, Utc};
use ndarray::Array2;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TestEphemeris {
    pub data: EphemerisData,
    pub body_queries: AtomicUsize,
}

impl TestEphemeris {
    pub fn new(n: usize) -> Self {
        let mut data = EphemerisData::new();
        let epoch = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        data.times = Some(
            (0..n)
                .map(|i| epoch + chrono::Duration::seconds(i as i64))
                .collect(),
        );
        data.gcrs = Some(Array2::from_shape_fn((n, 6), |(_, j)| {
            if j == 0 {
                -7000.0
            } else {
                0.0
            }
        }));
        data.sun_gcrs = Some(Array2::from_shape_fn((n, 6), |(_, j)| {
            if j == 0 {
                1e8
            } else {
                0.0
            }
        }));
        data.moon_gcrs = Some(Array2::from_shape_fn((n, 6), |(_, j)| {
            if j == 1 {
                1e6
            } else {
                0.0
            }
        }));
        Self {
            data,
            body_queries: AtomicUsize::new(0),
        }
    }
}

impl EphemerisBase for TestEphemeris {
    fn data(&self) -> &EphemerisData {
        &self.data
    }
    fn data_mut(&mut self) -> &mut EphemerisData {
        &mut self.data
    }
    fn get_itrs_data(&self) -> Option<&Array2<f64>> {
        None
    }
    fn get_itrs_skycoord_ref(&self) -> Option<&Py<PyAny>> {
        None
    }
    fn set_itrs_skycoord_cache(&self, value: Py<PyAny>) -> Result<(), Py<PyAny>> {
        Err(value)
    }
    fn radec_to_altaz(&self, _: f64, _: f64, _: Option<&[usize]>) -> PyResult<Array2<f64>> {
        unreachable!("not used by these tests")
    }
    fn get_any_body_gcrs_positions(&self, _: &str) -> PyResult<Array2<f64>> {
        self.body_queries.fetch_add(1, Ordering::Relaxed);
        Ok(self.data.moon_gcrs.as_ref().unwrap().clone())
    }
}
