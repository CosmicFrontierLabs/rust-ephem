use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

#[pyclass]
// Store TEME information in a structure
pub struct PositionVelocityData {
    pub position: Array2<f64>,
    pub velocity: Array2<f64>,
}

#[pymethods]
impl PositionVelocityData {
    #[getter]
    fn position(&self, py: Python) -> Py<PyArray2<f64>> {
        self.position.clone().into_pyarray(py).to_owned().into()
    }

    #[getter]
    fn velocity(&self, py: Python) -> Py<PyArray2<f64>> {
        self.velocity.clone().into_pyarray(py).to_owned().into()
    }

    /// Right ascension of the position vector in degrees (0-360), one per row.
    #[getter]
    fn ra_deg(&self, py: Python) -> Py<PyArray1<f64>> {
        self.position
            .rows()
            .into_iter()
            .map(|p| p[1].atan2(p[0]).to_degrees().rem_euclid(360.0))
            .collect::<Vec<f64>>()
            .into_pyarray(py)
            .unbind()
    }

    /// Declination of the position vector in degrees, one per row.
    #[getter]
    fn dec_deg(&self, py: Python) -> Py<PyArray1<f64>> {
        self.position
            .rows()
            .into_iter()
            .map(|p| p[2].atan2(p[0].hypot(p[1])).to_degrees())
            .collect::<Vec<f64>>()
            .into_pyarray(py)
            .unbind()
    }

    /// Length of the position vector in km, one per row.
    #[getter]
    fn distance(&self, py: Python) -> Py<PyArray1<f64>> {
        self.position
            .rows()
            .into_iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .collect::<Vec<f64>>()
            .into_pyarray(py)
            .unbind()
    }

    #[getter]
    fn position_unit(&self) -> &str {
        "km" // kilometers
    }
    #[getter]
    fn velocity_unit(&self) -> &str {
        "km/s" // kilometers per second
    }
}
