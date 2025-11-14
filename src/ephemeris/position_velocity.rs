use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
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

    #[getter]
    fn position_unit(&self) -> &str {
        "km" // kilometers
    }
    #[getter]
    fn velocity_unit(&self) -> &str {
        "km/s" // kilometers per second
    }
}
