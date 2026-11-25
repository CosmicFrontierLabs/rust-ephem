//! Optional WebGPU path for geometric dot-product masks.
use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub(super) fn try_mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> ([f64; 3], f64, Option<f64>),
) -> PyResult<Option<Array2<bool>>> {
    // The three geometric kernels beat CPU dispatch at this size on M1 Max.
    // Smaller grids rarely amortize queue submission and synchronous readback.
    if targets.nrows().saturating_mul(times) < 4_194_304
        || std::env::var_os("RUST_EPHEM_GPU").as_deref() != Some(std::ffi::OsStr::new("1"))
    {
        return Ok(None);
    }
    // Only bounded vectors participate in the shader's f32 error estimate.
    let bounded = |x: &f64| x.abs() <= 1.000001;
    if !targets.iter().all(bounded) {
        return Ok(None);
    }
    let mut geometry = Vec::with_capacity(times * 5);
    for t in 0..times {
        let (direction, min_cos, max_cos) = row(t);
        if !direction.iter().all(bounded) {
            return Ok(None);
        }
        geometry.extend(direction);
        // Finite sentinels lie outside the possible dot product. Comparisons
        // with NaN in the reference kernel are always false.
        geometry.push(if min_cos.is_nan() {
            16.0
        } else {
            min_cos.clamp(-16.0, 16.0)
        });
        geometry.push(
            max_cos
                .filter(|v| !v.is_nan())
                .unwrap_or(-16.0)
                .clamp(-16.0, 16.0),
        );
    }
    let geometry = Array2::from_shape_vec((times, 5), geometry).expect("five entries per row");
    Python::attach(|py| {
        let result = py
            .import("rust_ephem._gpu")?
            .getattr("cosine_mask")?
            .call1((targets.to_pyarray(py), geometry.into_pyarray(py)))?;
        let mask = result.extract::<PyReadonlyArray2<'_, bool>>()?;
        Ok(Some(mask.as_array().to_owned()))
    })
}
