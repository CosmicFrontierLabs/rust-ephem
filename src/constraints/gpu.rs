//! Optional WebGPU path for geometric dot-product masks.
use ndarray::Array2;
use pyo3::{exceptions::PyRuntimeError, PyResult};

#[cfg(feature = "gpu")]
mod native;

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
    #[cfg(feature = "gpu")]
    return native::mask(targets, times, row).map_err(PyRuntimeError::new_err);

    #[cfg(not(feature = "gpu"))]
    {
        let _ = row;
        Err(PyRuntimeError::new_err(
            "RUST_EPHEM_GPU=1 requires a build with the Cargo gpu feature",
        ))
    }
}
