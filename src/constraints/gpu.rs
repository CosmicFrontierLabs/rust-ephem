//! Shared angular comparisons and optional native GPU dispatch.
use ndarray::Array2;
use pyo3::{exceptions::PyRuntimeError, PyResult};

#[cfg(feature = "gpu")]
mod native;

#[derive(Clone, Copy)]
pub(super) enum Angle {
    Direction,
    NearestPole,
}

impl Angle {
    pub(super) fn cosine(self, dot: f64) -> f64 {
        match self {
            Self::Direction => dot,
            Self::NearestPole => dot.abs(),
        }
    }
}

pub(super) fn enabled() -> bool {
    std::env::var_os("RUST_EPHEM_GPU").as_deref() == Some(std::ffi::OsStr::new("1"))
}

pub(super) type AngularBounds = ([f64; 3], f64, Option<f64>);

/// A target-by-direction grid shared by rules with per-epoch angular bounds.
pub(super) fn angular_grid(
    targets: &Array2<f64>,
    epochs: &[AngularBounds],
    angle: Angle,
) -> PyResult<Array2<bool>> {
    if let Some(mask) = try_angle_mask(targets, epochs.len(), |t| epochs[t], angle)? {
        return Ok(mask);
    }
    Ok(Array2::from_shape_fn(
        (targets.nrows(), epochs.len()),
        |(i, t)| {
            let (unit, upper, lower) = epochs[t];
            let dot =
                targets[[i, 0]] * unit[0] + targets[[i, 1]] * unit[1] + targets[[i, 2]] * unit[2];
            let cosine = angle.cosine(dot);
            cosine > upper || lower.is_some_and(|bound| cosine < bound)
        },
    ))
}

pub(super) fn try_mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> ([f64; 3], f64, Option<f64>),
) -> PyResult<Option<Array2<bool>>> {
    try_angle_mask(targets, times, row, Angle::Direction)
}

fn try_angle_mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> AngularBounds,
    angle: Angle,
) -> PyResult<Option<Array2<bool>>> {
    if targets.nrows().saturating_mul(times) < 4_194_304 || !enabled() {
        return Ok(None);
    }
    #[cfg(feature = "gpu")]
    return native::mask(targets, times, row, angle).map_err(PyRuntimeError::new_err);

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (row, angle);
        Err(PyRuntimeError::new_err(
            "RUST_EPHEM_GPU=1 requires a build with the Cargo gpu feature",
        ))
    }
}

pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> PyResult<Option<Vec<bool>>> {
    #[cfg(feature = "gpu")]
    return native::catalogue_hits(targets, stars, threshold).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (targets, stars, threshold);
        Err(PyRuntimeError::new_err(
            "RUST_EPHEM_GPU=1 requires a build with the Cargo gpu feature",
        ))
    }
}
