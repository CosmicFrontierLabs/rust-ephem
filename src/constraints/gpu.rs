//! Shared angular comparisons and optional native GPU dispatch.
use ndarray::Array2;
use pyo3::{exceptions::PyRuntimeError, PyResult};

#[cfg(feature = "gpu")]
mod batch;
#[cfg(feature = "cubecl")]
mod cube;
#[cfg(feature = "gpu")]
mod native;

#[derive(Clone, Copy)]
enum Backend {
    #[cfg(feature = "gpu")]
    Wgpu,
    #[cfg(feature = "cubecl")]
    CubeCl,
}

impl Backend {
    fn selected() -> PyResult<Option<Self>> {
        match std::env::var("RUST_EPHEM_GPU").as_deref() {
            #[cfg(feature = "gpu")]
            Ok("1") => Ok(Some(Self::Wgpu)),
            #[cfg(not(feature = "gpu"))]
            Ok("1") => Err(PyRuntimeError::new_err(
                "RUST_EPHEM_GPU=1 requires a build with the Cargo gpu feature",
            )),
            #[cfg(feature = "cubecl")]
            Ok("cubecl") => Ok(Some(Self::CubeCl)),
            #[cfg(not(feature = "cubecl"))]
            Ok("cubecl") => Err(PyRuntimeError::new_err(
                "RUST_EPHEM_GPU=cubecl requires a build with the Cargo cubecl feature",
            )),
            _ => Ok(None),
        }
    }
}

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
    matches!(
        std::env::var("RUST_EPHEM_GPU").as_deref(),
        Ok("1" | "cubecl")
    )
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
    if targets.nrows().saturating_mul(times) < 4_194_304 {
        return Ok(None);
    }
    let Some(backend) = Backend::selected()? else {
        return Ok(None);
    };
    #[cfg(feature = "gpu")]
    return batch::mask(backend, targets, times, row, angle).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (row, angle);
        match backend {}
    }
}

pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> PyResult<Option<Vec<bool>>> {
    let Some(backend) = Backend::selected()? else {
        return Ok(None);
    };
    #[cfg(feature = "gpu")]
    return batch::catalogue_hits(backend, targets, stars, threshold)
        .map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (targets, stars, threshold);
        match backend {}
    }
}
