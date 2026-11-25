//! Shared angular comparisons and optional CubeCL dispatch.
use ndarray::Array2;
use pyo3::{exceptions::PyRuntimeError, PyResult};

#[cfg(feature = "cubecl")]
mod batch;
#[cfg(feature = "cubecl")]
mod cube;

#[cfg(not(feature = "cubecl"))]
const GPU_BUILD_REQUIRED: &str =
    "GPU execution requires a build with the Cargo gpu or cubecl feature";

#[derive(Clone, Copy)]
pub(super) enum Angle {
    Direction,
    NearestPole,
    Apparent,
}

impl Angle {
    pub(super) fn cosine(self, dot: f64) -> f64 {
        match self {
            Self::Direction | Self::Apparent => dot,
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

pub(super) const ANGULAR_MIN_PAIRS: usize = 4_194_304;
pub(super) const TRIG_MIN_PAIRS: usize = 131_072;

pub(super) fn mask_eligible(targets: usize, times: usize, minimum: usize) -> bool {
    targets.saturating_mul(times) >= minimum && enabled()
}

pub(super) type AngularBounds = ([f64; 3], f64, Option<f64>);

/// A target-by-direction grid shared by rules with per-epoch angular bounds.
pub(super) fn angular_grid(
    targets: &Array2<f64>,
    epochs: &[AngularBounds],
    angle: Angle,
) -> PyResult<Array2<bool>> {
    if mask_eligible(targets.nrows(), epochs.len(), ANGULAR_MIN_PAIRS) {
        if let Some(mask) = dispatch_angle_mask(targets, epochs.len(), |t| epochs[t], angle, None)?
        {
            return Ok(mask);
        }
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
    if !mask_eligible(targets.nrows(), times, ANGULAR_MIN_PAIRS) {
        return Ok(None);
    }
    dispatch_angle_mask(targets, times, row, Angle::Direction, None)
}

/// Dispatch after the caller checks eligibility and prepares its geometry.
/// The callback resolves ambiguous cells in the original target/time indices.
pub(super) fn dispatch_angle_mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> AngularBounds,
    angle: Angle,
    refine: Option<&dyn Fn(usize, usize) -> bool>,
) -> PyResult<Option<Array2<bool>>> {
    #[cfg(feature = "cubecl")]
    return batch::mask(targets, times, row, angle, refine).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, times, row, angle, refine);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}

pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> PyResult<Option<Vec<bool>>> {
    if !enabled() {
        return Ok(None);
    }
    #[cfg(feature = "cubecl")]
    return batch::catalogue_hits(targets, stars, threshold).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, stars, threshold);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}

pub(super) fn nearest_catalogue_cosines(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
) -> PyResult<Option<Vec<f64>>> {
    #[cfg(feature = "cubecl")]
    return batch::nearest_cosines(targets, stars).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, stars);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}
