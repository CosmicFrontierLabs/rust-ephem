//! Shared angular comparisons and optional CubeCL dispatch.
use ndarray::Array2;
use pyo3::{exceptions::PyRuntimeError, PyResult};

#[cfg(feature = "cubecl")]
mod batch;
#[cfg(feature = "cubecl")]
mod cube;
#[cfg(feature = "cubecl")]
pub(super) use batch::Catalogue;

#[cfg(not(feature = "cubecl"))]
pub(super) struct Catalogue;
#[cfg(not(feature = "cubecl"))]
impl Catalogue {
    pub(super) fn new(_: impl IntoIterator<Item = [f64; 3]>) -> Self {
        Self
    }
}

#[cfg(not(feature = "cubecl"))]
const GPU_BUILD_REQUIRED: &str =
    "GPU execution requires a build with the Cargo gpu or cubecl feature";

#[derive(Clone, Copy)]
pub(super) enum Angle {
    Direction,
    NearestPole,
    Apparent,
    RollWedge { wide: bool },
}

impl Angle {
    pub(super) fn projection(self, dot: impl Fn(usize) -> f64) -> f64 {
        match self {
            Self::Direction | Self::Apparent => dot(0),
            Self::NearestPole => dot(0).abs(),
            Self::RollWedge { wide: true } => dot(0).min(dot(3)),
            Self::RollWedge { wide: false } => dot(0).max(dot(3)),
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
            let cosine = angle.projection(|k| {
                targets[[i, k]] * unit[0]
                    + targets[[i, k + 1]] * unit[1]
                    + targets[[i, k + 2]] * unit[2]
            });
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
    refine: Option<&(dyn Fn(usize, usize) -> bool + Sync)>,
) -> PyResult<Option<Array2<bool>>> {
    #[cfg(feature = "cubecl")]
    return batch::mask(targets, times, row, angle, refine).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, times, row, angle, refine);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}

/// Finish the surviving polygon candidates in a densely packed screening mask.
pub(super) fn refine_polygon_mask(
    mask: &mut Array2<bool>,
    candidate: bool,
    exact: impl Fn(usize, usize) -> bool + Sync,
) {
    use rayon::prelude::*;
    let times = mask.ncols();
    if times == 0 {
        return;
    }
    mask.as_slice_mut()
        .expect("contiguous screening mask")
        .par_chunks_mut(times)
        .enumerate()
        .for_each(|(i, row)| {
            for (t, cell) in row.iter_mut().enumerate() {
                if *cell == candidate {
                    *cell = exact(i, t);
                }
            }
        });
}

pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &Catalogue,
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
    targets: &ndarray::ArrayRef2<f64>,
    stars: &Catalogue,
) -> PyResult<Option<Vec<f64>>> {
    #[cfg(feature = "cubecl")]
    return batch::nearest_cosines(targets, stars).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, stars);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}

pub(super) fn angle_values(
    targets: &Array2<f64>,
    bodies: &[[f64; 3]],
    upper: f64,
    lower: f64,
    grid: &mut super::core::LeafValueGrid,
) -> PyResult<bool> {
    #[cfg(feature = "cubecl")]
    return batch::angle_grid(targets, bodies, upper, lower, grid).map_err(PyRuntimeError::new_err);
    #[cfg(not(feature = "cubecl"))]
    {
        let _ = (targets, bodies, upper, lower, grid);
        Err(PyRuntimeError::new_err(GPU_BUILD_REQUIRED))
    }
}
