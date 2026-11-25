//! Host-side tiling, packed mask decoding, and f64 refinement for GPU kernels.
use super::{native, Angle, AngularBounds, Backend};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};

#[derive(Clone, Copy)]
pub(super) enum Kernel {
    Mask(Angle),
    Catalogue,
}

impl Kernel {
    pub(super) fn groups(self, records: usize) -> u32 {
        match self {
            Self::Mask(_) => records.div_ceil(64) as u32,
            Self::Catalogue => records as u32,
        }
    }
}

pub(super) trait Compute {
    type Buffer;
    fn storage(&self, data: &[f32]) -> Self::Buffer;
    /// Each output record consists of two u32 words.
    fn dispatch(
        &self,
        kernel: Kernel,
        inputs: [&Self::Buffer; 2],
        records: usize,
    ) -> Result<impl std::ops::Deref<Target = [u8]>, String>;
}

fn mask_tile<B: Compute>(
    backend: &B,
    targets: ArrayView2<'_, f64>,
    epochs: &[[f64; 5]],
    geometry: &B::Buffer,
    angle: Angle,
    mut mask: ArrayViewMut2<'_, bool>,
) -> Result<(), String> {
    let words = epochs.len().div_ceil(32);
    let count = words * targets.nrows();
    let target_buffer = backend.storage(&targets.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let data = backend.dispatch(Kernel::Mask(angle), [&target_buffer, geometry], count)?;
    let packed: &[[u32; 2]] = bytemuck::cast_slice(&data);
    for (i, mut row) in mask.rows_mut().into_iter().enumerate() {
        let target = targets.row(i);
        for (word, (cells, &[flags, mut uncertain])) in row
            .as_slice_mut()
            .expect("contiguous mask row")
            .chunks_mut(32)
            .zip(&packed[i * words..(i + 1) * words])
            .enumerate()
        {
            for (lane, cell) in cells.iter_mut().enumerate() {
                *cell = flags & (1 << lane) != 0;
            }
            while uncertain != 0 {
                let lane = uncertain.trailing_zeros() as usize;
                let [x, y, z, upper, lower] = epochs[word * 32 + lane];
                let dot = target[0] * x + target[1] * y + target[2] * z;
                let cosine = angle.cosine(dot);
                cells[lane] = cosine > upper || cosine < lower;
                uncertain &= uncertain - 1;
            }
        }
    }
    Ok(())
}

pub(super) fn mask(
    backend: Backend,
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> AngularBounds,
    angle: Angle,
) -> Result<Option<Array2<bool>>, String> {
    let bounded = |x: &f64| x.abs() <= 1.000001;
    if !targets.iter().all(bounded) {
        return Ok(None);
    }
    let mut epochs = Vec::with_capacity(times);
    for t in 0..times {
        let (direction, upper, lower) = row(t);
        if !direction.iter().all(bounded) {
            return Ok(None);
        }
        let [x, y, z] = direction;
        // All dot products are between -4 and 4. Use finite sentinels for
        // absent or unbounded thresholds; a comparison with NaN is false.
        let upper = if upper.is_nan() {
            16.0
        } else {
            upper.clamp(-16.0, 16.0)
        };
        let lower = lower
            .filter(|x| !x.is_nan())
            .unwrap_or(-16.0)
            .clamp(-16.0, 16.0);
        epochs.push([x, y, z, upper, lower]);
    }
    match backend {
        Backend::Wgpu => native::with_backend(|backend| mask_on(backend, targets, &epochs, angle)),
        #[cfg(feature = "cubecl")]
        Backend::CubeCl => mask_on(&super::cube::client()?, targets, &epochs, angle),
    }
    .map(Some)
}

fn mask_on<B: Compute>(
    backend: &B,
    targets: &Array2<f64>,
    epochs: &[[f64; 5]],
    angle: Angle,
) -> Result<Array2<bool>, String> {
    let times = epochs.len();
    let mut result = Array2::from_elem((targets.nrows(), times), false);
    for (block, epochs) in epochs.chunks(65_536).enumerate() {
        let t = block * 65_536;
        let geometry = backend.storage(
            &epochs
                .iter()
                .flatten()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
        );
        // Limit each dispatch to 262,144 pairs of mask/refinement words.
        let rows = 262_144 / epochs.len().div_ceil(32);
        for i in (0..targets.nrows()).step_by(rows) {
            let end = (i + rows).min(targets.nrows());
            mask_tile(
                backend,
                targets.slice(s![i..end, ..]),
                epochs,
                &geometry,
                angle,
                result.slice_mut(s![i..end, t..t + epochs.len()]),
            )?;
        }
    }
    Ok(result)
}

/// A catalogue tile returns one range per target, or a definite-hit sentinel.
pub(super) fn catalogue_hits(
    backend: Backend,
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> Result<Option<Vec<bool>>, String> {
    let bounded = |x: &f64| x.abs() <= 1.000001;
    if !threshold.is_finite()
        || !targets.iter().all(bounded)
        || !stars.iter().flatten().all(bounded)
    {
        return Ok(None);
    }
    match backend {
        Backend::Wgpu => {
            native::with_backend(|backend| catalogue_on(backend, targets, stars, threshold))
        }
        #[cfg(feature = "cubecl")]
        Backend::CubeCl => catalogue_on(&super::cube::client()?, targets, stars, threshold),
    }
    .map(Some)
}

fn catalogue_on<B: Compute>(
    backend: &B,
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> Result<Vec<bool>, String> {
    let mut hits = vec![false; targets.nrows()];
    for stars in stars.chunks(32_768) {
        let packed: Vec<f32> = stars
            .iter()
            .flat_map(|&[x, y, z]| [x as f32, y as f32, z as f32, 0.0])
            .collect();
        let catalogue = backend.storage(&packed);
        // Bound the duration of a submission and respect WebGPU's portable
        // one-dimensional workgroup limit.
        let batch = (134_217_728 / stars.len()).min(65_535);
        for (block, results) in hits.chunks_mut(batch).enumerate() {
            if results.iter().all(|&v| v) {
                continue;
            }
            let first = block * batch;
            let rows = targets.slice(s![first..first + results.len(), ..]);
            let packed: Vec<f32> = rows
                .rows()
                .into_iter()
                .flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, threshold as f32])
                .collect();
            let cones = backend.storage(&packed);
            let data = backend.dispatch(Kernel::Catalogue, [&cones, &catalogue], results.len())?;
            let ranges: &[[u32; 2]] = bytemuck::cast_slice(&data);
            for ((hit, row), &[first, end]) in results.iter_mut().zip(rows.rows()).zip(ranges) {
                *hit |= end == u32::MAX
                    || (first < end
                        && stars[first as usize..end as usize]
                            .iter()
                            .any(|s| row[0] * s[0] + row[1] * s[1] + row[2] * s[2] > threshold));
            }
        }
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKENDS: &[Backend] = &[
        Backend::Wgpu,
        #[cfg(feature = "cubecl")]
        Backend::CubeCl,
    ];

    #[test]
    #[ignore = "requires a hardware GPU adapter"]
    fn hardware_mask_preserves_ties_and_tile_order() {
        // Cross both tile dimensions and the final partial word. Adjacent
        // doubles on either side of 0.5 round to the same f32.
        let x = [0.5_f64.next_down(), 0.5, 0.5_f64.next_up()];
        let targets = Array2::from_shape_fn((129, 3), |(i, j)| {
            let v = x[i % 3];
            [v, (1.0 - v * v).sqrt(), 0.0][j]
        });
        let times = 65_539;
        let row = |t: usize| match t % 3 {
            0 => ([1.0, 0.0, 0.0], 0.5, None),
            1 => ([1.0, 0.0, 0.0], 1.0, Some(0.5)),
            _ => ([0.0, 1.0, 0.0], 0.8, None),
        };
        for &backend in BACKENDS {
            let mask = mask(backend, &targets, times, row, Angle::Direction)
                .unwrap()
                .unwrap();
            for ((i, t), &flag) in mask.indexed_iter() {
                assert_eq!(
                    flag,
                    match t % 3 {
                        0 => i % 3 == 2,
                        1 => i % 3 == 0,
                        _ => true,
                    }
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a hardware GPU adapter"]
    fn nearest_pole_and_catalogue_refine_strict_boundaries() {
        let values = [0.5_f64.next_down(), 0.5, 0.5_f64.next_up()];
        let targets = Array2::from_shape_fn((6, 3), |(i, j)| {
            let x = if i < 3 { values[i] } else { -values[i - 3] };
            [x, (1.0 - x * x).sqrt(), 0.0][j]
        });
        let axis = [1.0, 0.0, 0.0];
        for &backend in BACKENDS {
            let grid = mask(
                backend,
                &targets,
                2,
                |t| {
                    if t == 0 {
                        (axis, 0.5, None)
                    } else {
                        (axis, 1.0, Some(0.5))
                    }
                },
                Angle::NearestPole,
            )
            .unwrap()
            .unwrap();
            for i in 0..6 {
                assert_eq!(grid[[i, 0]], i % 3 == 2);
                assert_eq!(grid[[i, 1]], i % 3 == 0);
            }
            let hits = catalogue_hits(backend, &targets, &[axis], 0.5)
                .unwrap()
                .unwrap();
            assert_eq!(hits, [false, false, true, false, false, false]);

            // Two ambiguous stars in different lanes. Only the last clears the
            // strict boundary, so the refinement range must include both.
            let mut stars = vec![[0.0, 0.0, 1.0]; 100];
            stars[7] = [values[0], (1.0 - values[0] * values[0]).sqrt(), 0.0];
            stars[94] = [values[2], (1.0 - values[2] * values[2]).sqrt(), 0.0];
            let targets = ndarray::arr2(&[axis]);
            assert_eq!(
                catalogue_hits(backend, &targets, &stars, 0.5)
                    .unwrap()
                    .unwrap(),
                [true]
            );
        }
    }
}
