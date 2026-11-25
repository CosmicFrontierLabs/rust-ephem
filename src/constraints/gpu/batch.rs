//! Host-side tiling, packed mask decoding, and f64 refinement for GPU kernels.
use super::{cube, cube::Kernel, Angle, AngularBounds};
use cubecl::{prelude::Client, server::Handle};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};

fn mask_tile(
    client: &Client,
    targets: ArrayView2<'_, f64>,
    times: usize,
    geometry: &Handle,
    angle: Angle,
    refine: impl Fn(usize, usize) -> bool,
    mut mask: ArrayViewMut2<'_, bool>,
) -> Result<(), String> {
    let words = times.div_ceil(32);
    let count = words * targets.nrows();
    let target_buffer = cube::storage(
        client,
        &targets.iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let data = cube::dispatch(
        client,
        Kernel::Mask(angle),
        [&target_buffer, geometry],
        count,
    )?;
    let packed: &[[u32; 2]] = bytemuck::cast_slice(&data);
    for (i, mut row) in mask.rows_mut().into_iter().enumerate() {
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
                cells[lane] = refine(i, word * 32 + lane);
                uncertain &= uncertain - 1;
            }
        }
    }
    Ok(())
}

pub(super) fn mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> AngularBounds,
    angle: Angle,
    refine: Option<&dyn Fn(usize, usize) -> bool>,
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
    let client = cube::client()?;
    let mut result = Array2::from_elem((targets.nrows(), times), false);
    for (block, epochs) in epochs.chunks(65_536).enumerate() {
        let t = block * 65_536;
        let geometry = cube::storage(
            &client,
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
                &client,
                targets.slice(s![i..end, ..]),
                epochs.len(),
                &geometry,
                angle,
                |r, c| {
                    if let Some(exact) = refine {
                        return exact(i + r, t + c);
                    }
                    let target = targets.row(i + r);
                    let [x, y, z, upper, lower] = epochs[c];
                    let dot = target[0] * x + target[1] * y + target[2] * z;
                    let cosine = angle.cosine(dot);
                    cosine > upper || cosine < lower
                },
                result.slice_mut(s![i..end, t..t + epochs.len()]),
            )?;
        }
    }
    Ok(Some(result))
}

/// A catalogue tile returns one range per target, or a definite-hit sentinel.
pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: f64,
) -> Result<Option<Vec<bool>>, String> {
    let mut hits = vec![false; targets.nrows()];
    let done = scan_catalogue(
        targets,
        stars,
        Some(threshold),
        &mut hits,
        |chunk| chunk.iter().all(|&hit| hit),
        |hit, row, stars, [first, end]| {
            *hit |= end == u32::MAX
                || (first < end
                    && stars[first as usize..end as usize]
                        .iter()
                        .any(|s| row[0] * s[0] + row[1] * s[1] + row[2] * s[2] > threshold));
        },
    )?;
    Ok(done.then_some(hits))
}

pub(super) fn nearest_cosines(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
) -> Result<Option<Vec<f64>>, String> {
    let mut best = vec![f64::NEG_INFINITY; targets.nrows()];
    let done = scan_catalogue(
        targets,
        stars,
        None,
        &mut best,
        |_| false,
        |best, row, stars, [first, end]| {
            for s in &stars[first as usize..end as usize] {
                *best = best.max(row[0] * s[0] + row[1] * s[1] + row[2] * s[2]);
            }
        },
    )?;
    Ok(done.then_some(best))
}

/// Each workgroup returns a range containing every candidate still tied in
/// f32, or a definite-hit sentinel for a cone search. Reduce tiles in f64.
fn scan_catalogue<T>(
    targets: &Array2<f64>,
    stars: &[[f64; 3]],
    threshold: Option<f64>,
    results: &mut [T],
    finished: impl Fn(&[T]) -> bool,
    reduce: impl Fn(&mut T, ndarray::ArrayView1<'_, f64>, &[[f64; 3]], [u32; 2]),
) -> Result<bool, String> {
    let bounded = |x: &f64| x.abs() <= 1.000001;
    if threshold.is_some_and(|x| !x.is_finite())
        || !targets.iter().all(bounded)
        || !stars.iter().flatten().all(bounded)
    {
        return Ok(false);
    }
    let client = cube::client()?;
    for stars in stars.chunks(32_768) {
        let packed: Vec<f32> = stars
            .iter()
            .flat_map(|&[x, y, z]| [x as f32, y as f32, z as f32, 0.0])
            .collect();
        let catalogue = cube::storage(&client, &packed);
        // Bound the duration of a submission and respect WebGPU's portable
        // one-dimensional workgroup limit.
        let batch = (134_217_728 / stars.len()).min(65_535);
        for (block, results) in results.chunks_mut(batch).enumerate() {
            if finished(results) {
                continue;
            }
            let first = block * batch;
            let rows = targets.slice(s![first..first + results.len(), ..]);
            let packed: Vec<f32> = rows
                .rows()
                .into_iter()
                .flat_map(|v| {
                    [
                        v[0] as f32,
                        v[1] as f32,
                        v[2] as f32,
                        threshold.unwrap_or_default() as f32,
                    ]
                })
                .collect();
            let cones = cube::storage(&client, &packed);
            let data = cube::dispatch(
                &client,
                Kernel::Catalogue {
                    nearest: threshold.is_none(),
                },
                [&cones, &catalogue],
                results.len(),
            )?;
            let ranges: &[[u32; 2]] = bytemuck::cast_slice(&data);
            for ((output, row), &range) in results.iter_mut().zip(rows.rows()).zip(ranges) {
                reduce(output, row, stars, range);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mask = mask(&targets, times, row, Angle::Direction, None)
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

    #[test]
    #[ignore = "requires a hardware GPU adapter"]
    fn nearest_pole_and_catalogue_refine_strict_boundaries() {
        let values = [0.5_f64.next_down(), 0.5, 0.5_f64.next_up()];
        let targets = Array2::from_shape_fn((6, 3), |(i, j)| {
            let x = if i < 3 { values[i] } else { -values[i - 3] };
            [x, (1.0 - x * x).sqrt(), 0.0][j]
        });
        let axis = [1.0, 0.0, 0.0];
        let grid = mask(
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
            None,
        )
        .unwrap()
        .unwrap();
        for i in 0..6 {
            assert_eq!(grid[[i, 0]], i % 3 == 2);
            assert_eq!(grid[[i, 1]], i % 3 == 0);
        }
        let hits = catalogue_hits(&targets, &[axis], 0.5).unwrap().unwrap();
        assert_eq!(hits, [false, false, true, false, false, false]);

        // Two ambiguous stars in different lanes. Only the last clears the
        // strict boundary, so the refinement range must include both.
        let mut stars = vec![[0.0, 0.0, 1.0]; 100];
        stars[7] = [values[0], (1.0 - values[0] * values[0]).sqrt(), 0.0];
        stars[94] = [values[2], (1.0 - values[2] * values[2]).sqrt(), 0.0];
        let targets = ndarray::arr2(&[axis]);
        assert_eq!(
            catalogue_hits(&targets, &stars, 0.5).unwrap().unwrap(),
            [true]
        );
    }
}
