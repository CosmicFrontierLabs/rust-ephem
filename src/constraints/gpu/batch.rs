//! Host-side tiling, packed mask decoding, and f64 refinement for GPU kernels.
use super::{cube, cube::Kernel, Angle, AngularBounds};
use cubecl::{prelude::Client, server::Handle};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

fn mask_tile(
    client: &Client,
    targets: ArrayView2<'_, f64>,
    times: usize,
    geometry: &Handle,
    angle: Angle,
    refine: impl Fn(usize, usize) -> bool + Sync,
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
    // Each task owns whole rows, including when this tile is a strided view
    // into a mask with more than 65,536 epochs.
    let rows_per_task = (32_768 / times).max(1);
    let decode = |(block, mut rows): (usize, ArrayViewMut2<'_, bool>)| {
        for (offset, mut row) in rows.rows_mut().into_iter().enumerate() {
            let i = block * rows_per_task + offset;
            let cells = row.as_slice_mut().expect("contiguous mask row");
            for (word, (cells, &[flags, mut uncertain])) in cells
                .chunks_mut(32)
                .zip(&packed[i * words..(i + 1) * words])
                .enumerate()
            {
                if flags == 0 || flags == u32::MAX {
                    cells.fill(flags != 0);
                } else if let Ok(full) = <&mut [bool; 32]>::try_from(&mut *cells) {
                    *full = std::array::from_fn(|lane| flags & (1 << lane) != 0);
                } else {
                    for (lane, cell) in cells.iter_mut().enumerate() {
                        *cell = flags & (1 << lane) != 0;
                    }
                }
                while uncertain != 0 {
                    let lane = uncertain.trailing_zeros() as usize;
                    cells[lane] = refine(i, word * 32 + lane);
                    uncertain &= uncertain - 1;
                }
            }
        }
    };
    let parallel = mask.len() >= 131_072;
    let blocks = mask
        .axis_chunks_iter_mut(Axis(0), rows_per_task)
        .enumerate();
    if parallel {
        blocks.par_bridge().for_each(decode);
    } else {
        blocks.for_each(decode);
    }
    Ok(())
}

pub(super) fn mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> AngularBounds,
    angle: Angle,
    refine: Option<&(dyn Fn(usize, usize) -> bool + Sync)>,
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
        // Replace absent or unbounded thresholds with finite sentinels outside [-4, 4].
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
    let word_capacity = cube::mask_word_capacity(&client);
    let epoch_tile = 65_536.min(word_capacity.saturating_mul(32));
    for (block, epochs) in epochs.chunks(epoch_tile).enumerate() {
        let t = block * epoch_tile;
        let geometry = cube::storage(
            &client,
            &epochs
                .iter()
                .flatten()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
        );
        let rows = word_capacity / epochs.len().div_ceil(32);
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
                    let cosine =
                        angle.projection(|k| target[k] * x + target[k + 1] * y + target[k + 2] * z);
                    cosine > upper || cosine < lower
                },
                result.slice_mut(s![i..end, t..t + epochs.len()]),
            )?;
        }
    }
    Ok(Some(result))
}

/// Full geometric separations need a double-precision inverse cosine.
pub(super) fn angle_grid(
    targets: &Array2<f64>,
    bodies: &[[f64; 3]],
    upper: f64,
    lower: f64,
    grid: &mut crate::constraints::core::LeafValueGrid,
) -> Result<bool, String> {
    use cubecl::prelude::CubeElement;

    // wgpu and Metal lack a reliable f64 acos; use the CPU loop without creating a device.
    if matches!(
        std::env::var("RUST_EPHEM_CUBECL_BACKEND").as_deref(),
        Err(std::env::VarError::NotPresent) | Ok("wgpu" | "metal")
    ) {
        return Ok(false);
    }
    let client = cube::client()?;
    if !client.properties().supports_type(f64::cube_type()) || client.name().starts_with("wgpu") {
        return Ok(false);
    }
    let hardware = &client.properties().hardware;
    let width = 64
        .min(hardware.max_units_per_cube)
        .min(hardware.max_cube_dim.0);
    let capacity = (width as usize)
        .saturating_mul(hardware.max_cube_count.0 as usize)
        .min(1 << 22);
    let epoch_tile = capacity.min(16_384);
    let targets = targets
        .as_slice()
        .expect("contiguous target vectors")
        .as_chunks::<3>()
        .0;
    for (epoch_block, epochs) in bodies.chunks(epoch_tile).enumerate() {
        let start = epoch_block * epoch_tile;
        let geometry = client.create_from_slice(f64::as_bytes(epochs.as_flattened()));
        let rows = capacity / epochs.len();
        for (block, vectors) in targets.chunks(rows).enumerate() {
            let count = vectors.len() * epochs.len();
            let data = cube::separations(
                &client,
                vectors.as_flattened(),
                &geometry,
                count,
                width,
                upper,
                lower,
            )?;
            for (i, (target, signed)) in vectors
                .iter()
                .zip(f64::from_bytes(&data).chunks_exact(epochs.len()))
                .enumerate()
            {
                let row = block * rows + i;
                let range = s![row, start..start + epochs.len()];
                let mut angles = grid.values.slice_mut(range);
                let mut flags = grid.violated.slice_mut(range);
                for (t, ((angle, flag), &value)) in
                    angles.iter_mut().zip(&mut flags).zip(signed).enumerate()
                {
                    if value == 181.0 {
                        let dot = crate::utils::vector_math::dot_product(target, &epochs[t]);
                        *angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
                        *flag = dot > upper || dot < lower;
                    } else {
                        *angle = value.abs();
                        *flag = value.is_sign_negative();
                    }
                }
            }
        }
    }
    Ok(true)
}

/// Immutable host data shared by repeated searches on any selected device.
pub(in crate::constraints) struct Catalogue {
    stars: Vec<[f64; 3]>,
    packed: Vec<[f32; 4]>,
    max_norm_squared: Option<f64>,
}

impl Catalogue {
    pub(in crate::constraints) fn new(units: impl IntoIterator<Item = [f64; 3]>) -> Self {
        let mut stars: Vec<_> = units.into_iter().collect();
        stars.sort_unstable_by(|a, b| a[2].total_cmp(&b[2]));
        let max_norm_squared = stars.iter().try_fold(0.0_f64, |highest, s| {
            s.iter()
                .all(|v| v.abs() <= 1.000001)
                .then(|| highest.max(crate::utils::vector_math::dot_product(s, s)))
        });
        let packed = stars
            .iter()
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32, 0.0])
            .collect();
        Self {
            stars,
            packed,
            max_norm_squared,
        }
    }
}

/// A catalogue tile returns one range per target, or a definite-hit sentinel.
pub(super) fn catalogue_hits(
    targets: &Array2<f64>,
    stars: &Catalogue,
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
    targets: &ndarray::ArrayRef2<f64>,
    stars: &Catalogue,
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

/// A star that can reach the cutoff must also lie within the chord distance
/// along z. For nearest searches, actual stars near the target's z provide a
/// lower bound on the winning dot product, even for clustered catalogues.
fn catalogue_window(
    target: &[f64; 3],
    stars: &[[f64; 3]],
    max_norm_squared: f64,
    threshold: Option<f64>,
) -> std::ops::Range<usize> {
    use crate::utils::vector_math::dot_product;
    let z = target[2];
    let cutoff = threshold.unwrap_or_else(|| {
        let samples = stars.len().min(32);
        let middle = stars.partition_point(|s| s[2] < z);
        let start = middle
            .saturating_sub(samples / 2)
            .min(stars.len() - samples);
        stars[start..start + samples]
            .iter()
            .map(|s| dot_product(target, s))
            .fold(f64::NEG_INFINITY, f64::max)
    });
    // |target - star|² = |target|² + |star|² - 2 dot(target, star).
    // All coordinates are bounded by 1.000001. Round the squared radius
    // outward by more than the f64 error, including cancellation at a tie.
    let radius = (dot_product(target, target) + max_norm_squared - 2.0 * cutoff + 1e-12)
        .max(0.0)
        .sqrt();
    stars.partition_point(|s| s[2] < z - radius)..stars.partition_point(|s| s[2] <= z + radius)
}

/// Each workgroup returns a range containing every candidate still tied in
/// f32, or a definite-hit sentinel for a cone search. Reduce tiles in f64.
fn scan_catalogue<T>(
    targets: &ndarray::ArrayRef2<f64>,
    catalogue: &Catalogue,
    threshold: Option<f64>,
    results: &mut [T],
    finished: impl Fn(&[T]) -> bool,
    reduce: impl Fn(&mut T, ndarray::ArrayView1<'_, f64>, &[[f64; 3]], [u32; 2]),
) -> Result<bool, String> {
    let bounded = |x: &f64| x.abs() <= 1.000001;
    let Some(max_norm_squared) = catalogue.max_norm_squared else {
        return Ok(false);
    };
    if threshold.is_some_and(|x| !x.is_finite()) || !targets.iter().all(bounded) {
        return Ok(false);
    }
    let client = cube::client()?;
    for (stars, packed) in catalogue
        .stars
        .chunks(32_768)
        .zip(catalogue.packed.chunks(32_768))
    {
        let catalogue = cube::storage(&client, packed.as_flattened());
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
                    let range =
                        catalogue_window(&[v[0], v[1], v[2]], stars, max_norm_squared, threshold);
                    [
                        v[0] as f32,
                        v[1] as f32,
                        v[2] as f32,
                        threshold.unwrap_or_default() as f32,
                        range.start as f32,
                        range.end as f32,
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
    fn catalogue_window_keeps_brute_force_hits_and_nearest_ties() {
        use crate::utils::vector_math::dot_product;
        let catalogue = Catalogue::new(
            (0..257)
                .rev()
                .map(|i| {
                    let z = i as f64 / 128.0 - 1.0;
                    let (y, x) = (i as f64 * 2.4).sin_cos();
                    let r = (1.0 - z * z).sqrt();
                    [r * x, r * y, z]
                })
                .chain([[0.0, 0.0, 1.0], [0.3, -0.2, 0.7], [1.0, 1.0, 1.0]]),
        );
        let stars = &catalogue.stars;
        for target in [[0.0; 3], [0.0, 0.0, 1.0], [0.5, -0.75, -0.25], [-1.0; 3]] {
            let best = stars
                .iter()
                .map(|s| dot_product(&target, s))
                .fold(f64::NEG_INFINITY, f64::max);
            for cutoff in [
                None,
                Some(best),
                Some(best.next_down()),
                Some(best.next_up()),
                Some(-0.5),
            ] {
                let range =
                    catalogue_window(&target, stars, catalogue.max_norm_squared.unwrap(), cutoff);
                for (i, star) in stars.iter().enumerate() {
                    if dot_product(&target, star) >= cutoff.unwrap_or(best) {
                        assert!(range.contains(&i));
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a hardware GPU adapter"]
    fn hardware_mask_preserves_ties_and_tile_order() {
        // Cross the epoch tile, the CPU row tile and the final partial word. Adjacent
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
        let hits = catalogue_hits(&targets, &Catalogue::new([axis]), 0.5)
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
            catalogue_hits(&targets, &Catalogue::new(stars), 0.5)
                .unwrap()
                .unwrap(),
            [true]
        );
    }
}
