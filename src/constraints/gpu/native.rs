//! Native device ownership, tiled dispatch, readback, and f64 boundary checks.
use super::Angle;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use once_cell::sync::OnceCell;
use std::sync::{mpsc, Mutex};
use wgpu::util::DeviceExt;

struct Backend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    direction: wgpu::ComputePipeline,
    nearest_pole: wgpu::ComputePipeline,
    catalogue: wgpu::ComputePipeline,
}

// Error scopes belong to the device. Serialize scopes as well as submissions
// so that an error is reported to the call that issued the work.
static BACKEND: OnceCell<Mutex<Backend>> = OnceCell::new();

fn checked<T>(
    device: &wgpu::Device,
    work: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    for filter in [
        wgpu::ErrorFilter::Internal,
        wgpu::ErrorFilter::OutOfMemory,
        wgpu::ErrorFilter::Validation,
    ] {
        device.push_error_scope(filter);
    }
    let mut result = work();
    for _ in 0..3 {
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            result = Err(err.to_string());
        }
    }
    result
}

impl Backend {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|err| format!("RUST_EPHEM_GPU=1 requires a hardware GPU adapter: {err}"))?;
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return Err("RUST_EPHEM_GPU=1 requires a hardware GPU adapter".into());
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|err| err.to_string())?;
        let compile = |source: &'static str, entry| {
            checked(&device, || {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                Ok(
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
                )
            })
        };
        let shader = include_str!("cosine_mask.wgsl");
        let direction = compile(shader, "direction")?;
        let nearest_pole = compile(shader, "nearest_pole")?;
        let catalogue = compile(include_str!("catalogue.wgsl"), "catalogue_hits")?;
        Ok(Self {
            device,
            queue,
            direction,
            nearest_pole,
            catalogue,
        })
    }

    fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Submit two storage inputs and map the result once the compute pass completes.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        inputs: [&wgpu::Buffer; 2],
        groups: u32,
        size: u64,
    ) -> Result<wgpu::Buffer, String> {
        let device = &self.device;
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let output = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bindings = [inputs[0], inputs[1], &output];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &std::array::from_fn::<_, 3, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: bindings[i].as_entire_binding(),
            }),
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |status| {
            let _ = tx.send(status);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        Ok(readback)
    }

    fn tile(
        &self,
        targets: ArrayView2<'_, f64>,
        epochs: &[[f64; 5]],
        geometry: &wgpu::Buffer,
        angle: Angle,
        mut mask: ArrayViewMut2<'_, bool>,
    ) -> Result<(), String> {
        let words = epochs.len().div_ceil(32);
        let count = words * targets.nrows();
        let size = (count * 8) as u64;
        let target_buffer = self.storage(&targets.iter().map(|&v| v as f32).collect::<Vec<_>>());
        let pipeline = match angle {
            Angle::Direction => &self.direction,
            Angle::NearestPole => &self.nearest_pole,
        };
        let readback = self.dispatch(
            pipeline,
            [&target_buffer, geometry],
            count.div_ceil(64) as u32,
            size,
        )?;
        let data = readback.slice(..).get_mapped_range();
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
}

fn backend() -> Result<std::sync::MutexGuard<'static, Backend>, String> {
    BACKEND
        .get_or_try_init(|| Backend::new().map(Mutex::new))?
        .lock()
        .map_err(|_| "GPU backend lock was poisoned".to_string())
}

pub(super) fn mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> ([f64; 3], f64, Option<f64>),
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
    let backend = backend()?;
    let mut result = Array2::from_elem((targets.nrows(), times), false);
    checked(&backend.device, || {
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
                backend.tile(
                    targets.slice(s![i..end, ..]),
                    epochs,
                    &geometry,
                    angle,
                    result.slice_mut(s![i..end, t..t + epochs.len()]),
                )?;
            }
        }
        Ok(Some(result))
    })
}

/// A catalogue tile returns one range per target, or a definite-hit sentinel.
pub(super) fn catalogue_hits(
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
    let backend = backend()?;
    let mut hits = vec![false; targets.nrows()];
    checked(&backend.device, || {
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
                let buffer = backend.dispatch(
                    &backend.catalogue,
                    [&cones, &catalogue],
                    results.len() as u32,
                    (results.len() * 8) as u64,
                )?;
                let data = buffer.slice(..).get_mapped_range();
                let ranges: &[[u32; 2]] = bytemuck::cast_slice(&data);
                for ((hit, row), &[first, end]) in results.iter_mut().zip(rows.rows()).zip(ranges) {
                    *hit |= end == u32::MAX
                        || (first < end
                            && stars[first as usize..end as usize].iter().any(|s| {
                                row[0] * s[0] + row[1] * s[1] + row[2] * s[2] > threshold
                            }));
                }
            }
        }
        Ok(Some(hits))
    })
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
        let mask = mask(&targets, times, row, Angle::Direction)
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
