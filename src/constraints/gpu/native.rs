//! Native device ownership, tiled dispatch, readback, and f64 boundary checks.
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use once_cell::sync::OnceCell;
use std::sync::{mpsc, Mutex};
use wgpu::util::DeviceExt;

struct Backend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
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
        let pipeline = checked(&device, || {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("geometric cosine mask"),
                source: wgpu::ShaderSource::Wgsl(include_str!("cosine_mask.wgsl").into()),
            });
            Ok(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("geometric cosine mask"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            )
        })?;
        Ok(Self {
            device,
            queue,
            pipeline,
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

    fn tile(
        &self,
        targets: ArrayView2<'_, f64>,
        epochs: &[[f64; 5]],
        geometry: &wgpu::Buffer,
        mut mask: ArrayViewMut2<'_, bool>,
    ) -> Result<(), String> {
        let device = &self.device;
        let words = epochs.len().div_ceil(32);
        let count = words * targets.nrows();
        let size = (count * 8) as u64;
        let target_buffer = self.storage(&targets.iter().map(|&v| v as f32).collect::<Vec<_>>());
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
        let bindings = [&target_buffer, geometry, &output];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &std::array::from_fn::<_, 3, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: bindings[i].as_entire_binding(),
            }),
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
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
        let data = slice.get_mapped_range();
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
                    cells[lane] = dot > upper || dot < lower;
                    uncertain &= uncertain - 1;
                }
            }
        }
        Ok(())
    }
}

pub(super) fn mask(
    targets: &Array2<f64>,
    times: usize,
    row: impl Fn(usize) -> ([f64; 3], f64, Option<f64>),
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
    let backend = BACKEND
        .get_or_try_init(|| Backend::new().map(Mutex::new))?
        .lock()
        .map_err(|_| "GPU backend lock was poisoned")?;
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
                    result.slice_mut(s![i..end, t..t + epochs.len()]),
                )?;
            }
        }
        Ok(Some(result))
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
        let mask = mask(&targets, times, row).unwrap().unwrap();
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
