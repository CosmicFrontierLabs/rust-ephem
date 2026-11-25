//! Raw wgpu device initialization, compute pipelines, and readback.
use super::batch::{Compute, Kernel};
use super::Angle;
use once_cell::sync::OnceCell;
use std::sync::{mpsc, Mutex};
use wgpu::util::DeviceExt;

pub(super) struct Backend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    direction: wgpu::ComputePipeline,
    nearest_pole: wgpu::ComputePipeline,
    catalogue: wgpu::ComputePipeline,
}

// Serialize native submissions and their error scopes.
static BACKEND: OnceCell<Mutex<Backend>> = OnceCell::new();

fn checked<T>(
    device: &wgpu::Device,
    work: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let scopes = [
        wgpu::ErrorFilter::Internal,
        wgpu::ErrorFilter::OutOfMemory,
        wgpu::ErrorFilter::Validation,
    ]
    .map(|filter| device.push_error_scope(filter));
    let mut result = work();
    for scope in scopes.into_iter().rev() {
        if let Some(err) = pollster::block_on(scope.pop()) {
            result = Err(err.to_string());
        }
    }
    result
}

impl Backend {
    fn new() -> Result<Self, String> {
        let (_, _, device, queue) = request_device()?;
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
}

pub(super) fn request_device(
) -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|err| format!("GPU execution requires a hardware GPU adapter: {err}"))?;
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return Err("GPU execution requires a hardware GPU adapter".into());
    }
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|err| err.to_string())?;
    Ok((instance, adapter, device, queue))
}

impl Compute for Backend {
    type Buffer = wgpu::Buffer;

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
        kernel: Kernel,
        inputs: [&wgpu::Buffer; 2],
        records: usize,
    ) -> Result<impl std::ops::Deref<Target = [u8]>, String> {
        let groups = kernel.groups(records);
        let pipeline = match kernel {
            Kernel::Mask(Angle::Direction) => &self.direction,
            Kernel::Mask(Angle::NearestPole) => &self.nearest_pole,
            Kernel::Catalogue => &self.catalogue,
        };
        let size = (records * 8) as u64;
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
        slice.get_mapped_range().map_err(|err| err.to_string())
    }
}

pub(super) fn with_backend<T>(
    work: impl FnOnce(&Backend) -> Result<T, String>,
) -> Result<T, String> {
    let backend = BACKEND
        .get_or_try_init(|| Backend::new().map(Mutex::new))?
        .lock()
        .map_err(|_| "GPU backend lock was poisoned".to_string())?;
    checked(&backend.device, || work(&backend))
}
