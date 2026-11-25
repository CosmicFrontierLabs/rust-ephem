//! CubeCL implementations of the packed angular mask and star catalogue search.
use super::Angle;
use cubecl::device::{DeviceUnavailable, RuntimeId};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{try_init_setup, AutoGraphicsApi, WgpuDevice, WgpuDeviceKind, WgpuInitError};
use once_cell::sync::OnceCell;

/// Keep each initialized device when a caller switches runtimes between batches.
pub(super) fn client() -> Result<Client, String> {
    use std::env::{var, VarError};

    let backend = match var("RUST_EPHEM_CUBECL_BACKEND") {
        Ok(name) => runtime(&name)?,
        Err(VarError::NotPresent) => RuntimeId::Wgpu,
        Err(VarError::NotUnicode(_)) => return Err("Invalid RUST_EPHEM_CUBECL_BACKEND".into()),
    };

    static WGPU: OnceCell<cubecl::Device> = OnceCell::new();
    static METAL: OnceCell<cubecl::Device> = OnceCell::new();
    static CUDA: OnceCell<cubecl::Device> = OnceCell::new();
    static HIP: OnceCell<cubecl::Device> = OnceCell::new();
    static CPU: OnceCell<cubecl::Device> = OnceCell::new();
    let device = match backend {
        RuntimeId::Wgpu => WGPU.get_or_try_init(wgpu_device),
        RuntimeId::Metal => METAL.get_or_try_init(metal_device),
        RuntimeId::Cuda => CUDA.get_or_try_init(|| cubecl::Device::cuda(0).map_err(device_error)),
        RuntimeId::Hip => HIP.get_or_try_init(|| cubecl::Device::rocm(0).map_err(device_error)),
        RuntimeId::Cpu => CPU.get_or_try_init(|| cubecl::Device::cpu().map_err(device_error)),
    }?;
    Ok(device.client())
}

fn runtime(name: &str) -> Result<RuntimeId, String> {
    match name {
        "wgpu" => Ok(RuntimeId::Wgpu),
        "metal" => Ok(RuntimeId::Metal),
        "cuda" => Ok(RuntimeId::Cuda),
        "hip" | "rocm" => Ok(RuntimeId::Hip),
        "cpu" => Ok(RuntimeId::Cpu),
        _ => Err(format!("Unknown RUST_EPHEM_CUBECL_BACKEND={name:?}. Choose wgpu, metal, cuda, hip (rocm), or cpu.")),
    }
}

fn device_error(error: DeviceUnavailable) -> String {
    match error {
        DeviceUnavailable::NotLinked(runtime) => {
            let feature = match runtime {
                RuntimeId::Cuda => "cubecl-cuda",
                RuntimeId::Hip => "cubecl-hip",
                RuntimeId::Metal => "cubecl-metal",
                RuntimeId::Cpu => "cubecl-cpu",
                RuntimeId::Wgpu => "cubecl",
            };
            format!(
                "The CubeCL {runtime:?} backend requires a build with the Cargo {feature} feature"
            )
        }
        e => e.to_string(),
    }
}

fn wgpu_device() -> Result<cubecl::Device, String> {
    // Prefer a discrete GPU and exclude software rasterizers.
    for kind in [
        WgpuDeviceKind::DiscreteGpu(0),
        WgpuDeviceKind::IntegratedGpu(0),
        WgpuDeviceKind::VirtualGpu(0),
        WgpuDeviceKind::Other(0),
    ] {
        let device = WgpuDevice::new(kind);
        match try_init_setup::<AutoGraphicsApi>(&device, Default::default()) {
            Ok(_) => return Ok(cubecl::Device::Wgpu(device)),
            Err(WgpuInitError::AdapterUnavailable { .. }) => continue,
            Err(e) => return Err(format!("CubeCL initialization failed: {e}")),
        }
    }
    Err("CubeCL wgpu requires a hardware GPU adapter".into())
}

#[cfg(all(feature = "cubecl-metal", target_vendor = "apple"))]
fn metal_device() -> Result<cubecl::Device, String> {
    use objc2_metal::{MTLDevice, MTLGPUFamily};
    // CubeCL asserts Metal 3 support when creating a client. Register the
    // checked device so a missing or older adapter yields a Python error.
    let device = cubecl::metal::device::default_device()
        .filter(|d| d.supportsFamily(MTLGPUFamily::Metal3))
        .or_else(|| {
            cubecl::metal::device::all_devices()
                .into_iter()
                .find(|d| d.supportsFamily(MTLGPUFamily::Metal3))
        })
        .ok_or("CubeCL native Metal requires a Metal 3 capable device")?;
    Ok(cubecl::Device::Metal(cubecl::metal::register_device(
        device,
    )))
}

#[cfg(not(all(feature = "cubecl-metal", target_vendor = "apple")))]
fn metal_device() -> Result<cubecl::Device, String> {
    Err(
        "CubeCL native Metal requires an Apple target built with the Cargo cubecl-metal feature"
            .into(),
    )
}

#[cube(launch)]
fn angular_mask(
    targets: &[f32],
    epochs: &[f32],
    output: &mut [u32],
    #[comptime] nearest_pole: bool,
    #[comptime] apparent: bool,
) {
    let word = ABSOLUTE_POS;
    if word >= output.len() / 2 {
        terminate!();
    }
    let times = epochs.len() / 5;
    let words = times.div_ceil(32);
    let i = word / words * 3;
    let first = word % words * 32;
    let mut flags = 0u32;
    let mut uncertain = 0u32;
    for lane in 0..usize::min(32, times - first) {
        let t = (first + lane) * 5;
        let dot = targets[i] * epochs[t]
            + targets[i + 1] * epochs[t + 1]
            + targets[i + 2] * epochs[t + 2];
        let cosine = if nearest_pole { f32::abs(dot) } else { dot };
        let upper = epochs[t + 3];
        let lower = epochs[t + 4];
        let bit = 1u32 << (lane as u32);
        if cosine > upper || cosine < lower {
            flags |= bit;
        }
        let margin = if apparent { 0.002_f32 } else { 2e-6_f32 };
        if f32::abs(cosine - upper) <= margin || f32::abs(cosine - lower) <= margin {
            uncertain |= bit;
        }
    }
    output[word * 2] = flags;
    output[word * 2 + 1] = uncertain;
}

#[cube(launch)]
fn catalogue(cones: &[f32], stars: &[f32], output: &mut [u32], #[comptime] nearest: bool) {
    let target = CUBE_POS;
    let i = target * 4;
    let count = stars.len() / 4;
    let span = Shared::<[Atomic<u32>]>::new_slice(2usize);
    if UNIT_POS == 0 {
        span[0].store(count as u32);
        span[1].store(0u32);
    }
    sync_cube();
    let mut threshold = cones[i + 3];
    if nearest {
        // Quantize down to share the maximum through portable u32 atomics.
        // Adding four covers every dot of three components bounded by one.
        let peak = Shared::<[Atomic<u32>]>::new_slice(1usize);
        if UNIT_POS == 0 {
            peak[0].store(0);
        }
        sync_cube();
        let mut best = cones[i + 3] - 4.0;
        let mut index = UNIT_POS as usize;
        while index < count {
            let j = index * 4;
            let dot =
                cones[i] * stars[j] + cones[i + 1] * stars[j + 1] + cones[i + 2] * stars[j + 2];
            best = f32::max(best, dot);
            index += CUBE_DIM as usize;
        }
        peak[0].fetch_max(((best + 4.0) * 1048576.0) as u32);
        sync_cube();
        threshold = peak[0].load() as f32 / 1048576.0 - 4.0;
    }
    let mut first = count as u32;
    let mut end = 0u32;
    let mut star = UNIT_POS as usize;
    while star < count {
        let j = star * 4;
        let delta = cones[i] * stars[j] + cones[i + 1] * stars[j + 1] + cones[i + 2] * stars[j + 2]
            - threshold;
        if !nearest && delta > 2e-6_f32 {
            end = u32::MAX;
            break;
        }
        if delta >= -2e-6_f32 {
            first = u32::min(first, star as u32);
            end = star as u32 + 1;
        }
        star += CUBE_DIM as usize;
    }
    span[0].fetch_min(first);
    span[1].fetch_max(end);
    sync_cube();
    if UNIT_POS == 0 {
        output[target * 2] = span[0].load();
        output[target * 2 + 1] = span[1].load();
    }
}

#[derive(Clone, Copy)]
pub(super) enum Kernel {
    Mask(Angle),
    Catalogue { nearest: bool },
}

pub(super) fn storage(client: &Client, data: &[f32]) -> Handle {
    client.create_from_slice(f32::as_bytes(data))
}

pub(super) fn dispatch(
    client: &Client,
    kernel: Kernel,
    inputs: [&Handle; 2],
    records: usize,
) -> Result<impl std::ops::Deref<Target = [u8]>, String> {
    let output = client.empty(records * 8);
    // All three allocations contain contiguous 32-bit elements.
    let arg = |handle: &Handle| unsafe {
        BufferArg::from_raw_parts(handle.clone(), handle.size() as usize / 4)
    };
    // The CPU runtime limits a cube to the number of available cores.
    let hardware = &client.properties().hardware;
    let width = 64
        .min(hardware.max_units_per_cube)
        .min(hardware.max_cube_dim.0);
    let groups = match kernel {
        Kernel::Mask(_) => records.div_ceil(width as usize),
        Kernel::Catalogue { .. } => records,
    };
    let groups = CubeCount::Static(groups as u32, 1, 1);
    let dim = CubeDim::new_1d(width);
    let [a, b] = inputs.map(arg);
    match kernel {
        Kernel::Mask(angle) => angular_mask::launch(
            client,
            groups,
            dim,
            a,
            b,
            arg(&output),
            matches!(angle, Angle::NearestPole),
            matches!(angle, Angle::Apparent),
        ),
        Kernel::Catalogue { nearest } => {
            catalogue::launch(client, groups, dim, a, b, arg(&output), nearest)
        }
    }
    client
        .read_one(output)
        .map_err(|e| format!("CubeCL dispatch failed: {e}"))
}

/// One signed angle per cell. The sign carries the violation flag, including at zero.
#[cube(launch)]
fn separation(targets: &[f64], epochs: &[f64], output: &mut [f64], upper: f64, lower: f64) {
    let cell = ABSOLUTE_POS;
    if cell >= output.len() {
        terminate!();
    }
    let times = epochs.len() / 3;
    let i = cell / times * 3;
    let t = cell % times * 3;
    let dot =
        targets[i] * epochs[t] + targets[i + 1] * epochs[t + 1] + targets[i + 2] * epochs[t + 2];
    // Device contraction may change the last bits of the dot. Near a cutoff
    // this can change the mask, and near +/-1 acos magnifies roundoff.
    if f64::abs(dot) >= 0.999999 || f64::abs(dot - upper) <= 1e-13 || f64::abs(dot - lower) <= 1e-13
    {
        output[cell] = 181.0;
    } else {
        let angle = f64::acos(dot) * (180.0_f64 / std::f64::consts::PI);
        output[cell] = if dot > upper || dot < lower {
            -angle
        } else {
            angle
        };
    }
}

pub(super) fn separations(
    client: &Client,
    targets: &[f64],
    epochs: &Handle,
    cells: usize,
    width: u32,
    upper: f64,
    lower: f64,
) -> Result<impl std::ops::Deref<Target = [u8]>, String> {
    let inputs = client.create_from_slice(f64::as_bytes(targets));
    let output = client.empty(cells * size_of::<f64>());
    // All three buffers contain contiguous f64 values.
    let arg = |h: &Handle| unsafe {
        BufferArg::from_raw_parts(h.clone(), h.size() as usize / size_of::<f64>())
    };
    separation::launch(
        client,
        CubeCount::Static(cells.div_ceil(width as usize) as u32, 1, 1),
        CubeDim::new_1d(width),
        arg(&inputs),
        arg(epochs),
        arg(&output),
        upper,
        lower,
    );
    client
        .read_one(output)
        .map_err(|e| format!("CubeCL dispatch failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_names_select_exact_backends() {
        for (name, expected) in [
            ("wgpu", RuntimeId::Wgpu),
            ("metal", RuntimeId::Metal),
            ("cuda", RuntimeId::Cuda),
            ("hip", RuntimeId::Hip),
            ("rocm", RuntimeId::Hip),
            ("cpu", RuntimeId::Cpu),
        ] {
            assert_eq!(runtime(name).unwrap(), expected);
        }
        assert!(runtime("unknown").is_err());
    }

    #[test]
    #[ignore = "requires the configured CubeCL runtime"]
    fn selected_client_uses_requested_runtime() {
        let selected = std::env::var("RUST_EPHEM_CUBECL_BACKEND").unwrap_or_else(|_| "wgpu".into());
        let expected = match selected.as_str() {
            "rocm" => "hip",
            other => other,
        };
        let client = client().unwrap();
        // wgpu also reports its shader compiler, e.g. wgpu<wgsl>.
        assert_eq!(client.name().split('<').next(), Some(expected));
    }
}
