//! CubeCL implementations of the packed angular mask and star catalogue search.
use super::batch::{Compute, Kernel};
use super::{native, Angle};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{try_init_device, WgpuSetup};
use once_cell::sync::OnceCell;

pub(super) fn client() -> Result<Client, String> {
    static DEVICE: OnceCell<cubecl::Device> = OnceCell::new();
    let device = DEVICE.get_or_try_init(|| {
        let (instance, adapter, device, queue) = native::request_device()?;
        let setup = WgpuSetup {
            backend: adapter.get_info().backend,
            instance,
            adapter,
            device,
            queue,
        };
        try_init_device(setup, Default::default())
            .map(cubecl::Device::Wgpu)
            .map_err(|e| format!("CubeCL initialization failed: {e}"))
    })?;
    Ok(device.client())
}

#[cube(launch)]
fn angular_mask(
    targets: &[f32],
    epochs: &[f32],
    output: &mut [u32],
    #[comptime] nearest_pole: bool,
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
        if f32::abs(cosine - upper) <= 2e-6_f32 || f32::abs(cosine - lower) <= 2e-6_f32 {
            uncertain |= bit;
        }
    }
    output[word * 2] = flags;
    output[word * 2 + 1] = uncertain;
}

#[cube(launch)]
fn catalogue(cones: &[f32], stars: &[f32], output: &mut [u32]) {
    let target = CUBE_POS;
    let i = target * 4;
    let count = stars.len() / 4;
    let span = Shared::<[Atomic<u32>]>::new_slice(2usize);
    if UNIT_POS == 0 {
        span[0].store(count as u32);
        span[1].store(0u32);
    }
    sync_cube();
    let mut first = count as u32;
    let mut end = 0u32;
    let mut star = UNIT_POS as usize;
    while star < count {
        let j = star * 4;
        let delta = cones[i] * stars[j] + cones[i + 1] * stars[j + 1] + cones[i + 2] * stars[j + 2]
            - cones[i + 3];
        if delta > 2e-6_f32 {
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

impl Compute for Client {
    type Buffer = Handle;

    fn storage(&self, data: &[f32]) -> Handle {
        self.create_from_slice(f32::as_bytes(data))
    }

    fn dispatch(
        &self,
        kernel: Kernel,
        inputs: [&Handle; 2],
        records: usize,
    ) -> Result<impl std::ops::Deref<Target = [u8]>, String> {
        let output = self.empty(records * 8);
        // All three allocations contain contiguous 32-bit elements.
        let arg = |handle: &Handle| unsafe {
            BufferArg::from_raw_parts(handle.clone(), handle.size() as usize / 4)
        };
        let groups = CubeCount::Static(kernel.groups(records), 1, 1);
        let dim = CubeDim::new_1d(64);
        let [a, b] = inputs.map(arg);
        match kernel {
            Kernel::Mask(angle) => angular_mask::launch(
                self,
                groups,
                dim,
                a,
                b,
                arg(&output),
                matches!(angle, Angle::NearestPole),
            ),
            Kernel::Catalogue => catalogue::launch(self, groups, dim, a, b, arg(&output)),
        }
        self.read_one(output)
            .map_err(|e| format!("CubeCL dispatch failed: {e}"))
    }
}
