"""Optional WebGPU backend for geometric constraint masks."""

from functools import lru_cache
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Two bitsets per 32 cells: the mask and the cells requiring f64 refinement.
_SHADER = """
@group(0) @binding(0) var<storage, read> targets: array<f32>;
@group(0) @binding(1) var<storage, read> epochs: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mask)) { return; }
    let nt = arrayLength(&epochs) / 5u;
    let cells = (arrayLength(&targets) / 3u) * nt;
    let first = 32u * id.x;
    var target_index = first / nt;
    var time_index = first % nt;
    var flags = 0u;
    var uncertain = 0u;
    for (var lane = 0u; lane < min(32u, cells-first); lane++) {
        let i = target_index * 3u;
        let t = time_index * 5u;
        let cosine = targets[i]*epochs[t] + targets[i+1u]*epochs[t+1u]
                   + targets[i+2u]*epochs[t+2u];
        let upper = epochs[t+3u];
        let lower = epochs[t+4u];
        flags |= select(0u, 1u << lane, cosine > upper || cosine < lower);
        // Unit-vector components are bounded by one. 2e-6 covers conversion
        // to f32 plus rounding of the products and sum.
        uncertain |= select(0u, 1u << lane,
            abs(cosine-upper) <= 2e-6 || abs(cosine-lower) <= 2e-6);
        time_index++;
        if (time_index == nt) { time_index = 0u; target_index++; }
    }
    mask[id.x] = vec2<u32>(flags, uncertain);
}
"""


@lru_cache(maxsize=1)
def _pipeline() -> tuple[Any, Any, Any]:
    try:
        wgpu = import_module("wgpu")
    except ImportError as exc:
        raise RuntimeError(
            "RUST_EPHEM_GPU=1 requires pip install 'rust-ephem[gpu]'"
        ) from exc
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter.info["adapter_type"] == "CPU":
        raise RuntimeError("RUST_EPHEM_GPU=1 requires a hardware WebGPU adapter")
    device = adapter.request_device_sync()
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={
            "module": device.create_shader_module(code=_SHADER),
            "entry_point": "main",
        },
    )
    return device, pipeline, wgpu.BufferUsage


def _tile(
    targets: NDArray[np.float64], epochs: NDArray[np.float64]
) -> NDArray[np.uint8]:
    device, pipeline, usage = _pipeline()
    cells = len(targets) * len(epochs)
    words = (cells + 31) // 32
    directions = device.create_buffer_with_data(
        data=np.ascontiguousarray(targets, dtype="<f4"), usage=usage.STORAGE
    )
    geometry = device.create_buffer_with_data(
        data=np.ascontiguousarray(epochs, dtype="<f4"), usage=usage.STORAGE
    )
    output = device.create_buffer(size=words * 8, usage=usage.STORAGE | usage.COPY_SRC)
    group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": i, "resource": {"buffer": buf}}
            for i, buf in enumerate((directions, geometry, output))
        ],
    )
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, group)
    compute.dispatch_workgroups((words + 63) // 64)
    compute.end()
    device.queue.submit([encoder.finish()])
    bits = np.frombuffer(device.queue.read_buffer(output), dtype="<u4").reshape(-1, 2)
    flags = np.unpackbits(
        bits[:, 0].copy().view(np.uint8), bitorder="little", count=cells
    )
    # Expand only the words that contain ambiguous cells. Chunk the f64
    # refinement so a grid full of exact ties also has bounded scratch space.
    words_to_check = np.flatnonzero(bits[:, 1])
    for start in range(0, words_to_check.size, 1024):
        selected = words_to_check[start : start + 1024]
        lanes = np.flatnonzero(
            np.unpackbits(bits[selected, 1].view(np.uint8), bitorder="little")
        )
        cells_to_check = 32 * selected[lanes // 32] + lanes % 32
        rows, cols = np.divmod(cells_to_check, len(epochs))
        a, b = targets[rows], epochs[cols]
        cosine = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]
        flags[cells_to_check] = (cosine > b[:, 3]) | (cosine < b[:, 4])
    return flags.reshape(len(targets), len(epochs))


def cosine_mask(
    targets: NDArray[np.float64], epochs: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Evaluate a target-by-time grid with bounded GPU buffer and dispatch sizes."""
    mask = np.empty((len(targets), len(epochs)), dtype=np.bool_)
    for t in range(0, len(epochs), 65_536):
        block = epochs[t : t + 65_536]
        rows = max(1, 8_388_608 // len(block))
        for i in range(0, len(targets), rows):
            mask[i : i + rows, t : t + len(block)] = _tile(targets[i : i + rows], block)
    return mask
