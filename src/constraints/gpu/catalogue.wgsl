// One workgroup searches a star tile for a single target. A definite hit
// needs no refinement. Otherwise return the span of possible matches so
// the host only examines stars between the first and last ambiguous result.
@group(0) @binding(0) var<storage, read> targets: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> stars: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hits: array<vec2<u32>>;
var<workgroup> first_from_end: atomic<u32>;
var<workgroup> end: atomic<u32>;

@compute @workgroup_size(64)
fn catalogue_hits(@builtin(workgroup_id) group: vec3<u32>,
                  @builtin(local_invocation_index) lane: u32) {
    let cone = targets[group.x];
    let count = arrayLength(&stars);
    var first = count;
    var last = 0u;
    for (var i = lane; i < count; i += 64u) {
        let delta = dot(cone.xyz, stars[i].xyz) - cone.w;
        if (delta > 2e-6) { last = 0xffffffffu; break; }
        if (delta >= -2e-6) {
            first = min(first, i);
            last = i + 1u;
        }
    }
    atomicMax(&first_from_end, count - first);
    atomicMax(&end, last);
    workgroupBarrier();
    if (lane == 0u) {
        hits[group.x] = vec2<u32>(count - atomicLoad(&first_from_end), atomicLoad(&end));
    }
}
