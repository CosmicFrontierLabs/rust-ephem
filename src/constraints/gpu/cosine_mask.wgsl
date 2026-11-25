@group(0) @binding(0) var<storage, read> targets: array<f32>;
@group(0) @binding(1) var<storage, read> epochs: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<vec2<u32>>;

fn classify(id: vec3<u32>, nearest_pole: bool) {
    if (id.x >= arrayLength(&mask)) { return; }
    let nt = arrayLength(&epochs) / 5u;
    let words_per_target = (nt + 31u) / 32u;
    let i = (id.x / words_per_target) * 3u;
    let first = (id.x % words_per_target) * 32u;
    var flags = 0u;
    var uncertain = 0u;
    for (var lane = 0u; lane < min(32u, nt-first); lane++) {
        let t = (first + lane) * 5u;
        let dot_value = targets[i]*epochs[t] + targets[i+1u]*epochs[t+1u]
                   + targets[i+2u]*epochs[t+2u];
        let cosine = select(dot_value, abs(dot_value), nearest_pole);
        let upper = epochs[t+3u];
        let lower = epochs[t+4u];
        flags |= select(0u, 1u << lane, cosine > upper || cosine < lower);
        // Unit-vector components are bounded by one. 2e-6 covers conversion
        // to f32 plus rounding of the products and sum.
        uncertain |= select(0u, 1u << lane,
            abs(cosine-upper) <= 2e-6 || abs(cosine-lower) <= 2e-6);
    }
    mask[id.x] = vec2<u32>(flags, uncertain);
}

@compute @workgroup_size(64)
fn direction(@builtin(global_invocation_id) id: vec3<u32>) { classify(id, false); }

@compute @workgroup_size(64)
fn nearest_pole(@builtin(global_invocation_id) id: vec3<u32>) { classify(id, true); }
