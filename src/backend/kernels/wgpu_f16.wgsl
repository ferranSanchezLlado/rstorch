enable f16;

struct Params {
    rhs: f32,
    op: u32,
    len: u32,
    m: u32,
    k: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> lhs: array<f16>;
@group(0) @binding(1) var<storage, read> rhs_values: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<storage, read> params: Params;

fn apply_op_f16(left: f16, right: f16, op: u32) -> f16 {
    switch op {
        case 0u: { return left + right; }
        case 1u: { return left - right; }
        case 2u: { return left * right; }
        default: { return left / right; }
    }
}

@compute @workgroup_size(64)
fn binary_f16_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let gid = id.x;
    if gid >= params.len {
        return;
    }
    out[gid] = apply_op_f16(lhs[gid], rhs_values[gid], params.op);
}

@compute @workgroup_size(64)
fn scalar_f16_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let gid = id.x;
    if gid >= params.len {
        return;
    }
    out[gid] = apply_op_f16(lhs[gid], f16(params.rhs), params.op);
}

@compute @workgroup_size(64)
fn matmul_f16_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let gid = id.x;
    let total = params.m * params.n;
    if gid >= total {
        return;
    }

    let row = gid / params.n;
    let col = gid % params.n;
    var acc = f16(0.0);
    for (var inner = 0u; inner < params.k; inner = inner + 1u) {
        acc = acc + lhs[row * params.k + inner] * rhs_values[inner * params.n + col];
    }
    out[gid] = acc;
}

@compute @workgroup_size(64)
fn sum_f16_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }

    var acc = f16(0.0);
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        acc = acc + lhs[idx];
    }
    out[0] = acc;
}
