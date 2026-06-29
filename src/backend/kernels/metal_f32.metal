#include <metal_stdlib>
using namespace metal;

static float apply_op_f32(float lhs, float rhs, uint op) {
    switch (op) {
        case 0: return lhs + rhs;
        case 1: return lhs - rhs;
        case 2: return lhs * rhs;
        default: return lhs / rhs;
    }
}

kernel void binary_f32_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_op_f32(lhs[gid], rhs[gid], op);
}

kernel void scalar_f32_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& rhs [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_op_f32(input[gid], rhs, op);
}

kernel void matmul_f32_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = m * n;
    if (gid >= total) {
        return;
    }

    uint row = gid / n;
    uint col = gid % n;
    float acc = 0.0;
    for (uint inner = 0; inner < k; inner++) {
        acc += lhs[row * k + inner] * rhs[inner * n + col];
    }
    out[gid] = acc;
}

kernel void sum_f32_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }

    float acc = 0.0;
    for (uint idx = 0; idx < len; idx++) {
        acc += input[idx];
    }
    out[0] = acc;
}
