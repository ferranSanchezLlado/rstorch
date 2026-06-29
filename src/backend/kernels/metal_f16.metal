#include <metal_stdlib>
using namespace metal;

static half apply_op_f16(half lhs, half rhs, uint op) {
    switch (op) {
        case 0: return lhs + rhs;
        case 1: return lhs - rhs;
        case 2: return lhs * rhs;
        default: return lhs / rhs;
    }
}

kernel void binary_f16_kernel(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_op_f16(lhs[gid], rhs[gid], op);
}

kernel void scalar_f16_kernel(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant half& rhs [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_op_f16(input[gid], rhs, op);
}

kernel void matmul_f16_kernel(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* out [[buffer(2)]],
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
    half acc = half(0.0);
    for (uint inner = 0; inner < k; inner++) {
        acc += lhs[row * k + inner] * rhs[inner * n + col];
    }
    out[gid] = acc;
}

kernel void sum_f16_kernel(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }

    half acc = half(0.0);
    for (uint idx = 0; idx < len; idx++) {
        acc += input[idx];
    }
    out[0] = acc;
}
