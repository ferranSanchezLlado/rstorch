#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = lhs[id] + rhs[id];
    }
}

kernel void sub_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = lhs[id] - rhs[id];
    }
}

kernel void mul_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = lhs[id] * rhs[id];
    }
}

kernel void div_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = lhs[id] / rhs[id];
    }
}

kernel void add_scalar_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& rhs [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = input[id] + rhs;
    }
}

kernel void sub_scalar_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& rhs [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = input[id] - rhs;
    }
}

kernel void mul_scalar_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& rhs [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = input[id] * rhs;
    }
}

kernel void div_scalar_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& rhs [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = input[id] / rhs;
    }
}

kernel void powf_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& exponent [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = pow(input[id], exponent);
    }
}

kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = max(input[id], 0.0f);
    }
}

kernel void exp_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = exp(input[id]);
    }
}

kernel void ln_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < len) {
        out[id] = log(input[id]);
    }
}

kernel void sum_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        float total = 0.0f;
        for (uint i = 0; i < len; i++) {
            total += input[i];
        }
        out[0] = total;
    }
}

kernel void mean_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id == 0) {
        float total = 0.0f;
        for (uint i = 0; i < len; i++) {
            total += input[i];
        }
        out[0] = total / float(len);
    }
}

kernel void matmul_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint len = rows * cols;
    if (id >= len) {
        return;
    }

    uint row = id / cols;
    uint col = id % cols;
    float total = 0.0f;

    for (uint i = 0; i < inner; i++) {
        total += lhs[row * inner + i] * rhs[i * cols + col];
    }

    out[id] = total;
}

kernel void transpose_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint len = rows * cols;
    if (id >= len) {
        return;
    }

    uint row = id / cols;
    uint col = id % cols;
    out[col * rows + row] = input[id];
}

kernel void add_row_f32(
    device const float* input [[buffer(0)]],
    device const float* row_values [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint len = rows * cols;
    if (id < len) {
        out[id] = input[id] + row_values[id % cols];
    }
}

kernel void add_col_f32(
    device const float* input [[buffer(0)]],
    device const float* col_values [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint len = rows * cols;
    if (id < len) {
        out[id] = input[id] + col_values[id / cols];
    }
}
