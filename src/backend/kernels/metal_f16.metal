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

static half apply_unary_f16(half x, uint op) {
    float xf = float(x);
    switch (op) {
        case 0: return x > half(0.0) ? x : half(0.0);
        case 1: return -x;
        case 2: return half(exp(xf));
        case 3: return half(log(xf));
        case 4: return half(tanh(xf));
        case 5: return half(1.0 / (1.0 + exp(-xf)));
        case 6: return half(sqrt(xf));
        case 7: return half(fabs(xf));
        default: {
            float x3 = xf * xf * xf;
            return half(0.5 * xf * (1.0 + tanh(0.7978845608028654 * (xf + 0.044715 * x3))));
        }
    }
}

kernel void unary_f16_kernel(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& op [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_unary_f16(input[gid], op);
}

kernel void row_softmax_f16_kernel(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint start = row * cols;
    float max_value = float(input[start]);
    for (uint col = 1; col < cols; col++) {
        max_value = max(max_value, float(input[start + col]));
    }

    float sum = 0.0;
    for (uint col = 0; col < cols; col++) {
        sum += exp(float(input[start + col]) - max_value);
    }
    float logsumexp = max_value + log(sum);

    for (uint col = 0; col < cols; col++) {
        uint idx = start + col;
        if (op == 0) {
            out[idx] = half(exp(float(input[idx]) - max_value) / sum);
        } else {
            out[idx] = half(float(input[idx]) - logsumexp);
        }
    }
}

kernel void sum_last_f16_kernel(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    float acc = 0.0;
    uint start = row * cols;
    for (uint col = 0; col < cols; col++) {
        acc += float(input[start + col]);
    }
    out[row] = half(acc);
}

kernel void bmm_f16_kernel(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch * m * n;
    if (gid >= total) {
        return;
    }

    uint b = gid / (m * n);
    uint within = gid % (m * n);
    uint row = within / n;
    uint col = within % n;
    uint lhs_base = b * m * k;
    uint rhs_base = b * k * n;
    float acc = 0.0;
    for (uint inner = 0; inner < k; inner++) {
        acc += float(lhs[lhs_base + row * k + inner]) * float(rhs[rhs_base + inner * n + col]);
    }
    out[gid] = half(acc);
}

kernel void strided_matmul_f16_kernel(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& lhs_offset [[buffer(6)]],
    constant uint& lhs_row_stride [[buffer(7)]],
    constant uint& lhs_col_stride [[buffer(8)]],
    constant uint& rhs_offset [[buffer(9)]],
    constant uint& rhs_row_stride [[buffer(10)]],
    constant uint& rhs_col_stride [[buffer(11)]],
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
        float a = float(lhs[lhs_offset + row * lhs_row_stride + inner * lhs_col_stride]);
        float b = float(rhs[rhs_offset + inner * rhs_row_stride + col * rhs_col_stride]);
        acc += a * b;
    }
    out[gid] = half(acc);
}

kernel void broadcast_f16_kernel(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    constant uint& period [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    constant uint& op [[buffer(6)]],
    constant uint& rhs_len [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    uint rhs_idx;
    if (mode == 0) {
        rhs_idx = gid % period;
    } else if (mode == 1) {
        rhs_idx = gid / period;
    } else {
        rhs_idx = (gid / period) % rhs_len;
    }
    out[gid] = apply_op_f16(lhs[gid], rhs[rhs_idx], op);
}

kernel void mask_f16_kernel(
    device const half* input [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device half* out [[buffer(2)]],
    device const half* other [[buffer(3)]],
    constant half& fill [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    bool take = mask[gid] != 0;
    if (mode == 0) {
        out[gid] = take ? fill : input[gid];
    } else {
        out[gid] = take ? input[gid] : other[gid];
    }
}

kernel void index_select_rows_f16_kernel(
    device const half* input [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    uint out_row = gid / cols;
    uint col = gid % cols;
    out[gid] = input[indices[out_row] * cols + col];
}

kernel void cross_entropy_f16_kernel(
    device const half* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& ignore_index [[buffer(5)]],
    constant float& label_smoothing [[buffer(6)]],
    constant uint& mean_reduction [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    float loss_sum = 0.0;
    uint valid = 0;
    for (uint row = 0; row < rows; row++) {
        uint target = targets[row];
        if (target == ignore_index) {
            continue;
        }
        uint start = row * cols;
        float max_value = float(logits[start]);
        for (uint col = 1; col < cols; col++) {
            max_value = max(max_value, float(logits[start + col]));
        }
        float sum = 0.0;
        for (uint col = 0; col < cols; col++) {
            sum += exp(float(logits[start + col]) - max_value);
        }
        float logsumexp = max_value + log(sum);
        float nll = -(float(logits[start + target]) - logsumexp);
        float smooth = 0.0;
        for (uint col = 0; col < cols; col++) {
            smooth -= float(logits[start + col]) - logsumexp;
        }
        smooth /= float(cols);
        loss_sum += (1.0 - label_smoothing) * nll + label_smoothing * smooth;
        valid += 1;
    }
    if (valid == 0) {
        out[0] = half(0.0);
    } else if (mean_reduction != 0) {
        out[0] = half(loss_sum / float(valid));
    } else {
        out[0] = half(loss_sum);
    }
}

kernel void layer_norm_f16_kernel(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    uint start = row * cols;
    float mean = 0.0;
    for (uint col = 0; col < cols; col++) {
        mean += float(input[start + col]);
    }
    mean /= float(cols);
    float var = 0.0;
    for (uint col = 0; col < cols; col++) {
        float diff = float(input[start + col]) - mean;
        var += diff * diff;
    }
    float inv_std = rsqrt(var / float(cols) + eps);
    for (uint col = 0; col < cols; col++) {
        float xhat = (float(input[start + col]) - mean) * inv_std;
        out[start + col] = half(xhat * float(weight[col]) + float(bias[col]));
    }
}

kernel void rms_norm_f16_kernel(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    uint start = row * cols;
    float mean_sq = 0.0;
    for (uint col = 0; col < cols; col++) {
        float x = float(input[start + col]);
        mean_sq += x * x;
    }
    float inv_rms = rsqrt(mean_sq / float(cols) + eps);
    for (uint col = 0; col < cols; col++) {
        out[start + col] = half(float(input[start + col]) * inv_rms * float(weight[col]));
    }
}

kernel void sgd_step_f16_kernel(
    device const half* param [[buffer(0)]],
    device const half* grad [[buffer(1)]],
    device const half* velocity_in [[buffer(2)]],
    device half* out [[buffer(3)]],
    device half* velocity_out [[buffer(4)]],
    constant half& lr [[buffer(5)]],
    constant half& momentum [[buffer(6)]],
    constant half& weight_decay [[buffer(7)]],
    constant uint& use_momentum [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    half value = param[gid];
    half next = value;
    if (weight_decay != half(0.0)) {
        next -= lr * weight_decay * value;
    }
    if (use_momentum != 0) {
        half velocity = velocity_in == nullptr ? grad[gid] : velocity_in[gid] * momentum + grad[gid];
        velocity_out[gid] = velocity;
        next -= lr * velocity;
    } else {
        next -= lr * grad[gid];
    }
    out[gid] = next;
}

kernel void adam_step_f16_kernel(
    device const half* param [[buffer(0)]],
    device const half* grad [[buffer(1)]],
    device const half* m_in [[buffer(2)]],
    device const half* v_in [[buffer(3)]],
    device half* out [[buffer(4)]],
    device half* m_out [[buffer(5)]],
    device half* v_out [[buffer(6)]],
    constant half& lr [[buffer(7)]],
    constant half& beta1 [[buffer(8)]],
    constant half& beta2 [[buffer(9)]],
    constant half& eps [[buffer(10)]],
    constant half& weight_decay [[buffer(11)]],
    constant half& beta1_pow [[buffer(12)]],
    constant half& beta2_pow [[buffer(13)]],
    constant uint& has_state [[buffer(14)]],
    constant uint& len [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    half g = grad[gid];
    half m_prev = has_state != 0 ? m_in[gid] : half(0.0);
    half v_prev = has_state != 0 ? v_in[gid] : half(0.0);
    half m = beta1 * m_prev + (half(1.0) - beta1) * g;
    half v = beta2 * v_prev + (half(1.0) - beta2) * g * g;
    half m_hat = m / (half(1.0) - beta1_pow);
    half v_hat = v / (half(1.0) - beta2_pow);
    half decayed = param[gid] - lr * weight_decay * param[gid];
    out[gid] = decayed - lr * m_hat / (sqrt(v_hat) + eps);
    m_out[gid] = m;
    v_out[gid] = v;
}
