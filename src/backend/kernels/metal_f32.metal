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

static float apply_unary_f32(float x, uint op) {
    switch (op) {
        case 0: return x > 0.0 ? x : 0.0;
        case 1: return -x;
        case 2: return exp(x);
        case 3: return log(x);
        case 4: return tanh(x);
        case 5: return 1.0 / (1.0 + exp(-x));
        case 6: return sqrt(x);
        case 7: return fabs(x);
        default: {
            float x3 = x * x * x;
            return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x3)));
        }
    }
}

kernel void unary_f32_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& op [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = apply_unary_f32(input[gid], op);
}

kernel void row_softmax_f32_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint start = row * cols;
    float max_value = input[start];
    for (uint col = 1; col < cols; col++) {
        max_value = max(max_value, input[start + col]);
    }

    float sum = 0.0;
    for (uint col = 0; col < cols; col++) {
        sum += exp(input[start + col] - max_value);
    }
    float logsumexp = max_value + log(sum);

    for (uint col = 0; col < cols; col++) {
        uint idx = start + col;
        if (op == 0) {
            out[idx] = exp(input[idx] - max_value) / sum;
        } else {
            out[idx] = input[idx] - logsumexp;
        }
    }
}

kernel void sum_last_f32_kernel(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
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
        acc += input[start + col];
    }
    out[row] = acc;
}

kernel void bmm_f32_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
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
        acc += lhs[lhs_base + row * k + inner] * rhs[rhs_base + inner * n + col];
    }
    out[gid] = acc;
}

kernel void strided_matmul_f32_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
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
        float a = lhs[lhs_offset + row * lhs_row_stride + inner * lhs_col_stride];
        float b = rhs[rhs_offset + inner * rhs_row_stride + col * rhs_col_stride];
        acc += a * b;
    }
    out[gid] = acc;
}

kernel void broadcast_f32_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
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
    out[gid] = apply_op_f32(lhs[gid], rhs[rhs_idx], op);
}

kernel void mask_f32_kernel(
    device const float* input [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const float* other [[buffer(3)]],
    constant float& fill [[buffer(4)]],
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

kernel void index_select_rows_f32_kernel(
    device const float* input [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* out [[buffer(2)]],
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

kernel void cross_entropy_f32_kernel(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* out [[buffer(2)]],
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
        float max_value = logits[start];
        for (uint col = 1; col < cols; col++) {
            max_value = max(max_value, logits[start + col]);
        }
        float sum = 0.0;
        for (uint col = 0; col < cols; col++) {
            sum += exp(logits[start + col] - max_value);
        }
        float logsumexp = max_value + log(sum);
        float nll = -(logits[start + target] - logsumexp);
        float smooth = 0.0;
        for (uint col = 0; col < cols; col++) {
            smooth -= logits[start + col] - logsumexp;
        }
        smooth /= float(cols);
        loss_sum += (1.0 - label_smoothing) * nll + label_smoothing * smooth;
        valid += 1;
    }
    if (valid == 0) {
        out[0] = 0.0;
    } else if (mean_reduction != 0) {
        out[0] = loss_sum / float(valid);
    } else {
        out[0] = loss_sum;
    }
}

kernel void layer_norm_f32_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
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
        mean += input[start + col];
    }
    mean /= float(cols);
    float var = 0.0;
    for (uint col = 0; col < cols; col++) {
        float diff = input[start + col] - mean;
        var += diff * diff;
    }
    float inv_std = rsqrt(var / float(cols) + eps);
    for (uint col = 0; col < cols; col++) {
        float xhat = (input[start + col] - mean) * inv_std;
        out[start + col] = xhat * weight[col] + bias[col];
    }
}

kernel void rms_norm_f32_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
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
        float x = input[start + col];
        mean_sq += x * x;
    }
    float inv_rms = rsqrt(mean_sq / float(cols) + eps);
    for (uint col = 0; col < cols; col++) {
        out[start + col] = input[start + col] * inv_rms * weight[col];
    }
}

kernel void sgd_step_f32_kernel(
    device const float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const float* velocity_in [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* velocity_out [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant float& momentum [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    constant uint& use_momentum [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    float value = param[gid];
    float next = value;
    if (weight_decay != 0.0) {
        next -= lr * weight_decay * value;
    }
    if (use_momentum != 0) {
        float velocity = velocity_in == nullptr ? grad[gid] : velocity_in[gid] * momentum + grad[gid];
        velocity_out[gid] = velocity;
        next -= lr * velocity;
    } else {
        next -= lr * grad[gid];
    }
    out[gid] = next;
}

kernel void adam_step_f32_kernel(
    device const float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const float* m_in [[buffer(2)]],
    device const float* v_in [[buffer(3)]],
    device float* out [[buffer(4)]],
    device float* m_out [[buffer(5)]],
    device float* v_out [[buffer(6)]],
    constant float& lr [[buffer(7)]],
    constant float& beta1 [[buffer(8)]],
    constant float& beta2 [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant float& weight_decay [[buffer(11)]],
    constant float& beta1_pow [[buffer(12)]],
    constant float& beta2_pow [[buffer(13)]],
    constant uint& has_state [[buffer(14)]],
    constant uint& len [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    float g = grad[gid];
    float m_prev = has_state != 0 ? m_in[gid] : 0.0;
    float v_prev = has_state != 0 ? v_in[gid] : 0.0;
    float m = beta1 * m_prev + (1.0 - beta1) * g;
    float v = beta2 * v_prev + (1.0 - beta2) * g * g;
    float m_hat = m / (1.0 - beta1_pow);
    float v_hat = v / (1.0 - beta2_pow);
    float decayed = param[gid] - lr * weight_decay * param[gid];
    out[gid] = decayed - lr * m_hat / (sqrt(v_hat) + eps);
    m_out[gid] = m;
    v_out[gid] = v;
}
