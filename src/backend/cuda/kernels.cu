extern "C" __global__ void add_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = lhs[index] + rhs[index];
    }
}

extern "C" __global__ void sub_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = lhs[index] - rhs[index];
    }
}

extern "C" __global__ void mul_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = lhs[index] * rhs[index];
    }
}

extern "C" __global__ void div_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = lhs[index] / rhs[index];
    }
}

extern "C" __global__ void add_scalar_f32(
    const float *input,
    float *out,
    float rhs,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = input[index] + rhs;
    }
}

extern "C" __global__ void sub_scalar_f32(
    const float *input,
    float *out,
    float rhs,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = input[index] - rhs;
    }
}

extern "C" __global__ void mul_scalar_f32(
    const float *input,
    float *out,
    float rhs,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = input[index] * rhs;
    }
}

extern "C" __global__ void div_scalar_f32(
    const float *input,
    float *out,
    float rhs,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = input[index] / rhs;
    }
}

extern "C" __global__ void powf_f32(
    const float *input,
    float *out,
    float exponent,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = powf(input[index], exponent);
    }
}

extern "C" __global__ void relu_f32(
    const float *input,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        float value = input[index];
        out[index] = value > 0.0f ? value : 0.0f;
    }
}

extern "C" __global__ void exp_f32(
    const float *input,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = expf(input[index]);
    }
}

extern "C" __global__ void ln_f32(
    const float *input,
    float *out,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        out[index] = logf(input[index]);
    }
}

extern "C" __global__ void sum_f32(
    const float *input,
    float *out,
    unsigned int len
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned int index = 0; index < len; ++index) {
            total += input[index];
        }
        out[0] = total;
    }
}

extern "C" __global__ void mean_f32(
    const float *input,
    float *out,
    unsigned int len
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (unsigned int index = 0; index < len; ++index) {
            total += input[index];
        }
        out[0] = total / (float)len;
    }
}

extern "C" __global__ void matmul_f32(
    const float *lhs,
    const float *rhs,
    float *out,
    unsigned int rows,
    unsigned int inner,
    unsigned int cols
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int len = rows * cols;
    if (index < len) {
        unsigned int row = index / cols;
        unsigned int col = index % cols;
        float total = 0.0f;
        for (unsigned int k = 0; k < inner; ++k) {
            total += lhs[row * inner + k] * rhs[k * cols + col];
        }
        out[index] = total;
    }
}

extern "C" __global__ void transpose_f32(
    const float *input,
    float *out,
    unsigned int rows,
    unsigned int cols,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        unsigned int row = index / cols;
        unsigned int col = index % cols;
        out[col * rows + row] = input[index];
    }
}

extern "C" __global__ void add_row_f32(
    const float *input,
    const float *row,
    float *out,
    unsigned int rows,
    unsigned int cols,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        unsigned int col = index % cols;
        out[index] = input[index] + row[col];
    }
}

extern "C" __global__ void add_col_f32(
    const float *input,
    const float *col,
    float *out,
    unsigned int rows,
    unsigned int cols,
    unsigned int len
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        unsigned int row = index / cols;
        out[index] = input[index] + col[row];
    }
}
